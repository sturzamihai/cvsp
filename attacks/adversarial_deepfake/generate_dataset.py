from __future__ import annotations

import argparse
import io
import logging
import random
import sys
import tempfile
import time
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torchattacks
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from cvsp.models.lnclip_df import load_model as make_deepfake_model

EPS = 8 / 255
ALPHA = 2 / 255
PGD_STEPS = 25

_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


class NormalizeWrapper(nn.Module):
    def __init__(self, model: nn.Module, mean=_CLIP_MEAN, std=_CLIP_STD) -> None:
        super().__init__()
        self.model = model
        self.register_buffer(
            "mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = self.mean.to(x.device)
        std = self.std.to(x.device)
        x_norm = (x.float() - mean) / std
        model_dtype = next(self.model.parameters()).dtype
        return self.model(x_norm.to(model_dtype)).logits_labels.float()


def build_preprocess() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )


class Attack(ABC):
    label: int

    @abstractmethod
    def apply(
        self, imgs: torch.Tensor, labels: torch.Tensor
    ) -> list[tuple[str, torch.Tensor]]: ...


class FGSMAttack(Attack):
    label = 1

    def __init__(self, model: nn.Module, eps: float) -> None:
        self._attack = torchattacks.FGSM(model, eps=eps)

    def apply(self, imgs, labels):
        adv = self._attack(imgs, labels)
        return [("fgsm", adv[i]) for i in range(len(imgs))]


class PGDAttack(Attack):
    label = 1

    def __init__(self, model: nn.Module, eps: float, alpha: float, steps: int) -> None:
        self._attack = torchattacks.PGD(model, eps=eps, alpha=alpha, steps=steps)

    def apply(self, imgs, labels):
        adv = self._attack(imgs, labels)
        return [("pgd", adv[i]) for i in range(len(imgs))]


class UniformNoiseAugmentation(Attack):
    label = 0
    _levels = [2 / 255, 4 / 255, 8 / 255, 16 / 255]

    def __init__(self) -> None:
        self._counter = 0

    def apply(self, imgs, _):
        results = []
        for img in imgs:
            eps = self._levels[self._counter % len(self._levels)]
            name = f"noise{int(round(eps * 255))}"
            results.append(
                (name, (img + torch.empty_like(img).uniform_(-eps, eps)).clamp(0, 1))
            )
            self._counter += 1
        return results


class JPEGAugmentation(Attack):
    label = 0
    _qualities = [95, 75, 50]

    def __init__(self) -> None:
        self._counter = 0

    def apply(self, imgs, _):
        results = []
        for img in imgs:
            quality = self._qualities[self._counter % len(self._qualities)]
            results.append((f"jpeg{quality}", _jpeg_compress(img, quality)))
            self._counter += 1
        return results


def _jpeg_compress(x: torch.Tensor, quality: int) -> torch.Tensor:
    arr = _tensor_to_uint8(x).permute(1, 2, 0).cpu().numpy()
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return transforms.ToTensor()(Image.open(buf).convert("RGB"))


def _tensor_to_uint8(t: torch.Tensor) -> torch.Tensor:
    return (t.float() * 255).round().clamp(0, 255).to(torch.uint8)


def save_png(tensor: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = _tensor_to_uint8(tensor).permute(1, 2, 0).cpu().numpy()
    Image.fromarray(arr).save(str(path), format="PNG")


def load_png_as_tensor(path: Path) -> torch.Tensor:
    return transforms.ToTensor()(Image.open(path).convert("RGB"))


def collect_clips(celebdf_root: Path) -> list[tuple[Path, int]]:
    sources = [
        ("Celeb-real/frames", 0),
        ("YouTube-real/frames", 0),
        ("Celeb-synthesis/frames", 1),
    ]
    clips: list[tuple[Path, int]] = []
    for subdir, df_label in sources:
        frames_dir = celebdf_root / subdir
        if not frames_dir.exists():
            continue
        for clip_dir in sorted(frames_dir.iterdir()):
            if clip_dir.is_dir():
                clips.append((clip_dir, df_label))
    return clips


def stratified_clip_split(
    clips: list[tuple[Path, int]], seed: int = 42
) -> dict[str, list[tuple[Path, int]]]:
    rng = random.Random(seed)

    real = [(c, l) for c, l in clips if l == 0]
    fake = [(c, l) for c, l in clips if l == 1]

    def split_one(group: list) -> tuple[list, list, list]:
        g = group.copy()
        rng.shuffle(g)
        n = len(g)
        n_test = max(1, round(n * 0.15))
        n_val = max(1, round(n * 0.15))
        n_train = n - n_test - n_val
        return g[:n_train], g[n_train : n_train + n_val], g[n_train + n_val :]

    r_tr, r_va, r_te = split_one(real)
    f_tr, f_va, f_te = split_one(fake)
    return {"train": r_tr + f_tr, "val": r_va + f_va, "test": r_te + f_te}


def sample_frames(clip_dir: Path, n: int) -> list[Path]:
    frames = sorted({*clip_dir.glob("*.png"), *clip_dir.glob("*.jpg")})
    if not frames:
        return []
    if len(frames) <= n:
        return frames
    idxs = [round(i * (len(frames) - 1) / (n - 1)) for i in range(n)]
    seen: set[int] = set()
    result: list[Path] = []
    for i in idxs:
        if i not in seen:
            seen.add(i)
            result.append(frames[i])
    return result


def setup_logger(log_file: Path) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("adv_gen")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh = logging.FileHandler(log_file, mode="w")
    fh.setFormatter(fmt)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def verify(
    wrapped: NormalizeWrapper,
    preprocess: transforms.Compose,
    attacks: list[Attack],
    clips: list[tuple[Path, int]],
    device: torch.device,
    logger: logging.Logger,
) -> bool:
    logger.info("=" * 60)
    logger.info("VERIFICATION — 10 frames")
    logger.info("=" * 60)

    samples: list[tuple[Path, int]] = []
    for clip_dir, df_label in clips:
        for fp in sample_frames(clip_dir, 32):
            samples.append((fp, df_label))
            if len(samples) >= 10:
                break
        if len(samples) >= 10:
            break
    samples = samples[:10]
    logger.info(f"Collected {len(samples)} frames")

    imgs_t = torch.stack(
        [preprocess(Image.open(fp).convert("RGB")) for fp, _ in samples]
    ).to(device)
    labels_t = torch.tensor([l for _, l in samples], dtype=torch.long, device=device)

    ok = True

    lo, hi = imgs_t.min().item(), imgs_t.max().item()
    logger.info(f"[CHECK 1] Tensor range: [{lo:.4f}, {hi:.4f}]  (expected [0, 1])")
    if lo < -0.01 or hi > 1.01:
        logger.error(
            "FAIL — tensors not in [0, 1]; normalization still in preprocessing"
        )
        ok = False
    else:
        logger.info("PASS")

    with torch.no_grad():
        clean_preds = wrapped(imgs_t).argmax(dim=1).cpu()
    correct = (clean_preds == labels_t.cpu()).sum().item()
    logger.info(f"[CHECK 4] Clean accuracy: {correct}/10  (need ≥8)")
    if correct < 8:
        logger.error(f"FAIL — only {correct}/10 correct")
        ok = False
    else:
        logger.info("PASS")

    if not ok:
        logger.error("Stopping early — model or preprocessing broken.")
        return False

    adv_attacks = [a for a in attacks if a.label == 1]
    first_adv_tensor = None

    limit = EPS + 1e-6
    for attack in adv_attacks:
        results = attack.apply(imgs_t, labels_t)
        adv_t = torch.stack([t for _, t in results])
        attack_name = results[0][0]

        if first_adv_tensor is None:
            first_adv_tensor = adv_t[0].cpu()

        pert = (adv_t - imgs_t).abs().max().item()
        logger.info(
            f"[CHECK 2] {attack_name.upper()} max perturbation: {pert:.6f}  (limit {limit:.6f})"
        )
        if pert > limit:
            logger.error(
                f"FAIL — {attack_name.upper()} perturbation exceeds ε=8/255 in pixel space"
            )
            ok = False
        else:
            logger.info("PASS")

        with torch.no_grad():
            flips = (wrapped(adv_t).argmax(dim=1).cpu() != labels_t.cpu()).sum().item()
        min_flips = 4 if attack_name == "fgsm" else 7
        logger.info(
            f"[CHECK 5] {attack_name.upper()} flips: {flips}/10  (need ≥{min_flips})"
        )
        if flips < min_flips:
            logger.error(f"FAIL — {attack_name.upper()} flipped only {flips}/10")
            ok = False
        else:
            logger.info("PASS")

    if first_adv_tensor is not None:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            tmp = Path(f.name)
        save_png(first_adv_tensor, tmp)
        reloaded = load_png_as_tensor(tmp)
        tmp.unlink(missing_ok=True)
        rt_err = (reloaded - first_adv_tensor).abs().max().item()
        tol = 2 / 255
        logger.info(f"[CHECK 3] PNG round-trip error: {rt_err:.6f}  (limit {tol:.6f})")
        if rt_err > tol:
            logger.error(
                "FAIL — round-trip error > 2/255; PNG quantization lossy beyond tolerance"
            )
            ok = False
        else:
            logger.info("PASS")

    if ok:
        logger.info("=" * 60)
        logger.info("ALL VERIFICATION CHECKS PASSED")
        logger.info("=" * 60)
    else:
        logger.error("=" * 60)
        logger.error("VERIFICATION FAILED — halting")
        logger.error("=" * 60)

    return ok


def generate(
    preprocess: transforms.Compose,
    attacks: list[Attack],
    splits: dict[str, list[tuple[Path, int]]],
    output_dir: Path,
    frames_per_clip: int,
    batch_size: int,
    device: torch.device,
) -> list[dict]:
    manifest_rows: list[dict] = []
    total_clips = sum(len(v) for v in splits.values())

    with tqdm(total=total_clips, desc="Clips", unit="clip", dynamic_ncols=True) as pbar:
        for split_name, clip_list in splits.items():
            clean_dir = output_dir / split_name / "clean"
            adv_dir = output_dir / split_name / "adv"
            clean_dir.mkdir(parents=True, exist_ok=True)
            adv_dir.mkdir(parents=True, exist_ok=True)

            for clip_dir, df_label in clip_list:
                clip_name = clip_dir.name
                df_label_str = "real" if df_label == 0 else "fake"
                frame_paths = sample_frames(clip_dir, frames_per_clip)

                if not frame_paths:
                    pbar.update(1)
                    continue

                for b_start in range(0, len(frame_paths), batch_size):
                    batch = frame_paths[b_start : b_start + batch_size]
                    imgs = [preprocess(Image.open(fp).convert("RGB")) for fp in batch]
                    imgs_t = torch.stack(imgs).to(device)
                    labels_t = torch.full(
                        (len(imgs_t),), df_label, dtype=torch.long, device=device
                    )

                    for fp, img in zip(batch, imgs):
                        fname = f"{clip_name}_{fp.stem}.png"
                        out = clean_dir / fname
                        save_png(img, out)
                        manifest_rows.append(
                            {
                                "path": str(out.relative_to(output_dir)),
                                "split": split_name,
                                "label": 0,
                                "source_clip": clip_name,
                                "source_frame": fp.name,
                                "df_label": df_label_str,
                                "attack_name": "clean",
                            }
                        )

                    for attack in attacks:
                        out_dir = adv_dir if attack.label == 1 else clean_dir
                        for fp, (name, adv_img) in zip(
                            batch, attack.apply(imgs_t, labels_t)
                        ):
                            fname = f"{clip_name}_{fp.stem}_{name}.png"
                            out = out_dir / fname
                            save_png(adv_img, out)
                            manifest_rows.append(
                                {
                                    "path": str(out.relative_to(output_dir)),
                                    "split": split_name,
                                    "label": attack.label,
                                    "source_clip": clip_name,
                                    "source_frame": fp.name,
                                    "df_label": df_label_str,
                                    "attack_name": name,
                                }
                            )

                pbar.update(1)

    return manifest_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate adversarial perturbation detection dataset from CelebDF-v2"
    )
    parser.add_argument("--celebdf-root", type=Path, default="./data/Celeb-DF-v2")
    parser.add_argument("--checkpoint", type=Path, default="./weights/lnclip.ckpt")
    parser.add_argument("--output-dir", type=Path, default="./data/Adv-Celeb-DF")
    parser.add_argument("--frames-per-clip", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_path = args.output_dir / "generation.log"
    logger = setup_logger(log_path)
    t0 = time.time()

    logger.info(f"Device: {device}")
    logger.info(f"Args: {vars(args)}")

    logger.info(f"Loading checkpoint from {args.checkpoint}")
    model, _ = make_deepfake_model(str(args.checkpoint), device=str(device))
    model.eval()

    try:
        proc = model.feature_extractor._preprocess
        clip_mean = list(proc.image_mean)
        clip_std = list(proc.image_std)
        logger.info(
            f"Normalization from CLIPProcessor — mean={clip_mean} std={clip_std}"
        )
    except Exception:
        clip_mean = _CLIP_MEAN
        clip_std = _CLIP_STD
        logger.warning(
            f"Could not read normalization from CLIPProcessor; "
            f"using fallback mean={clip_mean} std={clip_std}"
        )

    wrapped = NormalizeWrapper(model, mean=clip_mean, std=clip_std).to(device)
    preprocess = build_preprocess()

    attacks: list[Attack] = [
        FGSMAttack(wrapped, eps=EPS),
        PGDAttack(wrapped, eps=EPS, alpha=ALPHA, steps=PGD_STEPS),
        UniformNoiseAugmentation(),
        JPEGAugmentation(),
    ]

    logger.info(f"Scanning clips at {args.celebdf_root}")
    clips = collect_clips(args.celebdf_root)
    logger.info(f"Found {len(clips)} clips total")

    random.seed(args.seed)
    splits = stratified_clip_split(clips, seed=args.seed)
    for sp, cl in splits.items():
        real_n = sum(1 for _, l in cl if l == 0)
        fake_n = sum(1 for _, l in cl if l == 1)
        logger.info(f"  {sp}: {len(cl)} clips  (real={real_n}, fake={fake_n})")

    all_clips = [item for cl in splits.values() for item in cl]
    ok = verify(wrapped, preprocess, attacks, all_clips, device, logger)
    if not ok:
        logger.error("Halting — verification failed.")
        sys.exit(1)

    if args.verify_only:
        logger.info("--verify-only: done.")
        return

    logger.info("Starting full dataset generation …")
    manifest_rows = generate(
        preprocess,
        attacks,
        splits,
        args.output_dir,
        args.frames_per_clip,
        args.batch_size,
        device,
    )

    manifest_path = args.output_dir / "manifest.parquet"
    df = pd.DataFrame(manifest_rows)
    df.to_parquet(manifest_path, index=False)
    logger.info(f"Manifest saved: {manifest_path}  ({len(df):,} rows)")

    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info("FINAL REPORT")
    logger.info("=" * 60)
    for sp in ["train", "val", "test"]:
        s = df[df["split"] == sp]
        n_clean_orig = (s["attack_name"] == "clean").sum()
        n_clean_aug = ((s["label"] == 0) & (s["attack_name"] != "clean")).sum()
        n_adv = (s["label"] == 1).sum()
        logger.info(
            f"  {sp:5s}: clean={n_clean_orig + n_clean_aug:6d} "
            f"(orig={n_clean_orig:5d} + aug={n_clean_aug:5d}) | "
            f"adv={n_adv:6d}"
        )
    logger.info(f"  Total images: {len(df):,}")
    logger.info(f"  Runtime: {elapsed:.1f}s")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
