#!/usr/bin/env python3
from __future__ import annotations

"""
Generate adversarial perturbation detection dataset from CelebDF-v2.

Produces paired clean/adversarial PNGs with labels for a 2-class detector:
  0 = clean image
  1 = adversarial image

Normalization fix: CLIPProcessor normalizes tensors to roughly [-3, 3], so
applying ε=8/255 there is not an 8/255 pixel-space perturbation. This script
strips normalization from preprocessing (attacks see [0,1] tensors) and folds
it into a NormalizeWrapper so the model still receives correctly normalized
inputs.
"""

import argparse
import logging
import random
import sys
import tempfile
import time
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

# Standard CLIP normalization constants (used by all openai/clip-vit-* variants)
_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------


class NormalizeWrapper(nn.Module):
    """
    Wraps LNCLIP-DF so attacks can operate on [0, 1] float32 tensors.

    Applies CLIP mean/std normalization inside forward(), after which the
    normalized tensor is cast to the underlying model's dtype (bf16/fp16/fp32)
    before being passed to the encoder. Logits are returned as float32 so
    torchattacks loss computation stays in full precision.
    """

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
        # x: float32 [0, 1]; keep float32 for the normalize step so gradients
        # flow cleanly, then cast only the model input to its stored dtype.
        mean = self.mean.to(x.device)
        std = self.std.to(x.device)
        x_norm = (x.float() - mean) / std

        model_dtype = next(self.model.parameters()).dtype
        return self.model(x_norm.to(model_dtype)).logits_labels.float()


# ---------------------------------------------------------------------------
# Preprocessing (no normalization — attacks own the [0, 1] space)
# ---------------------------------------------------------------------------


def build_preprocess() -> transforms.Compose:
    """
    Replicates CLIPProcessor spatial transforms but omits normalization.
    Output: float32 tensor in [0, 1], shape (3, 224, 224).
    """
    return transforms.Compose(
        [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),  # uint8 → float32 [0, 1]
        ]
    )


# ---------------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------------


def collect_clips(celebdf_root: Path) -> list[tuple[Path, int]]:
    """Return (clip_dir, df_label) for every clip in CelebDF-v2."""
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
    """
    70 / 15 / 15 train/val/test split, stratified by df_label.
    All frames from one clip stay in the same split.
    """
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

    return {
        "train": r_tr + f_tr,
        "val": r_va + f_va,
        "test": r_te + f_te,
    }


def sample_frames(clip_dir: Path, n: int) -> list[Path]:
    """Return up to n evenly-spaced frame paths from clip_dir."""
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


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def tensor_to_uint8(t: torch.Tensor) -> torch.Tensor:
    return (t.float() * 255).round().clamp(0, 255).to(torch.uint8)


def save_png(tensor: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = tensor_to_uint8(tensor).permute(1, 2, 0).cpu().numpy()
    Image.fromarray(arr).save(str(path), format="PNG")


def load_png_as_tensor(path: Path) -> torch.Tensor:
    return transforms.ToTensor()(Image.open(path).convert("RGB"))


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def verify(
    wrapped: NormalizeWrapper,
    preprocess: transforms.Compose,
    fgsm: torchattacks.Attack,
    pgd: torchattacks.Attack,
    clips: list[tuple[Path, int]],
    device: torch.device,
    logger: logging.Logger,
) -> bool:
    """Run 10-frame verification. Halts and returns False on any failure."""
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

    # --- Check 1: tensor range ---
    lo, hi = imgs_t.min().item(), imgs_t.max().item()
    logger.info(f"[CHECK 1] Tensor range: [{lo:.4f}, {hi:.4f}]  (expected [0, 1])")
    if lo < -0.01 or hi > 1.01:
        logger.error(
            "FAIL — tensors not in [0, 1]; normalization still in preprocessing"
        )
        ok = False
    else:
        logger.info("PASS")

    # --- Check 4: clean accuracy (run early; no point in attack checks if broken) ---
    with torch.no_grad():
        clean_logits = wrapped(imgs_t)
    clean_preds = clean_logits.argmax(dim=1).cpu()
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

    # --- Check 2: perturbation bound ---
    adv_fgsm = fgsm(imgs_t, labels_t)
    adv_pgd = pgd(imgs_t, labels_t)

    limit = EPS + 1e-6
    pert_fgsm = (adv_fgsm - imgs_t).abs().max().item()
    pert_pgd = (adv_pgd - imgs_t).abs().max().item()

    logger.info(
        f"[CHECK 2] FGSM max perturbation: {pert_fgsm:.6f}  (limit {limit:.6f})"
    )
    if pert_fgsm > limit:
        logger.error("FAIL — FGSM perturbation exceeds ε=8/255 in pixel space")
        ok = False
    else:
        logger.info("PASS")

    logger.info(f"[CHECK 2] PGD  max perturbation: {pert_pgd:.6f}  (limit {limit:.6f})")
    if pert_pgd > limit:
        logger.error("FAIL — PGD perturbation exceeds ε=8/255 in pixel space")
        ok = False
    else:
        logger.info("PASS")

    # --- Check 3: PNG round-trip ---
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        tmp = Path(f.name)
    save_png(adv_fgsm[0].cpu(), tmp)
    reloaded = load_png_as_tensor(tmp)
    tmp.unlink(missing_ok=True)
    rt_err = (reloaded - adv_fgsm[0].cpu()).abs().max().item()
    tol = 2 / 255
    logger.info(f"[CHECK 3] PNG round-trip error: {rt_err:.6f}  (limit {tol:.6f})")
    if rt_err > tol:
        logger.error(
            "FAIL — round-trip error > 2/255; PNG quantization lossy beyond tolerance"
        )
        ok = False
    else:
        logger.info("PASS")

    # --- Check 5: flip rates ---
    with torch.no_grad():
        fgsm_preds = wrapped(adv_fgsm).argmax(dim=1).cpu()
        pgd_preds = wrapped(adv_pgd).argmax(dim=1).cpu()

    fgsm_flips = (fgsm_preds != labels_t.cpu()).sum().item()
    pgd_flips = (pgd_preds != labels_t.cpu()).sum().item()

    logger.info(f"[CHECK 5] FGSM flips: {fgsm_flips}/10  (need ≥4)")
    if fgsm_flips < 4:
        logger.error(f"FAIL — FGSM flipped only {fgsm_flips}/10")
        ok = False
    else:
        logger.info("PASS")

    logger.info(f"[CHECK 5] PGD  flips: {pgd_flips}/10  (need ≥7)")
    if pgd_flips < 7:
        logger.error(f"FAIL — PGD flipped only {pgd_flips}/10")
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


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def generate(
    wrapped: NormalizeWrapper,
    preprocess: transforms.Compose,
    fgsm: torchattacks.Attack,
    pgd: torchattacks.Attack,
    splits: dict[str, list[tuple[Path, int]]],
    output_dir: Path,
    frames_per_clip: int,
    batch_size: int,
    device: torch.device,
    logger: logging.Logger,
) -> tuple[list[dict], dict]:
    """Generate full dataset. Returns (manifest_rows, stats)."""
    manifest_rows: list[dict] = []
    stats: dict[str, dict] = {
        sp: {"clean": 0, "fgsm": 0, "pgd": 0, "fgsm_flips": 0, "pgd_flips": 0}
        for sp in splits
    }

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

                # Batch attacks
                for b_start in range(0, len(frame_paths), batch_size):
                    batch = frame_paths[b_start : b_start + batch_size]
                    imgs = [preprocess(Image.open(fp).convert("RGB")) for fp in batch]
                    imgs_t = torch.stack(imgs).to(device)
                    labels_t = torch.full(
                        (len(imgs_t),), df_label, dtype=torch.long, device=device
                    )

                    # --- Save clean images ---
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
                    stats[split_name]["clean"] += len(batch)

                    # --- FGSM ---
                    adv_fgsm = fgsm(imgs_t, labels_t)
                    with torch.no_grad():
                        fgsm_preds = wrapped(adv_fgsm).argmax(dim=1).cpu()
                    fgsm_flips = (fgsm_preds != labels_t.cpu()).sum().item()

                    for i, fp in enumerate(batch):
                        fname = f"{clip_name}_{fp.stem}_fgsm.png"
                        out = adv_dir / fname
                        save_png(adv_fgsm[i], out)
                        manifest_rows.append(
                            {
                                "path": str(out.relative_to(output_dir)),
                                "split": split_name,
                                "label": 1,
                                "source_clip": clip_name,
                                "source_frame": fp.name,
                                "df_label": df_label_str,
                                "attack_name": "fgsm",
                            }
                        )
                    stats[split_name]["fgsm"] += len(batch)
                    stats[split_name]["fgsm_flips"] += fgsm_flips

                    # --- PGD ---
                    adv_pgd = pgd(imgs_t, labels_t)
                    with torch.no_grad():
                        pgd_preds = wrapped(adv_pgd).argmax(dim=1).cpu()
                    pgd_flips = (pgd_preds != labels_t.cpu()).sum().item()

                    for i, fp in enumerate(batch):
                        fname = f"{clip_name}_{fp.stem}_pgd.png"
                        out = adv_dir / fname
                        save_png(adv_pgd[i], out)
                        manifest_rows.append(
                            {
                                "path": str(out.relative_to(output_dir)),
                                "split": split_name,
                                "label": 1,
                                "source_clip": clip_name,
                                "source_frame": fp.name,
                                "df_label": df_label_str,
                                "attack_name": "pgd",
                            }
                        )
                    stats[split_name]["pgd"] += len(batch)
                    stats[split_name]["pgd_flips"] += pgd_flips

                    logger.debug(
                        f"{split_name} | {clip_name} | batch {b_start // batch_size} | "
                        f"FGSM flip={100 * fgsm_flips / len(batch):.1f}% | "
                        f"PGD flip={100 * pgd_flips / len(batch):.1f}%"
                    )

                pbar.update(1)

    return manifest_rows, stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


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
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Run 10-frame verification checks then exit",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_path = args.output_dir / "generation.log"
    logger = setup_logger(log_path)
    t0 = time.time()

    logger.info(f"Device: {device}")
    logger.info(f"Args: {vars(args)}")

    # --- Load model ---
    logger.info(f"Loading checkpoint from {args.checkpoint}")
    model, _ = make_deepfake_model(str(args.checkpoint), device=str(device))
    model.eval()

    # Extract normalization constants from the CLIPProcessor when possible,
    # so we stay in sync even if the backbone changes.
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
            "Could not read normalization from CLIPProcessor; "
            f"using fallback mean={clip_mean} std={clip_std}"
        )

    wrapped = NormalizeWrapper(model, mean=clip_mean, std=clip_std).to(device)
    preprocess = build_preprocess()

    # --- Build attacks (operate on [0, 1] tensors via wrapped model) ---
    fgsm = torchattacks.FGSM(wrapped, eps=EPS)
    pgd = torchattacks.PGD(wrapped, eps=EPS, alpha=ALPHA, steps=PGD_STEPS)

    # --- Collect clips and split ---
    logger.info(f"Scanning clips at {args.celebdf_root}")
    clips = collect_clips(args.celebdf_root)
    logger.info(f"Found {len(clips)} clips total")

    random.seed(args.seed)
    splits = stratified_clip_split(clips, seed=args.seed)
    for sp, cl in splits.items():
        real_n = sum(1 for _, l in cl if l == 0)
        fake_n = sum(1 for _, l in cl if l == 1)
        logger.info(f"  {sp}: {len(cl)} clips  (real={real_n}, fake={fake_n})")

    # --- Verification ---
    all_clips = [item for cl in splits.values() for item in cl]
    ok = verify(wrapped, preprocess, fgsm, pgd, all_clips, device, logger)
    if not ok:
        logger.error("Halting — verification failed.")
        sys.exit(1)

    if args.verify_only:
        logger.info("--verify-only: done.")
        return

    # --- Generate dataset ---
    logger.info("Starting full dataset generation …")
    manifest_rows, stats = generate(
        wrapped,
        preprocess,
        fgsm,
        pgd,
        splits,
        args.output_dir,
        args.frames_per_clip,
        args.batch_size,
        device,
        logger,
    )

    # --- Save manifest ---
    manifest_path = args.output_dir / "manifest.parquet"
    df = pd.DataFrame(manifest_rows)
    df.to_parquet(manifest_path, index=False)
    logger.info(f"Manifest saved: {manifest_path}  ({len(df):,} rows)")

    # --- Final report ---
    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info("FINAL REPORT")
    logger.info("=" * 60)
    for sp in ["train", "val", "test"]:
        s = stats[sp]
        fgsm_rate = 100 * s["fgsm_flips"] / max(s["fgsm"], 1)
        pgd_rate = 100 * s["pgd_flips"] / max(s["pgd"], 1)
        logger.info(
            f"  {sp:5s}: clean={s['clean']:6d} | "
            f"fgsm={s['fgsm']:6d} (flip={fgsm_rate:5.1f}%) | "
            f"pgd={s['pgd']:6d}  (flip={pgd_rate:5.1f}%)"
        )
    total_imgs = sum(s["clean"] + s["fgsm"] + s["pgd"] for s in stats.values())
    logger.info(f"  Total images: {total_imgs:,}")
    logger.info(f"  Runtime: {elapsed:.1f}s")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
