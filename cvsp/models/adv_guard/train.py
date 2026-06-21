from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from cvsp.models.adv_guard.model import AdvGuard
from cvsp.models.adv_guard.dataset import AdvGuardDataset


def setup_logger(log_file: Path) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("srnet_train")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh = logging.FileHandler(log_file, mode="w")
    fh.setFormatter(fmt)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    logger: logging.Logger,
):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch} [train]", dynamic_ncols=True)
    for batch_idx, (x, y) in enumerate(pbar):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        total_correct += (logits.argmax(1) == y).sum().item()
        total_samples += x.size(0)

        if batch_idx % 50 == 0:
            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "acc": f"{total_correct/total_samples:.4f}",
                }
            )

    return {
        "loss": total_loss / total_samples,
        "acc": total_correct / total_samples,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    desc: str = "val",
):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_probs: list[float] = []
    all_labels: list[int] = []

    for x, y in tqdm(loader, desc=desc, dynamic_ncols=True):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item() * x.size(0)
        total_correct += (logits.argmax(1) == y).sum().item()
        total_samples += x.size(0)

        probs = torch.softmax(logits, dim=1)[:, 1]
        all_probs.extend(probs.cpu().tolist())
        all_labels.extend(y.cpu().tolist())

    auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0

    return {
        "loss": total_loss / total_samples,
        "acc": total_correct / total_samples,
        "auc": auc,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train AdvGuard for adversarial detection"
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
    )
    parser.add_argument("--output-dir", type=Path, default=Path("./weights/adv_guard"))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(args.output_dir / "training.log")

    manifest = args.manifest or (args.data_root / "manifest.parquet")
    logger.info(f"Device: {device}")
    logger.info(f"Args: {vars(args)}")

    train_ds = AdvGuardDataset(manifest, args.data_root, "train", augment=True)
    val_ds = AdvGuardDataset(manifest, args.data_root, "val", augment=False)
    test_ds = AdvGuardDataset(manifest, args.data_root, "test", augment=False)

    logger.info(
        f"Train: {len(train_ds):,}  Val: {len(val_ds):,}  Test: {len(test_ds):,}"
    )

    for name, ds in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
        n_clean = (ds.df["label"] == 0).sum()
        n_adv = (ds.df["label"] == 1).sum()
        logger.info(
            f"  {name} balance: clean={n_clean:,}  adv={n_adv:,}  "
            f"ratio={n_adv/max(n_clean,1):.2f}"
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = AdvGuard(in_channels=3, num_classes=2).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {n_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-3 * 0.01
    )

    best_auc = 0.0
    best_epoch = 0
    epochs_no_improve = 0
    history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch, logger
        )
        val_metrics = evaluate(
            model, val_loader, criterion, device, desc=f"Epoch {epoch} [val]"
        )
        scheduler.step()

        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]
        logger.info(
            f"Epoch {epoch:3d}  ({elapsed:5.1f}s)  lr={lr_now:.2e}  | "
            f"train: loss={train_metrics['loss']:.4f} acc={train_metrics['acc']:.4f}  | "
            f"val: loss={val_metrics['loss']:.4f} acc={val_metrics['acc']:.4f} "
            f"auc={val_metrics['auc']:.4f}"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_acc": train_metrics["acc"],
                "val_loss": val_metrics["loss"],
                "val_acc": val_metrics["acc"],
                "val_auc": val_metrics["auc"],
                "lr": lr_now,
            }
        )

        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            best_epoch = epoch
            epochs_no_improve = 0
            ckpt_path = args.output_dir / "best.pt"
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_auc": best_auc,
                    "args": vars(args),
                },
                ckpt_path,
            )
            logger.info(f"    new best val AUC={best_auc:.4f}, saved to {ckpt_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                logger.info(
                    f"Early stopping: {args.patience} epochs without improvement"
                )
                break

    with open(args.output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    logger.info(
        f"Loading best checkpoint from epoch {best_epoch} (val AUC={best_auc:.4f})"
    )
    ckpt = torch.load(args.output_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])
    test_metrics = evaluate(model, test_loader, criterion, device, desc="test")

    logger.info("=" * 60)
    logger.info("FINAL TEST RESULTS")
    logger.info("=" * 60)
    logger.info(f"  Test loss: {test_metrics['loss']:.4f}")
    logger.info(f"  Test acc:  {test_metrics['acc']:.4f}")
    logger.info(f"  Test AUC:  {test_metrics['auc']:.4f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
