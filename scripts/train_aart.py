from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aart import AARTUNet2D, ce_dice_loss
from aart.data import LgeScarNPZDataset, parse_center_list


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def foreground_dice_from_logits(logits: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    pred = torch.argmax(logits, dim=1)
    values = {}
    for cls, name in ((1, "lv"), (2, "myo"), (3, "scar")):
        pred_c = pred == cls
        target_c = target == cls
        inter = torch.logical_and(pred_c, target_c).sum().float()
        denom = pred_c.sum().float() + target_c.sum().float()
        values[f"dice_{name}"] = float(((2.0 * inter + 1e-5) / (denom + 1e-5)).detach().cpu())
    values["dice_fg_mean"] = float(np.mean([values["dice_lv"], values["dice_myo"], values["dice_scar"]]))
    return values


def run_epoch(model, loader, optimizer, scaler, device, train: bool, args) -> dict[str, float]:
    model.train(train)
    totals: dict[str, float] = {}
    count = 0
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        context = torch.enable_grad() if train else torch.no_grad()
        with context:
            with torch.autocast(device_type="cuda", enabled=args.amp and device.type == "cuda"):
                logits = model(image)
                loss, loss_terms = ce_dice_loss(logits, target)
            if train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    optimizer.step()

        metrics = {key: float(value.detach().cpu()) for key, value in loss_terms.items()}
        metrics.update(foreground_dice_from_logits(logits.detach(), target.detach()))
        batch_size = image.shape[0]
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + value * batch_size
        count += batch_size
    return {key: value / max(count, 1) for key, value in totals.items()}


def make_loader(args, split: str, train: bool) -> DataLoader:
    dataset = LgeScarNPZDataset(
        root=args.data_root,
        split=split,
        image_size=args.image_size,
        include_centers=parse_center_list(args.train_centers if train else args.eval_centers),
        max_samples=args.max_train_samples if train else args.max_eval_samples,
        augment=train and args.augment,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size if train else args.eval_batch_size,
        shuffle=train,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train AART-Net on NPZ LGE-CMR slices.")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--train_split", default="train")
    parser.add_argument("--val_split", default="val")
    parser.add_argument("--train_centers", default="")
    parser.add_argument("--eval_centers", default="")
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=12.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--max_eval_samples", type=int, default=0)
    parser.add_argument("--patience", type=int, default=35)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--augment", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_loader = make_loader(args, args.train_split, train=True)
    val_loader = make_loader(args, args.val_split, train=False)
    model = AARTUNet2D(in_channels=1, num_classes=4, base_channels=args.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=args.amp and device.type == "cuda")

    best_score = -1.0
    best_epoch = -1
    history = []
    best_path = out_dir / "best_val_fg_mean.pth"
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, scaler, device, train=True, args=args)
        val_metrics = run_epoch(model, val_loader, optimizer=None, scaler=None, device=device, train=False, args=args)
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        print(json.dumps(row, indent=None, sort_keys=True))

        score = float(val_metrics.get("dice_fg_mean", 0.0))
        if score > best_score:
            best_score = score
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "metrics": val_metrics,
                    "config": vars(args),
                    "label_mapping": {"0": "background", "1": "LV_cavity", "2": "MYO", "3": "scar"},
                },
                best_path,
            )
        if args.patience > 0 and epoch - best_epoch >= args.patience:
            break

    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
