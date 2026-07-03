from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aart import AARTUNet2D
from aart.data import LgeScarNPZDataset, parse_center_list
from aart.metrics import PatientMetricAccumulator


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AART-Net inference and patient-level evaluation.")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--centers", default="")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--save_predictions", action="store_true")
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = LgeScarNPZDataset(
        root=args.data_root,
        split=args.split,
        image_size=args.image_size,
        include_centers=parse_center_list(args.centers),
        augment=False,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = AARTUNet2D(in_channels=1, num_classes=4, base_channels=args.base_channels).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state, strict=True)
    model.eval()

    accumulator = PatientMetricAccumulator(num_classes=4)
    pred_dir = out_dir / "predictions"
    if args.save_predictions:
        pred_dir.mkdir(exist_ok=True)

    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device)
            logits = model(image)
            pred = torch.argmax(logits, dim=1).cpu().numpy().astype(np.uint8)
            target = batch["target"].numpy().astype(np.uint8)
            for idx in range(pred.shape[0]):
                accumulator.update(
                    patient_id=str(batch["patient_id"][idx]),
                    center=str(batch["center"][idx]),
                    slice_idx=int(batch["slice_idx"][idx]),
                    pred=pred[idx],
                    target=target[idx],
                )
                if args.save_predictions:
                    stem = Path(str(batch["path"][idx])).stem
                    np.savez_compressed(pred_dir / f"{stem}_pred.npz", pred=pred[idx], target=target[idx])

    rows = accumulator.rows()
    summary = accumulator.summary()
    if rows:
        with (out_dir / "per_patient_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    with (out_dir / "summary_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)
    print(summary)


if __name__ == "__main__":
    main()
