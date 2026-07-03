from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy import ndimage


def dice_score(pred: np.ndarray, target: np.ndarray, eps: float = 1e-8) -> float:
    pred = np.asarray(pred).astype(bool)
    target = np.asarray(target).astype(bool)
    if not pred.any() and not target.any():
        return 1.0
    inter = np.logical_and(pred, target).sum(dtype=np.float64)
    denom = pred.sum(dtype=np.float64) + target.sum(dtype=np.float64)
    return float((2.0 * inter + eps) / (denom + eps))


def iou_score(pred: np.ndarray, target: np.ndarray, eps: float = 1e-8) -> float:
    pred = np.asarray(pred).astype(bool)
    target = np.asarray(target).astype(bool)
    if not pred.any() and not target.any():
        return 1.0
    inter = np.logical_and(pred, target).sum(dtype=np.float64)
    union = np.logical_or(pred, target).sum(dtype=np.float64)
    return float((inter + eps) / (union + eps))


def binary_surface(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask).astype(bool)
    if not mask.any():
        return mask
    structure = ndimage.generate_binary_structure(mask.ndim, 1)
    eroded = ndimage.binary_erosion(mask, structure=structure, border_value=0)
    return np.logical_xor(mask, eroded)


def _empty_penalty(shape: tuple[int, ...]) -> float:
    return float(np.sqrt(sum((size - 1) ** 2 for size in shape)))


def hd95_score(pred: np.ndarray, target: np.ndarray, spacing: tuple[float, ...] | None = None) -> float:
    pred = np.asarray(pred).astype(bool)
    target = np.asarray(target).astype(bool)
    if not pred.any() and not target.any():
        return 0.0
    if pred.any() != target.any():
        return _empty_penalty(pred.shape)
    pred_surface = binary_surface(pred)
    target_surface = binary_surface(target)
    if not pred_surface.any() or not target_surface.any():
        return _empty_penalty(pred.shape)
    dt_to_target = ndimage.distance_transform_edt(~target_surface, sampling=spacing)
    dt_to_pred = ndimage.distance_transform_edt(~pred_surface, sampling=spacing)
    distances = np.concatenate([dt_to_target[pred_surface], dt_to_pred[target_surface]]).astype(np.float64)
    return float(np.percentile(distances, 95)) if distances.size else 0.0


def nsd_score(pred: np.ndarray, target: np.ndarray, tau: float = 2.0, spacing: tuple[float, ...] | None = None) -> float:
    pred = np.asarray(pred).astype(bool)
    target = np.asarray(target).astype(bool)
    if not pred.any() and not target.any():
        return 1.0
    if pred.any() != target.any():
        return 0.0
    pred_surface = binary_surface(pred)
    target_surface = binary_surface(target)
    if not pred_surface.any() and not target_surface.any():
        return 1.0
    if not pred_surface.any() or not target_surface.any():
        return 0.0
    dt_to_target = ndimage.distance_transform_edt(~target_surface, sampling=spacing)
    dt_to_pred = ndimage.distance_transform_edt(~pred_surface, sampling=spacing)
    numerator = float((dt_to_target[pred_surface] <= tau).sum() + (dt_to_pred[target_surface] <= tau).sum())
    denominator = float(pred_surface.sum() + target_surface.sum())
    return numerator / denominator if denominator > 0 else 1.0


class PatientMetricAccumulator:
    def __init__(self, num_classes: int = 4) -> None:
        self.num_classes = int(num_classes)
        self._cases: dict[str, list[tuple[int, np.ndarray, np.ndarray]]] = defaultdict(list)

    def update(self, patient_id: str, center: str, slice_idx: int, pred: np.ndarray, target: np.ndarray) -> None:
        key = f"{center}::{patient_id}"
        self._cases[key].append((int(slice_idx), np.asarray(pred, dtype=np.uint8), np.asarray(target, dtype=np.uint8)))

    def rows(self, tau: float = 2.0) -> list[dict[str, float | str | int]]:
        rows: list[dict[str, float | str | int]] = []
        for key, slices in sorted(self._cases.items()):
            slices = sorted(slices, key=lambda item: item[0])
            pred_vol = np.stack([item[1] for item in slices], axis=0)
            target_vol = np.stack([item[2] for item in slices], axis=0)
            center, patient_id = key.split("::", 1)
            row: dict[str, float | str | int] = {
                "case_key": key,
                "center": center,
                "patient_id": patient_id,
                "num_slices": len(slices),
            }
            for cls, name in ((1, "lv"), (2, "myo"), (3, "scar")):
                pred_c = pred_vol == cls
                target_c = target_vol == cls
                row[f"dice_{name}"] = dice_score(pred_c, target_c)
                row[f"iou_{name}"] = iou_score(pred_c, target_c)
                row[f"hd95_{name}"] = hd95_score(pred_c, target_c)
                row[f"nsd_{name}"] = nsd_score(pred_c, target_c, tau=tau)
            rows.append(row)
        return rows

    def summary(self, tau: float = 2.0) -> dict[str, float]:
        rows = self.rows(tau=tau)
        out = {"num_patients": float(len(rows))}
        metric_keys = [key for row in rows for key in row if key.startswith(("dice_", "iou_", "hd95_", "nsd_"))]
        for key in sorted(set(metric_keys)):
            values = [float(row[key]) for row in rows if key in row]
            out[key] = float(np.mean(values)) if values else float("nan")
        return out
