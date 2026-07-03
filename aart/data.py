from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def normalize_minmax(image: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return np.zeros_like(image, dtype=np.float32)
    lo = float(finite.min())
    hi = float(finite.max())
    if hi > lo:
        image = (image - lo) / (hi - lo + eps)
    return np.clip(image, 0.0, 1.0).astype(np.float32)


def parse_case_slice(path: str | Path) -> tuple[str, int]:
    stem = Path(path).stem
    if "_z" not in stem:
        return stem, -1
    case, z_text = stem.rsplit("_z", 1)
    try:
        return case, int(z_text)
    except ValueError:
        return case, -1


def parse_center_list(text: str | None) -> set[str] | None:
    if not text:
        return None
    centers = {item.strip() for item in str(text).split(",") if item.strip()}
    return centers or None


class LgeScarNPZDataset(Dataset):
    """NPZ dataset for four-class LGE-CMR segmentation.

    Expected keys are ``image`` and ``label``. Labels follow the paper
    convention: 0 background, 1 LV cavity, 2 myocardium, 3 scar.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        image_size: int = 128,
        include_centers: set[str] | None = None,
        max_samples: int = 0,
        augment: bool = False,
        rotation_range: tuple[float, float] = (-15.0, 15.0),
        scale_range: tuple[float, float] = (0.92, 1.08),
        gamma_range: tuple[float, float] = (0.85, 1.20),
    ) -> None:
        self.root = Path(root)
        self.split = str(split)
        self.split_dir = self.root / self.split
        self.image_size = int(image_size)
        self.augment = bool(augment)
        self.rotation_range = tuple(float(x) for x in rotation_range)
        self.scale_range = tuple(float(x) for x in scale_range)
        self.gamma_range = tuple(float(x) for x in gamma_range)

        files = sorted(self.split_dir.rglob("*.npz"))
        if include_centers:
            files = [
                path
                for path in files
                if (path.parent.name if path.parent != self.split_dir else self.split) in include_centers
            ]
        if max_samples and int(max_samples) > 0:
            files = files[: int(max_samples)]
        if not files:
            raise FileNotFoundError(f"No NPZ files found under {self.split_dir}")
        self.files = files

    def __len__(self) -> int:
        return len(self.files)

    def _resize(self, image: np.ndarray, label: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if image.shape[-2:] == (self.image_size, self.image_size):
            return image, label
        image_t = torch.from_numpy(image).float().view(1, 1, *image.shape[-2:])
        label_t = torch.from_numpy(label).float().view(1, 1, *label.shape[-2:])
        size = (self.image_size, self.image_size)
        image = F.interpolate(image_t, size=size, mode="bilinear", align_corners=False)[0, 0].numpy()
        label = F.interpolate(label_t, size=size, mode="nearest")[0, 0].numpy().astype(np.uint8)
        return image, label

    def _augment_spatial(self, image: np.ndarray, label: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not self.augment:
            return image, label
        angle = 0.0
        if torch.rand(()) < 0.4:
            angle = float(torch.empty(()).uniform_(self.rotation_range[0], self.rotation_range[1])) * np.pi / 180.0
        scale = 1.0
        if torch.rand(()) < 0.15:
            scale = float(torch.empty(()).uniform_(self.scale_range[0], self.scale_range[1]))
        if angle == 0.0 and scale == 1.0:
            return image, label
        cos_a = np.cos(angle) * scale
        sin_a = np.sin(angle) * scale
        theta = torch.tensor([[[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0]]], dtype=torch.float32)
        image_t = torch.from_numpy(image).view(1, 1, *image.shape).float()
        label_t = torch.from_numpy(label).view(1, 1, *label.shape).float()
        grid = F.affine_grid(theta, image_t.shape, align_corners=False)
        image_t = F.grid_sample(image_t, grid, mode="bilinear", padding_mode="border", align_corners=False)
        label_t = F.grid_sample(label_t, grid, mode="nearest", padding_mode="zeros", align_corners=False)
        return image_t[0, 0].numpy(), label_t[0, 0].numpy().astype(np.uint8)

    def _augment_intensity(self, image: np.ndarray) -> np.ndarray:
        if not self.augment:
            return image.astype(np.float32)
        if torch.rand(()) < 0.2:
            gamma = float(torch.empty(()).uniform_(self.gamma_range[0], self.gamma_range[1]))
            image = np.clip(image, 0.0, 1.0) ** gamma
        if torch.rand(()) < 0.1:
            image = np.clip(image + float(torch.randn(()) * 0.03), 0.0, 1.0)
        if torch.rand(()) < 0.1:
            contrast = float(torch.empty(()).uniform_(0.9, 1.1))
            mean = float(image.mean())
            image = np.clip((image - mean) * contrast + mean, 0.0, 1.0)
        return image.astype(np.float32)

    def __getitem__(self, index: int) -> dict[str, object]:
        path = self.files[int(index)]
        with np.load(path) as data:
            image = normalize_minmax(np.asarray(data["image"], dtype=np.float32))
            label = np.asarray(data["label"], dtype=np.int64)
        if image.ndim == 3:
            image = image[0]
        label = np.asarray(label, dtype=np.uint8)
        image, label = self._resize(image, label)
        image, label = self._augment_spatial(image, label)
        image = self._augment_intensity(image)

        center = path.parent.name if path.parent != self.split_dir else self.split
        case, slice_idx = parse_case_slice(path)
        patient_id = f"{center}__{case}" if center != self.split else case
        return {
            "image": torch.from_numpy(np.ascontiguousarray(image)).unsqueeze(0).float(),
            "target": torch.from_numpy(np.ascontiguousarray(label)).long(),
            "center": center,
            "patient_id": patient_id,
            "slice_idx": torch.tensor(slice_idx, dtype=torch.long),
            "path": str(path),
        }
