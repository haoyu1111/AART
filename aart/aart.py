from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


AART8_BRANCHES: tuple[str, ...] = (
    "inner",
    "blood_side_mismatch",
    "radial_contrast",
    "tangent_mean",
    "tangent_range",
    "tangent_persistence",
    "blob_mean",
    "blob_range",
)


@dataclass(frozen=True)
class AARTKernelSpec:
    radial_steps: tuple[int, ...] = (-2, -1, 0, 1, 2)
    tangent_steps: tuple[int, ...] = (-3, -2, -1, 1, 2, 3)
    blob_radial_steps: tuple[int, ...] = (-1, 0, 1)
    blob_tangent_steps: tuple[int, ...] = (-1, 0, 1)
    dilation: float = 1.0
    center_detach: bool = True
    eps: float = 1e-6


def _mesh_xy(height: int, width: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    yy, xx = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    return xx[None, None], yy[None, None]


def soft_feature_center(x: torch.Tensor, detach: bool = True, eps: float = 1e-6) -> torch.Tensor:
    """Return the feature-energy barycenter in x-y pixel coordinates."""

    if x.ndim != 4:
        raise ValueError(f"x must have shape [B,C,H,W], got {tuple(x.shape)}")
    source = x.detach() if detach else x
    b, _, h, w = source.shape
    weight = source.float().abs().mean(dim=1, keepdim=True)
    xx, yy = _mesh_xy(h, w, source.device, source.dtype)
    denom = weight.sum(dim=(2, 3)).clamp_min(eps)
    cx = (weight * xx).sum(dim=(2, 3)) / denom
    cy = (weight * yy).sum(dim=(2, 3)) / denom
    return torch.cat([cx, cy], dim=1).view(b, 2)


def radial_tangent_frame(
    x: torch.Tensor,
    center: torch.Tensor | None = None,
    center_detach: bool = True,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build LV-centered radial and tangential unit vectors for each feature location."""

    b, _, h, w = x.shape
    if center is None:
        center = soft_feature_center(x, detach=center_detach, eps=eps)
    center = center.to(device=x.device, dtype=x.dtype)

    xx, yy = _mesh_xy(h, w, x.device, x.dtype)
    xx = xx.expand(b, 1, h, w)
    yy = yy.expand(b, 1, h, w)
    cx = center[:, 0].view(b, 1, 1, 1)
    cy = center[:, 1].view(b, 1, 1, 1)

    dx = xx - cx
    dy = yy - cy
    rho = torch.sqrt(dx.square() + dy.square() + eps)
    e_rx = dx / rho
    e_ry = dy / rho
    e_tx = -e_ry
    e_ty = e_rx
    return xx, yy, e_rx, e_ry, e_tx, e_ty


def sample_rt_offsets(
    x: torch.Tensor,
    offsets_rt: Iterable[tuple[float, float]],
    center: torch.Tensor | None = None,
    dilation: float = 1.0,
    center_detach: bool = True,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Sample features at radial-tangential offsets.

    Args:
        x: Input tensor with shape [B,C,H,W].
        offsets_rt: Sequence of (radial_step, tangential_step) offsets.

    Returns:
        Tensor with shape [B,C,N,H,W], where N is the number of offsets.
    """

    offsets = list(offsets_rt)
    if not offsets:
        raise ValueError("offsets_rt must contain at least one offset")

    b, c, h, w = x.shape
    xx, yy, e_rx, e_ry, e_tx, e_ty = radial_tangent_frame(x, center, center_detach, eps)
    grids = []
    scale = float(dilation)
    for radial_step, tangent_step in offsets:
        sx = xx + scale * (float(radial_step) * e_rx + float(tangent_step) * e_tx)
        sy = yy + scale * (float(radial_step) * e_ry + float(tangent_step) * e_ty)
        gx = 2.0 * sx / float(w - 1) - 1.0 if w > 1 else torch.zeros_like(sx)
        gy = 2.0 * sy / float(h - 1) - 1.0 if h > 1 else torch.zeros_like(sy)
        grids.append(torch.cat([gx, gy], dim=1).permute(0, 2, 3, 1))

    grid = torch.cat(grids, dim=0)
    sampled = F.grid_sample(
        x.repeat(len(offsets), 1, 1, 1),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return sampled.view(len(offsets), b, c, h, w).permute(1, 2, 0, 3, 4).contiguous()


def _group_count(channels: int, max_groups: int = 8) -> int:
    for group in range(min(max_groups, channels), 0, -1):
        if channels % group == 0:
            return group
    return 1


class AARTConv2d(nn.Module):
    """Eight-branch anatomy-aligned radial-tangential convolution."""

    valid_branch_names = AART8_BRANCHES

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_spec: AARTKernelSpec | None = None,
        branch_names: tuple[str, ...] = AART8_BRANCHES,
        gate_hidden: int | None = None,
    ) -> None:
        super().__init__()
        self.kernel_spec = kernel_spec or AARTKernelSpec()
        self.branch_names = tuple(branch_names)
        unknown = sorted(set(self.branch_names) - set(self.valid_branch_names))
        if unknown:
            raise ValueError(f"Unknown AART branch name(s): {unknown}")
        if not self.branch_names:
            raise ValueError("AARTConv2d requires at least one branch")

        self.branch_multiplier = len(self.branch_names)
        hidden = int(gate_hidden or max(8, self.branch_multiplier // 2))
        self.branch_gate = nn.Sequential(
            nn.Conv2d(self.branch_multiplier + 1, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, self.branch_multiplier, kernel_size=1),
        )
        nn.init.zeros_(self.branch_gate[-1].weight)
        nn.init.zeros_(self.branch_gate[-1].bias)

        self.fuse = nn.Conv2d(in_channels * self.branch_multiplier, out_channels, kernel_size=1, bias=False)
        self.last_center: torch.Tensor | None = None
        self.center_override: torch.Tensor | None = None
        self.branch_scales: dict[str, float] = {}

    def set_branch_scales(self, scales: dict[str, float] | None = None) -> None:
        self.branch_scales = dict(scales or {})

    def _scale_branch(self, name: str, feature: torch.Tensor) -> torch.Tensor:
        return feature * float(self.branch_scales.get(name, 1.0))

    def _rho_norm(self, x: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        xx, yy = _mesh_xy(h, w, x.device, x.dtype)
        xx = xx.expand(b, 1, h, w)
        yy = yy.expand(b, 1, h, w)
        cx = center[:, 0].view(b, 1, 1, 1)
        cy = center[:, 1].view(b, 1, 1, 1)
        rho = torch.sqrt((xx - cx).square() + (yy - cy).square() + self.kernel_spec.eps)
        return (rho / float(max(h, w, 1))).clamp(0.0, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spec = self.kernel_spec
        if self.center_override is None:
            center = soft_feature_center(x, detach=spec.center_detach, eps=spec.eps)
        else:
            center = self.center_override.to(device=x.device, dtype=x.dtype)
            if spec.center_detach:
                center = center.detach()
        self.last_center = center.detach()

        requested = set(self.branch_names)
        branch_map: dict[str, torch.Tensor] = {}

        if requested & {"inner", "blood_side_mismatch", "radial_contrast"}:
            radial = sample_rt_offsets(
                x,
                [(step, 0.0) for step in spec.radial_steps],
                center=center,
                dilation=spec.dilation,
                center_detach=spec.center_detach,
                eps=spec.eps,
            )
            inner = radial[:, :, :2].mean(dim=2)
            if "inner" in requested:
                branch_map["inner"] = inner
            if "blood_side_mismatch" in requested:
                branch_map["blood_side_mismatch"] = (x - inner).abs()
            if "radial_contrast" in requested:
                outer = radial[:, :, -2:].mean(dim=2)
                branch_map["radial_contrast"] = outer - inner

        if requested & {"tangent_mean", "tangent_range", "tangent_persistence"}:
            tangent = sample_rt_offsets(
                x,
                [(0.0, step) for step in spec.tangent_steps],
                center=center,
                dilation=spec.dilation,
                center_detach=spec.center_detach,
                eps=spec.eps,
            )
            if "tangent_mean" in requested:
                branch_map["tangent_mean"] = tangent.mean(dim=2)
            if "tangent_range" in requested:
                branch_map["tangent_range"] = tangent.max(dim=2).values - tangent.min(dim=2).values
            if "tangent_persistence" in requested:
                tangent_abs = tangent.abs()
                branch_map["tangent_persistence"] = F.relu(
                    tangent_abs.mean(dim=2)
                    - 0.25 * (tangent_abs.max(dim=2).values - tangent_abs.min(dim=2).values)
                )

        if requested & {"blob_mean", "blob_range"}:
            blob_offsets = [
                (r, t)
                for r in spec.blob_radial_steps
                for t in spec.blob_tangent_steps
                if not (int(r) == 0 and int(t) == 0)
            ]
            blob = sample_rt_offsets(
                x,
                blob_offsets,
                center=center,
                dilation=spec.dilation,
                center_detach=spec.center_detach,
                eps=spec.eps,
            )
            if "blob_mean" in requested:
                branch_map["blob_mean"] = blob.mean(dim=2)
            if "blob_range" in requested:
                branch_map["blob_range"] = blob.max(dim=2).values - blob.min(dim=2).values

        selected = [self._scale_branch(name, branch_map[name]) for name in self.branch_names]
        gate_input = torch.cat(
            [branch_map[name].detach().float().abs().mean(dim=1, keepdim=True) for name in self.branch_names]
            + [self._rho_norm(x, center).detach().float()],
            dim=1,
        ).to(dtype=x.dtype)
        gate = torch.softmax(self.branch_gate(gate_input), dim=1) * float(self.branch_multiplier)
        weighted = [feature * gate[:, idx : idx + 1] for idx, feature in enumerate(selected)]
        return self.fuse(torch.cat(weighted, dim=1))


class AARTConvNormAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dilation: float = 1.0) -> None:
        super().__init__()
        spec = AARTKernelSpec(dilation=dilation)
        self.aart = AARTConv2d(in_channels, out_channels, kernel_spec=spec)
        self.norm = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.aart(x)))


class AARTResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: float = 1.0) -> None:
        super().__init__()
        self.conv1 = AARTConvNormAct(channels, channels, dilation=dilation)
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(channels), channels),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.conv2(self.conv1(x)))


class AARTUpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, dilation: float = 1.0) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            AARTConvNormAct(in_channels + skip_channels, out_channels, dilation=dilation),
            AARTResidualBlock(out_channels, dilation=dilation),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class AARTUNet2D(nn.Module):
    """AART-ResUNet used for four-class LGE-CMR segmentation."""

    def __init__(self, in_channels: int = 1, num_classes: int = 4, base_channels: int = 32) -> None:
        super().__init__()
        c = int(base_channels)
        self.stem = nn.Sequential(
            AARTConvNormAct(in_channels, c),
            AARTResidualBlock(c),
        )
        self.down1 = nn.Sequential(nn.MaxPool2d(2), AARTConvNormAct(c, c * 2), AARTResidualBlock(c * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), AARTConvNormAct(c * 2, c * 4), AARTResidualBlock(c * 4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), AARTConvNormAct(c * 4, c * 8), AARTResidualBlock(c * 8))
        self.bridge = nn.Sequential(AARTConvNormAct(c * 8, c * 8), AARTResidualBlock(c * 8))
        self.up2 = AARTUpBlock(c * 8, c * 4, c * 4)
        self.up1 = AARTUpBlock(c * 4, c * 2, c * 2)
        self.up0 = AARTUpBlock(c * 2, c, c)
        self.head = nn.Conv2d(c, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.stem(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        xb = self.bridge(x3)
        y = self.up2(xb, x2)
        y = self.up1(y, x1)
        y = self.up0(y, x0)
        return self.head(y)


def build_aart_resunet(in_channels: int = 1, num_classes: int = 4, base_channels: int = 32) -> AARTUNet2D:
    return AARTUNet2D(in_channels=in_channels, num_classes=num_classes, base_channels=base_channels)
