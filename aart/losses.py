from __future__ import annotations

import torch
import torch.nn.functional as F


def _one_hot(target: torch.Tensor, num_classes: int) -> torch.Tensor:
    return F.one_hot(target.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()


def foreground_dice_loss(
    probabilities: torch.Tensor,
    target: torch.Tensor,
    class_indices: tuple[int, ...] = (1, 2, 3),
    eps: float = 1e-5,
) -> torch.Tensor:
    target_oh = _one_hot(target, probabilities.shape[1])
    losses = []
    for cls in class_indices:
        pred_c = probabilities[:, cls]
        target_c = target_oh[:, cls]
        inter = (pred_c * target_c).sum(dim=(1, 2))
        denom = pred_c.sum(dim=(1, 2)) + target_c.sum(dim=(1, 2))
        losses.append(1.0 - ((2.0 * inter + eps) / (denom + eps)))
    return torch.stack(losses, dim=1).mean()


def ce_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    class_weights: torch.Tensor | None = None,
    dice_classes: tuple[int, ...] = (1, 2, 3),
    lambda_ce: float = 1.0,
    lambda_dice: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    ce = F.cross_entropy(logits, target.long(), weight=class_weights)
    prob = torch.softmax(logits, dim=1)
    dice = foreground_dice_loss(prob, target, class_indices=dice_classes)
    total = float(lambda_ce) * ce + float(lambda_dice) * dice
    return total, {"loss": total.detach(), "ce": ce.detach(), "dice_loss": dice.detach()}
