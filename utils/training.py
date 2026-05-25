"""학습 루프 보조: Mixup, EMA, 클래스 가중치."""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def mixup_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    if alpha <= 0:
        return images, labels, labels, 1.0
    lam = float(np.random.beta(alpha, alpha))
    lam = max(lam, 1.0 - lam)
    idx = torch.randperm(images.size(0), device=images.device)
    mixed = lam * images + (1.0 - lam) * images[idx]
    return mixed, labels, labels[idx], lam


def mixup_loss(
    criterion: nn.Module,
    outputs: torch.Tensor,
    targets_a: torch.Tensor,
    targets_b: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    return lam * criterion(outputs, targets_a) + (1.0 - lam) * criterion(outputs, targets_b)


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """KL 기반 soft-target distillation loss."""
    t = max(float(temperature), 1.0)
    return F.kl_div(
        F.log_softmax(student_logits / t, dim=1),
        F.softmax(teacher_logits / t, dim=1),
        reduction="batchmean",
    ) * (t * t)


def compute_class_weights(dataset, num_classes: int, device: torch.device) -> torch.Tensor:
    counts = np.zeros(num_classes, dtype=np.float64)
    for _, label in dataset.samples:
        counts[label] += 1
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


class ModelEMA:
    """검증·저장 시 EMA 가중치 사용 → 일반화 성능 향상."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        shadow_state = self.shadow.state_dict()
        model_state = model.state_dict()
        for name, value in model_state.items():
            shadow_value = shadow_state[name]
            if torch.is_floating_point(shadow_value):
                shadow_value.mul_(self.decay).add_(value, alpha=1.0 - self.decay)
            else:
                shadow_value.copy_(value)

    def state_dict(self):
        return self.shadow.state_dict()
