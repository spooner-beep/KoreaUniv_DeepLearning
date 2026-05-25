"""
CustomCNN 학습·구조 하이퍼파라미터.

목표: ResNet18 Validation Accuracy (~44.1%) 를 CustomCNN 으로 상회.
기준선: CustomCNN 37.4% (Epoch 9 이후 과적합) — report.md 참고.

사용:
  from hyperparams import CustomCNNHyperParams
  hp = CustomCNNHyperParams.tuned_v2()
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import config


@dataclass
class CustomCNNHyperParams:
    """CustomCNN 전용 학습·모델 설정 (ResNet 경로와 분리)."""

    # ── 모델 구조 ─────────────────────────────────────────────
    channels: tuple[int, ...] = (64, 128, 256, 512)
    use_residual: bool = True
    conv_dropout: float = 0.15
    encoder_dropout: float = 0.4
    head_dropout: float = 0.25
    architecture: str = "classic"  # classic | modern_v4
    blocks_per_stage: tuple[int, ...] = (2, 2, 2, 2)

    # ── 옵티마이저 / 스케줄 ───────────────────────────────────
    optimizer: str = "adamw"
    lr: float = 3e-4
    weight_decay: float = 0.05
    max_epochs: int = 60
    patience: int = 15
    scheduler: str = "onecycle"  # onecycle | cosine
    grad_clip_norm: float = 1.0

    # ── 손실 / 정규화 ─────────────────────────────────────────
    label_smoothing: float = 0.12
    mixup_alpha: float = 0.3
    use_ema: bool = True
    ema_decay: float = 0.999
    use_distillation: bool = False
    distill_alpha: float = 0.65
    distill_temperature: float = 3.0

    # ── 데이터 ────────────────────────────────────────────────
    batch_size: int = 32
    augment: str = "strong"  # default | strong
    use_class_weights: bool = True

    @classmethod
    def tuned_v2(cls) -> CustomCNNHyperParams:
        """과적합 완화 + 표현력 확대를 반영한 권장 프리셋."""
        return cls()

    @classmethod
    def tuned_v3(cls) -> CustomCNNHyperParams:
        """
        ResNet18 상회 목표 + 로컬 GPU 학습 속도 균형.
        - 증강: default (ResNet과 동일, CPU 부하 최소)
        - 정규화: Mixup·EMA·클래스가중치·잔차 CNN으로 Val 일반화
        - epoch 35 + early stopping (과도한 50~80 epoch 방지)
        """
        return cls(
            channels=(64, 128, 256, 512),
            use_residual=True,
            conv_dropout=0.12,
            encoder_dropout=0.35,
            head_dropout=0.2,
            lr=3e-4,
            weight_decay=0.05,
            max_epochs=40,
            patience=12,
            scheduler="cosine",
            grad_clip_norm=1.0,
            label_smoothing=0.1,
            mixup_alpha=0.15,
            use_ema=True,
            ema_decay=0.999,
            batch_size=config.BATCH_SIZE,
            augment="default",
            use_class_weights=True,
        )

    @classmethod
    def tuned_v3_heavy(cls) -> CustomCNNHyperParams:
        """느리지만 증강 강함 (strong_fast). 시간 여유 있을 때만."""
        hp = cls.tuned_v3()
        hp.max_epochs = 50
        hp.patience = 15
        hp.augment = "strong_fast"
        hp.mixup_alpha = 0.25
        return hp

    @classmethod
    def tuned_v4(cls) -> CustomCNNHyperParams:
        """
        ResNet18과 동일한 30 epoch 안에서 학습하는 CustomCNN.

        - modern_v4: residual + SE + avg/max pooling으로 표현력 강화
        - default augmentation으로 CPU 부하 최소화
        - ResNet18은 teacher로만 사용; 최종 모델은 CustomCNN 구조/파라미터만 저장
        """
        return cls(
            channels=(48, 96, 192, 320, 512),
            blocks_per_stage=(1, 2, 2, 2, 1),
            architecture="modern_v4",
            use_residual=True,
            conv_dropout=0.05,
            encoder_dropout=0.25,
            head_dropout=0.15,
            lr=3e-4,
            weight_decay=0.03,
            max_epochs=30,
            patience=8,
            scheduler="cosine",
            grad_clip_norm=1.0,
            label_smoothing=0.05,
            mixup_alpha=0.1,
            use_ema=True,
            ema_decay=0.999,
            use_distillation=True,
            distill_alpha=0.65,
            distill_temperature=3.0,
            batch_size=config.BATCH_SIZE,
            augment="default",
            use_class_weights=False,
        )

    @classmethod
    def tuned_v4_scratch(cls) -> CustomCNNHyperParams:
        """Teacher 없이 동일 구조를 scratch로 학습하는 비교용 프리셋."""
        hp = cls.tuned_v4()
        hp.use_distillation = False
        hp.distill_alpha = 0.0
        return hp

    @classmethod
    def local_smoke(cls) -> CustomCNNHyperParams:
        """로컬 PC 동작 확인용 (짧은 epoch, 작은 batch)."""
        hp = cls.tuned_v2()
        hp.max_epochs = 2
        hp.patience = 2
        hp.batch_size = 8
        hp.use_ema = False
        hp.mixup_alpha = 0.0
        return hp

    @classmethod
    def baseline_v1(cls) -> CustomCNNHyperParams:
        """기존 train.py 와 유사한 설정 (비교 실험용)."""
        return cls(
            channels=(32, 64, 128, 256),
            use_residual=False,
            conv_dropout=0.0,
            encoder_dropout=0.5,
            head_dropout=0.0,
            optimizer="adam",
            lr=1e-3,
            weight_decay=1e-4,
            max_epochs=30,
            patience=10,
            scheduler="cosine",
            grad_clip_norm=0.0,
            label_smoothing=0.1,
            mixup_alpha=0.0,
            use_ema=False,
            augment="default",
            use_class_weights=False,
        )

    def to_model_kwargs(self) -> dict[str, Any]:
        return {
            "num_classes": config.NUM_CLASSES,
            "channels": self.channels,
            "use_residual": self.use_residual,
            "conv_dropout": self.conv_dropout,
            "encoder_dropout": self.encoder_dropout,
            "head_dropout": self.head_dropout,
            "architecture": self.architecture,
            "blocks_per_stage": self.blocks_per_stage,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> CustomCNNHyperParams:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data.get("channels"), list):
            data["channels"] = tuple(data["channels"])
        if isinstance(data.get("blocks_per_stage"), list):
            data["blocks_per_stage"] = tuple(data["blocks_per_stage"])
        valid = cls.__dataclass_fields__
        return cls(**{k: v for k, v in data.items() if k in valid})


# ResNet18 비교 목표 (문서 기준선; 실제 학습 후에는 resnet18_train_log.json 우선)
RESNET18_TARGET_VAL_ACC = 0.441


def resnet18_baseline_val_acc() -> float:
    log_path = config.RESULTS_DIR / "resnet18_train_log.json"
    if log_path.exists():
        data = json.loads(log_path.read_text(encoding="utf-8"))
        return float(data.get("best_val_acc", RESNET18_TARGET_VAL_ACC))
    return RESNET18_TARGET_VAL_ACC

# tune_custom.py 그리드 후보
TUNE_GRID: list[dict[str, Any]] = [
    {"lr": 2e-4, "weight_decay": 0.03, "mixup_alpha": 0.2, "encoder_dropout": 0.35},
    {"lr": 3e-4, "weight_decay": 0.05, "mixup_alpha": 0.3, "encoder_dropout": 0.4},
    {"lr": 5e-4, "weight_decay": 0.05, "mixup_alpha": 0.3, "encoder_dropout": 0.45},
    {"lr": 3e-4, "weight_decay": 0.08, "mixup_alpha": 0.4, "encoder_dropout": 0.5},
    {"lr": 3e-4, "weight_decay": 0.05, "mixup_alpha": 0.2, "encoder_dropout": 0.4, "conv_dropout": 0.2},
]
