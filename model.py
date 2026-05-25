import torch
import torch.nn as nn
from torchvision import models
import config


class ConvNormAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation: 시대별 색감/실루엣 채널을 동적으로 강조."""

    def __init__(self, channels: int, ratio: float = 0.25):
        super().__init__()
        hidden = max(16, int(channels * ratio))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(self.pool(x))


class ConvBlock(nn.Module):
    """기존 단순 블록 (baseline_v1 호환)."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.block(x)


class ResidualConvBlock(nn.Module):
    """
    2×Conv + skip + MaxPool + Dropout2d.
    잔차 연결로 gradient 흐름 개선 (ResNet18 대비 scratch 학습 안정화).
    """

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.15):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.skip = (
            nn.Conv2d(in_ch, out_ch, 1, bias=False)
            if in_ch != out_ch
            else nn.Identity()
        )
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        out = self.relu(self.conv(x) + self.skip(x))
        out = self.pool(out)
        return self.drop(out)


class ModernResidualBlock(nn.Module):
    """
    Conv-BN-SiLU 기반 residual block + SE.
    MaxPool 대신 stride conv로 downsampling 하여 정보 손실을 줄인다.
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, dropout: float = 0.05):
        super().__init__()
        self.conv1 = ConvNormAct(in_ch, out_ch, 3, stride)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.se = SEBlock(out_ch)
        self.skip = (
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
            if stride != 1 or in_ch != out_ch
            else nn.Identity()
        )
        self.act = nn.SiLU(inplace=True)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        out = self.conv2(self.conv1(x))
        out = self.se(out)
        out = self.act(out + self.skip(x))
        return self.drop(out)


class CustomCNN(nn.Module):
    """
    패션 시대 분류용 Custom CNN.

    tuned_v2 (기본):
      - ResidualConvBlock × 4, channels (64→512)
      - encoder/head 분리 Dropout
    baseline_v1:
      - ConvBlock × 4, channels (32→256) — 구 아키텍처 호환
    """

    def __init__(
        self,
        num_classes: int = config.NUM_CLASSES,
        channels: tuple[int, ...] = (64, 128, 256, 512),
        use_residual: bool = True,
        conv_dropout: float = 0.15,
        encoder_dropout: float = 0.4,
        head_dropout: float = 0.25,
        architecture: str = "classic",
        blocks_per_stage: tuple[int, ...] = (2, 2, 2, 2),
    ):
        super().__init__()
        self.architecture = architecture
        self.use_residual = use_residual

        if architecture == "modern_v4":
            self.features, feat_dim = self._build_modern_encoder(
                channels=channels,
                blocks_per_stage=blocks_per_stage,
                dropout=conv_dropout,
            )
            pooled_dim = feat_dim * 2
            self.encoder = nn.Sequential(
                nn.Flatten(),
                nn.Linear(pooled_dim, 768),
                nn.BatchNorm1d(768),
                nn.SiLU(inplace=True),
                nn.Dropout(encoder_dropout),
            )
            self.head = nn.Sequential(
                nn.Dropout(head_dropout) if head_dropout > 0 else nn.Identity(),
                nn.Linear(768, num_classes),
            )
            return

        in_ch = 3
        blocks = []
        for out_ch in channels:
            if use_residual:
                blocks.append(ResidualConvBlock(in_ch, out_ch, conv_dropout))
            else:
                blocks.append(ConvBlock(in_ch, out_ch))
            in_ch = out_ch
        self.features = nn.Sequential(*blocks)
        feat_dim = channels[-1]

        self.encoder = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(encoder_dropout),
        )
        self.head = nn.Sequential(
            nn.Dropout(head_dropout) if head_dropout > 0 else nn.Identity(),
            nn.Linear(512, num_classes),
        )

    def encode(self, x):
        features = self.features(x)
        if self.architecture == "modern_v4":
            avg = nn.functional.adaptive_avg_pool2d(features, 1)
            mx = nn.functional.adaptive_max_pool2d(features, 1)
            return self.encoder(torch.cat([avg, mx], dim=1))
        return self.encoder(features)

    def forward(self, x):
        return self.head(self.encode(x))

    @staticmethod
    def _build_modern_encoder(
        channels: tuple[int, ...],
        blocks_per_stage: tuple[int, ...],
        dropout: float,
    ) -> tuple[nn.Sequential, int]:
        if len(blocks_per_stage) != len(channels):
            raise ValueError("blocks_per_stage length must match channels length")

        layers: list[nn.Module] = [
            ConvNormAct(3, channels[0], 3, stride=2),
            ConvNormAct(channels[0], channels[0], 3, stride=1),
        ]
        in_ch = channels[0]
        for stage_idx, (out_ch, depth) in enumerate(zip(channels, blocks_per_stage)):
            for block_idx in range(depth):
                stride = 2 if block_idx == 0 and stage_idx > 0 else 1
                layers.append(ModernResidualBlock(in_ch, out_ch, stride=stride, dropout=dropout))
                in_ch = out_ch
        return nn.Sequential(*layers), channels[-1]


class ResNet18(nn.Module):
    """ImageNet 사전학습 ResNet18 (비교·베이스라인)."""

    def __init__(self, num_classes=config.NUM_CLASSES, pretrained=True):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        base = models.resnet18(weights=weights)
        self.backbone = nn.Sequential(*list(base.children())[:-1])
        self.head = nn.Linear(512, num_classes)

    def encode(self, x):
        return self.backbone(x).flatten(1)

    def forward(self, x):
        return self.head(self.encode(x))


def build_model(model_type: str, hp=None, pretrained: bool = True) -> nn.Module:
    if model_type == "custom":
        kwargs = hp.to_model_kwargs() if hp is not None else {}
        return CustomCNN(**kwargs)
    if model_type == "resnet18":
        return ResNet18(pretrained=pretrained)
    raise ValueError(f"Unknown model: {model_type}")
