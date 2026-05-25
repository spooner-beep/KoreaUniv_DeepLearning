"""체크포인트·하이퍼파라미터 기반 모델 로드."""
from pathlib import Path

import torch.nn as nn

import config
from hyperparams import CustomCNNHyperParams
from model import build_model
from utils.checkpoint import load_state_dict

HP_PATH = config.CHECKPOINT_DIR / "custom_hparams.json"


def load_model(model_type: str, device) -> nn.Module:
    if model_type == "custom":
        hp = (
            CustomCNNHyperParams.load(HP_PATH)
            if HP_PATH.is_file()
            else CustomCNNHyperParams.tuned_v2()
        )
        model = build_model("custom", hp)
    else:
        model = build_model("resnet18", hp=None, pretrained=False)

    ckpt = config.CHECKPOINT_DIR / f"best_{model_type}.pth"
    model.load_state_dict(load_state_dict(ckpt, map_location=device))
    return model.to(device)
