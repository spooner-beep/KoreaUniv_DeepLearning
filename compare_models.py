"""
CustomCNN vs ResNet18 벤치마크 비교 (Train Acc / Val Acc).

EC2 학습 후 검증 단계에서 실행:
  python compare_models.py
  python compare_models.py --json-only
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime

import torch
import config
from dataset import FashionEraDataset
from hyperparams import resnet18_baseline_val_acc
from utils.dataloader import make_dataloader
from utils.device import get_device
from utils.model_loader import load_model
from train import validate


@torch.no_grad()
def eval_split(model, split: str, device, batch_size: int, augment: str = "default"):
    ds = FashionEraDataset(split, augment=augment)
    loader = make_dataloader(ds, batch_size, shuffle=False)
    criterion = torch.nn.CrossEntropyLoss()
    _, acc = validate(model, loader, criterion, device)
    return acc, len(ds)


def compare():
    device = get_device()
    batch_size = config.BATCH_SIZE
    rows = []

    for model_type in ("resnet18", "custom"):
        try:
            model = load_model(model_type, device)
        except FileNotFoundError as e:
            print(f"[skip] {model_type}: {e}")
            continue

        # 비교 시 Train Acc는 증강 없이(default transform) 측정
        train_acc, n_train = eval_split(model, "train", device, batch_size, "default")
        val_acc, n_val = eval_split(model, "val", device, batch_size, "default")
        gap = train_acc - val_acc

        rows.append({
            "model": model_type,
            "train_acc": round(train_acc, 4),
            "val_acc": round(val_acc, 4),
            "train_val_gap": round(gap, 4),
            "n_train": n_train,
            "n_val": n_val,
        })
        print(
            f"{model_type:10s} | Train Acc: {train_acc:.4f} ({n_train}) | "
            f"Val Acc: {val_acc:.4f} ({n_val}) | gap: {gap:+.4f}"
        )

    if len(rows) == 2:
        custom, resnet = rows[1], rows[0]
        winner = "custom" if custom["val_acc"] > resnet["val_acc"] else "resnet18"
        beat = custom["val_acc"] > resnet["val_acc"]
        print(
            f"\nVal Acc 차이 (Custom - ResNet): {custom['val_acc'] - resnet['val_acc']:+.4f}"
        )
        target = resnet18_baseline_val_acc()
        print(f"ResNet18 기준선 (이번 학습): {target:.4f}")
        print(f"CustomCNN ResNet 상회: {'YES' if beat else 'NO'}")
        print(f"권장 추천 모델: {winner}")
    elif len(rows) == 1:
        winner = rows[0]["model"]
        beat = False
    else:
        winner = None
        beat = False

    summary = {
        "timestamp": datetime.now().isoformat(),
        "device": str(device),
        "resnet18_baseline": resnet18_baseline_val_acc(),
        "models": rows,
        "custom_beats_resnet": beat,
        "recommended_for_recommend": winner,
    }
    out = config.RESULTS_DIR / "model_comparison.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n저장: {out}")
    return summary


def main():
    argparse.ArgumentParser(description="CustomCNN vs ResNet18 비교").parse_args()
    compare()


if __name__ == "__main__":
    main()
