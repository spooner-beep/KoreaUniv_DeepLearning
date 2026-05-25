"""
학습된 CNN으로 데이터셋 전체 이미지를 512-dim 벡터로 변환하여 저장.
추천 시스템의 인덱스 역할.

실행: python extract_features.py --model resnet18
"""
import torch
import numpy as np
import json
from torch.utils.data import DataLoader
import argparse
import config
from dataset import FashionEraDataset
from utils.device import get_device
from utils.model_loader import load_model


@torch.no_grad()
def extract_all(model, device):
    all_feats, all_paths, all_eras = [], [], []

    for split in ["train", "val", "test"]:
        ds = FashionEraDataset(split, return_path=True)
        loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=config.NUM_WORKERS)

        model.eval()
        for imgs, labels, paths in loader:
            feats = model.encode(imgs.to(device)).cpu().numpy()
            all_feats.append(feats)
            all_paths.extend(paths)
            all_eras.extend(labels.tolist())

    return np.vstack(all_feats), all_paths, all_eras


def main(model_type):
    device = get_device()
    print(f"Device: {device}")

    model = load_model(model_type, device)
    print(f"체크포인트 로드: {config.CHECKPOINT_DIR / f'best_{model_type}.pth'}")

    features, paths, eras = extract_all(model, device)
    print(f"추출 완료: {features.shape[0]}개 이미지, {features.shape[1]}-dim 벡터")

    np.save(config.FEATURE_NPY, features)
    with open(config.FEATURE_META, "w", encoding="utf-8") as f:
        json.dump({"paths": paths, "eras": eras}, f, ensure_ascii=False)

    print(f"저장 완료: {config.FEATURE_NPY}, {config.FEATURE_META}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["custom", "resnet18"], default="custom")
    args = parser.parse_args()
    main(args.model)
