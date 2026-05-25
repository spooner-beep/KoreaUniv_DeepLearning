"""
사용자 입력 이미지와 가장 어울리는 코디 이미지를 추천.
학습된 CNN으로 스타일 벡터 추출 → 코사인 유사도 → Top-K 반환.

실행: python recommend.py --image path/to/your_clothes.jpg --model resnet18 --top_k 5
"""
import torch
import numpy as np
import json
import argparse
from utils.fonts import setup_matplotlib

setup_matplotlib()
import matplotlib.pyplot as plt
from PIL import Image
import config
from dataset import get_transforms
from utils.device import get_device
from utils.model_loader import load_model


def cosine_similarity(query, index):
    query = query / (np.linalg.norm(query) + 1e-8)
    norms = np.linalg.norm(index, axis=1, keepdims=True) + 1e-8
    return (index / norms) @ query


def load_index():
    features = np.load(config.FEATURE_NPY)
    with open(config.FEATURE_META, encoding="utf-8") as f:
        meta = json.load(f)
    return features, meta["paths"], meta["eras"]


def recommend(img_path, model, device, top_k=5, era_filter=2019):
    features, paths, eras = load_index()

    transform = get_transforms("test")
    img = transform(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        query_feat = model.encode(img).cpu().numpy()[0]

    eras_arr = np.array(eras)
    if era_filter:
        era_idx = config.ERA_TO_IDX[era_filter]
        mask = eras_arr == era_idx
        filtered_feats = features[mask]
        filtered_paths = [p for p, m in zip(paths, mask.tolist()) if m]
        filtered_eras  = eras_arr[mask].tolist()
    else:
        filtered_feats = features
        filtered_paths = paths
        filtered_eras  = eras

    sims = cosine_similarity(query_feat, filtered_feats)
    sorted_idx = np.argsort(sims)[::-1]

    results, seen = [], set()
    for i in sorted_idx:
        p = filtered_paths[i]
        if p not in seen:
            seen.add(p)
            results.append((p, config.IDX_TO_ERA[filtered_eras[i]], float(sims[i])))
        if len(results) == top_k:
            break

    return results


def visualize(query_path, results):
    n = len(results) + 1
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 5))

    axes[0].imshow(Image.open(query_path).convert("RGB"))
    axes[0].set_title("입력 이미지", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    for i, (path, era, sim) in enumerate(results, 1):
        axes[i].imshow(Image.open(path).convert("RGB"))
        axes[i].set_title(f"추천 {i}\n{era}년대  sim={sim:.3f}", fontsize=9)
        axes[i].axis("off")

    plt.suptitle("패션 코디 추천", fontsize=13)
    plt.tight_layout()
    out = config.RESULTS_DIR / "recommendation_result.png"
    plt.savefig(out, dpi=150)
    if config.SHOW_PLOTS:
        plt.show()
    else:
        plt.close()
    print(f"{out} 저장 완료")


def main(img_path, model_type, top_k, era_filter):
    device = get_device()
    print(f"Device: {device}")

    model = load_model(model_type, device).eval()

    results = recommend(img_path, model, device, top_k=top_k, era_filter=era_filter)

    print(f"\n[ 입력: {img_path} ]\n")
    for rank, (path, era, sim) in enumerate(results, 1):
        print(f"  {rank}위  유사도: {sim:.4f}  시대: {era}  {path}")

    visualize(img_path, results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",    required=True,  help="입력 의류 이미지 경로")
    parser.add_argument("--model",    choices=["custom", "resnet18"], default="custom")
    parser.add_argument("--top_k",    type=int,  default=5)
    parser.add_argument("--era",      type=int,  default=2019, help="추천 기준 연도 (0=전체)")
    args = parser.parse_args()
    main(args.image, args.model, args.top_k, args.era or None)
