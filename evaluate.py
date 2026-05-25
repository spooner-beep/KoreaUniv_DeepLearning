import torch
import numpy as np
from utils.fonts import setup_matplotlib

setup_matplotlib()
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader
import argparse
import config
from dataset import FashionEraDataset
from utils.device import get_device
from utils.model_loader import load_model


@torch.no_grad()
def get_predictions(model, loader, device):
    model.eval()
    all_preds, all_labels, all_imgs = [], [], []
    for imgs, labels in loader:
        preds = model(imgs.to(device)).argmax(1).cpu()
        all_preds.append(preds)
        all_labels.append(labels)
        all_imgs.append(imgs)
    return torch.cat(all_preds), torch.cat(all_labels), torch.cat(all_imgs)


def plot_confusion_matrix(preds, labels, model_type):
    cm = confusion_matrix(labels, preds)
    era_names = [str(e) for e in config.TARGET_ERAS]
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=era_names, yticklabels=era_names)
    plt.xlabel("예측")
    plt.ylabel("실제")
    plt.title(f"Confusion Matrix ({model_type})")
    plt.tight_layout()
    path = config.RESULTS_DIR / f"confusion_matrix_{model_type}.png"
    plt.savefig(path, dpi=150)
    if config.SHOW_PLOTS:
        plt.show()
    else:
        plt.close()
    print(f"{path} 저장 완료")


def plot_misclassified(preds, labels, imgs, model_type, n=8):
    wrong_idx = (preds != labels).nonzero(as_tuple=True)[0][:n]
    if len(wrong_idx) == 0:
        print("오분류 샘플 없음")
        return

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    for ax, idx in zip(axes.flatten(), wrong_idx):
        img = (imgs[idx] * std + mean).permute(1, 2, 0).clamp(0, 1).numpy()
        ax.imshow(img)
        ax.set_title(
            f"실제: {config.IDX_TO_ERA[labels[idx].item()]}\n"
            f"예측: {config.IDX_TO_ERA[preds[idx].item()]}",
            fontsize=9
        )
        ax.axis("off")
    plt.suptitle(f"오분류 샘플 ({model_type})")
    plt.tight_layout()
    path = config.RESULTS_DIR / f"misclassified_{model_type}.png"
    plt.savefig(path, dpi=150)
    if config.SHOW_PLOTS:
        plt.show()
    else:
        plt.close()
    print(f"{path} 저장 완료")


def main(model_type):
    device = get_device()
    print(f"Device: {device}")

    test_ds = FashionEraDataset("test")
    test_loader = DataLoader(
        test_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS
    )

    model = load_model(model_type, device)

    preds, labels, imgs = get_predictions(model, test_loader, device)

    acc = (preds == labels).float().mean().item()
    print(f"\nTest Accuracy: {acc:.4f}\n")
    print(classification_report(
        labels.numpy(), preds.numpy(),
        target_names=[str(e) for e in config.TARGET_ERAS]
    ))

    plot_confusion_matrix(preds.numpy(), labels.numpy(), model_type)
    plot_misclassified(preds, labels, imgs, model_type)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["custom", "resnet18"], default="custom")
    args = parser.parse_args()
    main(args.model)
