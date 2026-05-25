from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import config


def get_transforms(split: str, augment: str = "default"):
    """
    augment:
      default — 기존 증강 (ResNet·baseline 비교용)
      strong  — CustomCNN tuned_v2: RandomErasing + 회전 (느림)
      strong_fast — strong 유사, CPU 부담 축소 (권장)
    """
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    if split == "train":
        if augment == "strong_fast":
            return transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(config.IMG_SIZE, padding=8),
                transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.03),
                transforms.ToTensor(),
                normalize,
                transforms.RandomErasing(p=0.15, scale=(0.02, 0.08)),
            ])
        if augment == "strong":
            return transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.RandomCrop(config.IMG_SIZE, padding=16),
                transforms.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.35, hue=0.05),
                transforms.ToTensor(),
                normalize,
                transforms.RandomErasing(p=0.25, scale=(0.02, 0.12)),
            ])
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(config.IMG_SIZE, padding=16),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            transforms.ToTensor(),
            normalize,
        ])

    return transforms.Compose([
        transforms.CenterCrop(config.IMG_SIZE),
        transforms.ToTensor(),
        normalize,
    ])


class FashionEraDataset(Dataset):
    SPLIT_MAP = {
        "train": "Training",
        "val":   "Validation",
        "test":  "Validation",
    }

    def __init__(self, split, return_path=False, augment: str = "default"):
        self.transform = get_transforms(split, augment=augment)
        self.return_path = return_path
        self.samples = []

        split_dir = config.IMAGE_DIR / self.SPLIT_MAP[split]
        for gender in ["man", "woman"]:
            for era in config.TARGET_ERAS:
                era_dir = split_dir / gender / str(era)
                if not era_dir.exists():
                    continue
                for img_path in sorted(era_dir.glob("*.jpg")):
                    self.samples.append((img_path, config.ERA_TO_IDX[era]))

        print(f"[{split}] {len(self.samples)}장 로드 (augment={augment})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        with Image.open(img_path) as img:
            img = self.transform(img.convert("RGB"))
        if self.return_path:
            return img, label, str(img_path)
        return img, label
