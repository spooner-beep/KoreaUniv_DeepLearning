import os
import sys
from pathlib import Path

from utils.env import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = Path(os.getenv("FASHION_REPO_ROOT", PROJECT_ROOT.parent))

DATA_ROOT = Path(os.getenv("FASHION_DATA_ROOT", REPO_ROOT / "Data"))
IMAGE_DIR = Path(os.getenv("FASHION_IMAGE_DIR", PROJECT_ROOT / "data_resized"))

TARGET_ERAS = [1990, 2000, 2010, 2019]
ERA_TO_IDX = {era: idx for idx, era in enumerate(TARGET_ERAS)}
IDX_TO_ERA = {idx: era for era, idx in ERA_TO_IDX.items()}
NUM_CLASSES = len(TARGET_ERAS)

RESIZE_SIZE = 256  # 오프라인 전처리 출력 크기
IMG_SIZE = 224       # 학습 시 CenterCrop 크기
BATCH_SIZE = int(os.getenv("FASHION_BATCH_SIZE", "32"))
NUM_EPOCHS = 30  # ResNet18 경로용; CustomCNN 은 hyperparams.CustomCNNHyperParams 참고
LR = 1e-3

# 로컬 PC 발열/부하 제어용. 정확한 GPU 사용률 제한은 PyTorch 단독으로 어렵지만,
# batch/worker/thread/memory 상한으로 과열을 줄인다.
TORCH_NUM_THREADS = int(os.getenv("FASHION_TORCH_NUM_THREADS", "4"))
TORCH_NUM_INTEROP_THREADS = int(os.getenv("FASHION_TORCH_NUM_INTEROP_THREADS", "1"))
CUDA_MEMORY_FRACTION = float(os.getenv("FASHION_CUDA_MEMORY_FRACTION", "1.0"))
CUDNN_BENCHMARK = os.getenv("FASHION_CUDNN_BENCHMARK", "0").lower() in ("1", "true", "yes")


def _default_num_workers() -> int:
    if os.getenv("FASHION_NUM_WORKERS") is not None:
        return int(os.getenv("FASHION_NUM_WORKERS"))
    # Windows: DataLoader multiprocessing 이슈 방지
    if sys.platform == "win32":
        return 0
    # Amazon EC2 (Linux): GPU 학습 시 기본 4
    return 4


NUM_WORKERS = _default_num_workers()
# CustomCNN: Windows에서 worker 4는 CPU 100% + 첫 epoch 지연. 기본 2 권장.
CUSTOM_NUM_WORKERS = int(
    os.getenv("FASHION_CUSTOM_NUM_WORKERS", str(min(NUM_WORKERS, 2)))
)
PREFETCH_FACTOR = int(os.getenv("FASHION_PREFETCH_FACTOR", "2"))


def _pin_memory_default() -> bool:
    if os.getenv("FASHION_PIN_MEMORY") is not None:
        return os.getenv("FASHION_PIN_MEMORY", "").lower() in ("1", "true", "yes")
    return os.getenv("FASHION_DEVICE", "").lower() == "cuda"


PIN_MEMORY = _pin_memory_default()

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)

FEATURE_DIR = PROJECT_ROOT / "features"
FEATURE_DIR.mkdir(exist_ok=True)
FEATURE_NPY = FEATURE_DIR / "features.npy"
FEATURE_META = FEATURE_DIR / "metadata.json"

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

SHOW_PLOTS = os.getenv("FASHION_SHOW_PLOTS", "").lower() in ("1", "true", "yes")

LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)


def runtime_profile() -> str:
    """local | ec2 | auto"""
    explicit = os.getenv("FASHION_PROFILE", "").strip().lower()
    if explicit in ("local", "ec2"):
        return explicit
    if sys.platform == "win32":
        return "local"
    if os.getenv("FASHION_DEVICE", "").lower() == "cuda" or (
        sys.platform.startswith("linux") and os.getenv("FASHION_HEADLESS") == "1"
    ):
        return "ec2"
    return "auto"
