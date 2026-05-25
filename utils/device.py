"""PyTorch device selection (CUDA if available, else CPU; override with FASHION_DEVICE)."""
import os

import torch


def get_device() -> torch.device:
    forced = os.getenv("FASHION_DEVICE", "").strip().lower()
    if forced:
        if forced.startswith("cuda") and not torch.cuda.is_available():
            print(
                "[device] FASHION_DEVICE=cuda 이지만 CUDA를 사용할 수 없습니다. "
                "CPU로 학습합니다. (PyTorch cu128 nightly 설치 필요)",
                flush=True,
            )
            return torch.device("cpu")
        return torch.device(forced)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
