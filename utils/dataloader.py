"""DataLoader 공통 설정 (GPU 활용률·전송 최적화)."""
from __future__ import annotations

from torch.utils.data import DataLoader, Dataset

import config


def make_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int | None = None,
) -> DataLoader:
    workers = config.NUM_WORKERS if num_workers is None else num_workers
    kwargs: dict = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": workers,
    }
    if workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = config.PREFETCH_FACTOR
    if config.PIN_MEMORY:
        kwargs["pin_memory"] = True
    return DataLoader(dataset, **kwargs)


def to_device(batch, device):
    """pin_memory 사용 시 non_blocking GPU 전송."""
    imgs, labels = batch[0], batch[1]
    non_blocking = config.PIN_MEMORY and device.type == "cuda"
    return imgs.to(device, non_blocking=non_blocking), labels.to(device, non_blocking=non_blocking)
