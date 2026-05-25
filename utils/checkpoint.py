import torch


def load_state_dict(path, map_location=None):
    """PyTorch 2.x weights_only 지원, 구버전 호환."""
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)
