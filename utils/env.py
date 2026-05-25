"""Load optional .env from project root."""
from pathlib import Path


def load_dotenv():
    try:
        from dotenv import load_dotenv as _load
    except ImportError:
        return
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.is_file():
        _load(env_path)
