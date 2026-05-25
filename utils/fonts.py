"""Cross-platform Korean font helpers (Windows / Amazon Linux EC2)."""
import os
import sys
from pathlib import Path


def _first_existing(paths):
    for p in paths:
        if p and Path(p).is_file():
            return Path(p)
    return None


def matplotlib_font_family() -> str:
    if os.getenv("FASHION_FONT_MATPLOTLIB"):
        return os.getenv("FASHION_FONT_MATPLOTLIB")

    if sys.platform == "win32":
        return "Malgun Gothic"

    # Amazon Linux / Ubuntu (fonts-nanum 패키지)
    if Path("/usr/share/fonts/truetype/nanum/NanumGothic.ttf").is_file():
        return "NanumGothic"

    return "DejaVu Sans"


def setup_matplotlib():
    import matplotlib

    if os.getenv("FASHION_HEADLESS", "").lower() in ("1", "true", "yes"):
        matplotlib.use("Agg")
    elif sys.platform != "win32" and not os.environ.get("DISPLAY"):
        matplotlib.use("Agg")

    matplotlib.rcParams["font.family"] = matplotlib_font_family()
    matplotlib.rcParams["axes.unicode_minus"] = False


def ppt_font_name() -> str:
    if os.getenv("FASHION_FONT_PPT"):
        return os.getenv("FASHION_FONT_PPT")
    if sys.platform == "win32":
        return "Malgun Gothic"
    return "NanumGothic"


def reportlab_font_paths():
    """Return (regular_ttf, bold_ttf) for reportlab TTFont registration."""
    reg = os.getenv("FASHION_FONT_REGULAR")
    bold = os.getenv("FASHION_FONT_BOLD")
    if reg and bold:
        return Path(reg), Path(bold)

    if sys.platform == "win32":
        win = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
        return win / "malgun.ttf", win / "malgunbd.ttf"

    linux_candidates = [
        (
            Path("/usr/share/fonts/truetype/nanum/NanumGothic.ttf"),
            Path("/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf"),
        ),
        (
            Path("/usr/share/fonts/nanum/NanumGothic.ttf"),
            Path("/usr/share/fonts/nanum/NanumGothicBold.ttf"),
        ),
    ]
    for regular, bold in linux_candidates:
        if regular.is_file() and bold.is_file():
            return regular, bold

    raise FileNotFoundError(
        "한글 PDF 폰트를 찾을 수 없습니다. "
        "EC2: sudo yum install -y fontconfig && sudo yum install -y google-noto-sans-cjk-ttc "
        "또는 .env에 FASHION_FONT_REGULAR / FASHION_FONT_BOLD 경로를 지정하세요."
    )
