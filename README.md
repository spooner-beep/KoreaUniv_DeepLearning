# 패션 시대 분류 및 코디 추천 프로젝트

고려대학교 딥러닝 과목 팀 프로젝트  
**Windows 로컬** + **Amazon EC2 (CUDA GPU)** 환경을 기준으로 구성되어 있습니다.

## 프로젝트 개요

연도별 패션 이미지를 학습하여 스타일 벡터를 추출하고,
사용자의 옷 이미지를 입력하면 어울리는 코디를 추천하는 시스템입니다.

- **데이터**: [AI Hub — 연도별 패션 선호도 파악 및 추천](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71446) (CC BY-SA)
- **클래스**: 1990 / 2000 / 2010 / 2019 (4클래스)

## AI Hub 데이터 디렉터리 구조

`FASHION_DATA_ROOT` (예: `D:\Fashion Cycle\Data`) 기준:

```
Data/
├── Training/
│   ├── 01.원천데이터/          ← JPG 이미지
│   │   ├── TS_man_1990
│   │   ├── TS_man_2000  … TS_man_2019
│   │   └── TS_woman_1990 … TS_woman_2019
│   └── 02.라벨링데이터/          ← JSON 메타 (imgName, era 등)
│       ├── TL_man_1990          ※ TS_ 가 아니라 TL_
│       ├── TL_man_2000  … TL_man_2019
│       └── TL_woman_1990 … TL_woman_2019
└── Validation/
    ├── 01.원천데이터/
    │   ├── VS_man_1990  … VS_woman_2019
    └── 02.라벨링데이터/
        ├── VL_man_1990  … VL_woman_2019   ※ VS_ 가 아니라 VL_
```

| 접두사 | 의미 |
|--------|------|
| `TS_` | Training **원천**(Source) 이미지 |
| `TL_` | Training **라벨**(Label) JSON |
| `VS_` | Validation 원천 이미지 |
| `VL_` | Validation 라벨 JSON |

`resize_images.py`는 **라벨 JSON**(`TL_*` / `VL_*`)의 `imgName`으로 **원천**(`TS_*` / `VS_*`) JPG를 찾아 `data_resized/`에 저장합니다.  
1950~1980 등 다른 연도 폴더는 있어도, 학습에는 `config.TARGET_ERAS`(1990·2000·2010·2019)만 사용합니다.

## 환경 설정

### 1. 의존성 설치

```bash
cd ku_deep_learning_project
python -m venv .venv
# Windows
.venv\Scripts\activate
# EC2 Linux
source .venv/bin/activate

pip install -r requirements.txt
```

**EC2 (CUDA)** 예시 — 인스턴스 CUDA 버전에 맞게 [pytorch.org](https://pytorch.org)에서 wheel 설치:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

### 2. `.env` 설정

```bash
copy .env.example .env   # Windows
# cp .env.example .env   # Linux
```

| 변수 | 설명 |
|------|------|
| `FASHION_DATA_ROOT` | AI Hub `Data` 폴더 경로 |
| `FASHION_IMAGE_DIR` | 리사이즈 출력 (`data_resized`) |
| `FASHION_NUM_WORKERS` | Windows 기본 `0`, EC2 기본 `4` |
| `FASHION_HEADLESS` | EC2 SSH 시 `1` (그래프 파일만 저장) |
| `FASHION_DEVICE` | `cuda` / `cpu` 강제 지정 (선택) |

## CustomCNN 최적화 (ResNet18 상회 목표)

기존 CustomCNN 37.4% → ResNet18 44.1% 상회를 위해 하이퍼파라미터·구조·학습법을 개선했습니다.  
**상세 근거·실험 표·실행법**: [`docs/CUSTOMCNN_TUNING.md`](docs/CUSTOMCNN_TUNING.md)

```bash
# 하이퍼파라미터 탐색 (선택, EC2 권장)
python tune_custom.py --epochs 20

# 최적화된 CustomCNN 학습 (기본 프리셋 tuned_v2)
python train.py --model custom

# 구 설정과 비교
python train.py --model custom --preset baseline_v1
```

## 실행 방법

```bash
# 0. 원본 → 256×256 리사이즈 (최초 1회, 수십 분 소요)
python resize_images.py

# 1. 학습 (EC2 GPU 권장)
python train.py --model resnet18
python train.py --model custom

# 2. ResNet18 vs CustomCNN 비교
python compare_models.py

# 3. 평가
python evaluate.py --model resnet18

# 4. 특징 벡터 추출
python extract_features.py --model resnet18

# 5. 코디 추천 (로컬)
python recommend.py --image path\to\your_clothes.jpg --model custom --top_k 5
```

## 로컬 PC 실행 (Windows)

**상세 가이드**: [`docs/LOCAL_GUIDE.md`](docs/LOCAL_GUIDE.md)

```powershell
copy .env.local.example .env
.\scripts\local_pipeline.ps1 -Mode smoke    # 동작 확인 (2 epoch)
python watch_training.py                    # 다른 터미널에서 진행 상황 확인
```

학습 진행 파일: `logs/latest_status.json`, `logs/train_*.log`, `results/train_history_*.json`

## EC2 학습 + 로컬 추천 (권장 운영 방식)

**AMI**: Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.11 (Ubuntu 24.04)  
**인스턴스**: `g6.xlarge` (NVIDIA L4)

| 단계 | 위치 | 작업 |
|------|------|------|
| 리사이즈·학습·비교 | EC2 | `scripts/ec2_pipeline.sh` |
| Top-5 추천 | Windows 로컬 | `recommend.py` + EC2에서 받은 `checkpoints/`, `features/` |

**상세 플로우**: [`docs/EC2_WORKFLOW.md`](docs/EC2_WORKFLOW.md)

```bash
# EC2
cp .env.ec2.example .env
chmod +x scripts/ec2_pipeline.sh && ./scripts/ec2_pipeline.sh
```

```powershell
# Windows 로컬
.\scripts\local_recommend.ps1 -Image "내옷.jpg" -Model custom -TopK 5
```

## Amazon EC2 빠른 가이드

1. **인스턴스**: `g6.xlarge` + Deep Learning AMI (PyTorch 2.11, Ubuntu 24.04)
2. **데이터**: `Data/` + (권장) `data_resized/` 를 EC2에 업로드
3. **`.env`**: `.env.ec2.example` 참고
4. **한글 폰트** (평가 그래프용, 선택):

   ```bash
   # Amazon Linux 2023
   sudo yum install -y fontconfig
   sudo yum install -y google-noto-sans-cjk-ttc
   # 또는 Nanum
   sudo yum install -y nanum-fonts-all
   ```

5. **학습**:

   ```bash
   export FASHION_HEADLESS=1
   python resize_images.py
   python train.py --model resnet18
   ```

6. **체크포인트**를 로컬로 받은 뒤 Windows에서 `recommend.py` 실행 가능

## 파일 구조

```
├── config.py              # 경로·하이퍼파라미터 (.env 연동)
├── resize_images.py       # Data → data_resized (256×256)
├── dataset.py
├── model.py
├── train.py
├── compare_models.py      # ResNet18 vs CustomCNN Train/Val Acc
├── evaluate.py
├── scripts/ec2_pipeline.sh
├── scripts/local_recommend.ps1
├── docs/EC2_WORKFLOW.md
├── extract_features.py
├── recommend.py
├── utils/                 # device, fonts, checkpoint
├── .env.example
├── requirements.txt
├── checkpoints/
├── results/               # 평가·추천 결과 PNG
└── data_resized/          # git 제외, resize_images.py 로 생성
```

## 기술 스택

- Python 3.11+ / PyTorch 2.x
- Windows 10/11 (CPU·로컬 개발)
- Amazon EC2 + NVIDIA CUDA (학습)
