# KoreaUniv DeepLearning - Fashion Cycle Service Package

이 패키지는 학습 완료된 **ResNet18** 및 **CustomCNN** 모델을 서비스 적용/추론에 사용하기 위한 최소 구성입니다.

## 포함 모델

| 모델 | 체크포인트 | Best Val Accuracy | 비고 |
|---|---|---:|---|
| ResNet18 | `checkpoints/best_resnet18.pth` | 67.05% | ImageNet pretrained 기반 fine-tuning |
| CustomCNN | `checkpoints/best_custom.pth` | 54.50% | `modern_v4`, ResNet18 distillation으로 학습 |

CustomCNN의 최종 체크포인트에는 **CustomCNN 구조/파라미터만 포함**되며 ResNet18 backbone은 포함되지 않습니다.

## 폴더 구성

```text
.
├── checkpoints/
│   ├── best_resnet18.pth
│   ├── best_custom.pth
│   └── custom_hparams.json
├── results/
│   ├── custom_train_log.json
│   ├── resnet18_train_log.json
│   └── train_history_custom.json
├── utils/
├── config.py
├── model.py
├── hyperparams.py
├── dataset.py
├── evaluate.py
├── recommend.py
├── extract_features.py
├── compare_models.py
├── requirements.txt
├── final_customcnn_report.pdf
└── final_customcnn_report.pptx
```

## 환경 설정

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.local.example .env
```

RTX 50 시리즈 GPU에서는 CUDA 12.8 이상을 지원하는 PyTorch wheel이 필요할 수 있습니다.

## 모델 평가

Validation 데이터(`data_resized/Validation`)가 있는 경우:

```powershell
python evaluate.py --model resnet18
python evaluate.py --model custom
python compare_models.py
```

## 서비스 추론

단일 이미지 분류는 모델을 직접 로드하여 사용할 수 있습니다.

```python
from utils.device import get_device
from utils.model_loader import load_model

device = get_device()
model = load_model("custom", device).eval()
```

추천 기능(`recommend.py`)은 `features/features.npy`, `features/metadata.json`이 필요합니다. 패키지에 feature index가 없으면 데이터셋을 준비한 뒤 다음 명령으로 생성합니다.

```powershell
python extract_features.py --model custom
python recommend.py --image path\to\image.jpg --model custom --top_k 5
```

## 보고서

- `final_customcnn_report.pdf`
- `final_customcnn_report.pptx`

보고서에는 최초 CustomCNN과 최종 CustomCNN의 구조, 파라미터, 학습 전략, 성능 비교가 포함되어 있습니다.
