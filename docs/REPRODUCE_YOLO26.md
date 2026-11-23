# YOLO26 실험 실행

완료한 다섯 실험의 설정은 `configs/yolo26/{n640,s640,m640,n960,s960}.json`에 보존했다. `docs/experiments`의 공개 설정과 같으며 데이터·모델·출력 경로만 실행 시 지정한다.

## 환경

기존 학습 환경은 Windows, Python 3.13.5, RTX 4080 SUPER 16GB, PyTorch 2.13.0+cu130, Ultralytics 8.4.106이다. 기존 PC 러너용 `pyproject.toml`과 분리한 환경을 사용한다.

```sh
python -m venv .venv-training
```

Windows PowerShell에서는 `.\.venv-training\Scripts\Activate.ps1`, Linux에서는 `source .venv-training/bin/activate`로 활성화한다.

```sh
python -m pip install -r requirements-training.txt
python tools/yolo26.py --help
```

고정 의존성은 기존 Windows 환경의 기록이다. 다른 OS에서의 설치와 학습은 아직 검증하지 않았다. 실행기는 PyTorch·Ultralytics 버전과 CUDA를 검사한다.

## 데이터 준비

기존에 리샘플링한 YOLO 데이터가 필요하다. 원본 AIHub 전체에서 새 분할을 만드는 명령이 아니다. 이미지·라벨과 가중치는 저장소에 포함하지 않는다.

```text
/path/to/data/
  legacy_c4d/
    images/train/   images/val/
    labels/train/   labels/val/
  bollard_val_2146/
    images/val/     labels/val/
  nightowls_rec34/
    images/val/     labels/val/
/path/to/pretrained/
  yolo26n.pt
  yolo26s.pt
  yolo26m.pt
```

라벨 순서는 `0: person`, `1: stairs`, `2: bollard`다. 학습 10,919장과 개발 787장을 그대로 사용한다. 데이터 파일을 복사하거나 기존 YAML을 수정하지 않고, 작업 폴더에 검사 기록과 입력 목록을 만든다.

```sh
python tools/yolo26.py prepare --data-root /path/to/data --workspace outputs/reproduction
```

외부 평가 데이터도 준비했다면 같은 명령에 `--external`을 추가한다. 개발셋 중복을 제외한 볼라드 2,085장 목록도 생성한다.

이미지 디코딩, 라벨 좌표, 이미지·라벨 대응, 분할 중복과 원본 파일 내용의 해시를 검사한다. `configs/yolo26/datasets.json`에는 원본 manifest에서 계산한 `(split, 이미지 SHA-256, 라벨 SHA-256)` 목록의 정렬된 해시가 있다. 경로·운영체제 줄바꿈에 영향을 받는 기존 manifest 파일 자체의 해시와는 다른 값이다. 이름이나 위치를 바꿔도 파일 내용과 분할이 같아야 통과한다.

## 학습

먼저 데이터·사전학습 가중치 해시와 모델 로딩, CUDA, 설정 유효성을 확인한다.

```sh
python tools/yolo26.py train --run n640 --pretrained-dir /path/to/pretrained --workspace outputs/reproduction --check-only
```

검사가 통과하면 `--check-only`를 빼고 실행한다. `--run` 값만 바꿔 다섯 조건을 실행할 수 있다.

| run | 모델 | 입력 | epochs | batch | seed |
|---|---|---:|---:|---:|---:|
| n640 | YOLO26n | 640 | 50 | 16 | 0 |
| s640 | YOLO26s | 640 | 50 | 16 | 0 |
| m640 | YOLO26m | 640 | 50 | 16 | 0 |
| n960 | YOLO26n | 960 | 50 | 16 | 0 |
| s960 | YOLO26s | 960 | 50 | 16 | 0 |

각 결과는 `outputs/reproduction/runs/<run>/`에 저장한다. 기존 실행 폴더가 있으면 덮어쓰지 않고 중단한다. 이 공통 실행기는 중단 학습 재개를 지원하지 않으므로 재시도는 새 workspace를 사용한다.

## 별도 평가

```sh
python tools/yolo26.py evaluate --run n640 --weights outputs/reproduction/runs/n640/weights/best.pt --dataset legacy_c4d --workspace outputs/reproduction --output outputs/evaluation/n640-dev --check-only
```

`--check-only`를 빼면 평가한다. `--dataset`은 `legacy_c4d`, `bollard_val_2146`, `bollard_dev_disjoint`, `nightowls_rec34` 중 선택한다. `--run`은 평가 기록에 남기는 실험 식별자이며, 실제 평가 모델은 `--weights`로 지정한다.

별도 평가는 기존 `evaluation` 조건인 입력 640, batch 16, confidence 0.001, IoU 0.7, max_det 300, rect=False, half=False를 공통 적용한다. 학습 중 best 선택용 검증과 다르므로 README의 기존 점수를 대체하지 않는다. 클래스별 AP와 데스크톱 처리 시간을 기록하며 NightOwls는 person, 볼라드 세트는 person·bollard만 집계한다.

## 검증 범위

2026년 11월 23일 기존 Windows 환경에서 데이터 준비, 다섯 학습 조건과 n640 개발셋 평가의 `--check-only`가 통과했다. 데이터 오류 방지 테스트 3개도 통과했다. [검사 기록](experiments/reproduction-check.json) `--check-only`는 실제 학습 배치나 역전파를 실행하지 않는다. 기존 실험 결과는 [공개 근거](experiments/README.md)에 있으며, 이 코드로 새 학습을 완료한 결과는 아직 없다.
