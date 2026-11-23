# 밤마실

**야맹증 저시력자의 야간보행을 돕는 실시간 AI 시각보조 서비스**

스마트폰 카메라로 사람·계단·볼라드를 찾아 강조하고, 밝기·대비를 개선해 야간 보행을 돕습니다.

🏆 **제8회 K-디지털 트레이닝 해커톤 지정과제 최우수상 (고용노동부 장관상)**

2026.09.04 · [수상 발표](https://www.moel.go.kr/news/enews/report/enewsView.do?news_seq=19887) · [팀장 인터뷰](https://www.mt.co.kr/policy/2026/09/08/2026090814340425566)

## 시연

![밤마실 앱의 야간 볼라드 탐지](docs/assets/demo/bollard-demo.gif)

| 사람 | 계단 | 볼라드 |
|:---:|:---:|:---:|
| <img src="docs/assets/demo/person-detection.jpg" width="220" alt="사람 탐지"> | <img src="docs/assets/demo/stairs-detection.jpg" width="220" alt="계단 탐지"> | <img src="docs/assets/demo/bollard-detection.jpg" width="220" alt="볼라드 탐지"> |

[조도 개선 · Google Cardboard 시연](docs/DEMO.md)

## 기획 배경

전맹을 위한 음성 안내 서비스는 있지만, 시력을 잃어가거나 잔존시력이 있는 사람이 **남은 시력을 활용하도록 돕는 지원은 부족하다**는 문제의식에서 시작했습니다. 밤마실은 사용자가 밤길의 위험요소를 직접 보고 피할 수 있도록 화면을 개선하고 탐지 영역을 강조합니다.

사람·계단을 기본 탐지 대상으로 삼고, 당사자 인터뷰에서 반복적으로 충돌한다고 언급한 볼라드를 추가했습니다.

## 담당 역할 — 팀장

**서비스 기획부터 설계, 모델링·앱 개발 방향 설정, 실증까지 프로젝트 전반을 총괄했습니다.**

- **서비스 기획·설계:** 대상 사용자와 문제 정의, 핵심 기능·사용 흐름 설계, 개발 우선순위 결정, 기획서 작성
- **모델링:** 탐지 대상·학습·평가 방향 설정과 개선 작업 지시. AIHub 데이터 직접 정리·리샘플링, YOLO11n 학습
- **앱 파이프라인:** 입력·탐지·화면 표시 구조 설계, 개발 요구사항 수립과 구현·개선 작업 지시
- **실증:** Google Cardboard 착용 환경 실증 기획과 진행 지시
- **후속 개선:** YOLO26 모델 크기·입력 해상도 비교 실험과 결과 정리

## 앱 구조

탐지는 원본 영상으로, 화면 표시는 밝기·대비를 개선한 영상으로 처리합니다.

```mermaid
flowchart LR
    camera["CameraX"] --> detect["YOLO · ONNX Runtime"]
    camera --> enhance["Drago · CLAHE"]
    detect --> stabilize["표시 안정화 · FSM / IIR"]
    stabilize --> render["OpenGL · 위험영역 강조"]
    enhance --> render
```

## 실험 결과

### 예선 모델 비교

![모델별 정확도와 추론 시간](docs/assets/experiments/proposal-model-resolution.png)

주간 보도 2,146장, person·bollard 평균 mAP50과 데스크톱 추론 시간입니다. 640 조건은 50 epoch, 960 조건은 모델별 30·48 epoch입니다.

- 실야간 데이터 추가 후 보행자 mAP50 **0.127 → 0.684**
- 배경 이미지 추가 후 야간 계단 오탐 **22 → 9건** (본선 실험)
- Galaxy A34에서 화면 표시 **29.9 FPS**, 탐지 갱신 **3.4 FPS**

[예선 실험 표·학습 곡선·크기별 Recall](docs/PRELIMINARY_RESULTS.md) · [negative 학습 기록](docs/MODEL_EXPERIMENTS.md)

### 후속 YOLO26 학습

NMS 후처리 병목을 줄이고 더 큰 모델을 적용하기 위해 n·s·m을 비교했습니다. 공통 조건은 **50 epochs · batch 16 · seed 0**, RTX 4080 SUPER입니다.

| 모델 | 입력 | 개발셋 mAP50–95 | 학습 시간(분) |
|---|---:|---:|---:|
| YOLO26n | 640 | 0.59884 | 83.4 |
| YOLO26s | 640 | 0.63822 | 88.0 |
| YOLO26m | 640 | 0.64738 | 118.3 |
| YOLO26n | 960 | 0.61995 | 102.3 |
| YOLO26s | 960 | **0.66632** | 134.6 |

각 조건은 1회 학습했으며 개발셋 best checkpoint 기준입니다. **s960이 가장 높았고**, 최종 모델은 외부 평가와 스마트폰 지연시간을 확인한 뒤 선택할 예정입니다.

[5개 실험 설정·결과·곡선](docs/experiments/README.md) · [데이터 구성](docs/DATASET.md) · [학습·평가 실행 방법](docs/REPRODUCE_YOLO26.md)

## 팀 저장소

[비전 모델링](https://github.com/kty2001/KDT_Hackathon) · [파이프라인·앱](https://github.com/jh4995/night-walk)
