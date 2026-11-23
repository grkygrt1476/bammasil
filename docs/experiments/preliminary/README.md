# 예선 모델 비교 기록

기획서 5–6쪽 그래프의 원본 `benchmark_c4d.csv`와 여섯 실행의 epoch CSV를 보존했다. 새로 평가한 결과가 아니다.

- [전체 벤치마크 수치](benchmark.csv): 평균·클래스별 AP, 볼라드 크기별 Recall, 처리 시간, 오탐 집계.
- [원본 해시](source.json): 원본 벤치마크와 학습 CSV의 SHA-256.
- 학습 횟수: 640 조건은 50 epoch, YOLO26n 960은 48, YOLO11n 960은 30. 엄밀한 동일 학습 예산 비교는 아니다.
- 평가셋: 기존 주간 보도 2,146장, 볼라드 GT 6,988개. 개발셋과 61장이 겹치는 역사적 평가셋이다.
- `v2146_map50`은 person·bollard 평균이며 볼라드 단독 AP는 `v2146_bollard_mAP50`이다.
- 크기별 Recall: 각 실행의 오류 분석 기록 기준 confidence 0.25, 매칭 IoU 0.5. GT 너비는 원본 이미지 픽셀 기준이다. 원본 구간 이름을 그대로 유지했다.
- `inf_ms`는 데스크톱 모델 추론 시간만 나타낸다. 전처리와 후처리는 CSV의 별도 열이며 앱 지연시간과 다르다.

그래프 재생성:

```sh
python tools/plot_preliminary_results.py
```

Matplotlib·NumPy가 필요하다. 모델·해상도 그래프와 크기별 Recall은 `benchmark.csv`, 학습 곡선은 실행별 CSV에서 그린다. 10월의 YOLO26 후속 다섯 실험은 다른 분할·평가 기록으로 별도 표에 둔다.
