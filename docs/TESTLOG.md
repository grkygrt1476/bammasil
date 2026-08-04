# TESTLOG

## 2026-08-04 12:54 — 영상 stride 3 구동 확인

**결론**: 통과 — YOLO 로드 → 추론 → 렌더 연결 확인

| 항목 | 내용 |
|---|---|
| 입력 | data/sample_video_gemini.mp4 — 240프레임 (stride 3) |
| 모델 | bammasil_det_c4b_loli0_640.onnx / 640 static |
| 클래스 | person, stairs (ONNX 메타 names) |
| 환경 | PC CPU · onnxruntime 1.27.0 · CPUExecutionProvider |
| 결과 | person 0건, stairs 81건 / 탐지 0건 1/80 추론프레임 |
| 산출물 | outputs/20260804_1254_2/ |

**막힌 것**: 없음

## 2026-08-04 12:54 — 통합 구동 확인

**결론**: 통과 — YOLO 로드 → 추론 → 렌더 연결 확인

| 항목 | 내용 |
|---|---|
| 입력 | data/sample_gemini — 1장 |
| 모델 | bammasil_det_c4b_loli0_640.onnx / 640 static |
| 클래스 | person, stairs (ONNX 메타 names) |
| 환경 | PC CPU · onnxruntime 1.27.0 · CPUExecutionProvider |
| 결과 | person 0건, stairs 1건 / 탐지 0건 0장 |
| 산출물 | outputs/20260804_1254/ |

**막힌 것**: onnxruntime-gpu가 libcudart.so.13 부재로 import 실패 → venv 동봉 cudart preload로 우회(ADR-0001) · `data/sample_gemini` 폴더가 실제론 단일 jpg → 스템 일치 해석으로 대응

## 2026-08-04 11:47 — 스모크 — 영상 stride 3

**결론**: 통과 — YOLO 로드 → 추론 → 렌더 연결 확인

| 항목 | 내용 |
|---|---|
| 입력 | data/sample_video_gemini.mp4 — 240프레임 (stride 3) |
| 모델 | bammasil_det_c4b_loli0_640.onnx / 640 static |
| 클래스 | person, stairs (ONNX 메타 names) |
| 환경 | PC CPU · onnxruntime 1.27.0 · CPUExecutionProvider |
| 결과 | person 0건, stairs 81건 / 탐지 0건 1프레임 |
| 산출물 | outputs/20260804_1147_2/ |

**막힌 것**: 없음

## 2026-08-04 11:47 — 스모크 — 이미지 1장

**결론**: 통과 — YOLO 로드 → 추론 → 렌더 연결 확인

| 항목 | 내용 |
|---|---|
| 입력 | data/sample_gemini.jpg — 1장 |
| 모델 | bammasil_det_c4b_loli0_640.onnx / 640 static |
| 클래스 | person, stairs (ONNX 메타 names) |
| 환경 | PC CPU · onnxruntime 1.27.0 · CPUExecutionProvider |
| 결과 | person 0건, stairs 1건 / 탐지 0건 0장 |
| 산출물 | outputs/20260804_1147/ |

**막힌 것**: 없음

