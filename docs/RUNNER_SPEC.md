# RUNNER_SPEC — YOLO ONNX 추론 러너

`src/runner.py` — 이미지 1장 / 이미지 폴더 / 영상 파일을 입력받아 ONNX YOLO 모델로 탐지하고,
바운딩박스가 그려진 결과물과 프레임별 상세(`run.jsonl`)를 저장한다.
이미지는 길이 1짜리 영상과 같은 코드 경로로 처리한다.

모델 입출력·전처리 계약의 원문은 [model/bammasil_det_c4b_loli0_640/README.md](../model/bammasil_det_c4b_loli0_640/README.md)이다.

## 실행

```bash
uv run python -m src.runner --input data/sample_gemini --model model/bammasil_det_c4b_loli0_640/bammasil_det_c4b_loli0_640.onnx --note "통합 구동 확인"
uv run python -m src.runner --input <파일.mp4> --model <...> --stride 3
```

## CLI 인자

| 인자 | 필수 | 기본값 | 설명 |
|---|---|---|---|
| `--input` | ✔ | — | 이미지 파일 / 이미지 폴더 / 영상 파일 — 자동 판별. 경로가 없고 같은 스템의 파일이 정확히 하나 있으면 그 파일로 해석 (예: `data/sample_gemini` → `data/sample_gemini.jpg`) |
| `--model` | ✔ | — | ONNX 모델 경로. 클래스명·순서는 ONNX 메타데이터 `names`에서 읽으며, 없으면 실행을 중단한다 (하드코딩 금지) |
| `--conf` | | 0.25 | confidence 임계값. 모델 카드 권장값은 0.35 — [ADR-0003](decisions/0003-keep-requested-default-thresholds.md) |
| `--iou` | | 0.45 | NMS IoU 임계값. 모델 카드 권장값은 0.7 — [ADR-0003](decisions/0003-keep-requested-default-thresholds.md) |
| `--stride` | | 1 | 영상 전용: N프레임마다 탐지하고 나머지 프레임은 직전 박스를 유지해 렌더 — [ADR-0002](decisions/0002-hold-last-boxes-on-strided-frames.md) |
| `--note` | | 입력 파일명 | `docs/TESTLOG.md` 회차 제목 |

## 출력

`outputs/YYYYMMDD_HHMM/` (같은 분에 재실행하면 `_2`, `_3` … 접미사):

- 이미지 입력: 입력과 같은 파일명의 결과 이미지 (입력 수와 동일)
- 영상 입력: `<스템>_det.mp4` + 대표 프레임 `rep_fNNNNNN.jpg` 최대 3장
  (탐지 수 상위 순. 전 프레임 탐지 0건이면 첫/끝 프레임을 저장)
- `run.jsonl` — 프레임별 1줄:

  ```json
  {"frame": 0, "source": "sample_gemini.jpg", "inferred": true, "ms": 36.5,
   "boxes": [{"cls": "stairs", "conf": 0.8448, "xyxy": [738.4, 916.6, 2186.0, 1506.8]}]}
  ```

  `source`는 이미지 입력일 때만 파일명(영상은 `null`). stride로 탐지를 건너뛴 프레임은
  `inferred: false, ms: null`이고 `boxes`는 직전 탐지 결과의 유지본. `xyxy`는 원본 해상도 좌표.

콘솔은 6줄 요약(`[env] [model] [input] [run] [det] [out]`)만 출력한다.
실행이 성공하면 `docs/TESTLOG.md` 최상단에 회차 블록을 자동 추가한다.

**탐지 0건은 실패가 아니다** — `[det]` 줄에 0건임을 명시하고 원본을 표시 없이 저장한다.

집계 기준: `[det]`의 클래스별 건수와 "탐지 0건" 수는 **추론한 프레임**에 대해서만 센다
(stride로 유지된 프레임은 직전 결과의 복사본이므로 중복 집계하지 않는다).
영상 모드의 `[det]` 줄은 `탐지 0건: N/M 추론프레임` 형식으로 분모(추론 프레임 수)를 병기한다.

## 성공 판정 (전부 참이어야 완료)

- exit code 0
- outputs/에 입력과 같은 수의 결과물
- 콘솔에 클래스명·EP·탐지 수·평균 소요시간 출력됨
- 위 인터페이스의 첫 번째 명령이 인자 그대로 재실행 가능

## 환경 특이사항

`onnxruntime-gpu 1.27.0`은 import 시점에 `libcudart.so.13`을 요구한다. 시스템에 CUDA 13이
없는 환경(WSL 등)을 위해 러너가 venv에 torch cu130이 동봉한 런타임을 preload한다 —
[ADR-0001](decisions/0001-preload-bundled-cudart-for-onnxruntime.md).
CPU EP만 사용하므로 실제 CUDA 실행은 일어나지 않는다.
