# 0001 — preload-bundled-cudart-for-onnxruntime

## Status

Accepted

## Date

2026-08-04

## Context

`pyproject.toml`은 `onnxruntime-gpu==1.27.0`을 고정하고 있는데, 이 wheel은 **import 시점**에
`libcudart.so.13`(CUDA 13 런타임)을 dlopen한다. 현재 개발 환경(WSL2)에는 시스템 CUDA 13이
없어 `import onnxruntime`이 `ImportError: libcudart.so.13`으로 즉시 실패한다.
러너는 CPU EP만 쓰므로 CUDA 실행 자체는 필요 없고, import만 통과하면 된다.
venv에는 torch cu130이 의존성으로 끌어온 `nvidia/cu13/lib/libcudart.so.13`이 이미 존재한다.

## Decision

러너가 `onnxruntime`을 import하기 직전에 venv 동봉 `nvidia/cu13/lib/libcudart.so.13`을
`ctypes.CDLL(..., RTLD_GLOBAL)`로 preload한다 (`src/runner.py`의 `_preload_bundled_cudart`).
파일이 없으면 조용히 건너뛰어, CUDA가 정상 설치된 환경이나 CPU wheel 환경에서는 no-op이다.

## Considered Options

1. **venv 동봉 cudart를 ctypes preload** — 의존성 변경 없음, `uv run python -m src.runner`가
   환경변수 없이 그대로 동작, CUDA 있는 환경에서도 무해
2. `LD_LIBRARY_PATH` 설정을 실행 명령에 요구 — 인터페이스 계약("인자 그대로 재실행 가능") 위반
3. `onnxruntime`(CPU wheel)으로 교체/추가 — 같은 import 이름을 가진 패키지가 충돌하고,
   1.27.0 고정(주석: "WSL/Linux 호환")이라는 기존 결정을 침범
4. `import torch`를 먼저 수행해 CUDA 라이브러리를 로드 — 동작하지만 러너 시작이 수 초
   느려지고 의도가 불투명

## Consequences

- 의존성·실행 인터페이스를 건드리지 않고 WSL/CUDA-없는 환경에서 러너가 동작한다.
- torch cu130이 venv에서 빠지면 preload 대상이 사라져 다시 ImportError가 난다
  (그 경우 에러 원문이 그대로 노출되므로 원인 추적은 가능).
- onnxruntime-gpu가 요구하는 CUDA 메이저 버전이 바뀌면 경로(`cu13`)를 갱신해야 한다.
