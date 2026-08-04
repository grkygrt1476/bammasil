# Architecture Decision Records

구조적 선택을 기록한다. 새 ADR은 [_template.md](_template.md)를 복사해 작성하고,
번호는 기존 최대+1 (재사용 금지), 추가 시 아래 인덱스에 같은 커밋에서 한 줄 추가한다.

| 번호 | 제목 | 상태 | 날짜 |
|---|---|---|---|
| [0001](0001-preload-bundled-cudart-for-onnxruntime.md) | onnxruntime import를 위한 venv 동봉 cudart preload | Accepted | 2026-08-04 |
| [0002](0002-hold-last-boxes-on-strided-frames.md) | stride 미추론 프레임은 직전 박스 유지 렌더 | Accepted | 2026-08-04 |
| [0003](0003-keep-requested-default-thresholds.md) | conf/IoU 기본값은 발주 사양(0.25/0.45) 유지 | Accepted | 2026-08-04 |
