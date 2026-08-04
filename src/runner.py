"""밤마실 YOLO ONNX 추론 러너.

이미지 1장 / 이미지 폴더 / 영상 파일을 입력받아 ONNX YOLO 모델로 탐지하고,
바운딩박스를 그린 결과물과 프레임별 상세(run.jsonl)를 outputs/에 저장한다.
이미지는 길이 1짜리 영상과 같은 코드 경로로 처리한다.

사용법·계약: docs/RUNNER_SPEC.md
모델 계약 원문: model/bammasil_det_c4b_loli0_640/README.md
"""

import argparse
import ast
import ctypes
import json
import sys
import sysconfig
import time
from collections import Counter
from pathlib import Path
from typing import NoReturn

import cv2
import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VID_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
PAD_COLOR = 114  # letterbox 회색 패딩 (모델 카드 계약)
# 클래스 인덱스 순 BGR 색 (0: person=녹색, 1: stairs=적색 계열 위험색)
COLORS = [(96, 200, 96), (60, 76, 231), (255, 178, 50), (0, 130, 200)]


def _preload_bundled_cudart() -> None:
    """onnxruntime-gpu 1.27은 import 시점에 libcudart.so.13을 요구한다.

    시스템에 CUDA 13이 없어도 venv에 torch cu130이 동봉한 런타임으로 충족시킨다.
    CPU EP만 쓰므로 실제 CUDA 실행은 일어나지 않는다. (ADR-0001)
    """
    lib = Path(sysconfig.get_paths()["purelib"]) / "nvidia" / "cu13" / "lib" / "libcudart.so.13"
    if lib.exists():
        try:
            ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)
        except OSError:
            pass  # preload 실패 시 아래 import가 원인 메시지와 함께 실패한다


_preload_bundled_cudart()
import onnxruntime as ort  # noqa: E402


def die(msg: str) -> NoReturn:
    print(f"[abort] {msg}", file=sys.stderr)
    sys.exit(1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="src.runner", description="YOLO ONNX 추론 러너 (이미지/폴더/영상)")
    p.add_argument("--input", required=True, help="이미지 파일 / 이미지 폴더 / 영상 파일 (자동 판별)")
    p.add_argument("--model", required=True, help="ONNX 모델 경로")
    p.add_argument("--conf", type=float, default=0.25, help="confidence 임계값 (기본 0.25 · 모델 카드 권장 0.35)")
    p.add_argument("--iou", type=float, default=0.45, help="NMS IoU 임계값 (기본 0.45 · 모델 카드 권장 0.7)")
    p.add_argument("--stride", type=int, default=1, help="영상에서 N프레임마다 탐지, 나머지는 직전 박스 유지 (기본 1)")
    p.add_argument("--note", default=None, help="TESTLOG 회차 제목 (기본: 입력 파일명)")
    args = p.parse_args()
    if args.stride < 1:
        die(f"--stride 는 1 이상이어야 합니다: {args.stride}")
    if not 0.0 <= args.conf <= 1.0:
        die(f"--conf 는 0..1 범위여야 합니다: {args.conf}")
    return args


def load_session(model_path: Path):
    """세션 생성 + 모델 계약 검증. (input_size, class_names) 반환."""
    if not model_path.is_file():
        die(f"모델 파일이 없습니다: {model_path}")
    opts = ort.SessionOptions()
    opts.log_severity_level = 3  # CUDA 관련 경고 억제 — CPU EP만 사용
    sess = ort.InferenceSession(str(model_path), sess_options=opts, providers=["CPUExecutionProvider"])

    raw_names = sess.get_modelmeta().custom_metadata_map.get("names")
    if not raw_names:
        die("ONNX 메타데이터에 'names'(클래스명)가 없습니다. 클래스 순서를 임의 결정할 수 없어 중단합니다.")
    try:
        names_map = ast.literal_eval(raw_names)
        class_names = [str(names_map[i]) for i in range(len(names_map))]
    except (ValueError, SyntaxError, KeyError) as e:
        die(f"ONNX 메타데이터 'names' 를 해석할 수 없습니다: {raw_names!r} ({e})")

    ishape = sess.get_inputs()[0].shape
    if len(ishape) != 4 or not all(isinstance(d, int) for d in ishape) or ishape[2] != ishape[3]:
        die(f"예상 밖 입력 shape {ishape} — [1,3,S,S] static 이어야 합니다.")
    oshape = sess.get_outputs()[0].shape
    if len(oshape) != 3 or oshape[1] != 4 + len(class_names):
        die(
            f"예상 밖 출력 shape {oshape} — [1, 4+{len(class_names)}, anchors] 이어야 합니다. "
            "transpose 여부를 임의 결정하지 않고 중단합니다."
        )
    return sess, ishape[2], class_names


def resolve_input(path: Path):
    """입력 자동 판별 → ("images", [파일들]) 또는 ("video", 경로)."""
    if path.is_dir():
        files = sorted(p for p in path.iterdir() if p.suffix.lower() in IMG_EXTS)
        if not files:
            die(f"폴더에 이미지가 없습니다: {path} (지원 확장자 {sorted(IMG_EXTS)})")
        return "images", files
    if path.is_file():
        ext = path.suffix.lower()
        if ext in IMG_EXTS:
            return "images", [path]
        if ext in VID_EXTS:
            return "video", path
        die(f"지원하지 않는 확장자입니다: {path}")
    # 경로가 없을 때: 같은 스템의 파일이 있으면 그걸로 해석 (예: data/sample_gemini → data/sample_gemini.jpg)
    siblings = [p for p in path.parent.glob(path.name + ".*") if p.suffix.lower() in IMG_EXTS | VID_EXTS] \
        if path.parent.is_dir() else []
    if len(siblings) == 1:
        print(f"(입력 해석) {path} 없음 → 스템 일치 파일 {siblings[0]} 사용", file=sys.stderr)
        return resolve_input(siblings[0])
    hint = f" (비슷한 경로: {', '.join(str(s) for s in siblings)})" if siblings else ""
    die(f"입력 경로가 없습니다: {path}{hint}")


def letterbox(img: np.ndarray, size: int):
    """비율 유지 리사이즈 + 중앙 회색 패딩. (결과, scale, pad_left, pad_top) 반환."""
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    new_w, new_h = max(1, round(w * r)), max(1, round(h * r))  # 극단 종횡비에서 0 방지
    if (new_w, new_h) != (w, h):
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    left = (size - new_w) // 2
    top = (size - new_h) // 2
    canvas = np.full((size, size, 3), PAD_COLOR, dtype=np.uint8)
    canvas[top : top + new_h, left : left + new_w] = img
    return canvas, r, left, top


def preprocess(frame_bgr: np.ndarray, size: int):
    lb, r, left, top = letterbox(frame_bgr, size)
    x = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = np.ascontiguousarray(x.transpose(2, 0, 1)[None])  # HWC → 1CHW
    return x, r, left, top


def nms_per_class(xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray, iou_thres: float) -> np.ndarray:
    """클래스별 greedy NMS. 클래스 오프셋으로 한 번에 처리 (letterbox 좌표 < 4096)."""
    b = xyxy + (cls.astype(np.float32) * 4096.0)[:, None]
    areas = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    order = conf.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        rest = order[1:]
        xx1 = np.maximum(b[i, 0], b[rest, 0])
        yy1 = np.maximum(b[i, 1], b[rest, 1])
        xx2 = np.minimum(b[i, 2], b[rest, 2])
        yy2 = np.minimum(b[i, 3], b[rest, 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[iou <= iou_thres]
    return np.asarray(keep, dtype=int)


def postprocess(raw: np.ndarray, conf_thres: float, iou_thres: float, nc: int,
                r: float, left: int, top: int, orig_w: int, orig_h: int) -> list[dict]:
    """[1, 4+nc, anchors] → 원본 좌표계의 탐지 리스트.

    채널 0..3 = cx,cy,w,h (letterbox 픽셀), 채널 4.. = 클래스 점수(sigmoid 적용됨).
    """
    pred = raw[0].T  # (anchors, 4+nc)
    scores = pred[:, 4 : 4 + nc]
    cls = scores.argmax(axis=1)
    conf = scores[np.arange(len(cls)), cls]
    m = conf >= conf_thres
    if not m.any():
        return []
    boxes, conf, cls = pred[m, :4], conf[m], cls[m]
    xyxy = np.concatenate([boxes[:, :2] - boxes[:, 2:4] / 2, boxes[:, :2] + boxes[:, 2:4] / 2], axis=1)
    keep = nms_per_class(xyxy, conf, cls, iou_thres)
    xyxy, conf, cls = xyxy[keep], conf[keep], cls[keep]
    # letterbox 역변환 → 원본 좌표
    xyxy[:, [0, 2]] = np.clip((xyxy[:, [0, 2]] - left) / r, 0, orig_w)
    xyxy[:, [1, 3]] = np.clip((xyxy[:, [1, 3]] - top) / r, 0, orig_h)
    return [
        {"cls": int(c), "conf": round(float(s), 4), "xyxy": [round(float(v), 1) for v in b]}
        for b, s, c in zip(xyxy, conf, cls)
    ]


def draw(frame: np.ndarray, dets: list[dict], class_names: list[str]) -> np.ndarray:
    out = frame.copy()
    t = max(2, round(min(out.shape[:2]) / 400))
    fs = t / 3.5
    for d in dets:
        x1, y1, x2, y2 = (int(round(v)) for v in d["xyxy"])
        color = COLORS[d["cls"] % len(COLORS)]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, t)
        label = f"{class_names[d['cls']]} {d['conf']:.2f}"
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, t)
        ty = y1 - base if y1 - th - base >= 0 else y1 + th + base  # 위 공간 없으면 박스 안쪽
        cv2.rectangle(out, (x1, ty - th - base), (x1 + tw, ty + base), color, -1)
        cv2.putText(out, label, (x1, ty), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), t, cv2.LINE_AA)
    return out


def make_outdir() -> Path:
    base = Path("outputs") / time.strftime("%Y%m%d_%H%M")
    outdir, n = base, 2
    while outdir.exists():
        outdir = base.with_name(f"{base.name}_{n}")
        n += 1
    outdir.mkdir(parents=True)
    return outdir


def append_testlog(title: str, summary: dict) -> None:
    """docs/TESTLOG.md 최상단에 회차 추가. 파일 없으면 생성."""
    path = Path("docs/TESTLOG.md")
    entry = (
        f"## {time.strftime('%Y-%m-%d %H:%M')} — {title}\n\n"
        f"**결론**: 통과 — YOLO 로드 → 추론 → 렌더 연결 확인\n\n"
        f"| 항목 | 내용 |\n|---|---|\n"
        f"| 입력 | {summary['input']} |\n"
        f"| 모델 | {summary['model']} |\n"
        f"| 클래스 | {summary['classes']} |\n"
        f"| 환경 | {summary['env']} |\n"
        f"| 결과 | {summary['result']} |\n"
        f"| 산출물 | {summary['outdir']} |\n\n"
        f"**막힌 것**: 없음\n"
    )
    header = "# TESTLOG\n"
    if path.exists():
        text = path.read_text(encoding="utf-8")
        body = text[len(header):].lstrip("\n") if text.startswith(header) else text
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        body = ""
    path.write_text(f"{header}\n{entry}\n{body}", encoding="utf-8")


def main() -> None:
    args = parse_args()
    sess, size, class_names = load_session(Path(args.model))
    nc = len(class_names)
    input_name = sess.get_inputs()[0].name

    print(f"[env]   onnxruntime {ort.__version__} / EP: {sess.get_providers()[0]}")
    print(f"[model] {Path(args.model).name} / 입력 1x3x{size}x{size} / 클래스 {class_names}")

    mode, source = resolve_input(Path(args.input))
    outdir = make_outdir()
    det_counter = Counter()  # 추론 프레임의 클래스별 탐지 수
    ms_list = []          # 추론 소요 ms
    zero_frames = 0       # 추론했는데 탐지 0건인 프레임 수

    def jsonable(dets: list[dict]) -> list[dict]:
        """run.jsonl용 — cls 인덱스를 클래스명으로 바꿔 파일 단독으로도 읽히게 한다."""
        return [{**d, "cls": class_names[d["cls"]]} for d in dets]

    def infer(frame_bgr: np.ndarray) -> tuple[list[dict], float]:
        x, r, left, top = preprocess(frame_bgr, size)
        t0 = time.perf_counter()
        raw = sess.run(None, {input_name: x})[0]
        ms = (time.perf_counter() - t0) * 1000.0
        h, w = frame_bgr.shape[:2]
        return postprocess(raw, args.conf, args.iou, nc, r, left, top, w, h), ms

    # run.jsonl은 프레임마다 스트리밍 기록 — 중도 실패해도 그때까지의 레코드가 남는다
    with open(outdir / "run.jsonl", "w", encoding="utf-8") as run_fp:

        def emit(rec: dict) -> None:
            run_fp.write(json.dumps(rec, ensure_ascii=False) + "\n")

        if mode == "images":
            print(f"[input] {args.input} — 이미지 {len(source)}장")
            for i, f in enumerate(source):
                frame = cv2.imread(str(f))
                if frame is None:
                    die(f"이미지를 읽을 수 없습니다: {f}")
                dets, ms = infer(frame)
                ms_list.append(ms)
                det_counter.update(d["cls"] for d in dets)
                zero_frames += not dets
                cv2.imwrite(str(outdir / f.name), draw(frame, dets, class_names))
                emit({"frame": i, "source": f.name, "inferred": True,
                      "ms": round(ms, 1), "boxes": jsonable(dets)})
            n_total, unit = len(source), "장"

        else:  # video
            cap = cv2.VideoCapture(str(source))
            if not cap.isOpened():
                die(f"영상을 열 수 없습니다: {source}")
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            n_hint = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f"[input] {args.input} — 영상 {n_hint}프레임, stride {args.stride}")
            writer = cv2.VideoWriter(str(outdir / f"{source.stem}_det.mp4"),
                                     cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            if not writer.isOpened():
                die("VideoWriter(mp4v) 를 열 수 없습니다.")
            held: list[dict] = []
            reps: list[tuple[int, int, np.ndarray]] = []  # (탐지수, frame_idx, 렌더 프레임) 상위 3
            first_frame = last_frame = None
            i = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                inferred = i % args.stride == 0
                if inferred:
                    held, ms = infer(frame)
                    ms_list.append(ms)
                    det_counter.update(d["cls"] for d in held)
                    zero_frames += not held
                rendered = draw(frame, held, class_names)
                writer.write(rendered)
                emit({"frame": i, "source": None, "inferred": inferred,
                      "ms": round(ms, 1) if inferred else None, "boxes": jsonable(held)})
                if inferred and held:
                    reps.append((len(held), i, rendered))
                    reps = sorted(reps, key=lambda x: (-x[0], x[1]))[:3]
                if first_frame is None:
                    first_frame = (i, rendered)
                last_frame = (i, rendered)
                i += 1
            cap.release()
            writer.release()
            if not i:
                die(f"영상에서 프레임을 읽지 못했습니다: {source}")
            for _, idx, img in reps:
                cv2.imwrite(str(outdir / f"rep_f{idx:06d}.jpg"), img)
            if not reps:  # 탐지 0건 영상 — 첫/끝 프레임을 대표로 저장
                for idx, img in {first_frame[0]: first_frame, last_frame[0]: last_frame}.values():
                    cv2.imwrite(str(outdir / f"rep_f{idx:06d}.jpg"), img)
            n_total, unit = i, "프레임"

    avg_ms = sum(ms_list) / len(ms_list)
    per_class = ", ".join(f"{class_names[c]} {det_counter[c]}건" for c in range(nc))
    # stride>1이면 유지 프레임은 집계 대상이 아니므로 분모(추론 프레임 수)를 함께 표기
    zero_str = f"{zero_frames}/{len(ms_list)} 추론프레임" if mode == "video" else f"{zero_frames}장"
    print(f"[run]   {n_total}/{n_total} 완료 (avg {avg_ms:.0f}ms/frame, platform: pc-cpu — 성능 판정 불가)")
    print(f"[det]   {per_class} / 탐지 0건: {zero_str}")
    print(f"[out]   {outdir}/")

    stride_note = f" (stride {args.stride})" if mode == "video" else ""
    append_testlog(args.note or Path(args.input).name, {
        "input": f"{args.input} — {n_total}{unit}{stride_note}",
        "model": f"{Path(args.model).name} / {size} static",
        "classes": ", ".join(class_names) + " (ONNX 메타 names)",
        "env": f"PC CPU · onnxruntime {ort.__version__} · {sess.get_providers()[0]}",
        "result": f"{per_class} / 탐지 0건 {zero_str}",
        "outdir": f"{outdir}/",
    })


if __name__ == "__main__":
    main()
