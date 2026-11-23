"""Export completed local runs without copying datasets or model weights."""
import argparse
import csv
import hashlib
import io
import json
import re
from pathlib import Path


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write_json(path, value):
    path.write_bytes((json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_root", type=Path)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    dest = repo / "docs/experiments"
    summary = json.loads((dest / "yolo26-summary.json").read_text(encoding="utf-8"))
    rows = []
    for name, record in summary["runs"].items():
        src = args.source_root / record["run_id"]
        raw = {}
        for filename, expected in record["source_file_sha256"].items():
            if sha(src / filename) != expected:
                raise ValueError(f"Source changed: {name}/{filename}")
            raw[filename] = json.loads((src / filename).read_text(encoding="utf-8"))
        if sha(src / "weights/best.pt") != record["best_sha256"]:
            raise ValueError(f"Checkpoint mismatch: {name}")
        progress = raw["progress.json"]
        manifest = raw["run_manifest.json"]
        if progress["state"] != "complete" or progress["epoch"] != 50:
            raise ValueError(f"Incomplete run: {name}")
        if progress["metrics"] != record["metrics"]:
            raise ValueError(f"Metric mismatch: {name}")
        if manifest["train_manifest_sha256"] != record["data_manifest_sha256"]:
            raise ValueError(f"Dataset mismatch: {name}")
        for key, value in record["settings"].items():
            if manifest["resolved_args"][key] != value:
                raise ValueError(f"Setting mismatch: {name}/{key}")
        curve = list(csv.DictReader((src / "results.csv").read_text(encoding="utf-8").splitlines()))
        if [int(r["epoch"]) for r in curve] != list(range(1, 51)):
            raise ValueError(f"Epoch sequence mismatch: {name}")
        out = dest / name
        out.mkdir(exist_ok=True)
        for filename in ("results.csv", "results.png", "progress.json"):
            (out / filename).write_bytes((src / filename).read_bytes())
        settings = dict(manifest["resolved_args"])
        replacements = {
            "model": f"models/pretrained/{manifest['model']}.pt",
            "data": "data/legacy_c4d/data.yaml",
            "project": "outputs/reproduction",
            "save_dir": f"outputs/reproduction/{name}",
        }
        settings.update(replacements)
        payload = {
            "model": manifest["model"],
            "environment": {k: manifest[k] for k in ("python", "platform", "torch", "ultralytics", "gpu")},
            "resolved_args": settings,
            "path_fields_replaced": list(replacements),
            "note": "Recorded settings with portable path references; training entry point is not yet integrated.",
        }
        if re.search(r"[A-Za-z]:\\|/home/|/Users/", json.dumps(payload)):
            raise ValueError(f"Absolute path in export: {name}")
        write_json(out / "settings.json", payload)
        completion = dict(raw["training_complete.json"])
        completion["best"] = "weights/best.pt"
        write_json(out / "training_complete.json", completion)
        write_json(out / "provenance.json", {
            "source_run_id": record["run_id"],
            "source_sha256": {f: sha(src / f) for f in ("run_manifest.json", "args.yaml", "progress.json", "training_complete.json", "results.csv", "results.png")},
            "checkpoint_sha256": record["best_sha256"],
            "pretrained_sha256": record["pretrained_sha256"],
            "data_manifest_sha256": record["data_manifest_sha256"],
            "published_sha256": {f.name: sha(f) for f in sorted(out.iterdir()) if f.name != "provenance.json"},
        })
        rows.append({"run": name, "model": manifest["model"], **{k: settings[k] for k in ("imgsz", "epochs", "batch", "seed")}, **record["metrics"], "elapsed_seconds": record["elapsed_seconds"], "peak_allocated_gib": record["peak_allocated_gib"], "best_sha256": record["best_sha256"]})
        print(f"Verified and exported {name}: 50 epochs, settings, metrics, checkpoint SHA-256")
    buf = io.StringIO(newline="")
    writer = csv.DictWriter(buf, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    (dest / "comparison.csv").write_bytes(buf.getvalue().encode("utf-8"))


if __name__ == "__main__":
    main()
