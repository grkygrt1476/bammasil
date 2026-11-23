"""Combine verified historical and new fixed-640 external evaluations."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def load(p):
    return json.loads(p.read_text(encoding="utf-8"))

def sha(p):
    with p.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("training_root", type=Path)
    p.add_argument("evaluation_root", type=Path)
    args = p.parse_args()
    summary = load(ROOT / "docs/experiments/yolo26-summary.json")
    cfg = load(ROOT / "configs/yolo26/evaluation.json")
    datasets = load(ROOT / "configs/yolo26/datasets.json")
    dest = ROOT / "docs/experiments/external-640"
    dest.mkdir(exist_ok=True)
    table = []
    for run, details in summary["runs"].items():
        for dataset, tag in [("bollard_dev_disjoint", "bollard_dev_disjoint"), ("nightowls_rec34", "night_person")]:
            reused = run in ("n640", "m640")
            source = args.training_root / details["run_id"] / "evaluation" / f"{tag}.json" if reused else args.evaluation_root / f"{run}-{dataset}" / "evaluation.json"
            record = load(source)
            if record["weights_sha256"] != details["best_sha256"]:
                raise ValueError(f"Weights mismatch: {run}")
            normalized = dict(record["settings"])
            # Ultralytics 8.4.106 maps half=False to quantize=None in-place.
            if "half" not in normalized and "quantize" in normalized and normalized["quantize"] is None:
                normalized["half"] = False
            for k,v in cfg.items():
                if normalized.get(k) != v:
                    raise ValueError(f"Settings mismatch: {run}/{dataset}/{k}")
            manifest = args.training_root / "reports" / f"{dataset}.manifest.jsonl"
            entries = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
            canonical = sorted((r["split"], r["sha256"], r["label_sha256"]) for r in entries)
            data_hash = hashlib.sha256(json.dumps(canonical,separators=(",", ":")).encode()).hexdigest()
            if data_hash != datasets[dataset]["content_sha256"]:
                raise ValueError("Dataset content mismatch")
            if reused and sha(manifest) != record["manifest_sha256"]:
                raise ValueError("Original manifest mismatch")
            if not reused and data_hash != record["data_content_sha256"]:
                raise ValueError("New evaluation dataset mismatch")
            clean = {"run":run, "dataset":dataset, "weights_sha256":record["weights_sha256"], "data_content_sha256":data_hash, "settings":cfg, "precision_normalization":"half=False and quantize=None are equivalent in Ultralytics 8.4.106", "per_class":record["per_class"], "speed_ms":record["speed_ms"], "source_record_sha256":sha(source), "source_kind":"existing_2026-10-07_evaluation" if reused else "new_2026-11-23_evaluation"}
            (dest/f"{run}-{dataset}.json").write_bytes((json.dumps(clean,indent=2)+"\n").encode())
            for cls,metrics in clean["per_class"].items():
                table.append({"run":run,"dataset":dataset,"class":cls,"ap50":metrics["ap50"],"ap50_95":metrics["ap50_95"]})
    buf=io.StringIO(newline="")
    writer=csv.DictWriter(buf,fieldnames=list(table[0]),lineterminator="\n")
    writer.writeheader();writer.writerows(table)
    (dest/"comparison.csv").write_bytes(buf.getvalue().encode())
    print(f"Exported {len(table)} class metrics from 10 verified evaluations.")

if __name__ == "__main__":
    main()
