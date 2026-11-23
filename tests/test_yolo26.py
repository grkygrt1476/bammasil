import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("yolo26", Path(__file__).resolve().parents[1] / "tools/yolo26.py")
y = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(y)


class DatasetGuards(unittest.TestCase):
    def test_fingerprint_ignores_path_but_preserves_split_and_pairing(self):
        a = [{"split": "train", "sha256": "a", "label_sha256": "b", "image": "old/path"}]
        b = [dict(a[0], image="new/path")]
        self.assertEqual(y.fingerprint(a), y.fingerprint(b))
        self.assertNotEqual(y.fingerprint(a), y.fingerprint([dict(b[0], split="val")]))
        self.assertNotEqual(y.fingerprint(a), y.fingerprint([dict(b[0], label_sha256="changed")]))

    def test_missing_label_and_invalid_box_rejected(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base / "images/train").mkdir(parents=True)
            (base / "labels/train").mkdir(parents=True)
            Image.new("RGB", (8, 8)).save(base / "images/train/a.png")
            with self.assertRaisesRegex(ValueError, "mismatch"):
                y.scan(base, ["train"])
            label = base / "labels/train/a.txt"
            label.write_text("1 0.9 0.5 0.8 0.2")
            with self.assertRaisesRegex(ValueError, "Invalid box"):
                y.scan(base, ["train"])
            label.write_text("")
            self.assertEqual(len(y.scan(base, ["train"])), 1)

    def test_modified_file_rejected_after_preparation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "a.jpg"
            label = root / "a.txt"
            image.write_bytes(b"original")
            label.write_bytes(b"")
            records = [{"split": "train", "image": str(image), "label": str(label), "sha256": y.sha(image), "label_sha256": y.sha(label)}]
            y.write(root / "prepared/legacy_c4d/audit.json", {"records": records})
            cfg = root / "config"
            y.write(cfg / "datasets.json", {"legacy_c4d": {"content_sha256": y.fingerprint(records)}})
            image.write_bytes(b"changed")
            with patch.object(y, "CONFIG", cfg), self.assertRaisesRegex(ValueError, "changed after preparation"):
                y.check_data(root, "legacy_c4d")


if __name__ == "__main__":
    unittest.main()
