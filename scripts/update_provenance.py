"""Record installed versions and dataset overlap checks alongside frozen artifacts."""

import importlib.metadata
import json
from pathlib import Path

from training.data import load_records

ROOT = Path(__file__).resolve().parents[1]


def main():
    path = ROOT / "models" / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["library_versions"] = {
        name: importlib.metadata.version(name)
        for name in ("scikit-learn", "numpy", "scipy", "joblib")
    }
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    hashes = {
        s: {r["image_sha256"] for r in load_records(ROOT / "data/cord", s)}
        for s in ("train", "validation", "test")
    }
    report = {
        "official_document_counts": {"train": 800, "validation": 100, "test": 100},
        "exact_image_hash_overlap": {
            a + "_" + b: len(hashes[a] & hashes[b])
            for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))
        },
    }
    (ROOT / "reports" / "data-integrity.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
