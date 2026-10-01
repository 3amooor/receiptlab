from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

DATASET_ID = "naver-clova-ix/cord-v2"
EXPECTED_COUNTS = {"train": 800, "validation": 100, "test": 100}
LABELS = (
    "other",
    "item_description",
    "item_quantity",
    "item_unit_price",
    "item_amount",
    "subtotal",
    "tax",
    "discount",
    "total",
)
CATEGORY_MAP = {
    "menu.nm": "item_description",
    "menu.sub.nm": "item_description",
    "menu.sub_nm": "item_description",
    "menu.cnt": "item_quantity",
    "menu.sub.cnt": "item_quantity",
    "menu.sub_cnt": "item_quantity",
    "menu.unitprice": "item_unit_price",
    "menu.sub_unitprice": "item_unit_price",
    "menu.price": "item_amount",
    "menu.sub.price": "item_amount",
    "menu.sub_price": "item_amount",
    "menu.itemsubtotal": "item_amount",
    "sub_total.subtotal_price": "subtotal",
    "subtotal.subtotal_price": "subtotal",
    "sub_total.tax_price": "tax",
    "subtotal.tax_price": "tax",
    "sub_total.discount_price": "discount",
    "subtotal.discount_price": "discount",
    "total.total_price": "total",
}


def load_records(data_dir, split):
    destination = Path(data_dir)
    if split not in EXPECTED_COUNTS:
        raise ValueError(f"Unknown official split: {split}")
    path = destination / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"CORD cache missing: {path}. Run python -m training.data.")
    records = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if len(records) != EXPECTED_COUNTS[split]:
        raise ValueError(
            f"{split} requires {EXPECTED_COUNTS[split]} official documents; found {len(records)}"
        )
    if len({record["id"] for record in records}) != len(records):
        raise ValueError(f"Duplicate document identifiers in {split}")
    for record in records:
        image_path = Path(record["image_path"])
        if not image_path.is_absolute():
            record["image_path"] = str((destination / image_path).resolve())
    return records


def prepare_dataset(data_dir="data/cord", revision="main"):
    from datasets import load_dataset
    from huggingface_hub import HfApi

    destination = Path(data_dir)
    destination.mkdir(parents=True, exist_ok=True)
    if (
        all((destination / f"{split}.jsonl").exists() for split in EXPECTED_COUNTS)
        and (destination / "dataset.json").exists()
    ):
        for split in EXPECTED_COUNTS:
            load_records(destination, split)
        return json.loads((destination / "dataset.json").read_text(encoding="utf-8"))
    frozen_revision = HfApi().dataset_info(DATASET_ID, revision=revision, token=False).sha
    print(f"Downloading public CORD-v2 at {frozen_revision}", flush=True)
    dataset = load_dataset(
        DATASET_ID,
        revision=frozen_revision,
        cache_dir=str(destination.parent / "huggingface"),
        token=False,
    )
    fingerprints = {}
    for split, expected in EXPECTED_COUNTS.items():
        if len(dataset[split]) != expected:
            raise ValueError(f"Unexpected {DATASET_ID} split: {split}={len(dataset[split])}")
        fingerprints[split] = dataset[split]._fingerprint
        output = destination / f"{split}.jsonl"
        temporary = output.with_suffix(".jsonl.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            for index, sample in enumerate(dataset[split]):
                image = sample["image"].convert("RGB")
                width, height = image.size
                truth = sample["ground_truth"]
                if isinstance(truth, str):
                    truth = json.loads(truth)
                lines = []
                for annotation in truth.get("valid_line", []):
                    texts, xs, ys, cached_words = [], [], [], []
                    for word in annotation.get("words", []):
                        text = str(word.get("text", "")).strip()
                        quad = word.get("quad", {})
                        wx = [float(quad[f"x{n}"]) for n in range(1, 5)]
                        wy = [float(quad[f"y{n}"]) for n in range(1, 5)]
                        box = [min(wx) / width, min(wy) / height, max(wx) / width, max(wy) / height]
                        box = [min(1.0, max(0.0, value)) for value in box]
                        texts.append(text)
                        xs.extend(wx)
                        ys.extend(wy)
                        cached_words.append({"text": text, "bbox": box})
                    text = " ".join(texts).strip()
                    if not text or not xs:
                        continue
                    box = [min(xs) / width, min(ys) / height, max(xs) / width, max(ys) / height]
                    lines.append(
                        {
                            "text": text,
                            "bbox": [min(1.0, max(0.0, value)) for value in box],
                            "confidence": 1.0,
                            "label": CATEGORY_MAP.get(annotation.get("category"), "other"),
                            "source_category": annotation.get("category", ""),
                            "words": cached_words,
                        }
                    )
                lines.sort(
                    key=lambda line: ((line["bbox"][1] + line["bbox"][3]) / 2, line["bbox"][0])
                )
                relative = Path("images") / split / f"{split}-{index:04d}.png"
                image_path = destination / relative
                image_path.parent.mkdir(parents=True, exist_ok=True)
                image.save(image_path, "PNG")
                record = {
                    "id": f"{split}-{index:04d}",
                    "image_path": relative.as_posix(),
                    "width": width,
                    "height": height,
                    "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
                    "lines": lines,
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        temporary.replace(output)
        print(f"Cached {split}: {expected} documents", flush=True)
    metadata = {
        "dataset": DATASET_ID,
        "revision": frozen_revision,
        "counts": EXPECTED_COUNTS,
        "fingerprints": fingerprints,
        "labels": list(LABELS),
        "category_map": CATEGORY_MAP,
        "note": "Official document splits; categories merged into nine receipt line roles.",
    }
    (destination / "dataset.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cache official CORD-v2 document splits")
    parser.add_argument("--data-dir", default="data/cord")
    parser.add_argument("--revision", default="main")
    args = parser.parse_args()
    prepare_dataset(args.data_dir, args.revision)
