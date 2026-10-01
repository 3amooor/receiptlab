"""Additional operational evaluation: raw model and explicit text assistance."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from receiptlab.assisted import assisted_extract
from receiptlab.ml import get_predictor
from receiptlab.ocr import run_ocr
from training.data import load_records
from training.evaluate import field_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--documents", type=int, default=20)
    parser.add_argument("--data-dir", default="data/cord")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--reports-dir", default="reports")
    args = parser.parse_args()
    records = load_records(args.data_dir, args.split)
    if not 1 <= args.documents <= len(records):
        raise ValueError("Invalid evaluation document count")
    selected = [
        records[i]
        for i in sorted(
            np.random.RandomState(42).choice(len(records), args.documents, replace=False).tolist()
        )
    ]
    predictor = get_predictor(args.model_dir)
    outputs, raw, detail, timings = [], [], [], []
    for record in selected:
        started = time.perf_counter()
        with Image.open(record["image_path"]) as image:
            output = assisted_extract(run_ocr(image), predictor)
        outputs.append(output)
        raw.append(output["raw_extraction"])
        timings.append((time.perf_counter() - started) * 1000)
        detail.append(
            {
                "id": record["id"],
                "anchor_fields": {
                    k: v["value"]
                    for k, v in output["fields"].items()
                    if v.get("origin") == "text_anchor"
                },
                "filtered_items": output["filtered_item_count"],
            }
        )
        print(f"Assisted {args.split}: {len(detail)}/{args.documents}", flush=True)
    source = Path(__file__).resolve().parents[1] / "receiptlab" / "assisted.py"
    report = {
        "dataset": "naver-clova-ix/cord-v2",
        "split": args.split,
        "documents": args.documents,
        "seed": 42,
        "model_version": predictor.model_version,
        "assist_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "input": "Actual pixels -> RapidOCR -> frozen learned classifier -> explicit text anchors",
        "parameters_selected_on_test": False,
        "development_note": "Assisted design followed raw benchmark failure analysis; geometric rules were fixed before separate validation and additional test evaluation. No test values were used to tune parameters.",
        "raw_learned_fields": field_metrics(selected, raw),
        "assisted_fields": field_metrics(selected, outputs),
        "documents_requiring_review": sum(o["needs_review"] for o in outputs),
        "latency_ms": {
            "median": float(np.median(timings)),
            "p95": float(np.percentile(timings, 95)),
        },
        "detail": detail,
        "limitations": [
            "Text recovery is deterministic assistance, not trained model accuracy.",
            "Anchor confidence is zero and every document requires human review.",
            "This additional operational evaluation does not replace the original model benchmark.",
        ],
    }
    destination = Path(args.reports_dir)
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / f"{args.split}-assisted-evaluation.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "split": args.split,
                "raw_total": report["raw_learned_fields"]["total"],
                "assisted_total": report["assisted_fields"]["total"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
