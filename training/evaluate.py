from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from receiptlab.extraction import extract, parse_amount
from receiptlab.ml import RulesPredictor, get_predictor
from receiptlab.ocr import run_ocr
from training.data import load_records
from training.train import line_metrics

FIELDS = ("subtotal", "tax", "discount", "total")


def field_metrics(records, outputs):
    metrics = {}
    for field in FIELDS:
        present = correct_present = all_correct = ambiguous = 0
        for record, output in zip(records, outputs, strict=True):
            candidates = {
                parse_amount(line["text"]) for line in record["lines"] if line["label"] == field
            }
            candidates.discard(None)
            if len(candidates) > 1:
                ambiguous += 1
                continue
            expected = next(iter(candidates)) if candidates else None
            actual = output["fields"][field]["value"]
            all_correct += int(actual == expected)
            if expected is not None:
                present += 1
                correct_present += int(actual == expected)
        evaluated = len(records) - ambiguous
        metrics[field] = {
            "present_documents": present,
            "correct_present": correct_present,
            "present_value_accuracy": correct_present / present if present else None,
            "presence_and_value_accuracy": all_correct / evaluated if evaluated else None,
            "evaluated_documents": evaluated,
            "ambiguous_truth_documents_excluded": ambiguous,
        }
    return metrics


def annotated_evaluation(records, predictor):
    truth, guesses, outputs, milliseconds = [], [], [], []
    for record in records:
        started = time.perf_counter()
        result = extract(record["lines"], predictor)
        milliseconds.append((time.perf_counter() - started) * 1000)
        outputs.append(result)
        truth.extend(line["label"] for line in record["lines"])
        guesses.extend(line["label"] for line in result["lines"])
    result = line_metrics(truth, guesses)
    result.update(
        {
            "documents": len(records),
            "input": "CORD annotated text and boxes; excludes OCR errors",
            "fields": field_metrics(records, outputs),
            "latency_ms": {
                "median": float(np.median(milliseconds)),
                "p95": float(np.percentile(milliseconds, 95)),
            },
            "documents_requiring_review": sum(output["needs_review"] for output in outputs),
        }
    )
    return result


def intersection_over_union(left, right):
    width = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    height = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    intersection = width * height
    total = (
        (left[2] - left[0]) * (left[3] - left[1])
        + (right[2] - right[0]) * (right[3] - right[1])
        - intersection
    )
    return intersection / total if total > 0 else 0.0


def normalized_text(text):
    return re.sub(r"\s+", "", text).casefold()


def ocr_evaluation(records, predictor, count):
    if count < 1 or count > len(records):
        raise ValueError("OCR document count must be between 1 and test split size.")
    indices = sorted(np.random.RandomState(42).choice(len(records), count, replace=False).tolist())
    chosen = [records[index] for index in indices]
    outputs, timings, details = [], [], []
    total_annotations = detected_annotations = exact_text = 0
    for record in chosen:
        started = time.perf_counter()
        try:
            with Image.open(record["image_path"]) as image:
                lines = run_ocr(image)
            output = extract(lines, predictor)
            detected = exact = 0
            for reference in record["lines"]:
                overlaps = [
                    (intersection_over_union(reference["bbox"], line["bbox"]), line)
                    for line in lines
                ]
                best = max(overlaps, key=lambda pair: pair[0]) if overlaps else (0, None)
                if best[0] >= 0.25:
                    detected += 1
                    exact += int(
                        normalized_text(best[1]["text"]) == normalized_text(reference["text"])
                    )
            detected_annotations += detected
            exact_text += exact
            total_annotations += len(record["lines"])
            details.append(
                {
                    "id": record["id"],
                    "ocr_lines": len(lines),
                    "annotated_lines": len(record["lines"]),
                    "annotations_detected_iou_025": detected,
                    "annotations_exact_text_iou_025": exact,
                }
            )
        except Exception as error:
            output = {
                "fields": {
                    field: {"value": None, "confidence": 0.0, "bbox": None} for field in FIELDS
                },
                "needs_review": True,
            }
            total_annotations += len(record["lines"])
            details.append({"id": record["id"], "failed": type(error).__name__})
        outputs.append(output)
        timings.append((time.perf_counter() - started) * 1000)
        print(f"OCR evaluation {len(outputs)}/{count}", flush=True)
    return {
        "documents": count,
        "sampling": "Seed 42 without replacement from official test split; count fixed before evaluation",
        "input": "Actual CORD receipt pixels -> RapidOCR ONNX -> frozen line classifier -> deterministic extraction",
        "fields": field_metrics(chosen, outputs),
        "annotated_line_detection_recall_iou_025": detected_annotations / total_annotations
        if total_annotations
        else 0.0,
        "annotated_line_exact_text_recall_iou_025": exact_text / total_annotations
        if total_annotations
        else 0.0,
        "annotation_matching_note": "Matches reference lines to highest-IoU OCR box; split/merged lines conservatively penalized. Unannotated text has no reference. This is not character error rate.",
        "latency_ms": {
            "median": float(np.median(timings)),
            "p95": float(np.percentile(timings, 95)),
        },
        "documents_requiring_review": sum(output["needs_review"] for output in outputs),
        "failures": sum("failed" in entry for entry in details),
        "documents_detail": details,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Final evaluation on untouched official CORD test documents"
    )
    parser.add_argument("--data-dir", default="data/cord")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--ocr-documents", type=int, default=20)
    args = parser.parse_args()
    predictor = get_predictor(args.model_dir)
    records = load_records(args.data_dir, "test")
    print("Evaluating annotated-text baseline and frozen learned model...", flush=True)
    baseline = annotated_evaluation(records, RulesPredictor())
    learned = annotated_evaluation(records, predictor)
    print("Evaluating real OCR separately...", flush=True)
    real_ocr = ocr_evaluation(records, predictor, args.ocr_documents)
    report = {
        "dataset": "naver-clova-ix/cord-v2",
        "split": "official test",
        "test_documents": len(records),
        "model_version": predictor.model_version,
        "confidence_threshold": predictor.confidence_threshold,
        "test_used_for_parameter_selection": False,
        "test_split_sha256": hashlib.sha256(
            (Path(args.data_dir) / "test.jsonl").read_bytes()
        ).hexdigest(),
        "label_support": dict(
            Counter(line["label"] for record in records for line in record["lines"])
        ),
        "baseline": baseline,
        "learned": learned,
        "actual_ocr_pipeline": real_ocr,
        "limitations": [
            "These are Indonesian receipt results, not invoice or Arabic accuracy.",
            "Line label F1 and document financial extraction accuracy measure different tasks.",
            "Actual OCR benchmark is a seeded subset, separate from all 100 annotated-text receipts.",
            "Validation review threshold is not a correctness guarantee or permission to autoapprove expenses.",
        ],
    }
    output = Path(args.reports_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    total = learned["fields"]["total"]["present_value_accuracy"]
    ocr_total = real_ocr["fields"]["total"]["present_value_accuracy"]
    card = f"""# ReceiptLab line classifier model card

The frozen model classifies receipt lines into nine roles using word/character TF-IDF,
normalized geometry, adjacent line text, and multiclass logistic regression.
Amount parsing and review decisions remain deterministic application code.

## Data and experiment protocol

CORD-v2, official document splits: 800 train, 100 validation, 100 test.
Scalar probability temperature and review threshold were selected on validation only.
Test was evaluated after the artifact was frozen. Training uses annotated text and boxes,
not OCR-generated input. See models/manifest.json for the pinned dataset revision.

## Measured results

- Rules baseline test macro F1: {baseline["macro_f1"]:.4f}
- Learned model test macro F1: {learned["macro_f1"]:.4f}
- Learned target-role macro F1: {learned["target_macro_f1"]:.4f}
- Annotated-text total exact-value accuracy: {total}
- Actual OCR total exact-value accuracy ({args.ocr_documents} seeded test documents): {ocr_total}
- Validation-selected review threshold: {predictor.confidence_threshold}

See evaluation.json for class support, precision/recall, absent/present field handling,
OCR matching assumptions, and latency. These results do not establish invoice, Arabic,
or operational financial accuracy.

## Intended use and limitations

Portfolio research and assistive receipt review. Trained on Indonesian receipts.
Currency is not inferred. Ambiguous amounts, missing totals, low confidence and failed
arithmetic require human review. Line probabilities do not establish document confidence.
Uploaded documents cannot override model artifacts or load pickled objects.
"""
    (output / "model-card.md").write_text(card, encoding="utf-8")
    print(
        json.dumps(
            {
                "baseline_macro_f1": baseline["macro_f1"],
                "learned_macro_f1": learned["macro_f1"],
                "annotated_total_accuracy": total,
                "actual_ocr_total_accuracy": ocr_total,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
