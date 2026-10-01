from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.metrics import accuracy_score, classification_report, f1_score, log_loss

from receiptlab.ml import LineClassifier, calibrated_probabilities
from training.data import LABELS, load_records, prepare_dataset


def collect_probabilities(model, records):
    arrays, truth = [], []
    for record in records:
        arrays.append(model.probabilities(record["lines"]))
        truth.extend(line["label"] for line in record["lines"])
    return np.vstack(arrays), np.asarray(truth)


def choose_temperature(probabilities, truth, classes):
    lookup = {label: index for index, label in enumerate(classes)}
    target = np.asarray([lookup[label] for label in truth])

    def loss(temperature):
        adjusted = calibrated_probabilities(probabilities, temperature)
        return float(-np.mean(np.log(np.clip(adjusted[np.arange(len(target)), target], 1e-12, 1))))

    result = minimize_scalar(loss, bounds=(0.25, 5.0), method="bounded")
    return float(result.x)


def choose_threshold(probabilities, truth, classes, target_precision=0.9):
    guesses = np.asarray(classes)[np.argmax(probabilities, axis=1)]
    confidences = probabilities.max(axis=1)
    selected = []
    for step in range(30, 100):
        threshold = step / 100
        accepted = (confidences >= threshold) & (guesses != "other")
        count = int(accepted.sum())
        precision = float((guesses[accepted] == truth[accepted]).mean()) if count else 0.0
        selected.append(
            {
                "threshold": round(float(threshold), 2),
                "accepted_lines": count,
                "precision": precision,
                "coverage": count / len(truth),
            }
        )
    feasible = [
        row
        for row in selected
        if row["accepted_lines"] >= 30 and row["precision"] >= target_precision
    ]
    chosen = (
        max(feasible, key=lambda row: row["coverage"])
        if feasible
        else {"threshold": 1.0, "accepted_lines": 0, "precision": 0.0, "coverage": 0.0}
    )
    return float(chosen["threshold"]), {
        "target_precision": target_precision,
        "target_met": bool(feasible),
        "selection_scope": "validation non-other predicted lines",
        "minimum_accepted_lines": 30,
        "selected": chosen,
        "candidates": selected,
    }


def line_metrics(truth, predictions):
    return {
        "accuracy": float(accuracy_score(truth, predictions)),
        "macro_f1": float(
            f1_score(truth, predictions, labels=list(LABELS), average="macro", zero_division=0)
        ),
        "target_macro_f1": float(
            f1_score(truth, predictions, labels=list(LABELS[1:]), average="macro", zero_division=0)
        ),
        "weighted_f1": float(f1_score(truth, predictions, average="weighted", zero_division=0)),
        "per_class": classification_report(
            truth, predictions, labels=list(LABELS), output_dict=True, zero_division=0
        ),
        "lines": len(truth),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train receipt line classifier on official CORD document split"
    )
    parser.add_argument("--data-dir", default="data/cord")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--mlflow-uri", default=None)
    args = parser.parse_args()
    if args.prepare:
        prepare_dataset(args.data_dir)
    train = load_records(args.data_dir, "train")
    validation = load_records(args.data_dir, "validation")
    started = time.perf_counter()
    model = LineClassifier().fit(train)
    raw, truth = collect_probabilities(model, validation)
    classes = model.classifier.classes_
    model.temperature = choose_temperature(raw, truth, classes)
    calibrated = calibrated_probabilities(raw, model.temperature)
    model.confidence_threshold, selection = choose_threshold(calibrated, truth, classes)
    predictions = classes[np.argmax(calibrated, axis=1)]
    metrics = line_metrics(truth, predictions)
    metrics.update(
        {
            "split": "validation",
            "documents": len(validation),
            "temperature": model.temperature,
            "nll_before": float(log_loss(truth, raw, labels=classes)),
            "nll_after": float(log_loss(truth, calibrated, labels=classes)),
            "threshold_selection": selection,
            "training_seconds": time.perf_counter() - started,
        }
    )
    model_dir, reports_dir = Path(args.model_dir), Path(args.reports_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    artifact = model_dir / "line-classifier.joblib"
    joblib.dump(model, artifact, compress=3)
    manifest = {
        "model_version": model.model_version,
        "model_kind": "tfidf-logistic-line-classifier",
        "artifact": artifact.name,
        "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "training_documents": len(train),
        "validation_documents": len(validation),
        "training_lines": sum(len(record["lines"]) for record in train),
        "label_counts": dict(
            Counter(line["label"] for record in train for line in record["lines"])
        ),
        "labels": list(classes),
        "temperature": model.temperature,
        "confidence_threshold": model.confidence_threshold,
        "calibration": "Validation-only scalar temperature minimizes multiclass negative log likelihood. Review threshold seeks 90% accuracy among non-other predictions, minimum 30 accepted lines.",
        "training_input": "CORD annotated text + normalized boxes + adjacent line text, not raw OCR",
        "test_used_for_training": False,
        "python_version": platform.python_version(),
        "dataset": json.loads((Path(args.data_dir) / "dataset.json").read_text(encoding="utf-8")),
        "split_hashes": {
            s: hashlib.sha256((Path(args.data_dir) / f"{s}.jsonl").read_bytes()).hexdigest()
            for s in ["train", "validation"]
        },
        "limitations": [
            "Indonesian receipts; no invoice or Arabic validation.",
            "Line probabilities do not establish document-level correctness.",
            "OCR errors are evaluated separately; annotated-text metrics exclude OCR errors.",
            "Trusted local joblib artifact only; never load user-uploaded pickle files.",
        ],
    }
    (model_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (reports_dir / "validation.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if args.mlflow_uri:
        import mlflow

        mlflow.set_tracking_uri(args.mlflow_uri)
        mlflow.set_experiment("ReceiptLab CORD line classification")
        with mlflow.start_run():
            mlflow.log_params(
                {
                    "train_documents": 800,
                    "validation_documents": 100,
                    "word_ngram": "1,2",
                    "char_ngram": "2,4",
                    "C": 3.0,
                    "temperature": model.temperature,
                    "confidence_threshold": model.confidence_threshold,
                }
            )
            mlflow.log_metrics(
                {
                    key: metrics[key]
                    for key in (
                        "accuracy",
                        "macro_f1",
                        "target_macro_f1",
                        "nll_before",
                        "nll_after",
                        "training_seconds",
                    )
                }
            )
            mlflow.log_artifact(str(model_dir / "manifest.json"))
    print(
        json.dumps(
            {
                "validation_macro_f1": metrics["macro_f1"],
                "threshold": model.confidence_threshold,
                "temperature": model.temperature,
                "seconds": metrics["training_seconds"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
