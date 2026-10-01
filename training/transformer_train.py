"""Train a real multimodal model with official document-level held-out splits."""

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image

from receiptlab.transformer import TransformerPredictor


def path_for(record, data_dir):
    path = Path(record["image_path"])
    return path if path.is_absolute() else data_dir / path


def labelled_windows(records, data_dir, predictor, label2id):
    for record in records:
        with Image.open(path_for(record, data_dir)) as source:
            image = source.convert("RGB")
        lines = record["lines"]
        labels = [label2id.get(line["label"], label2id["other"]) for line in lines]
        for indices in predictor.chunks(lines):
            encoding = predictor.encode(image, lines, indices, labels)
            yield {k: v for k, v in encoding.items()}, record["id"]


def evaluate(records, data_dir, predictor, label2id):
    from sklearn.metrics import classification_report, f1_score

    actual, predicted, probabilities, timings = [], [], [], []
    for record in records:
        with Image.open(path_for(record, data_dir)) as source:
            image = source.convert("RGB")
        start = time.perf_counter()
        result = predictor.predict_image(record["lines"], image)
        timings.append((time.perf_counter() - start) * 1000)
        for line, answer in zip(record["lines"], result, strict=True):
            actual.append(line["label"])
            predicted.append(answer["label"])
            probabilities.append(answer["model_confidence"])
    labels = sorted(label2id)
    metrics = {
        "micro_f1": float(f1_score(actual, predicted, average="micro")),
        "macro_f1": float(
            f1_score(actual, predicted, labels=labels, average="macro", zero_division=0)
        ),
        "per_class": classification_report(
            actual, predicted, labels=labels, output_dict=True, zero_division=0
        ),
        "documents": len(records),
        "lines": len(actual),
        "latency_ms_p50": float(np.percentile(timings, 50)),
        "latency_ms_p95": float(np.percentile(timings, 95)),
        "unrepresented_lines": sum(c == 0 for c in probabilities),
    }
    return metrics, np.asarray(actual) == np.asarray(predicted), np.asarray(probabilities)


def select_threshold(correct, confidence):
    # Validation only: select highest coverage with at least 98% line precision.
    choices = []
    for threshold in np.arange(0.50, 1.0, 0.01):
        accepted = confidence >= threshold
        count = int(accepted.sum())
        precision = float(correct[accepted].mean()) if count else 0.0
        if count >= 30 and precision >= 0.98:
            choices.append((count, float(threshold), precision))
    if not choices:
        return {"threshold": 1.0, "validation_precision": None, "validation_coverage": 0.0}
    count, threshold, precision = max(choices)
    return {
        "threshold": round(threshold, 2),
        "validation_precision": precision,
        "validation_coverage": count / len(correct),
    }


def main():
    import torch
    from huggingface_hub import HfApi
    from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor

    from training.data import load_records

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/cord"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/layoutlmv3"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1 or args.accumulation < 1:
        parser.error("epochs and accumulation must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    train = load_records(args.data_dir, "train")
    validation = load_records(args.data_dir, "validation")
    test = load_records(args.data_dir, "test")
    for a, b in [(train, validation), (train, test), (validation, test)]:
        if {r["id"] for r in a} & {r["id"] for r in b}:
            raise ValueError("Document splits overlap")
    label_names = sorted(
        {line["label"] for record in train for line in record["lines"]} | {"other"}
    )
    label2id = {name: i for i, name in enumerate(label_names)}
    base = "microsoft/layoutlmv3-base"
    revision = HfApi().model_info(base, token=False).sha
    cache = str(args.data_dir.parent / "huggingface")
    processor = LayoutLMv3Processor.from_pretrained(
        base, revision=revision, apply_ocr=False, cache_dir=cache, token=False
    )
    model = LayoutLMv3ForTokenClassification.from_pretrained(
        base,
        revision=revision,
        cache_dir=cache,
        token=False,
        id2label={i: name for name, i in label2id.items()},
        label2id=label2id,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    if model.supports_gradient_checkpointing:
        model.gradient_checkpointing_enable()
    predictor = TransformerPredictor.__new__(TransformerPredictor)
    predictor.torch, predictor.device, predictor.processor, predictor.model = (
        torch,
        device,
        processor,
        model,
    )
    predictor.version, predictor.threshold = "layoutlmv3-receiptlab-v1", 0.95
    expected_windows = 0
    for record in train:
        chunks = list(predictor.chunks(record["lines"]))
        if [index for chunk in chunks for index in chunk] != list(range(len(record["lines"]))):
            raise RuntimeError("Training chunks must cover each line exactly once")
        expected_windows += len(chunks)
    print(
        json.dumps(
            {"training_documents": len(train), "expected_windows_per_epoch": expected_windows}
        ),
        flush=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    best_f1, history = -1.0, []
    for epoch in range(args.epochs):
        model.train()
        random.shuffle(train)
        optimizer.zero_grad(set_to_none=True)
        total_loss, steps = 0.0, 0
        seen_documents = set()
        for batch, _document in labelled_windows(train, args.data_dir, predictor, label2id):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda"):
                loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            total_loss += float(loss.detach())
            scaler.scale(loss / args.accumulation).backward()
            steps += 1
            seen_documents.add(_document)
            if steps > expected_windows:
                raise RuntimeError(
                    f"Training yielded {steps} windows from {len(seen_documents)} unique documents; last={_document}"
                )
            if steps % args.accumulation == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            if steps % 100 == 0:
                print(
                    f"epoch={epoch + 1} window={steps} docs={len(seen_documents)} mean_loss={total_loss / steps:.4f}",
                    flush=True,
                )
        if steps % args.accumulation:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        model.eval()
        if steps != expected_windows:
            raise RuntimeError("Training window count did not match the complete split")
        metrics, _, _ = evaluate(validation, args.data_dir, predictor, label2id)
        history.append(
            {
                "epoch": epoch + 1,
                "loss": total_loss / steps,
                "validation_macro_f1": metrics["macro_f1"],
            }
        )
        print(json.dumps(history[-1]), flush=True)
        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            model.save_pretrained(args.output_dir)
            processor.save_pretrained(args.output_dir)
    del optimizer, model
    if device == "cuda":
        torch.cuda.empty_cache()
    predictor = TransformerPredictor(args.output_dir, device)
    valid_metrics, correct, confidence = evaluate(validation, args.data_dir, predictor, label2id)
    policy = select_threshold(correct, confidence)
    test_metrics, test_correct, test_confidence = evaluate(test, args.data_dir, predictor, label2id)
    accepted = test_confidence >= policy["threshold"]
    test_metrics["accepted_line_precision"] = (
        float(test_correct[accepted].mean()) if accepted.any() else None
    )
    test_metrics["accepted_line_coverage"] = float(accepted.mean())
    report = {
        "model": predictor.version,
        "base_model": base,
        "base_revision": revision,
        "license": "CC-BY-NC-SA-4.0",
        "evaluation_mode": "CORD supplied text and bounding boxes plus receipt image",
        "training_documents": len(train),
        "validation_documents": len(validation),
        "test_documents": len(test),
        "label_count": len(label2id),
        "seed": args.seed,
        "epochs": args.epochs,
        "selected_by": "validation macro F1",
        "history": history,
        "validation": valid_metrics,
        "review_policy": policy,
        "test": test_metrics,
        "duration_seconds": time.perf_counter() - start,
        "device": device,
        "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
        "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / 1024**2
        if device == "cuda"
        else None,
        "split_hashes": {
            s: hashlib.sha256((args.data_dir / f"{s}.jsonl").read_bytes()).hexdigest()
            for s in ["train", "validation", "test"]
        },
        "limitations": [
            "Line-level classification, not full-document exact-match accuracy",
            "Not an end-to-end OCR metric",
            "Indonesian receipt benchmark, not arbitrary English or Arabic invoices",
        ],
    }
    report_dir = Path("reports")
    report_dir.mkdir(exist_ok=True)
    (report_dir / "transformer-evaluation.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (args.output_dir / "receiptlab-manifest.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "test_micro_f1": test_metrics["micro_f1"],
                "test_macro_f1": test_metrics["macro_f1"],
                "device": device,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
