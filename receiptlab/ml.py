from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

_NUMERIC = re.compile(r"\d")


def feature_rows(lines):
    texts, numeric = [], []
    count = max(1, len(lines))
    for index, line in enumerate(lines):
        text = str(line["text"]).strip()
        before = str(lines[index - 1]["text"]) if index else ""
        after = str(lines[index + 1]["text"]) if index + 1 < len(lines) else ""
        texts.append(f"self {text} previous {before} following {after}")
        box = line.get("bbox", [0, 0, 1, 1])
        x0, y0, x1, y1 = [float(v) for v in box]
        length = max(1, len(text))
        numeric.append(
            [
                x0,
                y0,
                x1,
                y1,
                x1 - x0,
                y1 - y0,
                (x0 + x1) / 2,
                (y0 + y1) / 2,
                index / count,
                len(text) / 100,
                sum(c.isdigit() for c in text) / length,
                sum(c.isalpha() for c in text) / length,
                float(bool(_NUMERIC.search(text))),
                float(bool(re.search(r"[.,]\d{2,3}", text))),
                float("rp" in text.lower() or "$" in text),
                float(text.isupper()),
            ]
        )
    return texts, np.asarray(numeric, dtype=np.float64)


def calibrated_probabilities(probabilities, temperature):
    logits = np.log(np.clip(probabilities, 1e-12, 1.0)) / float(temperature)
    logits -= logits.max(axis=1, keepdims=True)
    values = np.exp(logits)
    return values / values.sum(axis=1, keepdims=True)


class LineClassifier:
    def __init__(self):
        self.word = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_features=18000,
            sublinear_tf=True,
            strip_accents="unicode",
        )
        self.char = TfidfVectorizer(
            analyzer="char", ngram_range=(2, 4), min_df=2, max_features=25000, sublinear_tf=True
        )
        self.classifier = LogisticRegression(
            C=3.0, class_weight="balanced", max_iter=700, random_state=42
        )
        self.temperature = 1.0
        self.confidence_threshold = 0.8
        self.model_version = "cord-line-v1"

    def fit(self, records):
        texts, numbers, labels = [], [], []
        for record in records:
            text, number = feature_rows(record["lines"])
            texts.extend(text)
            numbers.extend(number)
            labels.extend(line["label"] for line in record["lines"])
        matrix = hstack(
            [self.word.fit_transform(texts), self.char.fit_transform(texts), csr_matrix(numbers)],
            format="csr",
        )
        self.classifier.fit(matrix, labels)
        return self

    def probabilities(self, lines):
        if not lines:
            return np.empty((0, len(self.classifier.classes_)))
        texts, numbers = feature_rows(lines)
        matrix = hstack(
            [self.word.transform(texts), self.char.transform(texts), csr_matrix(numbers)],
            format="csr",
        )
        return calibrated_probabilities(self.classifier.predict_proba(matrix), self.temperature)

    def predict(self, lines):
        probabilities = self.probabilities(lines)
        result = []
        for line, row in zip(lines, probabilities, strict=True):
            index = int(np.argmax(row))
            result.append(
                {
                    **line,
                    "label": str(self.classifier.classes_[index]),
                    "model_confidence": float(row[index]),
                }
            )
        return result


class RulesPredictor:
    model_version = "rules-v1"
    confidence_threshold = 0.8

    def predict(self, lines):
        output = []
        key_patterns = [
            ("subtotal", r"sub\s*total"),
            ("tax", r"tax|pajak|ppn"),
            ("discount", r"disc|discount|diskon|potongan"),
            ("total", r"grand\s*total|(?<!sub)\btotal\b|jumlah"),
        ]
        for line in lines:
            text = line["text"].lower()
            box = line.get("bbox", [0, 0, 1, 1])
            label, confidence = "other", 0.45
            if _NUMERIC.search(text):
                center = (box[1] + box[3]) / 2
                context = [text]
                for candidate in lines:
                    cb = candidate.get("bbox", [0, 0, 1, 1])
                    if abs((cb[1] + cb[3]) / 2 - center) <= max(0.012, (box[3] - box[1]) * 0.65):
                        context.append(candidate["text"].lower())
                joined = " ".join(context)
                for role, pattern in key_patterns:
                    if re.search(pattern, joined):
                        label, confidence = role, 0.8
                        break
                if label == "other" and 0.1 < center < 0.8:
                    if box[0] < 0.2 and re.fullmatch(r"\d{1,2}(?:[xX])?", text.strip()):
                        label, confidence = "item_quantity", 0.6
                    elif box[0] > 0.48:
                        label, confidence = "item_amount", 0.6
            elif sum(c.isalpha() for c in text) > 2 and 0.1 < (box[1] + box[3]) / 2 < 0.8:
                if not re.search(
                    r"total|tax|pajak|ppn|discount|cash|change|bayar|subtotal|tel|date|time", text
                ):
                    label, confidence = "item_description", 0.55
            output.append({**line, "label": label, "model_confidence": confidence})
        return output


def get_predictor(model_dir):
    directory = Path(model_dir)
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            "Trained model manifest missing. Run python -m training.train first."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact = directory / manifest.get("artifact", "line-classifier.joblib")
    if not artifact.exists():
        raise FileNotFoundError(f"Trained artifact missing: {artifact}")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    if digest != manifest.get("sha256"):
        raise ValueError("Model artifact checksum does not match its manifest.")
    predictor = joblib.load(artifact)
    if not isinstance(predictor, LineClassifier):
        raise TypeError("Unsupported trusted model artifact.")
    predictor.model_version = manifest["model_version"]
    predictor.confidence_threshold = float(manifest["confidence_threshold"])
    return predictor
