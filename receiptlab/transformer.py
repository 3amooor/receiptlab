"""Optional image-aware LayoutLMv3 inference; imports GPU dependencies lazily."""

import json
from pathlib import Path


def normalized_box(box):
    return [round(max(0.0, min(1.0, float(v))) * 1000) for v in box]


class TransformerPredictor:
    """A locally fine-tuned classifier with a fixed label map and model revision."""

    def __init__(self, model_dir: Path, device: str | None = None):
        import torch
        from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = LayoutLMv3Processor.from_pretrained(model_dir, apply_ocr=False)
        self.model = LayoutLMv3ForTokenClassification.from_pretrained(model_dir).to(self.device)
        self.model.eval()
        self.version = "layoutlmv3-receiptlab-v1"
        self.threshold = 0.95
        manifest = Path(model_dir) / "receiptlab-manifest.json"
        if manifest.exists():
            self.threshold = float(
                json.loads(manifest.read_text(encoding="utf-8"))["review_policy"]["threshold"]
            )

    @property
    def model_version(self):
        return self.version

    @property
    def confidence_threshold(self):
        return self.threshold

    def chunks(self, lines):
        """Reserve space for tokenizer boundaries and never silently discard lines."""
        current, cost = [], 0
        for i, line in enumerate(lines):
            # Explicit options reset retained fast-tokenizer padding/truncation state.
            tokens = self.processor.tokenizer(
                [line["text"]],
                boxes=[normalized_box(line["bbox"])],
                padding=False,
                truncation=False,
                add_special_tokens=False,
            )["input_ids"]
            count = len(tokens) + 2
            if current and cost + count > 480:
                yield current
                current, cost = [], 0
            current.append(i)
            cost += count
        if current:
            yield current

    def encode(self, image, lines, indices, labels=None):
        kwargs = {"word_labels": [labels[i] for i in indices]} if labels is not None else {}
        return self.processor(
            image.convert("RGB"),
            text=[lines[i]["text"] for i in indices],
            boxes=[normalized_box(lines[i]["bbox"]) for i in indices],
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt",
            **kwargs,
        )

    def predict_image(self, lines, image):
        predictions = [dict(line, label="other", model_confidence=0.0) for line in lines]
        with self.torch.inference_mode():
            for indices in self.chunks(lines):
                encoding = self.encode(image, lines, indices)
                word_ids = encoding.word_ids(batch_index=0)
                logits = self.model(**{k: v.to(self.device) for k, v in encoding.items()}).logits
                probabilities = logits.softmax(dim=-1)[0].cpu()
                seen = set()
                for token, word in enumerate(word_ids):
                    if word is None or word in seen:
                        continue
                    seen.add(word)
                    confidence, predicted = probabilities[token].max(dim=-1)
                    original = indices[word]
                    predictions[original]["label"] = self.model.config.id2label[int(predicted)]
                    predictions[original]["model_confidence"] = float(confidence)
                # Unseen lines retain confidence zero and must pass through human review.
        return predictions

    def bind(self, image):
        return BoundTransformer(self, image)


class BoundTransformer:
    """Adapt image-aware inference to the shared line extraction interface."""

    def __init__(self, predictor, image):
        self.predictor = predictor
        self.image = image
        self.version = predictor.version
        self.threshold = predictor.threshold
        self.model_version = predictor.version
        self.confidence_threshold = predictor.threshold

    def predict(self, lines):
        return self.predictor.predict_image(lines, self.image)
