"""Verify chunk accounting without downloading torch or model weights in CI."""

from types import SimpleNamespace

from receiptlab.transformer import TransformerPredictor


class StatefulTokenizer:
    def __init__(self):
        self.padded = True

    def __call__(self, text, *, boxes, padding, truncation, add_special_tokens):
        assert padding is False and truncation is False and add_special_tokens is False
        assert len(boxes) == len(text) == 1
        self.padded = padding
        return {"input_ids": list(range(len(text[0].split())))}


def test_short_lines_do_not_inherit_padding_and_cover_each_line_once():
    predictor = TransformerPredictor.__new__(TransformerPredictor)
    tokenizer = StatefulTokenizer()
    predictor.processor = SimpleNamespace(tokenizer=tokenizer)
    lines = [{"text": "Coffee milk", "bbox": [0.1, 0.1, 0.9, 0.2]} for _ in range(30)]
    assert list(predictor.chunks(lines)) == [list(range(30))]
    tokenizer.padded = True
    assert list(predictor.chunks(lines)) == [list(range(30))]


def test_long_receipts_partition_without_overlap_or_loss():
    predictor = TransformerPredictor.__new__(TransformerPredictor)
    predictor.processor = SimpleNamespace(tokenizer=StatefulTokenizer())
    lines = [{"text": "word " * 30, "bbox": [0.1, 0.1, 0.9, 0.2]} for _ in range(100)]
    chunks = list(predictor.chunks(lines))
    assert len(chunks) > 1
    assert all(chunks)
    assert [index for chunk in chunks for index in chunk] == list(range(100))
