import pytest

from receiptlab.extraction import extract, parse_amount


class FixedPredictor:
    model_version = "test-v1"
    confidence_threshold = 0.8

    def predict(self, lines):
        return [dict(line) for line in lines]


def line(text, label, confidence=0.95, box=None):
    return {
        "text": text,
        "label": label,
        "model_confidence": confidence,
        "confidence": 0.98,
        "bbox": box or [0.1, 0.1, 0.8, 0.14],
    }


@pytest.mark.parametrize(
    "text,expected",
    [
        ("$1,234.56", "1234.56"),
        ("Rp30.000", "30000.00"),
        ("12,50", "12.50"),
        ("1.234.567", "1234567.00"),
        ("(10.50)", "-10.50"),
        ("2 x10.00", "10.00"),
        ("Total0", "0.00"),
        ("not a number", None),
        ("1.23.4", None),
    ],
)
def test_amount_parser_handles_cord_and_decimal_notation(text, expected):
    assert parse_amount(text) == expected


def test_low_ocr_confidence_requires_review_even_with_confident_classifier():
    entry = line("15.00", "total")
    entry["confidence"] = 0.4
    result = extract([entry], FixedPredictor())
    assert result["fields"]["total"]["value"] == "15.00"
    assert result["fields"]["total"]["confidence"] == 0.4
    assert result["needs_review"]


def test_conflicting_totals_are_explicit_and_reviewed():
    result = extract([line("10.00", "total"), line("12.00", "total", 0.9)], FixedPredictor())
    assert any("Conflicting total" in warning for warning in result["warnings"])
    assert result["needs_review"]


def test_reconciled_receipt_pairs_items_and_requires_no_review():
    entries = [
        line("Coffee", "item_description", box=[0.1, 0.2, 0.4, 0.24]),
        line("2", "item_quantity", box=[0.02, 0.2, 0.06, 0.24]),
        line("5.00", "item_unit_price", box=[0.5, 0.2, 0.6, 0.24]),
        line("10.00", "item_amount", box=[0.75, 0.2, 0.9, 0.24]),
        line("10.00", "subtotal", box=[0.75, 0.5, 0.9, 0.54]),
        line("1.00", "tax", box=[0.75, 0.6, 0.9, 0.64]),
        line("11.00", "total", box=[0.75, 0.7, 0.9, 0.74]),
    ]
    result = extract(entries, FixedPredictor())
    assert result["items"][0]["description"] == "Coffee"
    assert result["items"][0]["quantity"] == "2"
    assert result["items"][0]["amount"] == "10.00"
    assert not result["needs_review"]


def test_receipt_arithmetic_mismatch_forces_review():
    result = extract(
        [line("10.00", "subtotal"), line("1.00", "tax"), line("15.00", "total")], FixedPredictor()
    )
    assert any("reconcile" in warning for warning in result["warnings"])
    assert result["needs_review"]


def test_missing_total_and_empty_input_fail_closed():
    result = extract([], FixedPredictor())
    assert result["confidence"] == 0
    assert result["fields"]["total"]["value"] is None
    assert result["needs_review"]


def test_description_cannot_steal_a_price_from_distant_row():
    result = extract(
        [
            line("Coffee", "item_description", box=[0.1, 0.1, 0.4, 0.14]),
            line("10.00", "item_amount", box=[0.7, 0.8, 0.9, 0.84]),
        ],
        FixedPredictor(),
    )
    assert result["items"][0]["amount"] is None
    assert result["needs_review"]
