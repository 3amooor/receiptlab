from receiptlab.assisted import assisted_extract


class FixedPredictor:
    model_version = "test-trained"
    confidence_threshold = 0.8

    def predict(self, lines):
        return [dict(line) for line in lines]


def line(text, label="item_description", y=0.2, x0=0.1, x1=0.4):
    return {
        "text": text,
        "label": label,
        "model_confidence": 0.95,
        "confidence": 0.99,
        "bbox": [x0, y, x1, y + 0.025],
    }


def test_same_row_anchors_preserve_model_output_and_require_review():
    lines = [
        line("Coffee", y=0.2),
        line("5.00", "item_amount", y=0.2, x0=0.7, x1=0.9),
        line("SUBTOTAL", y=0.5),
        line("5.00", "item_amount", y=0.5, x0=0.7, x1=0.9),
        line("TAX", y=0.6),
        line("0.50", "item_amount", y=0.6, x0=0.7, x1=0.9),
        line("TOTAL", y=0.7),
        line("5.50", "item_amount", y=0.7, x0=0.7, x1=0.9),
    ]
    result = assisted_extract(lines, FixedPredictor())
    assert result["fields"]["subtotal"]["value"] == "5.00"
    assert result["fields"]["tax"]["value"] == "0.50"
    total = result["fields"]["total"]
    assert total["value"] == "5.50" and total["confidence"] == 0.0
    assert total["origin"] == "text_anchor" and total["bbox"] == lines[-1]["bbox"]
    assert result["raw_extraction"]["fields"]["total"]["value"] is None
    assert result["lines"] == lines
    assert [item["description"] for item in result["items"]] == ["Coffee"]
    assert result["confidence"] == 0.0 and result["needs_review"]


def test_inline_indonesian_amounts_are_explicit_anchors():
    result = assisted_extract(
        [
            line("SUB TOTAL: Rp30.000", y=0.3),
            line("PPN (10%) Rp3.000", y=0.4),
            line("TOTAL BAYAR: Rp33.000", y=0.5),
        ],
        FixedPredictor(),
    )
    assert result["fields"]["subtotal"]["value"] == "30000.00"
    assert result["fields"]["tax"]["value"] == "3000.00"
    assert result["fields"]["total"]["value"] == "33000.00"


def test_distinct_anchor_amounts_are_withheld():
    result = assisted_extract(
        [
            line("TOTAL", y=0.5),
            line("12.00", "item_amount", y=0.5, x0=0.5, x1=0.6),
            line("15.00", "item_amount", y=0.5, x0=0.7, x1=0.9),
        ],
        FixedPredictor(),
    )
    assert result["fields"]["total"]["value"] is None
    assert any("Conflicting text-anchor total" in w for w in result["warnings"])


def test_identical_anchor_values_do_not_conflict():
    result = assisted_extract(
        [line("TOTAL 12.00", y=0.5), line("Grand Total:12.00", y=0.6)], FixedPredictor()
    )
    assert result["fields"]["total"]["value"] == "12.00"


def test_learned_disagreement_is_withheld():
    result = assisted_extract(
        [line("TOTAL 15.00", y=0.5), line("12.00", "total", y=0.8)], FixedPredictor()
    )
    assert result["fields"]["total"]["value"] is None
    assert result["raw_extraction"]["fields"]["total"]["value"] == "12.00"


def test_no_amount_borrowed_from_another_row_or_left_column():
    result = assisted_extract(
        [
            line("TOTAL", y=0.5, x0=0.4, x1=0.6),
            line("10.00", "item_amount", y=0.55, x0=0.7, x1=0.9),
            line("2", "item_amount", y=0.5, x0=0.1, x1=0.2),
        ],
        FixedPredictor(),
    )
    assert result["fields"]["total"]["value"] is None


def test_payment_count_percentage_malformed_are_not_amounts():
    for value in ["Cash 100.00", "Total items2", "Jumlah3", "TOTAL:1.23.4"]:
        assert assisted_extract([line(value)], FixedPredictor())["fields"]["total"]["value"] is None
    result = assisted_extract(
        [line("Tax 10%", y=0.4), line("TOTAL 15.00", y=0.6)], FixedPredictor()
    )
    assert result["fields"]["tax"]["value"] is None
    assert result["fields"]["total"]["value"] == "15.00"


def test_metadata_and_time_are_filtered():
    lines = [
        line("Date:2026-10-01", y=0.2),
        line("35.00", "item_amount", y=0.2, x0=0.7, x1=0.9),
        line("Coffee", y=0.4),
        line("5.00", "item_amount", y=0.4, x0=0.7, x1=0.9),
        line("MY COFFEE SHOP", y=0.6),
        line("12:35", "item_amount", y=0.6, x0=0.7, x1=0.9),
    ]
    result = assisted_extract(lines, FixedPredictor())
    assert [item["description"] for item in result["items"]] == ["Coffee"]
    assert result["filtered_item_count"] >= 2
