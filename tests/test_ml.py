import hashlib
import json

import joblib
import numpy as np
import pytest

from receiptlab.ml import LineClassifier, calibrated_probabilities, get_predictor
from training.train import choose_temperature, choose_threshold


def test_temperature_scaling_normalizes_and_softens_overconfidence():
    raw = np.array([[0.99, 0.005, 0.005], [0.01, 0.9, 0.09]])
    adjusted = calibrated_probabilities(raw, 2.0)
    assert np.allclose(adjusted.sum(axis=1), 1)
    assert adjusted[0, 0] < raw[0, 0]
    assert np.array_equal(adjusted.argmax(axis=1), raw.argmax(axis=1))


def test_validation_temperature_reduces_overconfident_wrong_prediction_loss():
    raw = np.array([[0.99, 0.01]] * 4)
    truth = np.array(["a", "a", "b", "b"])
    temperature = choose_temperature(raw, truth, np.array(["a", "b"]))
    assert temperature > 1
    adjusted = calibrated_probabilities(raw, temperature)
    raw_loss = -np.log([0.99, 0.99, 0.01, 0.01]).mean()
    loss = -np.log([adjusted[0, 0], adjusted[1, 0], adjusted[2, 1], adjusted[3, 1]]).mean()
    assert loss < raw_loss


def test_threshold_rejects_high_coverage_inaccurate_predictions():
    probabilities = np.array([[0.05, 0.95]] * 40 + [[0.3, 0.7]] * 60)
    truth = np.array(["total"] * 40 + ["other"] * 60)
    threshold, details = choose_threshold(probabilities, truth, np.array(["other", "total"]))
    assert threshold > 0.7
    assert details["selected"]["accepted_lines"] == 40
    assert details["selected"]["precision"] == 1.0


def training_records():
    return [
        {
            "id": str(i),
            "lines": [
                {
                    "text": f"Coffee Arabica{i}",
                    "label": "item_description",
                    "bbox": [0.1, 0.2, 0.4, 0.24],
                },
                {"text": f"{10 + i}.00", "label": "item_amount", "bbox": [0.75, 0.2, 0.9, 0.24]},
                {"text": "TOTAL", "label": "other", "bbox": [0.1, 0.8, 0.4, 0.84]},
                {"text": f"{10 + i}.00", "label": "total", "bbox": [0.75, 0.8, 0.9, 0.84]},
            ],
        }
        for i in range(12)
    ]


def test_classifier_distinguishes_identical_amounts_by_geometry_and_context():
    model = LineClassifier().fit(training_records())
    result = model.predict(training_records()[0]["lines"])
    assert result[1]["label"] == "item_amount"
    assert result[3]["label"] == "total"
    assert model.predict([]) == []
    assert all(0 <= entry["model_confidence"] <= 1 for entry in result)


def test_artifact_loader_checks_hash_before_deserialization(tmp_path):
    model = LineClassifier().fit(training_records())
    artifact = tmp_path / "line-classifier.joblib"
    joblib.dump(model, artifact)
    manifest = {
        "artifact": artifact.name,
        "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "model_version": "test-model",
        "confidence_threshold": 0.85,
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    assert get_predictor(tmp_path).model_version == "test-model"
    artifact.write_bytes(b"untrusted corrupt artifact")
    with pytest.raises(ValueError, match="checksum"):
        get_predictor(tmp_path)


def test_model_missing_is_explicit(tmp_path):
    with pytest.raises(FileNotFoundError, match="manifest"):
        get_predictor(tmp_path)
