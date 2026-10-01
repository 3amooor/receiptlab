"""API integration tests with real image decoding and isolated durable storage."""

import csv
import io
import time
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from receiptlab.api import create_app
from receiptlab.store import Store

EXTRACTION = {
    "fields": {
        "subtotal": {"value": "10.00", "confidence": 0.95, "bbox": [0.1, 0.7, 0.9, 0.8]},
        "tax": {"value": "1.00", "confidence": 0.95, "bbox": None},
        "discount": {"value": None, "confidence": 0.0, "bbox": None},
        "total": {"value": "11.00", "confidence": 0.96, "bbox": [0.1, 0.8, 0.9, 0.9]},
    },
    "items": [
        {
            "description": "Coffee",
            "quantity": "1",
            "unit_price": "10.00",
            "amount": "10.00",
            "confidence": 0.9,
        }
    ],
    "lines": [
        {
            "text": "TOTAL 11.00",
            "bbox": [0.1, 0.8, 0.9, 0.9],
            "confidence": 0.95,
            "label": "total",
            "model_confidence": 0.96,
        }
    ],
    "warnings": [],
    "model_version": "test-trained-artifact",
    "confidence": 0.95,
    "needs_review": False,
}


def png(size=(100, 180), color="white"):
    stream = io.BytesIO()
    Image.new("RGB", size, color).save(stream, format="PNG")
    return stream.getvalue()


def fake_processor(image):
    assert image.mode == "RGB"
    return deepcopy(EXTRACTION)


@pytest.fixture
def client(tmp_path):
    app = create_app(tmp_path / "data", tmp_path / "models", processor=fake_processor)
    with TestClient(app) as test_client:
        yield test_client


def upload(client, filename="receipt.png", data=None):
    response = client.post("/api/documents", files={"file": (filename, data or png(), "image/png")})
    assert response.status_code == 202, response.text
    return response.json()


def finished(client, document_id):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        response = client.get(f"/api/documents/{document_id}")
        assert response.status_code == 200
        document = response.json()
        if document["status"] not in ("queued", "processing"):
            return document
        time.sleep(0.02)
    pytest.fail("Durable worker did not complete the document")


def review_body(document, approve=False):
    return {
        "revision": document["revision"],
        "fields": {
            name: field["value"] for name, field in document["extraction"]["fields"].items()
        },
        "items": [
            {key: item.get(key) for key in ("description", "quantity", "unit_price", "amount")}
            for item in document["extraction"]["items"]
        ],
        "approve": approve,
    }


def test_upload_worker_image_and_idempotency(client):
    first = upload(client, "../../private/receipt.png")
    duplicate = upload(client, "retry.png")
    assert first["id"] == duplicate["id"]
    document = finished(client, first["id"])
    assert document["status"] == "ready"
    assert document["filename"] == "receipt.png"
    assert document["page_count"] == 1
    assert document["extraction"]["fields"]["total"]["value"] == "11.00"
    assert len(client.get("/api/documents").json()) == 1
    assert client.get(document["image_url"]).content.startswith(b"\x89PNG")
    assert client.get("/api/health").json()["model_ready"] is True
    assert len(list(client.app.state.store.images.glob("*.png"))) == 1


def test_upload_limits_and_mime_spoofing(client):
    assert (
        client.post(
            "/api/documents", files={"file": ("fake.png", b"<svg/>", "image/png")}
        ).status_code
        == 415
    )
    assert (
        client.post(
            "/api/documents", files={"file": ("broken.png", b"\x89PNG\r\n\x1a\n", "image/png")}
        ).status_code
        == 422
    )
    assert (
        client.post("/api/documents", files={"file": ("empty.png", b"", "image/png")}).status_code
        == 422
    )
    assert (
        client.post(
            "/api/documents",
            files={"file": ("large.png", b"a" * (10 * 1024 * 1024 + 1), "image/png")},
        ).status_code
        == 413
    )
    assert (
        client.post(
            "/api/documents", files={"file": ("wide.png", png((12001, 1)), "image/png")}
        ).status_code
        == 413
    )


def test_single_page_pdf_and_multi_page_rejection(client):
    image = Image.new("RGB", (100, 180), "white")
    single = io.BytesIO()
    image.save(single, format="PDF")
    document = finished(client, upload(client, "receipt.pdf", single.getvalue())["id"])
    assert document["page_count"] == 1
    multi = io.BytesIO()
    image.save(multi, format="PDF", save_all=True, append_images=[image])
    response = client.post(
        "/api/documents", files={"file": ("two.pdf", multi.getvalue(), "application/pdf")}
    )
    assert response.status_code == 422
    assert "single-page" in response.text


def test_revision_checks_approval_and_original_audit(client):
    document = finished(client, upload(client)["id"])
    body = review_body(document)
    body["fields"]["total"] = "12.00"
    body["approve"] = True
    url = f"/api/documents/{document['id']}/review"
    assert client.patch(url, json=body).status_code == 422
    body["approve"] = False
    corrected = client.patch(url, json=body)
    assert corrected.status_code == 200
    corrected = corrected.json()
    assert corrected["status"] == "needs_review"
    assert corrected["extraction"]["warnings"]
    assert client.patch(url, json=body).status_code == 409
    body = review_body(corrected, True)
    body["fields"]["total"] = "11.00"
    approved = client.patch(url, json=body).json()
    assert approved["status"] == "approved"
    events = client.get(f"/api/documents/{document['id']}/audit").json()
    assert [event["event"] for event in events] == [
        "uploaded",
        "extracted",
        "corrected",
        "approved",
    ]
    assert events[1]["after"]["fields"]["total"]["value"] == "11.00"
    assert events[2]["after"]["fields"]["total"]["value"] == "12.00"


@pytest.mark.parametrize("invalid", ["NaN", "Infinity", "-1", "1e4", "0.001", "100000000.00", 11.0])
def test_strict_money_review(client, invalid):
    document = finished(client, upload(client)["id"])
    body = review_body(document)
    body["fields"]["total"] = invalid
    assert client.patch(f"/api/documents/{document['id']}/review", json=body).status_code == 422


def test_missing_total_and_line_item_mismatch_cannot_be_approved(client):
    document = finished(client, upload(client)["id"])
    body = review_body(document, True)
    body["fields"]["total"] = None
    url = f"/api/documents/{document['id']}/review"
    assert client.patch(url, json=body).status_code == 422
    body["fields"]["total"] = "11.00"
    body["items"][0]["quantity"] = "2"
    assert client.patch(url, json=body).status_code == 422


def test_csv_formula_escape_json_and_unknown_format(client):
    document = finished(client, upload(client)["id"])
    body = review_body(document)
    body["items"][0]["description"] = "=SUM(1,2)"
    assert client.patch(f"/api/documents/{document['id']}/review", json=body).status_code == 200
    path = f"/api/documents/{document['id']}/export"
    exported = client.get(path + "?format=json").json()
    assert exported["extraction"]["items"][0]["description"] == "=SUM(1,2)"
    rows = list(csv.reader(io.StringIO(client.get(path + "?format=csv").text)))
    assert rows[-1][1] == "'=SUM(1,2)"
    assert client.get(path + "?format=html").status_code == 422


def test_restart_recovers_interrupted_job(tmp_path):
    directory = tmp_path / "persistent"
    first_app = create_app(directory, processor=fake_processor, worker_enabled=False)
    with TestClient(first_app) as first:
        document = upload(first)
        claimed = first.app.state.store.claim()
        assert claimed["status"] == "processing"
    second_app = create_app(directory, processor=fake_processor)
    with TestClient(second_app) as second:
        recovered = finished(second, document["id"])
        assert recovered["status"] == "ready"
        assert recovered["revision"] == 2
        assert len(second.get("/api/documents").json()) == 1


def test_job_claim_is_atomic(tmp_path):
    store = Store(tmp_path / "claim")
    store.create("00000000-0000-0000-0000-000000000001", "sha", "receipt.png", "upload", 100, 100)
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: store.claim(), range(2)))
    assert sum(result is not None for result in results) == 1


def test_failure_is_persistent_and_safe(tmp_path):
    def fail(image):
        raise RuntimeError("secret receipt data should never enter error responses")

    with TestClient(create_app(tmp_path, processor=fail)) as client:
        document = finished(client, upload(client)["id"])
        assert document["status"] == "failed"
        assert "secret" not in document["error"]
        assert client.get(f"/api/documents/{document['id']}/export").status_code == 409


def test_local_origin_guard_and_invalid_document_id(client):
    assert (
        client.post(
            "/api/documents",
            headers={"Origin": "https://evil.example"},
            files={"file": ("receipt.png", png(), "image/png")},
        ).status_code
        == 403
    )
    assert client.get("/api/documents/not-a-uuid/image").status_code == 422
    assert client.get("/api/documents/00000000-0000-0000-0000-000000000001").status_code == 404


def test_request_size_limit_before_multipart_parsing(client):
    response = client.post(
        "/api/documents",
        content=b"a" * (10 * 1024 * 1024 + 65537),
        headers={"Content-Type": "application/octet-stream"},
    )
    assert response.status_code == 413
    assert client.get("/api/documents").json() == []


def test_review_preserves_unchanged_item_confidence(client):
    doc = finished(client, upload(client)["id"])
    body = review_body(doc)
    saved = client.patch(f"/api/documents/{doc['id']}/review", json=body).json()
    assert saved["extraction"]["items"][0]["confidence"] == 0.9
    body = review_body(saved)
    body["items"][0]["description"] = "Corrected coffee"
    changed = client.patch(f"/api/documents/{doc['id']}/review", json=body).json()
    assert changed["extraction"]["items"][0]["confidence"] == 0.0
    assert changed["extraction"]["items"][0]["reviewed"]


def test_missing_model_cannot_report_ready(tmp_path):
    with TestClient(create_app(tmp_path / "data", models_dir=tmp_path / "absent")) as client:
        doc = finished(client, upload(client)["id"])
        assert not client.get("/api/health").json()["model_ready"]
        assert doc["status"] == "failed" and doc["extraction"] is None


def test_unknown_sample_is_not_a_filesystem_path(client):
    assert client.post("/api/samples/unknown").status_code == 404
