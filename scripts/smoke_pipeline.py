"""Exercise genuine OCR/model inference through the HTTP API on original fixtures."""

import argparse
import json
import os
import time
from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient

from receiptlab.api import create_app

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("classic", "transformer"), default="classic")
    args = parser.parse_args()
    os.environ["RECEIPTLAB_MODEL_BACKEND"] = args.backend
    results = []
    workspace = ROOT / "data" / ("smoke-" + args.backend + "-" + uuid4().hex)
    app = create_app(workspace, ROOT / "models")
    with TestClient(app) as client:
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            if client.get("/api/health").json()["model_ready"]:
                break
            time.sleep(0.1)
        else:
            raise RuntimeError("Model did not become ready for fresh inference")
        for name in ("sample-a", "sample-b", "sample-c"):
            source = ROOT / "frontend/public/samples" / (name + ".png")
            response = client.post(
                "/api/documents", files={"file": (source.name, source.read_bytes(), "image/png")}
            )
            assert response.status_code == 202, response.text
            identifier = response.json()["id"]
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                doc = client.get("/api/documents/" + identifier).json()
                if doc["status"] not in ("queued", "processing"):
                    break
                time.sleep(0.1)
            assert doc["extraction"] and doc["status"] == "needs_review", doc
            extraction = doc["extraction"]
            observed = {key: field["value"] for key, field in extraction["fields"].items()}
            assert client.get(doc["image_url"]).status_code == 200
            assert (
                client.get("/api/documents/" + identifier + "/export?format=csv").status_code == 200
            )
            results.append(
                {
                    "sample": name,
                    "fields": observed,
                    "items": len(extraction["items"]),
                    "model_version": extraction["model_version"],
                    "requires_review": extraction["needs_review"],
                }
            )
        assert client.get("/api/health").json()["model_ready"]
    report = {
        "backend": args.backend,
        "samples": results,
        "input": "Original synthetic images; real OCR and frozen model; actual API worker",
        "checks": "Upload, durable inference, source image, needs-review state, CSV export, model readiness",
    }
    (ROOT / "reports" / ("smoke-" + args.backend + ".json")).write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
