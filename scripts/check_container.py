"""Check a genuine container's UI, OCR, model readiness and synthetic sample."""

import time

import httpx


def main():
    with httpx.Client(base_url="http://127.0.0.1:8010", timeout=15) as client:
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            try:
                if client.get("/api/health").json()["model_ready"]:
                    break
            except (httpx.HTTPError, ValueError):
                pass
            time.sleep(0.5)
        else:
            raise RuntimeError("Container did not load its trusted model")
        assert "ReceiptLab" in client.get("/").text
        response = client.post("/api/samples/sample-a")
        response.raise_for_status()
        identifier = response.json()["id"]
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            document = client.get("/api/documents/" + identifier).json()
            if document["status"] not in ("queued", "processing"):
                break
            time.sleep(0.5)
        assert document["status"] == "needs_review", document
        assert document["extraction"]["fields"]["total"]["value"] == "12.42", document
        assert len(document["extraction"]["items"]) == 3
        print("Container UI, trained model and actual OCR inference passed.")


if __name__ == "__main__":
    main()
