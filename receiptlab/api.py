"""Local single-user ReceiptLab API; never expose this server publicly."""

import asyncio
import csv
import hashlib
import io
import json
import math
import os
import warnings
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated
from uuid import UUID, uuid4

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError

from .schemas import MONEY_FIELDS, ReviewRequest, financial_warnings
from .store import MissingDocument, RevisionConflict, Store
from .worker import Worker

ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 20_000_000
MAX_DIMENSION = 12_000
SAMPLE_NAMES = {"sample-a", "sample-b", "sample-c"}
ALLOWED_ORIGINS = {
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:8010",
    "http://127.0.0.1:8010",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
}


def check_dimensions(width: int, height: int):
    if (
        width < 1
        or height < 1
        or width > MAX_DIMENSION
        or height > MAX_DIMENSION
        or width * height > MAX_PIXELS
    ):
        raise HTTPException(413, "Image dimensions exceed the 20 megapixel receipt limit.")


def decode_receipt(data: bytes) -> Image.Image:
    if not data:
        raise HTTPException(422, "The uploaded file is empty.")
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "Receipts must be at most 10 MB.")
    try:
        if data.startswith(b"%PDF-"):
            import pypdfium2 as pdfium

            document = pdfium.PdfDocument(data)
            try:
                if len(document) != 1:
                    raise HTTPException(422, "Upload a single-page receipt PDF.")
                page = document[0]
                try:
                    width, height = page.get_size()
                    if not math.isfinite(width) or not math.isfinite(height):
                        raise HTTPException(422, "Invalid PDF page dimensions.")
                    check_dimensions(math.ceil(width * 2), math.ceil(height * 2))
                    bitmap = page.render(scale=2)
                    try:
                        return bitmap.to_pil().convert("RGB").copy()
                    finally:
                        bitmap.close()
                finally:
                    page.close()
            finally:
                document.close()
        if not (data.startswith(b"\x89PNG\r\n\x1a\n") or data.startswith(b"\xff\xd8\xff")):
            raise HTTPException(415, "Only PNG, JPEG and single-page PDF receipts are supported.")
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as image:
                if image.format not in ("PNG", "JPEG"):
                    raise HTTPException(415, "Unsupported image format.")
                check_dimensions(*image.size)
                image.verify()
            with Image.open(io.BytesIO(data)) as image:
                return ImageOps.exif_transpose(image).convert("RGB").copy()
    except HTTPException:
        raise
    except (Image.DecompressionBombWarning, Image.DecompressionBombError):
        raise HTTPException(413, "Image dimensions are too large.") from None
    except (UnidentifiedImageError, OSError, ValueError, RuntimeError):
        raise HTTPException(422, "The receipt file could not be safely decoded.") from None
    except Exception:
        raise HTTPException(422, "The receipt file could not be safely decoded.") from None


def safe_filename(value: str | None) -> str:
    value = (value or "receipt").replace("\\", "/").split("/")[-1]
    value = "".join(char for char in value if ord(char) >= 32 and ord(char) != 127)
    return value[:180] or "receipt"


def csv_safe(value) -> str:
    text = "" if value is None else str(value)
    if text.lstrip().startswith(("=", "+", "-", "@")) or text.startswith(("\t", "\r", "\n")):
        return "'" + text
    return text


class RequestBodyLimitMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("method") not in ("POST", "PATCH"):
            return await self.app(scope, receive, send)
        limit = MAX_BYTES + 65536 if scope["path"] == "/api/documents" else 512 * 1024
        chunks, length = [], 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            body = message.get("body", b"")
            length += len(body)
            if length > limit:
                return await JSONResponse(
                    {"detail": "Request body exceeds the local receipt limit."}, status_code=413
                )(scope, receive, send)
            chunks.append(body)
            if not message.get("more_body", False):
                break
        buffered, delivered = b"".join(chunks), False

        async def bounded_receive():
            nonlocal delivered
            if delivered:
                return await receive()
            delivered = True
            return {"type": "http.request", "body": buffered, "more_body": False}

        await self.app(scope, bounded_receive, send)


def create_app(
    data_dir: Path | None = None,
    models_dir: Path | None = None,
    samples_dir: Path | None = None,
    processor=None,
    worker_enabled: bool = True,
) -> FastAPI:
    store = Store(
        data_dir or Path(os.environ.get("RECEIPTLAB_DATA_DIR", ROOT / "data" / "workspace"))
    )
    models_dir = Path(models_dir or os.environ.get("RECEIPTLAB_MODEL_DIR", ROOT / "models"))
    samples_dir = Path(samples_dir or ROOT / "frontend" / "public" / "samples")
    worker = Worker(store, models_dir, processor)

    @asynccontextmanager
    async def lifespan(app):
        if worker_enabled:
            await worker.start()
        yield
        if worker_enabled:
            await worker.stop()

    app = FastAPI(
        title="ReceiptLab",
        version="1.0.0",
        lifespan=lifespan,
        description="Local receipt OCR, trained extraction, human review and export.",
    )
    app.state.store, app.state.worker = store, worker
    app.add_middleware(RequestBodyLimitMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=sorted(ALLOWED_ORIGINS),
        allow_credentials=False,
        allow_methods=["GET", "POST", "PATCH"],
        allow_headers=["Content-Type"],
    )

    @app.middleware("http")
    async def local_origin_guard(request: Request, call_next):
        origin = request.headers.get("origin")
        if origin and origin not in ALLOWED_ORIGINS:
            return JSONResponse(
                {"detail": "Only the local ReceiptLab interface is allowed."}, status_code=403
            )
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(MissingDocument)
    async def missing_document(request, error):
        return JSONResponse({"detail": "Receipt not found."}, status_code=404)

    @app.exception_handler(RevisionConflict)
    async def revision_conflict(request, error):
        return JSONResponse(
            {"detail": "Receipt changed. Refresh before saving your review."}, status_code=409
        )

    @app.get("/api/health")
    def health():
        return {
            "status": "ok",
            "model_ready": worker.model_ready,
            "storage": "sqlite",
            "scope": "single-user localhost",
            "max_upload_mb": 10,
            "max_pdf_pages": 1,
        }

    @app.get("/api/documents")
    def list_documents():
        return store.list()

    @app.get("/api/documents/{document_id}")
    def get_document(document_id: UUID):
        return store.get(str(document_id))

    def enqueue(data: bytes, filename: str, source: str):
        digest = hashlib.sha256(data).hexdigest()
        existing = store.find_hash(digest)
        if existing:
            return existing
        image = decode_receipt(data)
        document_id = str(uuid4())
        target, temporary = store.images / f"{document_id}.png", store.images / f"{document_id}.tmp"
        image.save(temporary, format="PNG")
        temporary.replace(target)
        result, created = store.create(
            document_id, digest, safe_filename(filename), source, *image.size
        )
        image.close()
        if not created:
            target.unlink(missing_ok=True)
        worker.wake()
        return result

    @app.post("/api/documents", status_code=202)
    async def upload_document(file: Annotated[UploadFile, File()]):
        try:
            data = await file.read(MAX_BYTES + 1)
            return await asyncio.to_thread(enqueue, data, file.filename or "receipt", "upload")
        finally:
            await file.close()

    @app.post("/api/samples/{name}", status_code=202)
    def load_sample(name: str):
        if name not in SAMPLE_NAMES:
            raise HTTPException(404, "Sample not found.")
        target = samples_dir / f"{name}.png"
        if not target.is_file():
            raise HTTPException(404, "Sample asset is unavailable.")
        return enqueue(target.read_bytes(), target.name, "sample")

    @app.get("/api/documents/{document_id}/image")
    def document_image(document_id: UUID):
        store.get(str(document_id))
        return FileResponse(store.images / f"{document_id}.png", media_type="image/png")

    @app.patch("/api/documents/{document_id}/review")
    def review_document(document_id: UUID, review: ReviewRequest):
        document = store.get(str(document_id))
        if document["revision"] != review.revision or document["status"] not in (
            "ready",
            "needs_review",
            "approved",
        ):
            raise RevisionConflict()
        extraction = dict(document["extraction"])
        fields = {
            name: {
                **extraction["fields"].get(name, {}),
                "value": getattr(review.fields, name),
                "reviewed": True,
            }
            for name in MONEY_FIELDS
        }
        previous_items = extraction.get("items", [])
        items = []
        for index, item in enumerate(review.items):
            data = item.model_dump()
            previous = previous_items[index] if index < len(previous_items) else {}
            unchanged = all(previous.get(key) == data.get(key) for key in data)
            items.append(
                {
                    **data,
                    "confidence": previous.get("confidence", 0.0) if unchanged else 0.0,
                    "reviewed": True,
                }
            )
        checks = financial_warnings(fields, items)
        if review.approve and checks:
            raise HTTPException(
                422, {"message": "Resolve financial checks before approval.", "warnings": checks}
            )
        extraction.update(
            fields=fields,
            items=items,
            warnings=checks,
            needs_review=not review.approve,
            reviewed=True,
        )
        return store.review(str(document_id), review.revision, extraction, review.approve)

    @app.get("/api/documents/{document_id}/audit")
    def document_audit(document_id: UUID):
        return store.audit(str(document_id))

    @app.get("/api/documents/{document_id}/export")
    def export_document(document_id: UUID, format: str = "json"):
        document = store.get(str(document_id))
        if not document["extraction"]:
            raise HTTPException(409, "Extraction must finish before exporting.")
        if format not in ("json", "csv"):
            raise HTTPException(422, "Export format must be json or csv.")
        headers = {"Content-Disposition": f'attachment; filename="receipt-{document_id}.{format}"'}
        if format == "json":
            return Response(
                json.dumps(document, indent=2, ensure_ascii=False),
                media_type="application/json",
                headers=headers,
            )
        buffer = io.StringIO(newline="")
        writer = csv.writer(buffer)
        writer.writerow(["record_type", "description", "quantity", "unit_price", "amount"])
        for name, field in document["extraction"]["fields"].items():
            writer.writerow(["field", name, "", "", csv_safe(field.get("value"))])
        for item in document["extraction"]["items"]:
            writer.writerow(
                [
                    "item",
                    *[
                        csv_safe(item.get(key))
                        for key in ("description", "quantity", "unit_price", "amount")
                    ],
                ]
            )
        return Response(buffer.getvalue(), media_type="text/csv; charset=utf-8", headers=headers)

    @app.get("/api/metrics")
    def metrics():
        report = ROOT / "reports" / "evaluation.json"
        if not report.is_file():
            return {
                "available": False,
                "message": "Run training and evaluation to generate measured results.",
            }
        data = json.loads(report.read_text(encoding="utf-8"))
        transformer = ROOT / "reports" / "transformer-evaluation.json"
        if transformer.exists():
            data["transformer"] = json.loads(transformer.read_text(encoding="utf-8"))
        assisted = ROOT / "reports" / "test-assisted-evaluation.json"
        if assisted.exists():
            data["assisted"] = json.loads(assisted.read_text(encoding="utf-8"))
        return data

    frontend = ROOT / "frontend" / "dist"
    if frontend.is_dir():
        app.mount("/", StaticFiles(directory=frontend, html=True), name="frontend")
    return app


app = create_app()
