"""Single-user SQLite persistence with durable jobs and immutable audit events."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


class MissingDocument(Exception):
    pass


class RevisionConflict(Exception):
    pass


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Store:
    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.images = self.directory / "images"
        self.images.mkdir(exist_ok=True)
        self.database = self.directory / "receiptlab.sqlite3"
        self.migrate()

    @contextmanager
    def connection(self):
        connection = sqlite3.connect(self.database, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def migrate(self):
        with self.connection() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            if version > 1:
                raise RuntimeError("Database schema is newer than this application.")
            if version == 0:
                connection.executescript("""
                    CREATE TABLE documents (
                        id TEXT PRIMARY KEY, sha256 TEXT NOT NULL UNIQUE,
                        filename TEXT NOT NULL, created_at TEXT NOT NULL,
                        status TEXT NOT NULL, source TEXT NOT NULL,
                        width INTEGER NOT NULL, height INTEGER NOT NULL,
                        revision INTEGER NOT NULL DEFAULT 1,
                        extraction TEXT, original_extraction TEXT, error TEXT
                    );
                    CREATE INDEX documents_queue ON documents(status, created_at);
                    CREATE TABLE audits (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                        revision INTEGER NOT NULL, event TEXT NOT NULL,
                        created_at TEXT NOT NULL, before_json TEXT, after_json TEXT
                    );
                    PRAGMA user_version = 1;
                """)

    @staticmethod
    def public(row) -> dict:
        document = dict(row)
        document.pop("sha256", None)
        document.pop("original_extraction", None)
        document["extraction"] = (
            json.loads(document["extraction"]) if document["extraction"] else None
        )
        document["page_count"] = 1
        document["image_url"] = f"/api/documents/{document['id']}/image"
        return document

    def get(self, document_id: str) -> dict:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM documents WHERE id = ?", (document_id,)
            ).fetchone()
        if row is None:
            raise MissingDocument()
        return self.public(row)

    def list(self) -> list[dict]:
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT * FROM documents ORDER BY created_at DESC LIMIT 200"
            ).fetchall()
        return [self.public(row) for row in rows]

    def find_hash(self, digest: str) -> dict | None:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM documents WHERE sha256 = ?", (digest,)
            ).fetchone()
        return self.public(row) if row else None

    def create(
        self, document_id: str, digest: str, filename: str, source: str, width: int, height: int
    ):
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM documents WHERE sha256 = ?", (digest,)
            ).fetchone()
            if row:
                return self.public(row), False
            connection.execute(
                "INSERT INTO documents(id,sha256,filename,created_at,status,source,width,height) VALUES (?,?,?,?,?,?,?,?)",
                (document_id, digest, filename, now(), "queued", source, width, height),
            )
            self._audit(connection, document_id, 1, "uploaded", None, {"status": "queued"})
        return self.get(document_id), True

    @staticmethod
    def _audit(connection, document_id, revision, event, before, after):
        connection.execute(
            "INSERT INTO audits(document_id,revision,event,created_at,before_json,after_json) VALUES (?,?,?,?,?,?)",
            (document_id, revision, event, now(), json.dumps(before), json.dumps(after)),
        )

    def recover(self):
        with self.connection() as connection:
            connection.execute("UPDATE documents SET status = 'queued' WHERE status = 'processing'")

    def claim(self) -> dict | None:
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM documents WHERE status = 'queued' ORDER BY created_at LIMIT 1"
            ).fetchone()
            if row is None:
                return None
            connection.execute(
                "UPDATE documents SET status = 'processing' WHERE id = ?", (row["id"],)
            )
        return self.get(row["id"])

    def finish(self, document_id: str, extraction: dict | None, error: str | None = None):
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM documents WHERE id = ?", (document_id,)
            ).fetchone()
            if row is None:
                raise MissingDocument()
            status = (
                "failed" if error else "needs_review" if extraction["needs_review"] else "ready"
            )
            encoded = json.dumps(extraction) if extraction else None
            revision = row["revision"] + 1
            connection.execute(
                "UPDATE documents SET status=?, extraction=?, original_extraction=?, error=?, revision=? WHERE id=?",
                (status, encoded, encoded, error, revision, document_id),
            )
            self._audit(
                connection,
                document_id,
                revision,
                "extracted" if not error else "failed",
                None,
                extraction,
            )
        return self.get(document_id)

    def review(self, document_id: str, revision: int, extraction: dict, approve: bool):
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM documents WHERE id = ?", (document_id,)
            ).fetchone()
            if row is None:
                raise MissingDocument()
            if row["revision"] != revision or row["status"] not in (
                "ready",
                "needs_review",
                "approved",
            ):
                raise RevisionConflict()
            status = "approved" if approve else "needs_review"
            connection.execute(
                "UPDATE documents SET extraction=?, revision=?, status=? WHERE id=?",
                (json.dumps(extraction), revision + 1, status, document_id),
            )
            self._audit(
                connection,
                document_id,
                revision + 1,
                "approved" if approve else "corrected",
                json.loads(row["extraction"]),
                extraction,
            )
        return self.get(document_id)

    def audit(self, document_id: str) -> list[dict]:
        self.get(document_id)
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT * FROM audits WHERE document_id=? ORDER BY id", (document_id,)
            ).fetchall()
        return [
            {
                "id": row["id"],
                "revision": row["revision"],
                "event": row["event"],
                "created_at": row["created_at"],
                "before": json.loads(row["before_json"]),
                "after": json.loads(row["after_json"]),
            }
            for row in rows
        ]
