"""Package locally trained noncommercial transformer weights with checksums."""

import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = ROOT / "models" / "layoutlmv3"
    files = sorted(
        p for p in source.iterdir() if p.is_file() and p.name != "artifact-checksums.json"
    )
    hashes = {p.name: hashlib.file_digest(p.open("rb"), "sha256").hexdigest() for p in files}
    (source / "artifact-checksums.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")
    destination = ROOT / "data" / "release"
    destination.mkdir(parents=True, exist_ok=True)
    output = destination / "receiptlab-layoutlmv3-v1.zip"
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for path in sorted(source.iterdir()):
            if path.is_file():
                archive.write(path, path.name)
    with output.open("rb") as opened:
        checksum = hashlib.file_digest(opened, "sha256").hexdigest()
    (destination / "receiptlab-layoutlmv3-v1.zip.sha256").write_text(
        checksum + "  " + output.name + "\n", encoding="utf-8"
    )
    report = {
        "asset": output.name,
        "sha256": checksum,
        "bytes": output.stat().st_size,
        "license": "CC-BY-NC-SA-4.0",
        "base_model": "microsoft/layoutlmv3-base",
        "release": "v1.0.0",
        "files": hashes,
    }
    (ROOT / "reports" / "transformer-artifact.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in report.items() if k != "files"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
