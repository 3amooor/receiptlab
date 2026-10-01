"""Download the optional release weights, verify SHA256 and extract bounded files."""

import argparse
import hashlib
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--accept-noncommercial-license",
        action="store_true",
        help="Acknowledge CC-BY-NC-SA-4.0 license terms for these optional weights",
    )
    args = parser.parse_args()
    if not args.accept_noncommercial_license:
        parser.error(
            "Read docs/data-and-models.md; pass --accept-noncommercial-license to download research weights."
        )
    report = json.loads((ROOT / "reports" / "transformer-artifact.json").read_text())
    directory = ROOT / "data" / "downloads"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / report["asset"]
    url = "https://github.com/3amooor/receiptlab/releases/download/v1.0.0/" + report["asset"]
    if not target.exists():
        urllib.request.urlretrieve(url, target)
    with target.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != report["sha256"]:
            raise ValueError("Model archive checksum does not match the trusted repository report")
    destination = ROOT / "models" / "layoutlmv3"
    destination.mkdir(parents=True, exist_ok=True)
    allowed = set(report["files"]) | {"artifact-checksums.json"}
    with zipfile.ZipFile(target) as archive:
        if sum(entry.file_size for entry in archive.infolist()) > 1_000_000_000:
            raise ValueError("Model archive exceeds expected size")
        for entry in archive.infolist():
            if entry.filename not in allowed or Path(entry.filename).name != entry.filename:
                raise ValueError("Unexpected model archive path")
            with archive.open(entry) as source, (destination / entry.filename).open("wb") as output:
                shutil.copyfileobj(source, output)
    print("Verified research weights saved to", destination)


if __name__ == "__main__":
    main()
