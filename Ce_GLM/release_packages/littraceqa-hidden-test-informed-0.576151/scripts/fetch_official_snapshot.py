#!/usr/bin/env python3
"""Download and hash-lock the public LitTraceQA files needed for schema validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads((ROOT / "release_manifest.json").read_text(encoding="utf-8"))
REVISION = MANIFEST["official_snapshot"]["revision"]
BASE_URL = f"https://huggingface.co/datasets/LitTraceQA/LitTraceQA/resolve/{REVISION}"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def download(relative: str, destination: Path, expected: str) -> str:
    if destination.exists() and sha256(destination) == expected:
        return "cached"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    url = f"{BASE_URL}/{relative}?download=true"
    request = urllib.request.Request(url, headers={"User-Agent": "GroundLM-reproducible-release/1.0"})
    with urllib.request.urlopen(request) as response, temporary.open("wb") as handle:
        while chunk := response.read(1024 * 1024):
            handle.write(chunk)
    actual = sha256(temporary)
    if actual != expected:
        temporary.unlink(missing_ok=True)
        raise AssertionError(f"Downloaded hash mismatch for {relative}: {actual} != {expected}")
    temporary.replace(destination)
    return "downloaded"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=".cache/official_snapshot")
    args = parser.parse_args()
    output = resolve(args.output)
    results = {}
    for relative, expected in MANIFEST["official_snapshot"]["files"].items():
        results[relative] = download(relative, output / relative, expected)
    print(json.dumps({"status": "PASS", "revision": REVISION, "output": str(output), "files": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
