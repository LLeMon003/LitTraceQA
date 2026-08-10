from __future__ import annotations

import hashlib
import json
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[1]
BASELINES = WORKSPACE / "records" / "immutable_baselines"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def main() -> int:
    directories = []
    overall_pass = True
    for baseline_path in sorted(BASELINES.glob("*.json")):
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        root = Path(baseline["root"])
        expected = {entry["path"].replace("\\", "/"): entry for entry in baseline["files"]}
        current_paths = {
            path.relative_to(root).as_posix(): path
            for path in root.rglob("*")
            if path.is_file()
        }
        missing = sorted(set(expected) - set(current_paths))
        added = sorted(set(current_paths) - set(expected))
        modified = []
        for relative in sorted(set(expected) & set(current_paths)):
            path = current_paths[relative]
            entry = expected[relative]
            if path.stat().st_size != entry["bytes"] or sha256(path) != entry["sha256"].upper():
                modified.append(relative)
        passed = not missing and not added and not modified
        overall_pass = overall_pass and passed
        directories.append({
            "name": baseline["name"],
            "root": str(root),
            "baseline_files": len(expected),
            "current_files": len(current_paths),
            "missing_count": len(missing),
            "added_count": len(added),
            "modified_count": len(modified),
            "missing_examples": missing[:10],
            "added_examples": added[:10],
            "modified_examples": modified[:10],
            "status": "pass" if passed else "fail",
        })
    result = {
        "experiment_id": "V2.3_FULL_CODE_REPRODUCTION_CONTINUATION",
        "status": "pass" if overall_pass else "fail",
        "directories": directories,
        "version_3_note": "No separate Version 3 top-level directory was discovered at Phase 0; the archived version_freeze baseline is verified here.",
    }
    print(json.dumps(result, indent=2))
    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
