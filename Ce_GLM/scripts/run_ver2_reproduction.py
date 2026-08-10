from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


WORKSPACE = Path(__file__).resolve().parents[1]
ROOT = WORKSPACE.parent
VER2 = ROOT / "littraceqa_baseline_Ver.2"
UQ = ROOT / "littraceqa_baseline_uq_experiments"
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from src.credential_resolver import CredentialUnavailable, resolve_provider_config
from src.jsonl_io import inspect_jsonl

CACHE_SHA = "54DA46600AFE81DAB5D8D2F10E87AC453FF5FCA206296DAF126EFFCD4D4C409D"
PARENT_19_SHA = "188D081433DA039171AFDF56EF6248F79520019EEA8A7DAD5B1B4FD336FD8344"
COMPLETENESS_SHA = "C3B51991E37EC1DC2E778E3FC41C2EA41D6D96DF4793C24A19F447D966915B0C"
DEV_OPTION_SHA = "9338B351E7DBC677F58C7BC847D8754E251D86E26746BDB9911677A34C741839"
EVIDENCE_SAFE_SHA = "A9A99BB552363E030F5876664AE8BB4024381A72C66A0782D993603A3D0D7B89"
SOURCE_GROUNDED_SHA = "D791909E0C342BA2393D617ABD007DE95D8127036545FE9B3D44AEC59A94DCAA"
DEV_TARGET_SHA = "2635F99AA8B1C22AD941AB2761C64622710528DA833A16014FF7DF021EF69364"
PRODUCTION_OPTION_SHA = "944513E5084522C3233842B9AE15DCD7BB2CEE1E44DC683DC827C8968FEDBBDE"
PRODUCTION_TARGET_SHA = "076C949F4B40FDBF8D963CA5A92C31838C9F24D6BE9D7ED97ABC69DCEC92AFF8"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def verify_fresh_asset_manifest(path: Path) -> dict[str, Any]:
    """Verify a relocatable fresh-input manifest without inspecting data contents."""
    manifest_path = require_file(path, "fresh asset manifest")
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    assets = document.get("assets")
    if not isinstance(assets, list):
        raise ValueError("Fresh asset manifest has no assets list")
    roots = {"release_root": VER2, "source_root": UQ}
    missing: list[str] = []
    mismatched: list[str] = []
    for item in assets:
        if not isinstance(item, dict):
            raise ValueError("Fresh asset manifest contains a non-object entry")
        role, root_name, relative = (str(item.get(key) or "") for key in ("role", "root", "relative_path"))
        expected = str(item.get("sha256") or "").upper()
        target_root = roots.get(root_name)
        if not role or target_root is None or not relative or len(expected) != 64:
            raise ValueError("Fresh asset manifest contains an invalid asset declaration")
        candidate = target_root / relative
        if not candidate.is_file():
            missing.append(role)
        elif sha256(candidate) != expected:
            mismatched.append(role)
    return {
        "schema_version": document.get("schema_version"),
        "manifest_sha256": sha256(manifest_path),
        "missing": missing,
        "mismatched": mismatched,
        "verified_asset_count": len(assets) - len(missing) - len(mismatched),
    }


def require_file(path: Path, label: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Missing {label}: {resolved}")
    return resolved


def configure_external_roots(args: argparse.Namespace) -> None:
    """Select explicitly supplied source releases without relying on sibling paths."""
    global VER2, UQ
    if args.release_root is not None:
        VER2 = args.release_root.expanduser().resolve()
    if args.source_root is not None:
        UQ = args.source_root.expanduser().resolve()


def raw_fresh_requirements() -> list[tuple[str, Path]]:
    """Only the immutable source inputs required before a raw-fresh API call."""
    return [
        ("Version 2 fresh runner", VER2 / "run_full_validation.py"),
        ("validation inputs", VER2 / "inputs" / "validation_inputs.jsonl"),
        ("paper metadata", VER2 / "inputs" / "paper_metadata.jsonl"),
        ("option-bearing inputs", VER2 / "inputs_with_options" / "validation_inputs_with_options.jsonl"),
        ("full Docling object index", VER2 / "artifacts" / "document_object_index_full_docling.jsonl"),
        ("sentence index", VER2 / "artifacts" / "sentence_index.jsonl"),
        ("production evidence parent", VER2 / "checkpoints" / "PRODUCTION_OPTION_AWARE_predictions.jsonl"),
        ("option-aware replay", UQ / "scripts" / "replay_option_aware_multiple_choice.py"),
        ("option-aware object index", UQ / "outputs" / "document_object_index.jsonl"),
    ]


def raw_fresh_preflight(manifest_path: Path, *, text_model: str | None = None) -> dict[str, Any]:
    """Return a secret-free readiness result without starting a model request."""
    missing = [label for label, path in raw_fresh_requirements() if not path.is_file()]
    provider: dict[str, str | bool]
    try:
        provider = resolve_provider_config().status()
        requested_model = str(text_model or "").strip()
        if requested_model:
            provider = {**provider, "model": requested_model, "model_source": "explicit_child_override"}
    except CredentialUnavailable as exc:
        provider = {"credential_present": False, "diagnostic": str(exc)}
    assets = verify_fresh_asset_manifest(manifest_path)
    ready = not missing and not assets["missing"] and not assets["mismatched"] and bool(provider.get("credential_present"))
    return {
        "mode": "raw-fresh",
        "status": "ready" if ready else "not_ready",
        "missing_required_assets": missing,
        "asset_manifest": assets,
        "provider": provider,
        "cache_boundary_used": False,
        "official_evaluator_invoked": False,
    }


def jsonl_count(path: Path) -> tuple[int, int]:
    profile = inspect_jsonl(path)
    if profile["duplicate_query_ids"]:
        raise ValueError(f"Duplicate query identifiers in {path}: {profile['duplicate_query_ids'][:5]}")
    return int(profile["records"]), int(profile["unique_query_ids"])


def file_profile(path: Path) -> dict[str, Any]:
    path = require_file(path, "stage artifact")
    profile: dict[str, Any] = {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }
    if path.suffix.lower() == ".jsonl":
        records, identifiers = jsonl_count(path)
        profile.update({"records": records, "unique_query_ids": identifiers})
    return profile


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: JSONL record is not an object")
            rows.append(value)
    return rows


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temp.replace(path)


def query_id(row: dict[str, Any]) -> str:
    return str(row.get("query_id") or row.get("id") or "")


def ordered_input_ids(path: Path) -> list[str]:
    return [query_id(row) for row in read_jsonl(path)]


def rows_by_unique_query_id(path: Path) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(path)
    counts: dict[str, int] = {}
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        qid = query_id(row)
        if not qid:
            raise ValueError(f"{path}: record without query_id")
        counts[qid] = counts.get(qid, 0) + 1
        by_id[qid] = row
    duplicates = sorted(qid for qid, count in counts.items() if count > 1)
    if duplicates:
        raise ValueError(f"{path}: duplicate query_id values: {duplicates[:5]}")
    return by_id


def predictions_from_raw_generation_log(path: Path) -> dict[str, dict[str, Any]]:
    raw_by_id = rows_by_unique_query_id(path)
    predictions: dict[str, dict[str, Any]] = {}
    for qid, row in raw_by_id.items():
        prediction = row.get("final_prediction")
        if not isinstance(prediction, dict):
            raise ValueError(f"{path}: raw log record {qid} has no final_prediction object")
        if query_id(prediction) != qid:
            raise ValueError(f"{path}: final_prediction query_id mismatch for {qid}")
        predictions[qid] = prediction
    return predictions


def terminate_process_tree(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except Exception:
            process.terminate()


@dataclass
class StageResult:
    name: str
    status: str
    input_files: list[Path]
    output_file: Path
    code_files: list[Path]
    started_at: str
    duration_sec: float
    warnings: list[str] = field(default_factory=list)
    command: list[str] | None = None
    stdout_log: str | None = None
    stderr_log: str | None = None

    def manifest(self) -> dict[str, Any]:
        return {
            "stage": self.name,
            "status": self.status,
            "cache_status": self.status,
            "started_at_utc": self.started_at,
            "duration_sec": round(self.duration_sec, 6),
            "input_hashes": [file_profile(path) for path in self.input_files],
            "output_hashes": [file_profile(self.output_file)],
            "record_count": file_profile(self.output_file).get("records"),
            "warnings": self.warnings,
            "code_files_used": [
                {"path": str(path.resolve()), "sha256": sha256(path.resolve())}
                for path in self.code_files
            ],
            "command": self.command,
            "stdout_log": self.stdout_log,
            "stderr_log": self.stderr_log,
            "gold_used_in_generation": False,
        }


class StopPipeline(Exception):
    def __init__(self, output: Path) -> None:
        super().__init__(str(output))
        self.output = output


class Runner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.child_environment = dict(os.environ)
        self.provider_status: dict[str, str | bool] | None = None
        if args.mode == "raw-fresh":
            self.configure_raw_fresh_credentials()
            if args.verify_hashes:
                assets = verify_fresh_asset_manifest(args.assets_manifest)
                if assets["missing"] or assets["mismatched"]:
                    raise RuntimeError("Fresh asset manifest verification failed before generation")
        self.output = args.output.resolve()
        self.run_root = self.output.parent
        self.manifest_dir = self.run_root / "stage_manifests"
        self.work_dir = self.run_root / "work"
        self.logs_dir = self.run_root / "logs"
        self.stages: list[dict[str, Any]] = []
        self.evaluator_calls = 0
        self.api_calls = 0
        self.started_at = utc_now()
        self.stage_order = self.order_for_mode(args.mode)
        self.start_index = self.stage_order.index(args.start_stage) if args.start_stage else 0
        self.stop_index = self.stage_order.index(args.stop_stage) if args.stop_stage else len(self.stage_order) - 1
        if self.start_index > self.stop_index:
            raise ValueError("--start-stage occurs after --stop-stage")
        self.resume = self.load_resume(args.resume_manifest)
        self.last_output: Path | None = None

    def configure_raw_fresh_credentials(self) -> None:
        """Inject a resolved credential only into child-process environment memory."""
        try:
            config = resolve_provider_config()
        except CredentialUnavailable as exc:
            raise RuntimeError(str(exc)) from exc
        self.provider_status = config.status()
        if not str(self.child_environment.get("SILICONFLOW_API_KEY") or "").strip():
            self.child_environment["SILICONFLOW_API_KEY"] = config.credential.value
        if not str(self.child_environment.get("SILICONFLOW_BASE_URL") or "").strip():
            self.child_environment["SILICONFLOW_BASE_URL"] = config.endpoint
        requested_model = str(getattr(getattr(self, "args", None), "text_model", "") or "").strip()
        if requested_model:
            self.child_environment["SILICONFLOW_TEXT_MODEL"] = requested_model
            self.provider_status = {**self.provider_status, "model": requested_model, "model_source": "explicit_child_override"}
            return
        if not str(self.child_environment.get("SILICONFLOW_TEXT_MODEL") or self.child_environment.get("SILICONFLOW_MODEL") or "").strip():
            self.child_environment["SILICONFLOW_TEXT_MODEL"] = config.model

    @staticmethod
    def order_for_mode(mode: str) -> list[str]:
        if mode == "cached-exact":
            return [
                "cache_boundary",
                "table_and_evidence_ancestry",
                "freeform_completeness",
                "option_aware_mc",
                "evidence_safe_cleanup",
                "source_grounded_mc",
                "typed_mc",
                "final_prediction",
            ]
        if mode == "production-cached-exact":
            return ["cache_boundary", "production_option_aware", "typed_mc", "final_prediction"]
        return [
            "raw_generation_and_freeform",
            "option_aware_mc",
            "evidence_safe_cleanup",
            "source_grounded_mc",
            "typed_mc",
            "final_prediction",
        ]

    @staticmethod
    def load_resume(path: Path | None) -> dict[str, Any] | None:
        if path is None:
            return None
        return json.loads(require_file(path, "resume manifest").read_text(encoding="utf-8"))

    def initialize(self) -> None:
        if self.output.exists() and self.resume is None and self.args.resume_run is None:
            raise FileExistsError(f"Refusing to overwrite output: {self.output}")
        if self.run_root.exists() and any(self.run_root.iterdir()) and self.resume is None and self.args.resume_run is None:
            raise FileExistsError(f"Refusing to reuse non-empty run directory: {self.run_root}")
        self.manifest_dir.mkdir(parents=True, exist_ok=True)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    def verify_expected(self, path: Path, expected: str, label: str) -> None:
        actual = sha256(require_file(path, label))
        if actual != expected:
            raise RuntimeError(f"{label} SHA-256 mismatch: expected {expected}, got {actual}")

    def record(self, result: StageResult) -> Path:
        manifest = result.manifest()
        index = self.stage_order.index(result.name)
        manifest_path = self.manifest_dir / f"{index:02d}_{result.name}.json"
        write_json(manifest_path, manifest)
        self.stages.append(manifest)
        self.last_output = result.output_file
        self.write_run_manifest(status="in_progress")
        return result.output_file

    def stage_gate(self, name: str) -> Path | None:
        index = self.stage_order.index(name)
        if index > self.stop_index:
            if self.last_output is None:
                raise RuntimeError("No completed output is available at --stop-stage")
            raise StopPipeline(self.last_output)
        if index >= self.start_index:
            return None
        if self.resume is None:
            raise RuntimeError(f"Stage {name} precedes --start-stage but no resume manifest was supplied")
        candidates = [stage for stage in self.resume.get("stages", []) if stage.get("stage") == name]
        if len(candidates) != 1:
            raise RuntimeError(f"Resume manifest must contain exactly one completed {name} stage")
        prior = candidates[0]
        outputs = prior.get("output_hashes") or []
        if len(outputs) != 1:
            raise RuntimeError(f"Resume stage {name} does not have one output artifact")
        path = require_file(Path(outputs[0]["path"]), f"resumed {name} output")
        actual = sha256(path)
        if actual != str(outputs[0]["sha256"]).upper():
            raise RuntimeError(f"Resumed {name} output hash changed")
        started = time.perf_counter()
        return self.record(
            StageResult(
                name=name,
                status="CACHE_HIT",
                input_files=[],
                output_file=path,
                code_files=[],
                started_at=utc_now(),
                duration_sec=time.perf_counter() - started,
                warnings=["Reused from --resume-manifest after verifying its recorded SHA-256."],
            )
        )

    def write_run_manifest(self, status: str, error: str | None = None) -> None:
        write_json(
            self.run_root / "run_manifest.json",
            {
                "schema_version": 1,
                "experiment_id": "V2.3_FULL_CODE_REPRODUCTION_CONTINUATION",
                "mode": self.args.mode,
                "status": status,
                "started_at_utc": self.started_at,
                "updated_at_utc": utc_now(),
                "start_stage": self.args.start_stage,
                "stop_stage": self.args.stop_stage,
                "verify_hashes": self.args.verify_hashes,
                "resume_run": str(self.args.resume_run) if self.args.resume_run else None,
                "output": str(self.output),
                "stages": self.stages,
                "evaluator_calls": self.evaluator_calls,
                "api_calls": self.api_calls,
                "error": error,
                "final_target_used_as_input": False,
                "gold_used_in_generation": False,
            },
        )

    def cache_stage(
        self,
        name: str,
        input_files: list[Path],
        output_file: Path,
        expected_sha: str,
        warning: str,
    ) -> Path:
        resumed = self.stage_gate(name)
        if resumed is not None:
            return resumed
        started = time.perf_counter()
        started_at = utc_now()
        self.verify_expected(output_file, expected_sha, name)
        return self.record(
            StageResult(
                name=name,
                status="VERIFIED_CACHE_INPUT" if name == "cache_boundary" else "CACHE_HIT",
                input_files=input_files,
                output_file=output_file,
                code_files=[],
                started_at=started_at,
                duration_sec=time.perf_counter() - started,
                warnings=[warning],
            )
        )

    def run_streaming_command(
        self,
        name: str,
        command: list[str],
        cwd: Path,
        stdout_path: Path,
        stderr_path: Path,
        heartbeat_files: list[Path] | None = None,
    ) -> dict[str, Any]:
        started_at = utc_now()
        heartbeat_files = heartbeat_files or []
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stderr_path.parent.mkdir(parents=True, exist_ok=True)
        last_activity = time.monotonic()
        last_sizes = {path: path.stat().st_size if path.exists() else 0 for path in heartbeat_files}
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        preexec_fn = None if os.name == "nt" else os.setsid
        with stdout_path.open("w", encoding="utf-8", newline="\n") as out, stderr_path.open("w", encoding="utf-8", newline="\n") as err:
            process = subprocess.Popen(
                command,
                cwd=cwd,
                env={**self.child_environment, "PYTHONDONTWRITEBYTECODE": "1"},
                text=True,
                encoding="utf-8",
                errors="replace",
                stdout=out,
                stderr=err,
                creationflags=creationflags,
                preexec_fn=preexec_fn,
            )
            write_json(
                self.logs_dir / f"{name}.process.json",
                {
                    "stage": name,
                    "pid": process.pid,
                    "command": command,
                    "cwd": str(cwd),
                    "started_at_utc": started_at,
                    "stall_timeout_sec": self.args.stall_timeout_sec,
                    "heartbeat_files": [str(path) for path in heartbeat_files],
                },
            )
            while True:
                exit_code = process.poll()
                current_sizes = {path: path.stat().st_size if path.exists() else 0 for path in heartbeat_files}
                if current_sizes != last_sizes:
                    last_activity = time.monotonic()
                    last_sizes = current_sizes
                    printable_sizes = {str(path): size for path, size in current_sizes.items()}
                    try:
                        print(json.dumps({"stage": name, "pid": process.pid, "heartbeat_bytes": printable_sizes}, sort_keys=True), flush=True)
                    except OSError:
                        pass
                if exit_code is not None:
                    return {"pid": process.pid, "started_at_utc": started_at, "exit_code": int(exit_code), "timed_out": False}
                if time.monotonic() - last_activity > float(self.args.stall_timeout_sec):
                    terminate_process_tree(process)
                    return {
                        "pid": process.pid,
                        "started_at_utc": started_at,
                        "exit_code": None,
                        "timed_out": True,
                        "stall_timeout_sec": self.args.stall_timeout_sec,
                    }
                time.sleep(float(self.args.heartbeat_interval_sec))

    def command_stage(
        self,
        name: str,
        command: list[str],
        cwd: Path,
        input_files: list[Path],
        output_file: Path,
        expected_sha: str | None,
        code_files: list[Path],
        heartbeat_files: list[Path] | None = None,
    ) -> Path:
        resumed = self.stage_gate(name)
        if resumed is not None:
            return resumed
        started = time.perf_counter()
        started_at = utc_now()
        stdout_path = self.logs_dir / f"{name}.stdout.log"
        stderr_path = self.logs_dir / f"{name}.stderr.log"
        process = self.run_streaming_command(name, command, cwd, stdout_path, stderr_path, heartbeat_files=heartbeat_files or [output_file])
        if process.get("timed_out"):
            raise RuntimeError(f"Stage {name} stalled for {process.get('stall_timeout_sec')} seconds; child tree terminated; see {stderr_path}")
        if process["exit_code"]:
            raise RuntimeError(f"Stage {name} failed with exit code {process['exit_code']}; see {stderr_path}")
        require_file(output_file, f"{name} output")
        if expected_sha is not None:
            self.verify_expected(output_file, expected_sha, name)
        return self.record(
            StageResult(
                name=name,
                status="EXECUTED",
                input_files=input_files,
                output_file=output_file,
                code_files=code_files,
                started_at=started_at,
                duration_sec=time.perf_counter() - started,
                command=command,
                stdout_log=str(stdout_path),
                stderr_log=str(stderr_path),
            )
        )

    def active(self, name: str) -> bool:
        index = self.stage_order.index(name)
        return self.start_index <= index <= self.stop_index

    def not_applicable(self, name: str, output_file: Path, reason: str) -> Path:
        started = time.perf_counter()
        return self.record(
            StageResult(
                name=name,
                status="NOT_APPLICABLE",
                input_files=[],
                output_file=output_file,
                code_files=[],
                started_at=utc_now(),
                duration_sec=time.perf_counter() - started,
                warnings=[reason],
            )
        )

    def run_cached_exact(self) -> Path:
        cache = UQ / "outputs" / "modular_composed_label_locator_predictions.jsonl"
        parent_19 = VER2 / "checkpoints" / "parent_19of26_25of27_predictions.jsonl"
        current = self.cache_stage(
            "cache_boundary", [], cache, CACHE_SHA,
            "Historical producer record is absent; see records/CACHE_BOUNDARY_DECISION.json.",
        )
        current = self.cache_stage(
            "table_and_evidence_ancestry", [current], parent_19, PARENT_19_SHA,
            "Archived accepted table/evidence ancestry is a verified cache hit; individual transitions remain recorded in VERSION2_LINEAGE_GRAPH.json.",
        )

        freeform_dir = self.work_dir / "freeform_completeness"
        command = [
            sys.executable, str(WORKSPACE / "scripts" / "run_freeform_chain.py"),
            "--release-root", str(VER2),
            "--parent", str(parent_19),
            "--input", str(VER2 / "inputs" / "validation_inputs.jsonl"),
            "--metadata", str(VER2 / "inputs" / "paper_metadata.jsonl"),
            "--object-index", str(VER2 / "artifacts" / "document_object_index_full_docling.jsonl"),
            "--facts", str(VER2 / "artifacts" / "paper_selection_active_facts.jsonl"),
            "--output-dir", str(freeform_dir),
        ]
        current = self.command_stage(
            "freeform_completeness", command, VER2, [current], freeform_dir / "predictions.jsonl",
            COMPLETENESS_SHA, [
                WORKSPACE / "scripts" / "run_freeform_chain.py",
                VER2 / "scripts" / "run_freeform_completeness_renderer.py",
                VER2 / "scripts" / "run_freeform_source_sentence_expander.py",
                VER2 / "scripts" / "run_complete_paper_selection_renderer.py",
                VER2 / "scripts" / "run_freeform_difference_completeness.py",
            ],
        )

        option_dir = self.work_dir / "option_aware_mc"
        command = [
            sys.executable, str(UQ / "scripts" / "replay_option_aware_multiple_choice.py"),
            "--input", str(VER2 / "inputs_with_options" / "validation_inputs_with_options.jsonl"),
            "--checkpoint", str(current),
            "--paper-metadata", str(VER2 / "inputs" / "paper_metadata.jsonl"),
            "--object-index", str(UQ / "outputs" / "document_object_index.jsonl"),
            "--output-dir", str(option_dir),
        ]
        current = self.command_stage(
            "option_aware_mc", command, UQ, [current], option_dir / "DEV_BEST_OPTION_AWARE_predictions.jsonl",
            DEV_OPTION_SHA, [UQ / "scripts" / "replay_option_aware_multiple_choice.py"],
        )

        evidence_dir = self.work_dir / "evidence_safe_cleanup"
        command = [
            sys.executable, str(VER2 / "scripts" / "replay_claim_evidence.py"),
            "--input", str(VER2 / "inputs_with_options" / "validation_inputs_with_options.jsonl"),
            "--parent", str(current),
            "--production-parent", str(VER2 / "checkpoints" / "PRODUCTION_OPTION_AWARE_predictions.jsonl"),
            "--paper-metadata", str(VER2 / "inputs" / "paper_metadata.jsonl"),
            "--object-index", str(VER2 / "artifacts" / "document_object_index_full_docling.jsonl"),
            "--output-dir", str(evidence_dir),
        ]
        current = self.command_stage(
            "evidence_safe_cleanup", command, VER2, [current], evidence_dir / "deterministic_shadow_predictions.jsonl",
            EVIDENCE_SAFE_SHA, [VER2 / "scripts" / "replay_claim_evidence.py"],
        )

        source_grounded = VER2 / "checkpoints" / "CANONICAL_DEV_BEST_SOURCE_GROUNDED_MC_predictions.jsonl"
        current = self.cache_stage(
            "source_grounded_mc", [current], source_grounded, SOURCE_GROUNDED_SHA,
            "The accepted source-grounded runner is release-path-bound; its frozen hash-verified output is used as an explicit cache hit.",
        )

        typed_dir = self.work_dir / "typed_mc"
        command = [
            sys.executable, str(VER2 / "scripts" / "run_typed_mc_pipeline.py"),
            "--parent", str(current),
            "--semantic-answers", str(VER2 / "artifacts" / "source_grounded_mc_semantic_answers.jsonl"),
            "--options", str(VER2 / "inputs_with_options" / "validation_inputs_with_options.jsonl"),
            "--sentence-index", str(VER2 / "artifacts" / "sentence_index.jsonl"),
            "--output-dir", str(typed_dir),
            "--expected-parent-sha256", SOURCE_GROUNDED_SHA,
        ]
        current = self.command_stage(
            "typed_mc", command, VER2, [current], typed_dir / "predictions.jsonl",
            DEV_TARGET_SHA, [VER2 / "scripts" / "run_typed_mc_pipeline.py"],
        )
        return self.final_stage(current, DEV_TARGET_SHA)

    def run_production_exact(self) -> Path:
        cache = UQ / "outputs" / "modular_composed_label_locator_predictions.jsonl"
        current = self.cache_stage(
            "cache_boundary", [], cache, CACHE_SHA,
            "Historical producer record is absent; see records/CACHE_BOUNDARY_DECISION.json.",
        )
        production_parent = VER2 / "checkpoints" / "PRODUCTION_OPTION_AWARE_predictions.jsonl"
        current = self.cache_stage(
            "production_option_aware", [current], production_parent, PRODUCTION_OPTION_SHA,
            "Production option-aware ancestry is a frozen verified cache hit.",
        )
        typed_dir = self.work_dir / "typed_mc"
        command = [
            sys.executable, str(VER2 / "scripts" / "run_typed_mc_pipeline.py"),
            "--parent", str(current),
            "--semantic-answers", str(VER2 / "artifacts" / "source_grounded_mc_semantic_answers.jsonl"),
            "--options", str(VER2 / "inputs_with_options" / "validation_inputs_with_options.jsonl"),
            "--sentence-index", str(VER2 / "artifacts" / "sentence_index.jsonl"),
            "--output-dir", str(typed_dir),
            "--expected-parent-sha256", PRODUCTION_OPTION_SHA,
        ]
        current = self.command_stage(
            "typed_mc", command, VER2, [current], typed_dir / "predictions.jsonl",
            PRODUCTION_TARGET_SHA, [VER2 / "scripts" / "run_typed_mc_pipeline.py"],
        )
        return self.final_stage(current, PRODUCTION_TARGET_SHA)

    def raw_generation_command(self, raw_dir: Path, input_path: Path) -> list[str]:
        cache_dir = self.run_root / "cache"
        pdf_dir = cache_dir / "pdfs"
        pdf_text_dir = cache_dir / "pdf_text_cache"
        pdf_image_dir = cache_dir / "pdf_page_images"
        pdf_vlm_dir = cache_dir / "pdf_vlm_cache"
        pdf_manifest = cache_dir / "pdf_download_manifest.jsonl"
        return [
            sys.executable, str(VER2 / "run_full_validation.py"),
            "--env-file", str(self.args.env_file or (WORKSPACE / "configs" / ".env.example")),
            "--input", str(input_path),
            "--object-index", str(VER2 / "artifacts" / "document_object_index_full_docling.jsonl"),
            "--output-dir", str(raw_dir),
            "--baseline-extra",
            "--pdf-dir", str(pdf_dir),
            "--pdf-text-dir", str(pdf_text_dir),
            "--pdf-image-dir", str(pdf_image_dir),
            "--pdf-vlm-cache-dir", str(pdf_vlm_dir),
            "--pdf-manifest", str(pdf_manifest),
        ]

    def merge_partial_raw_generation(self, resume_run: Path, raw_dir: Path) -> Path:
        source_raw = require_file(resume_run / "work" / "raw_full_validation" / "raw_generation.jsonl", "partial raw generation")
        source_run = require_file(resume_run / "work" / "raw_full_validation" / "run_log.jsonl", "partial run log")
        full_input = VER2 / "inputs" / "validation_inputs.jsonl"
        input_rows = read_jsonl(full_input)
        input_ids = [query_id(row) for row in input_rows]
        partial_raw_by_id = rows_by_unique_query_id(source_raw)
        partial_prediction_by_id = predictions_from_raw_generation_log(source_raw)
        partial_run_by_id = rows_by_unique_query_id(source_run)
        completed_ids = [qid for qid in input_ids if qid in partial_prediction_by_id]
        remaining_rows = [row for row in input_rows if query_id(row) not in partial_prediction_by_id]
        extra = sorted(set(partial_prediction_by_id) - set(input_ids))
        if extra:
            raise ValueError(f"Partial raw generation contains unexpected query IDs: {extra[:5]}")
        if len(partial_prediction_by_id) != len(partial_run_by_id):
            raise ValueError("Partial raw generation and run log have different completed ID counts")
        raw_dir.mkdir(parents=True, exist_ok=True)
        missing_input = raw_dir / "remaining_input.jsonl"
        write_jsonl_atomic(missing_input, remaining_rows)
        missing_dir = raw_dir / "resume_missing_generation"
        stdout_path = self.logs_dir / "raw_generation_resume_missing.stdout.log"
        stderr_path = self.logs_dir / "raw_generation_resume_missing.stderr.log"
        expected_missing = [query_id(row) for row in remaining_rows]
        missing_raw_log_path = missing_dir / "raw_generation.jsonl"
        missing_stage_path = missing_dir / "stage_00_generated.jsonl"
        missing_run_log_path = missing_dir / "run_log.jsonl"
        reusable_missing_generation = False
        if missing_raw_log_path.exists() and missing_stage_path.exists() and missing_run_log_path.exists():
            existing_raw_log = rows_by_unique_query_id(missing_raw_log_path)
            existing_predictions = rows_by_unique_query_id(missing_stage_path)
            existing_run = rows_by_unique_query_id(missing_run_log_path)
            reusable_missing_generation = (
                sorted(existing_predictions) == sorted(expected_missing)
                and sorted(existing_raw_log) == sorted(expected_missing)
                and sorted(existing_run) == sorted(expected_missing)
            )
        if reusable_missing_generation:
            process = {
                "exit_code": 0,
                "reused_completed_missing_generation": True,
                "reason": "Existing missing-only raw generation artifacts contain exactly the expected remaining query IDs.",
            }
        else:
            command = self.raw_generation_command(missing_dir, missing_input)
            process = self.run_streaming_command(
                "raw_generation_resume_missing",
                command,
                VER2,
                stdout_path,
                stderr_path,
                heartbeat_files=[missing_raw_log_path, missing_run_log_path],
            )
            if process.get("timed_out"):
                raise RuntimeError(f"Resume missing generation stalled for {process.get('stall_timeout_sec')} seconds; see {stderr_path}")
            if process["exit_code"]:
                raise RuntimeError(f"Resume missing generation failed with exit code {process['exit_code']}; see {stderr_path}")
        new_raw_log = rows_by_unique_query_id(require_file(missing_raw_log_path, "missing raw generation"))
        new_predictions = rows_by_unique_query_id(require_file(missing_stage_path, "missing stage_00 predictions"))
        new_run = rows_by_unique_query_id(require_file(missing_run_log_path, "missing run log"))
        if sorted(new_predictions) != sorted(expected_missing):
            raise RuntimeError("Missing generation did not produce exactly the remaining query IDs")
        if sorted(new_raw_log) != sorted(expected_missing):
            raise RuntimeError("Missing raw log did not produce exactly the remaining query IDs")
        if sorted(new_run) != sorted(expected_missing):
            raise RuntimeError("Missing run log did not produce exactly the remaining query IDs")
        combined_prediction_by_id = {**partial_prediction_by_id, **new_predictions}
        combined_raw_log_by_id = {**partial_raw_by_id, **new_raw_log}
        combined_run_by_id = {**partial_run_by_id, **new_run}
        if len(combined_prediction_by_id) != len(input_ids):
            raise RuntimeError("Combined raw generation does not contain 55 unique query IDs")
        merged_raw = raw_dir / "stage_00_generated.jsonl"
        merged_raw_log = raw_dir / "raw_generation.jsonl"
        merged_run_log = raw_dir / "run_log.jsonl"
        write_jsonl_atomic(merged_raw, [combined_prediction_by_id[qid] for qid in input_ids])
        write_jsonl_atomic(merged_raw_log, [combined_raw_log_by_id[qid] for qid in input_ids])
        write_jsonl_atomic(merged_run_log, [combined_run_by_id[qid] for qid in input_ids])
        write_json(
            raw_dir / "resume_summary.json",
            {
                "resume_run": str(resume_run),
                "completed_reused": len(completed_ids),
                "new_records_generated": len(expected_missing),
                "api_calls_avoided": len(completed_ids),
                "completed_ids": completed_ids,
                "remaining_ids": expected_missing,
                "partial_raw_sha256": sha256(source_raw),
                "partial_run_log_sha256": sha256(source_run),
                "merged_raw_sha256": sha256(merged_raw),
                "merged_run_log_sha256": sha256(merged_run_log),
                "missing_generation_stdout": str(stdout_path),
                "missing_generation_stderr": str(stderr_path),
                "process": process,
            },
        )
        return merged_raw

    def raw_generation_stage(self) -> Path:
        if self.args.resume_run is None:
            raw_dir = self.work_dir / "raw_full_validation"
            command = self.raw_generation_command(raw_dir, VER2 / "inputs" / "validation_inputs.jsonl")
            return self.command_stage(
                "raw_generation_and_freeform", command, VER2,
                [VER2 / "inputs" / "validation_inputs.jsonl"], raw_dir / "predictions.jsonl", None,
                [VER2 / "run_full_validation.py"],
                heartbeat_files=[
                    raw_dir / "raw_generation.jsonl",
                    raw_dir / "run_log.jsonl",
                    raw_dir / "stage_00_generated.jsonl",
                ],
            )
        resumed = self.stage_gate("raw_generation_and_freeform")
        if resumed is not None:
            return resumed
        started = time.perf_counter()
        started_at = utc_now()
        raw_dir = self.work_dir / "raw_full_validation"
        current = self.merge_partial_raw_generation(self.args.resume_run.resolve(), raw_dir)
        return self.record(
            StageResult(
                name="raw_generation_and_freeform",
                status="EXECUTED",
                input_files=[VER2 / "inputs" / "validation_inputs.jsonl"],
                output_file=current,
                code_files=[Path(__file__), VER2 / "run_full_validation.py"],
                started_at=started_at,
                duration_sec=time.perf_counter() - started,
                warnings=["Resumed interrupted raw generation: reused completed records and generated only missing IDs."],
                command=["resume-run", str(self.args.resume_run)],
                stdout_log=str(self.logs_dir / "raw_generation_resume_missing.stdout.log"),
                stderr_log=str(self.logs_dir / "raw_generation_resume_missing.stderr.log"),
            )
        )

    def run_raw_fresh(self) -> Path:
        current = self.raw_generation_stage()

        option_dir = self.work_dir / "option_aware_mc"
        command = [
            sys.executable, str(UQ / "scripts" / "replay_option_aware_multiple_choice.py"),
            "--input", str(VER2 / "inputs_with_options" / "validation_inputs_with_options.jsonl"),
            "--checkpoint", str(current),
            "--paper-metadata", str(VER2 / "inputs" / "paper_metadata.jsonl"),
            "--object-index", str(UQ / "outputs" / "document_object_index.jsonl"),
            "--output-dir", str(option_dir),
        ]
        current = self.command_stage(
            "option_aware_mc", command, UQ, [current], option_dir / "DEV_BEST_OPTION_AWARE_predictions.jsonl",
            None, [UQ / "scripts" / "replay_option_aware_multiple_choice.py"],
        )

        evidence_dir = self.work_dir / "evidence_safe_cleanup"
        command = [
            sys.executable, str(VER2 / "scripts" / "replay_claim_evidence.py"),
            "--input", str(VER2 / "inputs_with_options" / "validation_inputs_with_options.jsonl"),
            "--parent", str(current),
            "--production-parent", str(VER2 / "checkpoints" / "PRODUCTION_OPTION_AWARE_predictions.jsonl"),
            "--paper-metadata", str(VER2 / "inputs" / "paper_metadata.jsonl"),
            "--object-index", str(VER2 / "artifacts" / "document_object_index_full_docling.jsonl"),
            "--output-dir", str(evidence_dir),
        ]
        current = self.command_stage(
            "evidence_safe_cleanup", command, VER2, [current], evidence_dir / "deterministic_shadow_predictions.jsonl",
            None, [VER2 / "scripts" / "replay_claim_evidence.py"],
        )

        source_dir = self.work_dir / "source_grounded_mc"
        command = [
            sys.executable, str(WORKSPACE / "scripts" / "run_source_grounded_generic.py"),
            "--release-root", str(VER2),
            "--source-root", str(UQ),
            "--parent", str(current),
            "--options", str(VER2 / "inputs_with_options" / "validation_inputs_with_options.jsonl"),
            "--output-dir", str(source_dir),
        ]
        current = self.command_stage(
            "source_grounded_mc", command, WORKSPACE, [current], source_dir / "frozen_predictions.jsonl",
            None, [WORKSPACE / "scripts" / "run_source_grounded_generic.py", VER2 / "src" / "source_grounded_mc_replay.py"],
        )

        typed_dir = self.work_dir / "typed_mc"
        command = [
            sys.executable, str(VER2 / "scripts" / "run_typed_mc_pipeline.py"),
            "--parent", str(current),
            "--semantic-answers", str(source_dir / "semantic_answers.jsonl"),
            "--options", str(VER2 / "inputs_with_options" / "validation_inputs_with_options.jsonl"),
            "--sentence-index", str(VER2 / "artifacts" / "sentence_index.jsonl"),
            "--output-dir", str(typed_dir),
        ]
        current = self.command_stage(
            "typed_mc", command, VER2, [current, source_dir / "semantic_answers.jsonl"], typed_dir / "predictions.jsonl",
            None, [VER2 / "scripts" / "run_typed_mc_pipeline.py"],
        )
        return self.final_stage(current, None)

    def final_stage(self, current: Path, expected_sha: str | None) -> Path:
        resumed = self.stage_gate("final_prediction")
        if resumed is not None:
            return resumed
        started = time.perf_counter()
        started_at = utc_now()
        self.output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(current, self.output)
        if expected_sha is not None:
            self.verify_expected(self.output, expected_sha, "final prediction")
        return self.record(
            StageResult(
                name="final_prediction",
                status="EXECUTED",
                input_files=[current],
                output_file=self.output,
                code_files=[Path(__file__)],
                started_at=started_at,
                duration_sec=time.perf_counter() - started,
            )
        )

    def run(self) -> None:
        self.initialize()
        self.write_run_manifest(status="in_progress")
        try:
            if self.args.mode == "cached-exact":
                final = self.run_cached_exact()
            elif self.args.mode == "production-cached-exact":
                final = self.run_production_exact()
            else:
                final = self.run_raw_fresh()
            self.write_run_manifest(status="completed")
            print(json.dumps({"mode": self.args.mode, "output": str(final), "sha256": sha256(final)}, indent=2))
        except StopPipeline as stopped:
            self.write_run_manifest(status="completed_at_stop_stage")
            print(json.dumps({"mode": self.args.mode, "stopped_at": self.args.stop_stage, "output": str(stopped.output), "sha256": sha256(stopped.output)}, indent=2))
        except Exception as exc:
            self.write_run_manifest(status="failed", error=f"{type(exc).__name__}: {exc}")
            raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Integrated Version 2 reproduction runner")
    parser.add_argument("--mode", required=True, choices=("cached-exact", "production-cached-exact", "raw-fresh"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--start-stage")
    parser.add_argument("--stop-stage")
    parser.add_argument("--resume-manifest", type=Path)
    parser.add_argument("--resume-run", type=Path, help="Resume raw-fresh from an interrupted run directory.")
    parser.add_argument("--release-root", type=Path, help="External Version 2 source release root for fresh/replay stages.")
    parser.add_argument("--source-root", type=Path, help="External option-aware source root for fresh/replay stages.")
    parser.add_argument("--env-file", type=Path, help="Optional public dotenv settings file; credentials resolve centrally and are never read from a project .env.")
    parser.add_argument("--text-model", help="Optional explicit SiliconFlow text-model identifier for this child run; never stored as a credential.")
    parser.add_argument("--assets-manifest", type=Path, default=WORKSPACE / "configs" / "fresh_reproduction_assets.json", help="Hash manifest for raw-fresh external source inputs.")
    parser.add_argument("--preflight", action="store_true", help="Perform a secret-free raw-fresh asset and credential readiness check, then exit.")
    parser.add_argument("--verify-hashes", action="store_true")
    parser.add_argument("--stall-timeout-sec", type=float, default=1200.0)
    parser.add_argument("--heartbeat-interval-sec", type=float, default=30.0)
    args = parser.parse_args()
    valid = Runner.order_for_mode(args.mode)
    for label in ("start_stage", "stop_stage"):
        value = getattr(args, label)
        if value is not None and value not in valid:
            parser.error(f"--{label.replace('_', '-')} must be one of: {', '.join(valid)}")
    if args.start_stage and args.start_stage != valid[0] and args.resume_manifest is None:
        parser.error("A non-initial --start-stage requires --resume-manifest")
    if args.resume_run is not None and args.mode != "raw-fresh":
        parser.error("--resume-run is only valid with --mode raw-fresh")
    return args


def main() -> int:
    args = parse_args()
    configure_external_roots(args)
    if args.preflight:
        if args.mode != "raw-fresh":
            raise ValueError("--preflight is currently defined only for --mode raw-fresh")
        report = raw_fresh_preflight(args.assets_manifest, text_model=args.text_model)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["status"] == "ready" else 2
    Runner(args).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
