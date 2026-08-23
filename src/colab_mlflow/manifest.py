"""Validated, human-readable run manifests stored on Google Drive."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from .filesystem import validate_slug


MANIFEST_SCHEMA_VERSION = "1.0"
FINAL_RUN_STATUSES = {"FINISHED", "FAILED", "KILLED"}


@dataclass(frozen=True)
class RunManifest:
    """One completed MLflow run transported as one metadata document."""

    path: Path
    document: dict[str, Any]

    @property
    def project(self) -> str:
        return self.document["project"]

    @property
    def experiment(self) -> str:
        return self.document["experiment"]

    @property
    def worker(self) -> str:
        return self.document["worker"]

    @property
    def run(self) -> dict[str, Any]:
        return self.document["run"]

    @property
    def run_id(self) -> str:
        return self.run["id"]

    @property
    def artifact_root(self) -> Path:
        return self.path.parent / self.document["artifact_root"]


def discover_run_manifests(
    storage_root: Path, project_slug: str, worker_id: str | None = None
) -> list[RunManifest]:
    """Return every valid completed run for one project, optionally filtered by worker."""

    project_slug = validate_slug(project_slug)
    worker_id = validate_slug(worker_id) if worker_id else None
    project_root = storage_root / "runs" / project_slug
    if not project_root.is_dir():
        return []

    manifests: list[RunManifest] = []
    seen_run_ids: set[str] = set()
    for path in sorted(project_root.glob("*/*/manifest.json")):
        manifest = load_run_manifest(path)
        if manifest.project != project_slug:
            raise ValueError(
                f"Manifest project '{manifest.project}' does not match its Drive folder '{project_slug}': {path}"
            )
        if worker_id and manifest.worker != worker_id:
            continue
        if manifest.run_id in seen_run_ids:
            raise ValueError(f"Duplicate run ID '{manifest.run_id}' in Drive manifests.")
        seen_run_ids.add(manifest.run_id)
        manifests.append(manifest)
    return manifests


def load_run_manifest(path: Path) -> RunManifest:
    """Load one manifest and reject incomplete, unsafe, or unsupported documents."""

    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Run manifest is not valid JSON: {path}") from error
    if not isinstance(document, dict):
        raise ValueError(f"Run manifest must be a JSON object: {path}")
    if document.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported run manifest schema at {path}: {document.get('schema_version')!r}"
        )

    for key in ("project", "experiment", "worker"):
        value = document.get(key)
        if not isinstance(value, str):
            raise ValueError(f"Run manifest field '{key}' must be a slug: {path}")
        validate_slug(value)

    run = document.get("run")
    if not isinstance(run, dict):
        raise ValueError(f"Run manifest field 'run' must be an object: {path}")
    run_id = run.get("id")
    if not isinstance(run_id, str) or len(run_id) != 32 or any(
        character not in "0123456789abcdef" for character in run_id
    ):
        raise ValueError(f"Run ID must be 32 lowercase hexadecimal characters: {path}")
    if path.parent.name != run_id:
        raise ValueError(f"Run ID does not match its Drive folder: {path}")
    if not isinstance(run.get("name"), str) or not run["name"].strip():
        raise ValueError(f"Run name is required: {path}")
    if run.get("status") not in FINAL_RUN_STATUSES:
        raise ValueError(f"Run status must be final ({', '.join(sorted(FINAL_RUN_STATUSES))}): {path}")
    for key in ("start_time", "end_time"):
        if not isinstance(run.get(key), int) or run[key] < 0:
            raise ValueError(f"Run {key} must be a non-negative millisecond timestamp: {path}")
    if run["end_time"] < run["start_time"]:
        raise ValueError(f"Run end_time precedes start_time: {path}")

    for key in ("parameters", "metrics", "metric_history", "tags", "datasets"):
        if not isinstance(document.get(key), dict):
            raise ValueError(f"Run manifest field '{key}' must be an object: {path}")
    mlflow_inputs = document.get("mlflow_inputs")
    if not isinstance(mlflow_inputs, dict):
        raise ValueError(f"Run manifest field 'mlflow_inputs' must be an object: {path}")
    if set(mlflow_inputs) != {"dataset_inputs", "model_inputs"}:
        raise ValueError(
            f"Run manifest mlflow_inputs requires dataset_inputs and model_inputs: {path}"
        )
    if not isinstance(mlflow_inputs["dataset_inputs"], list) or not isinstance(
        mlflow_inputs["model_inputs"], list
    ):
        raise ValueError(f"Run manifest MLflow input collections must be lists: {path}")
    if not isinstance(document.get("summary"), dict):
        raise ValueError(f"Run manifest field 'summary' must be an object: {path}")
    if not isinstance(document.get("artifacts"), list):
        raise ValueError(f"Run manifest field 'artifacts' must be a list: {path}")
    if not isinstance(document.get("logged_models"), list):
        raise ValueError(f"Run manifest field 'logged_models' must be a list: {path}")
    if document.get("artifact_root") != "artifacts":
        raise ValueError(f"Run manifest artifact_root must be 'artifacts': {path}")

    _validate_string_map(document["parameters"], "parameters", path)
    _validate_string_map(document["tags"], "tags", path)
    for key, value in document["metrics"].items():
        _validate_storage_key(key, "metric", path)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"Metric '{key}' must be numeric: {path}")
    for key, history in document["metric_history"].items():
        _validate_storage_key(key, "metric", path)
        if not isinstance(history, list) or not history:
            raise ValueError(f"Metric history '{key}' must be a non-empty list: {path}")
        for point in history:
            if not isinstance(point, dict):
                raise ValueError(f"Metric history '{key}' contains a non-object point: {path}")
            if not isinstance(point.get("value"), (int, float)) or isinstance(point.get("value"), bool):
                raise ValueError(f"Metric history '{key}' has a non-numeric value: {path}")
            if not isinstance(point.get("timestamp"), int) or not isinstance(point.get("step"), int):
                raise ValueError(f"Metric history '{key}' requires integer timestamp and step: {path}")

    for artifact in document["artifacts"]:
        if not isinstance(artifact, dict) or not isinstance(artifact.get("path"), str):
            raise ValueError(f"Every artifact needs a relative path: {path}")
        relative = PurePosixPath(artifact["path"])
        if relative.is_absolute() or ".." in relative.parts or str(relative) in {"", "."}:
            raise ValueError(f"Unsafe artifact path '{artifact['path']}': {path}")
        if not isinstance(artifact.get("size_bytes"), int) or artifact["size_bytes"] < 0:
            raise ValueError(f"Artifact size must be a non-negative integer: {path}")
        checksum = artifact.get("sha256")
        if (
            not isinstance(checksum, str)
            or len(checksum) != 64
            or any(character not in "0123456789abcdef" for character in checksum)
        ):
            raise ValueError(f"Artifact sha256 must contain 64 hexadecimal characters: {path}")

    for dataset_input in mlflow_inputs["dataset_inputs"]:
        if not isinstance(dataset_input, dict) or not isinstance(dataset_input.get("dataset"), dict):
            raise ValueError(f"Every MLflow dataset input needs a dataset object: {path}")
        dataset = dataset_input["dataset"]
        for key in ("name", "digest", "source_type", "source"):
            if not isinstance(dataset.get(key), str):
                raise ValueError(f"MLflow dataset field '{key}' must be a string: {path}")
        for key in ("schema", "profile"):
            if dataset.get(key) is not None and not isinstance(dataset[key], str):
                raise ValueError(f"MLflow dataset field '{key}' must be a string or null: {path}")
        tags = dataset_input.get("tags")
        if not isinstance(tags, dict):
            raise ValueError(f"MLflow dataset input tags must be an object: {path}")
        _validate_string_map(tags, "dataset input tags", path)
    for model_input in mlflow_inputs["model_inputs"]:
        if (
            not isinstance(model_input, dict)
            or not isinstance(model_input.get("model_id"), str)
            or not model_input["model_id"].startswith("m-")
        ):
            raise ValueError(f"Every MLflow model input needs a model_id: {path}")

    for model in document["logged_models"]:
        if not isinstance(model, dict):
            raise ValueError(f"Every logged model must be an object: {path}")
        model_id = model.get("id")
        if not isinstance(model_id, str) or not model_id.startswith("m-"):
            raise ValueError(f"Logged model ID must start with 'm-': {path}")
        if model.get("source_run_id") != run_id:
            raise ValueError(f"Logged model source_run_id must match its run: {path}")
        artifact_path = model.get("artifact_path")
        relative_model_path = PurePosixPath(artifact_path) if isinstance(artifact_path, str) else None
        if (
            relative_model_path is None
            or relative_model_path.is_absolute()
            or ".." in relative_model_path.parts
            or len(relative_model_path.parts) < 2
            or relative_model_path.parts[0] != "logged-models"
        ):
            raise ValueError(f"Logged model artifact_path must be below logged-models/: {path}")
        for key in ("parameters", "tags"):
            if not isinstance(model.get(key), dict):
                raise ValueError(f"Logged model field '{key}' must be an object: {path}")
            _validate_string_map(model[key], f"model {key}", path)
        if not isinstance(model.get("metrics"), list):
            raise ValueError(f"Logged model metrics must be a list: {path}")

    return RunManifest(path=path, document=document)


def _validate_string_map(values: dict[str, object], label: str, path: Path) -> None:
    for key, value in values.items():
        _validate_storage_key(key, label.removesuffix("s"), path)
        if not isinstance(value, str):
            raise ValueError(f"Manifest {label} value for '{key}' must be a string: {path}")


def _validate_storage_key(key: object, label: str, path: Path) -> None:
    if not isinstance(key, str) or not key or "/" in key or "\\" in key or key in {".", ".."}:
        raise ValueError(f"Unsafe {label} key {key!r}: {path}")
