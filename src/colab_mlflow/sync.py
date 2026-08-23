"""Build a verified local MLflow SQLite database from compact Drive manifests."""

from __future__ import annotations

import hashlib
import io
import json
import os
import sqlite3
import tempfile
from contextlib import contextmanager, redirect_stdout
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator

from .filesystem import validate_slug
from .manifest import RunManifest, discover_run_manifests


@dataclass(frozen=True)
class SyncResult:
    """Result of one idempotent Drive-manifest-to-SQLite synchronization."""

    database_path: Path
    changed: bool
    experiments: int
    runs: int
    fingerprint: str


def project_state_directory(
    local_state_root: Path, project_slug: str, worker_id: str | None = None
) -> Path:
    project_slug = validate_slug(project_slug)
    suffix = f"--{validate_slug(worker_id)}" if worker_id else ""
    return local_state_root / "projects" / f"{project_slug}{suffix}"


def project_database_path(
    local_state_root: Path, project_slug: str, worker_id: str | None = None
) -> Path:
    return project_state_directory(local_state_root, project_slug, worker_id) / "mlflow.db"


def sqlite_uri(path: Path) -> str:
    return f"sqlite:///{path.resolve().as_posix()}"


def sync_project(
    *,
    storage_root: Path,
    local_state_root: Path,
    project_slug: str,
    worker_id: str | None = None,
    force: bool = False,
) -> SyncResult:
    """Rebuild this project's SQLite database from immutable completed-run manifests."""

    project_slug = validate_slug(project_slug)
    manifests = discover_run_manifests(storage_root, project_slug, worker_id)
    if not manifests:
        location = storage_root / "runs" / project_slug
        raise FileNotFoundError(f"No completed run manifests were found below: {location}")

    state_directory = project_state_directory(local_state_root, project_slug, worker_id)
    state_directory.mkdir(parents=True, exist_ok=True)
    database_path = state_directory / "mlflow.db"
    state_path = state_directory / "sync-state.json"

    with database_write_lock(state_directory):
        fingerprint = manifest_fingerprint(manifests)
        previous = _read_sync_state(state_path)
        if not force and database_path.is_file() and previous.get("fingerprint") == fingerprint:
            return SyncResult(
                database_path=database_path,
                changed=False,
                experiments=int(previous.get("experiments", 0)),
                runs=int(previous.get("runs", 0)),
                fingerprint=fingerprint,
            )

        with tempfile.TemporaryDirectory(prefix="sync-", dir=state_directory) as temporary:
            temporary_root = Path(temporary)
            snapshot = temporary_root / "mlruns"
            experiments = create_manifest_snapshot(manifests, snapshot)
            candidate = temporary_root / "mlflow.db"
            _migrate_snapshot(snapshot, candidate)
            expected_run_ids = {manifest.run_id for manifest in manifests}
            _verify_database(
                candidate,
                project_slug=project_slug,
                expected_experiments=experiments,
                expected_run_ids=expected_run_ids,
            )
            current = discover_run_manifests(storage_root, project_slug, worker_id)
            if manifest_fingerprint(current) != fingerprint:
                raise RuntimeError(
                    "Drive run manifests changed during synchronization. Wait for publishing to finish, then retry."
                )
            _activate_database(candidate, database_path)

        state = {
            "schema_version": 2,
            "source": "drive-run-manifests",
            "fingerprint": fingerprint,
            "experiments": experiments,
            "runs": len(manifests),
            "synced_at": datetime.now(timezone.utc).isoformat(),
        }
        _write_json_atomic(state_path, state)
        return SyncResult(
            database_path=database_path,
            changed=True,
            experiments=experiments,
            runs=len(manifests),
            fingerprint=fingerprint,
        )


def sync_needed(
    *,
    storage_root: Path,
    local_state_root: Path,
    project_slug: str,
    worker_id: str | None = None,
) -> bool:
    manifests = discover_run_manifests(storage_root, project_slug, worker_id)
    if not manifests:
        return False
    state_directory = project_state_directory(local_state_root, project_slug, worker_id)
    state = _read_sync_state(state_directory / "sync-state.json")
    return (
        not (state_directory / "mlflow.db").is_file()
        or state.get("fingerprint") != manifest_fingerprint(manifests)
    )


def manifest_fingerprint(manifests: list[RunManifest]) -> str:
    """Hash only compact manifest bytes; artifact content remains lazy on Drive."""

    digest = hashlib.sha256()
    for manifest in sorted(manifests, key=lambda item: item.path.as_posix()):
        digest.update(manifest.path.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(manifest.path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def create_manifest_snapshot(manifests: list[RunManifest], target_store: Path) -> int:
    """Expand compact manifests into a temporary local FileStore for MLflow's migrator."""

    grouped: dict[str, list[RunManifest]] = {}
    for manifest in manifests:
        grouped.setdefault(f"{manifest.project}/{manifest.experiment}", []).append(manifest)
    target_store.mkdir(parents=True, exist_ok=True)

    for number, (experiment_name, runs) in enumerate(sorted(grouped.items()), start=1):
        experiment_id = str(number)
        experiment = target_store / experiment_id
        experiment.mkdir()
        creation_time = min(run.run["start_time"] for run in runs)
        artifact_location = runs[0].artifact_root.parent.as_uri()
        _write_text(
            experiment / "meta.yaml",
            "\n".join(
                [
                    f"artifact_location: {json.dumps(artifact_location)}",
                    f"creation_time: {creation_time}",
                    f"experiment_id: {json.dumps(experiment_id)}",
                    f"last_update_time: {max(run.run['end_time'] for run in runs)}",
                    "lifecycle_stage: active",
                    f"name: {json.dumps(experiment_name)}",
                    "",
                ]
            ),
        )
        for manifest in sorted(runs, key=lambda item: (item.run["start_time"], item.run_id)):
            _write_manifest_run(manifest, experiment, experiment_id)
    return len(grouped)


def _write_manifest_run(manifest: RunManifest, experiment: Path, experiment_id: str) -> None:
    document = manifest.document
    run = document["run"]
    run_root = experiment / manifest.run_id
    run_root.mkdir()
    status = {"FINISHED": 3, "FAILED": 4, "KILLED": 5}[run["status"]]
    _write_text(
        run_root / "meta.yaml",
        "\n".join(
            [
                f"artifact_uri: {json.dumps(manifest.artifact_root.as_uri())}",
                f"end_time: {run['end_time']}",
                "entry_point_name: ''",
                f"experiment_id: {json.dumps(experiment_id)}",
                "lifecycle_stage: active",
                f"run_id: {manifest.run_id}",
                f"run_name: {json.dumps(run['name'])}",
                "source_name: ''",
                "source_type: 4",
                "source_version: ''",
                f"start_time: {run['start_time']}",
                f"status: {status}",
                "tags: []",
                f"user_id: {json.dumps(run.get('user_id', 'colab'))}",
                "",
            ]
        ),
    )
    for key, value in sorted(document["parameters"].items()):
        _write_text(run_root / "params" / key, value)
    histories = document["metric_history"]
    for key, value in sorted(document["metrics"].items()):
        points = histories.get(key) or [
            {"timestamp": run["end_time"], "value": value, "step": 0}
        ]
        _write_text(
            run_root / "metrics" / key,
            "".join(
                f"{point['timestamp']} {point['value']} {point['step']}\n" for point in points
            ),
        )
    tags = dict(document["tags"])
    tags.setdefault("mlflow.runName", run["name"])
    for key, value in sorted(tags.items()):
        _write_text(run_root / "tags" / key, value)
    _write_run_inputs(document["mlflow_inputs"], manifest, experiment, run_root)
    for model in document["logged_models"]:
        _write_logged_model(model, manifest, experiment, experiment_id, run_root)


def _write_run_inputs(
    inputs: dict[str, object], manifest: RunManifest, experiment: Path, run_root: Path
) -> None:
    """Recreate native MLflow dataset/model input vertices from the manifest."""

    for item in inputs["dataset_inputs"]:
        dataset_input = dict(item)
        dataset = dict(dataset_input["dataset"])
        dataset_id = hashlib.sha256(
            json.dumps(dataset, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:32]
        dataset_root = experiment / "datasets" / dataset_id
        _write_text(
            dataset_root / "meta.yaml",
            "\n".join(
                [
                    f"digest: {json.dumps(dataset['digest'])}",
                    f"name: {json.dumps(dataset['name'])}",
                    f"profile: {json.dumps(dataset.get('profile'))}",
                    f"schema: {json.dumps(dataset.get('schema'))}",
                    f"source: {json.dumps(dataset['source'])}",
                    f"source_type: {json.dumps(dataset['source_type'])}",
                    "",
                ]
            ),
        )
        tags = dict(dataset_input["tags"])
        relation_id = hashlib.sha256(
            json.dumps(
                [manifest.run_id, dataset_id, tags], sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()[:32]
        tag_lines = ["tags:"] + [
            f"  {json.dumps(key)}: {json.dumps(value)}" for key, value in sorted(tags.items())
        ]
        if not tags:
            tag_lines = ["tags: {}"]
        _write_text(
            run_root / "inputs" / relation_id / "meta.yaml",
            "\n".join(
                [
                    f"destination_id: {dataset_id}",
                    "destination_type: RUN",
                    f"source_id: {dataset_id}",
                    "source_type: DATASET",
                    *tag_lines,
                    "",
                ]
            ),
        )
    for item in inputs["model_inputs"]:
        model_id = str(dict(item)["model_id"])
        relation_id = hashlib.sha256(
            f"{manifest.run_id}\0{model_id}".encode("utf-8")
        ).hexdigest()[:32]
        _write_text(
            run_root / "inputs" / relation_id / "meta.yaml",
            "\n".join(
                [
                    f"destination_id: {manifest.run_id}",
                    "destination_type: RUN",
                    f"source_id: {model_id}",
                    "source_type: MODEL",
                    "tags: {}",
                    "",
                ]
            ),
        )


def _write_logged_model(
    model: dict[str, object],
    manifest: RunManifest,
    experiment: Path,
    experiment_id: str,
    run_root: Path,
) -> None:
    model_id = str(model["id"])
    model_root = experiment / "models" / model_id
    artifact_location = (manifest.artifact_root / str(model["artifact_path"])).as_uri()
    status = {"PENDING": 1, "READY": 2, "FAILED": 3}.get(str(model["status"]), 3)
    _write_text(
        model_root / "meta.yaml",
        "\n".join(
            [
                f"artifact_location: {json.dumps(artifact_location)}",
                f"creation_timestamp: {model['creation_timestamp']}",
                f"experiment_id: {json.dumps(experiment_id)}",
                f"last_updated_timestamp: {model['last_updated_timestamp']}",
                f"model_id: {model_id}",
                f"model_type: {json.dumps(model.get('model_type'))}",
                f"name: {json.dumps(model['name'])}",
                f"source_run_id: {manifest.run_id}",
                f"status: {status}",
                f"status_message: {json.dumps(model.get('status_message'))}",
                "",
            ]
        ),
    )
    for key, value in sorted(dict(model["parameters"]).items()):
        _write_text(model_root / "params" / key, str(value))
    for key, value in sorted(dict(model["tags"]).items()):
        _write_text(model_root / "tags" / key, str(value))
    metric_lines: dict[str, list[str]] = {}
    for metric_value in model["metrics"]:
        metric = dict(metric_value)
        fields = [
            metric["timestamp"],
            metric["value"],
            metric["step"],
            metric.get("run_id") or manifest.run_id,
        ]
        if metric.get("dataset_name") is not None or metric.get("dataset_digest") is not None:
            fields.extend([metric.get("dataset_name") or "", metric.get("dataset_digest") or ""])
        metric_lines.setdefault(str(metric["key"]), []).append(" ".join(map(str, fields)))
    for key, lines in metric_lines.items():
        _write_text(model_root / "metrics" / key, "\n".join(lines) + "\n")
    _write_text(
        run_root / "outputs" / model_id / "meta.yaml",
        "\n".join(
            [
                f"destination_id: {model_id}",
                "destination_type: MODEL_OUTPUT",
                f"source_id: {model_id}",
                "source_type: RUN_OUTPUT",
                "step: 0",
                "tags: {}",
                "",
            ]
        ),
    )


@contextmanager
def database_read_lock(state_directory: Path) -> Generator[None, None, None]:
    with _database_lock(state_directory, exclusive=False):
        yield


@contextmanager
def database_write_lock(state_directory: Path) -> Generator[None, None, None]:
    with _database_lock(state_directory, exclusive=True):
        yield


@contextmanager
def _database_lock(state_directory: Path, *, exclusive: bool) -> Generator[None, None, None]:
    try:
        import fcntl
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("SQLite synchronization requires macOS or Linux file locking.") from error
    state_directory.mkdir(parents=True, exist_ok=True)
    with (state_directory / "database.lock").open("a+", encoding="utf-8") as stream:
        operation = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        try:
            fcntl.flock(stream.fileno(), operation | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                "The local MLflow database is in use. Let 'colab-mlflow server' sync it, or stop the server before running sync manually."
            ) from error
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _migrate_snapshot(snapshot: Path, target_database: Path) -> None:
    try:
        from mlflow.store.fs2db import migrate
    except ImportError as error:
        raise RuntimeError("MLflow 3.15 or newer is required for SQLite synchronization.") from error
    with redirect_stdout(io.StringIO()):
        migrate(snapshot, sqlite_uri(target_database), progress=False)
    with sqlite3.connect(target_database) as connection:
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        result = connection.execute("PRAGMA quick_check").fetchone()
    if not result or result[0] != "ok":
        raise RuntimeError("MLflow created an invalid SQLite database.")


def _verify_database(
    database: Path,
    *,
    project_slug: str,
    expected_experiments: int,
    expected_run_ids: set[str],
) -> None:
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
        experiment_ids = [
            str(row[0])
            for row in connection.execute(
                "SELECT experiment_id FROM experiments WHERE name LIKE ?",
                (f"{project_slug}/%",),
            )
        ]
        if len(experiment_ids) != expected_experiments:
            raise RuntimeError("SQLite verification found a different experiment count than Drive.")
        placeholders = ",".join("?" for _ in experiment_ids)
        actual_run_ids = {
            str(row[0])
            for row in connection.execute(
                f"SELECT run_uuid FROM runs WHERE experiment_id IN ({placeholders})",
                experiment_ids,
            )
        }
    if actual_run_ids != expected_run_ids:
        raise RuntimeError("SQLite verification found different run IDs than Drive manifests.")


def _activate_database(candidate: Path, database: Path) -> None:
    database.parent.mkdir(parents=True, exist_ok=True)
    Path(f"{database}-wal").unlink(missing_ok=True)
    Path(f"{database}-shm").unlink(missing_ok=True)
    os.replace(candidate, database)


def _read_sync_state(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {}
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return document if isinstance(document, dict) else {}


def _write_json_atomic(path: Path, document: dict[str, object]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
