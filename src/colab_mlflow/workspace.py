"""Visible global workspace configuration and Drive workspace setup."""

from __future__ import annotations

import os
import json
import subprocess
import tomllib
from dataclasses import dataclass
from pathlib import Path

from .filesystem import validate_slug, write_text


@dataclass(frozen=True)
class WorkspaceSettings:
    """Drive paths, local SQL state, and default worker for one machine."""

    local_storage_root: Path
    colab_storage_root: str
    worker_id: str
    local_state_root: Path


@dataclass(frozen=True)
class DatasetLocation:
    """One immutable dataset version as seen locally and from Google Colab."""

    local_path: Path
    colab_path: str


def default_global_config_path() -> Path:
    """Output: the visible, user-editable global configuration path."""

    configured = os.environ.get("COLAB_MLFLOW_CONFIG")
    return (
        Path(configured).expanduser()
        if configured
        else Path.home() / ".config" / "colab-mlflow" / "config.toml"
    )


def tool_environment_file() -> Path | None:
    """Output: the editable .env beside a source checkout, when this is one."""

    candidate = Path(__file__).resolve().parents[2] / ".env"
    return candidate if candidate.is_file() else None


def load_default_workspace(config_path: Path | None = None) -> WorkspaceSettings:
    """Load an explicit global config, then a tool .env, then the standard global config."""

    if config_path:
        return load_global_workspace(config_path)
    environment_file = tool_environment_file()
    return load_workspace(environment_file) if environment_file else load_global_workspace()


def active_workspace_source(config_path: Path | None = None) -> tuple[str, Path]:
    """Output: a human-readable source type and the visible file currently in effect."""

    if config_path:
        return "global TOML", config_path.expanduser()
    environment_file = tool_environment_file()
    if environment_file:
        return "tool environment file", environment_file
    return "global TOML", default_global_config_path()


def load_global_workspace(config_path: Path | None = None) -> WorkspaceSettings:
    """Input: optional config location. Output: validated Linux/macOS workspace settings."""

    path = (config_path or default_global_config_path()).expanduser()
    if not path.is_file():
        raise FileNotFoundError(
            f"Global configuration was not found: {path}. "
            "Run 'colab-mlflow config init' once, then edit that visible file when needed."
        )
    document = tomllib.loads(path.read_text(encoding="utf-8"))
    values = {
        "local_root": document.get("drive_storage_root"),
        "colab_root": document.get("colab_storage_root"),
        "worker_id": document.get("default_worker_id", "worker-a"),
        "local_state_root": document.get("local_state_root"),
    }
    if any(not isinstance(value, str) or not value.strip() for value in values.values()):
        raise ValueError(
            f"Global configuration is incomplete: {path}. "
            "Set drive_storage_root, colab_storage_root, default_worker_id, and local_state_root."
        )
    return _workspace_settings(
        local_root=values["local_root"],
        colab_root=values["colab_root"],
        worker_id=values["worker_id"],
        local_state_root=values["local_state_root"],
    )


def create_global_workspace(
    *,
    config_path: Path | None,
    drive_storage_root: Path,
    colab_storage_root: str,
    default_worker_id: str,
    local_state_root: Path | None,
) -> tuple[Path, WorkspaceSettings]:
    """Create one visible global configuration and its required folders without overwrite."""

    path = (config_path or default_global_config_path()).expanduser()
    if path.exists():
        raise FileExistsError(
            f"Global configuration already exists: {path}. Edit it directly instead of overwriting it."
        )
    settings = _workspace_settings(
        local_root=str(drive_storage_root),
        colab_root=colab_storage_root,
        worker_id=default_worker_id,
        local_state_root=str(local_state_root or default_local_state_root()),
    )
    write_text(
        path,
        "\n".join(
            [
                "# colab-mlflow machine configuration",
                "# This file is intentionally visible and editable. It is not committed to projects.",
                'schema_version = "1.0"',
                f"drive_storage_root = {json.dumps(settings.local_storage_root.as_posix())}",
                f"colab_storage_root = {json.dumps(settings.colab_storage_root)}",
                f"default_worker_id = {json.dumps(settings.worker_id)}",
                "# SQLite metadata stays local to this Linux/macOS computer; never put it inside Drive.",
                f"local_state_root = {json.dumps(settings.local_state_root.as_posix())}",
                "",
            ]
        ),
    )
    prepare_workspace(settings)
    return path, settings


def load_workspace(env_file: Path) -> WorkspaceSettings:
    """Input: an environment file. Output: validated settings with process variables taking precedence."""

    if not env_file.is_file():
        raise FileNotFoundError(f"Environment file was not found: {env_file}")
    values = _read_env_file(env_file)
    local_root = _value("COLAB_MLFLOW_LOCAL_STORAGE_ROOT", values)
    colab_root = _value("COLAB_MLFLOW_COLAB_STORAGE_ROOT", values)
    worker_id = _value("COLAB_MLFLOW_WORKER_ID", values)
    local_state_root = _value("COLAB_MLFLOW_LOCAL_STATE_ROOT", values)
    if not local_root or not colab_root or not worker_id:
        raise ValueError(
            "COLAB_MLFLOW_LOCAL_STORAGE_ROOT, COLAB_MLFLOW_COLAB_STORAGE_ROOT, and COLAB_MLFLOW_WORKER_ID are required."
        )
    if not colab_root.startswith("/content/drive/"):
        raise ValueError("COLAB_MLFLOW_COLAB_STORAGE_ROOT must be under /content/drive/.")
    return _workspace_settings(
        local_root=local_root,
        colab_root=colab_root,
        worker_id=worker_id,
        local_state_root=local_state_root or str(default_local_state_root()),
    )


def setup_workspace(env_file: Path) -> WorkspaceSettings:
    """Input: an environment file. Output: ready dataset and compact-run folders."""

    return prepare_workspace(load_workspace(env_file))


def prepare_workspace(settings: WorkspaceSettings) -> WorkspaceSettings:
    """Input: validated settings. Output: ready Drive and local SQLite folders."""

    for relative_path in ("datasets", "runs"):
        (settings.local_storage_root / relative_path).mkdir(parents=True, exist_ok=True)
    (settings.local_state_root / "projects").mkdir(parents=True, exist_ok=True)
    _write_dataset_root_guide(settings)
    return settings


def init_dataset(*, settings: WorkspaceSettings, slug: str, version: str) -> DatasetLocation:
    """Input: a dataset name and immutable version. Output: a ready Drive folder and guide."""

    slug, version = validate_slug(slug), validate_slug(version)
    local_path = settings.local_storage_root / "datasets" / slug / version
    local_path.mkdir(parents=True, exist_ok=True)
    colab_path = f"{settings.colab_storage_root}/datasets/{slug}/{version}"
    guide = local_path / "README.md"
    if not guide.exists():
        write_text(
            guide,
            "\n".join(
                [
                    f"# Dataset: {slug} / {version}",
                    "",
                    "Copy this immutable dataset version into this folder.",
                    "Do not modify files here after a tracked run uses them; create a new version folder instead.",
                    "",
                    f"Colab path: `{colab_path}`",
                    "",
                    "Example notebook input:",
                    "```python",
                    "DATASET_PATHS = {",
                    f'    "train": "{colab_path}/train.csv",',
                    "}",
                    "```",
                    "",
                ]
            ),
        )
    return DatasetLocation(local_path=local_path, colab_path=colab_path)


def load_project_workspace(
    root: Path, worker_id: str | None = None, config_path: Path | None = None
) -> WorkspaceSettings:
    """Input: a project root. Output: machine-local Drive settings without an env argument."""

    global_settings = load_default_workspace(config_path)
    local_root = _git_config(root, "colab-mlflow.local-storage-root") or str(
        global_settings.local_storage_root
    )
    local_state_root = _git_config(root, "colab-mlflow.local-state-root") or str(
        global_settings.local_state_root
    )
    default_worker = (
        worker_id
        or _git_config(root, "colab-mlflow.default-worker")
        or global_settings.worker_id
    )
    configuration = root / ".colab-mlflow.toml"
    if not configuration.is_file():
        raise FileNotFoundError(f"Project configuration was not found: {configuration}")
    document = tomllib.loads(configuration.read_text(encoding="utf-8"))
    storage = document.get("storage")
    if not isinstance(storage, dict) or not isinstance(storage.get("colab_root"), str):
        raise ValueError("Project configuration has no Colab storage root.")
    return _workspace_settings(
        local_root=local_root,
        colab_root=storage["colab_root"],
        worker_id=default_worker,
        local_state_root=local_state_root,
    )


def default_local_state_root() -> Path:
    """Output: local non-Drive storage for persistent SQLite tracking state."""

    data_home = os.environ.get("XDG_DATA_HOME")
    base = Path(data_home).expanduser() if data_home else Path.home() / ".local" / "share"
    return base / "colab-mlflow"


def _workspace_settings(
    *, local_root: str, colab_root: str, worker_id: str, local_state_root: str
) -> WorkspaceSettings:
    """Validate paths shared by legacy environment and global TOML configuration."""

    if not colab_root.startswith("/content/drive/"):
        raise ValueError("colab_storage_root must be under /content/drive/.")
    local_storage_path = Path(local_root).expanduser()
    local_state_path = Path(local_state_root).expanduser()
    if local_state_path.resolve().is_relative_to(local_storage_path.resolve()):
        raise ValueError(
            "local_state_root must be local to this Linux/macOS computer and outside Google Drive storage."
        )
    return WorkspaceSettings(
        local_storage_root=local_storage_path,
        colab_storage_root=colab_root,
        worker_id=validate_slug(worker_id),
        local_state_root=local_state_path,
    )


def _value(name: str, values: dict[str, str | None]) -> str | None:
    """Input: environment variable name and file values. Output: process value or file value."""

    return os.environ.get(name) or values.get(name)


def _read_env_file(path: Path) -> dict[str, str]:
    """Input: a simple dotenv file. Output: key/value pairs without process mutation."""

    values: dict[str, str] = {}
    for number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").lstrip()
        if "=" not in line:
            raise ValueError(f"Invalid environment entry at {path}:{number}")
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if not key:
            raise ValueError(f"Empty environment name at {path}:{number}")
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[key] = value
    return values


def _git_config(root: Path, key: str) -> str | None:
    """Input: a project and local Git config key. Output: value when configured."""

    result = subprocess.run(
        ["git", "-C", str(root), "config", "--local", "--get", key],
        capture_output=True,
        check=False,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _write_dataset_root_guide(settings: WorkspaceSettings) -> None:
    """Input: workspace settings. Output: non-destructive guidance in Drive's dataset root."""

    guide = settings.local_storage_root / "datasets" / "README.md"
    if guide.exists():
        return
    write_text(
        guide,
        "\n".join(
            [
                "# colab-mlflow dataset storage",
                "",
                "Create each immutable dataset version with:",
                "```bash",
                "colab-mlflow dataset init --slug <dataset-name> --version <version>",
                "```",
                "",
                "Then copy source files into `datasets/<dataset-name>/<version>/`.",
                f"Use `{settings.colab_storage_root}/datasets/` in notebook `DATASET_PATHS`.",
                "Never overwrite a version that has already been used in a tracked run.",
                "",
            ]
        ),
    )
