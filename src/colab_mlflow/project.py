"""Source-project initialization and experiment definition services."""

from __future__ import annotations

import json
import subprocess
import tomllib
from pathlib import Path

from .filesystem import validate_slug, write_text
from .models import Experiment, Project
from .templates import agents_md, claude_md, experiment_contract_md, project_workflow_md, skill_md


def init_project(
    *, root: Path, slug: str, name: str, description: str, colab_storage_root: str
) -> Project:
    """Input: source-project identity. Output: project configuration and agent guidance."""

    slug = validate_slug(slug)
    if not name.strip() or not description.strip() or not colab_storage_root.strip():
        raise ValueError("Project name, description, and Colab storage root are required.")
    _ensure_own_git_repository(root)
    configuration = root / ".colab-mlflow.toml"
    write_text(
        configuration,
        "\n".join(
            [
                'schema_version = "0.4.0"',
                f"project_slug = {json.dumps(slug)}",
                f"name = {json.dumps(name)}",
                f"description = {json.dumps(description)}",
                'source_mode = "git"',
                "",
                "[storage]",
                f"colab_root = {json.dumps(colab_storage_root)}",
                "",
            ]
        ),
    )
    _write_managed_guidance(root / "AGENTS.md", agents_md())
    _write_managed_guidance(root / "CLAUDE.md", claude_md())
    _write_managed_guidance(root / "README.md", project_workflow_md())
    write_text(root / "docs/experiment-contract.md", experiment_contract_md())
    write_text(root / ".agents/skills/colab-mlflow-experiment/SKILL.md", skill_md())
    return Project(slug=slug, root=root, configuration=configuration)


def _write_managed_guidance(path: Path, content: str) -> None:
    """Input: an agent instruction path and managed content. Output: merged guidance."""

    start = "<!-- colab-mlflow:start -->"
    finish = "<!-- colab-mlflow:end -->"
    block = f"{start}\n{content.strip()}\n{finish}"
    existing = path.read_text(encoding="utf-8") if path.is_file() else ""
    if start in existing and finish in existing:
        before, remainder = existing.split(start, 1)
        _, after = remainder.split(finish, 1)
        merged = f"{before.rstrip()}\n\n{block}{after}".strip() + "\n"
    elif existing.strip():
        merged = f"{existing.rstrip()}\n\n{block}\n"
    else:
        merged = block + "\n"
    write_text(path, merged)


def load_project_identity(root: Path) -> dict[str, str]:
    """Input: a project root. Output: its validated identity and Drive root."""

    configuration = root / ".colab-mlflow.toml"
    if not configuration.is_file():
        raise FileNotFoundError(f"Project configuration was not found: {configuration}")
    document = tomllib.loads(configuration.read_text(encoding="utf-8"))
    storage = document.get("storage")
    required = ("project_slug", "name", "description")
    if any(not isinstance(document.get(key), str) for key in required):
        raise ValueError("Project configuration is missing identity fields.")
    if not isinstance(storage, dict) or not isinstance(storage.get("colab_root"), str):
        raise ValueError("Project configuration has no Colab storage root.")
    return {
        "slug": document["project_slug"],
        "name": document["name"],
        "description": document["description"],
        "colab_root": storage["colab_root"],
    }


def link_repository(*, root: Path, repository_url: str, branch: str = "main") -> tuple[str, str]:
    """Input: a project and Git remote. Output: the normalized origin URL and branch."""

    if not repository_url.strip():
        raise ValueError("A GitHub repository URL is required.")
    repository_url = repository_url.strip()
    if not (
        repository_url.startswith("https://github.com/")
        or repository_url.startswith("git@github.com:")
    ):
        raise ValueError("Repository must be a GitHub HTTPS or SSH clone URL.")
    if not branch.strip() or any(character.isspace() for character in branch):
        raise ValueError("Git branch must be a non-empty name without spaces.")
    _ensure_own_git_repository(root)
    current = subprocess.run(
        ["git", "-C", str(root), "remote", "get-url", "origin"],
        capture_output=True,
        check=False,
        text=True,
    )
    action = "set-url" if current.returncode == 0 else "add"
    subprocess.run(
        ["git", "-C", str(root), "remote", action, "origin", repository_url],
        check=True,
    )
    subprocess.run(["git", "-C", str(root), "branch", "-M", branch], check=True)
    return repository_url, branch


def repository_metadata(root: Path) -> dict[str, str]:
    """Input: a project root. Output: origin URL and current branch for Colab bootstrap."""

    _ensure_own_git_repository(root)
    origin = subprocess.run(
        ["git", "-C", str(root), "remote", "get-url", "origin"],
        capture_output=True,
        check=False,
        text=True,
    )
    if origin.returncode != 0 or not origin.stdout.strip():
        raise ValueError(
            "Project has no origin remote. Run 'colab-mlflow link --root . --repository <github-url>'."
        )
    branch = subprocess.run(
        ["git", "-C", str(root), "symbolic-ref", "--short", "HEAD"],
        capture_output=True,
        check=False,
        text=True,
    )
    if branch.returncode != 0 or not branch.stdout.strip():
        raise ValueError("Project must be on a named Git branch before generating a notebook.")
    return {"repository_url": origin.stdout.strip(), "branch": branch.stdout.strip()}


def load_experiment_definition(project_root: Path, slug: str) -> dict[str, str]:
    """Input: a project and experiment slug. Output: its stable experiment definition."""

    slug = validate_slug(slug)
    manifest = project_root / "experiments" / slug / "experiment.toml"
    if not manifest.is_file():
        raise FileNotFoundError(f"Experiment was not found: {manifest}")
    document = tomllib.loads(manifest.read_text(encoding="utf-8"))
    required = ("experiment_slug", "type", "objective", "primary_metric")
    if any(not isinstance(document.get(key), str) for key in required):
        raise ValueError(f"Experiment manifest is incomplete: {manifest}")
    return {key: document[key] for key in required}


def load_tracking_contract(project_root: Path, slug: str) -> dict[str, object]:
    """Load optional, user-defined comparison fields without imposing a schema."""

    slug = validate_slug(slug)
    manifest = project_root / "experiments" / slug / "experiment.toml"
    if not manifest.is_file():
        raise FileNotFoundError(f"Experiment was not found: {manifest}")
    document = tomllib.loads(manifest.read_text(encoding="utf-8"))
    contract = document.get("tracking_contract", {})
    if not isinstance(contract, dict):
        raise ValueError(f"tracking_contract must be a TOML table when present: {manifest}")
    return contract


def list_experiment_definitions(project_root: Path) -> dict[str, dict[str, str]]:
    """Input: a project root. Output: all locally defined experiments by slug."""

    experiments_root = project_root / "experiments"
    if not experiments_root.is_dir():
        return {}
    definitions: dict[str, dict[str, str]] = {}
    for manifest in sorted(experiments_root.glob("*/experiment.toml")):
        definition = load_experiment_definition(project_root, manifest.parent.name)
        definitions[definition["experiment_slug"]] = definition
    return definitions


def _ensure_own_git_repository(root: Path) -> None:
    """Input: a project root. Output: a Git repository rooted exactly there."""

    root.mkdir(parents=True, exist_ok=True)
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
            capture_output=True,
            check=False,
            text=True,
        )
    except FileNotFoundError as error:
        raise RuntimeError("Git is required to initialize a tracked project.") from error
    if result.returncode == 0:
        repository_root = Path(result.stdout.strip()).resolve()
        if repository_root != root.resolve():
            raise ValueError(
                f"Project root must be its own Git repository, not a subdirectory of '{repository_root}'."
            )
        return
    subprocess.run(["git", "init", "--quiet", str(root)], check=True)


def create_experiment(
    *, project_root: Path, slug: str, experiment_type: str, objective: str, primary_metric: str
) -> Experiment:
    """Input: a stable experiment definition. Output: manifest and editable pipeline without fixed datasets."""

    slug = validate_slug(slug)
    valid_types = {"training", "evaluation", "inference", "benchmark"}
    if experiment_type not in valid_types:
        raise ValueError(f"Experiment type must be one of: {', '.join(sorted(valid_types))}.")
    if not objective.strip() or not primary_metric.strip():
        raise ValueError("Objective and primary metric are required.")
    root = project_root / "experiments" / slug
    manifest = root / "experiment.toml"
    pipeline = root / "pipeline.yaml"
    if manifest.exists() or pipeline.exists():
        raise FileExistsError(
            f"Experiment '{slug}' already exists. Runs may change parameters only; create a new experiment for a new pipeline."
        )
    write_text(
        manifest,
        "\n".join(
            [
                'schema_version = "0.4.0"',
                f"experiment_slug = {json.dumps(slug)}",
                f"type = {json.dumps(experiment_type)}",
                f"objective = {json.dumps(objective)}",
                f"primary_metric = {json.dumps(primary_metric)}",
                'dataset_policy = "explicit_drive_paths_per_run"',
                "",
                "[tracking_contract]",
                'status = "needs-discussion"',
                "# Add only decisions relevant to this experiment after discussing them.",
                "",
            ]
        ),
    )
    write_text(
        pipeline,
        "\n".join(
            [
                "schema_version: 0.4.0",
                f"experiment: {slug}",
                f"type: {experiment_type}",
                "inputs: explicit_drive_paths_per_run",
                "tracking_contract:",
                "  status: needs-discussion",
                "  # Add stage-specific tracking decisions here when they are agreed.",
                "stages:",
                "  - name: main",
                f"    type: {experiment_type}",
                "",
            ]
        ),
    )
    return Experiment(slug=slug, root=root, manifest=manifest, pipeline=pipeline)
