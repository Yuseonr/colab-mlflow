"""Central CLI entry point for colab-mlflow."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from .notebook import create_notebook, github_colab_url
from .project import (
    create_experiment,
    init_project,
    link_repository,
    list_experiment_definitions,
    load_experiment_definition,
    load_tracking_contract,
    load_project_identity,
    repository_metadata,
)
from .server import start_server
from .sync import sync_project
from .tracking import (
    collect_project_runs,
    format_project_status,
    format_run_detail,
    select_run,
)
from .workspace import (
    active_workspace_source,
    create_global_workspace,
    load_default_workspace,
    init_dataset,
    load_project_workspace,
    load_workspace,
    prepare_workspace,
    setup_workspace,
)


def _examples(*commands: str) -> str:
    """Output: consistently formatted copy-paste examples for argparse help."""

    return "Examples:\n" + "\n".join(f"  {command}" for command in commands)


def _command(
    actions: argparse._SubParsersAction,
    name: str,
    *,
    help: str,
    description: str,
    examples: tuple[str, ...],
) -> argparse.ArgumentParser:
    """Output: a subcommand parser with a concise purpose and runnable examples."""

    return actions.add_parser(
        name,
        help=help,
        description=description,
        epilog=_examples(*examples),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )


def build_parser() -> argparse.ArgumentParser:
    """Output: the simple command parser."""

    parser = argparse.ArgumentParser(
        prog="colab-mlflow",
        description="Tracks reproducible Colab experiments through Google Drive and local SQLite.",
        epilog=_examples(
            "colab-mlflow config show",
            "colab-mlflow bootstrap --help",
            "colab-mlflow status",
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    commands = parser.add_subparsers(dest="command", required=True)
    config = _command(
        commands,
        "config",
        help="Shows or creates the visible workspace configuration.",
        description="Shows the active tool configuration or creates the optional TOML fallback.",
        examples=("colab-mlflow config show", "colab-mlflow config init --help"),
    )
    config_actions = config.add_subparsers(dest="config_action", required=True)
    config_init = _command(
        config_actions,
        "init",
        help="Creates fallback TOML configuration and storage folders.",
        description="Use this only when the tool has no persistent source checkout with a .env file.",
        examples=(
            "colab-mlflow config init --drive-storage-root \"/path/to/Drive/colab-mlflow-storage\" \\",
            "  --colab-storage-root /content/drive/MyDrive/colab-mlflow-storage --default-worker-id worker-a \\",
            "  --local-state-root /path/to/colab-mlflow/.state",
        ),
    )
    config_init.add_argument("--config", type=Path, help="Optional alternate visible TOML path.")
    config_init.add_argument("--drive-storage-root", required=True, type=Path, help="Local synced Google Drive folder for datasets and run manifests.")
    config_init.add_argument("--colab-storage-root", required=True, help="The same Drive folder as mounted in Colab, under /content/drive/.")
    config_init.add_argument("--default-worker-id", default="worker-a", help="Default unique label for this Colab runtime (default: worker-a).")
    config_init.add_argument("--local-state-root", type=Path, help="Local folder for SQLite metadata; keep it outside Google Drive.")
    config_show = _command(
        config_actions,
        "show",
        help="Shows the active configuration path and effective values.",
        description="Use this first when a Drive path, worker, or SQLite location is unexpected.",
        examples=("colab-mlflow config show",),
    )
    config_show.add_argument("--config", type=Path, help="Optional alternate visible TOML path.")
    setup = _command(
        commands,
        "setup",
        help="Ensures configured Drive and local state folders exist.",
        description="Run once after editing the tool .env, or again to recreate missing folders.",
        examples=("colab-mlflow setup",),
    )
    setup.add_argument("--config", type=Path, help="Optional alternate visible TOML path.")
    setup.add_argument("--env-file", type=Path, help="Optional .env override; normally the tool checkout .env is used.")
    init = _command(
        commands,
        "init",
        help="Initializes tracking context in an existing source project.",
        description="Creates project identity, contracts, agent guidance, and workflow README content.",
        examples=(
            "colab-mlflow init --project house-pricing --name \"House Pricing\" \\",
            "  --description \"Predict house prices.\"",
        ),
    )
    init.add_argument("--root", default=Path("."), type=Path, help="Project repository directory (default: current directory).")
    init.add_argument("--project", required=True, help="Stable lowercase project slug, for example house-pricing.")
    init.add_argument("--name", required=True, help="Human-readable project name.")
    init.add_argument("--description", required=True, help="One-sentence project goal.")
    init.add_argument("--config", type=Path, help="Optional alternate visible TOML path.")
    init.add_argument("--env-file", type=Path, help="Optional .env override; normally the tool checkout .env is used.")
    init.add_argument("--repository", help="Optional GitHub clone URL; configures origin immediately.")
    init.add_argument("--branch", default="main")
    bootstrap = _command(
        commands,
        "bootstrap",
        help="Creates a project, first experiment, and notebook in one command.",
        description="Use for a new project when the first experiment is already known.",
        examples=(
            "colab-mlflow bootstrap --project house-pricing --name \"House Pricing\" \\",
            "  --description \"Predict house prices.\" --repository https://github.com/<owner>/house-pricing.git \\",
            "  --experiment ridge-baseline --type training --objective \"Train a Ridge baseline.\" \\",
            "  --primary-metric validation.rmse",
        ),
    )
    bootstrap.add_argument("--root", default=Path("."), type=Path, help="Empty project repository directory (default: current directory).")
    bootstrap.add_argument("--project", required=True, help="Stable lowercase project slug.")
    bootstrap.add_argument("--name", required=True, help="Human-readable project name.")
    bootstrap.add_argument("--description", required=True, help="One-sentence project goal.")
    bootstrap.add_argument("--repository", required=True, help="GitHub clone URL for this project.")
    bootstrap.add_argument("--branch", default="main", help="Repository branch that will contain the notebook (default: main).")
    bootstrap.add_argument("--experiment", required=True, help="Stable lowercase slug for the first experiment.")
    bootstrap.add_argument("--type", required=True, dest="experiment_type", help="Experiment category, such as training or evaluation.")
    bootstrap.add_argument("--objective", required=True, help="Specific purpose of this experiment.")
    bootstrap.add_argument("--primary-metric", required=True, help="Metric path used to rank runs, for example validation.rmse.")
    bootstrap.add_argument("--config", type=Path, help="Optional alternate visible TOML path.")
    link = _command(
        commands,
        "link",
        help="Links a project Git repository to its GitHub origin.",
        description="Adds origin when absent or updates it when origin already exists.",
        examples=("colab-mlflow link --repository https://github.com/<owner>/house-pricing.git",),
    )
    link.add_argument("--root", default=Path("."), type=Path, help="Project repository directory (default: current directory).")
    link.add_argument("--repository", required=True, help="GitHub clone URL for this project.")
    link.add_argument("--branch", default="main", help="Repository branch for generated Colab URLs (default: main).")
    dataset = _command(
        commands,
        "dataset",
        help="Manages immutable dataset-version folders on Drive.",
        description="Dataset bytes stay in Drive; runs record a file SHA or directory fingerprint.",
        examples=("colab-mlflow dataset init --slug house-prices --version kaggle-v2",),
    )
    dataset_actions = dataset.add_subparsers(dest="dataset_action", required=True)
    dataset_init = _command(
        dataset_actions,
        "init",
        help="Creates one immutable dataset version folder and README guide.",
        description="Copy files into the printed Drive directory; never overwrite a version used by a run.",
        examples=("colab-mlflow dataset init --slug animal-images --version v1",),
    )
    dataset_init.add_argument("--root", default=Path("."), type=Path, help="Project repository directory (default: current directory).")
    dataset_init.add_argument("--config", type=Path, help="Optional alternate visible TOML path.")
    dataset_init.add_argument("--env-file", type=Path, help="Optional .env override.")
    dataset_init.add_argument("--slug", required=True, help="Stable dataset name, for example house-prices.")
    dataset_init.add_argument("--version", required=True, help="Immutable version label, for example kaggle-v2.")
    experiment = _command(
        commands,
        "experiment",
        help="Creates stable experiment definitions.",
        description="An experiment is a stable scientific/pipeline boundary; parameters vary between runs.",
        examples=("colab-mlflow experiment create --help",),
    )
    experiment_actions = experiment.add_subparsers(dest="experiment_action", required=True)
    create = _command(
        experiment_actions,
        "create",
        help="Creates an experiment definition and editable pipeline contract.",
        description="Use a new experiment for material model, preprocessing, split, or evaluation changes.",
        examples=(
            "colab-mlflow experiment create --slug ridge-baseline --type training \\",
            "  --objective \"Train a Ridge baseline.\" --primary-metric validation.rmse",
        ),
    )
    create.add_argument("--root", default=Path("."), type=Path, help="Project repository directory (default: current directory).")
    create.add_argument("--slug", required=True, help="Stable lowercase experiment slug.")
    create.add_argument("--type", required=True, dest="experiment_type", help="Experiment category, such as training or evaluation.")
    create.add_argument("--objective", required=True, help="Specific purpose of this experiment.")
    create.add_argument("--primary-metric", required=True, help="Metric path used to rank runs, for example validation.rmse.")
    notebook = _command(
        commands,
        "notebook",
        help="Generates standalone Colab notebooks.",
        description="Generation refreshes tracking cells; executing the notebook is what creates a run.",
        examples=("colab-mlflow notebook generate --experiment ridge-baseline",),
    )
    notebook_actions = notebook.add_subparsers(dest="notebook_action", required=True)
    generate = _command(
        notebook_actions,
        "generate",
        help="Creates or refreshes one experiment's standalone Colab notebook.",
        description="Review, commit, and push the generated notebook before opening it in Colab.",
        examples=("colab-mlflow notebook generate --experiment ridge-baseline",),
    )
    generate.add_argument("--root", default=Path("."), type=Path, help="Project repository directory (default: current directory).")
    generate.add_argument("--experiment", required=True, help="Existing experiment slug to generate.")
    generate.add_argument("--target", type=Path, help="Optional notebook output path; default is experiments/<slug>/run.ipynb.")
    status = _command(
        commands,
        "status",
        help="Lists project experiments and completed runs across Drive workers.",
        description="Reads compact manifests only; it does not download model or dataset artifacts.",
        examples=("colab-mlflow status",),
    )
    status.add_argument("--root", default=Path("."), type=Path, help="Project repository directory (default: current directory).")
    status.add_argument("--env-file", type=Path, help="Optional .env override.")
    status.add_argument("--config", type=Path, help="Optional alternate visible TOML path.")
    run = _command(
        commands,
        "run",
        help="Inspects one tracked run.",
        description="Use run show for complete parameters, metrics, source, artifact, and log details.",
        examples=("colab-mlflow run show --experiment ridge-baseline --number 1",),
    )
    run_actions = run.add_subparsers(dest="run_action", required=True)
    show = _command(
        run_actions,
        "show",
        help="Shows datasets, revision, results, logs, and artifacts for one run.",
        description="Use the run number shown by status; this preserves full autologged details.",
        examples=("colab-mlflow run show --experiment ridge-baseline --number 1",),
    )
    show.add_argument("--root", default=Path("."), type=Path, help="Project repository directory (default: current directory).")
    show.add_argument("--env-file", type=Path, help="Optional .env override.")
    show.add_argument("--config", type=Path, help="Optional alternate visible TOML path.")
    show.add_argument("--experiment", required=True, help="Experiment slug shown by status.")
    show.add_argument("--number", required=True, type=int, help="One-based run number shown by status.")
    compare = _command(
        commands,
        "compare",
        help="Shows concise contract-aware run comparison without hiding full data.",
        description="Columns come from tracking_contract comparison fields plus the primary metric.",
        examples=("colab-mlflow compare --experiment ridge-baseline",),
    )
    compare.add_argument("--root", default=Path("."), type=Path, help="Project repository directory (default: current directory).")
    compare.add_argument("--env-file", type=Path, help="Optional .env override.")
    compare.add_argument("--config", type=Path, help="Optional alternate visible TOML path.")
    compare.add_argument("--experiment", required=True, help="Experiment slug to compare.")
    sync = _command(
        commands,
        "sync",
        help="Synchronizes compact Drive run manifests into local SQLite.",
        description="SQLite is local metadata only; models and datasets remain in Drive.",
        examples=("colab-mlflow sync", "colab-mlflow sync --force"),
    )
    sync.add_argument("--root", default=Path("."), type=Path, help="Project repository directory (default: current directory).")
    sync.add_argument("--env-file", type=Path, help="Optional .env override.")
    sync.add_argument("--config", type=Path, help="Optional alternate visible TOML path.")
    sync.add_argument("--worker", help="Optional: synchronize only one worker into an isolated database.")
    sync.add_argument("--force", action="store_true", help="Rebuild SQLite even when Drive metadata is unchanged.")
    server = _command(
        commands,
        "server",
        help="Starts the SQL-backed MLflow UI and periodically syncs Drive metadata.",
        description="Do not run manual sync while this server is active; both protect SQLite with a lock.",
        examples=("colab-mlflow server --port 5000 --sync-interval 300",),
    )
    server.add_argument("--root", default=Path("."), type=Path, help="Project repository directory (default: current directory).")
    server.add_argument("--env-file", type=Path, help="Optional .env override.")
    server.add_argument("--config", type=Path, help="Optional alternate visible TOML path.")
    server.add_argument("--worker", help="Optional: show only one worker instead of all workers.")
    server.add_argument("--port", type=int, default=5000, help="Local MLflow UI port (default: 5000).")
    server.add_argument(
        "--sync-interval",
        type=int,
        default=300,
        metavar="SECONDS",
        help="Drive polling interval; use 0 to sync only at startup (default: 300).",
    )
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    """Input: CLI arguments. Output: process exit code."""

    namespace = build_parser().parse_args(arguments)
    if namespace.command == "config" and namespace.config_action == "init":
        path, settings = create_global_workspace(
            config_path=namespace.config,
            drive_storage_root=namespace.drive_storage_root,
            colab_storage_root=namespace.colab_storage_root,
            default_worker_id=namespace.default_worker_id,
            local_state_root=namespace.local_state_root,
        )
        print(f"Global configuration was created at '{path}'.")
        print("Edit this TOML file directly whenever your Drive or local state locations change.")
        print(f"Drive manifest workspace is ready at '{settings.local_storage_root}'.")
        print(f"Local SQLite state root: '{settings.local_state_root}'.")
        return 0
    if namespace.command == "config" and namespace.config_action == "show":
        source_type, path = active_workspace_source(namespace.config)
        settings = load_default_workspace(namespace.config)
        print(f"Active workspace source ({source_type}): '{path}'")
        print("Edit this file directly to change these defaults.")
        print(f"Drive storage root: '{settings.local_storage_root}'")
        print(f"Colab storage root: '{settings.colab_storage_root}'")
        print(f"Default worker ID: '{settings.worker_id}'")
        print(f"Local SQLite state root: '{settings.local_state_root}'")
        return 0
    if namespace.command == "setup":
        settings = setup_workspace(namespace.env_file) if namespace.env_file else prepare_workspace(
            load_default_workspace(namespace.config)
        )
        print(f"Drive manifest workspace is ready at '{settings.local_storage_root}'.")
        print(f"Dataset root: '{settings.local_storage_root / 'datasets'}'.")
        print(f"Run root: '{settings.local_storage_root / 'runs'}'.")
        print(f"Local SQLite state root: '{settings.local_state_root}'.")
        return 0
    if namespace.command == "init":
        settings = _workspace_for_new_project(namespace)
        project = init_project(root=namespace.root, slug=namespace.project, name=namespace.name, description=namespace.description, colab_storage_root=settings.colab_storage_root)
        if namespace.repository:
            link_repository(
                root=namespace.root,
                repository_url=namespace.repository,
                branch=namespace.branch,
            )
        print(f"Project '{project.slug}' is ready.")
        return 0
    if namespace.command == "bootstrap":
        settings = load_default_workspace(namespace.config)
        prepare_workspace(settings)
        project = init_project(
            root=namespace.root,
            slug=namespace.project,
            name=namespace.name,
            description=namespace.description,
            colab_storage_root=settings.colab_storage_root,
        )
        link_repository(
            root=namespace.root, repository_url=namespace.repository, branch=namespace.branch
        )
        experiment = create_experiment(
            project_root=namespace.root,
            slug=namespace.experiment,
            experiment_type=namespace.experiment_type,
            objective=namespace.objective,
            primary_metric=namespace.primary_metric,
        )
        target = namespace.root / "experiments" / experiment.slug / "run.ipynb"
        create_notebook(
            target=target,
            project=project.slug,
            experiment=experiment.slug,
            experiment_type=namespace.experiment_type,
            objective=namespace.objective,
            primary_metric=namespace.primary_metric,
            colab_storage_root=settings.colab_storage_root,
            repository_url=namespace.repository,
            repository_branch=namespace.branch,
            project_root=namespace.root,
        )
        print(f"Project '{project.slug}', experiment '{experiment.slug}', and notebook are ready.")
        print(f"Notebook: '{target}'.")
        colab_url = github_colab_url(namespace.repository, namespace.branch, target.relative_to(namespace.root))
        if colab_url:
            print(f"After commit and push, open it in Colab: {colab_url}")
        return 0
    if namespace.command == "link":
        repository, branch = link_repository(
            root=namespace.root,
            repository_url=namespace.repository,
            branch=namespace.branch,
        )
        print(f"Project is linked to '{repository}' on branch '{branch}'.")
        return 0
    if namespace.command == "dataset" and namespace.dataset_action == "init":
        settings = (
            load_workspace(namespace.env_file)
            if namespace.env_file
            else load_project_workspace(namespace.root, config_path=namespace.config)
        )
        dataset = init_dataset(settings=settings, slug=namespace.slug, version=namespace.version)
        print(f"Dataset folder is ready at '{dataset.local_path}'.")
        print(f"Copy files there, then use this Colab path: '{dataset.colab_path}'.")
        return 0
    if namespace.command == "experiment" and namespace.experiment_action == "create":
        experiment = create_experiment(project_root=namespace.root, slug=namespace.slug, experiment_type=namespace.experiment_type, objective=namespace.objective, primary_metric=namespace.primary_metric)
        print(f"Experiment '{experiment.slug}' was created.")
        return 0
    if namespace.command == "notebook" and namespace.notebook_action == "generate":
        project = load_project_identity(namespace.root)
        experiment = load_experiment_definition(namespace.root, namespace.experiment)
        repository = repository_metadata(namespace.root)
        target = namespace.target or namespace.root / "experiments" / namespace.experiment / "run.ipynb"
        create_notebook(
            target=target,
            project=project["slug"],
            experiment=experiment["experiment_slug"],
            experiment_type=experiment["type"],
            objective=experiment["objective"],
            primary_metric=experiment["primary_metric"],
            colab_storage_root=project["colab_root"],
            repository_url=repository["repository_url"],
            repository_branch=repository["branch"],
            project_root=namespace.root,
        )
        print(f"Notebook was created at '{target}'.")
        colab_url = github_colab_url(
            repository["repository_url"],
            repository["branch"],
            target.resolve().relative_to(namespace.root.resolve()),
        )
        if colab_url:
            print(f"After commit and push, open it in Colab: {colab_url}")
        return 0
    if namespace.command == "status":
        project = load_project_identity(namespace.root)
        settings = (
            load_workspace(namespace.env_file)
            if namespace.env_file
            else load_project_workspace(namespace.root, config_path=namespace.config)
        )
        records = collect_project_runs(
            storage_root=settings.local_storage_root, project_slug=project["slug"]
        )
        print(
            format_project_status(
                project["slug"], records, list_experiment_definitions(namespace.root)
            )
        )
        return 0
    if namespace.command == "run" and namespace.run_action == "show":
        if namespace.number < 1:
            raise ValueError("Run number must be at least 1.")
        project = load_project_identity(namespace.root)
        settings = (
            load_workspace(namespace.env_file)
            if namespace.env_file
            else load_project_workspace(namespace.root, config_path=namespace.config)
        )
        records = collect_project_runs(
            storage_root=settings.local_storage_root,
            project_slug=project["slug"],
            include_artifacts=True,
        )
        record = select_run(records, experiment=namespace.experiment, number=namespace.number)
        print(format_run_detail(project["slug"], record))
        return 0
    if namespace.command == "compare":
        project = load_project_identity(namespace.root)
        settings = (
            load_workspace(namespace.env_file)
            if namespace.env_file
            else load_project_workspace(namespace.root, config_path=namespace.config)
        )
        definition = load_experiment_definition(namespace.root, namespace.experiment)
        contract = load_tracking_contract(namespace.root, namespace.experiment)
        records = collect_project_runs(
            storage_root=settings.local_storage_root, project_slug=project["slug"]
        )
        from .tracking import format_experiment_comparison

        print(
            format_experiment_comparison(
                project_slug=project["slug"],
                experiment=namespace.experiment,
                records=records,
                primary_metric=definition["primary_metric"],
                contract=contract,
            )
        )
        return 0
    if namespace.command == "sync":
        project = load_project_identity(namespace.root)
        settings = (
            load_workspace(namespace.env_file)
            if namespace.env_file
            else load_project_workspace(namespace.root, config_path=namespace.config)
        )
        result = sync_project(
            storage_root=settings.local_storage_root,
            local_state_root=settings.local_state_root,
            project_slug=project["slug"],
            worker_id=namespace.worker,
            force=namespace.force,
        )
        action = "Synchronized" if result.changed else "Already synchronized"
        print(
            f"{action}: {result.experiments} experiment(s), {result.runs} run(s) -> '{result.database_path}'."
        )
        return 0
    if namespace.command == "server":
        project = load_project_identity(namespace.root)
        settings = (
            load_workspace(namespace.env_file)
            if namespace.env_file
            else load_project_workspace(namespace.root, config_path=namespace.config)
        )
        start_server(
            storage_root=settings.local_storage_root,
            local_state_root=settings.local_state_root,
            project_slug=project["slug"],
            worker_id=namespace.worker,
            port=namespace.port,
            sync_interval=namespace.sync_interval,
        )
        return 0
    raise RuntimeError("Unknown command.")


def _workspace_for_new_project(namespace: argparse.Namespace):
    """Load visible global defaults, while accepting the legacy .env migration path."""

    return (
        load_workspace(namespace.env_file)
        if namespace.env_file
        else load_default_workspace(namespace.config)
    )


if __name__ == "__main__":
    raise SystemExit(main())
