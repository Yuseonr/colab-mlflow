"""Local MLflow UI backed by periodically synchronized SQLite metadata."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from .filesystem import validate_slug
from .sync import (
    database_read_lock,
    project_state_directory,
    sqlite_uri,
    sync_needed,
    sync_project,
)


def start_server(
    *,
    storage_root: Path,
    local_state_root: Path,
    project_slug: str,
    worker_id: str | None,
    port: int,
    sync_interval: int,
) -> None:
    """Input: Drive and local state. Output: blocking SQL-backed UI with periodic refresh."""

    if not 1 <= port <= 65535:
        raise ValueError("Port must be between 1 and 65535.")
    if sync_interval < 0:
        raise ValueError("Sync interval must be zero or a positive number of seconds.")
    project_slug = validate_slug(project_slug)
    result = sync_project(
        storage_root=storage_root,
        local_state_root=local_state_root,
        project_slug=project_slug,
        worker_id=worker_id,
    )
    _print_sync_result(result.changed, result.experiments, result.runs, result.database_path)
    state_directory = project_state_directory(local_state_root, project_slug, worker_id)
    database_path = result.database_path
    if sync_interval:
        print(f"Periodic Drive sync: every {sync_interval} second(s).")
    else:
        print("Periodic Drive sync: disabled; restart the server or run sync on demand.")
    print(f"MLflow UI: http://127.0.0.1:{port}")

    while True:
        refresh = False
        with database_read_lock(state_directory):
            process = subprocess.Popen(
                _ui_command(database_path=database_path, port=port),
                cwd=state_directory,
            )
            try:
                while True:
                    try:
                        return_code = process.wait(
                            timeout=sync_interval if sync_interval else None
                        )
                    except subprocess.TimeoutExpired:
                        if sync_needed(
                            storage_root=storage_root,
                            local_state_root=local_state_root,
                            project_slug=project_slug,
                            worker_id=worker_id,
                        ):
                            refresh = True
                            print("New Drive tracking metadata detected; refreshing SQLite...")
                            break
                        continue
                    if return_code:
                        raise subprocess.CalledProcessError(
                            return_code, _ui_command(database_path=database_path, port=port)
                        )
                    return
            except KeyboardInterrupt:
                return
            finally:
                _stop_process(process)

        if not refresh:
            return
        result = sync_project(
            storage_root=storage_root,
            local_state_root=local_state_root,
            project_slug=project_slug,
            worker_id=worker_id,
            force=True,
        )
        _print_sync_result(result.changed, result.experiments, result.runs, result.database_path)
        print("Restarting the local MLflow UI; refresh the browser if needed.")


def _ui_command(*, database_path: Path, port: int) -> list[str]:
    """Input: SQLite database and port. Output: localhost-only MLflow UI command."""

    return [
        sys.executable,
        "-m",
        "mlflow",
        "ui",
        "--backend-store-uri",
        sqlite_uri(database_path),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]


def _stop_process(process: subprocess.Popen[bytes]) -> None:
    """Input: UI subprocess. Output: clean termination before database replacement."""

    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def _print_sync_result(changed: bool, experiments: int, runs: int, database_path: Path) -> None:
    """Input: concise sync state. Output: user-facing local database summary."""

    action = "Synchronized" if changed else "Already synchronized"
    print(f"{action}: {experiments} experiment(s), {runs} run(s) -> '{database_path}'.")
