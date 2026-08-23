from __future__ import annotations

import tempfile
import subprocess
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from colab_mlflow.server import _ui_command, start_server


class FinishedProcess:
    def wait(self, timeout: int | None = None) -> int:
        return 0

    def poll(self) -> int:
        return 0

    def terminate(self) -> None:
        raise AssertionError("A finished process should not be terminated.")


class RefreshingProcess:
    def __init__(self) -> None:
        self.running = True

    def wait(self, timeout: int | None = None) -> int:
        if self.running and timeout == 10:
            self.running = False
            return 0
        if self.running:
            raise subprocess.TimeoutExpired("mlflow", timeout)
        return 0

    def poll(self) -> int | None:
        return None if self.running else 0

    def terminate(self) -> None:
        self.running = False


class CompletingProcess(FinishedProcess):
    pass


class ServerTest(unittest.TestCase):
    def test_local_ui_uses_persistent_sqlite_and_startup_sync(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state = root / "state"
            database = state / "projects/dogs-vs-cats/mlflow.db"
            database.parent.mkdir(parents=True)
            database.touch()
            result = SimpleNamespace(
                changed=True,
                experiments=1,
                runs=2,
                database_path=database,
            )
            with patch("colab_mlflow.server.sync_project", return_value=result) as sync, patch(
                "colab_mlflow.server.subprocess.Popen", return_value=FinishedProcess()
            ) as popen:
                start_server(
                    storage_root=root / "drive",
                    local_state_root=state,
                    project_slug="dogs-vs-cats",
                    worker_id=None,
                    port=5000,
                    sync_interval=0,
                )

            sync.assert_called_once()
            command = popen.call_args.args[0]
            self.assertEqual(command, _ui_command(database_path=database, port=5000))
            self.assertIn("sqlite:", command[5])
            self.assertNotIn("MLFLOW_ALLOW_FILE_STORE", popen.call_args.kwargs)
            self.assertEqual(popen.call_args.kwargs["cwd"], database.parent)

    def test_periodic_change_stops_ui_syncs_and_restarts_it(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state = root / "state"
            database = state / "projects/dogs-vs-cats/mlflow.db"
            database.parent.mkdir(parents=True)
            database.touch()
            initial = SimpleNamespace(changed=False, experiments=1, runs=1, database_path=database)
            refreshed = SimpleNamespace(changed=True, experiments=1, runs=2, database_path=database)
            with patch("colab_mlflow.server.sync_project", side_effect=[initial, refreshed]) as sync, patch(
                "colab_mlflow.server.sync_needed", return_value=True
            ), patch(
                "colab_mlflow.server.subprocess.Popen",
                side_effect=[RefreshingProcess(), CompletingProcess()],
            ) as popen:
                start_server(
                    storage_root=root / "drive",
                    local_state_root=state,
                    project_slug="dogs-vs-cats",
                    worker_id=None,
                    port=5000,
                    sync_interval=60,
                )

            self.assertEqual(sync.call_count, 2)
            self.assertTrue(sync.call_args_list[1].kwargs["force"])
            self.assertEqual(popen.call_count, 2)


if __name__ == "__main__":
    unittest.main()
