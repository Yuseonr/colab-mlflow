from __future__ import annotations

import json
import io
import subprocess
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from colab_mlflow.cli import main
from colab_mlflow.notebook import github_colab_url
from colab_mlflow.workspace import load_default_workspace


class CliTest(unittest.TestCase):
    def test_every_command_help_includes_examples(self) -> None:
        command_paths = (
            (),
            ("config",),
            ("config", "init"),
            ("config", "show"),
            ("setup",),
            ("init",),
            ("bootstrap",),
            ("link",),
            ("dataset",),
            ("dataset", "init"),
            ("experiment",),
            ("experiment", "create"),
            ("notebook",),
            ("notebook", "generate"),
            ("notebook", "colab"),
            ("status",),
            ("run",),
            ("run", "show"),
            ("run", "colab"),
            ("compare",),
            ("sync",),
            ("server",),
        )

        for path in command_paths:
            with self.subTest(command=" ".join(path) or "root"):
                output = io.StringIO()
                with redirect_stdout(output), self.assertRaises(SystemExit) as raised:
                    main([*path, "--help"])

                self.assertEqual(raised.exception.code, 0)
                self.assertIn("Examples:", output.getvalue())

    def test_setup_rejects_sqlite_state_inside_google_drive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            drive = root / "drive"
            env_file = root / ".env"
            env_file.write_text(
                "\n".join(
                    [
                        f"COLAB_MLFLOW_LOCAL_STORAGE_ROOT={drive}",
                        "COLAB_MLFLOW_COLAB_STORAGE_ROOT=/content/drive/MyDrive/colab-mlflow",
                        "COLAB_MLFLOW_WORKER_ID=worker-a",
                        f"COLAB_MLFLOW_LOCAL_STATE_ROOT={drive / 'sqlite'}",
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "outside Google Drive"):
                main(["setup", "--env-file", str(env_file)])

    def test_github_notebook_gets_a_direct_colab_link(self) -> None:
        url = github_colab_url(
            "https://github.com/example/dogs-vs-cats.git",
            "main",
            Path("experiments/multihead/run.ipynb"),
        )
        self.assertEqual(
            url,
            "https://colab.research.google.com/github/example/dogs-vs-cats/blob/main/experiments/multihead/run.ipynb",
        )

    def test_notebook_colab_prints_current_branch_url_without_writing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            experiment_root = root / "experiments" / "multihead"
            experiment_root.mkdir(parents=True)
            (experiment_root / "experiment.toml").write_text(
                "\n".join(
                    [
                        'experiment_slug = "multihead"',
                        'type = "training"',
                        'objective = "Train a model."',
                        'primary_metric = "validation.accuracy"',
                    ]
                ),
                encoding="utf-8",
            )
            notebook = experiment_root / "run.ipynb"
            notebook.write_text("{}\n", encoding="utf-8")
            output = io.StringIO()
            with patch(
                "colab_mlflow.cli.repository_metadata",
                return_value={"repository_url": "https://github.com/example/project.git", "branch": "main"},
            ), redirect_stdout(output):
                self.assertEqual(
                    main(["notebook", "colab", "--root", str(root), "--experiment", "multihead"]),
                    0,
                )
            self.assertEqual(
                output.getvalue(),
                "https://colab.research.google.com/github/example/project/blob/main/experiments/multihead/run.ipynb\n",
            )
            self.assertEqual(notebook.read_text(encoding="utf-8"), "{}\n")

    def test_setup_reads_environment_file_and_creates_worker_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            env_file = root / ".env"
            storage_root = root / "drive" / "colab-mlflow"
            env_file.write_text(
                "\n".join(
                    [
                        f"COLAB_MLFLOW_LOCAL_STORAGE_ROOT={storage_root}",
                        "COLAB_MLFLOW_COLAB_STORAGE_ROOT=/content/drive/MyDrive/colab-mlflow",
                        "COLAB_MLFLOW_WORKER_ID=worker-a",
                        f"COLAB_MLFLOW_LOCAL_STATE_ROOT={root / 'state'}",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            exit_code = main(["setup", "--env-file", str(env_file)])

            self.assertEqual(exit_code, 0)
            self.assertTrue((storage_root / "runs").is_dir())
            self.assertFalse((storage_root / "tracking").exists())
            self.assertTrue((root / "state/projects").is_dir())
            dataset_guide = storage_root / "datasets" / "README.md"
            self.assertTrue(dataset_guide.is_file())
            self.assertIn("dataset init", dataset_guide.read_text(encoding="utf-8"))
            self.assertFalse((storage_root / "artifacts").exists())

    def test_dataset_init_creates_version_folder_and_colab_guide(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            env_file = root / ".env"
            storage_root = root / "drive"
            env_file.write_text(
                "\n".join(
                    [
                        f"COLAB_MLFLOW_LOCAL_STORAGE_ROOT={storage_root}",
                        "COLAB_MLFLOW_COLAB_STORAGE_ROOT=/content/drive/MyDrive/colab-mlflow-storage",
                        "COLAB_MLFLOW_WORKER_ID=worker-a",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(
                    main(
                        [
                            "dataset", "init", "--env-file", str(env_file),
                            "--slug", "house-prices", "--version", "kaggle-v1",
                        ]
                    ),
                    0,
                )
            guide = storage_root / "datasets" / "house-prices" / "kaggle-v1" / "README.md"
            self.assertTrue(guide.is_file())
            self.assertIn("train.csv", guide.read_text(encoding="utf-8"))
            self.assertIn("/content/drive/MyDrive/colab-mlflow-storage/datasets/house-prices/kaggle-v1", output.getvalue())

    def test_project_experiment_and_notebook_need_no_dataset_registration(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "AGENTS.md").write_text("# Existing project rules\n", encoding="utf-8")
            env_file = root / ".env"
            env_file.write_text(
                "\n".join(
                    [
                        f"COLAB_MLFLOW_LOCAL_STORAGE_ROOT={root / 'drive'}",
                        "COLAB_MLFLOW_COLAB_STORAGE_ROOT=/content/drive/MyDrive/colab-mlflow",
                        "COLAB_MLFLOW_WORKER_ID=worker-a",
                        f"COLAB_MLFLOW_LOCAL_STATE_ROOT={root / 'state'}",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            main(["setup", "--env-file", str(env_file)])

            init_code = main(
                [
                    "init", "--root", str(root), "--project", "dogs-vs-cats",
                    "--name", "Dogs vs Cats", "--description", "Image classification.",
                    "--env-file", str(env_file),
                    "--repository", "https://github.com/example/dogs-vs-cats.git",
                ]
            )
            experiment_code = main(
                [
                    "experiment", "create", "--root", str(root),
                    "--slug", "resnet-transfer", "--type", "training",
                    "--objective", "Compare ResNet configurations.",
                    "--primary-metric", "validation.accuracy",
                ]
            )
            notebook_code = main(
                [
                    "notebook", "generate", "--root", str(root),
                    "--experiment", "resnet-transfer",
                ]
            )

            self.assertEqual(init_code, 0)
            self.assertEqual(experiment_code, 0)
            self.assertEqual(notebook_code, 0)
            self.assertTrue((root / ".git").is_dir())
            saved_local_root = subprocess.run(
                ["git", "-C", str(root), "config", "--local", "--get", "colab-mlflow.local-storage-root"],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(saved_local_root.returncode, 0)
            with patch("colab_mlflow.cli.collect_project_runs", return_value=[]), redirect_stdout(io.StringIO()):
                self.assertEqual(main(["status", "--root", str(root), "--env-file", str(env_file)]), 0)
            agent_guidance = (root / "AGENTS.md").read_text(encoding="utf-8")
            self.assertIn("# Existing project rules", agent_guidance)
            self.assertIn("Experiment collaboration context", agent_guidance)
            self.assertIn("Contract-first checkpoint", agent_guidance)
            self.assertIn("do not silently change an agreed contract", agent_guidance)
            self.assertIn("parameter-only trials", agent_guidance)
            project_readme = (root / "README.md").read_text(encoding="utf-8")
            self.assertIn("## colab-mlflow workflow", project_readme)
            self.assertIn("colab-mlflow bootstrap", project_readme)
            self.assertIn("colab-mlflow experiment create", project_readme)
            self.assertIn("colab-mlflow notebook generate", project_readme)
            experiment_contract = (root / "docs/experiment-contract.md").read_text(encoding="utf-8")
            self.assertIn("one Git repository and broad ML objective", experiment_contract)
            self.assertIn("It is run metadata, not another project, experiment", experiment_contract)
            self.assertIn("parameter or seed changes within the same pipeline", experiment_contract)
            self.assertIn("Contract lifecycle", experiment_contract)
            experiment_definition = (root / "experiments/resnet-transfer/experiment.toml").read_text(encoding="utf-8")
            self.assertIn('[tracking_contract]', experiment_definition)
            self.assertIn('status = "needs-discussion"', experiment_definition)
            notebook = json.loads((root / "experiments/resnet-transfer/run.ipynb").read_text(encoding="utf-8"))
            source = "".join(line for cell in notebook["cells"] for line in cell.get("source", []))
            run_source = "".join(
                "".join(cell["source"])
                for cell in notebook["cells"]
                if "cmf-run" in cell.get("metadata", {}).get("tags", [])
            )
            compile(run_source, "generated-run-cell", "exec")
            for cell in notebook["cells"]:
                if cell["cell_type"] == "code":
                    compile("".join(cell["source"]), "generated-notebook-cell", "exec")
            self.assertIn("DATASET_PATHS", source)
            self.assertIn("WORKER_ID", source)
            self.assertIn("RUN_LABEL", source)
            self.assertIn("RUN_VARIANTS", source)
            self.assertIn("RUN_VARIANTS.get(WORKER_ID", source)
            self.assertIn("SOURCE_COMMIT", source)
            self.assertIn('"git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"', source)
            self.assertIn("import mlflow", source)
            self.assertIn("mlflow.start_run", source)
            self.assertIn("mlflow.autolog", source)
            self.assertIn("experiment.contract_sha256", source)
            self.assertIn('"schema_version": "1.0"', source)
            self.assertIn('DRIVE_ROOT / "runs" / PROJECT_SLUG', source)
            self.assertNotIn('DRIVE_ROOT / "tracking"', source)
            self.assertIn("logs/stdout.log", source)
            self.assertIn("logs/stderr.log", source)
            self.assertIn("source/environment.txt", source)
            self.assertIn("results/summary.json", source)
            self.assertIn("https://github.com/example/dogs-vs-cats.git", source)
            self.assertIn("/content/drive/MyDrive/colab-mlflow", source)
            self.assertNotIn("colab_mlflow", source)
            self.assertNotIn("download_datasets_to_colab", source)
            self.assertNotIn("dataset://", source)
            self.assertIn("generation_id", notebook["metadata"]["cmf"])
            self.assertIn("cmf:user-code", source)
            self.assertIn("DATASET_PATHS", "".join(notebook["cells"][1]["source"]))
            self.assertNotIn("NOTEBOOK_PATH = Path(\"/content/drive", source)

            old_generation = notebook["metadata"]["cmf"]["generation_id"]
            editable = next(
                cell
                for cell in notebook["cells"]
                if "cmf-user-code" in cell.get("metadata", {}).get("tags", [])
            )
            editable["source"] = ["def run_pipeline(dataset_paths, parameters, output_dir):\n", "    return {'metrics': {'accuracy': 1.0}}\n"]
            parameters = next(
                cell
                for cell in notebook["cells"]
                if "cmf-parameters" in cell.get("metadata", {}).get("tags", [])
            )
            parameters["source"] = [
                "RUN_VARIANTS = {'worker-a': {'learning_rate': 0.001}}\n"
            ]
            inputs = next(
                cell
                for cell in notebook["cells"]
                if "cmf-inputs" in cell.get("metadata", {}).get("tags", [])
            )
            inputs["source"] = [
                "WORKER_ID = 'worker-a'\n",
                "DATASET_PATHS = {'train': '/content/drive/MyDrive/data/train-v1'}\n",
            ]
            (root / "experiments/resnet-transfer/run.ipynb").write_text(
                json.dumps(notebook), encoding="utf-8"
            )
            main(["notebook", "generate", "--root", str(root), "--experiment", "resnet-transfer"])
            regenerated = json.loads(
                (root / "experiments/resnet-transfer/run.ipynb").read_text(encoding="utf-8")
            )
            regenerated_source = "".join(
                "".join(cell["source"])
                for cell in regenerated["cells"]
                if "cmf-user-code" in cell.get("metadata", {}).get("tags", [])
            )
            regenerated_parameters = "".join(
                "".join(cell["source"])
                for cell in regenerated["cells"]
                if "cmf-parameters" in cell.get("metadata", {}).get("tags", [])
            )
            regenerated_inputs = "".join(
                "".join(cell["source"])
                for cell in regenerated["cells"]
                if "cmf-inputs" in cell.get("metadata", {}).get("tags", [])
            )
            self.assertIn("return {'metrics': {'accuracy': 1.0}}", regenerated_source)
            self.assertIn("'learning_rate': 0.001", regenerated_parameters)
            self.assertIn("TRACKING_OPTIONS", regenerated_parameters)
            self.assertIn("/content/drive/MyDrive/data/train-v1", regenerated_inputs)
            self.assertIn("RUN_LABEL", regenerated_inputs)
            self.assertIn("DATASET_CACHE_MODE", regenerated_inputs)
            self.assertNotEqual(old_generation, regenerated["metadata"]["cmf"]["generation_id"])

    def test_visible_global_config_and_bootstrap_need_no_project_env_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = root / "visible-config" / "config.toml"
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(
                    main(
                        [
                            "config", "init", "--config", str(config_path),
                            "--drive-storage-root", str(root / "drive"),
                            "--colab-storage-root", "/content/drive/MyDrive/colab-mlflow-storage",
                            "--default-worker-id", "worker-a",
                            "--local-state-root", str(root / "local-state"),
                        ]
                    ),
                    0,
                )
                self.assertEqual(main(["config", "show", "--config", str(config_path)]), 0)
            config_text = config_path.read_text(encoding="utf-8")
            self.assertIn("intentionally visible and editable", config_text)
            self.assertIn("drive_storage_root", config_text)
            self.assertTrue((root / "drive" / "datasets").is_dir())
            self.assertTrue((root / "local-state" / "projects").is_dir())

            project_root = root / "house-pricing"
            with redirect_stdout(io.StringIO()):
                self.assertEqual(
                    main(
                        [
                            "bootstrap", "--config", str(config_path),
                            "--root", str(project_root), "--project", "house-pricing",
                            "--name", "House pricing", "--description", "Compare house-price models.",
                            "--repository", "https://github.com/example/house-pricing.git",
                            "--experiment", "ridge-baseline", "--type", "training",
                            "--objective", "Predict house prices.",
                            "--primary-metric", "validation.rmse",
                        ]
                    ),
                    0,
                )
                with patch("colab_mlflow.cli.collect_project_runs", return_value=[]):
                    self.assertEqual(
                        main(["status", "--root", str(project_root), "--config", str(config_path)]),
                        0,
                    )
            self.assertTrue((project_root / "experiments/ridge-baseline/run.ipynb").is_file())
            self.assertFalse((project_root / ".env").exists())
            self.assertIn("Active workspace source (global TOML):", output.getvalue())

    def test_tool_environment_file_is_preferred_over_global_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            environment_file = root / ".env"
            environment_file.write_text(
                "\n".join(
                    [
                        f"COLAB_MLFLOW_LOCAL_STORAGE_ROOT={root / 'drive'}",
                        "COLAB_MLFLOW_COLAB_STORAGE_ROOT=/content/drive/MyDrive/colab-mlflow-storage",
                        "COLAB_MLFLOW_WORKER_ID=worker-b",
                        f"COLAB_MLFLOW_LOCAL_STATE_ROOT={root / 'state'}",
                    ]
                ),
                encoding="utf-8",
            )
            with patch("colab_mlflow.workspace.tool_environment_file", return_value=environment_file):
                settings = load_default_workspace()
            self.assertEqual(settings.worker_id, "worker-b")
            self.assertEqual(settings.local_storage_root, root / "drive")

    def test_status_and_run_show_present_project_history(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / ".colab-mlflow.toml").write_text(
                "\n".join(
                    [
                        'project_slug = "dogs-vs-cats"',
                        'name = "Dogs vs Cats"',
                        'description = "Image classification."',
                        "[storage]",
                        'colab_root = "/content/drive/MyDrive/colab-mlflow"',
                    ]
                ),
                encoding="utf-8",
            )
            env_file = root / ".env"
            env_file.write_text(
                "\n".join(
                    [
                        f"COLAB_MLFLOW_LOCAL_STORAGE_ROOT={root / 'drive'}",
                        "COLAB_MLFLOW_COLAB_STORAGE_ROOT=/content/drive/MyDrive/colab-mlflow",
                        "COLAB_MLFLOW_WORKER_ID=worker-a",
                    ]
                ),
                encoding="utf-8",
            )
            experiment_root = root / "experiments" / "multihead"
            experiment_root.mkdir(parents=True)
            (experiment_root / "experiment.toml").write_text(
                "\n".join(
                    [
                        'experiment_slug = "multihead"',
                        'type = "training"',
                        'objective = "Train a model."',
                        'primary_metric = "accuracy"',
                        "",
                        "[tracking_contract]",
                        'comparison_parameters = ["learning_rate"]',
                        'comparison_metrics = ["accuracy"]',
                    ]
                ),
                encoding="utf-8",
            )
            records = [
                {
                    "worker_id": "worker-a", "experiment": "multihead", "number": 1,
                    "run_id": "abcdef123456", "run_name": "run", "status": "FINISHED",
                    "start_time": 1, "artifact_uri": "file:///drive/artifacts",
                    "parameters": {"learning_rate": "0.01"},
                    "metrics": {"accuracy": 0.9},
                    "tags": {
                        "experiment.type": "training", "experiment.primary_metric": "accuracy",
                        "source.repository": "https://github.com/example/project.git",
                        "source.commit": "abc", "source.notebook": "experiments/multihead/run.ipynb",
                        "dataset.train.path": "/content/drive/MyDrive/data/train-v1",
                    },
                    "artifacts": ["logs/stdout.log"],
                }
            ]
            output = io.StringIO()
            with patch("colab_mlflow.cli.collect_project_runs", return_value=records), redirect_stdout(output):
                self.assertEqual(main(["status", "--root", str(root), "--env-file", str(env_file)]), 0)
                self.assertEqual(
                    main(
                        [
                            "run", "show", "--root", str(root), "--env-file", str(env_file),
                            "--experiment", "multihead", "--number", "1",
                        ]
                    ),
                    0,
                )
                self.assertEqual(
                    main(
                        [
                            "compare", "--root", str(root), "--env-file", str(env_file),
                            "--experiment", "multihead",
                        ]
                    ),
                    0,
                )
            text = output.getvalue()
            self.assertIn("Experiment: multihead", text)
            self.assertIn("Run: #1", text)
            self.assertIn("logs/stdout.log", text)
            self.assertIn("learning_rate", text)
            self.assertIn("Use 'colab-mlflow run show'", text)
            colab_output = io.StringIO()
            with patch("colab_mlflow.cli.collect_project_runs", return_value=records), redirect_stdout(colab_output):
                self.assertEqual(
                    main(
                        [
                            "run", "colab", "--root", str(root), "--env-file", str(env_file),
                            "--experiment", "multihead", "--number", "1",
                        ]
                    ),
                    0,
                )
            self.assertEqual(
                colab_output.getvalue(),
                "https://colab.research.google.com/github/example/project/blob/abc/experiments/multihead/run.ipynb\n",
            )

    def test_experiment_definition_cannot_be_overwritten_by_a_run_change(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            subprocess.run(["git", "init", "--quiet", str(root)], check=True)

            arguments = [
                "experiment", "create", "--root", str(root), "--slug", "multihead",
                "--type", "training", "--objective", "Train a multi-head model.",
                "--primary-metric", "validation.loss",
            ]
            self.assertEqual(main(arguments), 0)
            with self.assertRaises(FileExistsError):
                main(arguments)

    def test_notebook_generation_requires_a_linked_repository(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            env_file = root / ".env"
            env_file.write_text(
                "\n".join(
                    [
                        f"COLAB_MLFLOW_LOCAL_STORAGE_ROOT={root / 'drive'}",
                        "COLAB_MLFLOW_COLAB_STORAGE_ROOT=/content/drive/MyDrive/colab-mlflow",
                        "COLAB_MLFLOW_WORKER_ID=worker-a",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            main(
                [
                    "init", "--root", str(root), "--project", "unlinked",
                    "--name", "Unlinked", "--description", "No remote yet.",
                    "--env-file", str(env_file),
                ]
            )
            main(
                [
                    "experiment", "create", "--root", str(root), "--slug", "baseline",
                    "--type", "training", "--objective", "Baseline.",
                    "--primary-metric", "loss",
                ]
            )

            with self.assertRaises(ValueError):
                main(["notebook", "generate", "--root", str(root), "--experiment", "baseline"])

    def test_init_rejects_a_project_nested_inside_another_repository(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            subprocess.run(["git", "init", "--quiet", str(root)], check=True)
            project_root = root / "nested-project"
            env_file = root / ".env"
            env_file.write_text(
                "\n".join(
                    [
                        f"COLAB_MLFLOW_LOCAL_STORAGE_ROOT={root / 'drive'}",
                        "COLAB_MLFLOW_COLAB_STORAGE_ROOT=/content/drive/MyDrive/colab-mlflow",
                        "COLAB_MLFLOW_WORKER_ID=worker-a",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                main(
                    [
                        "init", "--root", str(project_root), "--project", "nested-project",
                        "--name", "Nested project", "--description", "Must be independent.",
                        "--env-file", str(env_file),
                    ]
                )
