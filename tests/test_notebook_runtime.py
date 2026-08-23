"""Executable specification for the standalone notebook's publish behavior."""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from colab_mlflow.manifest import load_run_manifest
from colab_mlflow.notebook import create_notebook


class FakeRun:
    def __init__(self, mlflow: "FakeMlflow", record: object) -> None:
        self.mlflow = mlflow
        self.record = record
        self.info = record.info
        self.parent = mlflow.active

    def __enter__(self) -> "FakeRun":
        self.mlflow.active = self.record
        return self

    def __exit__(self, exception_type: object, exception: object, traceback: object) -> None:
        self.record.info.status = "FAILED" if exception_type else "FINISHED"
        self.record.info.end_time = self.record.info.start_time + 100
        self.mlflow.active = self.parent


class FakeClient:
    def __init__(self, mlflow: "FakeMlflow") -> None:
        self.mlflow = mlflow

    def get_run(self, run_id: str) -> object:
        return self.mlflow.runs[run_id]

    def search_runs(self, experiment_ids: list[str], order_by: list[str]) -> list[object]:
        return list(self.mlflow.runs.values())

    def get_metric_history(self, run_id: str, key: str) -> list[object]:
        value = self.mlflow.runs[run_id].data.metrics[key]
        return [types.SimpleNamespace(value=value, timestamp=2_000, step=0)]

    def search_logged_models(self, experiment_ids: list[str]) -> list[object]:
        return []


class FakeMlflow(types.ModuleType):
    """Small in-memory MLflow surface; artifact calls still write real local files."""

    def __init__(self) -> None:
        super().__init__("mlflow")
        self.runs: dict[str, object] = {}
        self.active: object | None = None
        self.tracking_uri = ""
        self.experiment = types.SimpleNamespace(experiment_id="1")
        self.MlflowClient = lambda tracking_uri=None: FakeClient(self)

    def set_tracking_uri(self, uri: str) -> None:
        self.tracking_uri = uri

    def set_experiment(self, name: str) -> object:
        self.experiment.name = name
        return self.experiment

    def autolog(self, **options: object) -> None:
        self.autolog_options = options

    def start_run(self, run_name: str, nested: bool = False) -> FakeRun:
        run_id = f"{len(self.runs) + 1:032x}"
        artifact_root = Path(self.tracking_uri.removeprefix("file://")) / "artifacts" / run_id
        artifact_root.mkdir(parents=True)
        record = types.SimpleNamespace(
            info=types.SimpleNamespace(
                run_id=run_id,
                run_name=run_name,
                status="RUNNING",
                start_time=1_000,
                end_time=None,
                artifact_uri=artifact_root.as_uri(),
            ),
            data=types.SimpleNamespace(
                params={},
                metrics={},
                tags={
                    "mlflow.user": "root",
                    **(
                        {"mlflow.parentRunId": self.active.info.run_id}
                        if nested and self.active is not None
                        else {}
                    ),
                },
            ),
            inputs=types.SimpleNamespace(
                to_dictionary=lambda: {"dataset_inputs": [], "model_inputs": []}
            ),
        )
        self.runs[run_id] = record
        return FakeRun(self, record)

    def log_params(self, values: dict[str, object]) -> None:
        self.active.data.params.update({key: str(value) for key, value in values.items()})

    def log_metrics(self, values: dict[str, float]) -> None:
        self.active.data.metrics.update(values)

    def set_tags(self, values: dict[str, str]) -> None:
        self.active.data.tags.update(values)

    def set_tag(self, key: str, value: str) -> None:
        self.active.data.tags[key] = value

    def log_dict(self, value: object, artifact_file: str) -> None:
        target = self._artifact(artifact_file)
        target.write_text(json.dumps(value), encoding="utf-8")

    def log_text(self, value: str, artifact_file: str) -> None:
        self._artifact(artifact_file).write_text(value, encoding="utf-8")

    def log_artifact(self, path: str, artifact_path: str) -> None:
        target = self._artifact(f"{artifact_path}/{Path(path).name}")
        shutil.copy2(path, target)

    def log_artifacts(self, path: str, artifact_path: str) -> None:
        source = Path(path)
        for item in source.rglob("*"):
            if item.is_file():
                target = self._artifact(f"{artifact_path}/{item.relative_to(source).as_posix()}")
                shutil.copy2(item, target)

    def _artifact(self, relative: str) -> Path:
        root = Path(self.active.info.artifact_uri.removeprefix("file://"))
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        return target


class NotebookRuntimeTest(unittest.TestCase):
    def test_successful_run_publishes_one_manifest_and_lazy_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project_root, drive_root, notebook_path, run_source = self._project(Path(temporary))
            mlflow = FakeMlflow()

            def run_pipeline(dataset_paths: dict[str, str], parameters: dict[str, object], output_dir: Path) -> dict[str, object]:
                print("training completed")
                (output_dir / "model.joblib").write_bytes(b"trained model")
                return {
                    "metrics": {"validation.accuracy": 0.94},
                    "summary": {"best_head": "classification"},
                }

            globals_document = self._runtime_globals(
                project_root, drive_root, notebook_path, run_pipeline, worker="worker-a"
            )
            globals_document["RUN_LABEL"] = "ridge-seed-42"
            with patch.dict(sys.modules, {"mlflow": mlflow}):
                exec(compile(run_source, "generated-run-cell", "exec"), globals_document)

            manifests = list((drive_root / "runs/dogs-vs-cats/multihead").glob("*/manifest.json"))
            self.assertEqual(len(manifests), 1)
            manifest = load_run_manifest(manifests[0])
            self.assertEqual(manifest.worker, "worker-a")
            self.assertEqual(manifest.run["name"], "ridge-seed-42")
            self.assertEqual(manifest.document["parameters"]["learning_rate"], "0.001")
            self.assertEqual(manifest.document["metrics"]["validation.accuracy"], 0.94)
            self.assertEqual(manifest.document["summary"]["best_head"], "classification")
            self.assertEqual(len(manifest.document["tags"]["experiment.contract_sha256"]), 64)
            self.assertEqual(
                manifest.document["mlflow_inputs"],
                {"dataset_inputs": [], "model_inputs": []},
            )
            self.assertEqual(manifest.document["datasets"]["train"]["sha256"], "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824")
            artifact_paths = {artifact["path"] for artifact in manifest.document["artifacts"]}
            self.assertIn("outputs/model.joblib", artifact_paths)
            self.assertIn("logs/stdout.log", artifact_paths)
            self.assertIn("notebook/run.ipynb", artifact_paths)
            self.assertIn("training completed", (manifest.artifact_root / "logs/stdout.log").read_text())
            self.assertNotIn("tracking", manifest.path.parts)

    def test_failed_run_is_still_published_with_exception_log(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project_root, drive_root, notebook_path, run_source = self._project(Path(temporary))
            mlflow = FakeMlflow()

            def failing_pipeline(dataset_paths: dict[str, str], parameters: dict[str, object], output_dir: Path) -> dict[str, object]:
                print("failure is captured")
                raise RuntimeError("training failed")

            globals_document = self._runtime_globals(
                project_root, drive_root, notebook_path, failing_pipeline, worker="worker-b"
            )
            with patch.dict(sys.modules, {"mlflow": mlflow}), self.assertRaisesRegex(
                RuntimeError, "training failed"
            ):
                exec(compile(run_source, "generated-run-cell", "exec"), globals_document)

            manifest_path = next((drive_root / "runs/dogs-vs-cats/multihead").glob("*/manifest.json"))
            manifest = load_run_manifest(manifest_path)
            self.assertEqual(manifest.document["run"]["status"], "FAILED")
            exception = manifest.artifact_root / "logs/exception.log"
            self.assertIn("RuntimeError: training failed", exception.read_text())
            self.assertIn("failure is captured", (manifest.artifact_root / "logs/stdout.log").read_text())

    def test_optional_contract_validation_warns_but_keeps_the_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project_root, drive_root, notebook_path, run_source = self._project(Path(temporary))
            contract = notebook_path.parent / "experiment.toml"
            contract.write_text(
                """experiment_slug = 'multihead'

[tracking_contract]
required_metrics = ['validation.loss']
summary_fields = ['model_family']
required_artifacts = ['report.json']
validation_mode = 'warn'
""",
                encoding="utf-8",
            )
            mlflow = FakeMlflow()

            def run_pipeline(dataset_paths: dict[str, str], parameters: dict[str, object], output_dir: Path) -> dict[str, object]:
                return {"metrics": {"validation.accuracy": 0.94}, "summary": {"model_family": "demo"}}

            globals_document = self._runtime_globals(
                project_root, drive_root, notebook_path, run_pipeline, worker="worker-a"
            )
            with patch.dict(sys.modules, {"mlflow": mlflow}):
                exec(compile(run_source, "generated-run-cell", "exec"), globals_document)

            manifest_path = next((drive_root / "runs/dogs-vs-cats/multihead").glob("*/manifest.json"))
            manifest = load_run_manifest(manifest_path)
            self.assertEqual(manifest.document["run"]["status"], "FINISHED")
            self.assertEqual(manifest.document["tags"]["experiment.contract_validation"], "warning")
            self.assertIn("results/contract-validation.json", {item["path"] for item in manifest.document["artifacts"]})
            self.assertIn("logs/contract-validation.log", {item["path"] for item in manifest.document["artifacts"]})

    def test_imagefolder_directory_can_be_cached_locally_without_changing_tracked_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project_root, drive_root, notebook_path, run_source = self._project(Path(temporary))
            image_root = drive_root / "datasets/animal-images/v1/train"
            (image_root / "cat").mkdir(parents=True)
            (image_root / "dog").mkdir()
            (image_root / "cat/001.jpg").write_bytes(b"cat-image")
            (image_root / "dog/001.jpg").write_bytes(b"dog-image")
            mlflow = FakeMlflow()
            observed_paths: dict[str, str] = {}

            def run_pipeline(dataset_paths: dict[str, str], parameters: dict[str, object], output_dir: Path) -> dict[str, object]:
                local_images = Path(dataset_paths["train_images"])
                observed_paths["train_images"] = str(local_images)
                self.assertTrue((local_images / "cat/001.jpg").is_file())
                self.assertTrue((local_images / "dog/001.jpg").is_file())
                return {"metrics": {"validation.accuracy": 0.94}, "summary": {}}

            globals_document = self._runtime_globals(
                project_root, drive_root, notebook_path, run_pipeline, worker="worker-a"
            )
            cache_root = Path(temporary) / "colab-cache"
            globals_document.update(
                {
                    "DATASET_PATHS": {"train_images": str(image_root)},
                    "DATASET_CACHE_MODE": "copy",
                    "DATASET_CACHE_ROOT": cache_root,
                }
            )
            with patch.dict(sys.modules, {"mlflow": mlflow}):
                exec(compile(run_source, "generated-run-cell", "exec"), globals_document)

            self.assertTrue(Path(observed_paths["train_images"]).is_relative_to(cache_root))
            manifest_path = next((drive_root / "runs/dogs-vs-cats/multihead").glob("*/manifest.json"))
            manifest = load_run_manifest(manifest_path)
            self.assertEqual(manifest.document["datasets"]["train_images"]["kind"], "directory")
            cache = manifest.document["dataset_cache"]["train_images"]
            self.assertEqual(cache["mode"], "copy")
            self.assertEqual(cache["source_path"], str(image_root))
            self.assertTrue(Path(cache["cache_path"]).is_relative_to(cache_root))
            reused_paths, reused_cache = globals_document["_cmf_cache_datasets"](
                {"train_images": str(image_root)},
                globals_document["dataset_records"],
                "copy",
                cache_root,
            )
            self.assertTrue(reused_cache["train_images"]["cache_hit"])
            self.assertEqual(reused_paths["train_images"], cache["cache_path"])

            archive_path = drive_root / "datasets/animal-images-v1.zip"
            shutil.make_archive(str(archive_path.with_suffix("")), "zip", root_dir=image_root)
            archive_record = globals_document["_cmf_dataset_record"](archive_path)
            archive_paths, archive_cache = globals_document["_cmf_cache_datasets"](
                {"archive": str(archive_path)},
                {"archive": archive_record},
                "archive",
                cache_root,
            )
            self.assertTrue((Path(archive_paths["archive"]) / "cat/001.jpg").is_file())
            self.assertEqual(archive_cache["archive"]["mode"], "archive")
            artifact_paths = {artifact["path"] for artifact in manifest.document["artifacts"]}
            self.assertIn("inputs/dataset-cache.json", artifact_paths)

    def test_strict_contract_validation_marks_the_run_failed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project_root, drive_root, notebook_path, run_source = self._project(Path(temporary))
            (notebook_path.parent / "experiment.toml").write_text(
                """experiment_slug = 'multihead'

[tracking_contract]
required_metrics = ['validation.loss']
validation_mode = 'strict'
""",
                encoding="utf-8",
            )
            mlflow = FakeMlflow()

            def run_pipeline(dataset_paths: dict[str, str], parameters: dict[str, object], output_dir: Path) -> dict[str, object]:
                return {"metrics": {"validation.accuracy": 0.94}, "summary": {}}

            globals_document = self._runtime_globals(
                project_root, drive_root, notebook_path, run_pipeline, worker="worker-a"
            )
            with patch.dict(sys.modules, {"mlflow": mlflow}), self.assertRaisesRegex(
                RuntimeError, "Contract validation warning"
            ):
                exec(compile(run_source, "generated-run-cell", "exec"), globals_document)

            manifest_path = next((drive_root / "runs/dogs-vs-cats/multihead").glob("*/manifest.json"))
            manifest = load_run_manifest(manifest_path)
            self.assertEqual(manifest.document["run"]["status"], "FAILED")
            self.assertEqual(manifest.document["tags"]["experiment.contract_validation"], "warning")

    def test_nested_tuning_trial_is_published_as_a_separate_linked_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project_root, drive_root, notebook_path, run_source = self._project(Path(temporary))
            mlflow = FakeMlflow()

            def tuning_pipeline(dataset_paths: dict[str, str], parameters: dict[str, object], output_dir: Path) -> dict[str, object]:
                with mlflow.start_run(run_name="trial-1", nested=True):
                    mlflow.log_params({"max_depth": 4})
                    mlflow.log_metrics({"validation.accuracy": 0.95})
                return {
                    "metrics": {"validation.accuracy": 0.95},
                    "summary": {"best_trial": "trial-1"},
                }

            globals_document = self._runtime_globals(
                project_root, drive_root, notebook_path, tuning_pipeline, worker="worker-a"
            )
            with patch.dict(sys.modules, {"mlflow": mlflow}):
                exec(compile(run_source, "generated-run-cell", "exec"), globals_document)

            manifests = [
                load_run_manifest(path)
                for path in (drive_root / "runs/dogs-vs-cats/multihead").glob("*/manifest.json")
            ]
            self.assertEqual(len(manifests), 2)
            child = next(item for item in manifests if item.run["name"] == "trial-1")
            parent = next(item for item in manifests if item.run["name"] != "trial-1")
            self.assertEqual(child.document["tags"]["mlflow.parentRunId"], parent.run_id)
            self.assertEqual(child.document["parameters"]["max_depth"], "4")

    def _project(self, root: Path) -> tuple[Path, Path, Path, str]:
        project_root = root / "project"
        drive_root = root / "drive"
        project_root.mkdir()
        dataset = drive_root / "datasets/train.csv"
        dataset.parent.mkdir(parents=True)
        dataset.write_text("hello", encoding="utf-8")
        notebook_path = project_root / "experiments/multihead/run.ipynb"
        notebook_path.parent.mkdir(parents=True)
        (notebook_path.parent / "experiment.toml").write_text("experiment_slug = 'multihead'\n")
        (notebook_path.parent / "pipeline.yaml").write_text("experiment: multihead\n")
        create_notebook(
            target=notebook_path,
            project="dogs-vs-cats",
            experiment="multihead",
            experiment_type="training",
            objective="Train a multi-head model.",
            primary_metric="validation.accuracy",
            colab_storage_root=str(drive_root),
            repository_url="https://github.com/example/dogs-vs-cats.git",
            repository_branch="main",
            project_root=project_root,
        )
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        run_source = "".join(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if "cmf-run" in cell.get("metadata", {}).get("tags", [])
        )
        return project_root, drive_root, notebook_path, run_source

    def _runtime_globals(
        self,
        project_root: Path,
        drive_root: Path,
        notebook_path: Path,
        pipeline: object,
        *,
        worker: str,
    ) -> dict[str, object]:
        generation = json.loads(notebook_path.read_text())["metadata"]["cmf"]["generation_id"]
        return {
            "WORKER_ID": worker,
            "RUN_LABEL": "",
            "DATASET_CACHE_MODE": "none",
            "DATASET_PATHS": {"train": str(drive_root / "datasets/train.csv")},
            "PROJECT_SLUG": "dogs-vs-cats",
            "EXPERIMENT_SLUG": "multihead",
            "EXPERIMENT_TYPE": "training",
            "EXPERIMENT_OBJECTIVE": "Train a multi-head model.",
            "PRIMARY_METRIC": "validation.accuracy",
            "REPOSITORY_URL": "https://github.com/example/dogs-vs-cats.git",
            "REPOSITORY_BRANCH": "main",
            "NOTEBOOK_RELATIVE_PATH": "experiments/multihead/run.ipynb",
            "DRIVE_ROOT_TEXT": str(drive_root),
            "GENERATION_ID": generation,
            "RUN_VARIANTS": {"worker-a": {"learning_rate": 0.001}, "worker-b": {"learning_rate": 0.01}},
            "TRACKING_OPTIONS": {"autolog": True, "log_models": True},
            "SOURCE_COMMIT": "abc123",
            "PROJECT_ROOT": project_root,
            "NOTEBOOK_PATH": notebook_path,
            "run_pipeline": pipeline,
            "sys": sys,
            "subprocess": __import__("subprocess"),
            "Path": Path,
            "json": json,
        }


if __name__ == "__main__":
    unittest.main()
