"""Integration tests: compact Drive manifests become a complete local MLflow database."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from colab_mlflow.sync import sqlite_uri, sync_project

try:
    from mlflow import MlflowClient
except ImportError:  # pragma: no cover - the dependency-backed run exercises these tests
    MlflowClient = None  # type: ignore[assignment,misc]


RUN_ID = "a" * 32
MODEL_ID = "m-" + "b" * 32


def publish_example_run(drive: Path) -> Path:
    """Given: one Drive run containing readable metadata and two lazy artifacts."""

    run_root = drive / "runs/house-pricing-test/linear-baseline" / RUN_ID
    artifacts = run_root / "artifacts"
    model = artifacts / "outputs/model.joblib"
    logged_model = artifacts / f"logged-models/{MODEL_ID}/model.pkl"
    model.parent.mkdir(parents=True)
    logged_model.parent.mkdir(parents=True)
    model.write_bytes(b"portable model")
    logged_model.write_bytes(b"native mlflow model")

    def artifact(path: Path) -> dict[str, object]:
        return {
            "path": path.relative_to(artifacts).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    manifest = {
        "schema_version": "1.0",
        "project": "house-pricing-test",
        "experiment": "linear-baseline",
        "worker": "worker-a",
        "run": {
            "id": RUN_ID,
            "name": "linear-baseline-worker-a",
            "status": "FINISHED",
            "start_time": 1_000,
            "end_time": 2_000,
            "user_id": "colab",
        },
        "parameters": {"alpha": "1.0", "fit_intercept": "True"},
        "metrics": {"validation.rmse": 12.3},
        "metric_history": {
            "validation.rmse": [
                {"value": 13.0, "timestamp": 1_500, "step": 0},
                {"value": 12.3, "timestamp": 2_000, "step": 1},
            ]
        },
        "tags": {
            "project": "house-pricing-test",
            "worker.id": "worker-a",
            "experiment.primary_metric": "validation.rmse",
            "source.commit": "abc123",
        },
        "datasets": {
            "train": {
                "path": "/content/drive/MyDrive/data/train.csv",
                "kind": "file",
                "size_bytes": 42,
                "sha256": "c" * 64,
            }
        },
        "mlflow_inputs": {
            "dataset_inputs": [
                {
                    "dataset": {
                        "name": "housing_train",
                        "digest": "dataset-digest",
                        "source_type": "local",
                        "source": '{"uri": "train.csv"}',
                        "schema": '{"mlflow_colspec": [{"name": "price", "type": "double"}]}',
                        "profile": '{"num_rows": 100}',
                    },
                    "tags": {"mlflow.data.context": "train"},
                }
            ],
            "model_inputs": [],
        },
        "summary": {"model_family": "Ridge"},
        "artifact_root": "artifacts",
        "artifacts": [artifact(model), artifact(logged_model)],
        "logged_models": [
            {
                "id": MODEL_ID,
                "name": "model",
                "source_run_id": RUN_ID,
                "status": "READY",
                "status_message": None,
                "model_type": None,
                "creation_timestamp": 1_500,
                "last_updated_timestamp": 1_900,
                "artifact_path": f"logged-models/{MODEL_ID}",
                "parameters": {"alpha": "1.0"},
                "metrics": [],
                "tags": {"framework": "sklearn"},
            }
        ],
    }
    path = run_root / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


@unittest.skipIf(MlflowClient is None, "MLflow dependency is not installed")
class ManifestSyncIntegrationTest(unittest.TestCase):
    def test_sync_preserves_run_identity_history_and_lazy_artifact_locations(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            drive = root / "drive"
            publish_example_run(drive)

            # When: the same Drive state is synchronized twice.
            first = sync_project(
                storage_root=drive,
                local_state_root=root / "state",
                project_slug="house-pricing-test",
            )
            second = sync_project(
                storage_root=drive,
                local_state_root=root / "state",
                project_slug="house-pricing-test",
            )

            # Then: the first import is complete and the second is idempotent.
            self.assertTrue(first.changed)
            self.assertFalse(second.changed)
            self.assertEqual((first.experiments, first.runs), (1, 1))

            client = MlflowClient(tracking_uri=sqlite_uri(first.database_path))
            experiment = next(
                item
                for item in client.search_experiments()
                if item.name == "house-pricing-test/linear-baseline"
            )
            run = client.get_run(RUN_ID)
            history = client.get_metric_history(RUN_ID, "validation.rmse")

            self.assertEqual(run.info.experiment_id, experiment.experiment_id)
            self.assertEqual(run.data.params["alpha"], "1.0")
            self.assertEqual(run.data.metrics["validation.rmse"], 12.3)
            self.assertEqual(len(run.inputs.dataset_inputs), 1)
            dataset_input = run.inputs.dataset_inputs[0]
            self.assertEqual(dataset_input.dataset.name, "housing_train")
            self.assertEqual(dataset_input.dataset.digest, "dataset-digest")
            self.assertEqual(
                {tag.key: tag.value for tag in dataset_input.tags}["mlflow.data.context"],
                "train",
            )
            self.assertEqual([point.value for point in history], [13.0, 12.3])
            self.assertEqual(run.data.tags["worker.id"], "worker-a")
            self.assertEqual(
                run.info.artifact_uri,
                (drive / f"runs/house-pricing-test/linear-baseline/{RUN_ID}/artifacts").as_uri(),
            )
            self.assertFalse(list((root / "state").rglob("*.joblib")))
            self.assertFalse(list((root / "state").rglob("model.pkl")))

            models = client.search_logged_models([experiment.experiment_id])
            self.assertEqual([model.model_id for model in models], [MODEL_ID])
            self.assertEqual(models[0].source_run_id, RUN_ID)


if __name__ == "__main__":
    unittest.main()
