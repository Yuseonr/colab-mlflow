"""Behavior tests for human-readable status and run detail output."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from colab_mlflow.tracking import (
    collect_project_runs,
    format_experiment_comparison,
    format_project_status,
    format_run_detail,
    select_run,
)


def publish_run(
    storage: Path, *, worker: str, run_id: str, start_time: int, accuracy: float
) -> None:
    run_root = storage / "runs/dogs-vs-cats/multihead" / run_id
    stdout = run_root / "artifacts/logs/stdout.log"
    stdout.parent.mkdir(parents=True)
    stdout.write_text(f"completed on {worker}", encoding="utf-8")
    manifest = {
        "schema_version": "1.0",
        "project": "dogs-vs-cats",
        "experiment": "multihead",
        "worker": worker,
        "run": {
            "id": run_id,
            "name": f"multihead-{worker}",
            "status": "FINISHED",
            "start_time": start_time,
            "end_time": start_time + 100,
            "user_id": "colab",
        },
        "parameters": {"learning_rate": "0.001" if worker == "worker-a" else "0.01"},
        "metrics": {"validation.accuracy": accuracy},
        "metric_history": {
            "validation.accuracy": [
                {"value": accuracy, "timestamp": start_time + 100, "step": 0}
            ]
        },
        "tags": {
            "experiment.type": "training",
            "experiment.primary_metric": "validation.accuracy",
            "source.repository": "https://github.com/example/dogs-vs-cats.git",
            "source.commit": run_id[:8],
            "source.notebook": "experiments/multihead/run.ipynb",
            "source.pipeline": "experiments/multihead/pipeline.yaml",
        },
        "datasets": {
            "train": {
                "path": "/content/drive/MyDrive/data/train-v1",
                "kind": "directory",
                "size_bytes": 100,
                "file_count": 2,
                "fingerprint": "d" * 64,
            }
        },
        "mlflow_inputs": {"dataset_inputs": [], "model_inputs": []},
        "summary": {"best_head": "classification"},
        "artifact_root": "artifacts",
        "artifacts": [
            {
                "path": "logs/stdout.log",
                "size_bytes": stdout.stat().st_size,
                "sha256": hashlib.sha256(stdout.read_bytes()).hexdigest(),
            }
        ],
        "logged_models": [],
    }
    (run_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


class TrackingTest(unittest.TestCase):
    def test_two_workers_appear_as_one_ordered_experiment_history(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            storage = Path(temporary)
            publish_run(storage, worker="worker-a", run_id="a" * 32, start_time=1_000, accuracy=0.91)
            publish_run(storage, worker="worker-b", run_id="b" * 32, start_time=2_000, accuracy=0.93)

            records = collect_project_runs(
                storage_root=storage,
                project_slug="dogs-vs-cats",
                include_artifacts=True,
            )

            self.assertEqual([record["number"] for record in records], [1, 2])
            status = format_project_status("dogs-vs-cats", records)
            self.assertIn("Experiment: multihead (training) — 2 run(s)", status)
            self.assertIn("#1 [worker-a]", status)
            self.assertIn("#2 [worker-b]", status)

            second_run = select_run(records, experiment="multihead", number=2)
            detail = format_run_detail("dogs-vs-cats", second_run)
            self.assertIn("Worker: worker-b", detail)
            self.assertIn("validation.accuracy: 0.93", detail)
            self.assertIn("train: /content/drive/MyDrive/data/train-v1", detail)
            self.assertIn("completed on worker-b", detail)
            self.assertIn('"best_head": "classification"', detail)
            self.assertIn("logs/stdout.log", detail)

    def test_missing_run_has_a_clear_error(self) -> None:
        with self.assertRaisesRegex(LookupError, "Run #3"):
            select_run([], experiment="multihead", number=3)

    def test_contract_comparison_is_concise_without_removing_full_details(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            storage = Path(temporary)
            publish_run(storage, worker="worker-a", run_id="a" * 32, start_time=1_000, accuracy=0.91)
            publish_run(storage, worker="worker-b", run_id="b" * 32, start_time=2_000, accuracy=0.93)
            records = collect_project_runs(storage_root=storage, project_slug="dogs-vs-cats")

            comparison = format_experiment_comparison(
                project_slug="dogs-vs-cats",
                experiment="multihead",
                records=records,
                primary_metric="validation.accuracy",
                contract={
                    "comparison_parameters": ["learning_rate"],
                    "comparison_metrics": ["validation.accuracy"],
                },
            )

            self.assertIn("learning_rate", comparison)
            self.assertIn("validation.accuracy", comparison)
            self.assertIn("0.001", comparison)
            self.assertIn("0.93", comparison)
            self.assertIn("run show", comparison)


if __name__ == "__main__":
    unittest.main()
