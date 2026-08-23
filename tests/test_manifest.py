"""Contract tests for the compact, completed-run document on Drive."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from colab_mlflow.manifest import discover_run_manifests, load_run_manifest
from colab_mlflow.sync import manifest_fingerprint


def minimal_manifest(run_id: str = "a" * 32) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "project": "house-pricing-test",
        "experiment": "baseline",
        "worker": "worker-a",
        "run": {
            "id": run_id,
            "name": "baseline-worker-a",
            "status": "FINISHED",
            "start_time": 100,
            "end_time": 200,
            "user_id": "colab",
        },
        "parameters": {},
        "metrics": {},
        "metric_history": {},
        "tags": {},
        "datasets": {},
        "mlflow_inputs": {"dataset_inputs": [], "model_inputs": []},
        "summary": {},
        "artifact_root": "artifacts",
        "artifacts": [],
        "logged_models": [],
    }


def write_manifest(storage: Path, document: dict[str, object]) -> Path:
    run = document["run"]
    path = storage / "runs/house-pricing-test/baseline" / run["id"] / "manifest.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return path


class ManifestContractTest(unittest.TestCase):
    def test_only_finalized_runs_can_be_published(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            document = minimal_manifest()
            document["run"]["status"] = "RUNNING"
            path = write_manifest(Path(temporary), document)

            with self.assertRaisesRegex(ValueError, "status must be final"):
                load_run_manifest(path)

    def test_run_folder_and_manifest_identity_must_match(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = write_manifest(Path(temporary), minimal_manifest())
            wrong_folder = path.parent.parent / ("b" * 32)
            wrong_folder.mkdir()
            moved = wrong_folder / "manifest.json"
            moved.write_bytes(path.read_bytes())

            with self.assertRaisesRegex(ValueError, "does not match its Drive folder"):
                load_run_manifest(moved)

    def test_artifact_bytes_do_not_participate_in_metadata_sync_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            storage = Path(temporary)
            path = write_manifest(storage, minimal_manifest())
            artifact = path.parent / "artifacts/model.bin"
            artifact.parent.mkdir()
            artifact.write_bytes(b"first large model")
            manifests = discover_run_manifests(storage, "house-pricing-test")
            before = manifest_fingerprint(manifests)

            artifact.write_bytes(b"replacement large model")
            after = manifest_fingerprint(manifests)

            self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
