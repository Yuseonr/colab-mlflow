"""Generation of standalone, manually editable Google Colab notebooks."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from urllib.parse import quote

from .filesystem import write_text


def _code_cell(tag: str, source: list[str]) -> dict[str, object]:
    """Input: a managed tag and source lines. Output: a notebook code cell."""

    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"tags": [tag]},
        "outputs": [],
        "source": source,
    }


def create_notebook(
    *,
    target: Path,
    project: str,
    experiment: str,
    experiment_type: str,
    objective: str,
    primary_metric: str,
    colab_storage_root: str,
    repository_url: str,
    repository_branch: str,
    project_root: Path,
) -> Path:
    """Input: stable project metadata. Output: a standalone, fully tracked Colab notebook."""

    try:
        relative_notebook = target.resolve().relative_to(project_root.resolve())
    except ValueError as error:
        raise ValueError("Generated notebook must be inside its project repository.") from error
    generation_id = uuid.uuid4().hex
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "cmf": {"generation_id": generation_id, "managed": True},
        },
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {"tags": ["cmf-title"]},
                "source": [
                    f"# {project} / {experiment}\n",
                    f"**Type:** `{experiment_type}`  \n",
                    f"**Objective:** {objective}  \n",
                    f"**Primary metric:** `{primary_metric}`\n",
                    "\nIn Colab, edit `WORKER_ID`, `RUN_LABEL`, `DATASET_PATHS`, and optionally `DATASET_CACHE_MODE`. The notebook's parameter and pipeline cells can be edited manually or by a coding agent.\n",
                ],
            },
            _code_cell(
                "cmf-inputs",
                [
                    "# The only per-Colab values you edit manually.\n",
                    'WORKER_ID = "worker-a"\n',
                    'RUN_LABEL = ""  # Optional; defaults to experiment-worker-timestamp.\n',
                    'DATASET_CACHE_MODE = "none"  # "none", "copy", or "archive"; cache lives only in this Colab runtime.\n',
                    "DATASET_PATHS = {\n",
                    f'    "train": "{colab_storage_root}/datasets/CHANGE-ME",\n',
                    "}\n",
                ],
            ),
            _code_cell(
                "cmf-managed-config",
                [
                    "# Generated project identity. Do not edit this cell manually.\n",
                    f"PROJECT_SLUG = {project!r}\n",
                    f"EXPERIMENT_SLUG = {experiment!r}\n",
                    f"EXPERIMENT_TYPE = {experiment_type!r}\n",
                    f'EXPERIMENT_OBJECTIVE = {objective!r}\n',
                    f"PRIMARY_METRIC = {primary_metric!r}\n",
                    f"REPOSITORY_URL = {repository_url!r}\n",
                    f"REPOSITORY_BRANCH = {repository_branch!r}\n",
                    f"NOTEBOOK_RELATIVE_PATH = {relative_notebook.as_posix()!r}\n",
                    f"DRIVE_ROOT_TEXT = {colab_storage_root!r}\n",
                    f"GENERATION_ID = {generation_id!r}\n",
                ],
            ),
            _code_cell(
                "cmf-parameters",
                [
                    "# <cmf:user-code name=\"parameters\">\n",
                    "# Edit manually, or ask a coding agent to edit this cell.\n",
                    "RUN_VARIANTS = {\n",
                    "    # Add optional worker-specific parameter presets here.\n",
                    "    \"default\": {},\n",
                    "}\n",
                    "# Autolog metadata is captured locally, then compacted into one Drive manifest.\n",
                    "TRACKING_OPTIONS = {\"autolog\": True, \"log_models\": True}\n",
                    "# </cmf:user-code>\n",
                ],
            ),
            _code_cell(
                "cmf-bootstrap",
                [
                    "# Self-contained Colab bootstrap: no colab-mlflow package is imported.\n",
                    "import json\n",
                    "import os\n",
                    "import subprocess\n",
                    "import sys\n",
                    "from pathlib import Path\n",
                    "from google.colab import drive\n",
                    'drive.mount("/content/drive")\n',
                    'subprocess.run([sys.executable, "-m", "pip", "install", "-q", "mlflow>=3.15,<4"], check=True)\n',
                    'PROJECT_ROOT = Path("/content") / f"cmf-{PROJECT_SLUG}-{GENERATION_ID[:8]}"\n',
                    "if not (PROJECT_ROOT / \".git\").is_dir():\n",
                    "    subprocess.run([\"git\", \"clone\", \"--branch\", REPOSITORY_BRANCH, \"--single-branch\", REPOSITORY_URL, str(PROJECT_ROOT)], check=True)\n",
                    "SOURCE_COMMIT = subprocess.check_output([\"git\", \"-C\", str(PROJECT_ROOT), \"rev-parse\", \"HEAD\"], text=True).strip()\n",
                    "NOTEBOOK_PATH = PROJECT_ROOT / NOTEBOOK_RELATIVE_PATH\n",
                    "committed_notebook = json.loads(NOTEBOOK_PATH.read_text(encoding=\"utf-8\"))\n",
                    "committed_generation = committed_notebook.get(\"metadata\", {}).get(\"cmf\", {}).get(\"generation_id\")\n",
                    "if committed_generation != GENERATION_ID:\n",
                    "    raise RuntimeError(\"The imported notebook is newer than GitHub. Commit and push the generated notebook, then restart Colab.\")\n",
                    "requirements = PROJECT_ROOT / \"requirements.txt\"\n",
                    "pyproject = PROJECT_ROOT / \"pyproject.toml\"\n",
                    "if requirements.is_file():\n",
                    "    subprocess.run([sys.executable, \"-m\", \"pip\", \"install\", \"-q\", \"-r\", str(requirements)], check=True)\n",
                    "elif pyproject.is_file():\n",
                    "    subprocess.run([sys.executable, \"-m\", \"pip\", \"install\", \"-q\", \"-e\", str(PROJECT_ROOT)], check=True)\n",
                    "os.chdir(PROJECT_ROOT)\n",
                    "print(f\"Ready: {REPOSITORY_URL}@{SOURCE_COMMIT[:8]}\")\n",
                ],
            ),
            _code_cell(
                "cmf-user-code",
                [
                    "# <cmf:user-code name=\"pipeline\">\n",
                    "# Implement manually, or ask a coding agent to edit this cell.\n",
                    "# It must return {'metrics': {...}, 'summary': {...}}; files under output_dir are auto-logged.\n",
                    "def run_pipeline(dataset_paths, parameters, output_dir):\n",
                    "    raise NotImplementedError(\"Implement this experiment pipeline before running the notebook.\")\n",
                    "# </cmf:user-code>\n",
                ],
            ),
            _code_cell(
                "cmf-run",
                [
                    "# Managed wrapper: MLflow runs locally, then publishes one manifest plus real artifacts to Drive.\n",
                    "import hashlib\n",
                    "import io\n",
                    "import os\n",
                    "import platform\n",
                    "import shutil\n",
                    "import tempfile\n",
                    "import tomllib\n",
                    "import traceback\n",
                    "import uuid\n",
                    "from contextlib import redirect_stderr, redirect_stdout\n",
                    "from datetime import datetime, timezone\n",
                    "from importlib.metadata import distributions\n",
                    "from urllib.parse import unquote, urlparse\n",
                    "os.environ.setdefault(\"MLFLOW_ALLOW_FILE_STORE\", \"true\")\n",
                    "import mlflow\n",
                    "from mlflow import MlflowClient\n",
                    "\n",
                    "def _cmf_sha256(path):\n",
                    "    digest = hashlib.sha256()\n",
                    "    with Path(path).open(\"rb\") as stream:\n",
                    "        for chunk in iter(lambda: stream.read(1024 * 1024), b\"\"):\n",
                    "            digest.update(chunk)\n",
                    "    return digest.hexdigest()\n",
                    "\n",
                    "def _cmf_dataset_record(path):\n",
                    "    path = Path(path)\n",
                    "    if path.is_file():\n",
                    "        return {\"path\": str(path), \"kind\": \"file\", \"size_bytes\": path.stat().st_size, \"sha256\": _cmf_sha256(path)}\n",
                    "    digest, total_size, file_count = hashlib.sha256(), 0, 0\n",
                    "    for child in sorted(item for item in path.rglob(\"*\") if item.is_file()):\n",
                    "        stat = child.stat(); relative = child.relative_to(path).as_posix()\n",
                    "        digest.update(f\"{relative}\\0{stat.st_size}\\0{stat.st_mtime_ns}\\n\".encode())\n",
                    "        total_size += stat.st_size; file_count += 1\n",
                    "    return {\"path\": str(path), \"kind\": \"directory\", \"size_bytes\": total_size, \"file_count\": file_count, \"fingerprint\": digest.hexdigest()}\n",
                    "\n",
                    "def _cmf_cache_datasets(dataset_paths, dataset_records, mode, cache_root):\n",
                    "    if mode not in {\"none\", \"copy\", \"archive\"}:\n",
                    "        raise ValueError(\"DATASET_CACHE_MODE must be none, copy, or archive.\")\n",
                    "    resolved, records = {}, {}\n",
                    "    for alias, raw_path in dataset_paths.items():\n",
                    "        source, source_record = Path(raw_path), dataset_records[alias]\n",
                    "        cache_key = source_record.get(\"sha256\") or source_record.get(\"fingerprint\")\n",
                    "        if mode == \"none\":\n",
                    "            resolved[alias] = str(source)\n",
                    "            records[alias] = {\"mode\": mode, \"source_path\": str(source), \"cache_key\": cache_key, \"cache_path\": None, \"cache_hit\": False}\n",
                    "            continue\n",
                    "        alias_key = hashlib.sha256(alias.encode(\"utf-8\")).hexdigest()[:12]\n",
                    "        target_root = Path(cache_root) / cache_key / alias_key\n",
                    "        marker = target_root / \"cache.json\"\n",
                    "        payload = target_root / \"payload\"\n",
                    "        expected = {\"mode\": mode, \"source_path\": str(source), \"cache_key\": cache_key}\n",
                    "        cache_hit = False\n",
                    "        if marker.is_file() and payload.exists():\n",
                    "            try:\n",
                    "                cache_hit = json.loads(marker.read_text(encoding=\"utf-8\")) == expected\n",
                    "            except json.JSONDecodeError:\n",
                    "                cache_hit = False\n",
                    "        if not cache_hit:\n",
                    "            if target_root.exists():\n",
                    "                shutil.rmtree(target_root)\n",
                    "            staging = target_root.with_name(f\".{target_root.name}.{uuid.uuid4().hex}.caching\")\n",
                    "            staging.mkdir(parents=True, exist_ok=False)\n",
                    "            staged_payload = staging / \"payload\"\n",
                    "            try:\n",
                    "                if mode == \"copy\":\n",
                    "                    if source.is_dir():\n",
                    "                        shutil.copytree(source, staged_payload)\n",
                    "                    else:\n",
                    "                        staged_payload.mkdir()\n",
                    "                        shutil.copy2(source, staged_payload / source.name)\n",
                    "                else:\n",
                    "                    if not source.is_file():\n",
                    "                        raise ValueError(f\"archive cache mode requires a .zip or tar file: {source}\")\n",
                    "                    staged_payload.mkdir()\n",
                    "                    shutil.unpack_archive(str(source), str(staged_payload))\n",
                    "                (staging / \"cache.json\").write_text(json.dumps(expected, sort_keys=True), encoding=\"utf-8\")\n",
                    "                target_root.parent.mkdir(parents=True, exist_ok=True)\n",
                    "                os.replace(staging, target_root)\n",
                    "            except Exception:\n",
                    "                shutil.rmtree(staging, ignore_errors=True)\n",
                    "                raise\n",
                    "        if mode == \"copy\" and source.is_file():\n",
                    "            cached_path = payload / source.name\n",
                    "        else:\n",
                    "            cached_path = payload\n",
                    "        resolved[alias] = str(cached_path)\n",
                    "        records[alias] = {**expected, \"cache_path\": str(cached_path), \"cache_hit\": cache_hit}\n",
                    "    return resolved, records\n",
                    "\n",
                    "def _cmf_file_uri_path(uri):\n",
                    "    parsed = urlparse(uri)\n",
                    "    if parsed.scheme != \"file\":\n",
                    "        raise ValueError(f\"Expected local MLflow artifact URI, got: {uri}\")\n",
                    "    return Path(unquote(parsed.path))\n",
                    "\n",
                    "def _cmf_copy_artifact_tree(source, destination, prefix, inventory):\n",
                    "    source, destination = Path(source), Path(destination)\n",
                    "    if not source.is_dir():\n",
                    "        return\n",
                    "    for item in sorted(path for path in source.rglob(\"*\") if path.is_file()):\n",
                    "        relative = item.relative_to(source)\n",
                    "        target = destination / relative\n",
                    "        target.parent.mkdir(parents=True, exist_ok=True)\n",
                    "        inventory.append({\"path\": (Path(prefix) / relative).as_posix(), \"size_bytes\": item.stat().st_size, \"sha256\": _cmf_sha256(item)})\n",
                    "        shutil.copy2(item, target)\n",
                    "\n",
                    "def _cmf_model_document(model, artifact_path):\n",
                    "    status = getattr(model.status, \"value\", str(model.status))\n",
                    "    metrics = []\n",
                    "    for metric in model.metrics:\n",
                    "        metrics.append({\"key\": metric.key, \"value\": float(metric.value), \"timestamp\": int(metric.timestamp), \"step\": int(metric.step), \"run_id\": metric.run_id, \"dataset_name\": metric.dataset_name, \"dataset_digest\": metric.dataset_digest})\n",
                    "    return {\"id\": model.model_id, \"name\": model.name, \"source_run_id\": model.source_run_id, \"status\": status, \"status_message\": model.status_message, \"model_type\": model.model_type, \"creation_timestamp\": int(model.creation_timestamp), \"last_updated_timestamp\": int(model.last_updated_timestamp), \"artifact_path\": artifact_path, \"parameters\": {key: str(value) for key, value in model.params.items()}, \"metrics\": metrics, \"tags\": {key: str(value) for key, value in model.tags.items()}}\n",
                    "\n",
                    "def _cmf_inputs_document(tracked):\n",
                    "    if tracked.inputs is None:\n",
                    "        return {\"dataset_inputs\": [], \"model_inputs\": []}\n",
                    "    inputs = tracked.inputs.to_dictionary()\n",
                    "    return {\"dataset_inputs\": inputs.get(\"dataset_inputs\", []), \"model_inputs\": inputs.get(\"model_inputs\", [])}\n",
                    "\n",
                    "def _cmf_contract_strings(contract, key):\n",
                    "    values = contract.get(key, [])\n",
                    "    return [value for value in values if isinstance(value, str) and value] if isinstance(values, list) else []\n",
                    "\n",
                    "def _cmf_validate_contract(metrics, summary, output_dir):\n",
                    "    document = tomllib.loads(EXPERIMENT_MANIFEST_PATH.read_text(encoding=\"utf-8\"))\n",
                    "    contract = document.get(\"tracking_contract\", {})\n",
                    "    if not isinstance(contract, dict):\n",
                    "        raise ValueError(\"tracking_contract must be a TOML table when present.\")\n",
                    "    required_metrics = _cmf_contract_strings(contract, \"required_metrics\")\n",
                    "    summary_fields = _cmf_contract_strings(contract, \"summary_fields\")\n",
                    "    required_artifacts = _cmf_contract_strings(contract, \"required_artifacts\")\n",
                    "    if not (required_metrics or summary_fields or required_artifacts):\n",
                    "        return None\n",
                    "    mode = contract.get(\"validation_mode\", \"warn\")\n",
                    "    if mode not in {\"warn\", \"strict\"}:\n",
                    "        raise ValueError(\"tracking_contract.validation_mode must be warn or strict.\")\n",
                    "    missing = {\n",
                    "        \"metrics\": [key for key in required_metrics if key not in metrics],\n",
                    "        \"summary_fields\": [key for key in summary_fields if key not in summary],\n",
                    "        \"artifacts\": [path for path in required_artifacts if not (output_dir / path).is_file()],\n",
                    "    }\n",
                    "    missing = {key: value for key, value in missing.items() if value}\n",
                    "    return {\"mode\": mode, \"status\": \"passed\" if not missing else \"warning\", \"missing\": missing}\n",
                    "\n",
                    "def _cmf_publish_run(client, experiment_id, run_id, dataset_records, dataset_cache_records, summary):\n",
                    "    tracked = client.get_run(run_id)\n",
                    "    if tracked.info.status not in {\"FINISHED\", \"FAILED\", \"KILLED\"} or tracked.info.end_time is None:\n",
                    "        raise RuntimeError(f\"Run {run_id} is not finalized and cannot be published.\")\n",
                    "    publish_root = DRIVE_ROOT / \"runs\" / PROJECT_SLUG / EXPERIMENT_SLUG / run_id\n",
                    "    publish_root.parent.mkdir(parents=True, exist_ok=True)\n",
                    "    if publish_root.exists():\n",
                    "        raise FileExistsError(f\"A published run already exists: {publish_root}\")\n",
                    "    staging = publish_root.with_name(f\".{run_id}.{uuid.uuid4().hex}.publishing\")\n",
                    "    artifact_root = staging / \"artifacts\"\n",
                    "    artifact_root.mkdir(parents=True, exist_ok=False)\n",
                    "    inventory = []\n",
                    "    _cmf_copy_artifact_tree(_cmf_file_uri_path(tracked.info.artifact_uri), artifact_root, \"\", inventory)\n",
                    "    logged_models = []\n",
                    "    for model in client.search_logged_models([experiment_id]):\n",
                    "        if model.source_run_id != run_id:\n",
                    "            continue\n",
                    "        relative = f\"logged-models/{model.model_id}\"\n",
                    "        _cmf_copy_artifact_tree(_cmf_file_uri_path(model.artifact_location), artifact_root / relative, relative, inventory)\n",
                    "        logged_models.append(_cmf_model_document(model, relative))\n",
                    "    history = {}\n",
                    "    for key in tracked.data.metrics:\n",
                    "        history[key] = [{\"value\": float(point.value), \"timestamp\": int(point.timestamp), \"step\": int(point.step)} for point in client.get_metric_history(run_id, key)]\n",
                    "    manifest = {\n",
                    "        \"schema_version\": \"1.0\", \"project\": PROJECT_SLUG, \"experiment\": EXPERIMENT_SLUG, \"worker\": WORKER_ID,\n",
                    "        \"run\": {\"id\": run_id, \"name\": tracked.info.run_name or run_id[:8], \"status\": tracked.info.status, \"start_time\": int(tracked.info.start_time or 0), \"end_time\": int(tracked.info.end_time), \"user_id\": tracked.data.tags.get(\"mlflow.user\", \"colab\")},\n",
                    "        \"parameters\": {key: str(value) for key, value in tracked.data.params.items()},\n",
                    "        \"metrics\": {key: float(value) for key, value in tracked.data.metrics.items()},\n",
                    "        \"metric_history\": history, \"tags\": {key: str(value) for key, value in tracked.data.tags.items()},\n",
                    "        \"datasets\": dataset_records, \"dataset_cache\": dataset_cache_records, \"mlflow_inputs\": _cmf_inputs_document(tracked), \"summary\": summary, \"artifact_root\": \"artifacts\",\n",
                    "        \"artifacts\": sorted(inventory, key=lambda item: item[\"path\"]), \"logged_models\": logged_models,\n",
                    "    }\n",
                    "    (staging / \"manifest.json\").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + \"\\n\", encoding=\"utf-8\")\n",
                    "    os.replace(staging, publish_root)\n",
                    "    return publish_root\n",
                    "\n",
                    "DRIVE_ROOT = Path(DRIVE_ROOT_TEXT)\n",
                    "EXPERIMENT_ROOT = PROJECT_ROOT / \"experiments\" / EXPERIMENT_SLUG\n",
                    "EXPERIMENT_MANIFEST_PATH = EXPERIMENT_ROOT / \"experiment.toml\"\n",
                    "PIPELINE_PATH = EXPERIMENT_ROOT / \"pipeline.yaml\"\n",
                    "CONTRACT_SHA256 = hashlib.sha256(EXPERIMENT_MANIFEST_PATH.read_bytes() + b\"\\0\" + PIPELINE_PATH.read_bytes()).hexdigest()\n",
                    "if not WORKER_ID.strip():\n",
                    "    raise ValueError(\"WORKER_ID is required.\")\n",
                    "allowed_worker = set(\"abcdefghijklmnopqrstuvwxyz0123456789-\")\n",
                    "if set(WORKER_ID) - allowed_worker or WORKER_ID.startswith(\"-\") or WORKER_ID.endswith(\"-\") or \"--\" in WORKER_ID:\n",
                    "    raise ValueError(\"WORKER_ID must use lowercase letters, numbers, and single dashes.\")\n",
                    "selected_parameters = RUN_VARIANTS.get(WORKER_ID, RUN_VARIANTS.get(\"default\"))\n",
                    "if selected_parameters is None:\n",
                    "    raise ValueError(f\"No RUN_VARIANTS entry for {WORKER_ID} and no default preset.\")\n",
                    "RUN_PARAMETERS = dict(selected_parameters)\n",
                    "RUN_LABEL = globals().get(\"RUN_LABEL\", \"\")\n",
                    "if not isinstance(RUN_LABEL, str):\n",
                    "    raise ValueError(\"RUN_LABEL must be a string.\")\n",
                    "DATASET_CACHE_MODE = globals().get(\"DATASET_CACHE_MODE\", \"none\")\n",
                    "if not isinstance(DATASET_CACHE_MODE, str):\n",
                    "    raise ValueError(\"DATASET_CACHE_MODE must be a string.\")\n",
                    "DATASET_CACHE_ROOT = Path(globals().get(\"DATASET_CACHE_ROOT\", \"/content/cmf-datasets\"))\n",
                    "TRACKING_OPTIONS = globals().get(\"TRACKING_OPTIONS\", {\"autolog\": True, \"log_models\": True})\n",
                    "if not isinstance(TRACKING_OPTIONS, dict):\n",
                    "    raise ValueError(\"TRACKING_OPTIONS must be a dictionary.\")\n",
                    "if not DATASET_PATHS:\n",
                    "    raise ValueError(\"At least one explicit dataset path is required.\")\n",
                    "dataset_records = {}\n",
                    "for alias, value in DATASET_PATHS.items():\n",
                    "    path = Path(value)\n",
                    "    if not path.exists() or not path.is_relative_to(DRIVE_ROOT):\n",
                    "        raise ValueError(f\"Dataset must exist under {DRIVE_ROOT}: {path}\")\n",
                    "    dataset_records[alias] = _cmf_dataset_record(path)\n",
                    "PIPELINE_DATASET_PATHS, dataset_cache_records = _cmf_cache_datasets(DATASET_PATHS, dataset_records, DATASET_CACHE_MODE, DATASET_CACHE_ROOT)\n",
                    "LOCAL_TRACKING_ROOT = Path(tempfile.mkdtemp(prefix=f\"cmf-{EXPERIMENT_SLUG}-\"))\n",
                    "LOCAL_MLRUNS = LOCAL_TRACKING_ROOT / \"mlruns\"\n",
                    "LOCAL_MLRUNS.mkdir()\n",
                    "mlflow.set_tracking_uri(LOCAL_MLRUNS.as_uri())\n",
                    "local_experiment = mlflow.set_experiment(f\"{PROJECT_SLUG}/{EXPERIMENT_SLUG}\")\n",
                    "if TRACKING_OPTIONS.get(\"autolog\", True):\n",
                    "    mlflow.autolog(log_models=bool(TRACKING_OPTIONS.get(\"log_models\", True)), silent=True)\n",
                    "stdout_log, stderr_log = io.StringIO(), io.StringIO()\n",
                    "RUN_METRICS, RUN_SUMMARY = {}, {}\n",
                    "RUN_OUTPUT_DIR = LOCAL_TRACKING_ROOT / \"outputs\"\n",
                    "RUN_OUTPUT_DIR.mkdir()\n",
                    "run_name = RUN_LABEL.strip() or f\"{EXPERIMENT_SLUG}-{WORKER_ID}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}\"\n",
                    "PRIMARY_RUN_ID = None\n",
                    "PUBLISH_SUCCEEDED = False\n",
                    "try:\n",
                    "    with mlflow.start_run(run_name=run_name) as active_run:\n",
                    "        PRIMARY_RUN_ID = active_run.info.run_id\n",
                    "        mlflow.log_params(RUN_PARAMETERS)\n",
                    "        mlflow.set_tags({\n",
                    "            \"project\": PROJECT_SLUG, \"experiment\": EXPERIMENT_SLUG,\n",
                    "            \"experiment.type\": EXPERIMENT_TYPE, \"experiment.objective\": EXPERIMENT_OBJECTIVE,\n",
                    "            \"experiment.primary_metric\": PRIMARY_METRIC, \"experiment.contract_sha256\": CONTRACT_SHA256, \"worker.id\": WORKER_ID,\n",
                    "            \"source.repository\": REPOSITORY_URL, \"source.branch\": REPOSITORY_BRANCH,\n",
                    "            \"source.commit\": SOURCE_COMMIT, \"source.notebook\": NOTEBOOK_RELATIVE_PATH,\n",
                    "            \"source.pipeline\": str(PIPELINE_PATH.relative_to(PROJECT_ROOT)),\n",
                    "            \"notebook.generation_id\": GENERATION_ID, \"transport.format\": \"drive-manifest-v1\",\n",
                    "        })\n",
                    "        mlflow.set_tags({f\"dataset.{alias}.path\": record[\"path\"] for alias, record in dataset_records.items()})\n",
                    "        mlflow.set_tags({f\"dataset.{alias}.cache_mode\": record[\"mode\"] for alias, record in dataset_cache_records.items()})\n",
                    "        mlflow.log_dict(dataset_records, \"inputs/datasets.json\")\n",
                    "        mlflow.log_dict(dataset_cache_records, \"inputs/dataset-cache.json\")\n",
                    "        mlflow.log_dict({\"repository\": REPOSITORY_URL, \"branch\": REPOSITORY_BRANCH, \"commit\": SOURCE_COMMIT, \"notebook\": NOTEBOOK_RELATIVE_PATH}, \"source/revision.json\")\n",
                    "        mlflow.log_dict({\"python\": platform.python_version(), \"platform\": platform.platform()}, \"source/runtime.json\")\n",
                    "        environment = \"\\n\".join(sorted(f\"{item.metadata['Name']}=={item.version}\" for item in distributions() if item.metadata.get('Name'))) + \"\\n\"\n",
                    "        mlflow.log_text(environment, \"source/environment.txt\")\n",
                    "        mlflow.log_artifact(str(EXPERIMENT_MANIFEST_PATH), artifact_path=\"source\")\n",
                    "        mlflow.log_artifact(str(PIPELINE_PATH), artifact_path=\"source\")\n",
                    "        failure = \"\"\n",
                    "        try:\n",
                    "            with redirect_stdout(stdout_log), redirect_stderr(stderr_log):\n",
                    "                result = run_pipeline(PIPELINE_DATASET_PATHS, RUN_PARAMETERS, RUN_OUTPUT_DIR)\n",
                    "                if isinstance(result, dict):\n",
                    "                    RUN_METRICS = result.get(\"metrics\", {})\n",
                    "                    RUN_SUMMARY = result.get(\"summary\", {})\n",
                    "        except Exception:\n",
                    "            failure = traceback.format_exc()\n",
                    "            raise\n",
                    "        finally:\n",
                    "            if RUN_METRICS:\n",
                    "                mlflow.log_metrics({key: float(value) for key, value in RUN_METRICS.items()})\n",
                    "            mlflow.log_dict(RUN_SUMMARY, \"results/summary.json\")\n",
                    "            mlflow.log_text(stdout_log.getvalue(), \"logs/stdout.log\")\n",
                    "            mlflow.log_text(stderr_log.getvalue(), \"logs/stderr.log\")\n",
                    "            if failure:\n",
                    "                mlflow.log_text(failure, \"logs/exception.log\")\n",
                    "            if any(RUN_OUTPUT_DIR.iterdir()):\n",
                    "                mlflow.log_artifacts(str(RUN_OUTPUT_DIR), artifact_path=\"outputs\")\n",
                    "            mlflow.log_artifact(str(NOTEBOOK_PATH), artifact_path=\"notebook\")\n",
                    "            mlflow.set_tag(\"notebook.sha256\", hashlib.sha256(NOTEBOOK_PATH.read_bytes()).hexdigest())\n",
                    "            if not failure:\n",
                    "                contract_validation = _cmf_validate_contract(RUN_METRICS, RUN_SUMMARY, RUN_OUTPUT_DIR)\n",
                    "                if contract_validation is not None:\n",
                    "                    mlflow.log_dict(contract_validation, \"results/contract-validation.json\")\n",
                    "                    mlflow.set_tag(\"experiment.contract_validation\", contract_validation[\"status\"])\n",
                    "                    if contract_validation[\"status\"] == \"warning\":\n",
                    "                        message = f\"Contract validation warning: {json.dumps(contract_validation['missing'], sort_keys=True)}\"\n",
                    "                        mlflow.log_text(message + \"\\n\", \"logs/contract-validation.log\")\n",
                    "                        if contract_validation[\"mode\"] == \"strict\":\n",
                    "                            raise RuntimeError(message)\n",
                    "finally:\n",
                    "    if PRIMARY_RUN_ID is not None:\n",
                    "        client = MlflowClient(tracking_uri=LOCAL_MLRUNS.as_uri())\n",
                    "        local_runs = client.search_runs([local_experiment.experiment_id], order_by=[\"attributes.start_time ASC\"])\n",
                    "        published = []\n",
                    "        for local_run in local_runs:\n",
                    "            summary = RUN_SUMMARY if local_run.info.run_id == PRIMARY_RUN_ID else {}\n",
                    "            published.append(_cmf_publish_run(client, local_experiment.experiment_id, local_run.info.run_id, dataset_records, dataset_cache_records, summary))\n",
                    "        PUBLISH_SUCCEEDED = True\n",
                    "        print(\"Published manifests:\", *published, sep=\"\\n- \" )\n",
                    "    if PUBLISH_SUCCEEDED:\n",
                    "        shutil.rmtree(LOCAL_TRACKING_ROOT, ignore_errors=True)\n",
                    "if PRIMARY_RUN_ID is not None:\n",
                    "    print(f\"Tracked run: {PRIMARY_RUN_ID}\")\n",
                ],
            ),
        ],
    }
    _preserve_editable_cells(target, notebook)
    write_text(target, json.dumps(notebook, ensure_ascii=False, indent=2) + "\n")
    return target


def _preserve_editable_cells(target: Path, notebook: dict[str, object]) -> None:
    """Input: an existing notebook and regenerated document. Output: preserved editable cells."""

    if not target.is_file():
        return
    existing = json.loads(target.read_text(encoding="utf-8"))
    editable_tags = {"cmf-inputs", "cmf-parameters", "cmf-user-code"}
    existing_by_tag: dict[str, list[str]] = {}
    for cell in existing.get("cells", []):
        for tag in cell.get("metadata", {}).get("tags", []):
            if tag in editable_tags and isinstance(cell.get("source"), list):
                existing_by_tag[tag] = cell["source"]
    for cell in notebook["cells"]:  # type: ignore[index]
        tags = cell.get("metadata", {}).get("tags", [])
        for tag in tags:
            if tag in existing_by_tag:
                preserved = list(existing_by_tag[tag])
                if tag == "cmf-parameters" and "TRACKING_OPTIONS" not in "".join(preserved):
                    option = 'TRACKING_OPTIONS = {"autolog": True, "log_models": True}\n'
                    closing = next(
                        (index for index, line in enumerate(preserved) if "</cmf:user-code>" in line),
                        len(preserved),
                    )
                    preserved.insert(closing, option)
                if tag == "cmf-inputs" and "RUN_LABEL" not in "".join(preserved):
                    insertion = 'RUN_LABEL = ""  # Optional; defaults to experiment-worker-timestamp.\n'
                    worker_line = next(
                        (index + 1 for index, line in enumerate(preserved) if "WORKER_ID" in line),
                        0,
                    )
                    preserved.insert(worker_line, insertion)
                if tag == "cmf-inputs" and "DATASET_CACHE_MODE" not in "".join(preserved):
                    insertion = 'DATASET_CACHE_MODE = "none"  # "none", "copy", or "archive"; cache lives only in this Colab runtime.\n'
                    label_line = next(
                        (index + 1 for index, line in enumerate(preserved) if "RUN_LABEL" in line),
                        0,
                    )
                    preserved.insert(label_line, insertion)
                cell["source"] = preserved


def github_colab_url(repository_url: str, branch: str, notebook_path: Path) -> str | None:
    """Input: GitHub repository metadata. Output: a direct Colab URL when supported."""

    repository = repository_url.strip()
    if repository.startswith("git@github.com:"):
        repository = repository.removeprefix("git@github.com:")
    elif repository.startswith("https://github.com/"):
        repository = repository.removeprefix("https://github.com/")
    else:
        return None
    repository = repository.removesuffix(".git").strip("/")
    if repository.count("/") != 1:
        return None
    return (
        "https://colab.research.google.com/github/"
        f"{repository}/blob/{quote(branch, safe='')}/{quote(notebook_path.as_posix(), safe='/')}"
    )
