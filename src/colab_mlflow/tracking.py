"""Read-only project and run inspection from compact Drive manifests."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .manifest import discover_run_manifests


def collect_project_runs(
    *, storage_root: Path, project_slug: str, include_artifacts: bool = False
) -> list[dict[str, Any]]:
    """Return runs across workers without opening MLflow metadata file-by-file."""

    records: list[dict[str, Any]] = []
    for manifest in discover_run_manifests(storage_root, project_slug):
        document = manifest.document
        artifact_paths = [artifact["path"] for artifact in document["artifacts"]]
        records.append(
            {
                "worker_id": manifest.worker,
                "experiment": manifest.experiment,
                "run_id": manifest.run_id,
                "run_name": document["run"]["name"],
                "status": document["run"]["status"],
                "start_time": document["run"]["start_time"],
                "artifact_uri": manifest.artifact_root.as_uri(),
                "manifest_path": str(manifest.path),
                "parameters": dict(document["parameters"]),
                "metrics": dict(document["metrics"]),
                "tags": dict(document["tags"]),
                "datasets": dict(document["datasets"]),
                "artifacts": artifact_paths if include_artifacts else [],
                "log_preview": (
                    _log_previews(manifest.artifact_root) if include_artifacts else {}
                ),
                "result_summary": document["summary"] if include_artifacts else None,
            }
        )
    records.sort(key=lambda record: (record["experiment"], record["start_time"], record["run_id"]))
    counters: defaultdict[str, int] = defaultdict(int)
    for record in records:
        counters[record["experiment"]] += 1
        record["number"] = counters[record["experiment"]]
    return records


def format_project_status(
    project_slug: str,
    records: list[dict[str, Any]],
    definitions: dict[str, dict[str, str]] | None = None,
) -> str:
    """Render a compact experiment inventory from manifest records."""

    lines = [f"Project: {project_slug}"]
    definitions = definitions or {}
    if not records and not definitions:
        return "\n".join(lines + ["No completed run manifests found in the Drive workspace."])
    experiments: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        experiments[record["experiment"]].append(record)
    for experiment in sorted(set(experiments) | set(definitions)):
        runs = experiments[experiment]
        experiment_type = (
            definitions.get(experiment, {}).get("type")
            or (runs[-1]["tags"].get("experiment.type") if runs else None)
            or "unknown"
        )
        lines.append(f"\nExperiment: {experiment} ({experiment_type}) — {len(runs)} run(s)")
        for run in runs:
            primary = run["tags"].get("experiment.primary_metric")
            result = run["metrics"].get(primary) if primary else None
            result_text = f" {primary}={result:g}" if isinstance(result, (int, float)) else ""
            lines.append(
                f"  #{run['number']} [{run['worker_id']}] {run['status']} {run['run_id'][:8]}{result_text}"
            )
    return "\n".join(lines)


def format_experiment_comparison(
    *,
    project_slug: str,
    experiment: str,
    records: list[dict[str, Any]],
    primary_metric: str,
    contract: dict[str, object],
) -> str:
    """Render a focused view declared by an optional tracking contract.

    This is additive: callers can still use ``run show`` for every autologged
    parameter, metric, tag, summary, and artifact.
    """

    selected = [record for record in records if record["experiment"] == experiment]
    if not selected:
        return f"Project: {project_slug}\nExperiment: {experiment}\nNo completed runs found."

    metric_names = _contract_names(contract, "comparison_metrics")
    if not metric_names:
        metric_names = _contract_names(contract, "required_metrics")
    if primary_metric not in metric_names:
        metric_names.insert(0, primary_metric)
    parameter_names = _contract_names(contract, "comparison_parameters")
    headers = ["Run", "Label", "Worker", "Status", *parameter_names, *metric_names]
    rows = []
    for record in selected:
        rows.append(
            [
                f"#{record['number']}",
                _table_value(record["run_name"]),
                _table_value(record["worker_id"]),
                _table_value(record["status"]),
                *[_table_value(record["parameters"].get(name)) for name in parameter_names],
                *[_table_value(record["metrics"].get(name)) for name in metric_names],
            ]
        )
    lines = [f"Project: {project_slug}", f"Experiment: {experiment}", ""]
    lines.extend(_format_table(headers, rows))
    lines.extend(
        [
            "",
            "Fields come from tracking_contract comparison_* (or required_metrics).",
            "Use 'colab-mlflow run show' for complete autologged details.",
        ]
    )
    return "\n".join(lines)


def select_run(
    records: list[dict[str, Any]], *, experiment: str, number: int
) -> dict[str, Any]:
    for record in records:
        if record["experiment"] == experiment and record["number"] == number:
            return record
    raise LookupError(f"Run #{number} was not found in experiment '{experiment}'.")


def format_run_detail(project_slug: str, record: dict[str, Any]) -> str:
    """Render full traceability while artifact bytes stay on Drive until requested."""

    tags = record["tags"]
    lines = [
        f"Project: {project_slug}",
        f"Experiment: {record['experiment']}",
        f"Run: #{record['number']} ({record['run_id']})",
        f"Worker: {record['worker_id']}",
        f"Status: {record['status']}",
        f"Repository: {tags.get('source.repository', '-')}",
        f"Commit: {tags.get('source.commit', '-')}",
        f"Notebook: {tags.get('source.notebook', '-')}",
        f"Pipeline: {tags.get('source.pipeline', '-')}",
        f"Manifest: {record.get('manifest_path', '-')}",
        f"Artifact store: {record['artifact_uri']}",
        "",
        "Datasets:",
    ]
    datasets = record.get("datasets", {})
    for alias, dataset in sorted(datasets.items()):
        checksum = dataset.get("sha256") or dataset.get("fingerprint") or "-"
        lines.append(f"  {alias}: {dataset.get('path', '-')} (fingerprint: {checksum})")
    if not datasets:
        lines.append("  -")
    lines.extend(["", "Parameters:"])
    lines.extend(f"  {key}: {value}" for key, value in sorted(record["parameters"].items()))
    if not record["parameters"]:
        lines.append("  -")
    lines.extend(["", "Metrics:"])
    lines.extend(f"  {key}: {value}" for key, value in sorted(record["metrics"].items()))
    if not record["metrics"]:
        lines.append("  -")
    lines.extend(["", "Result summary:"])
    summary = record.get("result_summary")
    if summary is None:
        lines.append("  -")
    else:
        lines.extend(
            f"  {line}" for line in json.dumps(summary, ensure_ascii=False, indent=2).splitlines()
        )
    lines.extend(["", "Log preview:"])
    previews = record.get("log_preview", {})
    for path, content in previews.items():
        lines.append(f"  [{path}]")
        lines.extend(f"    {line}" for line in content.splitlines())
    if not previews:
        lines.append("  -")
    lines.extend(["", "Artifacts:"])
    lines.extend(f"  {path}" for path in record["artifacts"])
    if not record["artifacts"]:
        lines.append("  -")
    return "\n".join(lines)


def _log_previews(root: Path, limit: int = 4000) -> dict[str, str]:
    previews: dict[str, str] = {}
    for relative in ("logs/stdout.log", "logs/stderr.log", "logs/exception.log"):
        path = root / relative
        if not path.is_file():
            continue
        with path.open("rb") as stream:
            stream.seek(0, 2)
            size = stream.tell()
            stream.seek(max(0, size - limit))
            previews[relative] = stream.read().decode("utf-8", errors="replace")
    return previews


def _contract_names(contract: dict[str, object], key: str) -> list[str]:
    values = contract.get(key, [])
    if not isinstance(values, list):
        return []
    return [value for value in values if isinstance(value, str) and value]


def _table_value(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.6g}"
    rendered = str(value).replace("\n", " ")
    return rendered if len(rendered) <= 40 else f"{rendered[:37]}..."


def _format_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))
    separator = "-+-".join("-" * width for width in widths)
    rendered = [" | ".join(header.ljust(widths[index]) for index, header in enumerate(headers)), separator]
    rendered.extend(
        " | ".join(value.ljust(widths[index]) for index, value in enumerate(row)) for row in rows
    )
    return rendered
