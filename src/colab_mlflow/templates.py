"""English templates for source projects and agent guidance."""

from __future__ import annotations


def agents_md() -> str:
    """Output: contract-first guidance for agents collaborating on experiments."""

    return """# Experiment collaboration context

Use these instructions when the user asks to discuss, recommend, create, or edit a generated experiment. The agent is a design and implementation collaborator; it must not silently redefine the user's experiment.

## Read the local context first

- Read `.colab-mlflow.toml`, `docs/experiment-contract.md`, and the target experiment's `experiment.toml` and `pipeline.yaml` for context.
- Treat `experiment.toml` and `pipeline.yaml` as versioned source artifacts. They are committed to Git, copied into every run artifact, and identified on every run by `experiment.contract_sha256`.
- Do not decide silently that a request belongs to a different experiment. Explain the mismatch and let the user choose.
- Do not run Git commands, push, manage Drive, start Colab, or create/delete/rename experiments unless the user explicitly asks.

## Contract-first checkpoint

Before recommending or editing pipeline logic, inspect whether the experiment contract answers the request. There is deliberately no universal metric, summary, artifact, or reproducibility schema: use only what is meaningful for this project and experiment.

- When the user has already stated the decisions, restate them briefly and offer to record them in the contract.
- When a material decision is missing, ask concise questions before changing code: what is being decided, which primary metric and direction matter, which other scalar metrics should be comparable, what context a human should see in the summary, which artifacts are needed, and which dataset/split/seed assumptions matter.
- A user may edit the contract manually. With explicit permission, an agent may draft or update `experiment.toml` and `pipeline.yaml` too. Explain the proposed contract diff; do not silently change an agreed contract.
- Record a contract revision in Git before an important run. A changed contract is a source change, not an untracked note.

## Implement the requested pipeline

- The `.ipynb` file is the primary executable artifact and remains manually editable.
- Preserve all `cmf-*` managed cells. Edit only cells tagged `cmf-parameters` and `cmf-user-code`, inside `<cmf:user-code>` boundaries.
- Implement or import `run_pipeline(dataset_paths, parameters, output_dir)`.
- Use `RUN_VARIANTS` for parameter presets. Multiple workers may select different presets while remaining runs of the same experiment.
- Do not guess or rewrite `WORKER_ID` or `DATASET_PATHS`; those are the user's Colab inputs.
- Put every generated model, plot, report, prediction, checkpoint, or export under `output_dir`. The wrapper logs that directory automatically.
- Return `{"metrics": {...}, "summary": {...}}`. Every agreed comparison quantity must be a scalar metric; the primary metric must be present in `metrics`. Use `summary` for explanatory, JSON-safe run context rather than quantities to rank.
- Use the optional `RUN_LABEL` for a human-readable run name; leave it empty to use the generated experiment-worker-timestamp name.
- Use framework-native training code normally; managed `mlflow.autolog()` captures supported framework detail. Autolog is supplementary: explicitly compute and return the metrics the contract requires, and use optional `mlflow.log_metric(..., step=...)` for custom training histories.
- Optional on-demand `mlflow.log_*` calls are allowed inside the pipeline. Keep portable models, plots, reports, predictions, and checkpoints under `output_dir` as well.
- Do not add manual MLflow boilerplate already covered by the wrapper: Git revision, repository URL, branch, notebook, environment, datasets, parameters, stdout, stderr, exceptions, outputs, and run summary.
- For parameter-only trials, create another run with the same source commit; do not create a commit for every hyperparameter value. For source or contract changes, recommend a new commit before an important run.

## Validate the edit

- Validate notebook JSON and compile the Python source of edited code cells.
- Run relevant local tests where practical.
- Summarize the contract decisions, files changed, metrics/artifacts expected, and any unresolved question.
"""


def claude_md() -> str:
    """Output: concise instructions for Claude Code."""

    return agents_md()


def skill_md() -> str:
    """Output: repository skill discoverable by coding agents."""

    return """---
name: colab-mlflow-experiment
description: Discuss and implement a tracked ML experiment contract, parameter presets, and pipeline code while preserving managed tracking cells.
---

# colab-mlflow experiment

Follow the root `AGENTS.md` experiment collaboration context. Start by reading the current contract, then ask about any material missing decision before proposing code. Keep contract changes in `experiment.toml` and `pipeline.yaml`, preserve managed notebook cells, and do not manage Git, Drive, or run history unless explicitly requested.
"""


def project_workflow_md() -> str:
    """Output: managed README guidance included in every initialized source project."""

    return """## colab-mlflow workflow

This repository is one tracked ML project. The global `colab-mlflow` command
uses the tool-level configuration; this project does not need its own `.env`.

### Starting a project

- `colab-mlflow bootstrap ...` is the quickest option for a new repository. It
  initializes this Git project, links its GitHub remote, creates the first
  experiment, and generates that experiment's Colab notebook. It never commits
  or pushes for you.
- `colab-mlflow init ...` initializes only the project identity, Git guidance,
  and shared experiment documentation. Use it when the first experiment should
  be discussed or created later.

### Adding and running experiments

- `colab-mlflow experiment create ...` creates one stable experiment definition
  under `experiments/<slug>/`. Use a new experiment for a material pipeline or
  evaluation-protocol change; use another run for parameter or seed trials.
- `colab-mlflow notebook generate --experiment <slug>` creates or refreshes
  `experiments/<slug>/run.ipynb`. The notebook contains editable Colab inputs,
  an editable pipeline area, and managed MLflow tracking/publishing cells.

For image folders, put categories in subfolders and use `DATASET_CACHE_MODE =
"copy"` when the pipeline should read a temporary local `/content` cache rather
than thousands of individual Drive files. The tracked dataset remains the
original Drive path and fingerprint.

Commit and push the generated notebook before opening its printed Colab link.
Use `colab-mlflow status`, `compare`, `run show`, `sync`, and `server` from the
project root to inspect tracked runs.
"""


def experiment_contract_md() -> str:
    """Output: the shared project contract."""

    return """# Experiment contract

## Hierarchy

- Project: one Git repository and broad ML objective.
- Experiment: one stable objective, model or pipeline topology, preprocessing,
  loss composition, and evaluation protocol.
- Run: one execution of an experiment. Parameters, random seed, dataset version,
  and compute worker may vary without creating a new experiment.
- Worker: one Colab runtime or Google account used by the same person for
  concurrent compute. It is run metadata, not another project, experiment, or
  user identity.

Create a new run for parameter or seed changes within the same pipeline. A
material architecture, preprocessing flow, loss, evaluation protocol, or
training-versus-inference change belongs to a new experiment. The notebook
editor must explain such a mismatch instead of changing experiment definitions.

## Contract lifecycle

An experiment contract is intentionally flexible. It is not a universal list of
metrics or artifacts. It records only the decisions useful for this experiment:
for example, the primary decision metric and direction, required comparison
metrics, summary context, expected artifacts, dataset/split assumptions, or
reproducibility controls.

- Discuss a missing or ambiguous contract decision before implementing a
  material pipeline change. An agent should ask rather than invent a scientific
  objective, metric, or summary policy.
- Record agreed decisions in the experiment's `experiment.toml` and/or
  `pipeline.yaml`. The user may edit them manually or authorize an agent to do
  so.
- Commit contract changes before an important run. Each run records the source
  commit, includes both contract files as artifacts, and exposes
  `experiment.contract_sha256` in MLflow so a run can be connected to the
  exact contract revision.
- Parameter-only trials normally create new runs under one unchanged contract.
  Source or contract changes should produce a new Git commit before an
  important run.

An optional shape, to use only when it helps, is:

```toml
[tracking_contract]
status = "draft" # draft, agreed, or revised
decision_rule = "minimize validation.rmse"
required_metrics = ["validation.rmse", "validation.mae"]
comparison_parameters = ["random_state", "alpha"]
comparison_metrics = ["validation.rmse", "validation.mae"]
summary_fields = ["model_family", "training_rows", "validation_rows"]
required_artifacts = ["model.joblib", "validation_predictions.csv"]
validation_mode = "warn" # use "strict" to fail an incomplete run
```

The field names, number of fields, and values may differ by experiment. Scalar
values used to compare or rank runs belong in MLflow metrics; summary fields
provide explanatory context and are stored as structured run output.

## Reproducibility rules

- Code is written locally and committed to Git before an important run.
- The generated notebook clones the configured GitHub branch and records the exact commit it executes.
- Every run automatically records its repository, commit, notebook, runtime environment, explicit Drive dataset paths and fingerprints, complete MLflow parameters and metric history, worker, result summary, stdout, stderr, exceptions, logged models, and output artifacts.
- Dataset paths are not registered or copied by this tool. Their ownership and lifecycle remain in Google Drive, so make each version path immutable once it has been used in a run.
- Initialize a dataset version with `colab-mlflow dataset init --slug <name> --version <version>`, then copy its files below `datasets/<name>/<version>/` in the configured Drive storage.
- Concurrent Colab runtimes use unique MLflow run IDs and publish immutable directories under `runs/<project>/<experiment>/<run-id>/`; worker identifiers remain searchable run metadata.
- MLflow writes its many small metadata files only to temporary Colab storage. At completion, the notebook publishes one human-readable `manifest.json` plus real artifacts to Drive.
- The local inspector reads manifests without downloading artifact bytes. The SQL-backed UI periodically rebuilds a local Linux/macOS SQLite database from those manifests, while artifact paths continue to point at the streamed Drive mount.
"""
