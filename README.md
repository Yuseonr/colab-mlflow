# colab-mlflow

colab-mlflow generates standalone Google Colab notebooks for reproducible ML
experiments. Each ML project remains in its own Git repository. Colab trains on
temporary local storage, publishes completed runs to Google Drive, and a local
Linux/macOS computer rebuilds SQLite metadata for the native MLflow UI.

Generated notebooks never import the colab-mlflow package.

## Supported local platform

Run the CLI on Linux or macOS with a local directory that is synchronized or
mounted from Google Drive. Google Drive Desktop is one option on macOS; on
Linux, use an equivalent Drive sync or mount solution. Windows is not currently
supported because SQLite coordination uses Linux/macOS file locking.

## Architecture

~~~text
Git project
  └── experiments/<experiment>/run.ipynb
                    │
                    ▼
Google Colab
  ├── clones GitHub and records the exact commit
  ├── trains with temporary MLflow storage in /content
  ├── reads immutable datasets from Google Drive
  └── publishes completed runs
                    │
                    ▼
Google Drive
  ├── datasets/<dataset>/<version>/
  └── runs/<project>/<experiment>/<run-id>/
      ├── manifest.json
      └── artifacts/     models, outputs, logs, source, notebook
                    │
                    ▼
Local Linux/macOS computer
  ├── .state/projects/<project>/mlflow.db
  ├── native MLflow UI on localhost
  └── Google Drive Desktop fetches artifact bytes on demand
~~~

Drive holds datasets and real artifacts. SQLite stores metadata only; models and
datasets are never copied into SQLite.

## Concepts

| Concept | Meaning |
|---|---|
| Project | One Git repository and broad ML objective. |
| Experiment | One stable objective, pipeline/model topology, preprocessing, and evaluation protocol. |
| Run | One execution; parameters, seed, dataset version, and worker may vary. |
| Worker | Colab compute metadata, not an experiment boundary. |

Use another run for parameter or seed trials. Use another experiment for a
material model, feature/preprocessing, split, or evaluation-protocol change.

## Install the tool once

Clone the tool into a persistent folder, edit its one visible configuration
file, then install the command globally in editable mode.

~~~bash
git clone <colab-mlflow-repository-url> ~/tools/colab-mlflow
cd ~/tools/colab-mlflow
cp .env.example .env
~~~

Edit .env:

~~~dotenv
COLAB_MLFLOW_LOCAL_STORAGE_ROOT=/path/to/Google Drive/My Drive/colab-mlflow-storage
COLAB_MLFLOW_COLAB_STORAGE_ROOT=/content/drive/MyDrive/colab-mlflow-storage
COLAB_MLFLOW_WORKER_ID=worker-a
# Keep SQLite local to this tool checkout, never inside Google Drive.
COLAB_MLFLOW_LOCAL_STATE_ROOT=/path/to/colab-mlflow/.state
~~~

SQLite must be outside Drive. Install and prepare the command:

~~~bash
uv tool install --editable --force .
colab-mlflow setup
colab-mlflow config show
~~~

Every project uses this tool-level .env. Projects do not need their own .env or
machine-specific Drive paths. Editing the tool .env takes effect immediately.

## Start a project

### Fast path: project and first experiment

Run from a new project directory, or add --root <directory>.

~~~bash
colab-mlflow bootstrap \
  --project house-pricing \
  --name "House Pricing" \
  --description "Predict house prices." \
  --repository https://github.com/<owner>/house-pricing.git \
  --experiment ridge-baseline \
  --type training \
  --objective "Train a Ridge baseline." \
  --primary-metric validation.rmse
~~~

Bootstrap prepares storage, initializes Git, links GitHub, creates the first
experiment, and generates its notebook. It never commits or pushes for you.

### Existing Git repository or staged setup

~~~bash
git clone https://github.com/<owner>/house-pricing.git
cd house-pricing

colab-mlflow init \
  --project house-pricing \
  --name "House Pricing" \
  --description "Predict house prices."
~~~

Init creates the project identity, contract documentation, AGENTS.md, and a
managed workflow section in the project README. If an origin is missing:

~~~bash
colab-mlflow link --repository https://github.com/<owner>/house-pricing.git
~~~

## Create experiments and notebooks

~~~bash
colab-mlflow experiment create \
  --slug hist-gradient-boosting \
  --type training \
  --objective "Predict house prices with histogram gradient boosting." \
  --primary-metric validation.rmse

colab-mlflow notebook generate --experiment hist-gradient-boosting
~~~

This creates:

~~~text
experiments/hist-gradient-boosting/
├── experiment.toml
├── pipeline.yaml
└── run.ipynb
~~~

Notebook generation creates or refreshes the managed tracking scaffold. It does
not create a run. It preserves designated editable cells. Review its diff,
commit, and push before opening the printed Colab URL.

## Version datasets

Datasets are immutable Drive folders. Create a new version instead of modifying
one used by a run.

~~~bash
colab-mlflow dataset init --slug house-prices --version kaggle-v2
~~~

~~~text
datasets/house-prices/
├── kaggle-v1/
└── kaggle-v2/
~~~

File inputs are tracked by SHA-256. Directory inputs are tracked by structural
fingerprint, file count, and total size.

### Image folders and Colab cache

Category folders work with loaders such as torchvision ImageFolder:

~~~text
datasets/animal-images/v1/train/
├── cat/001.jpg
├── dog/001.jpg
└── bird/001.jpg
~~~

~~~python
DATASET_PATHS = {
    "train_images": "/content/drive/MyDrive/colab-mlflow-storage/datasets/animal-images/v1/train",
}
DATASET_CACHE_MODE = "copy"  # none, copy, or archive
~~~

| Cache mode | Behavior |
|---|---|
| none | Default. The pipeline reads Drive directly. |
| copy | Copies a file/folder once to /content/cmf-datasets; later runs in that runtime reuse it. |
| archive | Extracts a ZIP or tar archive from Drive to /content; best for many small image files. |

The Drive path and fingerprint/SHA remain the tracked dataset. The pipeline
receives a local path for copy and archive modes. Cache is runtime-local and
disappears when Colab resets.

## Notebook and run flow

A generated notebook intentionally exposes these inputs:

~~~python
WORKER_ID = "worker-a"
RUN_LABEL = "ridge-seed-42"  # Empty uses an automatic timestamp name.
DATASET_CACHE_MODE = "none"
DATASET_PATHS = {
    "train": "/content/drive/MyDrive/colab-mlflow-storage/datasets/house-prices/kaggle-v2/train.csv",
}
~~~

It also exposes parameters and pipeline code:

~~~python
RUN_VARIANTS = {
    "worker-a": {"alpha": 1.0, "random_state": 42},
    "worker-b": {"alpha": 10.0, "random_state": 43},
}

def run_pipeline(dataset_paths, parameters, output_dir):
    # Store models, predictions, plots, and reports below output_dir.
    return {
        "metrics": {"validation.rmse": rmse},
        "summary": {"model_family": "Ridge"},
    }
~~~

For an important run:

~~~text
edit locally → git commit → git push → open Colab link → Runtime: Run all
~~~

Colab mounts Drive, clones the configured branch, records the exact commit, and
verifies the notebook matches GitHub before training.

There is no run-create CLI command: each notebook execution receives a unique
MLflow run ID. For run #2, change a parameter/seed and label, then execute the
same notebook. Do not create run-1.ipynb, run-2.ipynb, and so on. A snapshot of
the executed notebook is logged for auditability.

Two Colab runtimes can run concurrently. Give them different WORKER_ID values
or parameter presets; their unique runs are combined in the usual project view.

## Contract-aware tracking

Experiment contracts are flexible. Declare only fields that matter:

~~~toml
[tracking_contract]
decision_rule = "minimize validation.rmse"
comparison_parameters = ["alpha", "random_state"]
comparison_metrics = ["validation.rmse", "validation.mae"]
required_metrics = ["validation.rmse", "validation.mae"]
summary_fields = ["model_family", "training_rows"]
required_artifacts = ["model.joblib", "validation_predictions.csv"]
validation_mode = "warn" # or strict
~~~

- Compare uses comparison fields plus the primary metric.
- Warn publishes the run with a contract warning artifact.
- Strict publishes the evidence but marks an incomplete run as failed.
- Full MLflow autolog data is never removed.

AGENTS.md tells coding agents to read the contract before editing. The tool logs
source commit, contract hash, and notebook hash, but cannot automatically make
scientific experiment-boundary decisions from arbitrary code changes.

## What a run contains

Colab uses temporary MLflow storage and publishes only final runs:

~~~text
runs/<project>/<experiment>/<32-char-run-id>/
├── manifest.json
└── artifacts/
    ├── inputs/datasets.json
    ├── inputs/dataset-cache.json
    ├── logs/stdout.log
    ├── logs/stderr.log
    ├── notebook/run.ipynb
    ├── outputs/...
    ├── results/summary.json
    └── source/...
~~~

A run records final status, parameters, latest metrics and metric history, tags,
dataset fingerprints, cache details, source revision, notebook hash,
environment, summary, logs, exception information, output checksums, logged
models, and nested-run relationships. Failed runs are published too.

## Inspect, sync, and use MLflow UI

Run these inside a project:

~~~bash
colab-mlflow status

colab-mlflow run show \
  --experiment hist-gradient-boosting \
  --number 1

colab-mlflow compare --experiment hist-gradient-boosting

colab-mlflow sync

colab-mlflow server --port 5000 --sync-interval 300
~~~

| Command | Purpose |
|---|---|
| status | Fast manifest-only project and run summary. |
| run show | Full metadata, source, datasets, artifacts, and log previews for one run. |
| compare | Concise contract-aware view; never hides autolog details. |
| sync | Rebuilds/refreshes local SQLite from Drive manifests. |
| server | Syncs on startup and serves native local MLflow UI. |

Server checks manifests periodically. When metadata changes, it rebuilds SQLite
and restarts the UI safely. Do not run manual sync while server is active; a
local file lock protects the database.

## Full command reference

| Command | Example |
|---|---|
| Create fallback config | colab-mlflow config init ... |
| Show active config | colab-mlflow config show |
| Ensure storage folders | colab-mlflow setup |
| Project + first experiment | colab-mlflow bootstrap ... |
| Initialize project only | colab-mlflow init ... |
| Link GitHub origin | colab-mlflow link --repository <url> |
| Create dataset version | colab-mlflow dataset init --slug <name> --version <version> |
| Create experiment | colab-mlflow experiment create ... |
| Generate notebook | colab-mlflow notebook generate --experiment <slug> |
| List runs | colab-mlflow status |
| Inspect run | colab-mlflow run show --experiment <slug> --number 1 |
| Compare runs | colab-mlflow compare --experiment <slug> |
| Import Drive metadata | colab-mlflow sync |
| Open MLflow UI | colab-mlflow server |

## Current limits

- Colab execution is manual; the local Linux/macOS CLI does not start remote
  training.
- Runs appear after they finish and publish; there is no live progress UI.
- The local Linux/macOS computer only needs to be on for sync and the MLflow
  UI, not for Colab to publish completed Drive runs.
- The /content cache is temporary and has finite disk space.
- Archive mode needs enough Colab disk for the extracted dataset.
- Contract and agent guidance remain responsible for experiment-boundary
  decisions.
