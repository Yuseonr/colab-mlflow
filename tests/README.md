# Test map

The suite is organized by observable behavior rather than implementation
detail:

- `test_manifest.py`: the completed-run JSON contract, identity safety, and the
  guarantee that metadata polling does not read artifact bytes.
- `test_notebook_runtime.py`: a generated notebook publishes successful,
  failed, and nested hyperparameter-tuning runs into the new Drive layout.
- `test_sync.py`: real MLflow integration from manifest to verified SQLite,
  including metric history, native dataset inputs, logged models, stable run
  IDs, and lazy artifact URIs.
- `test_tracking.py`: multi-worker status and human-readable run inspection.
- `test_server.py`: startup synchronization, SQLite-backed UI launch, and safe
  periodic restart.
- `test_cli.py`: visible global configuration, bootstrap/project/setup/dataset/
  notebook commands, and editable-cell regeneration behavior.

Run everything with the dependency-backed environment:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Each scenario uses Given/When/Then comments where setup, action, and assertion
would otherwise be ambiguous. Temporary directories stand in for Drive and
local Linux/macOS state; no test writes to the real workspace.
