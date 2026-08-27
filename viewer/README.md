# Trajectory viewer

Export a completed record with the Habitat environment used by the run:

```bash
PYTHONPATH=src envs/habitat-r2r/bin/python scripts/export_trajectory.py \
  --manifest runs/r2r_janusvln_smoke/manifest.json \
  --runner config/runners/r2r_janusvln_smoke.yaml \
  --output viewer/data
```

Serve the repository root and open `/viewer/`:

```bash
python -m http.server 8787 --bind 127.0.0.1
```
