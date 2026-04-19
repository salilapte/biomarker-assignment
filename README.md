# Biomarker Assignment — Physiological Markers of Relaxation

Identifies physiological biomarkers of relaxation from VR eye-tracking data and
assesses their relationship with self-reported anxiety (STAI).

## Quick start

### Option A — uv (recommended)

[uv](https://docs.astral.sh/uv/) manages the virtualenv and dependencies in one step:

```bash
# Install uv (once)
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Create venv + install deps
uv sync

# Activate the venv
# macOS / Linux
source .venv/bin/activate
# Windows (Git Bash)
source .venv/Scripts/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

### Option B — pip

```bash
python -m venv .venv

# Activate
# macOS / Linux
source .venv/bin/activate
# Windows (Git Bash)
source .venv/Scripts/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

pip install -e .
```

### Run

```bash
python main.py
```

This runs both pipelines sequentially:
1. **Preprocessing** → `data/processed/`
2. **Analysis** → `output/plots/` + `output/results/`

The interactive notebook is a separate exploration layer:

```bash
jupyter notebook analysis.ipynb
```

## Project structure

```
main.py                  Thin entry point — calls both pipelines
analysis.ipynb           Interactive exploration & visualisation
src/
  pipeline.py            run_preprocessing() + run_analysis()
  constants.py           Shared thresholds and defaults
  validation.py          Sample masks, trial quality, subject scores
  features.py            39 per-trial feature extractors
  selection.py           Redundancy removal, z-score, PCA, table builders
  plotting.py            All figure & styled-table functions
data/
  raw/                   Input CSVs (subjects.csv, timeseries.csv)
  processed/             Pipeline outputs (features.parquet, trial_scores.parquet)
docs/
  approach.md            Analysis plan, hypotheses, feature definitions
  problem.md             Original assignment brief
output/
  plots/                 Saved figures (PNG, 150 dpi)
  results/               Summary tables (CSV)
```

## Expected input

Place raw CSV files in `data/raw/`:
- `subjects.csv` — participant metadata (SubjectID, STAI_T, STAI_S, ...)
- `timeseries.csv` — physiological signals (SubjectID, DeviceTimestamp, CycleID, ...)

## Pipeline outputs

### Preprocessing (`data/processed/`)

| File | Description |
|------|-------------|
| `features.parquet` | Subject-level features + quality summary |
| `trial_scores.parquet` | Per-trial quality scores |
| `per_trial_feature_catalog.csv` | All 39 features for every valid trial |

### Analysis — plots (`output/plots/`)

| File | Description |
|------|-------------|
| `raw_signals.png` | Representative pupil + BPM traces with phase shading |
| `quality_overview.png` | Valid trials histogram, quality by cycle, calibration vs quality |
| `confound_corr.png` | Top feature correlations with confounders |
| `feature_scatter_stai_s.png` | Top features vs STAI_S scatter |
| `feature_scatter_stai_t.png` | Top features vs STAI_T scatter |
| `scree.png` | PCA explained variance |
| `pca_colored.png` | PC1 vs PC2 colored by STAI, calibration, quality |

### Analysis — tables (`output/results/`)

| File | Description |
|------|-------------|
| `relevance_table.csv` | Top STAI features with stability + confound flags |
| `loadings_summary.csv` | PC2/PC3 loadings + H5 metadata tests |
| `pc_target_correlations.csv` | PCs vs STAI + confounders (Spearman) |
| `zscore_params.csv` | Feature mean/std for re-standardization (API-ready) |

## Backend integration

Both pipelines are parameterised functions in `src/pipeline.py`, designed for
direct use from a backend framework:

```python
from fastapi import FastAPI
from src.pipeline import run_preprocessing, run_analysis

app = FastAPI()

@app.post("/preprocess")
def preprocess(raw_dir: str = "data/raw", output_dir: str = "data/processed"):
    data = run_preprocessing(raw_dir=raw_dir, output_dir=output_dir)
    return {"status": "ok", "subjects": data["features"].shape[0]}

@app.post("/analyze")
def analyze(processed_dir: str = "data/processed", output_dir: str = "output"):
    run_analysis(output_dir, processed_dir=processed_dir)
    return {"status": "ok", "output_dir": output_dir}
```

**Scaling considerations:**

- **Input validation** — Check that uploaded CSVs have the expected columns and
  value ranges (e.g., via Pydantic schemas) before the pipeline runs, to fail fast
  with clear error messages.
- **Background processing** — With large datasets, preprocessing could take too long
  for a synchronous HTTP request. A background task queue would let the API accept
  the request immediately and return a job ID, with a status endpoint to poll for
  completion.
- **Cloud storage** — In production, raw uploads and outputs (parquet, plots) would
  be stored in object storage (e.g., S3) rather than the local filesystem.
- **Incremental updates** — When new participants are added, only the new subjects
  need feature extraction. The saved z-score parameters (`zscore_params.csv`) allow
  standardizing new data against the original distribution without reprocessing
  everything.
- **Reproducibility** — Pipeline parameters (thresholds, feature definitions) should
  be versioned alongside code (e.g., git tags) so that any result can be traced back
  to the exact configuration that produced it.

## Dependencies

- pandas, numpy, scipy, matplotlib — core data stack
- scikit-learn — PCA
- pyarrow — parquet I/O
- jinja2 — pandas Styler rendering
