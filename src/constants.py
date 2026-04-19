"""User-tunable constants — edit these when adapting the pipeline to a new dataset."""

import numpy as np


# ── Signal validation ────────────────────────────────────────────────────────

# Outlier detection sensitivity (higher = more permissive)
DEFAULT_MAD_MULTIPLIER = 5.0
# Minimum PPG signal-quality index to accept a heart-rate sample
DEFAULT_SQI_THRESHOLD = 0.4
# Physiological pupil diameter bounds in mm (doi:10.1167/12.10.12)
PUPIL_DIAMETER_MIN_MM = 2.0
PUPIL_DIAMETER_MAX_MM = 8.0
# Per-modality weights for the composite trial quality score
QUALITY_RATIO_WEIGHTS = {
    "pupil_valid_ratio": 0.4,
    "bpm_valid_ratio": 0.4,
    "gaze_valid_ratio": 0.2,
}
# Bin edges mapping continuous quality → discrete tiers
QUALITY_SCORE_BINS = (-np.inf, 0.6, 0.9, np.inf)
# Discrete labels matching the bins above
QUALITY_SCORE_LABELS = (0, 1, 2)


# ── Feature selection ────────────────────────────────────────────────────────

# Drop one of a correlated pair when |r| exceeds this
HIGH_CORRELATION_THRESHOLD = 0.95
# Number of top features to surface in tables and plots
DEFAULT_TOP_N_FEATURES = 10


# ── Dataset-specific columns ─────────────────────────────────────────────────

# Target anxiety scores from subjects.csv
DEFAULT_TARGET_COLS = ("STAI_S", "STAI_T")
# Metadata columns tested as potential confounders
DEFAULT_CONFOUNDER_COLS = ("Gender", "WearsGlasses", "Handedness", "BloodType")


# ── Output ───────────────────────────────────────────────────────────────────

# Saved figure resolution
DEFAULT_PLOT_DPI = 150