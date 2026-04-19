"""
Signal validation — sample masks, trial quality, subject aggregation.
"""

import numpy as np
import pandas as pd

from .constants import (
    DEFAULT_MAD_MULTIPLIER,
    DEFAULT_SQI_THRESHOLD,
    PUPIL_DIAMETER_MAX_MM,
    PUPIL_DIAMETER_MIN_MM,
    QUALITY_RATIO_WEIGHTS,
    QUALITY_SCORE_BINS,
    QUALITY_SCORE_LABELS,
)


def _mad(x):
    """Median absolute deviation."""
    return np.nanmedian(np.abs(x - np.nanmedian(x)))


# ── sample-level masks ──────────────────────────────────────────────────────

def pupil_valid_mask(series, k=DEFAULT_MAD_MULTIPLIER):
    """Non-missing pupil diameter in [2, 8] and within median ± k·MAD."""
    med = np.nanmedian(series)
    mad = _mad(series)
    in_range = series.between(PUPIL_DIAMETER_MIN_MM, PUPIL_DIAMETER_MAX_MM)
    in_mad_band = series.between(med - k * mad, med + k * mad)
    return series.notna() & in_range & in_mad_band


def bpm_valid_mask(sqi, threshold=DEFAULT_SQI_THRESHOLD):
    """PPG signal quality above threshold."""
    return sqi.notna() & (sqi >= threshold)


def gaze_valid_mask(gx, gy, gz, k=DEFAULT_MAD_MULTIPLIER):
    """Non-missing gaze without velocity jumps (MAD-based)."""
    present = gx.notna() & gy.notna() & gz.notna()
    velocity = np.sqrt(gx.diff()**2 + gy.diff()**2 + gz.diff()**2)
    vel_med = np.nanmedian(velocity)
    vel_mad = _mad(velocity)
    # first sample has NaN velocity → keep it
    vel_ok = velocity.isna() | (velocity <= vel_med + k * vel_mad)
    return present & vel_ok


def motion_valid_mask(motion, k=DEFAULT_MAD_MULTIPLIER):
    """Motion magnitude within median + k·MAD."""
    med = np.nanmedian(motion)
    return motion.notna() & (motion <= med + k * _mad(motion))


def compute_sample_masks(df, k=DEFAULT_MAD_MULTIPLIER, sqi_threshold=DEFAULT_SQI_THRESHOLD):
    """Add boolean mask columns for each modality, combined with motion."""
    out = df.copy()
    out["pupil_valid"] = pupil_valid_mask(out["PupilDiameter"], k)
    out["bpm_valid"] = bpm_valid_mask(out["PPG_SQI"], sqi_threshold)
    out["gaze_valid"] = gaze_valid_mask(out["GazeX"], out["GazeY"], out["GazeZ"], k)
    out["motion_valid"] = motion_valid_mask(out["MotionMag"], k)

    # modality ∧ motion
    out["pupil_mask"] = out["pupil_valid"] & out["motion_valid"]
    out["gaze_mask"] = out["gaze_valid"] & out["motion_valid"]
    out["bpm_mask"] = out["bpm_valid"] & out["motion_valid"]
    return out


# ── trial-level quality ─────────────────────────────────────────────────────

def trial_quality_metrics(df):
    """Per-trial ratio of valid samples for each modality."""
    g = df.groupby(["SubjectID", "CycleID"])
    return pd.DataFrame({
        "pupil_valid_ratio": g["pupil_mask"].mean(),
        "bpm_valid_ratio": g["bpm_mask"].mean(),
        "gaze_valid_ratio": g["gaze_mask"].mean(),
    })


def trial_quality_score(metrics):
    """Weighted quality score → discrete {0, 1, 2}."""
    out = metrics.copy()
    out["quality_continuous"] = (
        QUALITY_RATIO_WEIGHTS["pupil_valid_ratio"] * out["pupil_valid_ratio"]
        + QUALITY_RATIO_WEIGHTS["bpm_valid_ratio"] * out["bpm_valid_ratio"]
        + QUALITY_RATIO_WEIGHTS["gaze_valid_ratio"] * out["gaze_valid_ratio"]
    )
    out["quality_score"] = pd.cut(
        out["quality_continuous"],
        bins=QUALITY_SCORE_BINS,
        labels=QUALITY_SCORE_LABELS,
    ).astype(int)
    return out


# ── subject-level aggregation ───────────────────────────────────────────────

def subject_quality(trial_scores):
    """Mean quality score and number of valid trials per subject."""
    g = trial_scores.groupby("SubjectID")
    return pd.DataFrame({
        "mean_quality": g["quality_score"].mean(),
        "n_valid_trials": g["quality_score"].apply(lambda s: (s > 0).sum()),
    })


# ── convenience pipeline ────────────────────────────────────────────────────

def validate_signals(df, k=DEFAULT_MAD_MULTIPLIER, sqi_threshold=DEFAULT_SQI_THRESHOLD):
    """Run full validation: masks → trial scores → subject scores."""
    df_masked = compute_sample_masks(df, k, sqi_threshold)
    metrics = trial_quality_metrics(df_masked)
    t_scores = trial_quality_score(metrics)
    s_scores = subject_quality(t_scores)
    return df_masked, t_scores, s_scores
