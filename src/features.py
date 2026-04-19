"""
Feature extraction — 39 per-trial features from pupil, BPM, gaze, cross-modal signals.
"""

import numpy as np
import pandas as pd
from scipy import stats

# Safe defaults — fixed by design, not user-tunable
PHASE_NAMES = ("baseline", "relax", "break")
CORRELATION_MIN_SAMPLES = 5
PERCENTILE_LOW = 10
PERCENTILE_HIGH = 90
DEFAULT_AGG_FUNC = "median"
IQR_LOWER_QUANTILE = 0.25
IQR_UPPER_QUANTILE = 0.75
MIN_VALID_QUALITY_SCORE = 1


# ── helpers ──────────────────────────────────────────────────────────────────

def _masked(values, mask):
    """Values where mask is True, NaNs dropped."""
    return values.loc[mask].dropna().values


def _safe_mean(y):
    """Mean that returns NaN for empty arrays without warnings."""
    return np.mean(y) if len(y) > 0 else np.nan


def _slope(y):
    """Linear slope over integer index."""
    if len(y) < 2:
        return np.nan
    x = np.arange(len(y), dtype=float)
    return stats.linregress(x, y).slope


def _detrended_std(y):
    """Std of residuals after removing linear trend."""
    if len(y) < 3:
        return np.nan
    x = np.arange(len(y), dtype=float)
    reg = stats.linregress(x, y)
    residuals = y - (reg.slope * x + reg.intercept)
    return np.std(residuals, ddof=1)


def _p90_p10(y):
    if len(y) == 0:
        return np.nan
    return np.percentile(y, PERCENTILE_HIGH) - np.percentile(y, PERCENTILE_LOW)


def _corr(a, b):
    """Pearson r between two arrays, NaN if too few points or zero variance."""
    valid = ~(np.isnan(a) | np.isnan(b))
    a, b = a[valid], b[valid]
    if len(a) < CORRELATION_MIN_SAMPLES:
        return np.nan
    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return np.corrcoef(a, b)[0, 1]


# ── per-trial feature extractors ────────────────────────────────────────────

def _pupil_features(baseline, relax, brk):
    bl_mean = _safe_mean(baseline)
    rl_mean = _safe_mean(relax)
    bk_mean = _safe_mean(brk)

    half = len(relax) // 2
    third = len(relax) // 3
    early = relax[:half]
    late = relax[half:]
    mid = relax[third:2*third]

    early_mean = _safe_mean(early)
    late_mean = _safe_mean(late)
    mid_mean = _safe_mean(mid)

    return {
        "pupil_baseline_mean": bl_mean,
        "pupil_relax_mean": rl_mean,
        "pupil_break_mean": bk_mean,
        "pupil_delta": rl_mean - bl_mean,
        "pupil_relative_delta": (rl_mean - bl_mean) / bl_mean if bl_mean != 0 else np.nan,
        "pupil_rebound": bk_mean - rl_mean,
        "pupil_relax_slope": _slope(relax),
        "pupil_relax_std": np.std(relax, ddof=1) if len(relax) > 1 else np.nan,
        "pupil_relax_detrended_std": _detrended_std(relax),
        "pupil_relax_p90_p10": _p90_p10(relax),
        "pupil_auc_relax": np.mean(relax - bl_mean) * 30.0 if len(relax) > 0 and np.isfinite(bl_mean) else np.nan,
        "pupil_early_mean": early_mean,
        "pupil_late_mean": late_mean,
        "pupil_early_late_diff": late_mean - early_mean,
        "pupil_mid_mean": mid_mean,
        "pupil_nonlinearity": mid_mean - (early_mean + late_mean) / 2,
    }


def _bpm_features(baseline, relax, brk):
    bl_mean = _safe_mean(baseline)
    rl_mean = _safe_mean(relax)
    bk_mean = _safe_mean(brk)

    half = len(relax) // 2
    early_mean = _safe_mean(relax[:half])
    late_mean = _safe_mean(relax[half:])

    return {
        "bpm_baseline_mean": bl_mean,
        "bpm_relax_mean": rl_mean,
        "bpm_break_mean": bk_mean,
        "bpm_delta": rl_mean - bl_mean,
        "bpm_rebound": bk_mean - rl_mean,
        "bpm_relax_slope": _slope(relax),
        "bpm_relax_std": np.std(relax, ddof=1) if len(relax) > 1 else np.nan,
        "bpm_relax_p90_p10": _p90_p10(relax),
        "bpm_early_mean": early_mean,
        "bpm_early_late_diff": late_mean - early_mean,
    }


def _gaze_features(gx_bl, gy_bl, gx_rl, gy_rl, gz_rl):
    xy_var_bl = (np.var(gx_bl, ddof=1) + np.var(gy_bl, ddof=1)) if len(gx_bl) > 1 else np.nan
    xy_var_rl = (np.var(gx_rl, ddof=1) + np.var(gy_rl, ddof=1)) if len(gx_rl) > 1 else np.nan

    if len(gx_rl) < 2:
        return {
            "gaze_xy_var_baseline": xy_var_bl,
            "gaze_xy_var_relax": xy_var_rl,
            "gaze_xy_var_delta": xy_var_rl - xy_var_bl if np.isfinite(xy_var_rl) and np.isfinite(xy_var_bl) else np.nan,
            "gaze_z_var_relax": np.var(gz_rl, ddof=1) if len(gz_rl) > 1 else np.nan,
            "gaze_xy_p90_p10": np.nan,
            "gaze_z_p90_p10": np.nan,
            "gaze_drift": np.nan,
            "gaze_velocity_mean": np.nan,
        }

    xy_disp = np.sqrt(gx_rl**2 + gy_rl**2)

    # drift: euclidean distance from first to last relax sample
    drift = np.sqrt(
        (gx_rl[-1] - gx_rl[0])**2
        + (gy_rl[-1] - gy_rl[0])**2
        + (gz_rl[-1] - gz_rl[0])**2
    )

    # mean frame-to-frame displacement
    velocity = np.sqrt(np.diff(gx_rl)**2 + np.diff(gy_rl)**2 + np.diff(gz_rl)**2)

    return {
        "gaze_xy_var_baseline": xy_var_bl,
        "gaze_xy_var_relax": xy_var_rl,
        "gaze_xy_var_delta": xy_var_rl - xy_var_bl,
        "gaze_z_var_relax": np.var(gz_rl, ddof=1),
        "gaze_xy_p90_p10": _p90_p10(xy_disp),
        "gaze_z_p90_p10": _p90_p10(gz_rl),
        "gaze_drift": drift,
        "gaze_velocity_mean": np.mean(velocity),
    }


def _cross_modal_features(pupil_relax, bpm_relax):
    half = len(pupil_relax) // 2

    corr_full = _corr(pupil_relax, bpm_relax)
    corr_early = _corr(pupil_relax[:half], bpm_relax[:half])
    corr_late = _corr(pupil_relax[half:], bpm_relax[half:])

    return {
        "pupil_bpm_corr": corr_full,
        "pupil_bpm_corr_early": corr_early,
        "pupil_bpm_corr_late": corr_late,
        "pupil_bpm_corr_diff": corr_late - corr_early,
    }


# ── main extraction ─────────────────────────────────────────────────────────

def extract_trial_features(trial_df, quality_score=None):
    """Extract all 39 features from a single-trial DataFrame."""
    phases = {}
    for phase in PHASE_NAMES:
        phases[phase] = trial_df[trial_df["Phase"].str.lower() == phase]

    def _get(phase, col, mask_col):
        sub = phases[phase]
        return _masked(sub[col], sub[mask_col])

    pupil_bl = _get("baseline", "PupilDiameter", "pupil_mask")
    pupil_rl = _get("relax", "PupilDiameter", "pupil_mask")
    pupil_bk = _get("break", "PupilDiameter", "pupil_mask")

    bpm_bl = _get("baseline", "PulseBPM", "bpm_mask")
    bpm_rl = _get("relax", "PulseBPM", "bpm_mask")
    bpm_bk = _get("break", "PulseBPM", "bpm_mask")

    gx_bl = _get("baseline", "GazeX", "gaze_mask")
    gy_bl = _get("baseline", "GazeY", "gaze_mask")
    gx_rl = _get("relax", "GazeX", "gaze_mask")
    gy_rl = _get("relax", "GazeY", "gaze_mask")
    gz_rl = _get("relax", "GazeZ", "gaze_mask")

    # For cross-modal features, align pupil and BPM to the same time points
    relax = phases["relax"]
    both_valid = relax["pupil_mask"] & relax["bpm_mask"]
    pupil_rl_aligned = relax.loc[both_valid, "PupilDiameter"].dropna().values
    bpm_rl_aligned = relax.loc[both_valid, "PulseBPM"].dropna().values

    feats = {}
    feats.update(_pupil_features(pupil_bl, pupil_rl, pupil_bk))
    feats.update(_bpm_features(bpm_bl, bpm_rl, bpm_bk))
    feats.update(_gaze_features(gx_bl, gy_bl, gx_rl, gy_rl, gz_rl))
    feats.update(_cross_modal_features(pupil_rl_aligned, bpm_rl_aligned))

    if quality_score is not None:
        feats["quality_score"] = quality_score

    return feats


def extract_all_trial_features(df_masked, trial_scores=None):
    """Extract features for every (SubjectID, CycleID)."""
    rows = []
    for (sid, cid), group in df_masked.groupby(["SubjectID", "CycleID"]):
        qs = None
        if trial_scores is not None:
            qs = int(trial_scores.loc[(sid, cid), "quality_score"])
            if qs == 0:
                continue
        feats = extract_trial_features(group, quality_score=qs)
        feats["SubjectID"] = sid
        feats["CycleID"] = cid
        rows.append(feats)

    return pd.DataFrame(rows).set_index(["SubjectID", "CycleID"])


# ── subject-level aggregation ───────────────────────────────────────────────

def aggregate_subject_features(
    trial_features,
    agg_func=DEFAULT_AGG_FUNC,
    min_quality=MIN_VALID_QUALITY_SCORE,
):
    """Aggregate trial features to subject level using median (default).

    Filters by quality_score >= min_quality. Also computes trial-to-trial IQR.
    """
    feature_cols = [c for c in trial_features.columns if c != "quality_score"]

    if "quality_score" in trial_features.columns:
        valid = trial_features[trial_features["quality_score"] >= min_quality]
    else:
        valid = trial_features

    agg = valid.groupby("SubjectID")[feature_cols].agg(agg_func)
    n_trials = valid.groupby("SubjectID").size().rename("n_trials_used")

    iqr = valid.groupby("SubjectID")[feature_cols].agg(
        lambda s: s.quantile(IQR_UPPER_QUANTILE) - s.quantile(IQR_LOWER_QUANTILE)
    )
    iqr = iqr.add_suffix("_iqr")

    return agg.join(n_trials).join(iqr)
