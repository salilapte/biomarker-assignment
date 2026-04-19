"""
Pipelines — preprocessing and analysis.

Two independent, parameterised functions designed for both CLI and API use:

- :func:`run_preprocessing` — raw CSVs → validated features + trial scores
- :func:`run_analysis` — features → plots (``output/plots/``) + tables (``output/results/``)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # headless — safe for API / CLI

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .constants import DEFAULT_PLOT_DPI, DEFAULT_TOP_N_FEATURES
from .features import aggregate_subject_features, extract_all_trial_features
from .plotting import (
    plot_candidate_summary,
    plot_candidate_loadings,
    plot_confound_corr,
    plot_feature_scatter,
    plot_pca_colored,
    plot_permutation_null,
    plot_quality_overview,
    plot_raw_signals,
    plot_scree,
)
from .selection import (
    build_candidate_table,
    build_relevance_table,
    dual_correlation,
    feature_target_correlation,
    fit_pca,
    format_h5_display,
    metadata_group_tests,
    permutation_test,
    remove_near_constant,
    remove_redundant_correlated,
    zscore_standardize,
)
from .validation import validate_signals


# ── helpers ──────────────────────────────────────────────────────────────────

def _save_fig(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, bbox_inches="tight", dpi=DEFAULT_PLOT_DPI)
    plt.close(fig)


# ── preprocessing pipeline ──────────────────────────────────────────────────

def run_preprocessing(
    raw_dir: str | Path = "data/raw",
    output_dir: str | Path = "data/processed",
) -> dict[str, Any]:
    """Raw CSVs → validated, aggregated, de-duplicated features.

    Returns a dict of in-memory DataFrames so :func:`run_analysis` can chain
    without re-reading parquet (also usable standalone via paths).
    """
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load
    subjects = pd.read_csv(raw_dir / "subjects.csv")
    timeseries = pd.read_csv(raw_dir / "timeseries.csv")
    print(f"Loaded {len(subjects)} subjects, {len(timeseries)} timeseries rows")

    # 2. Signal validation
    df_masked, trial_scores, subject_scores = validate_signals(timeseries)
    excluded = (trial_scores["quality_score"] == 0).sum()
    print(f"Validation: {len(trial_scores)} trials scored, {excluded} excluded (score=0)")
    trial_scores.to_parquet(output_dir / "trial_scores.parquet")

    # 3. Feature extraction
    trial_features = extract_all_trial_features(df_masked, trial_scores)
    print(f"Extracted {trial_features.shape[1]} features × {len(trial_features)} valid trials")

    # 3b. Save per-trial feature catalog
    trial_features.to_csv(output_dir / "per_trial_feature_catalog.csv")

    # 4. Subject-level aggregation
    subject_features = aggregate_subject_features(trial_features, agg_func="median", min_quality=1)
    print(f"Aggregated to {len(subject_features)} subjects, {subject_features.shape[1]} columns")

    # 5. Redundancy removal
    feature_cols = [c for c in subject_features.columns if c != "n_trials_used"]
    features = subject_features[feature_cols].copy()
    features, dropped_const = remove_near_constant(features)
    features, dropped_corr = remove_redundant_correlated(features)
    print(
        f"Redundancy removal: dropped {len(dropped_const)} near-constant, "
        f"{len(dropped_corr)} correlated → {features.shape[1]} remain"
    )

    # 6. Save
    features = features.join(subject_scores)
    features.to_parquet(output_dir / "features.parquet")
    print(f"\nPreprocessing artifacts → {output_dir}/")
    print(f"  features.parquet               ({features.shape[0]} × {features.shape[1]})")
    print(f"  trial_scores.parquet           ({len(trial_scores)} rows)")
    print(f"  per_trial_feature_catalog.csv  ({len(trial_features)} rows)")

    return {
        "features": features,
        "trial_scores": trial_scores,
        "subjects": subjects,
        "timeseries": timeseries,
    }


# ── analysis pipeline ───────────────────────────────────────────────────────

def run_analysis(
    output_dir: str | Path = "output",
    *,
    data: dict[str, Any] | None = None,
    processed_dir: str | Path = "data/processed",
    raw_dir: str | Path = "data/raw",
) -> None:
    """Feature exploration, correlation, PCA → saved plots and summary tables.

    Parameters
    ----------
    output_dir : root output folder (plots/ and results/ created underneath)
    data : optional dict from :func:`run_preprocessing` to avoid re-reading
    processed_dir : fallback path to load parquet files when *data* is None
    raw_dir : fallback path to load raw CSVs when *data* is None
    """
    output_dir = Path(output_dir)
    plots_dir = output_dir / "plots"
    results_dir = output_dir / "results"
    plots_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── load data ───────────────────────────────────────────────────────
    if data is not None:
        features = data["features"]
        trial_scores = data["trial_scores"]
        subjects = data["subjects"]
        timeseries = data["timeseries"]
    else:
        processed_dir = Path(processed_dir)
        raw_dir = Path(raw_dir)
        features = pd.read_parquet(processed_dir / "features.parquet")
        trial_scores = pd.read_parquet(processed_dir / "trial_scores.parquet")
        subjects = pd.read_csv(raw_dir / "subjects.csv")
        timeseries = pd.read_csv(raw_dir / "timeseries.csv")

    # Split quality cols back out
    quality_cols = ["mean_quality", "n_valid_trials"]
    subject_scores = features[quality_cols]
    features = features.drop(columns=quality_cols)

    print("── Analysis pipeline ──────────────────────────────────────")
    print(f"Features: {features.shape[0]} subjects × {features.shape[1]} features")

    # ── Plot 1: raw signals ─────────────────────────────────────────────
    sample_subjects = timeseries["SubjectID"].unique()[:3]
    fig = plot_raw_signals(timeseries, sample_subjects, cycle_ids=[1, 2])
    _save_fig(fig, plots_dir / "raw_signals.png")
    del timeseries  # free memory

    # ── Plot 2: quality overview ────────────────────────────────────────
    fig = plot_quality_overview(trial_scores, subject_scores, subjects_df=subjects)
    _save_fig(fig, plots_dir / "quality_overview.png")

    # ── Z-score standardize ─────────────────────────────────────────────
    features_z, zscore_params = zscore_standardize(features)

    # ── Correlations ────────────────────────────────────────────────────
    stai = subjects.set_index("SubjectID")[["STAI_S", "STAI_T"]]
    stai_spearman, _stai_pearson, stai_stable = dual_correlation(
        features_z, stai.loc[features_z.index], top_n=DEFAULT_TOP_N_FEATURES,
    )

    confounds = subjects.set_index("SubjectID")[["CalibrationError"]].loc[features_z.index]
    confounds["mean_quality"] = subject_scores.loc[features_z.index, "mean_quality"]
    confound_corr = feature_target_correlation(features_z, confounds, method="spearman")

    # ── Relevance table ─────────────────────────────────────────────
    tbl, top_union, floor, candidates = build_relevance_table(
        stai_spearman, stai_stable, confound_corr, top_n=DEFAULT_TOP_N_FEATURES,
    )

    # ── Plot 3: confound correlations ─────────────────────────────
    fig = plot_confound_corr(
        confound_corr,
        confound_cols=("CalibrationError", "mean_quality"),
        top_n=DEFAULT_TOP_N_FEATURES,
        stable=stai_stable,
    )
    _save_fig(fig, plots_dir / "confound_corr.png")

    # ── Plot 4: feature scatter ─────────────────────────────────────────
    cal_error = subjects.set_index("SubjectID")["CalibrationError"].loc[features_z.index]
    cal_error.name = "CalibrationError"
    figs = plot_feature_scatter(
        features_z,
        stai.loc[features_z.index],
        color_by=cal_error,
        target_cols=["STAI_S", "STAI_T"],
        top_n=3,
        stai_feature_correlations=stai_spearman,
    )
    for fig, suffix in zip(figs, ["stai_s", "stai_t"]):
        _save_fig(fig, plots_dir / f"feature_scatter_{suffix}.png")

    # ── Permutation test — biomarker sensitivity ────────────────────────
    perm_observed, perm_pvalues, perm_null = permutation_test(
        features_z, stai.loc[features_z.index], sorted(candidates),
    )
    figs = plot_permutation_null(
        perm_observed, perm_null, perm_pvalues, target_cols=["STAI_S", "STAI_T"],
    )
    for fig, suffix in zip(figs, ["stai_s", "stai_t"]):
        _save_fig(fig, plots_dir / f"permutation_null_{suffix}.png")

    # ── PCA ─────────────────────────────────────────────────────────────
    pca_scores, pca_var, pca_loadings = fit_pca(features_z)

    # ── Plot 5: scree ──────────────────────────────────────────────────
    fig = plot_scree(pca_var)
    _save_fig(fig, plots_dir / "scree.png")

    # ── Plot 6: PCA colored ─────────────────────────────────────────────
    color_vars = {
        "STAI_S": stai.loc[features_z.index, "STAI_S"],
        "STAI_T": stai.loc[features_z.index, "STAI_T"],
        "CalibrationError": cal_error,
        "mean_quality": subject_scores.loc[features_z.index, "mean_quality"],
    }
    fig = plot_pca_colored(pca_scores, color_vars)
    _save_fig(fig, plots_dir / "pca_colored.png")

    # ── PC–target correlations ──────────────────────────────────────────
    n_pcs = min(5, pca_scores.shape[1])
    pc_cols = pca_scores[pca_scores.columns[:n_pcs]]
    targets = stai.loc[features_z.index].copy()
    targets["CalibrationError"] = cal_error
    targets["mean_quality"] = subject_scores.loc[features_z.index, "mean_quality"]
    pc_corr = feature_target_correlation(pc_cols, targets, method="spearman")

    # ── Loadings table + H5 metadata tests ──────────────────────────────
    candidate_tbl, cand_feats = build_candidate_table(
        pca_loadings, stai_spearman, candidates, stai_stable,
        confound_corr, floor, pcs=("PC2", "PC3"), top_n=DEFAULT_TOP_N_FEATURES,
    )
    meta = subjects.set_index("SubjectID")[["Gender", "WearsGlasses", "Handedness", "BloodType"]]
    meta = meta.loc[features_z.index]
    h5_pvals, h5_effects = metadata_group_tests(features_z[cand_feats], meta)

    binary_cols = ["Gender", "WearsGlasses", "Handedness"]
    multi_cols = ["BloodType"]
    h5_display = format_h5_display(h5_pvals, h5_effects, binary_cols, multi_cols)
    candidate_tbl = candidate_tbl.join(h5_display, on="feature")

    # ── Plot 7: candidate loadings bar chart ───────────────────────────
    fig = plot_candidate_loadings(
        pca_loadings, candidates, pcs=("PC2", "PC3"), top_n=DEFAULT_TOP_N_FEATURES,
    )
    _save_fig(fig, plots_dir / "candidate_loadings.png")

    # ── Plot 8: candidate summary (correlations + H5 effect sizes) ────
    fig = plot_candidate_summary(
        candidates, stai_spearman, h5_effects, h5_pvals,
    )
    _save_fig(fig, plots_dir / "candidate_summary.png")

    # ── Save tables ─────────────────────────────────────────────────────
    tbl.to_csv(results_dir / "relevance_table.csv")
    candidate_tbl.to_csv(results_dir / "loadings_summary.csv")
    pc_corr.to_csv(results_dir / "pc_target_correlations.csv")
    zscore_params.to_csv(results_dir / "zscore_params.csv")

    print(f"\nPlots  → {plots_dir}/  ({len(list(plots_dir.glob('*.png')))} files)")
    print(f"Tables → {results_dir}/  ({len(list(results_dir.glob('*.csv')))} files)")
