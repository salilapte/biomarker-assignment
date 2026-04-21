"""
Plotting utilities for biomarker exploration and quality assessment.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

from .constants import DEFAULT_CONFOUNDER_COLS, DEFAULT_TARGET_COLS, DEFAULT_TOP_N_FEATURES

# ── Cosmetic / fixed plotting defaults ───────────────────────────────────────
MICROSECONDS_PER_SECOND = 1e6
DEFAULT_CYCLE_IDS = (1,)
PHASE_COLORS = {"baseline": "#7fb3d8", "relax": "#a8d5a2", "break": "#f4a582"}
QUALITY_LABELS = {0: "poor", 1: "acceptable", 2: "good"}
QUALITY_COLORS = {0: "#d73027", 1: "#fee08b", 2: "#1a9850"}
QUALITY_PLOT_ORDER = (2, 1, 0)
QUALITY_SUBJECT_EXTREME_FRACTION = 0.05
POSITIVE_CORRELATION_COLOR = "#2166ac"
NEGATIVE_CORRELATION_COLOR = "#b2182b"
DEFAULT_FEATURE_SCATTER_TOP_N = 3
DEFAULT_TOP_N_BIOMARKERS = 15
DEFAULT_RAW_SIGNALS_FIGSIZE = (14, 4)
DEFAULT_RAW_SIGNALS_PRESENTATION_FIGSIZE = (14, 4)
MOTION_CMAP = "RdYlGn_r"
DEFAULT_QUALITY_OVERVIEW_FIGSIZE = (16, 4.5)
DEFAULT_CORRELATION_MATRIX_FIGSIZE = (12, 10)
DEFAULT_CORRELATION_MATRIX_TITLE = "Feature correlation matrix"
DEFAULT_PCA_FIGSIZE = (7, 6)
DEFAULT_PCA_TITLE = "PCA"
DEFAULT_CLUSTER_CONFOUNDERS_FIGSIZE = (14, 6)
DEFAULT_CLUSTER_CONFOUNDERS_TITLE = "Cluster vs. potential confounders"
DEFAULT_BIOMARKER_SENSITIVITY_FIGSIZE = (10, 6)
DEFAULT_BIOMARKER_SENSITIVITY_TITLE = "Biomarker sensitivity (top features by |correlation|)"


# ── Figure 1: raw signals ───────────────────────────────────────────────────

def plot_raw_signals(df, subject_ids, cycle_ids=None, figsize=DEFAULT_RAW_SIGNALS_FIGSIZE):
    """Pupil + BPM traces for selected subjects/trials with phase shading."""
    cycle_ids = cycle_ids or list(DEFAULT_CYCLE_IDS)
    n = len(subject_ids)
    fig, axes = plt.subplots(n, 2, figsize=(figsize[0], figsize[1] * n), squeeze=False)

    for row, sid in enumerate(subject_ids):
        sub = df[df["SubjectID"] == sid]
        for cid in cycle_ids:
            trial = sub[sub["CycleID"] == cid]
            t0 = trial["DeviceTimestamp"].iloc[0]
            t_sec = (trial["DeviceTimestamp"] - t0) / MICROSECONDS_PER_SECOND

            # shade phases
            for ax in axes[row]:
                for phase, color in PHASE_COLORS.items():
                    mask = trial["Phase"].str.lower() == phase
                    if mask.any():
                        t_phase = t_sec[mask]
                        ax.axvspan(t_phase.iloc[0], t_phase.iloc[-1],
                                   alpha=0.2, color=color,
                                   label=phase if cid == cycle_ids[0] else None)

            axes[row, 0].plot(t_sec.values, trial["PupilDiameter"].values, lw=0.7, label=f"C{cid}")
            axes[row, 1].plot(t_sec.values, trial["PulseBPM"].values, lw=0.7, label=f"C{cid}")

        axes[row, 0].set_ylabel("Pupil (mm)")
        axes[row, 1].set_ylabel("BPM")
        axes[row, 0].set_title(f"{sid} — Pupil Diameter")
        axes[row, 1].set_title(f"{sid} — Pulse BPM")

    for ax in axes[-1]:
        ax.set_xlabel("Time (s)")
    for ax in axes.flat:
        ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    return fig


# ── Figure 1b: raw signals by quality tier (presentation) ───────────────────

def plot_raw_signals_by_quality(
    df,
    trial_scores,
    subject_scores,
    figsize=DEFAULT_RAW_SIGNALS_PRESENTATION_FIGSIZE,
):
    """Presentation figure: 1×3 grid, one panel per quality tier (high / medium / low subject).

    Subject selection:
      - High:   subject whose mean_quality is closest to 2.0
      - Medium: subject whose mean_quality is closest to 1.2 (excluding the high/low picks)
      - Low:    subject with the lowest mean_quality

    Each panel shows pupil diameter only.  All other trials are drawn as faint grey
    context traces.  The highlighted trial (whose quality_score matches the tier)
    is coloured by MotionMag using the RdYlGn_r colormap (green = calm, red = high motion).
    """
    ts = trial_scores.reset_index()  # ensures SubjectID, CycleID are columns

    sq = subject_scores["mean_quality"]
    high_sid = (sq - 2.0).abs().idxmin()
    low_sid = sq.idxmin()
    remaining = sq.drop([high_sid, low_sid])
    med_sid = (remaining - 1.2).abs().idxmin()

    tier_subjects = [
        (high_sid, 2, "High quality"),
        (med_sid,  1, "Medium quality"),
        (low_sid,  0, "Low quality"),
    ]

    motion_vmin = df["MotionMag"].quantile(0.02)
    motion_vmax = df["MotionMag"].quantile(0.98)
    cmap = plt.cm.get_cmap(MOTION_CMAP)
    norm = plt.Normalize(vmin=motion_vmin, vmax=motion_vmax)

    n_tiers = len(tier_subjects)
    fig, axes = plt.subplots(
        1, n_tiers,
        figsize=(figsize[0], figsize[1]),
        constrained_layout=True,
        squeeze=False,
    )
    axes = axes[0]  # shape (n_tiers,)

    for col, (sid, target_score, tier_label) in enumerate(tier_subjects):
        ax = axes[col]
        sub = df[df["SubjectID"] == sid]
        sub_ts = ts[ts["SubjectID"] == sid].copy()
        all_cycle_ids = sorted(sub["CycleID"].dropna().unique().tolist())

        # Pick highlighted cycle matching the tier score; fall back to closest
        matching = sub_ts[sub_ts["quality_score"] == target_score]
        if matching.empty:
            sub_ts["_dist"] = (sub_ts["quality_score"] - target_score).abs()
            matching = sub_ts.nsmallest(1, "_dist")
        highlight_cid = int(matching.iloc[0]["CycleID"])

        # Phase shading from highlighted trial time axis
        hl_trial = sub[sub["CycleID"] == highlight_cid]
        t0_hl = hl_trial["DeviceTimestamp"].iloc[0]
        t_hl = (hl_trial["DeviceTimestamp"] - t0_hl) / MICROSECONDS_PER_SECOND

        for phase, color in PHASE_COLORS.items():
            mask = hl_trial["Phase"].str.lower() == phase
            if mask.any():
                t_seg = t_hl[mask]
                ax.axvspan(t_seg.iloc[0], t_seg.iloc[-1], alpha=0.18, color=color)

        # Context traces — all other cycles, faint grey
        for cid in all_cycle_ids:
            if cid == highlight_cid:
                continue
            trial = sub[sub["CycleID"] == cid]
            if trial.empty:
                continue
            t0_c = trial["DeviceTimestamp"].iloc[0]
            t_c = (trial["DeviceTimestamp"] - t0_c) / MICROSECONDS_PER_SECOND
            ax.plot(t_c.values, trial["PupilDiameter"].values,
                    lw=0.6, color="#9ca3af", alpha=0.20, zorder=1)

        # Highlighted trial coloured by MotionMag via LineCollection
        motion = hl_trial["MotionMag"].values

        def _motion_lc(t, y, _ax, _motion=motion, _cmap=cmap, _norm=norm):
            valid = np.isfinite(y.astype(float)) & np.isfinite(t.astype(float))
            tv = t[valid].astype(float)
            yv = y[valid].astype(float)
            mv = _motion[valid].astype(float)
            if len(tv) < 2:
                return None
            pts = np.array([tv, yv]).T.reshape(-1, 1, 2)
            segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
            motion_mid = (mv[:-1] + mv[1:]) / 2
            lc = LineCollection(segs, cmap=_cmap, norm=_norm, linewidth=1.8, zorder=3)
            lc.set_array(motion_mid)
            _ax.add_collection(lc)
            _ax.autoscale_view()
            return lc

        _motion_lc(t_hl.values, hl_trial["PupilDiameter"].values, ax)

        # Labels
        q_row = sub_ts[sub_ts["CycleID"] == highlight_cid]
        if "quality_continuous" in q_row.columns and len(q_row):
            q_val = f"{float(q_row['quality_continuous'].values[0]):.2f}"
        else:
            q_val = "?"
        score_str = QUALITY_LABELS.get(target_score, str(target_score))

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Pupil diameter (mm)" if col == 0 else "")
        ax.set_title(f"{tier_label}\n{sid}  C{highlight_cid}  q={q_val}  ({score_str})")
        ax.axvline(5,  color="#6b7280", lw=0.8, ls="--", alpha=0.6)
        ax.axvline(35, color="#6b7280", lw=0.8, ls="--", alpha=0.6)

    # Shared y-axis limits: clamp to physiological range, ignore −1 sentinel values
    from .constants import PUPIL_DIAMETER_MIN_MM, PUPIL_DIAMETER_MAX_MM
    all_ylims = [ax.get_ylim() for ax in axes]
    y_min = max(min(lo for lo, _ in all_ylims), PUPIL_DIAMETER_MIN_MM - 0.5)
    y_max = min(max(hi for _, hi in all_ylims), PUPIL_DIAMETER_MAX_MM + 0.5)
    for ax in axes:
        ax.set_ylim(y_min, y_max)

    # Legend on first panel — bottom-left avoids covering the pupil trace
    phase_handles = [
        mpatches.Patch(color=c, alpha=0.35, label=p.capitalize())
        for p, c in PHASE_COLORS.items()
    ]
    context_handle = Line2D([0], [0], color="#9ca3af", lw=1.0, alpha=0.5, label="other trials")
    axes[0].legend(handles=phase_handles + [context_handle], fontsize=7, loc="lower left")

    # MotionMag colorbar anchored to the last panel only
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[-1], shrink=0.85, pad=0.02)
    cbar.set_label("Motion magnitude", fontsize=8)

    fig.suptitle("Pupil diameter by signal quality tier", fontsize=10)
    return fig


# ── Figure 2: quality overview (3-panel) ────────────────────────────────────

def plot_quality_overview(
    trial_scores,
    subject_scores,
    subjects_df=None,
    figsize=DEFAULT_QUALITY_OVERVIEW_FIGSIZE,
):
    """Histogram of valid trials, quality proportions by cycle, calibration error scatter."""
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # A — valid trials per subject
    ax = axes[0]
    counts = subject_scores["n_valid_trials"]
    ax.hist(counts, bins=range(0, int(counts.max()) + 2), edgecolor="black", alpha=0.75)
    ax.set_xlabel("Valid trials")
    ax.set_ylabel("Subjects")
    ax.set_title("A) Valid trials per subject")

    # B — quality score proportions per CycleID
    ax = axes[1]
    ts = trial_scores.reset_index()
    proportions = ts.groupby("CycleID")["quality_score"].value_counts(normalize=True).unstack(fill_value=0)
    for score in QUALITY_PLOT_ORDER:
        if score in proportions.columns:
            ax.plot(proportions.index, proportions[score], marker="o", markersize=4,
                    label=QUALITY_LABELS[score], color=QUALITY_COLORS[score])

    ax.set_xlabel("CycleID")
    ax.set_ylabel("Proportion")
    ax.set_title("B) Quality score by cycle")
    ax.legend(fontsize=7)

    # C — calibration error vs quality
    ax = axes[2]
    if subjects_df is not None:
        merged = subject_scores.join(subjects_df.set_index("SubjectID")[["CalibrationError"]])
        ax.scatter(merged["CalibrationError"], merged["mean_quality"], s=20, alpha=0.6)
        ax.set_xlabel("Calibration Error")
        ax.set_ylabel("Mean quality score")
        ax.set_title("C) Calibration error vs quality")
    else:
        ax.set_visible(False)

    fig.tight_layout()
    return fig


# ── Correlation heatmap ─────────────────────────────────────────────────────

def plot_correlation_matrix(
    features,
    cols=None,
    figsize=DEFAULT_CORRELATION_MATRIX_FIGSIZE,
    title=DEFAULT_CORRELATION_MATRIX_TITLE,
):
    """Pearson correlation heatmap."""
    if cols is None:
        cols = [c for c in features.select_dtypes(include=[np.number]).columns
                if not c.endswith("_iqr") and c != "n_trials_used"]
    corr = features[cols].corr()

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=90, fontsize=7)
    ax.set_yticklabels(cols, fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(title)
    fig.tight_layout()
    return fig


# ── PCA scatter ─────────────────────────────────────────────────────────────

def plot_pca(
    pca_result,
    labels=None,
    explained_variance=None,
    figsize=DEFAULT_PCA_FIGSIZE,
    title=DEFAULT_PCA_TITLE,
):
    """Scatter of first two PCA components, optionally colored by labels."""
    fig, ax = plt.subplots(figsize=figsize)

    if labels is not None:
        unique = sorted(labels.unique())
        cmap = plt.cm.get_cmap("tab10", len(unique))
        for i, val in enumerate(unique):
            mask = labels.values == val
            ax.scatter(pca_result[mask, 0], pca_result[mask, 1],
                       s=25, alpha=0.7, color=cmap(i), label=str(val))
        ax.legend(fontsize=8)
    else:
        ax.scatter(pca_result[:, 0], pca_result[:, 1], s=25, alpha=0.7)

    xlabel = "PC1"
    ylabel = "PC2"
    if explained_variance is not None:
        xlabel += f" ({explained_variance[0]:.1%})"
        ylabel += f" ({explained_variance[1]:.1%})"
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    return fig


# ── Cluster scatter with confounders ────────────────────────────────────────

def plot_cluster_confounders(pca_result, cluster_labels, metadata,
                             confounder_cols=DEFAULT_CONFOUNDER_COLS,
                             figsize=DEFAULT_CLUSTER_CONFOUNDERS_FIGSIZE):
    """2x2 scatter colored by potential confounders, cluster boundaries overlaid."""
    n = len(confounder_cols)
    ncols = min(n, 2)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

    for idx, col in enumerate(confounder_cols):
        ax = axes[idx // ncols, idx % ncols]
        vals = metadata[col].astype(str)
        unique = sorted(vals.unique())
        cmap = plt.cm.get_cmap("Set2", len(unique))
        for i, v in enumerate(unique):
            m = vals.values == v
            ax.scatter(pca_result[m, 0], pca_result[m, 1], s=20, alpha=0.6, color=cmap(i), label=v)
        # cluster outlines
        for cl in np.unique(cluster_labels):
            m = cluster_labels == cl
            ax.scatter(pca_result[m, 0], pca_result[m, 1],
                       s=60, facecolors="none", edgecolors="black", linewidths=0.5, alpha=0.3)
        ax.set_title(col)
        ax.legend(fontsize=7)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")

    # hide unused
    for idx in range(n, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    fig.suptitle(DEFAULT_CLUSTER_CONFOUNDERS_TITLE, fontsize=12)
    fig.tight_layout()
    return fig


# ── Biomarker sensitivity bar chart ─────────────────────────────────────────

def plot_biomarker_sensitivity(
    correlations,
    target_cols=DEFAULT_TARGET_COLS,
    top_n=DEFAULT_TOP_N_BIOMARKERS,
    figsize=DEFAULT_BIOMARKER_SENSITIVITY_FIGSIZE,
):
    """Horizontal bar chart of top features by |correlation| with STAI scores."""
    fig, axes = plt.subplots(1, len(target_cols), figsize=figsize, sharey=True)
    if len(target_cols) == 1:
        axes = [axes]

    for ax, col in zip(axes, target_cols):
        vals = correlations[col].dropna().abs().nlargest(top_n).sort_values()
        colors = [
            POSITIVE_CORRELATION_COLOR if correlations.loc[f, col] >= 0 else NEGATIVE_CORRELATION_COLOR
            for f in vals.index
        ]
        ax.barh(vals.index, vals.values, color=colors, edgecolor="black", linewidth=0.4)
        ax.set_xlabel(f"|r| with {col}")
        ax.set_title(col)
        pos_patch = mpatches.Patch(color=POSITIVE_CORRELATION_COLOR, label="positive r")
        neg_patch = mpatches.Patch(color=NEGATIVE_CORRELATION_COLOR, label="negative r")
        ax.legend(handles=[pos_patch, neg_patch], fontsize=7)

    fig.suptitle(DEFAULT_BIOMARKER_SENSITIVITY_TITLE, fontsize=12)
    fig.tight_layout()
    return fig


# ── Feature-target correlation bar chart ────────────────────────────────────

def plot_feature_target_corr(
    correlations,
    target_cols=DEFAULT_TARGET_COLS,
    top_n=DEFAULT_TOP_N_FEATURES,
    stable=None,
    figsize=(12, 5),
    title="Top feature correlations with targets",
):
    """Horizontal bar chart of top-*n* features by |correlation| with each target.

    If *stable* is provided (a set of feature names), those bars get a star marker.
    """
    n_targets = len(target_cols)
    fig, axes = plt.subplots(1, n_targets, figsize=figsize, sharey=False)
    if n_targets == 1:
        axes = [axes]

    for ax, col in zip(axes, target_cols):
        vals = correlations[col].dropna().abs().nlargest(top_n).sort_values()
        colors = [
            POSITIVE_CORRELATION_COLOR if correlations.loc[f, col] >= 0 else NEGATIVE_CORRELATION_COLOR
            for f in vals.index
        ]
        bars = ax.barh(vals.index, vals.values, color=colors, edgecolor="black", linewidth=0.4)

        # Mark stable features (top-N under both Spearman & Pearson)
        if stable:
            for bar, feat in zip(bars, vals.index):
                if feat in stable:
                    ax.text(
                        bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                        "\u2605", va="center", fontsize=9, color="goldenrod",
                    )

        ax.set_xlabel(f"|r| with {col}")
        ax.set_title(col)
        handles = [
            mpatches.Patch(color=POSITIVE_CORRELATION_COLOR, label="positive r"),
            mpatches.Patch(color=NEGATIVE_CORRELATION_COLOR, label="negative r"),
        ]
        if stable:
            handles.append(mpatches.Patch(facecolor="white", edgecolor="goldenrod", label="\u2605 stable"))
        ax.legend(handles=handles, fontsize=7)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    return fig


# ── Confound correlation bar chart ──────────────────────────────────────────

def plot_confound_corr(
    correlations,
    confound_cols=("CalibrationError", "mean_quality"),
    top_n=DEFAULT_TOP_N_FEATURES,
    stable=None,
    figsize=(12, 5),
    title="Top feature correlations with confounders",
):
    """Horizontal bar chart of top features correlated with confounders."""
    return plot_feature_target_corr(
        correlations,
        target_cols=confound_cols,
        top_n=top_n,
        stable=stable,
        figsize=figsize,
        title=title,
    )


# ── Feature vs target scatter ───────────────────────────────────────────────

def plot_feature_scatter(
    features,
    targets,
    color_by,
    target_cols=DEFAULT_TARGET_COLS,
    top_n=DEFAULT_FEATURE_SCATTER_TOP_N,
    stai_feature_correlations=None,
    figsize_per_panel=(4.5, 4),
):
    """Scatter of top features vs each target, colored by *color_by* Series.

    If *stai_feature_correlations* is provided, the top features are selected
    per target column by |correlation|. Otherwise the first *top_n* columns
    of *features* are used.
    """
    figs = []
    for target_col in target_cols:
        if stai_feature_correlations is not None and target_col in stai_feature_correlations.columns:
            top_feats = (
                stai_feature_correlations[target_col]
                .dropna().abs().nlargest(top_n).index.tolist()
            )
        else:
            top_feats = features.columns[:top_n].tolist()

        n = len(top_feats)
        fig, axes = plt.subplots(
            1, n,
            figsize=(figsize_per_panel[0] * n, figsize_per_panel[1]),
            constrained_layout=True,
            squeeze=False,
        )

        for idx, feat in enumerate(top_feats):
            ax = axes[0, idx]
            sc = ax.scatter(
                features[feat], targets[target_col],
                c=color_by, cmap="coolwarm", s=20, alpha=0.7, edgecolors="none",
            )
            # trend line
            mask = features[feat].notna() & targets[target_col].notna()
            if mask.sum() > 2:
                z = np.polyfit(features.loc[mask, feat], targets.loc[mask, target_col], 1)
                x_line = np.linspace(features[feat].min(), features[feat].max(), 50)
                ax.plot(x_line, np.polyval(z, x_line), "--", color="grey", lw=1)
            ax.set_xlabel(feat, fontsize=8)
            ax.set_ylabel(target_col if idx == 0 else "")
            ax.tick_params(labelsize=7)

        fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.7, pad=0.02, label=color_by.name)
        fig.suptitle(f"Top features vs {target_col}", fontsize=11)
        figs.append(fig)
    return figs


# ── PCA plots ───────────────────────────────────────────────────────────────

def plot_scree(explained_variance_ratio, figsize=(8, 4)):
    """Scree curve with cumulative explained variance."""
    n = len(explained_variance_ratio)
    cumulative = np.cumsum(explained_variance_ratio)

    fig, ax1 = plt.subplots(figsize=figsize)
    ax1.bar(range(1, n + 1), explained_variance_ratio, color="#4393c3", edgecolor="black", lw=0.4, label="Individual")
    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Explained variance ratio")

    ax2 = ax1.twinx()
    ax2.plot(range(1, n + 1), cumulative, "o-", color="#d6604d", markersize=4, label="Cumulative")
    ax2.set_ylabel("Cumulative variance")
    ax2.set_ylim(0, 1.05)

    # combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="center right")

    ax1.set_title("Scree plot — explained variance per PC")
    fig.tight_layout()
    return fig


def plot_pca_colored(scores, color_series_dict, figsize_per_panel=(5, 4.5)):
    """PC1 vs PC2 scatter, one panel per continuous color variable.

    Parameters
    ----------
    scores : DataFrame with PC1, PC2 columns (subjects × PCs)
    color_series_dict : dict  name → Series aligned to scores.index
    """
    n = len(color_series_dict)
    fig, axes = plt.subplots(1, n, figsize=(figsize_per_panel[0] * n, figsize_per_panel[1]), squeeze=False)

    for idx, (name, series) in enumerate(color_series_dict.items()):
        ax = axes[0, idx]
        vals = series.loc[scores.index]
        sc = ax.scatter(scores["PC1"], scores["PC2"], c=vals, cmap="coolwarm", s=20, alpha=0.7, edgecolors="none")
        fig.colorbar(sc, ax=ax, shrink=0.75, label=name)
        ax.set_xlabel(scores.columns[0])
        ax.set_ylabel(scores.columns[1])
        ax.set_title(name)

    fig.suptitle("PC1 vs PC2", fontsize=12)
    fig.tight_layout()
    return fig


def plot_loadings(loadings, pcs=("PC1", "PC2"), top_n=10, figsize=(12, 5)):
    """Horizontal bar chart of top loading features per PC."""
    n_pcs = len(pcs)
    fig, axes = plt.subplots(1, n_pcs, figsize=figsize, sharey=False)
    if n_pcs == 1:
        axes = [axes]

    for ax, pc in zip(axes, pcs):
        vals = loadings[pc].abs().nlargest(top_n).sort_values()
        colors = [POSITIVE_CORRELATION_COLOR if loadings.loc[f, pc] >= 0 else NEGATIVE_CORRELATION_COLOR for f in vals.index]
        ax.barh(vals.index, vals.values, color=colors, edgecolor="black", lw=0.4)
        ax.set_xlabel(f"|loading| on {pc}")
        ax.set_title(pc)

    fig.suptitle("Top PCA loadings", fontsize=12)
    fig.tight_layout()
    return fig


# ── Styled table display ────────────────────────────────────────────────────

def style_loadings_table(loadings_tbl, meta_cols):
    """Apply conditional styling to the candidate summary table.

    Green = candidate, amber = quality-driven, red cell = meaningful
    metadata effect (⚠ requires both p < 0.05 and |d| ≥ 0.3 / η² ≥ 0.04).
    """
    def _color_row(row):
        n = len(row)
        base = [""] * n
        if row.get("stable") is True and row.get("quality-driven") is False:
            base = ["background-color: #d4edda"] * n
        elif row.get("quality-driven") is True:
            base = ["background-color: #fff3cd"] * n
        return base

    def _color_h5(val):
        if isinstance(val, str) and "⚠" in val:
            return "background-color: #f8d7da"
        return ""

    return (
        loadings_tbl.style
        .apply(_color_row, axis=1)
        .map(_color_h5, subset=meta_cols)
        .set_caption(
            "Candidate biomarkers & other top PC loaders — green = candidate, "
            "amber = quality-driven, red cell = meaningful metadata effect "
            "(p < 0.05 AND |d| ≥ 0.3 / η² ≥ 0.04)"
        )
    )


# ── Permutation null distribution ───────────────────────────────────────────

def plot_permutation_null(observed, null_distributions, pvalues, target_cols=DEFAULT_TARGET_COLS, figsize_per_panel=(5, 4)):
    """Histogram of null |ρ| with observed |ρ| marked, 2×3 grid per target."""
    import math
    figs = []
    for col in target_cols:
        feat_names = observed.index.tolist()
        n = len(feat_names)
        ncols = min(n, 3)
        nrows = math.ceil(n / ncols)
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
            squeeze=False,
        )
        null = null_distributions[col]

        for idx, feat in enumerate(feat_names):
            r, c = divmod(idx, ncols)
            ax = axes[r, c]
            null_vals = np.abs(null[:, idx])
            obs_val = abs(observed.loc[feat, col])
            p = pvalues.loc[feat, col]

            ax.hist(null_vals, bins=40, color="#bdbdbd", edgecolor="white", lw=0.3)
            ax.axvline(obs_val, color="#d73027", lw=2, label=f"observed |ρ|={obs_val:.3f}")
            ax.set_title(f"{feat}\np={p:.4f}", fontsize=8)
            ax.set_xlabel("|ρ|")
            if c == 0:
                ax.set_ylabel("Count")
            ax.legend(fontsize=6)

        # Hide unused axes
        for idx in range(n, nrows * ncols):
            r, c = divmod(idx, ncols)
            axes[r, c].set_visible(False)

        fig.suptitle(f"Permutation null — {col}", fontsize=11)
        fig.tight_layout()
        figs.append(fig)
    return figs


# ── Candidate-centric plots ────────────────────────────────────────────────

CANDIDATE_COLOR = "#2ca02c"
OTHER_COLOR = "#bdbdbd"


def plot_candidate_loadings(pca_loadings, candidates, pcs=("PC2", "PC3"), top_n=10, figsize=(12, 5)):
    """Top loading bar chart per PC, with candidate features highlighted."""
    n_pcs = len(pcs)
    fig, axes = plt.subplots(1, n_pcs, figsize=figsize, sharey=False)
    if n_pcs == 1:
        axes = [axes]

    for ax, pc in zip(axes, pcs):
        vals = pca_loadings[pc].abs().nlargest(top_n).sort_values()
        colors = [CANDIDATE_COLOR if f in candidates else OTHER_COLOR for f in vals.index]
        bars = ax.barh(vals.index, vals.values, color=colors, edgecolor="black", lw=0.4)
        for bar, feat in zip(bars, vals.index):
            if feat in candidates:
                ax.text(
                    bar.get_width() + 0.003, bar.get_y() + bar.get_height() / 2,
                    "★", va="center", fontsize=9, color=CANDIDATE_COLOR,
                )
        ax.set_xlabel(f"|loading| on {pc}")
        ax.set_title(pc)

    handles = [
        mpatches.Patch(color=CANDIDATE_COLOR, label="Candidate"),
        mpatches.Patch(color=OTHER_COLOR, label="Other"),
    ]
    axes[-1].legend(handles=handles, fontsize=7, loc="lower right")
    fig.suptitle("PCA loadings — candidate biomarkers highlighted", fontsize=12)
    fig.tight_layout()
    return fig


def plot_candidate_summary(
    candidates, stai_spearman, h5_effects, h5_pvals,
    d_threshold=0.3, eta2_threshold=0.04,
    binary_cols=("Gender", "WearsGlasses", "Handedness"),
    multi_cols=("BloodType",),
    figsize=(13, 5),
):
    """1×2 summary figure for candidate biomarkers.

    Left : grouped horizontal bars — Spearman ρ with STAI_S and STAI_T per candidate.
    Right: dot chart of H5 effect sizes per candidate × metadata column,
           with threshold lines (Cohen, 1988).
    """
    feats = sorted(candidates)

    fig, (ax_rho, ax_h5) = plt.subplots(1, 2, figsize=figsize,
                                         gridspec_kw={"width_ratios": [1, 1.2]})

    # ── Left: STAI correlations ──────────────────────────────────────────
    y = np.arange(len(feats))
    bar_h = 0.35
    rho_s = [stai_spearman.loc[f, "STAI_S"] for f in feats]
    rho_t = [stai_spearman.loc[f, "STAI_T"] for f in feats]

    ax_rho.barh(y - bar_h / 2, rho_s, height=bar_h, color="#2166ac",
                edgecolor="black", lw=0.4, label="STAI_S")
    ax_rho.barh(y + bar_h / 2, rho_t, height=bar_h, color="#b2182b",
                edgecolor="black", lw=0.4, label="STAI_T")
    ax_rho.axvline(0, color="grey", lw=0.5)
    ax_rho.set_yticks(y)
    ax_rho.set_yticklabels(feats, fontsize=8)
    ax_rho.set_xlabel("Spearman ρ")
    ax_rho.set_title("STAI correlations", fontsize=10, fontweight="bold")
    ax_rho.legend(fontsize=7, loc="lower right")
    ax_rho.invert_yaxis()

    # ── Right: H5 effect sizes ───────────────────────────────────────────
    meta_cols = list(binary_cols) + list(multi_cols)
    markers = ["o", "s", "D", "^", "v", "P"]
    colors = plt.cm.Set2(np.linspace(0, 1, len(meta_cols)))

    for j, col in enumerate(meta_cols):
        vals = [h5_effects.loc[f, col] for f in feats]
        ax_h5.scatter(vals, y, marker=markers[j % len(markers)],
                      color=colors[j], s=50, zorder=3, label=col,
                      edgecolors="black", linewidths=0.4)

    # Threshold lines
    ax_h5.axvline(d_threshold, color="grey", ls="--", lw=1,
                  label=f"|d| = {d_threshold} threshold")
    ax_h5.axvline(eta2_threshold, color="grey", ls=":", lw=1,
                  label=f"η² = {eta2_threshold} threshold")

    ax_h5.set_yticks(y)
    ax_h5.set_yticklabels([])
    ax_h5.set_xlabel("Effect size (|d| or η²)")
    ax_h5.set_title("H5 metadata effect sizes", fontsize=10, fontweight="bold")
    ax_h5.legend(fontsize=7, loc="lower right")
    ax_h5.invert_yaxis()
    ax_h5.set_xlim(left=0)

    # Annotate "all permutation p = 0.001" as text
    fig.text(0.5, -0.02, "All candidates: permutation p ≤ 0.001 (1000 shuffles)",
             ha="center", fontsize=8, fontstyle="italic", color="grey")

    fig.suptitle("Candidate biomarker summary", fontsize=11)
    fig.tight_layout()
    return fig
