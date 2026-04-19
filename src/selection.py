"""
Feature selection — redundancy removal, standardization, relevance & confound analysis.
"""

import numpy as np
import pandas as pd
from scipy import stats

from .constants import HIGH_CORRELATION_THRESHOLD, DEFAULT_TOP_N_FEATURES

# Numerical epsilon — features with std below this are effectively constant
NEAR_CONSTANT_STD_THRESHOLD = 1e-6


# ── Feature redundancy ──────────────────────────────────────────────────────

def remove_near_constant(features, threshold=NEAR_CONSTANT_STD_THRESHOLD):
    """Drop features whose std is below *threshold*.

    Returns (filtered DataFrame, list of dropped column names).
    """
    stds = features.std()
    drop = stds[stds < threshold].index.tolist()
    return features.drop(columns=drop), drop


def remove_redundant_correlated(features, threshold=HIGH_CORRELATION_THRESHOLD):
    """For pairs with |r| > threshold, drop the feature with higher IQR across subjects.

    The IQR (Q75 − Q25) of each feature column over the subject axis measures
    cross-subject spread; the less stable member of a correlated pair is removed.

    Returns (filtered DataFrame, list of dropped column names).
    """
    corr = features.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))

    # IQR of each feature across subjects
    iqr_values = features.quantile(0.75) - features.quantile(0.25)

    drop = set()
    for col in upper.columns:
        paired = upper.index[upper[col] > threshold].tolist()
        for partner in paired:
            if col in drop or partner in drop:
                continue
            drop.add(col if iqr_values[col] >= iqr_values[partner] else partner)
    drop = sorted(drop)
    return features.drop(columns=drop), drop


# ── Standardization ─────────────────────────────────────────────────────────

def zscore_standardize(features):
    """Z-score standardize each column.

    Returns (standardized DataFrame, params DataFrame with 'mean' and 'std' rows).
    """
    mu = features.mean()
    sigma = features.std()
    # Avoid division by zero for any remaining near-constant columns
    sigma = sigma.replace(0, np.nan)
    standardized = (features - mu) / sigma
    params = pd.DataFrame({"mean": mu, "std": sigma})
    return standardized, params


# ── Correlation helpers ─────────────────────────────────────────────────────

def feature_target_correlation(features, targets, method="spearman"):
    """Compute correlation between each feature column and each target column.

    Parameters
    ----------
    features : DataFrame  (subjects × features)
    targets : DataFrame   (subjects × target columns, e.g. STAI_S, STAI_T)
    method : str          'pearson' or 'spearman'

    Returns DataFrame (features × targets) of correlation coefficients.
    """
    results = {}
    for col in targets.columns:
        if method == "spearman":
            results[col] = features.apply(
                lambda f, t=targets[col]: stats.spearmanr(f, t, nan_policy="omit").statistic
            )
        else:
            results[col] = features.corrwith(targets[col])
    return pd.DataFrame(results)


def feature_confound_correlation(features, confounds, method="spearman"):
    """Correlation between features and continuous confounders (e.g. CalibrationError, quality).

    Same interface as feature_target_correlation.
    """
    return feature_target_correlation(features, confounds, method=method)


def dual_correlation(features, targets, top_n=DEFAULT_TOP_N_FEATURES):
    """Run Spearman (primary) and Pearson (sanity check), flag stable features.

    A feature is *stable* if it appears in the top-*n* by |r| for both methods
    for at least one target.

    Returns
    -------
    spearman_corr : DataFrame  (features × targets)
    pearson_corr  : DataFrame  (features × targets)
    stable        : set of feature names that rank in top-*n* under both methods
    """
    spearman_corr = feature_target_correlation(features, targets, method="spearman")
    pearson_corr = feature_target_correlation(features, targets, method="pearson")

    stable = set()
    for col in targets.columns:
        top_sp = set(spearman_corr[col].abs().nlargest(top_n).index)
        top_pe = set(pearson_corr[col].abs().nlargest(top_n).index)
        stable |= top_sp & top_pe

    return spearman_corr, pearson_corr, stable


# ── Permutation test ────────────────────────────────────────────────────────

DEFAULT_N_PERMUTATIONS = 1000


def permutation_test(features, targets, feature_subset, n_permutations=DEFAULT_N_PERMUTATIONS, seed=42):
    """Permutation test for Spearman ρ between features and targets.

    Shuffles target labels *n_permutations* times, recomputes ρ for each
    feature–target pair, and returns the observed ρ and empirical p-value.

    Returns
    -------
    observed : DataFrame  (features × targets) — observed Spearman ρ
    pvalues : DataFrame   (features × targets) — empirical p-values
    null_distributions : dict  {target_col: ndarray (n_permutations × n_features)}
    """
    rng = np.random.default_rng(seed)
    feats = features[feature_subset]
    observed = feature_target_correlation(feats, targets, method="spearman")

    null_distributions = {}
    for col in targets.columns:
        null = np.empty((n_permutations, len(feature_subset)))
        target_vals = targets[col].values.copy()
        for i in range(n_permutations):
            rng.shuffle(target_vals)
            shuffled = pd.Series(target_vals, index=targets.index, name=col)
            null[i, :] = feats.apply(
                lambda f, t=shuffled: stats.spearmanr(f, t, nan_policy="omit").statistic
            ).values
        null_distributions[col] = null

    pvalues = pd.DataFrame(index=feature_subset, columns=targets.columns, dtype=float)
    for col in targets.columns:
        obs_abs = observed[col].abs().values
        null_abs = np.abs(null_distributions[col])
        pvalues[col] = ((null_abs >= obs_abs).sum(axis=0) + 1) / (n_permutations + 1)

    return observed, pvalues, null_distributions


# ── PCA helpers ─────────────────────────────────────────────────────────────

def fit_pca(features_z, n_components=None):
    """Fit PCA on z-scored features, imputing NaNs with column median.

    Parameters
    ----------
    features_z : DataFrame  (subjects × features), z-scored
    n_components : int or None
        Number of components to keep.  None keeps min(n_samples, n_features).

    Returns
    -------
    scores : DataFrame  (subjects × PCs) with columns PC1, PC2, …
    explained_variance_ratio : ndarray  per-component variance explained
    loadings : DataFrame  (features × PCs) — PCA component loadings
    """
    from sklearn.decomposition import PCA

    X = features_z.copy()
    # Impute remaining NaNs with column median so PCA doesn't fail
    X = X.fillna(X.median())

    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(X.values)

    pc_cols = [f"PC{i+1}" for i in range(transformed.shape[1])]
    scores = pd.DataFrame(transformed, index=features_z.index, columns=pc_cols)
    loadings = pd.DataFrame(
        pca.components_.T,
        index=features_z.columns,
        columns=pc_cols,
    )
    return scores, pca.explained_variance_ratio_, loadings


def top_loadings(loadings, pc="PC1", top_n=DEFAULT_TOP_N_FEATURES):
    """Return the top-*n* features by absolute loading for a given PC."""
    return loadings[pc].abs().nlargest(top_n).sort_values(ascending=False)


# ── Metadata confounder tests ───────────────────────────────────────────────

def metadata_group_tests(features_z, metadata):
    """Run group-difference tests for each feature against each metadata column.

    Binary columns → Mann-Whitney U + Cohen's d;
    columns with >2 groups → Kruskal-Wallis + eta squared (η²).

    Parameters
    ----------
    features_z : DataFrame  (subjects × features) — z-scored feature values
    metadata : DataFrame    (subjects × metadata columns), aligned index

    Returns
    -------
    pvals : DataFrame  (features × metadata columns) with p-values
    effects : DataFrame  (features × metadata columns) with effect sizes
              (Cohen's d for binary, η² for multi-group)
    """
    p_rows, e_rows = [], []
    for feat in features_z.columns:
        vals = features_z[feat].dropna()
        p_row = {"feature": feat}
        e_row = {"feature": feat}
        for col in metadata.columns:
            groups = metadata.loc[vals.index, col].dropna()
            shared = vals.loc[groups.index]
            unique = groups.unique()
            if len(unique) == 2:
                g1 = shared[groups == unique[0]]
                g2 = shared[groups == unique[1]]
                _, p = stats.mannwhitneyu(g1, g2, alternative="two-sided")
                # Cohen's d (pooled std)
                n1, n2 = len(g1), len(g2)
                pooled_std = np.sqrt(
                    ((n1 - 1) * g1.std(ddof=1) ** 2 + (n2 - 1) * g2.std(ddof=1) ** 2)
                    / (n1 + n2 - 2)
                )
                d = (g1.mean() - g2.mean()) / pooled_std if pooled_std > 0 else 0.0
                e_row[col] = abs(d)
            elif len(unique) > 2:
                group_vals = [shared[groups == g] for g in unique if len(shared[groups == g]) > 1]
                if len(group_vals) >= 2:
                    H, p = stats.kruskal(*group_vals)
                    # Eta squared: η² = (H - k + 1) / (N - k)
                    N = sum(len(g) for g in group_vals)
                    k = len(group_vals)
                    eta2 = (H - k + 1) / (N - k) if N > k else 0.0
                    e_row[col] = max(eta2, 0.0)
                else:
                    p = float("nan")
                    e_row[col] = float("nan")
            else:
                p = float("nan")
                e_row[col] = float("nan")
            p_row[col] = p
        p_rows.append(p_row)
        e_rows.append(e_row)
    pvals = pd.DataFrame(p_rows).set_index("feature")
    effects = pd.DataFrame(e_rows).set_index("feature")
    return pvals, effects


# ── Table builders ──────────────────────────────────────────────────────────

def build_relevance_table(stai_spearman, stai_stable, confound_corr, top_n=10):
    """Build the STAI relevance summary table.

    Returns (table, top_union, floor, candidates).
    """
    top_s = set(stai_spearman["STAI_S"].abs().nlargest(top_n).index)
    top_t = set(stai_spearman["STAI_T"].abs().nlargest(top_n).index)
    top_union = sorted(top_s | top_t)
    floor = min(
        stai_spearman["STAI_S"].abs().nlargest(top_n).min(),
        stai_spearman["STAI_T"].abs().nlargest(top_n).min(),
    )
    candidates = {
        f for f in stai_stable
        if confound_corr.loc[f].abs().max() <= floor
    }
    rows = []
    for feat in top_union:
        rho_s = stai_spearman.loc[feat, "STAI_S"]
        rho_t = stai_spearman.loc[feat, "STAI_T"]
        rel_s = "✓" if feat in stai_stable and feat in top_s else ""
        rel_t = "✓" if feat in stai_stable and feat in top_t else ""
        max_conf = confound_corr.loc[feat].abs().max()
        conf = "⚠" if max_conf > floor else ""
        rows.append({
            "feature": feat,
            "ρ STAI_S": f"{rho_s:+.4f}", "stable_S": rel_s,
            "ρ STAI_T": f"{rho_t:+.4f}", "stable_T": rel_t,
            "quality-driven": conf,
        })
    tbl = pd.DataFrame(rows).set_index("feature")
    return tbl, top_union, floor, candidates


def build_candidate_table(
    pca_loadings, stai_spearman, candidates, stai_stable,
    confound_corr, floor, pcs=("PC2", "PC3"), top_n=10,
):
    """Build candidate-centric summary table with PC loadings and flags.

    Section A ("Candidate"): the selected biomarker candidates.
    Section B ("Other PC loader"): top-loading features per PC that are not candidates,
    showing why they were excluded (unstable or quality-driven).

    Returns (table, feature_list_for_h5).
    """
    # Section A — candidates
    rows = []
    for feat in sorted(candidates):
        row = {
            "section": "Candidate",
            "feature": feat,
            "ρ STAI_S": f"{stai_spearman.loc[feat, 'STAI_S']:+.4f}",
            "ρ STAI_T": f"{stai_spearman.loc[feat, 'STAI_T']:+.4f}",
            "stable": True,
            "quality-driven": False,
        }
        for pc in pcs:
            row[f"{pc} loading"] = f"{pca_loadings.loc[feat, pc]:+.3f}"
        rows.append(row)

    # Section B — non-candidate top PC loaders
    other_feats = set()
    for pc in pcs:
        other_feats |= set(top_loadings(pca_loadings, pc=pc, top_n=top_n).index)
    other_feats -= set(candidates)

    for feat in sorted(other_feats):
        row = {
            "section": "Other PC loader",
            "feature": feat,
            "ρ STAI_S": f"{stai_spearman.loc[feat, 'STAI_S']:+.4f}",
            "ρ STAI_T": f"{stai_spearman.loc[feat, 'STAI_T']:+.4f}",
            "stable": feat in stai_stable,
            "quality-driven": bool(confound_corr.loc[feat].abs().max() > floor),
        }
        for pc in pcs:
            row[f"{pc} loading"] = f"{pca_loadings.loc[feat, pc]:+.3f}"
        rows.append(row)

    tbl = pd.DataFrame(rows).set_index(["section", "feature"])
    return tbl, sorted(candidates)


# Effect-size thresholds for H5 confounder flagging (Cohen, 1988)
H5_COHENS_D_THRESHOLD = 0.3      # small-to-medium
H5_ETA_SQUARED_THRESHOLD = 0.04  # small-to-medium


def format_h5_display(h5_pvals, h5_effects, binary_cols, multi_cols):
    """Format H5 p-values + effect sizes into display strings.

    Returns a DataFrame with "p (d=X.XX)" or "p (η²=X.XXX)" per cell.
    ⚠ flag requires both p < 0.05 AND meaningful effect size
    (|d| ≥ 0.3 for binary, η² ≥ 0.04 for multi-group).
    """
    h5_display = pd.DataFrame(index=h5_pvals.index)
    for col in binary_cols:
        h5_display[col] = (
            h5_pvals[col].apply(lambda p: f"{p:.3f}")
            + h5_effects[col].apply(lambda d: f" (d={d:.2f})")
            + pd.Series(
                [
                    " ⚠" if p < 0.05 and d >= H5_COHENS_D_THRESHOLD else ""
                    for p, d in zip(h5_pvals[col], h5_effects[col])
                ],
                index=h5_pvals.index,
            )
        )
    for col in multi_cols:
        h5_display[col] = (
            h5_pvals[col].apply(lambda p: f"{p:.3f}")
            + h5_effects[col].apply(lambda e: f" (η²={e:.3f})")
            + pd.Series(
                [
                    " ⚠" if p < 0.05 and e >= H5_ETA_SQUARED_THRESHOLD else ""
                    for p, e in zip(h5_pvals[col], h5_effects[col])
                ],
                index=h5_pvals.index,
            )
        )
    return h5_display
