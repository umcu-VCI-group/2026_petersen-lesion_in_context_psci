"""Shared plotting helpers for stats/figs notebooks.

- GROUP_ORDER, PALETTE: canonical model order and colors used across all panels.
- compute_nb_sig_table: per-(target, focal, reference) Nadeau-Bengio one-sided
  test, with BH-FDR within each target.
- get_sig_flags / plot_sig_markers / sig_legend_entries: triangle/square/pentagon
  markers placed above each violin (or box) indicating the focal model is
  significantly better than Majority / Permuted / Baseline.
"""

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from analysis.lib.stats import nadeau_bengio_test, bh_fdr


GROUP_ORDER = ["Majority", "Permuted", "Baseline", "Lesion", "fLNM", "sLNM", "Stacking"]

PALETTE = {
    "Majority": "#D3D3D3",
    "Permuted": "#E8B4B8",
    "Baseline": "#AEC6CF",
    "Lesion":   "#E76F51",
    "fLNM":     "#52A55C",
    "sLNM":     "#779ECB",
    "Stacking": "#9B7EBD",
}

# Long-form labels for legends. Internal column values stay as the short
# keys (so DataFrames / CSVs are unchanged), but in plots we re-label
# the legend entries through this dict for readability.
DISPLAY_NAMES = {
    "Majority": "Majority",
    "Permuted": "Permuted",
    "Baseline": "Baseline (demographics, clinical information, infarct volume)",
    "Lesion":   "Infarct location",
    "fLNM":     "fLNM",
    "sLNM":     "sLNM",
    "Stacking": "Integrated: baseline + acute infarct topography and connectivity",
}

# Markers: shape -> reference model the focal model must beat to earn it.
SIG_MARKERS = [
    ("Majority", "^"),  # triangle
    ("Permuted", "s"),  # square
    ("Baseline", "p"),  # pentagon
]
SIG_ALPHA = 0.05

REFERENCE_GROUPS = {"Majority", "Permuted", "Baseline"}
FOCAL_GROUPS = ["Lesion", "fLNM", "sLNM", "Stacking"]

# Cognitive-target nice labels (matches legacy 06_external_validation).
VARIABLE_LABELS = {
    "cognition-PSCI_present":            "Any-domain\nPSCI",
    "cognition-aef_domain_score":        "Attention /\nExecutive function",
    "cognition-ps_domain_score":         "Processing speed",
    "cognition-language_domain_score":   "Language",
    "cognition-verbalmemory_domain_score":  "Verbal memory",
    "cognition-vsfunctions_domain_score":   "Visuospatial\nfunctions",
    "cognition-vsmemory_domain_score":      "Visuospatial\nmemory",
}

# Short single-line variants (for tight panels / horizontal bars).
VARIABLE_LABELS_SHORT = {
    "cognition-PSCI_present":            "Any-domain PSCI",
    "cognition-aef_domain_score":        "Attn/Exec.",
    "cognition-ps_domain_score":         "Proc. speed",
    "cognition-language_domain_score":   "Language",
    "cognition-verbalmemory_domain_score":  "Verbal memory",
    "cognition-vsfunctions_domain_score":   "Visuosp. fn.",
    "cognition-vsmemory_domain_score":      "Visuosp. mem.",
}

# Modality display labels.
MODALITY_LABELS = {
    "lesion":       "Infarct location",
    "fLNM":         "Functional LNM",
    "sLNM":         "Structural LNM",
    "lesion_acute": "Infarct location",
    "flnm_acute":   "Functional LNM",
    "slnm_acute":   "Structural LNM",
    "baseline":     "Baseline (demographics, clinical information, infarct volume)",
}

# Cohort display names. Key = internal cohort label used in data; value = paper-facing name.
COHORT_LABELS = {
    "SB":         "Bundang VCI",
    "CAS":        "CASPER",
    "COAST":      "COAST",
    "CODECS":     "CODECS",
    "CROMIS":     "CROMIS-2",
    "DEDEMAS":    "DEDEMAS",
    "ETZ":        "PROCRAS",
    "GRECOGVASC": "GRECogVASC",
    "H":          "Hallym VCI",
    "L":          "STROKDEM",
    "MSSB":       "MSS-2",
    "STRIDE":     "CU-STRIDE",
    "USCOG":      "USCOG",
}

# Manuscript-wide matplotlib rcParams.
# Apply via `apply_manuscript_style()` at the top of any figure-building
# notebook/script so panels share font sizes / weights / spines.
# Sized so individual panels remain readable when composed into multi-panel
# manuscript figures.
MANUSCRIPT_RCPARAMS = {
    "font.size":         12,
    "font.weight":       "bold",
    "axes.titlesize":    14,
    "axes.titleweight":  "bold",
    "axes.labelsize":    13,
    "axes.labelweight":  "bold",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "xtick.labelsize":   11,
    "ytick.labelsize":   11,
    "legend.fontsize":   11,
    "legend.title_fontsize": 11,
    "legend.frameon":    False,
    "figure.titlesize":  15,
    "figure.titleweight":"bold",
}

def apply_manuscript_style():
    """Apply the manuscript-wide matplotlib rcParams."""
    import matplotlib as mpl
    mpl.rcParams.update(MANUSCRIPT_RCPARAMS)


def compute_nb_sig_table(plot_long, n_splits=10, test_size=0.1,
                          focal_groups=None, reference_groups=None,
                          metric="AUPRC"):
    """Run one-sided Nadeau-Bengio tests for all pairs.

    plot_long must have columns: Target, Group, Fold, <metric>. Pairs are
    aligned by Fold within each Target. BH-FDR is applied within each target
    (across all focal x reference comparisons for that target).

    focal_groups, reference_groups: optional lists overriding the defaults
    (FOCAL_GROUPS for focal, [Majority, Permuted, Baseline] for reference).
    Set focal_groups=["Stacking"] to skip the per-modality tests entirely.

    metric: column name of the per-fold scoring metric (default "AUPRC"). The
    AUROC sibling notebook (07b) passes "AUROC".

    Returns DataFrame with: Target, Focal, Reference, mean_diff, t_stat, p, p_adj_BH.
    """
    if focal_groups is None:
        focal_groups = FOCAL_GROUPS
    if reference_groups is None:
        reference_groups = ["Majority", "Permuted", "Baseline"]
    rows = []
    for target, sub in plot_long.groupby("Target"):
        target_pvals = []
        target_idx = []
        for focal in focal_groups:
            f_series = sub.loc[sub["Group"] == focal].set_index("Fold")[metric]
            if len(f_series) < 2:
                continue
            for reference in reference_groups:
                r_series = sub.loc[sub["Group"] == reference].set_index("Fold")[metric]
                common = f_series.index.intersection(r_series.index)
                if len(common) < 2:
                    continue
                out = nadeau_bengio_test(
                    f_series.loc[common].values,
                    r_series.loc[common].values,
                    n_splits=n_splits,
                    test_size=test_size,
                    alternative="greater",
                )
                rows.append({
                    "Target": target, "Focal": focal, "Reference": reference,
                    "mean_diff": out["mean_diff"], "t_stat": out["t_stat"],
                    "p": out["p"], "n_paired": out["n_paired"],
                })
                target_pvals.append(out["p"])
                target_idx.append(len(rows) - 1)
        if target_pvals:
            adj = bh_fdr(np.asarray(target_pvals, dtype=float))
            for i, row_idx in enumerate(target_idx):
                rows[row_idx]["p_adj_BH"] = float(adj[i])
    return pd.DataFrame(rows)


def get_sig_flags(nb_df, target, focal):
    """Return {reference: bool} indicating significance after BH-FDR + positive direction."""
    out = {ref: False for ref, _ in SIG_MARKERS}
    sub = nb_df[(nb_df["Target"] == target) & (nb_df["Focal"] == focal)]
    for ref, _ in SIG_MARKERS:
        row = sub[sub["Reference"] == ref]
        if len(row) == 0:
            continue
        p = row["p_adj_BH"].iloc[0] if "p_adj_BH" in row.columns else row["p"].iloc[0]
        d = row["mean_diff"].iloc[0]
        if pd.notna(p) and p < SIG_ALPHA and pd.notna(d) and d > 0:
            out[ref] = True
    return out


def plot_sig_markers(ax, cx, sig_flags, base_y, row_step,
                     marker_size=80, color="#222"):
    """Stack significance markers vertically above position cx."""
    for i, (ref, mk) in enumerate(SIG_MARKERS):
        if sig_flags[ref]:
            ax.scatter(cx, base_y + i * row_step, marker=mk, s=marker_size,
                       color=color, edgecolor="black", linewidth=0.8,
                       zorder=6, clip_on=False)


def sig_legend_entries(test_name="NB"):
    """Legend handles for the three significance markers.

    test_name labels the inference test in the legend ("NB" for pooled CV,
    "Wilcoxon" for LOCO).
    """
    # \mathbf so the p_FDR mathtext gets bold rendering even when the
    # surrounding legend text inherits a bold prop (matplotlib's mathtext
    # uses a separate font path that ignores the prop weight).
    return [
        Line2D([0], [0], marker=mk, color="w", markerfacecolor="black",
               markeredgecolor="black", markersize=9, linewidth=0,
               label=f"> {ref} ($\\mathbf{{p_{{FDR}}<{SIG_ALPHA}}}$, {test_name})")
        for ref, mk in SIG_MARKERS
    ]


def compute_wilcoxon_sig_table(plot_long, focal_groups=None,
                                 reference_groups=None, metric="AUPRC"):
    """Per-cohort paired Wilcoxon signed-rank significance table.

    LOCO-appropriate alternative to `compute_nb_sig_table`. Test sets are
    disjoint across LOCO folds (each cohort held out exactly once), so the
    Nadeau-Bengio variance correction (designed for k-fold CV with
    overlapping training sets *and* overlapping test material) is not
    applicable. Wilcoxon signed-rank is the field-standard non-parametric
    paired test for this setting.

    Mirrors `compute_nb_sig_table`'s output schema: one row per
    (Target, Focal, Reference) with mean_diff, t_stat (=NaN here),
    p, p_adj_BH, n_paired.
    """
    from scipy.stats import wilcoxon
    if focal_groups is None:
        focal_groups = FOCAL_GROUPS
    if reference_groups is None:
        reference_groups = ["Majority", "Permuted", "Baseline"]
    rows = []
    for target, sub in plot_long.groupby("Target"):
        target_pvals = []
        target_idx = []
        for focal in focal_groups:
            f_series = sub.loc[sub["Group"] == focal].set_index("Fold")[metric]
            if len(f_series) < 2:
                continue
            for reference in reference_groups:
                r_series = sub.loc[sub["Group"] == reference].set_index("Fold")[metric]
                common = f_series.index.intersection(r_series.index)
                if len(common) < 2:
                    continue
                diffs = (f_series.loc[common].values
                          - r_series.loc[common].values)
                diffs = diffs[~np.isnan(diffs)]
                n = len(diffs)
                if n < 2:
                    continue
                mean_diff = float(np.mean(diffs))
                # Drop exact zeros (Wilcoxon ties) by mode='wilcox'; alternative
                # 'greater' tests if positive differences dominate.
                try:
                    if np.all(diffs == 0):
                        p = 1.0
                    else:
                        _, p = wilcoxon(diffs, alternative="greater",
                                          zero_method="wilcox")
                        p = float(p)
                except ValueError:
                    p = 1.0  # fully degenerate (e.g. all NaN-equivalent)
                rows.append({
                    "Target": target, "Focal": focal, "Reference": reference,
                    "mean_diff": mean_diff, "t_stat": np.nan,
                    "p": p, "n_paired": n,
                })
                target_pvals.append(p)
                target_idx.append(len(rows) - 1)
        if target_pvals:
            adj = bh_fdr(np.asarray(target_pvals, dtype=float))
            for i, row_idx in enumerate(target_idx):
                rows[row_idx]["p_adj_BH"] = float(adj[i])
    return pd.DataFrame(rows)
