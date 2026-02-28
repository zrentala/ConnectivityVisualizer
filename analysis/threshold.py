"""
thresholding.py — cleaned and refactored

- Threshold class holds settings + dispatches stat tests
- All computational/stat functions are module-level
- No two-group tests
"""

from dataclasses import dataclass
from typing import Union, Tuple
import numpy as np

from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.stats import (
    ttest_1samp,
    norm,
    zscore,
    wilcoxon,
)

def mst_mask(conn_norm: np.ndarray) -> np.ndarray:
    """Maximum spanning tree mask from normalized connectivity matrix."""
    mst = minimum_spanning_tree(-conn_norm)
    return (mst.toarray() != 0).astype(bool)


# ============================================================
# STATISTICAL TEST COMPUTATIONS (module-level)
# ============================================================

def one_sample_permutation_test(
    data2d: np.ndarray,
    popmean: float = 0.0,
    n_permutations: int = 2000,
    bonferroni: bool = True,
):
    """
    One-sample permutation (sign-flip) for array shape (samples × features).
    Returns p-values.
    """
    arr = np.asarray(data2d)
    n_samples, n_features = arr.shape

    observed = np.mean(arr - popmean, axis=0)
    pvals = np.zeros(n_features)

    for i in range(n_features):
        col = arr[:, i]
        count = 0
        for _ in range(n_permutations):
            signs = np.random.choice([-1, 1], size=n_samples)
            pm = np.mean((col - popmean) * signs)
            if abs(pm) >= abs(observed[i]):
                count += 1
        pvals[i] = (count + 1) / (n_permutations + 1)

    if bonferroni:
        pvals = np.minimum(pvals * n_features, 1.0)

    return pvals


def compute_stattest(
    conn_stack: np.ndarray,
    stat_test: str,
    alpha: float,
    n_permutations: int = 2000,
):
    """
    Compute:
      - p-value map   (n x n)
      - boolean mask  (n x n) where p < alpha

    This function contains *all* statistical test logic and
    is intentionally kept OUTSIDE any class.
    """
    arr = np.asarray(conn_stack)
    if arr.ndim != 3:
        raise ValueError("Statistical tests require 3D array (samples × n × n).")

    n_samples, n_elec, _ = arr.shape
    flat = arr.reshape(n_samples, n_elec * n_elec)

    method = stat_test.lower()

    # --- t test ---
    if method == "t":
        _, pvals = ttest_1samp(flat, popmean=0.0, axis=0)

    # --- z (normal) test ---
    elif method == "z":
        zs = zscore(flat, axis=0)
        pvals = 2 * (1 - norm.cdf(np.abs(zs)))

    # --- Wilcoxon ---
    elif method == "wilcoxon":
        pvals = np.array([
            (
                wilcoxon(flat[:, i] - 0).pvalue
                if np.any(flat[:, i] != 0)
                else 1.0
            )
            for i in range(flat.size // n_samples)
        ])

    # --- permutation test ---
    elif method == "permutation":
        pvals = one_sample_permutation_test(
            flat,
            popmean=0.0,
            n_permutations=n_permutations,
            bonferroni=True,
        )

    else:
        raise ValueError(f"Unknown stat_test '{stat_test}'.")

    pmap = pvals.reshape(n_elec, n_elec)
    mask = pmap < alpha

    return pmap, mask

def compute_load_thresh(n_elec, n_min=1, n_max=80, t_min=0.5):
    n_clamped = max(n_min, min(n_elec, n_max))
    scale = (n_clamped - n_min) / (n_max - n_min)
    return t_min + scale * (0.99 - t_min)

# ============================================================
# THRESHOLD CLASS (dispatcher only)
# ============================================================

@dataclass
class Threshold:
    """
    Stores threshold settings and dispatches thresholds & stat tests.
    All computations remain in module-level functions.

    Parameters
    ----------
    threshold        : percent (0–100) used in Basic mode
    threshold_type   : "None", "Basic", "MST", "Statistical Test"
    alpha            : percent (0–100) for significance
    stat_test        : "t", "z", "wilcoxon", "permutation"
    """
    threshold: float = 0.5
    threshold_type: str = "Basic"
    alpha: float = 5.0
    stat_test: str = "t"

    # -------------------------
    # MAIN ENTRY POINT
    # -------------------------
    def apply_threshold(
        self,
        conn_mat: np.ndarray,
        idx: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Applies threshold to the matrix `conn_mat[idx]`.
        Returns (C_thresholded, mask).
        """
        C = conn_mat[idx]

        # No thresholding
        if self.threshold_type in {None, "", "None"}:
            mask = np.ones_like(C, dtype=bool)
            return C, mask

        cmin = float(np.min(C))
        cmax = float(np.max(C))
        conn_norm = (C - cmin) / ((cmax - cmin) + 1e-12)

        ttype = self.threshold_type.lower()

        # -------------------------
        # BASIC THRESHOLD
        # -------------------------
        if ttype == "basic":
            percent = self.threshold  # 0–100

            flat = np.abs(C.flatten())
            cutoff = np.percentile(flat, percent)

            mask = np.abs(C) >= cutoff
            return C * mask, mask

        # -------------------------
        # MST THRESHOLD
        # -------------------------
        if ttype in {"mst", "minimum spanning tree"}:
            mask = mst_mask(conn_norm)
            return C * mask, mask

        # -------------------------
        # STATISTICAL THRESHOLD
        # -------------------------
        if ttype == "statistical test":
            alpha_float = self.alpha / 100.0
            pvals, mask = compute_stattest(
                conn_mat,
                stat_test=self.stat_test,
                alpha=alpha_float,
                n_permutations=2000,
            )
            return C * mask, mask

        raise ValueError(f"Unknown threshold_type: {self.threshold_type}")