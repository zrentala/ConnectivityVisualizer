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


# ============================================================
# BASIC MASKS (module-level)
# ============================================================

def basic_mask(conn_norm: np.ndarray, thr: float) -> np.ndarray:
    """Mask for normalized connectivity >= threshold."""
    return (conn_norm >= thr).astype(bool)


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
            thr = self.threshold / 100.0
            mask = basic_mask(conn_norm, thr)
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


# from dataclasses import dataclass
# from typing import Optional, Union, Tuple

# import numpy as np
# from scipy.sparse.csgraph import minimum_spanning_tree
# from scipy.stats import ttest_1samp, norm, zscore, wilcoxon
# from statsmodels.stats.multitest import multipletests  # still available if you want BH, etc.
# from utils.braindata import BrainData

# @dataclass
# class Threshold:
#     """
#     Threshold settings used to derive a boolean mask from a connectivity matrix.

#     Attributes
#     ----------
#     threshold : float
#         For 'Basic', interpreted as a percentage (0-100) of normalized connectivity.
#     threshold_type : Optional[str]
#         One of:
#             - None / "None"             → no thresholding (all True mask)
#             - "Basic"                   → simple value threshold on normalized 0-1 matrix
#             - "Statistical Test"        → p < alpha using get_stattest_map
#             - "MST" / "Minimum Spanning Tree"
#     alpha : float
#         For 'Statistical Test', interpreted as a percentage (0-100).
#     """
#     threshold: float = 0.5
#     threshold_type: str = "Basic"
#     alpha: float = 5.0   # percent (5% default)

#     # ------------------------------------------------------------------
#     # Public: main entry point
#     # ------------------------------------------------------------------

#     def apply_threshold(self, conn_mat: np.ndarray, idx: int) -> Tuple[np.ndarray, np.ndarray]:
#         """
#         Given a connectivity matrix C (2D or 3D slice flattened to 2D),
#         return a boolean mask with the same shape as C.

#         For visualization we usually pass a single (n_elec, n_elec) matrix.
#         """
#         C = conn_mat[idx]
#         if C.ndim != 2:
#             raise ValueError("Threshold.apply_threshold expects a 2D matrix (n_elec, n_elec).")

#         # No thresholding: identity mask
#         if self.threshold_type is None or str(self.threshold_type).strip() in {"", "None"}:
#             return np.ones_like(C, dtype=bool)

#         # Normalize to [0,1] for percentage-based thresholds
#         c_min = float(np.min(C))
#         c_max = float(np.max(C))
#         denom = (c_max - c_min) + 1e-12
#         conn_normalized = (C - c_min) / denom

#         ttype = str(self.threshold_type)

#         if ttype == "Basic":
#             # 'threshold' is given in percent [0,100]
#             norm_threshold = float(self.threshold) / 100.0
#             mask = self.get_basic_map(conn_normalized, norm_threshold)
#             C = C * mask

#         elif ttype == "Statistical Test":
#             # 'alpha' also given in percent
#             Cs = conn_mat
#             if Cs.ndim != 3:
#                 raise ValueError("Statistical tests require multiple samples (3D array: samples×n×n).")
#             norm_alpha = float(self.alpha) / 100.0
#             mask = self.get_stattest_mask(Cs, alpha=norm_alpha)
#             C = C*mask

#         elif ttype in {"MST", "Minimum Spanning Tree"}:
#             mask = self.get_mst_map(conn_normalized)
#             C = C*mask
#         else:
#             raise ValueError(f"Unknown threshold_type: {self.threshold_type!r}")
#         # print(f"C: {C}")
#         # print(f"mask: {mask}")
#         return C, mask
#     # ------------------------------------------------------------------
#     # Basic / MST masks
#     # ------------------------------------------------------------------

#     @staticmethod
#     def get_basic_map(conn_mat: np.ndarray, threshold: float) -> np.ndarray:
#         """
#         Simple absolute threshold on a normalized matrix in [0,1].
#         Returns a boolean mask.
#         """
#         conn_mat = np.asarray(conn_mat)
#         return (conn_mat >= threshold).astype(bool)

#     @staticmethod
#     def get_mst_map(conn_mat: np.ndarray) -> np.ndarray:
#         """
#         Build maximum spanning tree mask from a (n_elec, n_elec) connectivity matrix.
#         We negate the weights and compute a minimum spanning tree of -conn_mat.
#         """
#         conn_mat = np.asarray(conn_mat)
#         if conn_mat.ndim != 2:
#             raise ValueError("get_mst_map expects a 2D matrix (n_elec, n_elec).")

#         # negate to get maximum spanning tree via minimum_spanning_tree
#         mst = minimum_spanning_tree(-conn_mat)
#         mst_map = mst.toarray() != 0
#         return mst_map.astype(bool)

#     # ------------------------------------------------------------------
#     # Multiple testing helpers
#     # ------------------------------------------------------------------

#     @staticmethod
#     def _bonferroni_correction(p_values: np.ndarray, alpha: float) -> np.ndarray:
#         """
#         Return Bonferroni-adjusted p-values (not a boolean mask).

#         p_adj = min(p * m, 1.0)
#         """
#         p = np.asarray(p_values, dtype=float)
#         m = p.size
#         if m == 0:
#             return p
#         p_adj = np.minimum(p * float(m), 1.0)
#         return p_adj

#     # ------------------------------------------------------------------
#     # One-sample permutation test
#     # ------------------------------------------------------------------

#     def _permutation_test(
#         self,
#         data,
#         popmean: float = 0.0,
#         axis: int = 0,
#         n_permutations: int = 1000,
#         correct_alpha: bool = True,
#         alpha: float = 0.05,
#     ):
#         """
#         One-sample permutation test (sign-flip) across rows for each feature (column).

#         Parameters
#         ----------
#         data : array-like, shape (n_samples, n_features)
#         popmean : float
#             Population mean under H0.
#         axis : int
#             Axis corresponding to samples (default 0).
#         n_permutations : int
#         correct_alpha : bool
#             If True, return Bonferroni-adjusted p-values.

#         Returns
#         -------
#         (None, p_values) with p_values shape (n_features,)
#         """
#         arr = np.asarray(data)
#         if axis != 0:
#             arr = np.swapaxes(arr, 0, axis)

#         n_samples, n_features = arr.shape
#         observed_means = np.mean(arr - popmean, axis=0)
#         p_values = np.zeros(n_features, dtype=float)

#         for i in range(n_features):
#             count = 0
#             col = arr[:, i]
#             for _ in range(int(n_permutations)):
#                 signs = np.random.choice([-1, 1], size=n_samples)
#                 permuted = (col - popmean) * signs
#                 permuted_mean = np.mean(permuted)
#                 if np.abs(permuted_mean) >= np.abs(observed_means[i]):
#                     count += 1
#             p_values[i] = (count + 1) / (n_permutations + 1)

#         if correct_alpha:
#             return None, self._bonferroni_correction(p_values, alpha=alpha)
#         return None, p_values

#     # ------------------------------------------------------------------
#     # Two-sample permutation test
#     # ------------------------------------------------------------------

#     def perm_test_between_groups(
#         self,
#         ctrl_stack: np.ndarray,
#         action_stack: np.ndarray,
#         n_permutations: int = 5000,
#     ) -> np.ndarray:
#         """
#         Two-sample permutation test comparing means between two groups for each feature.

#         ctrl_stack : (n_ctrl, n_features) or (n_ctrl, n_elec, n_elec)
#         action_stack : (n_action, n_features) or (n_action, n_elec, n_elec)
#         """
#         ctrl = np.asarray(ctrl_stack)
#         act = np.asarray(action_stack)
#         if ctrl.ndim == 1:
#             ctrl = ctrl[:, None]
#         if act.ndim == 1:
#             act = act[:, None]

#         n_ctrl, n_features = ctrl.shape
#         n_act = act.shape[0]
#         combined = np.vstack([ctrl, act])
#         n_total = combined.shape[0]

#         observed = np.mean(ctrl, axis=0) - np.mean(act, axis=0)
#         counts = np.zeros(n_features, dtype=int)

#         for _ in range(int(n_permutations)):
#             perm_idx = np.random.choice(n_total, size=n_ctrl, replace=False)
#             perm_ctrl = combined[perm_idx, :]
#             mask = np.ones(n_total, dtype=bool)
#             mask[perm_idx] = False
#             perm_act = combined[mask, :]
#             perm_diff = np.mean(perm_ctrl, axis=0) - np.mean(perm_act, axis=0)
#             counts += (np.abs(perm_diff) >= np.abs(observed))

#         p_values = (counts + 1) / (n_permutations + 1)
#         return p_values

#     # ------------------------------------------------------------------
#     # Public: statistical test → mask
#     # ------------------------------------------------------------------

#     def get_stattest_mask(
#         self,
#         conn_mat: np.ndarray,
#         alpha: float = 0.05,
#         test: str = "t",
#         n_permutations: int = 1000,
#     ) -> np.ndarray:
#         """
#         Return boolean mask (p < alpha) using get_stattest_map p-values.
#         """
#         pvals = self.get_stattest_map(
#             conn_mat,
#             alpha=alpha,
#             test=test,
#             n_permutations=n_permutations,
#         )
#         return (pvals < alpha).astype(bool)

#     def get_stattest_map(
#         self,
#         conn_mat: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]],
#         alpha: float = 0.05,
#         test: str = "t",
#         n_permutations: int = 1000,
#     ) -> np.ndarray:
#         """
#         Compute p-values on a 3D connectivity stack.

#         conn_mat :
#             - If 3D array: (n_samples, n_elec, n_elec)
#             - If tuple(list): (ctrl_stack, action_stack) for two-sample permutation tests

#         idx :
#             The index of the 2D matrix you are viewing visually.
#         """
#         # ------------------------------------------------------------------
#         # Case 1: Two-sample permutation test
#         # ------------------------------------------------------------------
#         if isinstance(conn_mat, (tuple, list)) and test.startswith("permutation"):
#             ctrl, act = conn_mat
#             ctrl = np.asarray(ctrl)
#             act = np.asarray(act)

#             # Flatten for testing
#             if ctrl.ndim == 3:
#                 n_ctrl, n_elecs, _ = ctrl.shape
#                 ctrl_flat = ctrl.reshape(n_ctrl, n_elecs * n_elecs)
#                 act_flat = act.reshape(act.shape[0], n_elecs * n_elecs)
#             else:
#                 ctrl_flat = ctrl
#                 act_flat = act
#                 n_elecs = int(np.sqrt(ctrl_flat.shape[1]))

#             pvals = self.perm_test_between_groups(
#                 ctrl_flat,
#                 act_flat,
#                 n_permutations=n_permutations,
#             )

#             return pvals.reshape(n_elecs, n_elecs)

#         # ------------------------------------------------------------------
#         # Case 2: Standard 1-sample tests on 3D stack
#         # ------------------------------------------------------------------
#         arr = np.asarray(conn_mat)

#         if arr.ndim != 3:
#             raise ValueError("Expected 3D array: (n_samples, n_elec, n_elec)")

#         n_samples, n_elecs, _ = arr.shape

#         # Flatten features → shape (n_samples, n_elecs*n_elecs)
#         flat = arr.reshape(n_samples, n_elecs * n_elecs)

#         # ------------------------------------------------------------------
#         # Perform statistical test
#         # ------------------------------------------------------------------
#         if test == "t":
#             _, pvals = ttest_1samp(flat, popmean=0, axis=0)

#         elif test == "z":
#             zscores = zscore(flat, axis=0)
#             pvals = 2 * (1 - norm.cdf(np.abs(zscores)))

#         elif test == "wilcoxon":
#             pvals = []
#             for col in range(flat.shape[1]):
#                 try:
#                     stat = wilcoxon(flat[:, col] - 0)
#                     pvals.append(stat.pvalue)
#                 except Exception:
#                     pvals.append(np.nan)
#             pvals = np.asarray(pvals)

#         elif test == "permutation w/o correction":
#             _, pvals = self._permutation_test(
#                 flat,
#                 popmean=0,
#                 axis=0,
#                 correct_alpha=False,
#                 n_permutations=n_permutations,
#                 alpha=alpha,
#             )

#         elif test == "permutation w correction":
#             _, pvals = self._permutation_test(
#                 flat,
#                 popmean=0,
#                 axis=0,
#                 correct_alpha=True,
#                 n_permutations=n_permutations,
#                 alpha=alpha,
#             )

#         else:
#             raise ValueError(f"Unsupported stat test: {test}")

#         # Reshape back into (n_elec, n_elec)
#         return pvals.reshape(n_elecs, n_elecs)
