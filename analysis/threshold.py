from dataclasses import dataclass
from typing import Optional, Union, Tuple

import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.stats import ttest_1samp, norm, zscore, wilcoxon
from statsmodels.stats.multitest import multipletests  # still available if you want BH, etc.


@dataclass
class Threshold:
    """
    Threshold settings used to derive a boolean mask from a connectivity matrix.

    Attributes
    ----------
    threshold : float
        For 'Basic', interpreted as a percentage (0-100) of normalized connectivity.
    threshold_type : Optional[str]
        One of:
            - None / "None"             → no thresholding (all True mask)
            - "Basic"                   → simple value threshold on normalized 0-1 matrix
            - "Statistical Test"        → p < alpha using get_stattest_map
            - "MST" / "Minimum Spanning Tree"
    alpha : float
        For 'Statistical Test', interpreted as a percentage (0-100).
    """
    threshold: float = 0.5
    threshold_type: Optional[str] = None
    alpha: float = 5.0   # percent (5% default)

    # ------------------------------------------------------------------
    # Public: main entry point
    # ------------------------------------------------------------------

    def apply_threshold(self, C: np.ndarray) -> np.ndarray:
        """
        Given a connectivity matrix C (2D or 3D slice flattened to 2D),
        return a boolean mask with the same shape as C.

        For visualization we usually pass a single (n_elec, n_elec) matrix.
        """
        C = np.asarray(C)
        if C.ndim != 2:
            raise ValueError("Threshold.apply_threshold expects a 2D matrix (n_elec, n_elec).")

        # No thresholding: identity mask
        if self.threshold_type is None or str(self.threshold_type).strip() in {"", "None"}:
            return np.ones_like(C, dtype=bool)

        # Normalize to [0,1] for percentage-based thresholds
        c_min = float(np.min(C))
        c_max = float(np.max(C))
        denom = (c_max - c_min) + 1e-12
        conn_normalized = (C - c_min) / denom

        ttype = str(self.threshold_type)

        if ttype == "Basic":
            # 'threshold' is given in percent [0,100]
            norm_threshold = float(self.threshold) / 100.0
            return self.get_basic_map(conn_normalized, norm_threshold)

        elif ttype == "Statistical Test":
            # 'alpha' also given in percent
            norm_alpha = float(self.alpha) / 100.0
            return self.get_stattest_mask(conn_normalized, alpha=norm_alpha)

        elif ttype in {"MST", "Minimum Spanning Tree"}:
            return self.get_mst_map(conn_normalized)

        else:
            raise ValueError(f"Unknown threshold_type: {self.threshold_type!r}")

    # ------------------------------------------------------------------
    # Basic / MST masks
    # ------------------------------------------------------------------

    @staticmethod
    def get_basic_map(conn_mat: np.ndarray, threshold: float) -> np.ndarray:
        """
        Simple absolute threshold on a normalized matrix in [0,1].
        Returns a boolean mask.
        """
        conn_mat = np.asarray(conn_mat)
        return (conn_mat >= threshold).astype(bool)

    @staticmethod
    def get_mst_map(conn_mat: np.ndarray) -> np.ndarray:
        """
        Build maximum spanning tree mask from a (n_elec, n_elec) connectivity matrix.
        We negate the weights and compute a minimum spanning tree of -conn_mat.
        """
        conn_mat = np.asarray(conn_mat)
        if conn_mat.ndim != 2:
            raise ValueError("get_mst_map expects a 2D matrix (n_elec, n_elec).")

        # negate to get maximum spanning tree via minimum_spanning_tree
        mst = minimum_spanning_tree(-conn_mat)
        mst_map = mst.toarray() != 0
        return mst_map.astype(bool)

    # ------------------------------------------------------------------
    # Multiple testing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _bonferroni_correction(p_values: np.ndarray, alpha: float) -> np.ndarray:
        """
        Return Bonferroni-adjusted p-values (not a boolean mask).

        p_adj = min(p * m, 1.0)
        """
        p = np.asarray(p_values, dtype=float)
        m = p.size
        if m == 0:
            return p
        p_adj = np.minimum(p * float(m), 1.0)
        return p_adj

    # ------------------------------------------------------------------
    # One-sample permutation test
    # ------------------------------------------------------------------

    def _permutation_test(
        self,
        data,
        popmean: float = 0.0,
        axis: int = 0,
        n_permutations: int = 1000,
        correct_alpha: bool = True,
        alpha: float = 0.05,
    ):
        """
        One-sample permutation test (sign-flip) across rows for each feature (column).

        Parameters
        ----------
        data : array-like, shape (n_samples, n_features)
        popmean : float
            Population mean under H0.
        axis : int
            Axis corresponding to samples (default 0).
        n_permutations : int
        correct_alpha : bool
            If True, return Bonferroni-adjusted p-values.

        Returns
        -------
        (None, p_values) with p_values shape (n_features,)
        """
        arr = np.asarray(data)
        if axis != 0:
            arr = np.swapaxes(arr, 0, axis)

        n_samples, n_features = arr.shape
        observed_means = np.mean(arr - popmean, axis=0)
        p_values = np.zeros(n_features, dtype=float)

        for i in range(n_features):
            count = 0
            col = arr[:, i]
            for _ in range(int(n_permutations)):
                signs = np.random.choice([-1, 1], size=n_samples)
                permuted = (col - popmean) * signs
                permuted_mean = np.mean(permuted)
                if np.abs(permuted_mean) >= np.abs(observed_means[i]):
                    count += 1
            p_values[i] = (count + 1) / (n_permutations + 1)

        if correct_alpha:
            return None, self._bonferroni_correction(p_values, alpha=alpha)
        return None, p_values

    # ------------------------------------------------------------------
    # Two-sample permutation test
    # ------------------------------------------------------------------

    def perm_test_between_groups(
        self,
        ctrl_stack: np.ndarray,
        action_stack: np.ndarray,
        n_permutations: int = 5000,
    ) -> np.ndarray:
        """
        Two-sample permutation test comparing means between two groups for each feature.

        ctrl_stack : (n_ctrl, n_features) or (n_ctrl, n_elec, n_elec)
        action_stack : (n_action, n_features) or (n_action, n_elec, n_elec)
        """
        ctrl = np.asarray(ctrl_stack)
        act = np.asarray(action_stack)
        if ctrl.ndim == 1:
            ctrl = ctrl[:, None]
        if act.ndim == 1:
            act = act[:, None]

        n_ctrl, n_features = ctrl.shape
        n_act = act.shape[0]
        combined = np.vstack([ctrl, act])
        n_total = combined.shape[0]

        observed = np.mean(ctrl, axis=0) - np.mean(act, axis=0)
        counts = np.zeros(n_features, dtype=int)

        for _ in range(int(n_permutations)):
            perm_idx = np.random.choice(n_total, size=n_ctrl, replace=False)
            perm_ctrl = combined[perm_idx, :]
            mask = np.ones(n_total, dtype=bool)
            mask[perm_idx] = False
            perm_act = combined[mask, :]
            perm_diff = np.mean(perm_ctrl, axis=0) - np.mean(perm_act, axis=0)
            counts += (np.abs(perm_diff) >= np.abs(observed))

        p_values = (counts + 1) / (n_permutations + 1)
        return p_values

    # ------------------------------------------------------------------
    # Public: statistical test → mask
    # ------------------------------------------------------------------

    def get_stattest_mask(
        self,
        conn_mat: np.ndarray,
        alpha: float = 0.05,
        test: str = "t",
        n_permutations: int = 1000,
    ) -> np.ndarray:
        """
        Return boolean mask (p < alpha) using get_stattest_map p-values.
        """
        pvals = self.get_stattest_map(
            conn_mat,
            alpha=alpha,
            test=test,
            n_permutations=n_permutations,
        )
        return (pvals < alpha).astype(bool)

    def get_stattest_map(
        self,
        conn_mat: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]],
        alpha: float = 0.05,
        test: str = "t",
        n_permutations: int = 1000,
    ) -> np.ndarray:
        """
        Compute p-values for statistical tests applied to connectivity data.

        conn_mat:
            - 2D: (n_samples, n_features)
            - 3D: (n_samples, n_elec, n_elec)
            - Tuple (ctrl_stack, action_stack) for two-sample permutation tests
              when test in {"permutation w/o correction", "permutation w correction"}.
        """
        # Two-sample permutation case
        if isinstance(conn_mat, (list, tuple)) and test in {
            "permutation w/o correction",
            "permutation w correction",
        }:
            ctrl, action = conn_mat
            ctrl_arr = np.asarray(ctrl)
            action_arr = np.asarray(action)

            if ctrl_arr.ndim == 3:
                n_samples_ctrl, n_elec, _ = ctrl_arr.shape
                ctrl_flat = ctrl_arr.reshape(n_samples_ctrl, n_elec * n_elec)
            else:
                ctrl_flat = ctrl_arr
                n_elec = int(np.sqrt(ctrl_flat.shape[1])) if ctrl_flat.ndim == 2 else None

            if action_arr.ndim == 3:
                n_samples_act = action_arr.shape[0]
                action_flat = action_arr.reshape(n_samples_act, n_elec * n_elec)
            else:
                action_flat = action_arr

            p_values = self.perm_test_between_groups(ctrl_flat, action_flat, n_permutations=n_permutations)
            if n_elec is not None:
                return p_values.reshape((n_elec, n_elec))
            return p_values

        # One-sample tests: flatten if 3D
        arr = np.asarray(conn_mat)
        if arr.ndim == 3:
            n_samples, n_elec, _ = arr.shape
            flat = arr.reshape(n_samples, n_elec * n_elec)
        elif arr.ndim == 2:
            flat = arr
            n_samples = flat.shape[0]
            n_elec = None
        else:
            raise ValueError("conn_mat must be 2D or 3D array of samples x features")

        if test == "t":
            _, p_values = ttest_1samp(flat, popmean=0, axis=0)

        elif test == "z":
            z_scores = zscore(flat, axis=0)
            p_values = 2 * (1 - norm.cdf(np.abs(z_scores)))

        elif test == "wilcoxon":
            p_vals = []
            for i in range(flat.shape[1]):
                try:
                    stat = wilcoxon(flat[:, i] - 0)
                    p_vals.append(stat.pvalue)
                except Exception:
                    p_vals.append(np.nan)
            p_values = np.asarray(p_vals)

        elif test == "permutation w/o correction":
            _, p_values = self._permutation_test(
                flat,
                popmean=0,
                axis=0,
                n_permutations=n_permutations,
                correct_alpha=False,
                alpha=alpha,
            )
        elif test == "permutation w correction":
            _, p_values = self._permutation_test(
                flat,
                popmean=0,
                axis=0,
                n_permutations=n_permutations,
                correct_alpha=True,
                alpha=alpha,
            )
        else:
            raise ValueError(f"Unsupported statistical test: {test}")

        if arr.ndim == 3 and n_elec is not None:
            return np.asarray(p_values).reshape((n_elec, n_elec))
        return np.asarray(p_values)

# from dataclasses import dataclass
# import numpy as np
# from scipy.sparse.csgraph import minimum_spanning_tree
# from scipy.stats import ttest_1samp, norm, zscore, wilcoxon
# from typing import Union, Tuple
# from typing import Optional
# from statsmodels.stats.multitest import multipletests
     
# @dataclass
# class Threshold():
#     # Optional visualization parameters
#     threshold: float = 0.5
#     threshold_type: Optional[str] = None
#     alpha: float = 5.0


#     def apply_threshold(self, C: np.ndarray) -> np.ndarray:
#         conn_normalized = (C - np.min(C)) / (np.max(C) - np.min(C) + 1e-12)
        
#         if self.threshold_type == "Basic":
#             # Convert percentage to normalized threshold
#             norm_threshold = self.threshold / 100.0
#             return self.get_basic_map(conn_normalized, norm_threshold)
#         elif self.threshold_type == "Statistical Test":
#             # Convert percentage to normalized threshold
#             norm_alpha = self.alpha / 100.0
#             return self.get_stattest_mask(conn_normalized, norm_alpha)
#         elif self.threshold_type == "MST" or self.threshold_type == "Minimum Spanning Tree":  # Minimum Spanning Tree
#             return self.get_mst_map(conn_normalized)
    

#     def get_basic_map(self, conn_mat: np.array, threshold:float):
#         return (conn_mat >= threshold).astype(bool)

#     def get_mst_map(conn_mat: np.array) -> np.array:
#         mst = minimum_spanning_tree(-conn_mat)  # negate to get maximum spanning tree
#         mst_map = mst.toarray() != 0
#         return mst_map.astype(bool)

#     def _bonferroni_correction(self, p_values, alpha):
#         """
#         Return Bonferroni-adjusted p-values (not a boolean mask).

#         The Bonferroni correction multiplies raw p-values by the number of tests and
#         clips to 1.0. This function returns the adjusted p-values so callers can
#         use them or threshold them as needed (e.g., p_adj < alpha).
#         """
#         p = np.asarray(self, p_values, dtype=float)
#         m = p.size
#         if m == 0:
#             return p
#         p_adj = np.minimum(p * float(m), 1.0)
#         return p_adj
#     def _permutation_test(self, data, popmean=0, axis=0, n_permutations=1000, correct_alpha=True):
#         """
#         One-sample permutation test (sign-flip) across rows for each feature (column).

#         Parameters:
#             data: array-like, shape (n_samples, n_features)
#             popmean: value to center data against (default 0)
#             axis: which axis corresponds to samples (default 0)
#             n_permutations: number of random sign-flip permutations
#             correct_alpha: if True, return Bonferroni-adjusted p-values instead of raw p-values

#         Returns:
#             (None, p_values) where p_values is array of length n_features
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
#             # +1 correction to avoid zeros
#             p_values[i] = (count + 1) / (n_permutations + 1)

#         if correct_alpha:
#             return None, self._bonferroni_correction(p_values, alpha=0.05)
#         return None, p_values


#     def perm_test_between_groups(self, ctrl_stack: np.ndarray, action_stack: np.ndarray, n_permutations: int = 5000) -> np.ndarray:
#         """
#         Two-sample permutation test that compares means between two groups for each feature.

#         Parameters
#         ----------
#         ctrl_stack : array-like, shape (n_ctrl, n_features)
#         action_stack : array-like, shape (n_action, n_features)
#         n_permutations : int
#             Number of random reallocations.

#         Returns
#         -------
#         p_values : np.ndarray
#             Array of p-values with shape (n_features,)
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


#     def get_stattest_mask(self, conn_mat: np.array, alpha: float = 0.05, test: str = "t", n_permutations: int = 1000) -> np.array:
#         """Compatibility wrapper: return boolean mask (p < alpha) using get_stattest_map p-values.

#         Supports passing a tuple/list (ctrl_stack, action_stack) when using `test='permutation'`.
#         """
#         print(conn_mat)
#         pvals = self.get_stattest_map(conn_mat, alpha=alpha, test=test, n_permutations=n_permutations)
#         print(pvals)
#         print(pvals < alpha)
#         return (pvals < alpha).astype(bool)

#     def get_stattest_map(self, conn_mat: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]], alpha: float = 0.05, test: str = "t", n_permutations: int = 1000) -> np.array:
#         """
#         Compute p-values for statistical tests applied to connectivity data.

#         Parameters:
#             conn_mat: array-like with shape (n_samples, n_features) or (n_samples, n_elec, n_elec)
#             alpha: (ignored here) significance level (kept for compatibility)
#             test: one of 't', 'z', 'wilcoxon', 'permutation'

#         Returns:
#             np.array of p-values. Shape depends on input: if conn_mat is 2D (n_samples, n_features)
#             then returns (n_features,). If conn_mat is 3D (n_samples, n_elec, n_elec) it will
#             return a matrix (n_elec, n_elec) of p-values.
#         """
#         arr = np.asarray(conn_mat)

#         # If caller provided a tuple/list for two-sample permutation testing, handle that
#         if isinstance(conn_mat, (list, tuple)) and test == 'permutation w/o correction' or test == 'permutation w correction':
#             # expect (ctrl_stack, action_stack) where each is (n_samples, n_elec, n_elec) or (n_samples, n_features)
#             ctrl, action = conn_mat
#             ctrl_arr = np.asarray(ctrl)
#             action_arr = np.asarray(action)
#             # flatten per-sample matrices if needed
#             if ctrl_arr.ndim == 3:
#                 n_samples_ctrl, n_elec, _ = ctrl_arr.shape
#                 ctrl_flat = ctrl_arr.reshape(n_samples_ctrl, n_elec * n_elec)
#             else:
#                 ctrl_flat = ctrl_arr
#                 n_elec = int(np.sqrt(ctrl_flat.shape[1])) if ctrl_flat.ndim == 2 else None

#             if action_arr.ndim == 3:
#                 n_samples_act = action_arr.shape[0]
#                 action_flat = action_arr.reshape(n_samples_act, n_elec * n_elec)
#             else:
#                 action_flat = action_arr

#             p_values = self.perm_test_between_groups(ctrl_flat, action_flat, n_permutations=n_permutations)
#             # reshape to matrix if we know n_elec
#             if n_elec is not None:
#                 return p_values.reshape((n_elec, n_elec))
#             return p_values

#         # If 3D (samples x n_elec x n_elec), flatten features into columns for testing
#         if arr.ndim == 3:
#             n_samples, n_elec, _ = arr.shape
#             flat = arr.reshape(n_samples, n_elec * n_elec)
#         elif arr.ndim == 2:
#             flat = arr
#             n_samples = flat.shape[0]
#         else:
#             raise ValueError("conn_mat must be 2D or 3D array of samples x features")

#         if test == "t":
#             _, p_values = ttest_1samp(flat, popmean=0, axis=0)

#         elif test == "z":
#             z_scores = zscore(flat, axis=0)
#             p_values = 2 * (1 - norm.cdf(np.abs(z_scores)))

#         elif test == "wilcoxon":
#             # Wilcoxon requires at least 2 samples; compute per-feature
#             p_vals = []
#             for i in range(flat.shape[1]):
#                 try:
#                     stat = wilcoxon(flat[:, i] - 0)
#                     p_vals.append(stat.pvalue)
#                 except Exception:
#                     p_vals.append(np.nan)
#             p_values = np.asarray(p_vals)

#         elif test == "permutation w/o correction":
#             # _permutation_test returns (None, p_values)
#             _, p_values = self._permutation_test(flat, popmean=0, axis=0, n_permutations=n_permutations, correct_alpha=False)
#         elif test == "permutation w correction":
#             _, p_values = self._permutation_test(flat, popmean=0, axis=0, n_permutations=n_permutations, correct_alpha=True)

#         else:
#             raise ValueError(f"Unsupported statistical test: {test}")

#         # If input was 3D, reshape p-values back into (n_elec, n_elec)
#         if arr.ndim == 3:
#             return np.asarray(p_values).reshape((n_elec, n_elec))
#         return np.asarray(p_values)

