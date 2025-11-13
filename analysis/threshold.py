import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.stats import ttest_1samp, norm, zscore, wilcoxon
from typing import Union, Tuple

from statsmodels.stats.multitest import multipletests
     


def get_basic_map(conn_mat: np.array, threshold:float):
    return (conn_mat >= threshold).astype(bool)

def get_mst_map(conn_mat: np.array) -> np.array:
    mst = minimum_spanning_tree(-conn_mat)  # negate to get maximum spanning tree
    mst_map = mst.toarray() != 0
    return mst_map.astype(bool)

def _bonferroni_correction(p_values, alpha):
    """
    Return Bonferroni-adjusted p-values (not a boolean mask).

    The Bonferroni correction multiplies raw p-values by the number of tests and
    clips to 1.0. This function returns the adjusted p-values so callers can
    use them or threshold them as needed (e.g., p_adj < alpha).
    """
    p = np.asarray(p_values, dtype=float)
    m = p.size
    if m == 0:
        return p
    p_adj = np.minimum(p * float(m), 1.0)
    return p_adj
def _permutation_test(data, popmean=0, axis=0, n_permutations=1000, correct_alpha=True):
    """
    One-sample permutation test (sign-flip) across rows for each feature (column).

    Parameters:
        data: array-like, shape (n_samples, n_features)
        popmean: value to center data against (default 0)
        axis: which axis corresponds to samples (default 0)
        n_permutations: number of random sign-flip permutations
        correct_alpha: if True, return Bonferroni-adjusted p-values instead of raw p-values

    Returns:
        (None, p_values) where p_values is array of length n_features
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
        # +1 correction to avoid zeros
        p_values[i] = (count + 1) / (n_permutations + 1)

    if correct_alpha:
        return None, _bonferroni_correction(p_values, alpha=0.05)
    return None, p_values


def perm_test_between_groups(ctrl_stack: np.ndarray, action_stack: np.ndarray, n_permutations: int = 5000) -> np.ndarray:
    """
    Two-sample permutation test that compares means between two groups for each feature.

    Parameters
    ----------
    ctrl_stack : array-like, shape (n_ctrl, n_features)
    action_stack : array-like, shape (n_action, n_features)
    n_permutations : int
        Number of random reallocations.

    Returns
    -------
    p_values : np.ndarray
        Array of p-values with shape (n_features,)
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


def get_stattest_mask(conn_mat: np.array, alpha: float = 0.05, test: str = "t", n_permutations: int = 1000) -> np.array:
    """Compatibility wrapper: return boolean mask (p < alpha) using get_stattest_map p-values.

    Supports passing a tuple/list (ctrl_stack, action_stack) when using `test='permutation'`.
    """
    print(conn_mat)
    pvals = get_stattest_map(conn_mat, alpha=alpha, test=test, n_permutations=n_permutations)
    print(pvals)
    print(pvals < alpha)
    return (pvals < alpha).astype(bool)

def get_stattest_map(conn_mat: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]], alpha: float = 0.05, test: str = "t", n_permutations: int = 1000) -> np.array:
    """
    Compute p-values for statistical tests applied to connectivity data.

    Parameters:
        conn_mat: array-like with shape (n_samples, n_features) or (n_samples, n_elec, n_elec)
        alpha: (ignored here) significance level (kept for compatibility)
        test: one of 't', 'z', 'wilcoxon', 'permutation'

    Returns:
        np.array of p-values. Shape depends on input: if conn_mat is 2D (n_samples, n_features)
        then returns (n_features,). If conn_mat is 3D (n_samples, n_elec, n_elec) it will
        return a matrix (n_elec, n_elec) of p-values.
    """
    arr = np.asarray(conn_mat)

    # If caller provided a tuple/list for two-sample permutation testing, handle that
    if isinstance(conn_mat, (list, tuple)) and test == 'permutation w/o correction' or test == 'permutation w correction':
        # expect (ctrl_stack, action_stack) where each is (n_samples, n_elec, n_elec) or (n_samples, n_features)
        ctrl, action = conn_mat
        ctrl_arr = np.asarray(ctrl)
        action_arr = np.asarray(action)
        # flatten per-sample matrices if needed
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

        p_values = perm_test_between_groups(ctrl_flat, action_flat, n_permutations=n_permutations)
        # reshape to matrix if we know n_elec
        if n_elec is not None:
            return p_values.reshape((n_elec, n_elec))
        return p_values

    # If 3D (samples x n_elec x n_elec), flatten features into columns for testing
    if arr.ndim == 3:
        n_samples, n_elec, _ = arr.shape
        flat = arr.reshape(n_samples, n_elec * n_elec)
    elif arr.ndim == 2:
        flat = arr
        n_samples = flat.shape[0]
    else:
        raise ValueError("conn_mat must be 2D or 3D array of samples x features")

    if test == "t":
        _, p_values = ttest_1samp(flat, popmean=0, axis=0)

    elif test == "z":
        z_scores = zscore(flat, axis=0)
        p_values = 2 * (1 - norm.cdf(np.abs(z_scores)))

    elif test == "wilcoxon":
        # Wilcoxon requires at least 2 samples; compute per-feature
        p_vals = []
        for i in range(flat.shape[1]):
            try:
                stat = wilcoxon(flat[:, i] - 0)
                p_vals.append(stat.pvalue)
            except Exception:
                p_vals.append(np.nan)
        p_values = np.asarray(p_vals)

    elif test == "permutation w/o correction":
        # _permutation_test returns (None, p_values)
        _, p_values = _permutation_test(flat, popmean=0, axis=0, n_permutations=n_permutations, correct_alpha=False)
    elif test == "permutation w correction":
        _, p_values = _permutation_test(flat, popmean=0, axis=0, n_permutations=n_permutations, correct_alpha=True)

    else:
        raise ValueError(f"Unsupported statistical test: {test}")

    # If input was 3D, reshape p-values back into (n_elec, n_elec)
    if arr.ndim == 3:
        return np.asarray(p_values).reshape((n_elec, n_elec))
    return np.asarray(p_values)

