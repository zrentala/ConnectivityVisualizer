# test_threshold.py

import numpy as np
import pytest

# Adjust this import to your actual module path
# e.g., from analysis.threshold import (
from analysis.threshold import (
    get_basic_map,
    get_mst_map,
    _bonferroni_correction,
    _permutation_test,
    perm_test_between_groups,
    get_stattest_mask,
    get_stattest_map,
)


def test_get_basic_map():
    conn = np.array([[0.1, 0.5],
                     [0.7, 0.2]])
    mask = get_basic_map(conn, threshold=0.5)
    expected = np.array([[False, True],
                         [True, False]])
    assert mask.dtype == bool
    assert np.array_equal(mask, expected)


def test_get_mst_map_simple():
    # 3-node fully connected graph with distinct weights
    conn = np.array([
        [0.0, 1.0, 0.5],
        [1.0, 0.0, 0.8],
        [0.5, 0.8, 0.0],
    ])
    mst_mask = get_mst_map(conn)
    # Maximum spanning tree should pick edges with weights 1.0 and 0.8 (between nodes (0,1) and (1,2))
    expected = np.array([
        [False, True,  False],
        [True,  False, True ],
        [False, True,  False],
    ])
    assert mst_mask.shape == conn.shape
    assert mst_mask.dtype == bool
    # Because the MST is undirected in this symmetric case, masks should match expected
    assert np.array_equal(mst_mask, expected)


def test_bonferroni_correction_basic():
    pvals = np.array([0.01, 0.02, 0.5])
    adj = _bonferroni_correction(pvals, alpha=0.05)
    # m = 3, so multiply by 3 and clip at 1
    expected = np.array([0.03, 0.06, 1.0])
    assert np.allclose(adj, expected)
    assert adj.shape == pvals.shape


def test_bonferroni_correction_empty():
    pvals = np.array([])
    adj = _bonferroni_correction(pvals, alpha=0.05)
    assert adj.size == 0
    assert adj.dtype == float


def test_permutation_test_returns_shape_and_bounds():
    rng = np.random.default_rng(123)
    data = rng.normal(loc=0.5, scale=1.0, size=(20, 5))  # 20 samples, 5 features

    _, pvals_raw = _permutation_test(data, popmean=0, axis=0, n_permutations=200, correct_alpha=False)
    _, pvals_adj = _permutation_test(data, popmean=0, axis=0, n_permutations=200, correct_alpha=True)

    assert pvals_raw.shape == (5,)
    assert pvals_adj.shape == (5,)

    assert np.all((pvals_raw >= 0.0) & (pvals_raw <= 1.0))
    assert np.all((pvals_adj >= 0.0) & (pvals_adj <= 1.0))

    # Adjusted p-values should be >= raw (Bonferroni multiplies then clips)
    assert np.all(pvals_adj >= pvals_raw)


def test_perm_test_between_groups_detects_difference():
    rng = np.random.default_rng(42)
    # Strong separation: means 0 vs 3
    ctrl = rng.normal(loc=0.0, scale=1.0, size=(30, 1))
    act = rng.normal(loc=3.0, scale=1.0, size=(30, 1))

    pvals = perm_test_between_groups(ctrl, act, n_permutations=1000)
    assert pvals.shape == (1,)
    # Should be highly significant
    assert pvals[0] < 0.01


def test_get_stattest_map_t_2d():
    rng = np.random.default_rng(7)
    data = rng.normal(loc=1.0, scale=1.0, size=(40, 10))  # mean > 0, t-test vs 0

    pvals = get_stattest_map(data, alpha=0.05, test="t")
    assert pvals.shape == (10,)
    # Most features should be significant at 0.05
    assert (pvals < 0.05).sum() >= 7


def test_get_stattest_map_t_3d():
    rng = np.random.default_rng(8)
    n_samples, n_elec = 30, 4
    data = rng.normal(loc=0.5, scale=1.0, size=(n_samples, n_elec, n_elec))

    pvals = get_stattest_map(data, alpha=0.05, test="t")
    assert pvals.shape == (n_elec, n_elec)
    assert np.all((pvals >= 0.0) & (pvals <= 1.0))


def test_get_stattest_map_z():
    rng = np.random.default_rng(9)
    data = rng.normal(loc=1.0, scale=1.0, size=(50, 6))

    pvals = get_stattest_map(data, alpha=0.05, test="z")
    assert pvals.shape == (6,)
    assert np.all((pvals >= 0.0) & (pvals <= 1.0))


def test_get_stattest_map_wilcoxon():
    rng = np.random.default_rng(10)
    data = rng.normal(loc=0.8, scale=1.0, size=(25, 5))

    pvals = get_stattest_map(data, alpha=0.05, test="wilcoxon")
    assert pvals.shape == (5,)
    # Wilcoxon can be a bit less powerful, but at least a majority should be < 0.05
    assert (pvals < 0.05).sum() >= 3


def test_get_stattest_map_permutation_no_correction():
    rng = np.random.default_rng(11)
    data = rng.normal(loc=1.0, scale=1.0, size=(25, 4))

    pvals = get_stattest_map(data, alpha=0.05, test="permutation w/o correction", n_permutations=500)
    assert pvals.shape == (4,)
    assert np.all((pvals >= 0.0) & (pvals <= 1.0))


def test_get_stattest_map_permutation_with_correction():
    rng = np.random.default_rng(12)
    data = rng.normal(loc=1.0, scale=1.0, size=(25, 4))

    pvals = get_stattest_map(data, alpha=0.05, test="permutation w correction", n_permutations=500)
    assert pvals.shape == (4,)
    assert np.all((pvals >= 0.0) & (pvals <= 1.0))


def test_get_stattest_mask():
    rng = np.random.default_rng(13)
    data = rng.normal(loc=1.5, scale=1.0, size=(40, 5))

    mask = get_stattest_mask(data, alpha=0.05, test="t")
    assert mask.shape == (5,)
    assert mask.dtype == bool
    # Most features should be significant at 0.05
    assert mask.sum() >= 3


def test_get_stattest_map_two_sample_permutation_flat():
    rng = np.random.default_rng(14)
    ctrl = rng.normal(loc=0.0, scale=1.0, size=(30, 6))
    act = rng.normal(loc=2.0, scale=1.0, size=(30, 6))

    pvals = get_stattest_map((ctrl, act), alpha=0.05, test="permutation w/o correction", n_permutations=1000)
    assert pvals.shape == (6,)
    assert np.all((pvals >= 0.0) & (pvals <= 1.0))
    # Most should reflect strong difference
    assert (pvals < 0.01).sum() >= 4


def test_get_stattest_map_two_sample_permutation_matrix():
    rng = np.random.default_rng(15)
    n_samples, n_elec = 20, 3
    # ctrl group: mean 0; action group: mean 1
    ctrl = rng.normal(loc=0.0, scale=1.0, size=(n_samples, n_elec, n_elec))
    act = rng.normal(loc=1.0, scale=1.0, size=(n_samples, n_elec, n_elec))

    pvals = get_stattest_map((ctrl, act), alpha=0.05, test="permutation w/o correction", n_permutations=500)
    assert pvals.shape == (n_elec, n_elec)
    assert np.all((pvals >= 0.0) & (pvals <= 1.0))


def test_get_stattest_map_invalid_dim_raises():
    # 1D array should be invalid
    data = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        get_stattest_map(data, alpha=0.05, test="t")
