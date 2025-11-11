import numpy as np
import analysis.statisticaltests as tests
from scipy.sparse.csgraph import minimum_spanning_tree

def get_basic_map(conn_mat: np.array, threshold:float):
    return (conn_mat >= threshold).astype(bool)

def get_mst_map(conn_mat: np.array) -> np.array:
    mst = minimum_spanning_tree(-conn_mat)  # negate to get maximum spanning tree
    mst_map = mst.toarray() != 0
    return mst_map.astype(bool)

# def get_stattest_map(conn_mat: np.array, alpha: float, test: str = "t") -> np.array:
#     n = conn_mat.shape[0]
#     test_func = tests.get_test(test)
#     _, p_values = test_func(conn_mat, popmean=0, axis=0)
#     stat_map = p_values < alpha
#     return stat_map.astype(bool)