import numpy as np
import pandas as pd
import pyvista as pv
from dataclasses import dataclass

@dataclass
class BrainData:
    """Class to hold brain data parameters for visualization."""
    conn_mat: np.ndarray # Connectivity matrices, shape (n_matrices, n_nodes, n_nodes)
    chanlocs: pd.DataFrame
    brain_mesh: pv.PolyData
    n_nodes: int
    directed: bool
    labels: np.ndarray

    def __init__(self, conn_mat: np.ndarray, chanlocs: pd.DataFrame, brain_mesh: pv.PolyData, directed: bool = False):
        self.conn_mat = conn_mat
        self.chanlocs = chanlocs
        self.brain_mesh = brain_mesh
        self.n_nodes = conn_mat.shape[0]
        self.is_directed = directed
        self.labels = chanlocs['label'].values