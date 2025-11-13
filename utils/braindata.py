import numpy as np
import pandas as pd
import pyvista as pv
from dataclasses import dataclass

@dataclass
class BrainData:
    """Class to hold brain data parameters for visualization."""
    conn_mat: np.ndarray
    chanlocs: pd.DataFrame
    brain_mesh: pv.PolyData
    n_nodes: int
    directed: bool
    labels: np.ndarray
