from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pyvista as pv


@dataclass
class BrainData:
    """
    Container for brain connectivity and geometry for visualization.
    Only core inputs are stored; derived fields are computed automatically.
    """
    conn_mat: np.ndarray
    chanlocs: pd.DataFrame
    brain_mesh: pv.PolyData
    directed: bool = False

    # Derived fields populated post-init
    n_nodes: int = field(init=False)
    labels: np.ndarray = field(init=False)

    def __post_init__(self):
        # Validate inputs
        if self.conn_mat.ndim > 3:
            raise ValueError("conn_mat must be a 2D matrix (n_nodes × n_nodes).")

        if self.conn_mat.shape[1] != self.conn_mat.shape[2]:
            raise ValueError("conn_mat must be square.")

        if "label" not in self.chanlocs.columns:
            raise KeyError("chanlocs must have a 'label' column.")

        # Compute derived fields
        self.n_nodes = self.conn_mat.shape[0]
        self.labels = self.chanlocs["label"].values