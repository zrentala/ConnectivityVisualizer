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
        if self.conn_mat.ndim == 3:
            n_mat, n1, n2 = self.conn_mat.shape
            if n1 != n2:
                raise ValueError("For 3D conn_mat, last two dims must be square: (n_mat, n_nodes, n_nodes).")
            n_nodes = n1
        elif self.conn_mat.ndim == 2:
            n1, n2 = self.conn_mat.shape
            if n1 != n2:
                raise ValueError("2D conn_mat must be square (n_nodes × n_nodes).")
            n_nodes = n1
        else:
            raise ValueError("conn_mat must be 2D or 3D (n_mat, n_nodes, n_nodes).")

        if "label" not in self.chanlocs.columns:
            raise KeyError("chanlocs must have a 'label' column.")
        if len(self.chanlocs) != n_nodes:
            raise ValueError(
                f"chanlocs has {len(self.chanlocs)} rows but conn_mat implies {n_nodes} nodes."
            )

        self.n_nodes = n_nodes
        self.labels = self.chanlocs["label"].values
