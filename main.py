# app.py
from __future__ import annotations
import numpy as np
import pandas as pd
import mne
import pyvista as pv
from dash import Dash
import dash_bootstrap_components as dbc

from visualization.brainvisualizer import ConnectivityVisualizer
from visualization.ui import create_layout
from data.simulation import Simulation
from interaction.callbacks import register_callbacks   # <-- new

class GlobalAppState:
    # brain_mesh
    # data
    # elec_locs
    # figs_cache
    

    def __init__(self):
        self.app = Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
        # ---- Simulated data ----
        cfg = {"n_elec": 20, "directed": False, "n_mat": 10}
        sim_data = Simulation(cfg)  # (n_mat, n, n)
        self.chanlocs = pd.DataFrame(
            {
                "label": [f"E{i}" for i in range(sim_data.n_elec)],
                "x": sim_data.locations[:, 0] * 100,
                "y": sim_data.locations[:, 1] * 100,
                "z": sim_data.locations[:, 2] * 100,
            }
        )
        self.data = sim_data

        self.brain_mesh = sim_data.build_brain_mesh()

        self.figs_cache = {
            "2D": [],
            "3D": [],
            "Heatmap": [],
        }

        n_mat = self.data.conn_matrices.shape[0]
        for i in range(n_mat):
            viz_i = ConnectivityVisualizer(sim_data.conn_matrices[i], self.chanlocs, brain_mesh=self.brain_mesh)
            self.figs_cache ["2D"].append(viz_i.figure_2d())
            self.figs_cache ["3D"].append(viz_i.figure_3d())
            self.figs_cache ["Heatmap"].append(viz_i.figure_heatmap())

        

    def create_app(self) -> Dash:  
        app = Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
        n_mat = self.data.conn_matrices.shape[0]
        initial_figs = {label: figs[0] for label, figs in self.figs_cache.items()}
        app.layout = create_layout(
            initial_figs=initial_figs,
            n_mat=n_mat,
        )

        # ---- Register callbacks, passing the frame_figs dict ----
        register_callbacks(app, self.figs_cache)

        return app


if __name__ == "__main__":
    global_state = GlobalAppState()
    app = global_state.create_app()
    app.run(debug=True)
