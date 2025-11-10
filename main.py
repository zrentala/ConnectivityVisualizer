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

class App:
    def __init__(self):
        pass

def build_brain_mesh() -> pv.PolyData:
    fs_dir = mne.datasets.fetch_fsaverage(verbose=True)
    lh_pial = fs_dir / "surf" / "lh.pial"
    rh_pial = fs_dir / "surf" / "rh.pial"

    coords_lh, faces_lh = mne.read_surface(lh_pial)
    coords_rh, faces_rh = mne.read_surface(rh_pial)

    faces_lh_vtk = np.column_stack((np.full(len(faces_lh), 3), faces_lh))
    faces_rh_vtk = np.column_stack((np.full(len(faces_rh), 3), faces_rh + len(coords_lh)))

    coords_combined = np.vstack((coords_lh, coords_rh))
    faces_combined  = np.vstack((faces_lh_vtk, faces_rh_vtk))

    return pv.PolyData(coords_combined, faces_combined)


def create_app() -> Dash:
    app = Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])

    # ---- Simulated data ----
    cfg = {"n_elec": 20, "directed": False, "n_mat": 10}
    sim_data = Simulation(cfg)  # (n_mat, n, n)

    locs = pd.DataFrame(
        {
            "label": [f"E{i}" for i in range(sim_data.n_elec)],
            "x": sim_data.locations[:, 0] * 100,
            "y": sim_data.locations[:, 1] * 100,
            "z": sim_data.locations[:, 2] * 100,
        }
    )

    brain_mesh = build_brain_mesh()

    n_mat = sim_data.conn_matrices.shape[0]

    # ---- Prebuild *all* frames for each view type ----
    frame_figs: dict[str, list] = {
        "2D": [],
        "3D": [],
        "Heatmap": [],
    }

    for i in range(n_mat):
        viz_i = ConnectivityVisualizer(sim_data.conn_matrices[i], locs, brain_mesh=brain_mesh)
        frame_figs["2D"].append(viz_i.figure_2d())
        frame_figs["3D"].append(viz_i.figure_3d())
        frame_figs["Heatmap"].append(viz_i.figure_heatmap())

    # Initial figures for the layout: just take frame 0 of each
    initial_figs = {label: figs[0] for label, figs in frame_figs.items()}

    # ---- Layout ----
    app.layout = create_layout(
        initial_figs=initial_figs,
        n_mat=n_mat,
    )

    # ---- Register callbacks, passing the frame_figs dict ----
    register_callbacks(app, frame_figs)

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)
