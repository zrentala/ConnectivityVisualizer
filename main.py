from __future__ import annotations
import numpy as np
import pandas as pd
import mne
import pyvista as pv
from dash import Dash, Input, Output, State, exceptions
import dash_bootstrap_components as dbc

from visualization.brainvisualizer import ConnectivityVisualizer
from visualization.ui import create_layout
from data.simulation import Simulation
# from analysis.graph import GraphAnalysis  # if you need it

app = Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])

if __name__ == "__main__":
    # ---- Simulated data ----
    cfg = {'n_elec': 20, 'directed': False, 'n_mat': 10}
    sim_data = Simulation(cfg)  # sim_data.conn_matrices -> (n_mat, n, n)

    locs = pd.DataFrame({
        'label': [f'E{i}' for i in range(sim_data.n_elec)],
        'x': sim_data.locations[:, 0] * 100,
        'y': sim_data.locations[:, 1] * 100,
        'z': sim_data.locations[:, 2] * 100
    })

    # print(f"All conns: {sim_data.conn_matrices}")
    # ---- Build brain mesh (fsaverage) ----
    fs_dir = mne.datasets.fetch_fsaverage(verbose=True)
    lh_pial = fs_dir / 'surf' / 'lh.pial'
    rh_pial = fs_dir / 'surf' / 'rh.pial'
    coords_lh, faces_lh = mne.read_surface(lh_pial)
    coords_rh, faces_rh = mne.read_surface(rh_pial)
    faces_lh_vtk = np.column_stack((np.full(len(faces_lh), 3), faces_lh))
    faces_rh_vtk = np.column_stack((np.full(len(faces_rh), 3), faces_rh + len(coords_lh)))
    coords_combined = np.vstack((coords_lh, coords_rh))
    faces_combined  = np.vstack((faces_lh_vtk, faces_rh_vtk))
    brain_mesh = pv.PolyData(coords_combined, faces_combined)

    # ---- Visualizer seeded with the first frame ----
    viz = ConnectivityVisualizer(sim_data.conn_matrices[0], locs, brain_mesh=brain_mesh)

    # ---- Layout with the cube pre-loaded into dcc.Store ----
    app.layout = create_layout(
        viz,
        n_mat=sim_data.conn_matrices.shape[0],
        conn_cube_as_list=sim_data.conn_matrices.tolist()
    )

    FIGS_2D = [ConnectivityVisualizer(sim_data.conn_matrices[i], locs, brain_mesh).figure_2d()
           for i in range(sim_data.conn_matrices.shape[0])]
    FIGS_3D = [ConnectivityVisualizer(sim_data.conn_matrices[i], locs, brain_mesh).figure_3d()
           for i in range(sim_data.conn_matrices.shape[0])]
    # ---- Callback: update both figures when slider moves ----
    @app.callback(
        Output("viz-2d", "figure"),
        Output("viz-3d", "figure"),
        Output("frame-label", "children"),
        Input("mat-idx", "value"),
        prevent_initial_call=False,
    )
    def update_figures(idx):
        idx = int(np.clip(idx or 0, 0, len(FIGS_2D)-1))
        f2d, f3d = FIGS_2D[idx], FIGS_3D[idx]
        f2d.update_layout(uirevision="keep")
        f3d.update_layout(uirevision="keep")
        return f2d, f3d, f"Frame: {idx+1} / {len(FIGS_2D)}"



    app.run(debug=True)
