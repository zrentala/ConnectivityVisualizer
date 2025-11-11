from dash import Input, Output
import numpy as np
import plotly.graph_objects as go
from visualization.brainvisualizer import ConnectivityVisualizer


def register_slider_callback(app, global_state):
    """Updates all figures when the slider changes (precomputed frames)."""
    labels = list(global_state.figs_cache.keys())
    n_frames = len(next(iter(global_state.figs_cache.values())))
    id_map = global_state.viz_tabs_builder.id_map
    frame_figs = global_state.figs_cache

    @app.callback(
        [Output(id_map[label], "figure") for label in labels]
        + [Output("frame-label", "children")],
        Input("mat-idx", "value"),
        prevent_initial_call=False,
    )
    def update_figures_from_slider(idx):
        idx = int(np.clip(idx or 0, 0, n_frames - 1))
        out_figs = []
        for label in labels:
            fig = frame_figs[label][idx]
            fig.update_layout(uirevision="keep")
            out_figs.append(fig)

        frame_text = f"Frame: {idx + 1} / {n_frames}"
        return (*out_figs, frame_text)


def register_tab_callback(app, global_state):
    """Recomputes the active tab’s figure dynamically when switching tabs."""
    labels = list(global_state.figs_cache.keys())
    id_map = global_state.viz_tabs_builder.id_map
    conn_matrices = global_state.data.conn_matrices
    chanlocs = global_state.chanlocs
    brain_mesh = global_state.brain_mesh
    n_frames = conn_matrices.shape[0]

    @app.callback(
        [Output(id_map[label], "figure") for label in labels],
        Input("viz-tabs", "active_tab"),
        Input("mat-idx", "value"),
        prevent_initial_call=True,
    )
    def recompute_active_tab(active_tab, idx):
        idx = int(np.clip(idx or 0, 0, n_frames - 1))
        out_figs = []

        for label in labels:
            if label == active_tab:
                viz = ConnectivityVisualizer(
                    conn_matrices[idx],
                    chanlocs,
                    brain_mesh=brain_mesh,
                )
                if label == "2D":
                    fig = viz.figure_2d()
                elif label == "3D":
                    fig = viz.figure_3d()
                else:
                    fig = viz.figure_heatmap()
            else:
                fig = go.Figure()
            out_figs.append(fig)

        return out_figs


def register_callbacks(app, global_state):
    """Attach all Dash callbacks."""
    register_slider_callback(app, global_state)
    register_tab_callback(app, global_state)
