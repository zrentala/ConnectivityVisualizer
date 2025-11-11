from dash import Input, Output, Dash, Patch
import numpy as np
import plotly.graph_objects as go
from visualization.brainvisualizer import ConnectivityVisualizer
from utils.global_app_state import GlobalAppState


def register_slider_callback(app: Dash, global_state: GlobalAppState):
    """Register callback to update visualization based on matrix index, threshold, and viz type."""
    n_frames = global_state.data.conn_matrices.shape[0]
    conn_matrices = global_state.data.conn_matrices
    chanlocs = global_state.chanlocs
    brain_mesh = global_state.brain_mesh

    @app.callback(
        Output("main-visualization", "figure"),
        Input("mat-idx", "value"),
        Input("thresh-comp", "value"),
        Input("thresh-comp-slider", "value"),
        Input("viz-type-dropdown", "value"),
        prevent_initial_call=False,
    )
    def update_visualization(idx, thresh_type, thresh_value, viz_type):
        """Update the main visualization with efficient patching."""
        idx = int(np.clip(idx or 0, 0, n_frames - 1))
        viz_type = viz_type or "2D"

        # Create visualizer for current matrix
        viz = ConnectivityVisualizer(
            conn_matrices[idx],
            chanlocs,
            brain_mesh=brain_mesh,
        )

        # Generate the appropriate figure based on viz type
        if viz_type == "2D":
            fig = viz.figure_2d(threshold=thresh_value, threshold_type=thresh_type)
        elif viz_type == "3D":
            fig = viz.figure_3d(threshold=thresh_value, threshold_type=thresh_type)
        elif viz_type == "Heatmap":
            fig = viz.figure_heatmap(threshold=thresh_value, threshold_type=thresh_type)
        else:
            fig = go.Figure()
        
        fig.update_layout(uirevision="keep")
        return fig


def register_threshold_callback(app: Dash, global_state: GlobalAppState):
    """Register callback to show/hide threshold slider based on threshold type."""
    @app.callback(
        Output("thresh-comp-slider-container", "style"),
        Input("thresh-comp", "value"),
        prevent_initial_call=False,
    )
    def toggle_threshold_slider(thresh_type):
        """Show slider for 'Basic' threshold, hide for 'Minimum Spanning Tree'."""
        if thresh_type == "Basic":
            return {"display": "block"}
        else:
            return {"display": "none"}


# def register_tab_callback(app: Dash,  global_state: GlobalAppState):
#     """Recomputes the active tab’s figure dynamically when switching tabs."""
#     labels = list(global_state.figs_cache.keys())
#     id_map = global_state.viz_tabs_builder.id_map
#     conn_matrices = global_state.data.conn_matrices
#     chanlocs = global_state.chanlocs
#     brain_mesh = global_state.brain_mesh
#     n_frames = conn_matrices.shape[0]

#     @app.callback(
#         [Output(id_map[label], "figure") for label in labels],
#         Input("viz-tabs", "active_tab"),
#         Input("mat-idx", "value"),
#         prevent_initial_call=True,
#     )
#     def recompute_active_tab(active_tab, idx):
#         idx = int(np.clip(idx or 0, 0, n_frames - 1))
#         out_figs = []

#         for label in labels:
#             if label == active_tab:
#                 viz = ConnectivityVisualizer(
#                     conn_matrices[idx],
#                     chanlocs,
#                     brain_mesh=brain_mesh,
#                 )
#                 if label == "2D":
#                     fig = viz.figure_2d()
#                 elif label == "3D":
#                     fig = viz.figure_3d()
#                 else:
#                     fig = viz.figure_heatmap()
#             else:
#                 fig = go.Figure()
#             out_figs.append(fig)

#         return out_figs


def register_callbacks(app: Dash, global_state: GlobalAppState):
    """Attach all Dash callbacks."""
    register_slider_callback(app, global_state)
    register_threshold_callback(app, global_state)
    # register_tab_callback(app, global_state)
