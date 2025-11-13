from dash import Input, Output, Dash
import numpy as np
import plotly.graph_objects as go
import plotly.colors as plc
from visualization.brainvisualizer import ConnectivityVisualizer
from utils.global_app_state import GlobalAppState


def register_visualization_callback(app: Dash, global_state: GlobalAppState):
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
        Input("color-type-dropdown", "value"),
        Input("conn-range", "value"),
        prevent_initial_call=False,
    )
    def update_visualization(idx, thresh_type, thresh_value, viz_type, color_name, conn_range):
        """Update the main visualization with efficient patching.

        Inputs must match the decorated Inputs exactly (idx, thresh_type, thresh_value,
        viz_type, color_name, conn_range).
        """
        idx = int(np.clip(idx or 0, 0, n_frames - 1))
        viz_type = (viz_type or "2D")

        # conn_range is expected to be a sequence [min, max]
        try:
            conn_min, conn_max = float(conn_range[0]), float(conn_range[1])
        except Exception:
            conn_min, conn_max = 0.0, 1.0

        # Create visualizer for current matrix
        viz = ConnectivityVisualizer(conn_matrices[idx], chanlocs, brain_mesh=brain_mesh)

        # Generate the appropriate figure based on viz type using helper builders
        if viz_type == "2D":
            fig = _build_2d_figure(viz, thresh_type, thresh_value, color_name, conn_min, conn_max)
        elif viz_type == "3D":
            fig = _build_3d_figure(viz, thresh_type, thresh_value, color_name, conn_min, conn_max)
        elif viz_type == "Heatmap":
            fig = _build_heatmap_figure(viz, thresh_type, thresh_value, color_name, conn_min, conn_max)
        else:
            fig = go.Figure()

        fig.update_layout(uirevision="keep")
        return fig


def _map_colors_for_name(name: str):
    """Return a small color mapping (pos, neg, node) for a given color map name.

    If the named sequential scale exists in plotly.colors.sequential, use its
    endpoints and midpoint. Otherwise fall back to simple color choices.
    """
    name = (name or "Viridis")
    seq = getattr(plc.sequential, name, None)
    if seq and len(seq) >= 3:
        pos = seq[-1]
        neg = seq[0]
        node = seq[len(seq) // 2]
        return {"pos_color": pos, "neg_color": neg, "node_fill": node, "colorscale": name}
    # fallback
    return {"pos_color": "red", "neg_color": "blue", "node_fill": "lightgreen", "colorscale": "RdBu"}


def _build_2d_figure(viz: ConnectivityVisualizer, thresh_type: str, thresh_value: float, color_name: str, conn_min: float = 0.0, conn_max: float = 1.0) -> go.Figure:
    """Helper to build a 2D figure from a ConnectivityVisualizer instance with color mapping."""
    cmap = _map_colors_for_name(color_name)
    return viz.figure_2d(
        threshold=thresh_value,
        threshold_type=thresh_type,
        pos_color=cmap["pos_color"],
        neg_color=cmap["neg_color"],
        node_fill=cmap["node_fill"],
        colorscale=cmap["colorscale"],
        conn_min=conn_min,
        conn_max=conn_max,
    )


def _build_3d_figure(viz: ConnectivityVisualizer, thresh_type: str, thresh_value: float, color_name: str, conn_min: float = 0.0, conn_max: float = 1.0) -> go.Figure:
    """Helper to build a 3D figure from a ConnectivityVisualizer instance with color mapping."""
    cmap = _map_colors_for_name(color_name)
    # 3D uses single line color; use pos_color for edges and node_fill for markers
    fig = viz.figure_3d(
        threshold=thresh_value,
        threshold_type=thresh_type,
        line_width=3.0,
        opacity=0.6,
        title=None,
        directed=True,
        conn_min=conn_min,
        conn_max=conn_max,
        colorscale=cmap["colorscale"],
    )

    # Post-process traces: set node marker color and edge line color where applicable
    for tr in fig.data:
        # scatter3d nodes: set marker color
        if getattr(tr, "type", "") == "scatter3d":
            if getattr(tr, "mode", "") and "markers" in tr.mode:
                if hasattr(tr, "marker"):
                    tr.marker.color = cmap["node_fill"]
            # leave line colors as produced by the visualizer (per-edge colorscale sampling)
    return fig


def _build_heatmap_figure(viz: ConnectivityVisualizer, thresh_type: str, thresh_value: float, color_name: str, conn_min: float = 0.0, conn_max: float = 1.0) -> go.Figure:
    """Helper to build a heatmap figure from a ConnectivityVisualizer instance with colorscale."""
    cmap = _map_colors_for_name(color_name)
    fig = viz.figure_heatmap(threshold=thresh_value, threshold_type=thresh_type, colorscale=cmap["colorscale"], conn_min=conn_min, conn_max=conn_max)

    # conn_min/conn_max are passed into the heatmap builder which maps them
    # into the data range; no need to modify the figure here.
    return fig


def register_threshold_callback(app: Dash, global_state: GlobalAppState):
    """Register callback to show/hide threshold slider and statistical test input."""
    @app.callback(
        Output("thresh-comp-slider-container", "style"),
        Output("thresh-comp-stat-test-container", "style"),
        Input("thresh-comp", "value"),
        prevent_initial_call=False,
    )
    def toggle_threshold_slider(thresh_type):
        """
        Show the slider if 'Basic' is selected,
        show the stat-test component if 'Statistical Test' is selected,
        hide both otherwise.
        """
        show = {"display": "block"}
        hide = {"display": "none"}

        if thresh_type == "Basic":
            return show, hide
        elif thresh_type == "Statistical Test":
            return hide, show
        else:
            return hide, hide



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
    register_visualization_callback(app, global_state)
    register_threshold_callback(app, global_state)
    # register_tab_callback(app, global_state)
