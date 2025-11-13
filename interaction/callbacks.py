from dash import Input, Output, Dash
import numpy as np
import plotly.graph_objects as go
import plotly.colors as plc
from visualization.brainvisualizer import ConnectivityVisualizer
from utils.global_app_state import GlobalAppState


def register_visualization_callback(app: Dash, global_state: GlobalAppState):
    """Register callback to update visualization based on matrix index, threshold, and viz type."""
    n_frames = int(global_state.data.conn_matrices.shape[0])
    conn_matrices = global_state.data.conn_matrices
    chanlocs = global_state.chanlocs
    brain_mesh = global_state.brain_mesh

    @app.callback(
        Output("main-visualization", "figure"),
        Input("mat-idx", "value"),
        # threshold type dropdown (Basic / MST / Statistical Test)
        Input("thresh-comp-type-dropdown", "value"),
        # numeric threshold slider (percent)
        Input("thresh-comp-slider", "value"),
        Input("viz-type-dropdown", "value"),
        Input("color-type-dropdown", "value"),
        Input("conn-range", "value"),
        # alpha slider inside the statistical-test subcomponent
        Input("thresh-comp-alpha-slider", "value"),
        prevent_initial_call=False,
    )
    def update_visualization(idx, thresh_type, thresh_value, viz_type, color_name, conn_range, alpha):
        """Update the main visualization figure.

        The signature must match the decorated Inputs exactly.
        """
        idx = int(np.clip(idx or 0, 0, n_frames - 1))
        viz_type = (viz_type or "2D")

        # conn_range is expected to be a two-element sequence [min, max]
        try:
            conn_min, conn_max = float(conn_range[0]), float(conn_range[1])
        except Exception:
            conn_min, conn_max = 0.0, 1.0

        viz = global_state.viz

        if viz_type == "2D":
            fig = _build_2d_figure(viz, thresh_type, thresh_value, color_name, conn_min, conn_max, alpha)
        elif viz_type == "3D":
            fig = _build_3d_figure(viz, thresh_type, thresh_value, color_name, conn_min, conn_max, alpha)
        elif viz_type == "Heatmap":
            fig = _build_heatmap_figure(viz, thresh_type, thresh_value, color_name, conn_min, conn_max, alpha)
        else:
            fig = go.Figure()

        fig.update_layout(uirevision="keep")
        return fig


def _map_colors_for_name(name: str):
    """Return a small color mapping (pos, neg, node) and colorscale name.

    Falls back to simple defaults if the named sequential scale is not found.
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


def _build_2d_figure(viz: ConnectivityVisualizer, thresh_type: str, thresh_value: float, color_name: str, conn_min: float = 0.0, conn_max: float = 1.0, alpha: float = 5.0) -> go.Figure:
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
        alpha=alpha,
    )


def _build_3d_figure(viz: ConnectivityVisualizer, thresh_type: str, thresh_value: float, color_name: str, conn_min: float = 0.0, conn_max: float = 1.0, alpha:float = 5.0) -> go.Figure:
    cmap = _map_colors_for_name(color_name)
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
        alpha=alpha
    )

    # Ensure node marker coloring uses the node fill choice
    for tr in fig.data:
        if getattr(tr, "type", "") == "scatter3d":
            if getattr(tr, "mode", "") and "markers" in tr.mode:
                if hasattr(tr, "marker"):
                    tr.marker.color = cmap["node_fill"]
    return fig


def _build_heatmap_figure(viz: ConnectivityVisualizer, thresh_type: str, thresh_value: float, color_name: str, conn_min: float = 0.0, conn_max: float = 1.0, alpha:float = 5.0) -> go.Figure:
    cmap = _map_colors_for_name(color_name)
    fig = viz.figure_heatmap(threshold=thresh_value, threshold_type=thresh_type, colorscale=cmap["colorscale"], conn_min=conn_min, conn_max=conn_max, alpha=alpha)
    return fig


def register_threshold_callback(app: Dash, global_state: GlobalAppState):
    """Show/hide the threshold slider and stat-test container based on selection."""
    @app.callback(
        Output("thresh-comp-slider-container", "style"),
        Output("thresh-comp-stat-test-container", "style"),
        Input("thresh-comp-type-dropdown", "value"),
        prevent_initial_call=False,
    )
    def toggle_threshold_slider(thresh_type):
        show = {"display": "block"}
        hide = {"display": "none"}

        if thresh_type == "Basic":
            return show, hide
        elif thresh_type == "Statistical Test":
            return hide, show
        else:
            return hide, hide


def register_callbacks(app: Dash, global_state: GlobalAppState):
    """Attach all interaction callbacks to the Dash app."""
    register_visualization_callback(app, global_state)
    register_threshold_callback(app, global_state)
    
