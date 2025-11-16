from dash import Input, Output, State, Dash, no_update, callback_context
import numpy as np
import plotly.graph_objects as go
import plotly.colors as plc
from data.loader import DataLoader, PRESET_CONFIGS  # adjust import path if needed


from visualization.brainvisualizer import ConnectivityVisualizer
from utils.global_app_state import GlobalAppState

# NEW:
from data.simulation import Simulation
from utils.braindata import BrainData
import pandas as pd
import base64
import io

PRESET_CONFIGS = {
    "small_undirected": {"n_elec": 10, "directed": False, "n_mat": 5},
    "medium_directed": {"n_elec": 20, "directed": True, "n_mat": 10},
    "large_undirected": {"n_elec": 64, "directed": False, "n_mat": 20},
}


def register_visualization_callback(app: Dash, global_state: GlobalAppState):
    """Register callback to update visualization based on matrix index, threshold, and viz type."""
    n_frames = int(global_state.brain_data.conn_mat.shape[0])
    # conn_mat = global_state.brain_data.conn_mat
    # chanlocs = global_state.brain_data.chanlocs
    # brain_mesh = global_state.brain_data.brain_mesh

    @app.callback(
        Output("main-visualization", "figure"),
        Input("data-comp-mat-idx", "value"),
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

        # viz = ConnectivityVisualizer(conn_mat[idx], chanlocs, brain_mesh=brain_mesh)

        viz = global_state.viz
        viz.update_fields(
            brain_data=global_state.brain_data,
            viz_type=viz_type,
            conn_idx=idx,
            thresh_type=thresh_type,
            thresh_value=thresh_value,
            color_name=color_name,
            conn_min=conn_min,
            conn_max=conn_max,
            alpha=alpha,
        )

        fig = viz.get_figure(brain_data=global_state.brain_data)
        cmap = _map_colors_for_name(color_name)
        fig.update_layout(uirevision="keep")
        return fig
        # if viz_type == "2D":
        #     fig = _build_2d_figure(viz, thresh_type, thresh_value, color_name, conn_min, conn_max, alpha)
        # elif viz_type == "3D":
        #     fig = _build_3d_figure(viz, thresh_type, thresh_value, color_name, conn_min, conn_max, alpha)
        # elif viz_type == "Heatmap":
        #     fig = _build_heatmap_figure(viz, thresh_type, thresh_value, color_name, conn_min, conn_max, alpha)
        # else:
        #     fig = go.Figure()

        # fig.update_layout(uirevision="keep")
        # return fig

def _brain_data_from_sim(cfg: dict) -> BrainData:
    sim = Simulation(cfg)
    chanlocs = pd.DataFrame(
        {
            "label": [f"E{i}" for i in range(sim.n_elec)],
            "x": sim.locations[:, 0] * 100,
            "y": sim.locations[:, 1] * 100,
            "z": sim.locations[:, 2] * 100,
        }
    )
    brain_mesh = sim.build_brain_mesh()
    return BrainData(sim.conn_matrices, chanlocs, brain_mesh, directed=sim.is_directed)


def _brain_data_from_uploaded_array(arr: np.ndarray) -> BrainData:
    """
    Build a BrainData object from a 3D connectivity array of shape (n_mat, n_elec, n_elec).
    """
    if arr.ndim != 3:
        raise ValueError("Uploaded data must be 3D: (n_mat, n_elec, n_elec)")

    n_mat, n_elec, _ = arr.shape
    angles = np.linspace(0, 2 * np.pi, n_elec, endpoint=False)
    x = np.cos(angles)
    y = np.sin(angles)
    z = np.zeros_like(x)

    chanlocs = pd.DataFrame(
        {
            "label": [f"E{i}" for i in range(n_elec)],
            "x": x * 100,
            "y": y * 100,
            "z": z,
        }
    )
    brain_mesh = None
    return BrainData(arr, chanlocs, brain_mesh, directed=False)


def register_data_callbacks(app: Dash, global_state: GlobalAppState, id_prefix: str = "data-comp"):
    modal_id = f"{id_prefix}-data-modal"
    btn_id = f"{id_prefix}-add-data-btn"
    close_id = f"{id_prefix}-data-modal-close"
    upload_id = f"{id_prefix}-upload"
    preset_id = f"{id_prefix}-preset-dropdown"
    gen_btn_id = f"{id_prefix}-gen-btn"
    gen_n_elec_id = f"{id_prefix}-gen-n-elec"
    gen_n_mat_id = f"{id_prefix}-gen-n-mat"
    gen_directed_id = f"{id_prefix}-gen-directed"
    label_id = f"{id_prefix}-dataset-label"
    store_id = f"{id_prefix}-data-store"
    slider_id = f"{id_prefix}-mat-idx"

    loader = DataLoader(global_state, preset_configs=PRESET_CONFIGS)

    @app.callback(
        Output(modal_id, "is_open"),
        Output(label_id, "children"),
        Output(store_id, "data"),
        Output(slider_id, "max"),
        Output(slider_id, "marks"),
        Output(slider_id, "value"),
        Input(btn_id, "n_clicks"),
        Input(close_id, "n_clicks"),
        Input(upload_id, "contents"),
        Input(preset_id, "value"),
        Input(gen_btn_id, "n_clicks"),
        State(upload_id, "filename"),
        State(modal_id, "is_open"),
        State(store_id, "data"),
        State(slider_id, "max"),
        State(slider_id, "marks"),
        State(slider_id, "value"),
        State(gen_n_elec_id, "value"),
        State(gen_n_mat_id, "value"),
        State(gen_directed_id, "value"),
        prevent_initial_call=False,
    )
    def data_modal_and_dataset(
        open_clicks,
        close_clicks,
        upload_contents,
        preset_value,
        gen_clicks,
        filename,
        is_open,
        store_data,
        slider_max,
        slider_marks,
        slider_value,
        gen_n_elec,
        gen_n_mat,
        gen_directed,
    ):
        ctx = callback_context

        # ---------- Initial load ----------
        if not ctx.triggered:
            label, store, slider = loader.initial_ui_state(
                store_data,
                slider_max,
                slider_marks,
                slider_value,
            )
            return (
                is_open,
                label,
                store,
                slider.max_idx,
                slider.marks,
                slider.value,
            )

        trigger = ctx.triggered[0]["prop_id"].split(".")[0]

        # ---------- Case 1: modal toggle (+ / Close) ----------
        if trigger in (btn_id, close_id):
            new_is_open = not (is_open or False)
            label = (store_data or {}).get("name") or "No dataset loaded"
            return (
                new_is_open,
                label,
                store_data,
                slider_max,
                slider_marks,
                slider_value,
            )

        # ---------- Case 2: option 1 – upload your own ----------
        if trigger == upload_id and upload_contents is not None:
            try:
                meta, slider = loader.load_uploaded(upload_contents, filename, store_data)
                new_store = {
                    "name": meta.name,
                    "source": meta.source,
                    **meta.extra,
                }
                return (
                    False,              # close modal
                    meta.name,
                    new_store,
                    slider.max_idx,
                    slider.marks,
                    slider.value,
                )
            except Exception as exc:
                label = f"Upload failed: {exc}"
                return (
                    False,
                    label,
                    store_data,
                    slider_max,
                    slider_marks,
                    slider_value,
                )

        # ---------- Case 3: option 2 – preset dataset ----------
        if trigger == preset_id and preset_value:
            try:
                meta, slider = loader.load_preset(preset_value)
                new_store = {
                    "name": meta.name,
                    "source": meta.source,
                    **meta.extra,
                }
                return (
                    False,
                    meta.name,
                    new_store,
                    slider.max_idx,
                    slider.marks,
                    slider.value,
                )
            except Exception as exc:
                label = f"Preset failed: {exc}"
                return (
                    False,
                    label,
                    store_data,
                    slider_max,
                    slider_marks,
                    slider_value,
                )

        # ---------- Case 4: option 3 – generate your own ----------
        if trigger == gen_btn_id:
            try:
                meta, slider = loader.load_simulated_custom(
                    n_elec=gen_n_elec,
                    n_mat=gen_n_mat,
                    directed=gen_directed,
                )
                new_store = {
                    "name": meta.name,
                    "source": meta.source,
                    **meta.extra,
                }
                return (
                    False,
                    meta.name,
                    new_store,
                    slider.max_idx,
                    slider.marks,
                    slider.value,
                )
            except Exception as exc:
                label = f"Simulation failed: {exc}"
                return (
                    False,
                    label,
                    store_data,
                    slider_max,
                    slider_marks,
                    slider_value,
                )

        # ---------- Fallback: no-op ----------
        label = (store_data or {}).get("name") or "No dataset loaded"
        return (
            is_open,
            label,
            store_data,
            slider_max,
            slider_marks,
            slider_value,
        )



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
    register_data_callbacks(app, global_state, id_prefix="data-comp")
