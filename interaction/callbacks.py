from dash import Input, Output, State, Dash, no_update, callback_context
import numpy as np
import plotly.graph_objects as go
import plotly.colors as plc
from data.loaders import DataLoader, PRESET_CONFIGS  # adjust import path if needed
import visualization.vizhelpers as helpers
from dash.exceptions import PreventUpdate

from visualization.vizuimanager import VizType, VizUIManager
from visualization.vizhelpers import UpdateType
from utils.global_app_state import GlobalAppState
from utils.update import update_attributes
from analysis.threshold import Threshold

# NEW:
from data.simulation import Simulation
from utils.braindata import BrainData
import pandas as pd

PRESET_CONFIGS = {
    "small_undirected": {"n_elec": 10, "directed": False, "n_mat": 5},
    "medium_directed": {"n_elec": 20, "directed": True, "n_mat": 10},
    "large_undirected": {"n_elec": 64, "directed": False, "n_mat": 20},
}


def determine_update_type_from_trigger(trigger_id: str) -> UpdateType:

    # Threshold changes
    if trigger_id in {
        "thresh-thresh_type-dropdown",
        "thresh-percent-slider",
        "thresh-stat-alpha-slider",
    }:
        return UpdateType.THRESHOLD

    # Node changes (2D & 3D)
    if trigger_id in {
        "viz-node-node_size-slider",
        #  "viz-node-node_opacity-slider",
    }:
        return UpdateType.NODES

    # Visibility / edge / color
    if trigger_id in {
        "viz-color_type-dropdown",
        "viz-color-range_slider",
        "viz-node-edge_width-range_slider",
        "viz-node-edge_opacity-slider",
        "viz-node-arc_radius-slider"
    }:
        return UpdateType.VISIBLE

    # Switching figures (2D <-> 3D)
    if trigger_id == "viz-fig_type-dropdown":
        return UpdateType.ALL

    # Frame change
    if trigger_id == "data-conn_idx-slider":
        return UpdateType.ALL

    # # Hemi toggles
    # if trigger_id in {
    #     "viz-3d-show_left_hem-checklist",
    #     "viz-3d-show_right_hem-checklist",
    # }:
    #     return UpdateType.VISIBLE

    # Default fallback
    return UpdateType.NONE

def register_visualization_callback(app: Dash, global_state: GlobalAppState):
    n_frames = int(global_state.brain_data.conn_mat.shape[0])
    @app.callback(
        Output("split-right-fig", "figure"),

        Input("data-conn_idx-slider", "value"),

        # threshold type dropdown (Basic / MST / Statistical Test)
        Input("thresh-thresh_type-dropdown", "value"),
        Input("thresh-stat-alpha-slider", "value"),
        Input("thresh-percent-slider", "value"),

        # numeric threshold slider (percent)
        Input("viz-fig_type-dropdown", "value"),
        Input("viz-color_type-dropdown", "value"),
        Input("viz-color-range_slider", "value"),

        # 2D visualization options
        Input("viz-node-node_size-slider", "value"),
        # Input("viz-node-node_opacity-slider", "value"),
        Input("viz-node-edge_width-range_slider", "value"),
        Input("viz-node-edge_opacity-slider", "value"),
        Input("viz-node-arc_radius-slider", "value"),

        Input("viz-3d-show_right_hem-checklist", "value"),
        Input("viz-3d-show_left_hem-checklist", "value"),
        Input("viz-3d-brain_mesh_opacity-slider", "value"),
        prevent_initial_call=False,
    )

    def update_visualization(conn_idx, 
                             
                            thresh_type, 
                            thresh_alpha,
                            thresh_percent, 

                            viz_fig_type,
                            color_type, 
                            color_range,
                               
                             node_size, 
                            # node_opacity, 
                             edge_width_range, 
                             edge_opacity, 
                             arc_radius,
                             show_hemi_right_3d, 
                             show_hemi_left_3d,
                             brain_mesh_opacity):

        """Update the main visualization figure.

        The signature must match the decorated Inputs exactly.
        """
        conn_idx = int(np.clip(conn_idx or 0, 0, n_frames - 1))
        viz_type = helpers.str_to_viz_type(viz_fig_type)

        # conn_range is expected to be a two-element sequence [min, max]
        try:
            color_min, color_max = float(color_range[0]), float(color_range[1])
        except Exception:
            color_min, color_max = 0.0, 1.0


        # -----------------------------
        # Build update flags
        # -----------------------------
        viz_updates = {
            "conn_idx": conn_idx,
            "colorscale": color_type,
            "color_min": color_min,
            "color_max": color_max,
            # "update_xyz": global_state.brain_data.chanlocs,
            "viz_type": viz_type,
            "node_size": node_size,
            # "node_opacity": node_opacity,
            "edge_width_range": edge_width_range,
            "edge_opacity": edge_opacity,
            "arc_radius": arc_radius,

            "show_hemi_left_3d": bool(show_hemi_left_3d and len(show_hemi_left_3d) > 0),
            "show_hemi_right_3d": bool(show_hemi_right_3d and len(show_hemi_right_3d) > 0),
            "brain_mesh_opacity_3d": brain_mesh_opacity,
        }

        threshold_updates = {
            "threshold_type": thresh_type,
            "threshold": thresh_percent,
            "alpha": thresh_alpha,
        }

        # -----------------------------
        # Determine update type
        # -----------------------------
        trigger = callback_context.triggered[0]["prop_id"].split(".")[0]
        update_type = determine_update_type_from_trigger(trigger)
        print("update_type:", update_type)

        # -----------------------------
        # Run the visualization update
        # -----------------------------
        # switch_fig = global_state.viz.viz_type != viz_updates['viz_type']
        # print(f"{switch_fig=}")
        update_attributes(global_state.threshold, **threshold_updates)
        global_state.viz.update_attributes(viz_updates=viz_updates)
        global_state.viz.update_figure(brain_data=global_state.brain_data, threshold=global_state.threshold, update_type=update_type)
        # if switch_fig and global_state.viz.viz_dict[ global_state.viz.viz_type] is None:
        #     global_state.viz.build_figure(brain_data=global_state.brain_data, threshold=global_state.threshold)
        # else:
        #     global_state.viz.update_figure(brain_data=global_state.brain_data, threshold=global_state.threshold, update_type=update_type)
        fig = global_state.viz.get_figure()
        fig.update_layout(uirevision="keep")
        return fig
    
    @app.callback(
        Output("stat-collapse-total_nodes-container", "children"),
        Output("stat-collapse-total_edges-container", "children"),
        Output("stat-collapse-visible_edges-container", "children"),

        # Trigger whenever visualization updates
        Input("split-right-fig", "figure"),
    )
    def update_stats(_):
        mask = global_state.viz._mask_cache
        # I need to fix the diagonals
        # print(mask)
        np.fill_diagonal(mask, False)

        n_nodes = global_state.brain_data.n_nodes
                
        # Total edges
        if global_state.brain_data.directed:
            total_edges = n_nodes * (n_nodes - 1)
        else:
            total_edges = n_nodes * (n_nodes - 1) // 2  # integer division for undirected

        # Mask for visible edges
        mask = global_state.viz._mask_cache.copy()
        np.fill_diagonal(mask, False)

        if global_state.brain_data.directed:
            visible_edges = int(mask.sum())
        else:
            visible_edges = int(np.triu(mask, k=1).sum())   

        return (
            n_nodes,
            total_edges,
            visible_edges,
        )


    
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

 
# def register_data_callbacks(app: Dash, global_state: GlobalAppState):
#     modal_id = "data-modal"
#     btn_id = "data-add_dataset-button"
#     close_id = "data-modal-close-button"
#     upload_id = "data-modal-upload"
#     preset_id = "data-modal-dataset_preset-dropdown"
#     gen_btn_id = "data-modal-gen-button"
#     gen_n_elec_id = "data-modal-gen_n_elec-input"
#     gen_n_mat_id = "data-modal-gen_n_mats-input"
#     gen_directed_id = "data-modal-gen_directed-checkbox"
#     label_id = "data-dataset-label"
#     store_id = "data-store"
#     slider_id = "data-conn_idx-slider"

#     loader = DataLoader(global_state, preset_configs=PRESET_CONFIGS)

#     @app.callback(
#         Output(modal_id, "is_open"),
#         Output(label_id, "children"),
#         Output(store_id, "data"),
#         Output(slider_id, "max"),
#         Output(slider_id, "marks"),
#         Output(slider_id, "value"),
#         Input(btn_id, "n_clicks"),
#         Input(close_id, "n_clicks"),
#         Input(upload_id, "contents"),
#         Input(preset_id, "value"),
#         Input(gen_btn_id, "n_clicks"),
#         State(upload_id, "filename"),
#         State(modal_id, "is_open"),
#         State(store_id, "data"),
#         State(slider_id, "max"),
#         State(slider_id, "marks"),
#         State(slider_id, "value"),
#         State(gen_n_elec_id, "value"),
#         State(gen_n_mat_id, "value"),
#         State(gen_directed_id, "value"),
#         prevent_initial_call=False,
#     )
#     def data_modal_and_dataset(
#         open_clicks,
#         close_clicks,
#         upload_contents,
#         preset_value,
#         gen_clicks,
#         filename,
#         is_open,
#         store_data,
#         slider_max,
#         slider_marks,
#         slider_value,
#         gen_n_elec,
#         gen_n_mat,
#         gen_directed,
#     ):
#         ctx = callback_context

#         # ---------- Initial load ----------
#         if not ctx.triggered:
#             label, store, slider = loader.initial_ui_state(
#                 store_data,
#                 slider_max,
#                 slider_marks,
#                 slider_value,
#             )
#             return (
#                 is_open,
#                 label,
#                 store,
#                 slider.max_idx,
#                 slider.marks,
#                 slider.value,
#             )

#         trigger = ctx.triggered[0]["prop_id"].split(".")[0]

#         # ---------- Case 1: modal toggle (+ / Close) ----------
#         if trigger in (btn_id, close_id):
#             new_is_open = not (is_open or False)
#             label = (store_data or {}).get("name") or "No dataset loaded"
#             return (
#                 new_is_open,
#                 label,
#                 store_data,
#                 slider_max,
#                 slider_marks,
#                 slider_value,
#             )

#         # ---------- Case 2: option 1 – upload your own ----------
#         if trigger == upload_id and upload_contents is not None:
#             try:
#                 meta, slider = loader.load_uploaded(upload_contents, filename, store_data)
#                 new_store = {
#                     "name": meta.name,
#                     "source": meta.source,
#                     **meta.extra,
#                 }
#                 return (
#                     False,              # close modal
#                     meta.name,
#                     new_store,
#                     slider.max_idx,
#                     slider.marks,
#                     slider.value,
#                 )
#             except Exception as exc:
#                 label = f"Upload failed: {exc}"
#                 return (
#                     False,
#                     label,
#                     store_data,
#                     slider_max,
#                     slider_marks,
#                     slider_value,
#                 )

#         # ---------- Case 3: option 2 – preset dataset ----------
#         if trigger == preset_id and preset_value:
#             try:
#                 meta, slider = loader.load_preset(preset_value)
#                 new_store = {
#                     "name": meta.name,
#                     "source": meta.source,
#                     **meta.extra,
#                 }
#                 return (
#                     False,
#                     meta.name,
#                     new_store,
#                     slider.max_idx,
#                     slider.marks,
#                     slider.value,
#                 )
#             except Exception as exc:
#                 label = f"Preset failed: {exc}"
#                 return (
#                     False,
#                     label,
#                     store_data,
#                     slider_max,
#                     slider_marks,
#                     slider_value,
#                 )

#         # ---------- Case 4: option 3 – generate your own ----------
#         if trigger == gen_btn_id:
#             try:
#                 meta, slider = loader.load_simulated_custom(
#                     n_elec=gen_n_elec,
#                     n_mat=gen_n_mat,
#                     directed=gen_directed,
#                 )
#                 new_store = {
#                     "name": meta.name,
#                     "source": meta.source,
#                     **meta.extra,
#                 }
#                 return (
#                     False,
#                     meta.name,
#                     new_store,
#                     slider.max_idx,
#                     slider.marks,
#                     slider.value,
#                 )
#             except Exception as exc:
#                 label = f"Simulation failed: {exc}"
#                 return (
#                     False,
#                     label,
#                     store_data,
#                     slider_max,
#                     slider_marks,
#                     slider_value,
#                 )

#         # ---------- Fallback: no-op ----------
#         label = (store_data or {}).get("name") or "No dataset loaded"
#         return (
#             is_open,
#             label,
#             store_data,
#             slider_max,
#             slider_marks,
#             slider_value,
#         )
def register_data_callbacks(app: Dash, global_state: GlobalAppState):
    modal_id = "data-modal"
    btn_id = "data-add_dataset-button"
    close_id = "data-modal-close-button"

    # Step 1 – FC data
    fc_upload_id = "data-fc-upload"
    fc_preset_id = "data-fc-preset-dropdown"
    fc_gen_btn_id = "data-fc-gen-btn"

    # Step 2 – location data
    loc_upload_id = "data-loc-upload"
    loc_preset_id = "data-loc-preset-dropdown"
    loc_gen_btn_id = "data-loc-gen-btn"

    # Step 3 – directed
    directed_id = "data-directed-checkbox"

    label_id = "data-dataset-label"
    store_id = "data-store"
    step_label_id = "data-step-indicator"
    slider_id = "data-conn_idx-slider"

    loader = DataLoader(global_state, preset_configs=PRESET_CONFIGS)

    @app.callback(
        Output(modal_id, "is_open"),
        Output(label_id, "children"),
        Output(store_id, "data"),
        Output(step_label_id, "children"),
        Output(slider_id, "max"),
        Output(slider_id, "marks"),
        Output(slider_id, "value"),
        Input(btn_id, "n_clicks"),
        Input(close_id, "n_clicks"),
        Input(fc_upload_id, "contents"),
        Input(fc_preset_id, "value"),
        Input(fc_gen_btn_id, "n_clicks"),
        Input(loc_upload_id, "contents"),
        Input(loc_preset_id, "value"),
        Input(loc_gen_btn_id, "n_clicks"),
        Input(directed_id, "value"),
        State(fc_upload_id, "filename"),
        State(loc_upload_id, "filename"),
        State(modal_id, "is_open"),
        State(store_id, "data"),
        State(slider_id, "max"),
        State(slider_id, "marks"),
        State(slider_id, "value"),
        prevent_initial_call=False,
    )
    def handle_data_modal(
        btn_click, close_click,
        fc_contents, fc_preset, fc_gen_click,
        loc_contents, loc_preset, loc_gen_click,
        directed_val,
        fc_filename, loc_filename,
        is_open, store_data,
        slider_max, slider_marks, slider_value,
    ):
        ctx = callback_context
        trigger = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else None

        # Normalize store_data
        if not store_data:
            store_data = {"step": 1, "fc": None, "loc": None, "directed": None}
        else:
            store_data.setdefault("step", 1)
            store_data.setdefault("fc", None)
            store_data.setdefault("loc", None)
            store_data.setdefault("directed", None)
        ctx = callback_context
        trigger = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else None

        # Normalize store_data
        if not store_data:
            store_data = {"step": 1, "fc": None, "loc": None, "directed": None}
        else:
            store_data.setdefault("step", 1)
            store_data.setdefault("fc", None)
            store_data.setdefault("loc", None)
            store_data.setdefault("directed", None)


        # ---------- Case: Open/Close modal ----------
        if trigger in (btn_id, close_id):
            new_is_open = not (is_open or False)
            step_label = f"Step {store_data['step']}: " + \
                ("Load FC data" if store_data["step"] == 1 else
                "Load location data" if store_data["step"] == 2 else
                "Directed / undirected")
            return new_is_open, store_data.get("fc", "No dataset loaded"), store_data, step_label, slider_max, slider_marks, slider_value

        # ---------- Step 1: Functional connectivity ----------
        if store_data["step"] == 1 and trigger in (fc_upload_id, fc_preset_id, fc_gen_btn_id):
            try:
                if trigger == fc_upload_id and fc_contents:
                    meta, slider = loader.load_uploaded(fc_contents, fc_filename, store_data)
                elif trigger == fc_preset_id and fc_preset:
                    meta, slider = loader.load_preset(fc_preset)
                elif trigger == fc_gen_btn_id:
                    meta, slider = loader.load_simulated_custom(
                        n_elec=20, n_mat=10, directed=False
                    )
                else:
                    raise PreventUpdate

                store_data["fc"] = {"name": meta.name, "source": meta.source, **meta.extra}
                store_data["step"] = 2  # advance to next step
                step_label = "Step 2: Load location data"
                return is_open, meta.name, store_data, step_label, slider.max_idx, slider.marks, slider.value
            except Exception as exc:
                return is_open, f"FC load failed: {exc}", store_data, f"Step 1: Load FC data", slider_max, slider_marks, slider_value

        # ---------- Step 2: Location data ----------
        if store_data["step"] == 2 and trigger in (loc_upload_id, loc_preset_id, loc_gen_btn_id):
            try:
                if trigger == loc_upload_id and loc_contents:
                    meta, _ = loader.load_location(loc_contents, loc_filename)
                elif trigger == loc_preset_id and loc_preset:
                    meta, _ = loader.load_location_preset(loc_preset)
                elif trigger == loc_gen_btn_id:
                    meta, _ = loader.load_location_simulated()
                else:
                    raise PreventUpdate

                store_data["loc"] = {"name": meta.name, "source": meta.source, **meta.extra}
                store_data["step"] = 3  # advance to final step
                step_label = "Step 3: Directed / undirected"
                return is_open, store_data["fc"]["name"], store_data, step_label, slider_max, slider_marks, slider_value
            except Exception as exc:
                return is_open, f"Location load failed: {exc}", store_data, "Step 2: Load location data", slider_max, slider_marks, slider_value

        # ---------- Step 3: Directed / undirected ----------
        if store_data["step"] == 3 and trigger == directed_id:
            store_data["directed"] = bool(directed_val)
            step_label = "Completed"
            return is_open, store_data["fc"]["name"], store_data, step_label, slider_max, slider_marks, slider_value

        # ---------- Fallback ----------
        step_label = f"Step {store_data['step']}: " + \
            ("Load FC data" if store_data["step"] == 1 else
            "Load location data" if store_data["step"] == 2 else
            "Directed / undirected")
        return is_open, store_data.get("fc", "No dataset loaded"), store_data, step_label, slider_max, slider_marks, slider_value



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


def register_threshold_callback(app: Dash):
    """Show/hide the threshold slider and stat-test container based on selection."""
    @app.callback(
        Output("thresh-slider_container", "style"),
        Output("thresh-stat_test_container", "style"),
        Input("thresh-thresh_type-dropdown", "value"),
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

def register_viz_control_callback(app: Dash):
    """Show/hide the threshold slider and stat-test container based on selection."""
    @app.callback(
        Output("viz-node-container", "style"),
        # Output("viz-2d-container", "style"),
        Output("viz-3d-container", "style"),
        Input("viz-fig_type-dropdown", "value"),
        prevent_initial_call=False,
    )
    def toggle_viz_container(fig_type):
        show = {"display": "block"}
        hide = {"display": "none"}
        viz_type = helpers.str_to_viz_type(s=fig_type)
        if viz_type == VizType.FIG2D:
            return show, hide
        elif viz_type == VizType.FIG3D:
            return show, show
        elif viz_type == VizType.FIGHEATMAP:
            return hide, hide
        else:
            return hide, hide

def register_stat_toggle_callback(app:Dash):
    @app.callback(
        Output("stats-collapse", "is_open"),
        Output("right-stats-container", "style"),
        Output("stat-toggle-btn", "children"),

        Input("stat-toggle-btn", "n_clicks"),
        State("stats-collapse", "is_open"),
    )
    def toggle_stats(n, is_open):
        if n is None:
            raise PreventUpdate

        new_open = not is_open

        if new_open:
            new_style = {
                "flex": "0 0 260px",  # EXPANDED WIDTH
                "overflow": "hidden",
                "transition": "flex-basis 0.3s",
                "borderLeft": "1px solid #ccc",
            }
            arrow = ">"  # arrow pointing right = collapse
        else:
            new_style = {
                "flex": "0 0 0px",
                "overflow": "hidden",
                "transition": "flex-basis 0.3s",
                "borderLeft": "1px solid #ccc",
            }
            arrow = "<"

        return new_open, new_style, arrow



def register_callbacks(app: Dash, global_state: GlobalAppState):
    """Attach all interaction callbacks to the Dash app."""
    register_visualization_callback(app, global_state)
    register_threshold_callback(app)
    register_data_callbacks(app, global_state)
    register_viz_control_callback(app)
    register_stat_toggle_callback(app)
