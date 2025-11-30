from dash import Input, Output, State, Dash, no_update, callback_context
import numpy as np
import plotly.graph_objects as go
import plotly.colors as plc
from data.loaders import DataLoader, PRESET_CONFIGS  # adjust import path if needed
import visualization.vizhelpers as helpers


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

# def determine_update_type(
#         viz_manager: VizUIManager,
#         threshold: Threshold,
#         updates: dict
#     ) -> UpdateType:

#     def strip_fig_type_tag(s: str) -> str:
#         return s[:-3] if s.endswith(("_2d", "_3d")) else s

#     def check_fields(fields, objs, update_type):
#         """
#         General field comparison function.

#         fields: list of field names (may include _2d/_3d)
#         objs: list of objects to compare against (viz_manager, fig, threshold)
#         update_type: UpdateType to return on mismatch
#         """
#         for field in fields:
#             if field not in updates:
#                 continue

#             base = strip_fig_type_tag(field)
#             # print(f"{base=}")
#             new_value = updates[field]

#             for obj in objs:
#                 # print(f"{obj}")
                
#                 # exact field first (viz_manager stores UI values this way)
#                 if hasattr(obj, field):
#                     old_value = getattr(obj, field)
#                 # fallback to base field (figure might store base)
#                 elif hasattr(obj, base):
#                     old_value = getattr(obj, base)
#                 else:
#                     continue  # skip missing attributes entirely

#                 if old_value != new_value:
#                     return update_type

#         return None


#     # ---------------------------------------------------------
#     # Objects used for comparison
#     # ---------------------------------------------------------
#     fig = viz_manager.viz_dict[viz_manager.viz_type]

#     # ---------------------------------------------------------
#     # 3. Threshold-related updates
#     # ---------------------------------------------------------
#     threshold_fields = ["threshold", "threshold_type", "alpha"]

#     result = check_fields(
#         fields=threshold_fields,
#         objs=[threshold],
#         update_type=UpdateType.THRESHOLD
#     )
#     if result:
#         return result


#     # ---------------------------------------------------------
#     # 1. Visualization-related updates (COLORSCALE + EDGE WIDTHS)
#     # ---------------------------------------------------------
#     color_fields = [
#         "colorscale",
#         "color_min",
#         "color_max",
#         "edge_width_range_2d",
#         "edge_width_range_3d",
#         "edge_opacity_2d",
#         "edge_opacity_3d"
#     ]

#     result = check_fields(
#         fields=color_fields,
#         objs=[viz_manager, fig],
#         update_type=UpdateType.VISIBLE
#     )
   
#     if result:
#         print("VISIBLE")
#         return result

#     # ---------------------------------------------------------
#     # 2. NODE updates (→ NODES update)
#     # ---------------------------------------------------------
#     node_fields = [
#         "node_size_2d",
#         "node_size_3d",
#     ]

#     result = check_fields(
#         fields=node_fields,
#         objs=[viz_manager, fig],
#         update_type=UpdateType.NODES
#     )
#     # print(result)
#     if result:
#         print("NODES")
#         return result

    
#     # ---------------------------------------------------------
#     # 4. Other (conn_idx → full ALL update)
#     # ---------------------------------------------------------
#     result = check_fields(
#         fields=["conn_idx"],
#         objs=[viz_manager],
#         update_type=UpdateType.ALL
#     )
#     if result:
#         print("ALL")
#         return result

#     # ---------------------------------------------------------
#     # 5. No update
#     # ---------------------------------------------------------
#     return UpdateType.NONE


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
        "viz-2d-node_size-slider",
        "viz-3d-node_size-slider",
    }:
        return UpdateType.NODES

    # Visibility / edge / color
    if trigger_id in {
        "viz-color_type-dropdown",
        "viz-color-range_slider",
        "viz-2d-edge_width-range_slider",
        "viz-2d-edge_opacity-slider",
        "viz-3d-edge_width-range_slider",
        "viz-3d-edge_opacity-slider",
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
        Input("viz-2d-node_size-slider", "value"),
        # Input("viz-2d-node_opacity-slider", "value"),
        Input("viz-2d-edge_width-range_slider", "value"),
        Input("viz-2d-edge_opacity-slider", "value"),

        # 3D visualization options
        Input("viz-3d-node_size-slider", "value"),
        # Input("viz-3d-node_opacity-slider", "value"),
        Input("viz-3d-edge_width-range_slider", "value"),
        Input("viz-3d-edge_opacity-slider", "value"),

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
                               
                             node_size_2d, 
                            #  node_opacity_2d, 
                             edge_width_range_2d, 
                             edge_opacity_2d, 
                             
                             node_size_3d, 
                            #  node_opacity_3d,
                             edge_width_range_3d, 
                             edge_opacity_3d,

                             show_hemi_right_3d, 
                             show_hemi_left_3d,
                             brain_mesh_opacity):

        """Update the main visualization figure.

        The signature must match the decorated Inputs exactly.
        """
        print(f"{node_size_2d=}")
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
            "node_size_2d": node_size_2d,
            # "node_opacity_2d": node_opacity_2d,
            "edge_width_range_2d": edge_width_range_2d,
            "edge_opacity_2d": edge_opacity_2d,

            "node_size_3d": node_size_3d,
            # "node_opacity_3d": node_opacity_3d,
            "edge_width_range_3d": edge_width_range_3d,
            "edge_opacity_3d": edge_opacity_3d,

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
    
# def register_visualization_callback(app: Dash, global_state: GlobalAppState):
#     """Register callback to update visualization based on matrix index, threshold, and viz type."""
#     n_frames = int(global_state.brain_data.conn_mat.shape[0])
#     # conn_mat = global_state.brain_data.conn_mat
#     # chanlocs = global_state.brain_data.chanlocs
#     # brain_mesh = global_state.brain_data.brain_mesh

#     @app.callback(
#         Output("main-visualization", "figure"),
#         Input("data-comp-mat-idx", "value"),
#         # threshold type dropdown (Basic / MST / Statistical Test)
#         Input("thresh-comp-type-dropdown", "value"),
#         # numeric threshold slider (percent)
#         Input("thresh-comp-slider", "value"),
#         Input("viz-type-dropdown", "value"),
#         Input("color-type-dropdown", "value"),
#         Input("conn-range", "value"),
#         # alpha slider inside the statistical-test subcomponent
#         Input("thresh-comp-alpha-slider", "value"),
#         prevent_initial_call=False,
#     )
#     def update_visualization(idx, thresh_type, thresh_value, viz_type, color_name, conn_range, alpha):
#         """Update the main visualization figure.

#         The signature must match the decorated Inputs exactly.
#         """
#         idx = int(np.clip(idx or 0, 0, n_frames - 1))
#         viz_type = (viz_type or "2D")

#         # conn_range is expected to be a two-element sequence [min, max]
#         try:
#             color_min, color_max = float(conn_range[0]), float(conn_range[1])
#         except Exception:
#             color_min, color_max = 0.0, 1.0

#         # viz = ConnectivityVisualizer(conn_mat[idx], chanlocs, brain_mesh=brain_mesh)
#         threshold_updates = {
#             "threshold_type": thresh_type,
#             "threshold": thresh_value,
#             "alpha": alpha,
#         }

#         viz_updates = {
#             "conn_idx": idx,
#             "colorscale": color_name,
#             "color_min": color_min,
#             "color_max": color_max,
#             # "update_xyz": global_state.brain_data.chanlocs,
#             "viz_type": viz_type
#         }

#         def determine_update_type(
#             viz,
#             threshold: Threshold,
#             updates: dict
#         ) -> UpdateType:
#             """
#             Determine what type of update is required given:
#             - viz: ConnectivityVisualizer instance
#             - threshold: Threshold instance
#             - updates: dict with new UI parameters
#             """

#             # ---------------------------------------------------------
#             # 1. Check visualization-related changes → FULL UPDATE
#             # ---------------------------------------------------------
#             color_fields = ["colorscale", "color_min", "color_max"]
#             for field in color_fields:
#                 if getattr(viz, field) != updates[field]:
#                     return UpdateType.COLOR

#             # ---------------------------------------------------------
#             # 2. Check threshold-related changes → THRESHOLD update
#             # ---------------------------------------------------------
#             # threshold fields to compare
#             threshold_fields = ["threshold", "threshold_type", "alpha"]

#             for field in threshold_fields:
#                 if getattr(threshold, field) != updates[field]:
#                     return UpdateType.THRESHOLD

#             # Also include conn_idx in threshold update logic:
#             if viz.conn_idx != updates["conn_idx"]:
#                 return UpdateType.THRESHOLD

#             # ---------------------------------------------------------
#             # 3. No update needed
#             # ---------------------------------------------------------
#             return UpdateType.NONE


#         update_type = determine_update_type(
#             global_state.viz,
#             global_state.threshold,
#             threshold_updates | viz_updates
#         )
#         print(update_type)
#         update_attributes(global_state.threshold, **threshold_updates)
#         global_state.viz.update(brain_data=global_state.brain_data, threshold=global_state.threshold, update_type=update_type, viz_updates=viz_updates)
#         fig = global_state.viz.get_figure()
#         fig.update_layout(uirevision="keep")
#         return fig
    
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


def register_data_callbacks(app: Dash, global_state: GlobalAppState):
    modal_id = "data-modal"
    btn_id = "data-add_dataset-button"
    close_id = "data-modal-close-button"
    upload_id = "data-modal-upload"
    preset_id = "data-modal-dataset_preset-dropdown"
    gen_btn_id = "data-modal-gen-button"
    gen_n_elec_id = "data-modal-gen_n_elec-input"
    gen_n_mat_id = "data-modal-gen_n_mats-input"
    gen_directed_id = "data-modal-gen_directed-checkbox"
    label_id = "data-dataset-label"
    store_id = "data-store"
    slider_id = "data-conn_idx-slider"

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
        Output("viz-2d-container", "style"),
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
            return hide, show
        else:
            return hide, hide

def register_callbacks(app: Dash, global_state: GlobalAppState):
    """Attach all interaction callbacks to the Dash app."""
    register_visualization_callback(app, global_state)
    register_threshold_callback(app)
    register_data_callbacks(app, global_state)
    register_viz_control_callback(app)
