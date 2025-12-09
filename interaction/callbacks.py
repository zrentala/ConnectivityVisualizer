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

def register_data_callbacks(app: Dash, global_state: GlobalAppState):
    modal_id = "data-modal"
    btn_id = "data-add_dataset-button"
    close_id = "data-modal-close-button"
    next_id = "data-next-button"
    back_id = "data-back-button"

    # IDs for file / dropdown inputs
    fc_upload_id = "data-fc-upload"
    fc_preset_id = "data-fc-preset-dropdown"
    loc_upload_id = "data-loc-upload"
    loc_preset_id = "data-loc-preset-dropdown"

    # Radio groups (dbc.RadioItems)
    fc_radio_id = "data-fc-radio"
    loc_radio_id = "data-loc-radio"

    fc_sim_nelec_id = "data-fc-sim-nelec"
    fc_sim_nmat_id = "data-fc-sim-nmat"

    directed_id = "data-directed-checkbox"

    label_id = "data-dataset-label"
    store_id = "data-store"
    step_label_id = "data-step-indicator"
    slider_id = "data-conn_idx-slider"
    error_id = "data-error-message"
    fc_summary_id = "data-fc-summary"

    step1_view_id = "data-step1-view"
    step2_view_id = "data-step2-view"

    # Card IDs for highlighting – must match create_data_component
    fc_upload_card_id = "data-fc-radio-upload-card"
    fc_preset_card_id = "data-fc-radio-preset-card"
    fc_sim_card_id = "data-fc-radio-sim-card"
    loc_upload_card_id = "data-loc-radio-upload-card"
    loc_preset_card_id = "data-loc-radio-preset-card"
    loc_sim_card_id = "data-loc-radio-sim-card"

    loader = DataLoader()

    def overall_step_text(sd: dict) -> str:
        if sd.get("fc") and sd.get("loc"):
            return "Data loaded"
        if sd.get("fc"):
            return "Step 2: Load location data"
        return "Step 1: Load FC data"

    def fc_summary_text(sd: dict) -> str:
        fc_meta = sd.get("fc") or {}
        if not fc_meta:
            return "No FC data loaded yet."
        n_elec = fc_meta.get("n_elec") or fc_meta.get("n_elecs")
        n_mat = fc_meta.get("n_mat") or fc_meta.get("n_mats")
        directed = fc_meta.get("directed", False)
        d_str = "directed" if directed else "undirected"
        return f"FC: n_elec={n_elec}, n_mats={n_mat}, {d_str}"

    def current_label(sd: dict) -> str:
        return sd.get("fc", {}).get("name", "No dataset loaded")

    # ---------------------------
    # Main state / loading logic
    # ---------------------------
    @app.callback(
        Output(modal_id, "is_open"),
        Output(label_id, "children"),
        Output(store_id, "data"),
        Output(step_label_id, "children"),
        Output(slider_id, "max"),
        Output(slider_id, "marks"),
        Output(slider_id, "value"),
        Output(error_id, "children"),
        Output(fc_summary_id, "children"),
        Output(step1_view_id, "style"),
        Output(step2_view_id, "style"),
        # Card className outputs (blue outline highlight)
        Output(fc_upload_card_id, "className"),
        Output(fc_preset_card_id, "className"),
        Output(fc_sim_card_id, "className"),
        Output(loc_upload_card_id, "className"),
        Output(loc_preset_card_id, "className"),
        Output(loc_sim_card_id, "className"),
        Input(btn_id, "n_clicks"),
        Input(close_id, "n_clicks"),
        Input(next_id, "n_clicks"),
        Input(back_id, "n_clicks"),
        Input(fc_upload_id, "contents"),
        Input(fc_preset_id, "value"),
        Input(loc_upload_id, "contents"),
        Input(loc_preset_id, "value"),
        Input(fc_radio_id, "value"),
        Input(loc_radio_id, "value"),
        Input(directed_id, "value"),
        State(fc_sim_nelec_id, "value"),
        State(fc_sim_nmat_id, "value"),
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
        btn_click,
        close_click,
        next_click,
        back_click,
        fc_contents,
        fc_preset_val,
        loc_contents,
        loc_preset_val,
        fc_radio_value,
        loc_radio_value,
        directed_val,
        fc_sim_nelec,
        fc_sim_nmat,
        fc_filename,
        loc_filename,
        is_open,
        store_data,
        slider_max,
        slider_marks,
        slider_value,
    ):
        # Normalize store
        if not isinstance(store_data, dict):
            store_data = {}
        store_data.setdefault("fc", {})
        store_data.setdefault("loc", {})
        store_data.setdefault("directed", False)
        store_data.setdefault("step", 1)

        error_msg = ""
        ctx = callback_context
        trigger = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else None

        # Current source selections
        fc_source = fc_radio_value or "upload"
        loc_source = loc_radio_value or "upload"

        # Helper for card classes (blue outline)
        base = "p-3 mb-2 border rounded bg-light"
        selected = base + " border-primary shadow-sm"

        def card_classes():
            fc_upload_cls = selected if fc_source == "upload" else base
            fc_preset_cls = selected if fc_source == "preset" else base
            fc_sim_cls = selected if fc_source == "simulate" else base

            loc_upload_cls = selected if loc_source == "upload" else base
            loc_preset_cls = selected if loc_source == "preset" else base
            loc_sim_cls = selected if loc_source == "simulate" else base

            return (
                fc_upload_cls,
                fc_preset_cls,
                fc_sim_cls,
                loc_upload_cls,
                loc_preset_cls,
                loc_sim_cls,
            )

        # ------------------------
        # Open / close modal only
        # ------------------------
        if trigger in (btn_id, close_id):
            if trigger == btn_id:
                is_open = not (is_open or False)
            else:
                is_open = False

            step = store_data.get("step", 1)
            step1_style = {"display": "block"} if step == 1 else {"display": "none"}
            step2_style = {"display": "block"} if step == 2 else {"display": "none"}

            (
                fc_upload_cls,
                fc_preset_cls,
                fc_sim_cls,
                loc_upload_cls,
                loc_preset_cls,
                loc_sim_cls,
            ) = card_classes()

            return (
                is_open,
                current_label(store_data),
                store_data,
                overall_step_text(store_data),
                slider_max,
                slider_marks,
                slider_value,
                error_msg,
                fc_summary_text(store_data),
                step1_style,
                step2_style,
                fc_upload_cls,
                fc_preset_cls,
                fc_sim_cls,
                loc_upload_cls,
                loc_preset_cls,
                loc_sim_cls,
            )

        # ----------------------
        # Immediate loads
        # ----------------------
        # STEP 1: upload/preset FC load immediately
        if trigger in (fc_upload_id, fc_preset_id):
            try:
                if trigger == fc_upload_id and fc_source == "upload" and fc_contents:
                    meta, slider = loader.load_uploaded(fc_contents, fc_filename, store_data)
                elif trigger == fc_preset_id and fc_source == "preset" and fc_preset_val:
                    meta, slider = loader.load_preset(fc_preset_val)
                else:
                    raise PreventUpdate

                store_data["fc"] = {"name": meta.name, "source": meta.source, **meta.extra}
                store_data["directed"] = store_data["fc"].get("directed", bool(directed_val))
                slider_max = slider.max_idx
                slider_marks = slider.marks
                slider_value = slider.value
                error_msg = ""

            except PreventUpdate:
                pass
            except Exception as exc:
                error_msg = f"FC load failed: {exc}"

        # STEP 2: upload/preset locations load immediately
        if trigger in (loc_upload_id, loc_preset_id):
            try:
                if not store_data.get("fc"):
                    raise RuntimeError("Load FC data in Step 1 before loading locations.")

                if trigger == loc_upload_id and loc_source == "upload" and loc_contents:
                    meta, _ = loader.load_location(loc_contents, loc_filename)
                elif trigger == loc_preset_id and loc_source == "preset" and loc_preset_val:
                    meta, _ = loader.load_location_preset(loc_preset_val)
                else:
                    raise PreventUpdate

                # check n_elec consistency
                fc_meta = store_data["fc"]
                fc_nelec = fc_meta.get("n_elec") or fc_meta.get("n_elecs")
                loc_nelec = meta.extra.get("n_elec") or meta.extra.get("n_elecs")
                if fc_nelec is not None and loc_nelec is not None and fc_nelec != loc_nelec:
                    raise ValueError(
                        f"Location data has n_elec={loc_nelec}, but FC data has n_elec={fc_nelec}."
                    )

                store_data["loc"] = {"name": meta.name, "source": meta.source, **meta.extra}
                error_msg = ""

            except PreventUpdate:
                pass
            except Exception as exc:
                error_msg = f"Location load failed: {exc}"

        # -------------
        # Navigation + simulate on Next
        # -------------
        if trigger == next_id:
            if store_data["step"] == 1:
                # If simulate selected, generate FC here
                if fc_source == "simulate":
                    try:
                        n_elec = int(fc_sim_nelec or 20)
                        n_mat = int(fc_sim_nmat or 10)
                        directed = bool(directed_val)
                        meta, slider = loader.load_simulated_custom(
                            n_elec=n_elec,
                            n_mat=n_mat,
                            directed=directed,
                        )
                        store_data["fc"] = {"name": meta.name, "source": meta.source, **meta.extra}
                        store_data["directed"] = directed
                        slider_max = slider.max_idx
                        slider_marks = slider.marks
                        slider_value = slider.value
                        error_msg = ""
                    except Exception as exc:
                        error_msg = f"Simulation failed: {exc}"

                # Require FC loaded
                if not store_data.get("fc"):
                    error_msg = error_msg or "Please load FC data before continuing."
                else:
                    store_data["step"] = 2

            elif store_data["step"] == 2:
                # If simulate selected, generate locations here
                if loc_source == "simulate":
                    try:
                        if not store_data.get("fc"):
                            raise RuntimeError("Load FC data first.")
                        meta, _ = loader.load_location_simulated()
                        fc_meta = store_data["fc"]
                        fc_nelec = fc_meta.get("n_elec") or fc_meta.get("n_elecs")
                        loc_nelec = meta.extra.get("n_elec") or meta.extra.get("n_elecs")
                        if fc_nelec is not None and loc_nelec is not None and fc_nelec != loc_nelec:
                            raise ValueError(
                                f"Simulated locations have n_elec={loc_nelec}, but FC data has n_elec={fc_nelec}."
                            )
                        store_data["loc"] = {"name": meta.name, "source": meta.source, **meta.extra}
                        error_msg = ""
                    except Exception as exc:
                        error_msg = f"Location simulation failed: {exc}"

                # Require locations loaded
                if not store_data.get("loc"):
                    error_msg = error_msg or "Please load location data before finishing."
                else:
                    is_open = False  # everything done

        elif trigger == back_id:
            if store_data["step"] == 2:
                store_data["step"] = 1
                error_msg = ""

        # Which step view is visible?
        step = store_data.get("step", 1)
        step1_style = {"display": "block"} if step == 1 else {"display": "none"}
        step2_style = {"display": "block"} if step == 2 else {"display": "none"}

        (
            fc_upload_cls,
            fc_preset_cls,
            fc_sim_cls,
            loc_upload_cls,
            loc_preset_cls,
            loc_sim_cls,
        ) = card_classes()

        return (
            is_open,
            current_label(store_data),
            store_data,
            overall_step_text(store_data),
            slider_max,
            slider_marks,
            slider_value,
            error_msg,
            fc_summary_text(store_data),
            step1_style,
            step2_style,
            fc_upload_cls,
            fc_preset_cls,
            fc_sim_cls,
            loc_upload_cls,
            loc_preset_cls,
            loc_sim_cls,
        )




# def register_data_callbacks(app: Dash, global_state: GlobalAppState):
#     modal_id = "data-modal"
#     btn_id = "data-add_dataset-button"
#     close_id = "data-modal-close-button"

#     # Step 1 – FC data
#     fc_upload_id = "data-fc-upload"
#     fc_preset_id = "data-fc-preset-dropdown"
#     fc_gen_btn_id = "data-fc-gen-btn"

#     # Step 2 – location data
#     loc_upload_id = "data-loc-upload"
#     loc_preset_id = "data-loc-preset-dropdown"
#     loc_gen_btn_id = "data-loc-gen-btn"

#     # Step 3 – directed flag
#     directed_id = "data-directed-checkbox"

#     label_id = "data-dataset-label"
#     store_id = "data-store"
#     step_label_id = "data-step-indicator"
#     slider_id = "data-conn_idx-slider"

#     loader = DataLoader()

#     def step_text(step: int) -> str:
#         return (
#             "Step 1: Load FC data" if step == 1 else
#             "Step 2: Load location data" if step == 2 else
#             "Step 3: Directed / undirected" if step == 3 else
#             "Completed"
#         )

#     @app.callback(
#         Output(modal_id, "is_open"),
#         Output(label_id, "children"),
#         Output(store_id, "data"),
#         Output(step_label_id, "children"),
#         Output(slider_id, "max"),
#         Output(slider_id, "marks"),
#         Output(slider_id, "value"),
#         Input(btn_id, "n_clicks"),
#         Input(close_id, "n_clicks"),
#         Input(fc_upload_id, "contents"),
#         Input(fc_preset_id, "value"),
#         Input(fc_gen_btn_id, "n_clicks"),
#         Input(loc_upload_id, "contents"),
#         Input(loc_preset_id, "value"),
#         Input(loc_gen_btn_id, "n_clicks"),
#         Input(directed_id, "value"),
#         State(fc_upload_id, "filename"),
#         State(loc_upload_id, "filename"),
#         State(modal_id, "is_open"),
#         State(store_id, "data"),
#         State(slider_id, "max"),
#         State(slider_id, "marks"),
#         State(slider_id, "value"),
#         prevent_initial_call=False,
#     )
#     def handle_data_modal(*args):
#         (
#             btn_click, close_click,
#             fc_contents, fc_preset, fc_gen_click,
#             loc_contents, loc_preset, loc_gen_click,
#             directed_val,
#             fc_filename, loc_filename,
#             is_open, store_data,
#             slider_max, slider_marks, slider_value
#         ) = args

#         # --- Normalize stored modal state ---
#         if not isinstance(store_data, dict):
#             store_data = {}
#         # store_data = store_data or {}
#         store_data.setdefault("fc", {})
#         store_data.setdefault("loc", {})
#         # store_data.setdefault("brain", {})
#         store_data.setdefault("directed", False)
#         store_data.setdefault("step", 1)
#         print(store_data)

#         ctx = callback_context
#         trigger = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else None

        

#         # ------------------------------------------------------------
#         # A. Modal open / close
#         # ------------------------------------------------------------
#         if trigger in (btn_id, close_id):
#             return (
#                 not (is_open or False),
#                 store_data.get("fc", {}).get("name", "No dataset loaded"),
#                 store_data,
#                 step_text(store_data["step"]),
#                 slider_max, slider_marks, slider_value
#             )

#         # ------------------------------------------------------------
#         # B. STEP 1: Load FC data
#         # ------------------------------------------------------------
#         if store_data["step"] == 1 and trigger in (fc_upload_id, fc_preset_id, fc_gen_btn_id):
#             try:
#                 # Uploaded FC file
#                 if trigger == fc_upload_id and fc_contents:
#                     meta, slider = loader.load_uploaded(fc_contents, fc_filename, store_data)

#                 # Preset FC
#                 elif trigger == fc_preset_id and fc_preset:
#                     meta, slider = loader.load_preset(fc_preset)

#                 # Generate FC data
#                 elif trigger == fc_gen_btn_id:
#                     meta, slider = loader.load_simulated_custom(n_elec=20, n_mat=10, directed=False)

#                 else:
#                     raise PreventUpdate

#                 store_data["fc"] = {"name": meta.name, "source": meta.source, **meta.extra}
#                 store_data["step"] = 2

#                 return (
#                     is_open,
#                     meta.name,
#                     store_data,
#                     step_text(2),
#                     slider.max_idx, slider.marks, slider.value
#                 )

#             except Exception as exc:
#                 return (
#                     is_open,
#                     f"FC load failed: {exc}",
#                     store_data,
#                     step_text(1),
#                     slider_max, slider_marks, slider_value
#                 )

#         # ------------------------------------------------------------
#         # C. STEP 2: Load location data
#         # ------------------------------------------------------------
#         if store_data["step"] == 2 and trigger in (loc_upload_id, loc_preset_id, loc_gen_btn_id):
#             try:
#                 # Location uploaded
#                 if trigger == loc_upload_id and loc_contents:
#                     meta, _ = loader.load_location(loc_contents, loc_filename)

#                 # Preset location
#                 elif trigger == loc_preset_id and loc_preset:
#                     meta, _ = loader.load_location_preset(loc_preset)

#                 # Generated location
#                 elif trigger == loc_gen_btn_id:
#                     meta, _ = loader.load_location_simulated()

#                 else:
#                     raise PreventUpdate

#                 store_data["loc"] = {"name": meta.name, "source": meta.source, **meta.extra}
#                 store_data["step"] = 3

#                 return (
#                     is_open,
#                     store_data["fc"]["name"],
#                     store_data,
#                     step_text(3),
#                     slider_max, slider_marks, slider_value
#                 )

#             except Exception as exc:
#                 return (
#                     is_open,
#                     f"Location load failed: {exc}",
#                     store_data,
#                     step_text(2),
#                     slider_max, slider_marks, slider_value
#                 )

#         # ------------------------------------------------------------
#         # D. STEP 3: Directed / Undirected
#         # ------------------------------------------------------------
#         if store_data["step"] == 3 and trigger == directed_id:
#             store_data["directed"] = bool(directed_val)

#             return (
#                 is_open,
#                 store_data["fc"]["name"],
#                 store_data,
#                 step_text(4),
#                 slider_max, slider_marks, slider_value
#             )

#         # ------------------------------------------------------------
#         # Fallback
#         # ------------------------------------------------------------
#         return (
#             is_open,
#             store_data.get("fc", {}).get("name", "No dataset loaded"),
#             store_data,
#             step_text(store_data["step"]),
#             slider_max, slider_marks, slider_value
#         )



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
