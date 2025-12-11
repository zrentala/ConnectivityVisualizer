from dash import Input, Output, State, Dash, no_update, callback_context
import numpy as np
import plotly.graph_objects as go
import plotly.colors as plc
from data.loaders import DataLoader, PRESET_CONFIGS, PRESET_LOCS, PRESET_LOCS_REVERSED  # adjust import path if needed
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

    # if not trigger_id or trigger_id == "data-add_dataset-button":
    #     return UpdateType.NONE
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
        if global_state.brain_data is None or global_state.viz is None:
            return go.Figure()  # empty placeholder

        n_frames = global_state.brain_data.conn_mat.shape[0]

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
        if global_state.viz is None or global_state.brain_data is None:
            return 0,0,0
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

def update_loc_options(sd):
    print("Updating loc options with sd:", sd)
    # No FC loaded yet → no options
    if sd is None or "n_elec" not in sd:
        return [{"label": "No available presets", "value": "none", "disabled": True}]

    n = sd["n_elec"]
    presets = PRESET_LOCS_REVERSED.get(n)

    # If nothing matches → disabled placeholder
    if not presets:
        return [{"label": "No available presets", "value": "none", "disabled": True}]

    # Normal case → real selectable options
    return [{"label": name, "value": name} for name in presets]

def update_fc_options():
    presets = PRESET_CONFIGS

    # Nothing matches → disabled placeholder
    if not presets:
        return [{"label": "No available presets", "value": "none", "disabled": True}]

    # Normal case → real options
    return [
        {
            "label": key,
            "value": f"{key} (n={value['n_elec']}, mats={value['n_mat']})"
        }
        for key, value in presets.items()
    ]


def register_data_callbacks(app: Dash, global_state: GlobalAppState):
    loader = DataLoader()

    modal_id = "data-modal"
    btn_id = "data-add_dataset-button"
    next_id = "data-next-button"
    back_id = "data-back-button"

    fc_upload_id = "data-fc-upload"
    fc_preset_id = "data-fc-preset-dropdown"
    loc_upload_id = "data-loc-upload"
    loc_preset_id = "data-loc-preset-dropdown"

    fc_radio_id = "data-fc-radio"
    loc_radio_id = "data-loc-radio"

    fc_sim_nelec_id = "data-fc-sim-nelec"
    fc_sim_nmat_id = "data-fc-sim-nmat"

    directed_id = "data-directed-checkbox"

    label_id = "data-dataset-label"
    store_id = "data-store"
    # step_label_id = "data-step-indicator"
    slider_id = "data-conn_idx-slider"
    error_id = "data-error-message"
    fc_summary_id = "data-fc-summary"

    step1_view_id = "data-step1-view"
    step2_view_id = "data-step2-view"

    # Card highlight IDs
    fc_upload_card_id = "data-fc-radio-upload-card"
    fc_preset_card_id = "data-fc-radio-preset-card"
    fc_sim_card_id = "data-fc-radio-sim-card"
    loc_upload_card_id = "data-loc-radio-upload-card"
    loc_preset_card_id = "data-loc-radio-preset-card"
    loc_sim_card_id = "data-loc-radio-sim-card"

    # ---------------------------------------------------
    # Small helpers
    # ---------------------------------------------------
    def current_label(sd: dict) -> str:
        print(sd)
        return (sd.get("fc_meta") or {}).get("name", "No dataset loaded")

    def fc_summary_text(sd):
        fc = sd.get("fc_cfg")
        if not fc:
            return "No FC data loaded yet."
        d = "directed" if fc.get("directed") else "undirected"
        return f"FC: # electrodes = {fc.get('n_elec')}, # FC matrices={fc.get('n_mat')}, {d}"

    def overall_step_text(sd):
        step = sd.get("step", 1)
        return "Step 1: Load FC data" if step == 1 else "Step 2: Load location data"

    def card_classes(fc_src, loc_src):
        base = "p-3 mb-2 border rounded bg-light"
        sel = base + " border-primary shadow-sm"
        return (
            sel if fc_src == "upload" else base,
            sel if fc_src == "preset" else base,
            sel if fc_src == "simulate" else base,
            sel if loc_src == "upload" else base,
            sel if loc_src == "preset" else base,
            sel if loc_src == "simulate" else base,
        )

    # ---------------------------------------------------
    # MAIN CALLBACK
    # ---------------------------------------------------
    @app.callback(
        (
            Output(modal_id, "is_open"),
            Output(label_id, "children"),
            Output(store_id, "data"),
            # Output(step_label_id, "children"),
            Output(slider_id, "max"),
            Output(slider_id, "marks"),
            Output(slider_id, "value"),
            Output(error_id, "children"),
            Output(fc_summary_id, "children"),
            Output(step1_view_id, "style"),
            Output(step2_view_id, "style"),
            Output(next_id, "children"),
            Output(back_id, "style"),
            
            # Card classes
            Output(fc_upload_card_id, "className"),
            Output(fc_preset_card_id, "className"),
            Output(fc_sim_card_id, "className"),
            Output(loc_upload_card_id, "className"),
            Output(loc_preset_card_id, "className"),
            Output(loc_sim_card_id, "className"),
            Output(fc_preset_id, "options"),
            Output(loc_preset_id, "options"),
        ),
        (
            Input(btn_id, "n_clicks"),
            Input(next_id, "n_clicks"),
            Input(back_id, "n_clicks"),
            Input(fc_upload_id, "contents"),
            Input(fc_preset_id, "value"),
            Input(loc_upload_id, "contents"),
            Input(loc_preset_id, "value"),
            Input(fc_radio_id, "value"),
            Input(loc_radio_id, "value"),
            Input(directed_id, "value"),
        ),
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
    def handle_modal(
        btn_click,
        next_click,
        back_click,
        fc_contents,
        fc_preset_val,
        loc_contents,
        loc_preset_val,
        fc_radio_val,
        loc_radio_val,
        directed_val,
        fc_sim_nelec,
        fc_sim_nmat,
        fc_filename,
        loc_filename,
        is_open,
        store_data,
        slider_max,
        slider_marks,
        slider_val,
    ):
        # -------------------------
        # Normalize store
        # -------------------------
        if not isinstance(store_data, dict):
            store_data = {}
        store_data.setdefault("step", 1)
        store_data.setdefault("fc_cfg", {})
        store_data.setdefault("loc_cfg",{})
        store_data.setdefault("fc_meta", {})
        store_data.setdefault("loc_meta", {})

        error_msg = ""

        ctx = callback_context
        trig = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else None
        print(f"Triggered by: {trig}")

        fc_src = fc_radio_val or "upload"
        loc_src = loc_radio_val or "upload"

        loc_options = update_loc_options(store_data)
        fc_options = update_fc_options()

        # ---------------------------------------------------
        # Open modal
        # ---------------------------------------------------
        if trig == btn_id:
            is_open = not is_open
            store_data = {"step": 1, "fc_cfg": {}, "loc_cfg": {}, "fc_meta": {}, "loc_meta": {}}
            step = store_data["step"]

            styles = (
                {"display": "block"} if step == 1 else {"display": "none"}, # slide 1
                {"display": "block"} if step == 2 else {"display": "none"}, # slide 2
            )
            next_text = "Submit" if step == 2 else "Next"
            back_style = {"display": "block"} if step == 2 else {"display": "none"}
            print("Opening modal")
            return (
                is_open,
                current_label(store_data),
                store_data,
                # overall_step_text(store_data),
                slider_max,
                slider_marks,
                slider_val,
                error_msg,
                fc_summary_text(store_data),
                styles[0],
                styles[1],
                next_text,      # ← MISSING (Output 12)
                back_style,     # ← MISSING (Output 13)
                *card_classes(fc_src, loc_src),
                fc_options,
                loc_options,
            )


        # ---------------------------------------------------
        # Step navigation
        # ---------------------------------------------------
        if trig == back_id and store_data["step"] == 2:
            print("Going back to step 1")
            store_data["step"] = 1

        # ---------------------------------------------------
        # NEXT → Step logic
        # ---------------------------------------------------
        if trig == next_id:

            # -------------------------
            # STEP 1 → Save FC config
            # -------------------------
            if store_data["step"] == 1:
                print("Processing FC step")
                fc_cfg = loader.make_fc_cfg(
                    source=fc_src,
                    upload=(fc_contents, fc_filename),
                    preset=fc_preset_val,
                    simulate=(fc_sim_nelec, fc_sim_nmat),
                )
                # print(fc_cfg)
                if fc_cfg["type"] == "sim":
                    fc_cfg["directed"] = bool(directed_val)

                store_data["fc_cfg"] = fc_cfg
                store_data["directed"] = bool(directed_val)
                
                store_data["step"] = 2
                # print(store_data)

            # -------------------------
            # STEP 2 → Save LOC config & BUILD DATASET
            # -------------------------
            elif store_data["step"] == 2:
                print("Processing LOC step")
                if store_data["fc_cfg"]["type"] == "sim":
                    print(store_data["fc_cfg"])
                    n_elec = store_data["fc_cfg"]["n_elec"]
                else:
                    n_elec = None

                loc_cfg = loader.make_loc_cfg(
                    source=loc_src,
                    upload=(loc_contents, loc_filename),
                    preset=loc_preset_val,
                    n_elec=n_elec,
                )

                # print(loc_cfg)
                store_data["loc_cfg"] = loc_cfg

                try:
                    # Build the final dataset
                    bd, meta, slider = loader.build_braindata(
                        store_data["fc_cfg"],
                        store_data["loc_cfg"],
                        store_data["directed"],
                    )

                    # Save into global_state
                    global_state.brain_data = bd
                    global_state.threshold = Threshold()
                    ## FIX
                    global_state.viz = VizUIManager(bd, global_state.threshold)

                    # Update store with metadata
                    store_data["fc_meta"] = meta["fc"].__dict__
                    store_data["loc_meta"] = meta["loc"].__dict__

                    slider_max = slider.max_idx
                    slider_marks = slider.marks
                    slider_val = slider.value

                    is_open = False

                except Exception as exc:
                    error_msg = f"Data load failed: {exc}"

        # ---------------------------------------------------
        # Compute output UI state
        # ---------------------------------------------------
        step = store_data["step"]
        step1_style = {"display": "block"} if step == 1 else {"display": "none"}
        step2_style = {"display": "block"} if step == 2 else {"display": "none"}
        next_text = "Submit" if step == 2 else "Next"
        back_style = {"display": "block"} if step == 2 else {"display": "none"}

        return (
            is_open,
            current_label(store_data),
            store_data,
            # overall_step_text(store_data),
            slider_max,
            slider_marks,
            slider_val,
            error_msg,
            fc_summary_text(store_data),
            step1_style,
            step2_style,
            next_text,      # ← MISSING (Output 12)
            back_style,     # ← MISSING (Output 13)
            *card_classes(fc_src, loc_src),
            fc_options,
            loc_options,
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
