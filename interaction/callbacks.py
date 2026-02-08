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
    print("=== REGISTERING DATA CALLBACKS ===")
    loader = DataLoader()

    # IDs
    label_id = "data-dataset-label"
    store_id = "data-store"
    slider_id = "data-conn_idx-slider"

    # FC mode-related IDs
    fc_mode_selector_id = "data-fc-mode-selector"
    fc_mode_store_id = "data-fc-mode-store"
    fc_upload_settings_id = "data-fc-upload-settings"
    fc_preset_settings_id = "data-fc-preset-settings"
    fc_simulate_settings_id = "data-fc-simulate-settings"

    # Location mode-related IDs
    loc_mode_selector_id = "data-loc-mode-selector"
    loc_mode_store_id = "data-loc-mode-store"
    loc_upload_settings_id = "data-loc-upload-settings"
    loc_preset_settings_id = "data-loc-preset-settings"
    loc_simulate_settings_id = "data-loc-simulate-settings"

    # Upload/Preset/Simulate IDs
    fc_upload_id = "data-fc-upload"
    fc_preset_id = "data-fc-preset-dropdown"
    fc_sim_nelec_id = "data-fc-sim-nelec"
    fc_sim_nmat_id = "data-fc-sim-nmat"
    directed_id = "data-directed-checkbox"

    loc_upload_id = "data-loc-upload"
    loc_preset_id = "data-loc-preset-dropdown"

    # ---------------------------------------------------
    # MODE SWITCHING CALLBACKS
    # ---------------------------------------------------
    @app.callback(
        Output(fc_mode_store_id, "data"),
        Output(fc_upload_settings_id, "style"),
        Output(fc_preset_settings_id, "style"),
        Output(fc_simulate_settings_id, "style"),
        Input(fc_mode_selector_id, "value"),
        prevent_initial_call=False,
    )
    def switch_fc_mode(mode):
        if mode is None:
            mode = "upload"
        
        upload_style = {"display": "block"} if mode == "upload" else {"display": "none"}
        preset_style = {"display": "block"} if mode == "preset" else {"display": "none"}
        simulate_style = {"display": "block"} if mode == "simulate" else {"display": "none"}
        
        return mode, upload_style, preset_style, simulate_style

    @app.callback(
        Output(loc_mode_store_id, "data"),
        Output(loc_upload_settings_id, "style"),
        Output(loc_preset_settings_id, "style"),
        Output(loc_simulate_settings_id, "style"),
        Input(loc_mode_selector_id, "value"),
        prevent_initial_call=False,
    )
    def switch_loc_mode(mode):
        if mode is None:
            mode = "upload"
        
        upload_style = {"display": "block"} if mode == "upload" else {"display": "none"}
        preset_style = {"display": "block"} if mode == "preset" else {"display": "none"}
        simulate_style = {"display": "block"} if mode == "simulate" else {"display": "none"}
        
        return mode, upload_style, preset_style, simulate_style

    # ---------------------------------------------------
    # COMBINED DATA LOADING AND SUBMISSION CALLBACK
    # ---------------------------------------------------
    @app.callback(
        Output(label_id, "children"),
        Output(store_id, "data"),
        Output(slider_id, "max"),
        Output(slider_id, "marks"),
        Output(slider_id, "value"),
        
        Input("data-submit-button", "n_clicks"),
        Input(fc_upload_id, "contents"),
        Input(fc_preset_id, "value"),
        Input(fc_sim_nelec_id, "value"),
        Input(fc_sim_nmat_id, "value"),
        Input(loc_upload_id, "contents"),
        Input(loc_preset_id, "value"),
        Input(fc_mode_store_id, "data"),
        Input(loc_mode_store_id, "data"),
        Input(directed_id, "value"),
        
        State(fc_upload_id, "filename"),
        State(loc_upload_id, "filename"),
        State(store_id, "data"),
        prevent_initial_call=False,
    )
    def load_and_submit_data(
        submit_clicks,
        fc_contents,
        fc_preset_val,
        fc_sim_nelec,
        fc_sim_nmat,
        loc_contents,
        loc_preset_val,
        fc_mode_val,
        loc_mode_val,
        directed_val,
        fc_filename,
        loc_filename,
        store_data,
    ):
        """Combined callback: update configuration and load data when submitted."""
        
        if not isinstance(store_data, dict):
            store_data = {}
        
        ctx = callback_context
        if not ctx.triggered:
            trigger_id = None
        else:
            trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
        
        fc_src = fc_mode_val or "upload"
        loc_src = loc_mode_val or "upload"
        
        try:
            # Always update configuration
            fc_cfg = loader.make_fc_cfg(
                source=fc_src,
                upload=(fc_contents, fc_filename),
                preset=fc_preset_val,
                simulate=(fc_sim_nelec, fc_sim_nmat),
            )
            
            if fc_cfg["type"] == "sim":
                fc_cfg["directed"] = bool(directed_val)
            
            store_data["fc_cfg"] = fc_cfg
            store_data["directed"] = bool(directed_val)
            
            n_elec = fc_cfg.get("n_elec", 0)
            loc_cfg = loader.make_loc_cfg(
                source=loc_src,
                upload=(loc_contents, loc_filename),
                preset=loc_preset_val,
                n_elec=n_elec,
            )
            
            store_data["loc_cfg"] = loc_cfg
            
            n_mats = fc_cfg.get("n_mat", 0)
            slider_max = max(0, n_mats - 1)
            marks = {0: "0", slider_max: str(slider_max)} if slider_max > 0 else {0: "0"}
            slider_value = 0
            
            # Check if submit was clicked
            if trigger_id == "data-submit-button" and submit_clicks and submit_clicks > 0:
                if not fc_cfg or not loc_cfg:
                    return "Please select both FC data and locations", store_data, slider_max, marks, slider_value
                
                bd, meta, slider = loader.build_braindata(fc_cfg, loc_cfg, bool(directed_val))
                
                global_state.brain_data = bd
                global_state.threshold = Threshold()
                global_state.viz = VizUIManager(bd, global_state.threshold)
                
                store_data["fc_meta"] = meta["fc"].__dict__
                store_data["loc_meta"] = meta["loc"].__dict__
                
                fc_name = fc_cfg.get("name", "Unknown FC")
                loc_name = loc_cfg.get("name", "Unknown Locations")
                label = f"✓ Loaded: {fc_name} | {loc_name}"
                
                return label, store_data, slider_max, marks, slider_value
            
            else:
                fc_name = fc_cfg.get("name", "Unknown")
                loc_name = loc_cfg.get("name", "Unknown")
                label = f"Ready to load: {fc_name} | {loc_name}"
                
                return label, store_data, slider_max, marks, slider_value
        
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            return f"Error: {str(e)}", store_data, 0, {0: "0"}, 0

# def register_data_callbacks(app: Dash, global_state: GlobalAppState):
#     print("=== REGISTERING DATA CALLBACKS ===")
#     loader = DataLoader()

#     # IDs
#     label_id = "data-dataset-label"
#     store_id = "data-store"
#     slider_id = "data-conn_idx-slider"
#     error_id = "data-error-message"

#     # FC mode-related IDs
#     fc_mode_selector_id = "data-fc-mode-selector"
#     fc_mode_store_id = "data-fc-mode-store"
#     fc_upload_settings_id = "data-fc-upload-settings"
#     fc_preset_settings_id = "data-fc-preset-settings"
#     fc_simulate_settings_id = "data-fc-simulate-settings"

#     # Location mode-related IDs
#     loc_mode_selector_id = "data-loc-mode-selector"
#     loc_mode_store_id = "data-loc-mode-store"
#     loc_upload_settings_id = "data-loc-upload-settings"
#     loc_preset_settings_id = "data-loc-preset-settings"
#     loc_simulate_settings_id = "data-loc-simulate-settings"

#     # Upload/Preset/Simulate IDs
#     fc_upload_id = "data-fc-upload"
#     fc_preset_id = "data-fc-preset-dropdown"
#     fc_sim_nelec_id = "data-fc-sim-nelec"
#     fc_sim_nmat_id = "data-fc-sim-nmat"
#     directed_id = "data-directed-checkbox"

#     loc_upload_id = "data-loc-upload"
#     loc_preset_id = "data-loc-preset-dropdown"
    
#     print(f"FC Mode Selector ID: {fc_mode_selector_id}")
#     print(f"FC Mode Store ID: {fc_mode_store_id}")
#     print(f"FC Upload Settings ID: {fc_upload_settings_id}")

#     # ---------------------------------------------------
#     # ---------------------------------------------------
#     # MODE SWITCHING CALLBACKS
#     # ---------------------------------------------------
#     # FC Mode switching
#     @app.callback(
#         Output(fc_mode_store_id, "data"),
#         Output(fc_upload_settings_id, "style"),
#         Output(fc_preset_settings_id, "style"),
#         Output(fc_simulate_settings_id, "style"),
#         Input(fc_mode_selector_id, "value"),
#     )
#     def switch_fc_mode(mode):
#         """Switch between Upload, Preset, and Simulate modes for FC data."""
#         print(f"=== FC Mode callback triggered ===")
#         print(f"Mode value: {mode}")
#         print(f"Mode type: {type(mode)}")
        
#         upload_style = {"display": "block"} if mode == "upload" else {"display": "none"}
#         preset_style = {"display": "block"} if mode == "preset" else {"display": "none"}
#         simulate_style ={"display": "block"} if mode == "simulate" else {"display": "none"}
        
#         print(f"Returning: store={mode}, upload_style={upload_style}, preset_style={preset_style}, simulate_style={simulate_style}")
#         return mode, upload_style, preset_style, simulate_style

#     # Location Mode switching
#     @app.callback(
#         Output(loc_mode_store_id, "data"),
#         Output(loc_upload_settings_id, "style"),
#         Output(loc_preset_settings_id, "style"),
#         Output(loc_simulate_settings_id, "style"),
#         Input(loc_mode_selector_id, "value"),
#     )
#     def switch_loc_mode(mode):
#         """Switch between Upload, Preset, and Simulate modes for locations."""
#         print(f"Location Mode switched to: {mode}")
        
#         upload_style = {} if mode == "upload" else {"display": "none"}
#         preset_style = {} if mode == "preset" else {"display": "none"}
#         simulate_style = {} if mode == "simulate" else {"display": "none"}
        
#         return mode, upload_style, preset_style, simulate_style
#     # ---------------------------------------------------
#     # DATA LOADING CALLBACK
#     # ---------------------------------------------------
#     @app.callback(
#         Output(label_id, "children"),
#         Output(store_id, "data"),
#         Output(slider_id, "max"),
#         Output(slider_id, "marks"),
#         Output(slider_id, "value"),
#         Input(fc_upload_id, "contents"),
#         Input(fc_preset_id, "value"),
#         Input(fc_sim_nelec_id, "value"),
#         Input(fc_sim_nmat_id, "value"),
#         Input(loc_upload_id, "contents"),
#         Input(loc_preset_id, "value"),
#         Input(fc_mode_store_id, "data"),
#         Input(loc_mode_store_id, "data"),
#         Input(directed_id, "value"),
#         State(fc_upload_id, "filename"),
#         State(loc_upload_id, "filename"),
#         State(store_id, "data"),
#         prevent_initial_call=False,
#     )
#     def load_data(
#         fc_contents,
#         fc_preset_val,
#         fc_sim_nelec,
#         fc_sim_nmat,
#         loc_contents,
#         loc_preset_val,
#         fc_mode_val,
#         loc_mode_val,
#         directed_val,
#         fc_filename,
#         loc_filename,
#         store_data,
#     ):
#         """Load FC and location data based on selections."""
#         if not isinstance(store_data, dict):
#             store_data = {}

#         fc_src = fc_mode_val or "upload"
#         loc_src = loc_mode_val or "upload"

#         try:
#             # Load FC data
#             fc_cfg = loader.make_fc_cfg(
#                 source=fc_src,
#                 upload=(fc_contents, fc_filename),
#                 preset=fc_preset_val,
#                 simulate=(fc_sim_nelec, fc_sim_nmat),
#             )
            
#             if fc_cfg["type"] == "sim":
#                 fc_cfg["directed"] = bool(directed_val)
            
#             store_data["fc_cfg"] = fc_cfg
#             store_data["directed"] = bool(directed_val)

#             # Load location data
#             n_elec = fc_cfg.get("n_elec", 0)
#             loc_cfg = loader.make_loc_cfg(
#                 source=loc_src,
#                 upload=(loc_contents, loc_filename),
#                 preset=loc_preset_val,
#                 n_elec=n_elec,
#             )
            
#             store_data["loc_cfg"] = loc_cfg

#             # Create dataset label
#             fc_name = fc_cfg.get("name", "Unknown")
#             loc_name = loc_cfg.get("name", "Unknown")
#             label = f"FC: {fc_name} | Locations: {loc_name}"
            
#             # Create slider marks
#             n_mats = fc_cfg.get("n_mat", 0)
#             slider_max = max(0, n_mats - 1)
#             marks = {0: "0", slider_max: str(slider_max)} if slider_max > 0 else {0: "0"}
#             slider_value = 0

#             return label, store_data, slider_max, marks, slider_value

#         except Exception as e:
#             print(f"Error loading data: {e}")
#             return "Error loading data", store_data, 0, {0: "0"}, 0

#     # ---------------------------------------------------
#     # SUBMIT BUTTON CALLBACK - Build and Load Data
#     # ---------------------------------------------------
#     @app.callback(
#         Output(label_id, "children"),
#         Output(store_id, "data"),
#         Input("data-submit-button", "n_clicks"),
#         State(store_id, "data"),
#         prevent_initial_call=True,
#     )
#     def handle_submit(submit_clicks, store_data):
#         """Build the brain data and update global state when submit is clicked."""
#         if not isinstance(store_data, dict):
#             store_data = {}

#         fc_cfg = store_data.get("fc_cfg", {})
#         loc_cfg = store_data.get("loc_cfg", {})
#         directed = store_data.get("directed", False)

#         # Check if we have both FC and location configs
#         if not fc_cfg or not loc_cfg:
#             return "Please select both FC data and locations", store_data

#         try:
#             # Build the final dataset
#             bd, meta, slider = loader.build_braindata(fc_cfg, loc_cfg, directed)

#             # Save into global_state
#             global_state.brain_data = bd
#             global_state.threshold = Threshold()
#             global_state.viz = VizUIManager(bd, global_state.threshold)

#             # Update store with metadata
#             store_data["fc_meta"] = meta["fc"].__dict__
#             store_data["loc_meta"] = meta["loc"].__dict__

#             # Create success label
#             fc_name = fc_cfg.get("name", "Unknown FC")
#             loc_name = loc_cfg.get("name", "Unknown Locations")
#             label = f"✓ Loaded: {fc_name} | {loc_name}"

#             return label, store_data

#         except Exception as e:
#             print(f"Error building dataset: {e}")
#             return f"Error: {str(e)}", store_data


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
