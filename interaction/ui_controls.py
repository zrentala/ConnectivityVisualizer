def create_graph_controls(show_controls=True, directed=True):
    """
    Graph Controls UI: slider, metric/community radio, legend.
    """

    if not show_controls:
        return html.Div()

    shade_slider = html.Div(
        create_slider(
            id="graph-shade-top-x-slider",
            data_min=1,
            data_max=20,
            step=1,
            label="Shades top x for metric",
            default=5,
        ),
        id="graph-shade-top-x-container",
    )

    if directed:
        metric_options = [
            {"label": "In Degree", "value": "in_degree"},
            {"label": "Out Degree", "value": "out_degree"},
            {"label": "Bidirectional Degree", "value": "bidirectional"},
            {"label": "Node Connection Strength", "value": "node_connection_strengths"},
            {"label": "Community Partition", "value": "community"},
            {"label": "None", "value": "none"},
        ]
        default_metric = "in_degree"
    else:
        metric_options = [
            {"label": "Degree", "value": "in_degree"},
            {"label": "Node Connection Strength", "value": "node_connection_strengths"},
            {"label": "Community Partition", "value": "community"},
            {"label": "None", "value": "none"},
        ]
        default_metric = "in_degree"

    metric_radio = dbc.RadioItems(
        id="graph-metric-radio",
        options=metric_options,
        value=default_metric,
        inline=False,
        className="mb-2",
    )

    legend = html.Div(id="graph-legend", className="mt-2")

    return html.Div(
        [
            html.H5("Graph Controls", className="mb-2"),
            shade_slider,
            metric_radio,
            legend,
        ],
        className="bg-light p-3 rounded shadow-sm mb-3",
    )

from dash import html, dcc
import dash_bootstrap_components as dbc
from dataclasses import dataclass
from typing import Optional, Any, Dict


PRESET_CONFIGS: Dict[str, Dict[str, Any]] = {
    "small_undirected": {"n_elec": 16, "directed": False, "n_mat": 5},
    "medium_directed": {"n_elec": 8, "directed": True, "n_mat": 10},
    "large_undirected": {"n_elec": 64, "directed": False, "n_mat": 20},
}

PRESET_LOCS: Dict[str, int] = {
    "standard_1005": 343,           # 10-05 system ~343 electrodes (EEG positions) :contentReference[oaicite:1]{index=1}
    "standard_1020": 94,            # 10-20 system ~94 electrodes :contentReference[oaicite:2]{index=2}
    "standard_alphabetic": 65,      # alphabetic labeling ~65 electrodes :contentReference[oaicite:3]{index=3}
    "standard_postfixed": 100,      # postfix intermed. ~100 electrodes :contentReference[oaicite:4]{index=4}
    "standard_prefixed": 74,        # prefix intermed. ~74 electrodes :contentReference[oaicite:5]{index=5}
    "standard_primed": 100,         # primed ~100 electrodes :contentReference[oaicite:6]{index=6}

    "biosemi16": 16,                # BioSemi 16 channels :contentReference[oaicite:7]{index=7}
    "biosemi32": 32,                # BioSemi 32 channels :contentReference[oaicite:8]{index=8}
    "biosemi64": 64,                # BioSemi 64 channels :contentReference[oaicite:9]{index=9}
    "biosemi128": 128,              # BioSemi 128 channels :contentReference[oaicite:10]{index=10}
    "biosemi160": 160,              # BioSemi 160 channels :contentReference[oaicite:11]{index=11}
    "biosemi256": 256,              # BioSemi 256 channels :contentReference[oaicite:12]{index=12}

    "easycap-M1": 74,               # EasyCap M1 ~74 electrodes :contentReference[oaicite:13]{index=13}
    "easycap-M10": 61,              # EasyCap M10 ~61 electrodes :contentReference[oaicite:14]{index=14}
    "easycap-M43": 64,              # EasyCap M43 ~64 electrodes (MNE listing) :contentReference[oaicite:15]{index=15}

    "EGI_256": 256,                 # EGI Net 256 channels :contentReference[oaicite:16]{index=16}

    "GSN-HydroCel-32": 33,          # HydroCel 32 + Cz (~33) :contentReference[oaicite:17]{index=17}
    "GSN-HydroCel-64_1.0": 64,      # HydroCel 64 channels :contentReference[oaicite:18]{index=18}
    "GSN-HydroCel-65_1.0": 65,      # HydroCel 64 + Cz (~65) :contentReference[oaicite:19]{index=19}
    "GSN-HydroCel-128": 128,        # HydroCel 128 channels :contentReference[oaicite:20]{index=20}
    "GSN-HydroCel-129": 129,        # HydroCel 128 + Cz (~129) :contentReference[oaicite:21]{index=21}
    "GSN-HydroCel-256": 256,        # HydroCel 256 channels :contentReference[oaicite:22]{index=22}
    "GSN-HydroCel-257": 257,        # HydroCel 256 + Cz (~257) :contentReference[oaicite:23]{index=23}

    "mgh60": 60,                    # MGH 60 channels :contentReference[oaicite:24]{index=24}
    "mgh70": 70,                    # MGH 70 channels :contentReference[oaicite:25]{index=25}

    "artinis-octamon": 8            # Artinis OctaMon ~8 sources (not classic EEG) :contentReference[oaicite:26]{index=26}
}


container_class = "p-3 my-3 rounded shadow-sm border border-dark"

ID_LIST= [
    'thresh-stat-alpha-slider',
    'thresh-stat-stat_type-dropdown',
    'thresh-thresh_type-dropdown',




]


def create_slider(id: str, data_min: float, data_max: float, step: float,
                  label: str = "Frame", default: Optional[float] =None) -> html.Div:
    """Create a slider for selecting a value within a numeric range."""
    # Ensure valid ordering
    if data_max < data_min:
        raise ValueError("data_max must be >= data_min")
    if default is None:
        default = (data_max + data_min) / 2
    return html.Div(
        [
            dbc.Label(label),
            dcc.Slider(
                id=id,
                min=data_min,
                max=data_max,
                step=step,
                value=default,
                updatemode="mouseup",
                tooltip={"placement": "bottom", "always_visible": True},
                marks={data_min: str(data_min), data_max: str(data_max)},
            ),
        ]
    )

def create_range_slider(id: str, data_min: float, data_max: float, step: float, default, label:str) -> html.Div:
    # Compute midpoint
    mid = data_min + (data_max - data_min) / 2

    # Format labels nicely (avoid long decimals)
    def fmt(x):
        return f"{x:.3g}"  # adjust precision if needed

    marks = {
        data_min: fmt(data_min),
        mid: fmt(mid),
        data_max: fmt(data_max),
    }
    return html.Div(
        [
            dbc.Label(label),
            dcc.RangeSlider(
                id=id,
                min=data_min,
                max=data_max,
                step=step,
                value=default,
                allowCross=False,
                marks=marks,
            ),
        ]
    )

def create_dropdown(id:str, options: list[dict], label: str = "Select Option", clearable=False, default: str = None) -> html.Div:
    """Create a flexible dropdown component."""
    id_tag='dropdown'
    return html.Div(
        [
            dbc.Label(label),
            dcc.Dropdown(
                id=id,
                options=options,
                value=default if default is not None else (options[0]["value"] if options else None),
                clearable=clearable,
            )
        ],
        className="mb-3",
    )

def create_thresh_component() -> html.Div:
    label='Threshold'
    def _create_stat_test_component() -> html.Div:
        test_type_options = [{"label": "t-test", "value": "t"},
            {"label": "z-test", "value": "z"},
            {"label": "Wilcoxon", "value": "wilcoxon"},
            {"label": "Permutation w/o Correction", "value": "permutation w/o correction"},
            {"label": "Permutation with FDR Correction", "value": "permutation w correction"}
        ]
        test_type_dropdown = create_dropdown(
            id='thresh-stat-stat_type-dropdown',
            options=test_type_options,
            label="Statistical Test Type",
            default="t",
        )

        alpha_slider = create_slider(id='thresh-stat-alpha-slider', data_min=0, data_max=10, step=0.1, label="Alpha Level (%)", default=0.75)
        return html.Div(
            [
                test_type_dropdown,
                alpha_slider,
            ],
            className="mt-2",
        )
    stat_test_component = _create_stat_test_component()

    thresh_dropdown_options = [
        {"label":"Basic", "value": "Basic"},
        {"label":"MST", "value": "Minimum Spanning Tree"},
        {"label":"Statistical Test", "value": "Statistical Test"}
    ]



    thresh_dropdown = create_dropdown(
        id='thresh-thresh_type-dropdown',
        options=thresh_dropdown_options,
        label="Threshold Type",
        default="Basic",
    )

    """Create a threshold input component with optional slider."""
    return html.Div(
        children =[
            thresh_dropdown,
            html.Div(
                id='thresh-slider_container',
                children=[
                    create_slider(id=f"thresh-percent-slider", data_min=0, data_max=100, step=1, label="Threshold Value (%)", default=0)
                ],
                className="mt-2",
            ),
            html.Div(
                id=f"thresh-stat_test_container",
                children=[stat_test_component],
                className="mt-2",
            ),
        ],
        className=container_class,
    )

def create_node_options():
    node_size_slider = create_slider(id="viz-node-node_size-slider", data_min=15, data_max=50, step=1, label="Node Size", default=30)
    # node_opacity_slider = create_slider(id="viz-node-node_opacity-slider", data_min=0, data_max=1, step=0.01, label="Node Opacity (%)")

    edge_width_range = create_range_slider(id="viz-node-edge_width-range_slider", data_min=0, data_max=10, step=0.1, default=[0.4, 5.0], label="Edge Width Size")
    edge_opacity_slider = create_slider(id="viz-node-edge_opacity-slider", data_min=0, data_max=1, step=0.01, label="Edge Opacity (%)", default=0.8)
    arc_radius_slider = create_slider(id="viz-node-arc_radius-slider", data_min=0, data_max=1, step=0.01, label="Arc Radius", default= 0)
    return html.Div(
        id="viz-node-container",
        children=[
            html.Hr(),
            html.H5("Graph Visualization Options"),
            node_size_slider,
            # node_opacity_slider,
            edge_width_range,
            edge_opacity_slider,
            arc_radius_slider
        ],
        style={"display": "none"},
    )

def create_2d_options() -> html.Div:
    return html.Div(
        id="viz-2d-container",
        children=[
            html.Hr(),
            html.H5("2D Visualization Options"),
        ],
        style={"display": "none"},
    )

def create_3d_options() -> html.Div:
    brain_mesh_opacity_slider = create_slider(id="viz-3d-brain_mesh_opacity-slider", data_min=0, data_max=1, step=0.01, label="Brain Mesh Opacity (%)")

    hemisphere_row = dbc.Row(
        [
            dbc.Col(
                dbc.Checklist(
                    id="viz-3d-show_left_hem-checklist",
                    options=[{"label": "Show Left Hemisphere", "value": True}],
                    value=[True],
                    switch=True,
                ),
                width=6,
            ),
            dbc.Col(
                dbc.Checklist(
                    id="viz-3d-show_right_hem-checklist",
                    options=[{"label": "Show Right Hemisphere", "value": True}],
                    value=[True],
                    switch=True,
                ),
                width=6,
            ),
        ]
    )

    return html.Div(
        id="viz-3d-container",
        children=[
            html.Hr(),
            html.H5("3D Visualization Options"),
            hemisphere_row,
            brain_mesh_opacity_slider
        ],
        style={"display": "none"},
    )


def create_viz_controls() -> html.Div:

    viz_type_dropdown = create_dropdown(
        id="viz-fig_type-dropdown",
        label="Visualization Type",
        default="2D",
        options=[
            {"label": "2D", "value": "2D"},
            {"label": "3D", "value": "3D"},
            {"label": "Heatmap", "value": "Heatmap"},
        ],
    )

    color_map_options = [
        {"label": "Viridis", "value": "Viridis"},
        {"label": "Cividis", "value": "Cividis"},
        {"label": "Plasma", "value": "Plasma"},
        {"label": "Inferno", "value": "Inferno"},
        {"label": "Magma", "value": "Magma"},
        {"label": "Turbo", "value": "Turbo"},
        {"label": "Hot", "value": "Hot"},
        {"label": "Cool", "value": "Cool"},
        {"label": "Rainbow", "value": "Rainbow"},
        {"label": "Cubehelix", "value": "Cubehelix"},
        {"label": "Greys", "value": "Greys"},
        {"label": "YlGnBu", "value": "YlGnBu"},
        {"label": "RdBu", "value": "RdBu"},
        {"label": "Picnic", "value": "Picnic"},
        {"label": "Portland", "value": "Portland"},
        {"label": "Jet", "value": "Jet"},
        {"label": "Hotpink", "value": "Hotpink"},
        {"label": "Electric", "value": "Electric"},
        {"label": "Blackbody", "value": "Blackbody"},
        {"label": "Earth", "value": "Earth"},
        {"label": "Balance", "value": "Balance"},
        {"label": "RdYlGn", "value": "RdYlGn"},
        {"label": "Spectral", "value": "Spectral"},
    ]

    color_map_dropdown = create_dropdown(
        id="viz-color_type-dropdown",
        label="Color Map",
        default="Viridis",
        options=color_map_options
    )

    color_range = create_range_slider(id="viz-color-range_slider", data_max=1, data_min=0, step=0.01, default=[0.0,1.0], label="Color Range")

    # 2D/3D option blocks
    options_node = create_node_options()
    options_2d = create_2d_options()
    options_3d = create_3d_options()

    return dbc.Container(
        children=[
            viz_type_dropdown,
            color_map_dropdown,
            color_range,
            options_node,
            # options_2d,
            options_3d,
        ],
        fluid=True,
        className=container_class,
    )

def get_loc_options():
    return [
        {
            "label": f"{name} ({count} elecs)",
            "value": name
        }
        for name, count in PRESET_LOCS.items()
    ]

def get_fc_options():
    presets = PRESET_CONFIGS

    return [
        {
            "label": f"{key.replace('_', ' ').title()} "
                     f"(n={cfg['n_elec']}, mats={cfg['n_mat']})",
            "value": key
        }
        for key, cfg in presets.items()
    ]


def create_data_component() -> html.Div:
    """Data loader UI with two sections: FC data (top) and locations (bottom)."""

    # =====================================================================
    # SECTION 1: FC Data
    # =====================================================================
    
    # Mode selector for FC
    fc_mode_selector = html.Div(
        [
            html.H5("Functional Connectivity Data", className="mb-3"),
            dcc.Dropdown(
                id="data-fc-mode-selector",
                options=[
                    {"label": "Upload", "value": "upload"},
                    {"label": "Preset", "value": "preset"},
                    {"label": "Simulate", "value": "simulate"},
                ],
                value="upload",
                style={"maxWidth": "200px"},
                clearable=False,
            ),
            dcc.Store(
                id="data-fc-mode-store",
                data="upload",
            ),
        ],
        className="mb-4",
    )

    # Upload settings for FC
    fc_upload_settings = dbc.Card(
        id="data-fc-upload-settings",
        children=[
            dbc.CardBody([
                html.H6("Upload FC data"),
                dcc.Upload(
                    id="data-fc-upload",
                    children=html.Div(["Drag and drop or ", html.A("select a file")]),
                    multiple=False,
                    className="border p-3 text-center",
                ),
            ]),
        ],
        className="p-3 mb-2 border rounded bg-light",
    )

    # Preset settings for FC
    fc_preset_settings = dbc.Card(
        id="data-fc-preset-settings",
        style={"display": "none"},
        children=[
            dbc.CardBody([
                html.H6("Select FC dataset", className="mb-3"),
                create_dropdown(
                    id="data-fc-preset-dropdown",
                    options=get_fc_options(),
                    clearable=True,
                ),
            ]),
        ],
        className="p-3 mb-2 border rounded bg-light",
    )

    # Simulate settings for FC
    fc_simulate_settings = dbc.Card(
        id="data-fc-simulate-settings",
        style={"display": "none"},
        children=[
            dbc.CardBody([
                html.H6("Simulated FC Data Parameters", className="mb-3"),
                dbc.Row([
                    dbc.Col([
                        dbc.Label("# electrodes"),
                        dbc.Input(id="data-fc-sim-nelec", type="number", value=20, min=1),
                    ], md=6),
                    dbc.Col([
                        dbc.Label("# FC matrices"),
                        dbc.Input(id="data-fc-sim-nmat", type="number", value=10, min=1),
                    ], md=6),
                    dbc.Col([
                        # dbc.Label("Directed graph"),
                        dbc.Checklist(
                            id="data-directed-checkbox",
                            options=[{"label": "Directed graph", "value": True}],
                            value=[],
                            inline=True,
                        )
                    ], md=6)
                ]),
            ]),
        ],
        className="p-3 mb-2 border rounded bg-light",
    )

    # FC section container
    fc_section = html.Div(
        [
            fc_mode_selector,
            fc_upload_settings,
            fc_preset_settings,
            fc_simulate_settings,
        ],
    )

    # =====================================================================
    # SECTION 2: Location Data
    # =====================================================================

    # Mode selector for locations
    loc_mode_selector = html.Div(
        [
            html.H5("Location Data", className="mb-3"),
            dcc.Dropdown(
                id="data-loc-mode-selector",
                options=[
                    {"label": "Upload", "value": "upload"},
                    {"label": "Preset", "value": "preset"},
                    {"label": "Simulate", "value": "simulate"},
                ],
                value="upload",
                style={"maxWidth": "200px"},
                clearable=False,
            ),
            dcc.Store(
                id="data-loc-mode-store",
                data="upload",
            ),
        ],
        className="mb-4",
    )

    # Upload settings for locations
    loc_upload_settings = dbc.Card(
        id="data-loc-upload-settings",
        children=[
            dbc.CardBody([
                html.H6("Upload locations"),
                dcc.Upload(
                    id="data-loc-upload",
                    children=html.Div(["Drag and drop or ", html.A("select a file")]),
                    className="border p-3 text-center",
                ),
            ]),
        ],
        className="p-3 mb-2 border rounded bg-light",
    )

    # Preset settings for locations
    loc_preset_settings = dbc.Card(
        id="data-loc-preset-settings",
        style={"display": "none"},
        children=[
            dbc.CardBody([
                html.H6("Select locations", className="mb-3"),
                create_dropdown(
                    id="data-loc-preset-dropdown",
                    options=get_loc_options(),
                ),
            ]),
        ],
        className="p-3 mb-2 border rounded bg-light",
    )

    # Simulate settings for locations
    loc_simulate_settings = dbc.Card(
        id="data-loc-simulate-settings",
        style={"display": "none"},
        children=[
            dbc.CardBody([
                html.H6("Simulated locations will be generated based on FC data."),
            ]),
        ],
        className="p-3 mb-2 border rounded bg-light",
    )

    # Location section container
    loc_section = html.Div(
        [
            loc_mode_selector,
            loc_upload_settings,
            loc_preset_settings,
            loc_simulate_settings,
        ],
    )

    # =====================================================================
    # Dataset status and slider
    # =====================================================================

    data_label = html.Span(
        id="data-dataset-label",
        children="No dataset loaded",
        className="ms-2 fw-bold text-success",
    )

    animation_slider = create_slider(
        id="data-conn_idx-slider",
        data_min=0,
        data_max=0,
        step=1,
        label="Connectivity Matrix Index",
        default=0,
    )

    data_store = dcc.Store(
        id="data-store",
        data={"fc_cfg": {}, "loc_cfg": {}, "fc_meta": {}, "loc_meta": {}},
    )

    # =====================================================================
    # Submit Button
    # =====================================================================

    submit_button = dbc.Button(
        "Load Dataset",
        id="data-submit-button",
        color="success",
        size="lg",
        className="w-100",
        n_clicks=0,
    )

    # =====================================================================
    # Collapsible DataLoader UI
    # =====================================================================
    toggle_btn = dbc.Button(
        "˅",
        id="data-loader-toggle-btn",
        color="secondary",
        size="sm",
        className="mb-2",
        n_clicks=0,
        style={"width": "100%"},
    )
    return dbc.Card(
            [
                # Header (stable)
                dbc.CardHeader(
                    html.Div(
                        [
                            html.H5("Load Data", className="mb-0"),
                            toggle_btn,
                        ],
                        className="d-flex justify-content-between align-items-center",
                    )
                ),
        dbc.Collapse(
            html.Div(
                id="data-loader-container",
                children=[
                    dbc.Container([
                        # Header with dataset status
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Dataset:"), data_label
                            ], width="auto")
                        ], className="mb-4"),
                        # FC Section
                        dbc.Card(dbc.CardBody(fc_section), className="mb-4"),
                        # Location Section
                        dbc.Card(dbc.CardBody(loc_section), className="mb-4"),
                        # Submit button
                        submit_button,
                        # Data store
                        data_store,
                    ], fluid=True, className=container_class),
                ]
            ),
            id="data-loader-collapse",
            is_open=True,
        ),
        # Connectivity Matrix Index slider OUTSIDE collapse
        html.Div(
            animation_slider,
            className="mt-4 mb-4"
        ),
    ], className="mb-4")


def create_stat_component():
    return html.Div(
        [
            dbc.Collapse(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.H5("Network Statistics", className="card-title mb-3"),
                            html.Div([
                                html.Div([
                                    html.Div("Total Nodes:", className="fw-bold me-2"),
                                    html.Div(id="stat-collapse-total_nodes-container")
                                ], className="d-flex justify-content-between"),
                                html.Hr(className="my-2"),
                                html.Div([
                                    html.Div("Total Edges:", className="fw-bold me-2"),
                                    html.Div(id="stat-collapse-total_edges-container")
                                ], className="d-flex justify-content-between"),
                                html.Hr(className="my-2"),
                                html.Div([
                                    html.Div("Visible Edges:", className="fw-bold me-2"),
                                    html.Div(id="stat-collapse-visible_edges-container")
                                ], className="d-flex justify-content-between"),
                                html.Hr(className="my-2"),
                                html.Div([
                                    html.Div("Connection Density:", className="fw-bold me-2"),
                                    html.Div(id="stat-collapse-conn_density-container")
                                ], className="d-flex justify-content-between"),
                                html.Hr(className="my-2"),
                                html.Div([
                                    html.Div("Global Efficiency:", className="fw-bold me-2"),
                                    html.Div(id="stat-collapse-global_efficiency-container")
                                ], className="d-flex justify-content-between"),
                                html.Hr(className="my-2"),
                                html.Div([
                                    html.Div("Local Efficiency:", className="fw-bold me-2"),
                                    html.Div(id="stat-collapse-local_efficiency-container")
                                ], className="d-flex justify-content-between"),
                                html.Hr(className="my-2"),
                                html.Div([
                                    html.Div("Modularity:", className="fw-bold me-2"),
                                    html.Div(id="stat-collapse-modularity-container")
                                ], className="d-flex justify-content-between"),
                                html.Hr(className="my-2"),
                                html.Div([
                                    html.Div("Top Node Degrees:", className="fw-bold me-2"),
                                    html.Div(id="stat-collapse-top_node_degrees-container")
                                ], className="d-flex justify-content-between"),
                                html.Hr(className="my-2"),
                                html.Div([
                                    html.Div("Top Node Strengths:", className="fw-bold me-2"),
                                    html.Div(id="stat-collapse-top_node_strengths-container")
                                ], className="d-flex justify-content-between"),
                            ]),
                        ],
                    ),
                ),
                id="stats-collapse",
                is_open=True,
            ),
        ],
        style={
            "display": "flex",
            "flexDirection": "column",
            "alignItems": "flex-end",
            "justifyContent": "flex-start",
        },
    )