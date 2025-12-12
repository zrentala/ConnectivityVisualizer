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

# def create_data_component() -> html.Div:
#     """Create a 3-step data selection component with add/load dataset controls."""

#     # Step indicator
#     step_indicator = html.Div(id="data-step-indicator", children="Step 1: Load FC data", className="mb-2 fw-bold")

#     # "+" button to open modal
#     add_data_button = dbc.Button(
#         "+", id="data-add_dataset-button", color="primary", size="sm", className="ms-2",
#         title="Add or replace dataset", n_clicks=0
#     )

#     # Label showing current dataset status
#     data_label = html.Span(
#         id="data-dataset-label", children="No dataset loaded", className="ms-2"
#     )

#     # Modal layout
#     data_modal = dbc.Modal(
#         [
#             dbc.ModalHeader("Add or replace dataset"),
#             dbc.ModalBody(
#                 [
#                     # Step 1: Functional connectivity data
#                     html.Div(
#                         [
#                             html.H5("Step 1: Load functional connectivity (FC) data"),
#                             dcc.Upload(
#                                 id="data-fc-upload",
#                                 children=html.Div(["Drag and drop or ", html.A("select a file")]),
#                                 multiple=False, className="border p-3 text-center mb-2",
#                             ),
#                             create_dropdown(
#                                 id="data-fc-preset-dropdown",
#                                 options=[
#                                     {"label": "Small undirected (n=10, mats=5)", "value": "small_undirected"},
#                                     {"label": "Medium directed (n=20, mats=10)", "value": "medium_directed"},
#                                     {"label": "Large undirected (n=64, mats=20)", "value": "large_undirected"},
#                                 ],
#                                 clearable=True
#                             ),
#                             dbc.Button("Simulate FC data", id="data-fc-gen-btn", color="secondary", className="mt-2", n_clicks=0),
#                         ],
#                         id="step-1-container", className="mb-3"
#                     ),
                    
#                     # Step 2: Location data
#                     html.Div(
#                         [
#                             html.H5("Step 2: Load location data"),
#                             dcc.Upload(
#                                 id="data-loc-upload",
#                                 children=html.Div(["Drag and drop or ", html.A("select a file")]),
#                                 multiple=False, className="border p-3 text-center mb-2",
#                             ),
#                             create_dropdown(
#                                 id="data-loc-preset-dropdown",
#                                 options=[
#                                     {"label": "10-channel 10-20 layout", "value": "standard_10_20"},
#                                     {"label": "64-channel layout", "value": "standard_64"},
#                                 ],
#                                 clearable=True
#                             ),
#                             dbc.Button("Generate locations", id="data-loc-gen-btn", color="secondary", className="mt-2", n_clicks=0),
#                         ],
#                         id="step-2-container", className="mb-3"
#                     ),
                    
#                     # Step 3: Directed/undirected
#                     html.Div(
#                         [
#                             html.H5("Step 3: Directed or undirected?"),
#                             dbc.Checkbox(id="data-directed-checkbox", value=False, className="me-2"),
#                             html.Label("Directed"),
#                         ],
#                         id="step-3-container", className="mb-2"
#                     ),
#                 ]
#             ),
#             dbc.ModalFooter(
#                 dbc.Button("Close", id="data-modal-close-button", className="ms-auto", n_clicks=0)
#             ),
#         ],
#         id="data-modal",
#         is_open=False,
#         centered=True,
#         backdrop="static",
#     )

#     # Store for metadata
#     data_store = dcc.Store(id="data-store", data={"fc": {}, "loc": {}, "directed": False, 'step': 1})

#     # Slider for FC matrices
#     animation_slider = create_slider(id="data-conn_idx-slider", data_min=0, data_max=0, step=1, label="Connectivity Matrix Index", default=0)

#     # Container layout
#     return dbc.Container(
#         [
#             dbc.Row([dbc.Col([dbc.Label("Dataset:"), data_label, add_data_button], width="auto")], className="mb-3"),
#             step_indicator,
#             animation_slider,
#             data_modal,
#             data_store,
#         ],
#         fluid=True, className=container_class
#     )

def get_loc_options():
    presets = PRESET_LOCS.keys()
    # Normal case → real selectable options
    return [{"label": name, "value": name} for name in presets]

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
    """Two-step data selection UI (Step 1: FC, Step 2: locations) with radios wrapping each section."""

    # step_indicator = html.Div(
    #     id="data-step-indicator",
    #     children="Step 1: Load FC data",
    #     className="mb-2 fw-bold",
    # )

    add_data_button = dbc.Button(
        "+",
        id="data-add_dataset-button",
        color="primary",
        size="sm",
        className="ms-2",
        title="Add or replace dataset",
        n_clicks=0,
    )

    data_label = html.Span(
        id="data-dataset-label",
        children="No dataset loaded",
        className="ms-2",
    )

    # ---------- STEP 1: FC ----------
    fc_source = dbc.RadioItems(
        id="data-fc-radio",
        options=[
            {"label": html.Div([
                dbc.Card(
                    id="data-fc-radio-upload-card",
                    children=[
                        html.H6("Upload FC data"),
                        dcc.Upload(
                            id="data-fc-upload",
                            children=html.Div(["Drag and drop or ", html.A("select a file")]),
                            multiple=False,
                            className="border p-3 text-center",
                        ),
                    ],
                    className="p-3 mb-2 border rounded bg-light",
                )
            ]), "value": "upload"},

            {"label": html.Div([
                dbc.Card(
                    id="data-fc-radio-preset-card",
                    children=[
                        html.H6("Preset FC dataset"),
                        create_dropdown(
                            id="data-fc-preset-dropdown",
                            options=get_fc_options(),
                            clearable=True,
                        ),
                    ],
                    className="p-3 mb-2 border rounded bg-light",
                )
            ]), "value": "preset"},

            {"label": html.Div([
                dbc.Card(
                    id="data-fc-radio-sim-card",
                    children=[
                        html.H6("Simulated FC data"),
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("# electrodes"),
                                dbc.Input(id="data-fc-sim-nelec", type="number", value=20, min=1),
                            ], md=4),

                            dbc.Col([
                                dbc.Label("# FC matrices"),
                                dbc.Input(id="data-fc-sim-nmat", type="number", value=10, min=1),
                            ], md=4),

                            dbc.Col([
                                dbc.Label("Directed?"),
                                dbc.Checkbox(id="data-directed-checkbox", value=False),
                            ], md=4),
                        ]),
                        # html.Div("Simulated FC will be generated on Next.", className="text-muted mt-2"),
                    ],
                    className="p-3 mb-2 border rounded bg-light",
                )
            ]), "value": "simulate"},
        ],

        value="upload",
        inline=False,
        className="radio-wrapped-group w-100",
    )


    step1_view = html.Div(
        id="data-step1-view",
        children=[
            html.H5("Step 1: Load functional connectivity (FC) data", className="mb-3"),
            fc_source
        ],
        className="w-100"
    )

    # ---------- STEP 2: Locations ----------
    loc_source = dbc.RadioItems(
        id="data-loc-radio",
        options=[
            {"label": html.Div([
                dbc.Card(
                    id="data-loc-radio-upload-card",
                    children=[
                        html.H6("Upload locations"),
                        dcc.Upload(
                            id="data-loc-upload",
                            children=html.Div(["Drag and drop or ", html.A("select a file")]),
                            className="border p-3 text-center",
                        ),
                    ],
                    className="p-3 mb-2 border rounded bg-light",
                )
            ]), "value": "upload"},

            {"label": html.Div([
                dbc.Card(
                    id="data-loc-radio-preset-card",
                    children=[
                        html.H6("Preset locations"),
                        create_dropdown(
                            id="data-loc-preset-dropdown",
                            options=get_loc_options(),
                        ),
                    ],
                    className="p-3 mb-2 border rounded bg-light",
                )
            ]), "value": "preset"},

            {"label": html.Div([
                dbc.Card(
                    id="data-loc-radio-sim-card",
                    children=[
                        html.H6("Simulated locations"),
                        # html.Div("Locations will be generated based on FC when you click Next."),
                    ],
                    className="p-3 mb-2 border rounded bg-light",
                )
            ]), "value": "simulate"},
        ],
        value="upload",
        # inline=False,
    )


    step2_view = html.Div(
        id="data-step2-view",
        style={"display": "none"},
        children=[
            html.H5("Step 2: Load location data", className="mb-3"),
            html.Div(
                id="data-fc-summary",
                className="mb-3 fst-italic",
                children="No FC data loaded yet.",
            ),
            loc_source
        ],
    )

    data_modal = dbc.Modal(
        [
            dbc.ModalHeader("Add or replace dataset"),
            dbc.ModalBody(
                [
                    step1_view,
                    step2_view,
                    html.Div(id="data-error-message", className="text-danger mt-2"),
                ]
            ),
            dbc.ModalFooter(
                [
                    dbc.Button(
                        "Back",
                        id="data-back-button",
                        color="secondary",
                        n_clicks=0,
                        className="me-auto",
                    ),
                    dbc.Button(
                        "Next",
                        id="data-next-button",
                        color="primary",
                        n_clicks=0,
                        className="me-2",
                    ),
                ]),
        ],
        id="data-modal",
        is_open=False,
        centered=True,
        backdrop="static",
        size="",
    )

    data_store = dcc.Store(
        id="data-store",
        data={"step": 1, "fc_cfg": {}, "loc_cfg": {}, "fc_meta": {}, "loc_meta": {}},
    )

    animation_slider = create_slider(
        id="data-conn_idx-slider",
        data_min=0,
        data_max=0,
        step=1,
        label="Connectivity Matrix Index",
        default=0,
    )

    return dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [dbc.Label("Dataset:"), data_label, add_data_button],
                        width="auto",
                    )
                ],
                className="mb-3",
            ),
            # step_indicator,
            animation_slider,
            data_modal,
            data_store,
        ],
        fluid=True,
        className=container_class,
    )


def create_stat_component():
    return html.Div(
        [
            dbc.Card(
                dbc.CardBody(
                    [
                        html.H5("Network Statistics", className="card-title mb-3"),

                        html.Div(
                            [
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
                            ],
                        ),
                    ],
                ),
            
            ),
        ],
        style={
            "display": "flex",
            "flexDirection": "column",
            "alignItems": "flex-end",  # right aligned
            "justifyContent": "flex-start",
            # "marginLeft": "auto",  # pushes component to the right!
        },
    )

