from dash import html, dcc
import dash_bootstrap_components as dbc
from dataclasses import dataclass

container_class = "p-3 my-3 rounded shadow-sm border border-dark"

def create_slider(id: str, n_frames: int, label: str = "Frame") -> html.Div:
    """Create a slider for selecting connectivity matrix index."""
    return html.Div(
        [
            dbc.Label(label),
            dcc.Slider(
                id=id,
                min=0,
                max=max(n_frames - 1, 0),
                step=1,
                value=0,
                updatemode="mouseup",
                tooltip={"placement": "bottom", "always_visible": True},
                marks={0: "0", n_frames: str(n_frames - 1)} if n_frames > 1 else None,
            ),
        ],
        # className="m-3",
    )

def create_thesh_component(id: str, label: str = "Threshold") -> html.Div:
    def _create_stat_test_component(id: str) -> html.Div:
        test_type_options = [{"label": "t-test", "value": "t"},
            {"label": "z-test", "value": "z"},
            {"label": "Wilcoxon", "value": "wilcoxon"},
            {"label": "Permutation w/o Correction", "value": "permutation w/o correction"},
            {"label": "Permutation with FDR Correction", "value": "permutation w correction"}
        ]
        test_type_dropdown = create_dropdown(
            id=f"{id}-test-type",
            options=test_type_options,
            label="Statistical Test Type",
            default="t",
        )
        return html.Div(
            [
                test_type_dropdown,
                html.Div(
                [
                    dbc.Label("Alpha Level (%)"),
                    dcc.Slider(
                        id=f"{id}-alpha-slider",
                        min=0,
                        max=max(10 - 1, 0),
                        step=0.1,
                        value=0,
                        updatemode="mouseup",
                        tooltip={"placement": "bottom", "always_visible": True},
                        marks={0: "0", 10: str(10 - 1)} if 10 > 1 else None,
                    ),
                ],
                className="mt-3"
                ),
            ],
            className="mt-2",
        )
    stat_test_component = _create_stat_test_component(id)

    thresh_dropdown_options = [
        {"label":"Basic", "value": "Basic"},
        {"label":"MST", "value": "Minimum Spanning Tree"},
        {"label":"Statistical Test", "value": "Statistical Test"}
    ]

    thresh_dropdown = create_dropdown(
        id=f'{id}-type-dropdown',
        options=thresh_dropdown_options,
        label="Threshold Type",
        default="Basic",
    )

    """Create a threshold input component with optional slider."""
    return html.Div(
        children =[
            thresh_dropdown,
            html.Div(
                id=f"{id}-slider-container",
                children=[
                    create_slider(id=f"{id}-slider", n_frames=100, label="Threshold Value (%)")
                ],
                className="mt-2",
            ),
            html.Div(
                id=f"{id}-stat-test-container",
                children=[stat_test_component],
                className="mt-2",
            ),
        ],
        className=container_class,
    )

def create_dropdown(id: str, options: list[dict], label: str = "Select Option", default: str = None) -> html.Div:
    """Create a flexible dropdown component."""
    return html.Div(
        [
            dbc.Label(label),
            dcc.Dropdown(
                id=id,
                options=options,
                value=default if default is not None else (options[0]["value"] if options else None),
                clearable=False,
            )
        ],
        className="mb-3",
    )
def create_2d_options(id_prefix: str) -> html.Div:
    node_size_slider = create_slider(
        id=f"{id_prefix}-node-size-2d",
        label="Node Size",
        n_frames=10,
        step=1,
        default=8,
    )

    edge_min_slider = create_slider(
        id=f"{id_prefix}-edge-min-2d",
        label="Edge Width (Min)",
        n_frames=100,
        step=0.1,
        default=0.5,
    )

    edge_max_slider = create_slider(
        id=f"{id_prefix}-edge-max-2d",
        label="Edge Width (Max)",
        n_frames=10,
        step=0.1,
        default=5.0,
    )

    return html.Div(
        id=f"{id_prefix}-options-2d",
        children=[
            html.Hr(),
            html.H5("2D Visualization Options"),
            node_size_slider,
            edge_min_slider,
            edge_max_slider,
        ],
        style={"display": "none"},
    )

def create_3d_options(id_prefix: str) -> html.Div:
    node_size_slider = create_slider(
        id=f"{id_prefix}-node-size-3d",
        label="Node Size",
        n_frames=20,
        step=1,
        default=8,
    )

    edge_min_slider = create_slider(
        id=f"{id_prefix}-edge-min-3d",
        label="Edge Width (Min)",
        n_frames=10,
        step=0.1,
        default=0.5,
    )

    edge_max_slider = create_slider(
        id=f"{id_prefix}-edge-max-3d",
        label="Edge Width (Max)",
        n_frames=10,
        step=0.1,
        default=5.0,
    )

    arc_points_slider = create_slider(
        id=f"{id_prefix}-arc-points-3d",
        label="Arc Curve Resolution (# Points)",
        n_frames=10,
        step=1,
        default=50,
    )

    hemisphere_row = dbc.Row(
        [
            dbc.Col(
                dbc.Checklist(
                    id=f"{id_prefix}-show-left-3d",
                    options=[{"label": "Show Left Hemisphere", "value": True}],
                    value=[True],
                    switch=True,
                ),
                width=6,
            ),
            dbc.Col(
                dbc.Checklist(
                    id=f"{id_prefix}-show-right-3d",
                    options=[{"label": "Show Right Hemisphere", "value": True}],
                    value=[True],
                    switch=True,
                ),
                width=6,
            ),
        ]
    )

    return html.Div(
        id=f"{id_prefix}-3d-options",
        children=[
            html.Hr(),
            html.H5("3D Visualization Options"),
            node_size_slider,
            edge_min_slider,
            edge_max_slider,
            arc_points_slider,
            html.Br(),
            hemisphere_row,
        ],
        style={"display": "none"},
    )


def create_viz_controls(id_prefix: str, n_mat: int) -> html.Div:

    viz_type_dropdown = create_dropdown(
        id="viz-type-dropdown",
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
        id="color-type-dropdown",
        label="Color Map",
        default="Viridis",
        options=color_map_options
    )

    color_range = html.Div(
        [
            dbc.Label("Color Range (0–1)"),
            dcc.RangeSlider(
                id="conn-range",
                min=0,
                max=1,
                step=0.01,
                value=[0.0, 1.0],
                allowCross=False,
                marks={0: "0.0", 0.5: "0.5", 1.0: "1.0"},
            ),
        ]
    )

    # 2D/3D option blocks
    options_2d = create_2d_options(id_prefix)
    options_3d = create_3d_options(id_prefix)

    return dbc.Container(
        children=[
            viz_type_dropdown,
            color_map_dropdown,
            color_range,
            options_2d,
            options_3d,
        ],
        fluid=True,
        className="p-2",
    )



def create_data_component(id_prefix: str, n_mat: int) -> html.Div:
    """Create data selection component with add/load dataset controls."""
    # Slider over connectivity matrices
    animation_slider = create_slider(
        id=f"{id_prefix}-mat-idx",
        n_frames=n_mat,
        label="Connectivity Matrix Index",
    )

    # Label that shows either "No dataset loaded" or current dataset name
    data_label = html.Span(
        id=f"{id_prefix}-dataset-label",
        children="No dataset loaded",
        className="ms-2",
    )

    # "+" button to add / replace data
    add_data_button = dbc.Button(
        "+",
        id=f"{id_prefix}-add-data-btn",
        color="primary",
        size="sm",
        className="ms-2",
        title="Add or replace dataset",
        n_clicks=0,
    )

    # Modal that appears when you click the "+" button
    data_modal = dbc.Modal(
        [
            dbc.ModalHeader("Add or replace dataset"),
            dbc.ModalBody(
                [
                    # 1) Load your own
                    html.H5("1. Load your own data"),
                    dcc.Upload(
                        id=f"{id_prefix}-upload",
                        children=html.Div(
                            [
                                "Drag and drop or ",
                                html.A("select a file"),
                            ]
                        ),
                        multiple=False,
                        className="border p-3 text-center mb-3",
                    ),

                    html.Hr(),

                    # 2) Choose from preset
                    html.H5("2. Choose a preset dataset"),
                    dcc.Dropdown(
                        id=f"{id_prefix}-preset-dropdown",
                        placeholder="Select preset dataset...",
                        options=[
                            {"label": "Small undirected (n=10, mats=5)", "value": "small_undirected"},
                            {"label": "Medium directed (n=20, mats=10)", "value": "medium_directed"},
                            {"label": "Large undirected (n=64, mats=20)", "value": "large_undirected"},
                        ],
                        clearable=True,
                        className="mb-3",
                    ),

                    html.Hr(),

                    # 3) Generate your own
                    html.H5("3. Generate simulated data"),
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dbc.Label("Number of nodes"),
                                    dbc.Input(
                                        id=f"{id_prefix}-gen-n-elec",
                                        type="number",
                                        min=2,
                                        step=1,
                                        value=20,
                                    ),
                                ],
                                md=4,
                            ),
                            dbc.Col(
                                [
                                    dbc.Label("Number of mats"),
                                    dbc.Input(
                                        id=f"{id_prefix}-gen-n-mat",
                                        type="number",
                                        min=1,
                                        step=1,
                                        value=10,
                                    ),
                                ],
                                md=4,
                            ),
                            dbc.Col(
                                [
                                    dbc.Label("Directed?"),
                                    dbc.Checkbox(
                                        id=f"{id_prefix}-gen-directed",
                                        value=False,
                                    ),
                                ],
                                # md=4,
                                className="d-flex align-items-end",
                            ),
                        ],
                        className="mb-2",
                    ),
                    dbc.Button(
                        "Generate",
                        id=f"{id_prefix}-gen-btn",
                        color="secondary",
                        className="mt-2",
                        n_clicks=0,
                    ),
                ]
            ),
            dbc.ModalFooter(
                dbc.Button(
                    "Close",
                    id=f"{id_prefix}-data-modal-close",
                    className="ms-auto",
                    n_clicks=0,
                )
            ),
        ],
        id=f"{id_prefix}-data-modal",
        is_open=False,
        centered=True,
        backdrop="static",
    )

    # Store to keep current dataset metadata (name, source, etc.)
    data_store = dcc.Store(
        id=f"{id_prefix}-data-store",
        data={
            "name": None,
            "source": None,  # "simulated", "uploaded", "preset"
        },
    )

    # Layout: dataset controls row + slider + modal + store
    return dbc.Container(
        children=[
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Label("Dataset:"),
                            data_label,
                            add_data_button,
                        ],
                        width="auto",
                    )
                ],
                className="mb-3",
            ),
            animation_slider,
            data_modal,
            data_store,
        ],
        fluid=True,
        className=container_class,
    )