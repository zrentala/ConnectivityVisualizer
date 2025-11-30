from dash import html, dcc
import dash_bootstrap_components as dbc
from dataclasses import dataclass
from typing import Optional

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

        alpha_slider = create_slider(id='thresh-stat-alpha-slider', data_min=0, data_max=10, step=0.1, label="Alpha Level (%)")
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


def create_2d_options() -> html.Div:
    node_size_slider = create_slider(id="viz-2d-node_size-slider", data_min=5, data_max=50, step=1, label="Node Size")
    node_opacity_slider = create_slider(id="viz-2d-node_opacity-slider", data_min=0, data_max=1, step=0.01, label="Node Opacity (%)")

    edge_width_range = create_range_slider(id="viz-2d-edge_width-range_slider", data_min=0, data_max=10, step=0.1, default=[0.4, 5.0], label="Edge Width Size")
    edge_opacity_slider = create_slider(id="viz-2d-edge_opacity-slider", data_min=0, data_max=1, step=0.01, label="Edge Opacity (%)")
    return html.Div(
        id="viz-2d-container",
        children=[
            html.Hr(),
            html.H5("2D Visualization Options"),
            node_size_slider,
            # node_opacity_slider,
            edge_width_range,
            edge_opacity_slider
        ],
        style={"display": "none"},
    )

def create_3d_options() -> html.Div:
    node_size_slider = create_slider(id="viz-3d-node_size-slider", data_min=1, data_max=50, step=1, label="Node Size")
    node_opacity_slider = create_slider(id="viz-3d-node_opacity-slider", data_min=0, data_max=1, step=0.01, label="Node Opacity (%)")

    edge_opacity_slider = create_slider(id="viz-3d-edge_opacity-slider", data_min=0, data_max=1, step=0.01, label="Edge Opacity (%)")
    edge_width_range = create_range_slider("viz-3d-edge_width-range_slider", data_min=0, data_max=10, step=0.1, default=[0.4, 5.0],label="Edge Width Size")

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
            node_size_slider,
            # node_opacity_slider,
            edge_width_range,
            edge_opacity_slider,
            html.Br(),
            hemisphere_row,
            brain_mesh_opacity_slider
        ],
        style={"display": "none"},
    )


def create_viz_controls(n_mat: int) -> html.Div:

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
    options_2d = create_2d_options()
    options_3d = create_3d_options()

    return dbc.Container(
        children=[
            viz_type_dropdown,
            color_map_dropdown,
            color_range,
            options_2d,
            options_3d,
        ],
        fluid=True,
        className=container_class,
    )



def create_data_component(n_mat: int) -> html.Div:
    """Create data selection component with add/load dataset controls."""
    # Slider over connectivity matrices
    animation_slider = create_slider(
        id="data-conn_idx-slider",
        data_max=n_mat-1,
        data_min=0,
        step=1,
        label="Connectivity Matrix Index",
        default=0,
    )

    # Label that shows either "No dataset loaded" or current dataset name
    data_label = html.Span(
        id="data-dataset-label",
        children="No dataset loaded",
        className="ms-2",
    )

    # "+" button to add / replace data
    add_data_button = dbc.Button(
        "+",
        id="data-add_dataset-button",
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
                        id="data-modal-upload",
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
                    create_dropdown(id="data-modal-dataset_preset-dropdown", options=[
                            {"label": "Small undirected (n=10, mats=5)", "value": "small_undirected"},
                            {"label": "Medium directed (n=20, mats=10)", "value": "medium_directed"},
                            {"label": "Large undirected (n=64, mats=20)", "value": "large_undirected"},
                        ],
                        clearable=True),

                    html.Hr(),

                    # 3) Generate your own
                    html.H5("3. Generate simulated data"),
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dbc.Label("Number of nodes"),
                                    dbc.Input(
                                        # id=f"{id_prefix}-gen-n-elec",
                                        id="data-modal-gen_n_elec-input",
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
                                        id="data-modal-gen_n_mats-input",
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
                                    dbc.Label("Directed? "),
                                    dbc.Checkbox(
                                        # id=f"{id_prefix}-gen-directed",
                                        id="data-modal-gen_directed-checkbox",
                                        value=False,
                                    ),
                                ],
                                md=4,
                                className="d-flex align-items-end",
                            ),
                        ],
                        className="mb-2",
                    ),
                    dbc.Button(
                        "Generate",
                        # id=f"{id_prefix}-gen-btn",
                        id="data-modal-gen-button",
                        color="secondary",
                        className="mt-2",
                        n_clicks=0,
                    ),
                ]
            ),
            dbc.ModalFooter(
                dbc.Button(
                    "Close",
                    id="data-modal-close-button",
                    # id=f"{id_prefix}-data-modal-close",
                    className="ms-auto",
                    n_clicks=0,
                )
            ),
        ],
        id="data-modal",
        is_open=False,
        centered=True,
        backdrop="static",
    )

    # Store to keep current dataset metadata (name, source, etc.)
    data_store = dcc.Store(
        # id=f"{id_prefix}-data-store",
        id="data-store",
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