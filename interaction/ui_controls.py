from dash import html, dcc
import dash_bootstrap_components as dbc
from dataclasses import dataclass

# @dataclass(frozen=True)
# class VizConfig:
#     graph_height: str = "70vh"
#     id_prefix: str = "viz"


# class VizTabsBuilder:
#     def __init__(self, config: VizConfig | None = None):
#         self.config = config or VizConfig()
#         self._id_map = {}

#     def graph_id(self, label: str) -> str:
#         key = label.lower().replace(" ", "-")
#         return f"{self.config.id_prefix}-{key}"

#     def create_viz_tabs(self, viz_dict) -> dbc.Card:
#         """
#         viz_dict: {label: figure}
#         """
#         self._id_map.clear()
#         tabs = []

#         for label, fig in viz_dict.items():
#             gid = self.graph_id(label)
#             self._id_map[label] = gid

#             tabs.append(
#                 dbc.Tab(
#                     dcc.Graph(
#                         id=gid,
#                         figure=fig,
#                         style={"height": self.config.graph_height},
#                     ),
#                     label=label,
#                 )
#             )

#         return dbc.Card(dbc.Tabs(tabs))

#     @property
#     def id_map(self):
#         return dict(self._id_map)

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
    return dbc.Container(
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

def create_viz_controls(id_prefix: str, n_mat: int) -> html.Div:
    
    viz_type_options = [
        {"label": "2D", "value": "2D"},
        {"label": "3D", "value": "3D"},
        {"label": "Heatmap", "value": "Heatmap"},
    ]
    
    viz_type_dropdown = create_dropdown(
        id='viz-type-dropdown',
        options=viz_type_options,
        label="Visualization Type",
        default="2D",
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
        id=f'color-type-dropdown',
        options=color_map_options,
        label="Color Type",
        default="Viridis",
    )

    
    min_max_input_group = html.Div(
        [
            dbc.Label("Color Range (0-1)"),
            html.Div(
                dcc.RangeSlider(
                    id="conn-range",
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    value=[0.0, 1.0],
                    allowCross=False,
                    marks={0: "0.0", 0.5: "0.5", 1.0: "1.0"},
                    tooltip={"placement": "bottom", "always_visible": False},
                ),
            ),
        ],
    )

    return dbc.Container(
        children=[
            viz_type_dropdown,
            color_map_dropdown,
            min_max_input_group],
        fluid=True,
        className=container_class,
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