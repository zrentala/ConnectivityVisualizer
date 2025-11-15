from dash import html, dcc
import dash_bootstrap_components as dbc
from dataclasses import dataclass

@dataclass(frozen=True)
class VizConfig:
    graph_height: str = "70vh"
    id_prefix: str = "viz"


class VizTabsBuilder:
    def __init__(self, config: VizConfig | None = None):
        self.config = config or VizConfig()
        self._id_map = {}

    def graph_id(self, label: str) -> str:
        key = label.lower().replace(" ", "-")
        return f"{self.config.id_prefix}-{key}"

    def create_viz_tabs(self, viz_dict) -> dbc.Card:
        """
        viz_dict: {label: figure}
        """
        self._id_map.clear()
        tabs = []

        for label, fig in viz_dict.items():
            gid = self.graph_id(label)
            self._id_map[label] = gid

            tabs.append(
                dbc.Tab(
                    dcc.Graph(
                        id=gid,
                        figure=fig,
                        style={"height": self.config.graph_height},
                    ),
                    label=label,
                )
            )

        return dbc.Card(dbc.Tabs(tabs))

    @property
    def id_map(self):
        return dict(self._id_map)

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
        className="m-3",
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
        className="p-3 my-3 rounded shadow-sm border border-dark",
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
        className="p-3 my-3 rounded shadow-sm border border-dark",
    )