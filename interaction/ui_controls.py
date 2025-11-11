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
                updatemode="drag",
                tooltip={"placement": "bottom", "always_visible": True},
                marks={0: "0", n_frames: str(n_frames - 1)} if n_frames > 1 else None,
            ),
        ],
        className="mb-3",
    )

def create_thesh_component(id: str, label: str = "Threshold") -> html.Div:
    """Create a threshold input component with optional slider."""
    return html.Div(
        [
            dbc.Label(label),
            dcc.Dropdown(
                id=id,
                options=[
                    {"label":"Basic", "value": "Basic"},
                    {"label":"MST", "value": "Minimum Spanning Tree"},
                ],
                value="Basic",
                clearable=False,
            ),
            html.Div(
                id=f"{id}-slider-container",
                children=[
                    create_slider(id=f"{id}-slider", n_frames=100, label="Threshold Value (%)")
                ],
                className="mt-2",
            )
        ],
        className="mb-3",
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