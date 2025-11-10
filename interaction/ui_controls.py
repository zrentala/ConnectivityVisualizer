from dash import html, dcc
import dash_bootstrap_components as dbc

@dataclass(frozen=True)
class VizConfig:
    graph_height: str = "40vh"
    id_prefix: str = "viz"


class VizTabsBuilder:
    def __init__(self, config: VizConfig | None = None):
        self.config = config or VizConfig()
        self._id_map: Dict[str, str] = {}

    def graph_id(self, label: str) -> str:
        key = label.lower().replace(" ", "-")
        return f"{self.config.id_prefix}-{key}"

    def create_viz_tabs(self, viz_dict: Dict[str, go.Figure]) -> dbc.Card:
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
    def id_map(self) -> Dict[str, str]:
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
                marks={0: "0", n_frames - 1: str(n_frames - 1)} if n_mat > 1 else None,
            ),
            html.Small(id="frame-label", className="text-muted"),
        ],
        className="mb-3",
    )