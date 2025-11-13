# visualization/ui.py
from dash import html, dcc
import dash_bootstrap_components as dbc
from interaction.ui_controls import create_slider, create_thesh_component, create_viz_controls
import dash_split_pane as dsp


def create_layout(n_mat, initial_fig):
    """Responsive layout that fits the viewport and keeps margins/padding (no fixed pixels)."""
    animation_slider = create_slider(id="mat-idx", n_frames=n_mat, label="Connectivity Matrix Index")
    threshold_comp = create_thesh_component(id="thresh-comp", label="Threshold")
    viz_controls = create_viz_controls(id_prefix="main", n_mat=n_mat)

    left = html.Div(
        [
            html.H3("Controls / Info Panel", className="mb-3"),
            html.H4("Visualization Controls"),
            viz_controls,
            animation_slider,
            html.H4("Threshold Controls"),
            threshold_comp,
        ],
        className="bg-light p-3 rounded shadow-sm",
        style={"height": "100%", "overflowY": "auto"},
    )

    right = html.Div(
        [
            html.H3("Brain Connectivity Visualization", className="mb-3"),
            dcc.Graph(
                id="main-visualization",
                figure=initial_fig,
                className="main-graph",
                # useResizeHandler=True,
                config={"responsive": True},
                style={"height": "100%", "width": "100%", "flex": "1 1 auto", "minHeight": 0},
            ),
        ],
        className="bg-light p-3 rounded shadow-sm",
        style={"height": "100%", "overflow": "hidden", "display": "flex", "flexDirection": "column", "flex": "1 1 auto"},
    )

    split_pane = dsp.DashSplitPane(
        id="main-split",
        children=[left, right],
        split="vertical",
        size="30%",
        minSize="20%",
        maxSize="70%",
        primary="first",
        allowResize=True,
        style={
            "position": "relative",
            "height": "90%",
            "width": "90%",
        },
        pane1Style={"height": "100%"},
        pane2Style={"height": "100%"},
        resizerStyle={"cursor": "col-resize"},
        className="rounded shadow-sm p-3 m-3 bg-light"
    )

    # The container itself uses viewport units — no fixed pixels
    return dbc.Container(
        split_pane,
        fluid=True,
        style={
            "height": "100vh",
            "width": "100vw",
            "overflow": "hidden",
            "display": "flex",
            "flexDirection": "column",
            "margin": "0",
            "alignItems": "center",
            "justifyContent": "center",
        },
    )
