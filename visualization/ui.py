# visualization/ui.py
from dash import html, dcc
import dash_bootstrap_components as dbc
from interaction.ui_controls import create_stat_component, create_thresh_component, create_viz_controls, create_data_component
import dash_split_pane as dsp
from plotly import graph_objs as go


def create_layout():
    """Responsive layout that fits the viewport and keeps margins/padding (no fixed pixels)."""
    threshold_comp = create_thresh_component()
    viz_controls = create_viz_controls()
    data_component = create_data_component()
    stat_component = create_stat_component()
    initial_fig = None

    left = html.Div(
        [
            html.H4("Data Controls"),
            data_component,
            html.H4("Visualization Controls"),
            viz_controls,
            html.H4("Threshold Controls"),
            threshold_comp,
            # html.H4("Graph Controls"),
        ],
        className="bg-light p-3 rounded shadow-sm",
        style={"height": "100%", "overflowY": "auto"},
    )

    right = html.Div(
        [
            html.Div(
                [
                    dcc.Graph(
                        id="split-right-fig",
                        figure=initial_fig,
                        config={"responsive": True},
                        style={
                            "height": "100%",
                            "width": "100%",
                            "minHeight": 0,
                            "flex": "1 1 auto",
                        },
                    ),
                ],
                id="right-figure-container",
                style={"height": "100%", "flex": "1 1 auto", "minWidth": 0},
                className="m-3"
            ),

            html.Div(
                stat_component,
                id="right-stats-container",

            ),
        ],
        style={
            "height": "100%",
            "display": "flex",
            "flexDirection": "row",
            "overflow": "hidden",
        },
    )

    
    # fig_col = html.Div(
    #     [
    #         html.H3("Brain Connectivity Visualization", className="mb-3"),
    #         dcc.Graph(
    #             id="split-right-fig",
    #             figure=initial_fig,
    #             className="main-graph",
    #             # useResizeHandler=True,
    #             config={"responsive": True},
    #             style={"height": "100%", "width": "100%", "flex": "1 1 auto", "minHeight": 0},
    #         ),
    #     ]
    # )

    # stat_col = html.Div(
    #     [
    #         html.H3("Statistics", className="mb-3"),
    #         stat_component,  # <- your custom stat component
    #     ],
    #     className="bg-light p-3 rounded shadow-sm",
    #     style={
    #         "height": "100%",
    #         "overflowY": "auto",
    #         "display": "flex",
    #         "flexDirection": "column",
    #     },
    # )

    # right = dbc.Row(
    #     [
    #         dbc.Col(fig_col, width=3, style={"height": "100vh", "overflow": "hidden"}),
    #         dbc.Col(stat_col, width=9, style={"height": "100vh", "overflow": "hidden"}),

    #     ],
    #     className="bg-light p-3 rounded shadow-sm",
    #     style={"height": "100%", "overflow": "hidden", "display": "flex", "flexDirection": "column", "flex": "1 1 auto"},
    # )

    split_pane = dsp.DashSplitPane(
        id="split",
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
