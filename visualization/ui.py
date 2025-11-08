# # visualization/ui.py
# from dash import html, dcc
# import dash_bootstrap_components as dbc

# def create_layout(viz, n_mat: int, conn_cube_as_list):
#     """
#     viz: ConnectivityVisualizer (used only to build initial figures)
#     n_mat: number of matrices
#     conn_cube_as_list: pre-serialized list (n_mat, n, n) to drop into dcc.Store
#     """
#     # Hidden store holding the connectivity cube
#     store = dcc.Store(id="conn-cube", data=conn_cube_as_list)

#     # Slider to choose frame
#     frame_ctrl = html.Div(
#         [
#             dbc.Label("Frame"),
#             dcc.Slider(
#                 id="mat-idx",
#                 min=0, max=max(n_mat - 1, 0), step=1, value=0,
#                 updatemode="drag",
#                 tooltip={"placement": "bottom", "always_visible": True},
#                 marks={0: "0", n_mat - 1: str(n_mat - 1)} if n_mat > 1 else None,
#             ),
#             html.Small(id="frame-label", className="text-muted"),
#         ],
#         className="mb-3"
#     )

#     # Initial figures from the first matrix already loaded in `viz`
#     tab2d = dbc.Tab(
#         [
#             dcc.Graph(
#                 id="graph-2d",
#                 figure=viz.figure_2d(threshold=0.2, use_arcs=True, curvature=0.2, directed=True),
#                 style={"height": "80vh"}   # container scales; 2D stays square via scaleanchor in figure
#             )
#         ],
#         label="2D"
#     )

#     tab3d = dbc.Tab(
#         [
#             dcc.Graph(
#                 id="graph-3d",
#                 figure=viz.figure_3d(threshold=0.7, edge_style="arc", arc_radius=None),
#                 style={"height": "80vh"}
#             )
#         ],
#         label="3D"
#     )

#     tabs = dbc.Card(dbc.Tabs([tab3d, tab2d]))

#     return dbc.Container(
#         [
#             store,
#             dbc.Row(
#                 [
#                     dbc.Col(
#                         [
#                             html.H3("Controls / Info Panel", className="mb-3"),
#                             frame_ctrl,
#                         ],
#                         width=3,
#                         className="bg-light p-3 rounded shadow-sm",
#                     ),
#                     dbc.Col(
#                         [
#                             html.H3("Brain Connectivity Visualization", className="mb-3"),
#                             tabs,
#                         ],
#                         width=9,
#                         className="bg-light p-3 rounded shadow-sm",
#                     ),
#                 ],
#                 align="start",
#                 className="mt-3 g-3",   # Bootstrap gap utilities
#             ),
#         ],
#         fluid=True,
#     )

from dash import html, dcc
import dash_bootstrap_components as dbc

def create_layout(viz, n_mat, conn_cube_as_list):
    tabs = dbc.Card(
        dbc.Tabs([
            dbc.Tab([dcc.Graph(id="viz-2d", figure=viz.figure_2d(threshold=0.2, use_arcs=True, curvature=0.2, directed=True), style={"height": "90vh"})],
                    label="2D Visualization"),
            dbc.Tab([dcc.Graph(id="viz-3d", figure=viz.figure_3d(threshold=0.7, edge_style="arc", arc_radius=None), style={"height": "90vh"})],
                    label="3D Visualization"),
        ])
    )

    return dbc.Container(
        [
            dcc.Store(id="conn-cube", data=conn_cube_as_list),  # <-- Store is in the layout
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H3("Controls / Info Panel", className="mb-3"),
                            html.Div([
                                dbc.Label("Frame"),
                                dcc.Slider(
                                    id="mat-idx", min=0, max=max(n_mat - 1, 0), step=1, value=0,
                                    updatemode="drag",
                                    tooltip={"placement": "bottom", "always_visible": True},
                                    marks={0: "0", n_mat - 1: str(n_mat - 1)} if n_mat > 1 else None,
                                ),
                                html.Small(id="frame-label", className="text-muted"),
                            ])
                        ],
                        width=3, className="bg-light p-3 rounded shadow-sm",
                    ),
                    dbc.Col(
                        [
                            html.H3("Brain Connectivity Visualization", className="mb-3"),
                            tabs,
                        ],
                        width=9, className="bg-light p-3 rounded shadow-sm",
                    ),
                ],
                className="mt-3 p-3",
            ),
        ],
        fluid=True,
    )

