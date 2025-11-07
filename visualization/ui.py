# visualization/ui.py
from dash import html, dcc
import dash_bootstrap_components as dbc

def create_layout(fig):
    return dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H3("Controls / Info Panel", className="mb-3"),
                            html.P("Adjust settings here."),
                        ],
                        width=3,
                        className="bg-light p-3 rounded shadow-sm",
                    ),
                    dbc.Col(
                        [
                            dcc.Graph(id="brain-3d", figure=fig, style={"height": "90vh"}),
                        ],
                        width=9,
                    ),
                ],
                align="start",  # optional: aligns columns vertically
                className="mt-3 g-3",
            ),
        ],
        fluid=True,
    )
