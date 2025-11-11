# visualization/ui.py
from dash import html, dcc
import dash_bootstrap_components as dbc
import interaction.ui_controls as ui_controls
from interaction.ui_controls import VizTabsBuilder, create_slider

viz_tabs_builder = VizTabsBuilder()


def create_layout(n_mat, initial_figs):
    tabs = viz_tabs_builder.create_viz_tabs(initial_figs)
    slider = create_slider(id="mat-idx", n_frames=n_mat, label="Connectivity Matrix Index")

    return dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H3("Controls / Info Panel", className="mb-3"),
                            slider,
                        ],
                        width=3,
                        className="bg-light p-3 m-3 rounded shadow-sm",
                    ),
                    dbc.Col(
                        [
                            html.H3("Brain Connectivity Visualization", className="mb-3"),
                            tabs,
                        ],
                        width=8,
                        className="bg-light p-3 m-3 rounded shadow-sm",
                    ),
                ],
                className="mt-3 p-3 justify-content-center",
            ),
        ],
        fluid=True,
    )
