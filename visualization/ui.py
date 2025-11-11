# visualization/ui.py
from dash import html, dcc
import dash_bootstrap_components as dbc
import interaction.ui_controls as ui_controls
from interaction.ui_controls import create_slider, create_thesh_component

def create_layout(n_mat, initial_fig):
    """Create the main layout with visualization type dropdown instead of tabs."""
    animation_slider = create_slider(id="mat-idx", n_frames=n_mat, label="Connectivity Matrix Index")
    threshold_comp = create_thesh_component(id="thresh-comp", label="Threshold")
    
    # Visualization type dropdown
    viz_type_dropdown = html.Div(
        [
            dbc.Label("Visualization Type"),
            dcc.Dropdown(
                id="viz-type-dropdown",
                options=[
                    {"label": "2D", "value": "2D"},
                    {"label": "3D", "value": "3D"},
                    {"label": "Heatmap", "value": "Heatmap"},
                ],
                value="2D",
                clearable=False,
            )
        ],
        className="mb-3",
    )

    return dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H3("Controls / Info Panel", className="mb-3"),
                            animation_slider,
                            threshold_comp,
                            
                        ],
                        width=3,
                        className="bg-light p-3 m-3 rounded shadow-sm",
                    ),
                    dbc.Col(
                        [
                            html.H3("Brain Connectivity Visualization", className="mb-3"),
                            viz_type_dropdown,
                            dcc.Graph(
                                id="main-visualization",
                                figure=initial_fig,
                                style={"height": "70vh"},
                            )
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
