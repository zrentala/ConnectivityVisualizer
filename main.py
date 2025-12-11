# app.py
from __future__ import annotations
import numpy as np
import pandas as pd
import pyvista as pv
from dash import Dash
import dash_bootstrap_components as dbc
from interaction.callbacks import register_callbacks
from visualization.ui import create_layout

from utils.global_app_state import GlobalAppState
from analysis.threshold import Threshold
        

def create_app(global_state: GlobalAppState) -> Dash:  
    """Create and initialize the Dash application."""
    app = Dash(__name__, external_stylesheets=[dbc.themes.CYBORG], suppress_callback_exceptions=True)
    app.layout = create_layout()
    

    # Register callbacks
    register_callbacks(app, global_state)

    return app


if __name__ == "__main__":
    global_state = GlobalAppState()
    app = create_app(global_state)
    app.run(debug=True)

