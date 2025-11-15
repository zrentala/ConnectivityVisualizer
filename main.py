# app.py
from __future__ import annotations
import numpy as np
import pandas as pd
import mne
import pyvista as pv
from dash import Dash
import dash_bootstrap_components as dbc

from visualization.brainvisualizer import ConnectivityVisualizer
from visualization.ui import create_layout
from data.simulation import Simulation
from interaction.callbacks import register_callbacks
from utils.global_app_state import GlobalAppState
        

def create_app(global_state: GlobalAppState) -> Dash:  
    """Create and initialize the Dash application."""
    app = Dash(__name__, external_stylesheets=[dbc.themes.CYBORG], suppress_callback_exceptions=True)
    n_mat = global_state.brain_data.conn_mat.shape[0]
    
    # Create initial 2D figure for display
    viz = ConnectivityVisualizer(
        global_state.brain_data
    )
    initial_fig = viz.get_figure(global_state.brain_data)
    
    app.layout = create_layout(
        initial_fig=initial_fig,
        n_mat=n_mat,
    )

    # Register callbacks
    register_callbacks(app, global_state)

    return app


if __name__ == "__main__":
    global_state = GlobalAppState()
    app = create_app(global_state)
    app.run(debug=True)
