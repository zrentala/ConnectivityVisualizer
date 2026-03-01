from __future__ import annotations
import os
import numpy as np
import pandas as pd
import pyvista as pv
from dash import Dash
import dash_bootstrap_components as dbc

from interaction.callbacks import register_callbacks
from visualization.ui import create_layout
from utils.global_app_state import GlobalAppState
from analysis.threshold import Threshold


# -------------------------
# Global state
# -------------------------
global_state = GlobalAppState()

# -------------------------
# Dash app
# -------------------------
app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG],
    suppress_callback_exceptions=True,  # safer for multi-page / dynamic layouts
)

server = app.server  # Required for Hugging Face

app.layout = create_layout()

# Register callbacks
register_callbacks(app, global_state)


# -------------------------
# Run (HF-compatible)
# -------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))

    app.run(
        host="0.0.0.0",   # REQUIRED for HF
        port=port,
        debug=False,      # NEVER True on HF
        use_reloader=False
    )