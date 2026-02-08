# # app.py
# from __future__ import annotations
# import numpy as np
# import pandas as pd
# import pyvista as pv
# from dash import Dash
# import dash_bootstrap_components as dbc
# from interaction.callbacks import register_callbacks
# from visualization.ui import create_layout

# from utils.global_app_state import GlobalAppState
# from analysis.threshold import Threshold
# global_state = GlobalAppState()

# app = Dash(__name__, external_stylesheets=[dbc.themes.CYBORG], suppress_callback_exceptions=True)
# app.layout = create_layout()


# # Register callbacks
# register_callbacks(app, global_state)

# if __name__ == "__main__":
#     app.run(debug=True)

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

global_state = GlobalAppState()

app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG],
)

server = app.server  # <-- expose Flask server explicitly

app.layout = create_layout()

# Register callbacks
register_callbacks(app, global_state)

if __name__ == "__main__":
    app.run(debug=True)
