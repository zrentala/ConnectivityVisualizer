from data.simulation import Simulation
import dash_bootstrap_components as dbc
from visualization.brainvisualizer import ConnectivityVisualizer
from dash import Dash
import pandas as pd

class GlobalAppState:
    """Global application state - stores data and configuration."""

    def __init__(self):
        self.app = Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
        
        # ---- Simulated data ----
        cfg = {"n_elec": 10, "directed": False, "n_mat": 10}
        self.data = Simulation(cfg)
        
        self.chanlocs = pd.DataFrame(
            {
                "label": [f"E{i}" for i in range(self.data.n_elec)],
                "x": self.data.locations[:, 0] * 100,
                "y": self.data.locations[:, 1] * 100,
                "z": self.data.locations[:, 2] * 100,
            }
        )

        self.brain_mesh = self.data.build_brain_mesh()
        self.viz = ConnectivityVisualizer(self.data, self.chanlocs, self.brain_mesh)
        # self.thresh_type = "Basic"
        
        # No pre-computed figures - they are generated on demand in callbacks
