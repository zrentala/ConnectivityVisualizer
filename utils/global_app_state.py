from data.simulation import Simulation
import dash_bootstrap_components as dbc
from visualization.brainvisualizer import ConnectivityVisualizer
from dash import Dash
import pandas as pd
from utils.braindata import BrainData

class GlobalAppState:
    """Global application state - stores data and configuration."""
    

    def __init__(self):
        self.app = Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
        
        # ---- Simulated data ----
        cfg = {"n_elec": 10, "directed": False, "n_mat": 10}
        data = Simulation(cfg)
        
        chanlocs = pd.DataFrame(
            {
                "label": [f"E{i}" for i in range(data.n_elec)],
                "x": data.locations[:, 0] * 100,
                "y": data.locations[:, 1] * 100,
                "z": data.locations[:, 2] * 100,
            }
        )

        brain_mesh = data.build_brain_mesh()
        self.brain_data = BrainData(data.conn_matrices, chanlocs, brain_mesh)
        self.viz = ConnectivityVisualizer(self.brain_data)
        
        
        # No pre-computed figures - they are generated on demand in callbacks
