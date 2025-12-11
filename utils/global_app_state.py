from data.simulation import *
import dash_bootstrap_components as dbc
from visualization.vizuimanager import VizUIManager
from dash import Dash
import pandas as pd
from utils.braindata import BrainData
from analysis.threshold import Threshold
from visualization.ui import create_layout
import mne

class GlobalAppState:
    """Global application state - stores data and configuration."""
    

    def __init__(self):
        self.app = Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
        self.brain_data = None       # No data yet
        self.threshold = None
        self.viz = None

    
        
        # # ---- Simulated data ----
        # cfg = {"n_elec": 65, "directed": False, "n_mat": 10}

        # data = Simulation(cfg)
        # montage = mne.channels.make_standard_montage("standard_alphabetic")
        # # Extract channel positions (in meters)
        # pos = montage.get_positions()["ch_pos"]  # dict: {label: [x,y,z]}
        # # print(pos)
        # # Convert into DataFrame
        # chanlocs = pd.DataFrame(
        #     {
        #         "label": list(pos.keys()),
        #         "x": [coord[0] * 1000 for coord in pos.values()],  # convert to cm to match your scale
        #         "y": [coord[1]  * 1000 for coord in pos.values()],
        #         "z": [coord[2] * 1000 for coord in pos.values()],
        #     }
        # )
        # del montage
        # # chanlocs = pd.DataFrame(
        # #     {
        # #         "label": [f"E{i}" for i in range(data.n_elec)],
        # #         "x": data.locations[:, 0] * 100,
        # #         "y": data.locations[:, 1] * 100,
        # #         "z": data.locations[:, 2] * 100,
        # #     }
        # # )

        # brain_mesh = build_brain_mesh()
        # self.brain_data = BrainData(data.conn_matrices, chanlocs, brain_mesh, directed=False)
        # # print( self.brain_data)
        # self.threshold = Threshold()
        # self.viz = VizUIManager(self.brain_data, self.threshold)
        # print(self.viz)
        