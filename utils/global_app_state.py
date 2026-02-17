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
        self.graph_analysis = None