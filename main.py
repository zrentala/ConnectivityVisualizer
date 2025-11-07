from dash import Dash
import dash_bootstrap_components as dbc
from visualization.ui import create_layout
from data.simulation import Simulation
from analysis.graph import GraphAnalysis
from visualization.brainvisualizer import visualize_connectivity_plotly
import pandas as pd

# ✅ Include Bootstrap stylesheet
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Example simulated data
sim_data = Simulation({'n_elec': 20, 'directed': False})
graph_analysis = GraphAnalysis(sim_data.conn_matrix)
graph_analysis.summary()

locs = pd.DataFrame({
    'label': [f'E{i}' for i in range(sim_data.n_elec)],
    'x': sim_data.locations[:, 0] * 100,
    'y': sim_data.locations[:, 1] * 100,
    'z': sim_data.locations[:, 2] * 100
})

fig = visualize_connectivity_plotly(sim_data.conn_matrix, locs, threshold=0.75)

app.layout = create_layout(fig)

if __name__ == "__main__":
    app.run(debug=True)
