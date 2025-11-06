# main.py
import numpy as np
import pandas as pd
import pyvista as pv
import mne
from analysis.graph import GraphAnalysis
from data.simulation import Simulation 

def main():
    sim_data = Simulation({'n_elec': 20, 'directed': False})
    graph_analysis = GraphAnalysis(sim_data.conn_matrix)
    graph_analysis.summary()
    print(sim_data.conn_matrix)
    print(sim_data.locations)
    print(sim_data.n_elec)
    print(sim_data.is_directed)


if __name__ == "__main__":
    main()