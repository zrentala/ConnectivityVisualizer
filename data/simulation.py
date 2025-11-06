# connectivity_visualizer/simulation.py
from pathlib import Path
import numpy as np
from utils import io # import your io module

class Simulation:
    def __init__(self, cfg):
        self.n_elec = cfg.get('n_elec', 10)
        self.is_directed = cfg.get('directed', False)
        self.conn_matrix = self.generate_conn()
        self.locations = self.generate_locs()

    def generate_conn(self):
        conn = np.random.rand(self.n_elec, self.n_elec)
        if not self.is_directed:
            conn = (conn + conn.T) / 2
        np.fill_diagonal(conn, 0)
        return conn

    def generate_locs(self):
        phi = np.random.uniform(0, np.pi * 2, self.n_elec)
        costheta = np.random.uniform(-1, 1, self.n_elec)
        theta = np.arccos(costheta)
        r = np.random.uniform(0, 1, self.n_elec) ** (1/3)

        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        return np.column_stack((x, y, z))

    def save(self, folder="output"):
        """Save connectivity and location data."""
        folder = Path(folder)
        folder.mkdir(exist_ok=True)
        io.save_connectivity(folder / "conn.npy", self.conn_matrix)
        io.save_locations(folder / "locs.csv", self.locations)
