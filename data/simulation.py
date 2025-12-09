# connectivity_visualizer/simulation.py
from pathlib import Path
import numpy as np
from utils import io # import your io module
import mne
import pyvista as pv

class Simulation:
    def __init__(self, cfg):
        self.n_elec = cfg.get('n_elec', 10)
        self.directed = cfg.get('directed', False)
        self.conn_matrices = generate_conn(n_mat=cfg.get('n_mat', 1), n_elec=self.n_elec, directed=self.directed)
        self.locations = generate_locs(self.n_elec)

    def save(self, folder="output"):
        """Save connectivity and location data."""
        folder = Path(folder)
        folder.mkdir(exist_ok=True)
        io.save_connectivity(folder / "conn.npy", self.conn_matrices)
        io.save_locations(folder / "locs.csv", self.locations)

def generate_conn(n_mat: int, n_elec: int, directed: bool) -> np.ndarray:
    conns = np.empty((n_mat, n_elec, n_elec), dtype=float)

    for i in range(n_mat):
        conn = np.random.rand(n_elec, n_elec)
        if not directed:
            conn = (conn + conn.T) / 2
        np.fill_diagonal(conn, 0)
        conns[i] = conn
    print(conns.shape)
    return conns

def generate_locs(n_elec: int):
    phi = np.random.uniform(0, np.pi * 2, n_elec)
    costheta = np.random.uniform(-1, 1, n_elec)
    theta = np.arccos(costheta)
    r = np.random.uniform(0, 1, n_elec) ** (1/3)

    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.column_stack((x, y, z))

def build_brain_mesh() -> pv.PolyData:
    fs_dir = mne.datasets.fetch_fsaverage(verbose=True)
    lh_pial = fs_dir / "surf" / "lh.pial"
    rh_pial = fs_dir / "surf" / "rh.pial"

    coords_lh, faces_lh = mne.read_surface(lh_pial)
    coords_rh, faces_rh = mne.read_surface(rh_pial)

    faces_lh_vtk = np.column_stack((np.full(len(faces_lh), 3), faces_lh))
    faces_rh_vtk = np.column_stack((np.full(len(faces_rh), 3), faces_rh + len(coords_lh)))

    coords_combined = np.vstack((coords_lh, coords_rh))
    faces_combined  = np.vstack((faces_lh_vtk, faces_rh_vtk))

    return pv.PolyData(coords_combined, faces_combined)
