# connectivity_visualizer/io.py

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.io import loadmat, savemat

# ---------------------------
# CONFIG FILES (.json)
# ---------------------------
def load_config(path: str | Path) -> dict:
    """Load a JSON configuration file."""
    path = Path(path)
    with open(path, "r") as f:
        return json.load(f)


def save_config(path: str | Path, config: dict):
    """Save a configuration dictionary to JSON."""
    path = Path(path)
    with open(path, "w") as f:
        json.dump(config, f, indent=4)


# ---------------------------
# CONNECTIVITY MATRICES (.npy, .csv, .mat)
# ---------------------------
def load_connectivity(path: str | Path) -> np.ndarray:
    """Load connectivity matrix from .npy, .csv, or .mat"""
    path = Path(path)
    if path.suffix == ".npy":
        return np.load(path)
    elif path.suffix == ".csv":
        return np.loadtxt(path, delimiter=",")
    elif path.suffix == ".mat":
        data = loadmat(path)
        # assume variable is named 'conn_matrix' or similar
        for key in data:
            if not key.startswith("__") and isinstance(data[key], np.ndarray):
                return data[key]
        raise KeyError("No valid matrix found in .mat file.")
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")


def save_connectivity(path: str | Path, matrix: np.ndarray):
    """Save connectivity matrix to .npy, .csv, or .mat"""
    path = Path(path)
    if path.suffix == ".npy":
        np.save(path, matrix)
    elif path.suffix == ".csv":
        np.savetxt(path, matrix, delimiter=",")
    elif path.suffix == ".mat":
        savemat(path, {"conn_matrix": matrix})
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")


# ---------------------------
# ELECTRODE LOCATIONS
# ---------------------------
def save_locations(path: str | Path, locations: np.ndarray):
    """Save electrode 3D coordinates (x, y, z)."""
    path = Path(path)
    df = pd.DataFrame(locations, columns=["x", "y", "z"])
    df.to_csv(path, index=False)


def load_locations(path: str | Path) -> np.ndarray:
    """Load electrode coordinates."""
    path = Path(path)
    df = pd.read_csv(path)
    return df.to_numpy()
