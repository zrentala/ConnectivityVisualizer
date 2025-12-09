# data_loader.py

from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd

from dash import no_update
from scipy.io import loadmat
from data.simulation import *
from utils.braindata import BrainData
from utils.global_app_state import GlobalAppState
from visualization.vizuimanager import VizUIManager
import mne

# You can move / edit these presets here instead of in callbacks.py
PRESET_CONFIGS: Dict[str, Dict[str, Any]] = {
    "small_undirected": {"n_elec": 10, "directed": False, "n_mat": 5},
    "medium_directed": {"n_elec": 20, "directed": True, "n_mat": 10},
    "large_undirected": {"n_elec": 64, "directed": False, "n_mat": 20},
}


@dataclass
class Meta:
    name: str
    source: str
    extra: dict

@dataclass
class SliderMeta:
    max_idx: int
    marks: dict
    value: int = 0

class DataLoader:
    """
    Factory that assembles a BrainData instance.
    Does NOT modify global state.
    Returns: (BrainData, Meta, SliderMeta)
    """

    def __init__(self, preset_configs=None):
        self.preset_configs = PRESET_CONFIGS

    # ============================================================
    #  Public Entry Point
    # ============================================================

    def build_braindata(
        self,
        fc_source: dict,
        loc_source: dict,
        directed: bool,
        brain_mesh
    ):
        """
        fc_source: {"type": "sim" | "preset" | "upload", ...}
        loc_source: same
        directed: bool
        """

        # ---- Load FC ----
        conn_mat, fc_meta, slider_meta = self._load_fc(fc_source)

        # ---- Load LOC ----
        chanlocs, loc_meta = self._load_locs(loc_source)

        # ---- Create BrainData instance ----
        bd = BrainData(
            conn_mat=conn_mat,
            chanlocs=chanlocs,
            brain_mesh=brain_mesh,
            directed=directed,
        )

        # combine metadata
        combined_meta = {
            "fc": fc_meta,
            "loc": loc_meta,
            "directed": directed,
        }

        return bd, combined_meta, slider_meta

    # ============================================================
    #  FC ROUTING
    # ============================================================

    def _load_fc(self, cfg):
        t = cfg["type"]

        if t == "sim":
            return self._fc_sim(cfg)
        elif t == "preset":
            return self._fc_preset(cfg)
        elif t == "upload":
            return self._fc_upload(cfg)
        else:
            raise ValueError(f"Unknown FC type: {t}")

    def _fc_sim(self, cfg):
        n_mat = cfg["n_mat"]
        n_elec = cfg["n_elec"]
        directed = cfg["directed"]

        conn = generate_conn(n_mat, n_elec, directed)

        meta = Meta(
            name=f"Simulated FC ({n_elec}x{n_elec}, {n_mat} mats)",
            source="sim",
            extra={"n_mat": n_mat, "n_elec": n_elec, "directed": directed},
        )

        slider = SliderMeta(
            max_idx=n_mat - 1,
            marks={i: str(i) for i in range(n_mat)},
            value=0,
        )

        return conn, meta, slider

    def _fc_preset(self, cfg):
        name = cfg["name"]
        preset = self.preset_configs[name]

        conn = generate_conn(
            preset["n_mat"], preset["n_elec"], preset["directed"]
        )

        meta = Meta(
            name=f"{name} (preset FC)",
            source="preset",
            extra=preset,
        )

        slider = SliderMeta(
            max_idx=preset["n_mat"] - 1,
            marks={i: str(i) for i in range(preset["n_mat"])},
            value=0,
        )

        return conn, meta, slider

    def _fc_upload(self, cfg):
        contents = cfg["contents"]
        filename = cfg["filename"]

        buf = decode_uploaded(contents, filename)
        arr = load_connectivity(buf)

        # Ensure 3D format
        if arr.ndim == 2:
            arr = arr[np.newaxis, ...]

        n_mat, n_elec, _ = arr.shape

        meta = Meta(
            name=f"{filename} (uploaded FC)",
            source="upload",
            extra={"n_mat": n_mat, "n_elec": n_elec},
        )

        slider = SliderMeta(
            max_idx=n_mat - 1,
            marks={i: str(i) for i in range(n_mat)},
            value=0,
        )

        return arr, meta, slider

    # ============================================================
    #  LOCATION ROUTING
    # ============================================================

    def _load_locs(self, cfg):
        t = cfg["type"]

        if t == "sim":
            return self._loc_sim(cfg)
        elif t == "preset":
            return self._loc_preset(cfg)
        elif t == "upload":
            return self._loc_upload(cfg)
        else:
            raise ValueError(f"Unknown loc type: {t}")

    def _loc_sim(self, cfg):
        sim = Simulation(cfg["sim_cfg"])
        df = load_locs_simulated(sim)

        meta = Meta(
            name="Simulated locations",
            source="sim",
            extra={"n_elec": len(df)},
        )

        return df, meta

    def _loc_preset(self, cfg):
        name = cfg["name"]
        df = load_locs_preset(name)

        meta = Meta(
            name=f"{name} (preset locs)",
            source="preset",
            extra={"n_elec": len(df)},
        )

        return df, meta

    def _loc_upload(self, cfg):
        contents = cfg["contents"]
        filename = cfg["filename"]

        buf = decode_uploaded(contents, filename)
        df = load_locs_input(buf)

        meta = Meta(
            name=f"{filename} (uploaded locs)",
            source="upload",
            extra={"n_elec": len(df)},
        )

        return df, meta



def decode_uploaded(contents, filename):
    content_type, b64data = contents.split(',')
    raw = base64.b64decode(b64data)
    buf = io.BytesIO(raw)
    buf.name = filename
    return buf


### HANDLE FC MATRIX

## right now use simualted but evently will connect to real data
def load_conn_mat_preset(kind: str):
    cfg = PRESET_CONFIGS[kind]
    return generate_conn(cfg["n_mat", cfg['n_elec']], cfg['directed'])

## 
def load_conn_mat_sim(cfg):
    keys = ['n_mat', 'n_elec', 'directed']
    if not all(k in cfg for k in keys):
        KeyError(f"Config input: {cfg} has an incorrect key")
    return generate_conn(cfg["n_mat", cfg['n_elec']], cfg['directed'])

##  from input
# ---------------------------
def load_connectivity(path: str | Path) -> np.ndarray:
    """Load connectivity matrix from .npy, .npz, .csv, or .mat"""
    path = Path(path)
    ext = path.suffix.lower()

    # -------- .npy --------
    if ext == ".npy":
        return np.load(path)

    # -------- .npz --------
    elif ext == ".npz":
        data = np.load(path)
        if "conn" in data:
            return data["conn"]
        # fallback: return first ndarray
        for key in data:
            if isinstance(data[key], np.ndarray):
                return data[key]
        raise KeyError("No valid array found in .npz file.")

    # -------- .csv --------
    elif ext == ".csv":
        arr = np.loadtxt(path, delimiter=",")
        # If it's a flat 2D matrix, wrap into (1, n, n)
        if arr.ndim == 2:
            return arr[np.newaxis, ...]
        return arr

    # -------- .mat --------
    elif ext == ".mat":
        data = loadmat(path)
        for key, val in data.items():
            if not key.startswith("__") and isinstance(val, np.ndarray):
                return val
        raise KeyError("No valid matrix found in .mat file.")

    else:
        raise ValueError(f"Unsupported file type: {ext}")



### HANDLE LOCS
def locs_dict_to_dataframe(pos: dict):
    """
    Convert a dict {label: [x, y, z]} in meters to a standardized
    Pandas DataFrame in millimeters.
    """
    return pd.DataFrame({
        "label": list(pos.keys()),
        "x": [coord[0] * 1000 for coord in pos.values()],
        "y": [coord[1] * 1000 for coord in pos.values()],
        "z": [coord[2] * 1000 for coord in pos.values()],
    })

def load_locs_preset(kind: str):
    montage = mne.channels.make_standard_montage(kind)
    pos = montage.get_positions()["ch_pos"]
    return locs_dict_to_dataframe(pos)
    
def load_locs_simulated(sim: Simulation):
    pos = sim.get_sensor_positions()  # dict like {"Fz":[x,y,z], ...}
    return locs_dict_to_dataframe(pos)


def load_locs_input(path):
    path = str(path).lower()

    # ---------------------------
    # EEG/standard montage formats
    # ---------------------------
    if path.endswith(".set"):
        raw = mne.io.read_raw_eeglab(path, preload=False)
        montage = raw.get_montage()

    elif path.endswith(".vhdr"):
        raw = mne.io.read_raw_brainvision(path, preload=False)
        montage = raw.get_montage()

    elif path.endswith((".bdf", ".edf")):
        raw = mne.io.read_raw(path, preload=False)
        montage = raw.get_montage()

    # ---------------------------
    # CSV / TXT custom coordinates
    # ---------------------------
    elif path.endswith((".csv", ".txt")):
        df = pd.read_csv(path)

        required = {"label", "x", "y", "z"}
        if not required.issubset(df.columns):
            raise ValueError(
                f"CSV must contain columns: {required}, but has: {set(df.columns)}"
            )

        # Convert DataFrame → dict({label: [x,y,z]})
        pos = {
            row["label"]: np.array([row["x"], row["y"], row["z"]], dtype=float)
            for _, row in df.iterrows()
        }

        return locs_dict_to_dataframe(pos)

    # ---------------------------
    # Other montage files (ced, locs, elc, etc.)
    # ---------------------------
    elif path.endswith((".ced", ".locs", ".elc", ".elp", ".sfp")):
        montage = mne.channels.read_custom_montage(path)

    else:
        raise ValueError(f"Unsupported channel location format: {path}")

    # --------------------------------------------------------
    # Extract MNE montage → dictionary → DataFrame conversion
    # --------------------------------------------------------
    pos = montage.get_positions()["ch_pos"]  # {label: np.array([x,y,z])}
    return locs_dict_to_dataframe(pos)
