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
    "small_undirected": {"n_elec": 16, "directed": False, "n_mat": 5},
    "medium_directed": {"n_elec": 8, "directed": True, "n_mat": 10},
    "large_undirected": {"n_elec": 64, "directed": False, "n_mat": 20},
}

PRESET_LOCS: Dict[str, int] = {
    "standard_1005": 343,           # 10-05 system ~343 electrodes (EEG positions) :contentReference[oaicite:1]{index=1}
    "standard_1020": 94,            # 10-20 system ~94 electrodes :contentReference[oaicite:2]{index=2}
    "standard_alphabetic": 65,      # alphabetic labeling ~65 electrodes :contentReference[oaicite:3]{index=3}
    "standard_postfixed": 100,      # postfix intermed. ~100 electrodes :contentReference[oaicite:4]{index=4}
    "standard_prefixed": 74,        # prefix intermed. ~74 electrodes :contentReference[oaicite:5]{index=5}
    "standard_primed": 100,         # primed ~100 electrodes :contentReference[oaicite:6]{index=6}

    "biosemi16": 16,                # BioSemi 16 channels :contentReference[oaicite:7]{index=7}
    "biosemi32": 32,                # BioSemi 32 channels :contentReference[oaicite:8]{index=8}
    "biosemi64": 64,                # BioSemi 64 channels :contentReference[oaicite:9]{index=9}
    "biosemi128": 128,              # BioSemi 128 channels :contentReference[oaicite:10]{index=10}
    "biosemi160": 160,              # BioSemi 160 channels :contentReference[oaicite:11]{index=11}
    "biosemi256": 256,              # BioSemi 256 channels :contentReference[oaicite:12]{index=12}

    "easycap-M1": 74,               # EasyCap M1 ~74 electrodes :contentReference[oaicite:13]{index=13}
    "easycap-M10": 61,              # EasyCap M10 ~61 electrodes :contentReference[oaicite:14]{index=14}
    "easycap-M43": 64,              # EasyCap M43 ~64 electrodes (MNE listing) :contentReference[oaicite:15]{index=15}

    "EGI_256": 256,                 # EGI Net 256 channels :contentReference[oaicite:16]{index=16}

    "GSN-HydroCel-32": 33,          # HydroCel 32 + Cz (~33) :contentReference[oaicite:17]{index=17}
    "GSN-HydroCel-64_1.0": 64,      # HydroCel 64 channels :contentReference[oaicite:18]{index=18}
    "GSN-HydroCel-65_1.0": 65,      # HydroCel 64 + Cz (~65) :contentReference[oaicite:19]{index=19}
    "GSN-HydroCel-128": 128,        # HydroCel 128 channels :contentReference[oaicite:20]{index=20}
    "GSN-HydroCel-129": 129,        # HydroCel 128 + Cz (~129) :contentReference[oaicite:21]{index=21}
    "GSN-HydroCel-256": 256,        # HydroCel 256 channels :contentReference[oaicite:22]{index=22}
    "GSN-HydroCel-257": 257,        # HydroCel 256 + Cz (~257) :contentReference[oaicite:23]{index=23}

    "mgh60": 60,                    # MGH 60 channels :contentReference[oaicite:24]{index=24}
    "mgh70": 70,                    # MGH 70 channels :contentReference[oaicite:25]{index=25}

    "artinis-octamon": 8            # Artinis OctaMon ~8 sources (not classic EEG) :contentReference[oaicite:26]{index=26}
}

PRESET_LOCS_REVERSED = {
    343: ["standard_1005"],
    94: ["standard_1020"],
    65: ["standard_alphabetic"],
    100: ["standard_postfixed", "standard_primed"],
    74: ["standard_prefixed", "easycap-M1"],
    16: ["biosemi16"],
    32: ["biosemi32"],
    64: ["biosemi64", "easycap-M43"],
    128: ["biosemi128", "GSN-HydroCel-128"],
    160: ["biosemi160"],
    256: ["biosemi256", "EGI_256", "GSN-HydroCel-256"],
    61: ["easycap-M10"],
    33: ["GSN-HydroCel-32"],
    65: ["GSN-HydroCel-65_1.0"],  # note: duplicates merged
    129: ["GSN-HydroCel-129"],
    257: ["GSN-HydroCel-257"],
    60: ["mgh60"],
    70: ["mgh70"],
    8: ["artinis-octamon"],
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
        brainmesh=None,
    ):
        """
        fc_source: {"type": "sim" | "preset" | "upload", ...}
        loc_source: same
        directed: bool
        """
        print(f"{fc_source=}")
        print(f"{loc_source=}")
        # ---- Load FC ----
        conn_mat, fc_meta, slider_meta = self._load_fc(fc_source)

        # ---- Load LOC ----
        chanlocs, loc_meta = self._load_locs(loc_source)

        if brainmesh is None:
            brain_mesh = build_brain_mesh()

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
    
    def make_fc_cfg(self, source, upload=None, preset=None, simulate=None):
        """
        Standardizes FC config so ALL return:
        {
            "type": ...,
            "n_elec": int,
            "n_mat": int,
            "directed": bool,
            "contents"/"filename"/"name" depending on source
        }
        """

        # ------------------------------
        # Upload
        # ------------------------------
        if source == "upload":
            contents, filename = upload
            buf = decode_uploaded(contents, filename)
            arr = load_connectivity(buf)

            # ensure 3D
            if arr.ndim == 2:
                arr = arr[np.newaxis, ...]
            n_mat, n_elec, _ = arr.shape

            return {
                "type": "upload",
                "contents": contents,
                "filename": filename,
                "n_elec": n_elec,
                "n_mat": n_mat,
                "directed": False,   # assume undirected unless user supplies otherwise
            }

        # ------------------------------
        # Preset
        # ------------------------------
        if source == "preset":
            cfg = self.preset_configs[preset]  # already validated outside
            return {
                "type": "preset",
                "name": preset,
                "n_elec": cfg["n_elec"],
                "n_mat": cfg["n_mat"],
                "directed": cfg["directed"],
            }

        # ------------------------------
        # Simulated
        # ------------------------------
        if source == "simulate":
            n_elec, n_mat = simulate
            return {
                "type": "sim",
                "n_elec": n_elec,
                "n_mat": n_mat,
                "directed": None,   # the callback will fill this in
            }

        raise ValueError(f"Unknown FC source: {source}")

    # def make_loc_cfg(self, source, upload=None, preset=None):
    #     if source == "upload":
    #         contents, filename = upload
    #         return {
    #             "type": "upload",
    #             "contents": contents,
    #             "filename": filename,
    #         }

    #     if source == "preset":
    #         return {"type": "preset", "name": preset}

    #     if source == "simulate":
    #         return {"type": "sim", "sim_cfg": {}}

    #     raise ValueError(f"Unknown LOC source: {source}")
    def make_loc_cfg(self, source, upload=None, preset=None, n_elec=None):
        if source == "upload":
            contents, filename = upload
            return {
                "type": "upload",
                "contents": contents,
                "filename": filename,
            }

        if source == "preset":
            return {"type": "preset", "name": preset}

        if source == "simulate":
            if n_elec is None:
                raise ValueError("LOC simulation requires n_elec for electrode count.")
            return {"type": "sim", "sim_cfg": {"n_elec": n_elec}}

        raise ValueError(f"Unknown LOC source: {source}")


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
    import io
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
def load_connectivity(source) -> np.ndarray:
    """
    Load connectivity matrix from:
    - file path (.npy, .npz, .csv, .mat)
    - file-like object (BytesIO)
    """

    # ---------------------------------------------------
    # If it's a file-like object (Dash upload)
    # ---------------------------------------------------
    if hasattr(source, "read"):
        name = getattr(source, "name", "")
        ext = Path(name).suffix.lower()

        if ext == ".npy":
            return np.load(source)

        elif ext == ".npz":
            data = np.load(source)
            if "conn" in data:
                return data["conn"]
            for key in data:
                if isinstance(data[key], np.ndarray):
                    return data[key]
            raise KeyError("No valid array found in .npz file.")

        elif ext == ".csv":
            source.seek(0)
            arr = np.loadtxt(source, delimiter=",")
            if arr.ndim == 2:
                return arr[np.newaxis, ...]
            return arr

        elif ext == ".mat":
            data = loadmat(source)
            for key, val in data.items():
                if not key.startswith("__") and isinstance(val, np.ndarray):
                    return val
            raise KeyError("No valid matrix found in .mat file.")

        else:
            raise ValueError(f"Unsupported file type: {ext}")

    # ---------------------------------------------------
    # Otherwise assume it's a path
    # ---------------------------------------------------
    path = Path(source)
    ext = path.suffix.lower()

    if ext == ".npy":
        return np.load(path)

    elif ext == ".npz":
        data = np.load(path)
        if "conn" in data:
            return data["conn"]
        for key in data:
            if isinstance(data[key], np.ndarray):
                return data[key]
        raise KeyError("No valid array found in .npz file.")

    elif ext == ".csv":
        arr = np.loadtxt(path, delimiter=",")
        if arr.ndim == 2:
            return arr[np.newaxis, ...]
        return arr

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
    pos = generate_locs(sim.n_elec)  # dict like {"Fz":[x,y,z], ...}
    return locs_dict_to_dataframe(pos)


def load_locs_input(source):
    """
    Load channel locations from:
    - file path
    - file-like object (BytesIO from Dash upload)
    """

    # ----------------------------------------------------
    # If it's a file-like object
    # ----------------------------------------------------
    if hasattr(source, "read"):
        name = getattr(source, "name", "")
        ext = Path(name).suffix.lower()

        # CSV / TXT
        if ext in (".csv", ".txt"):
            source.seek(0)
            df = pd.read_csv(source)

            required = {"label", "x", "y", "z"}
            if not required.issubset(df.columns):
                raise ValueError(
                    f"CSV must contain columns: {required}, but has: {set(df.columns)}"
                )

            return df

        # MNE-supported raw formats
        elif ext == ".set":
            raw = mne.io.read_raw_eeglab(source, preload=False)
            montage = raw.get_montage()

        elif ext == ".vhdr":
            raw = mne.io.read_raw_brainvision(source, preload=False)
            montage = raw.get_montage()

        elif ext in (".bdf", ".edf"):
            raw = mne.io.read_raw(source, preload=False)
            montage = raw.get_montage()

        elif ext in (".ced", ".locs", ".elc", ".elp", ".sfp"):
            montage = mne.channels.read_custom_montage(source)

        else:
            raise ValueError(f"Unsupported channel location format: {ext}")

        pos = montage.get_positions()["ch_pos"]
        return locs_dict_to_dataframe(pos)

    # ----------------------------------------------------
    # Otherwise assume it's a file path
    # ----------------------------------------------------
    path = Path(source)
    ext = path.suffix.lower()

    if ext == ".set":
        raw = mne.io.read_raw_eeglab(path, preload=False)
        montage = raw.get_montage()

    elif ext == ".vhdr":
        raw = mne.io.read_raw_brainvision(path, preload=False)
        montage = raw.get_montage()

    elif ext in (".bdf", ".edf"):
        raw = mne.io.read_raw(path, preload=False)
        montage = raw.get_montage()

    elif ext in (".ced", ".locs", ".elc", ".elp", ".sfp"):
        montage = mne.channels.read_custom_montage(path)

    elif ext in (".csv", ".txt"):
        df = pd.read_csv(path)

        required = {"label", "x", "y", "z"}
        if not required.issubset(df.columns):
            raise ValueError(
                f"CSV must contain columns: {required}, but has: {set(df.columns)}"
            )

        return df

    else:
        raise ValueError(f"Unsupported channel location format: {ext}")

    pos = montage.get_positions()["ch_pos"]
    return locs_dict_to_dataframe(pos)