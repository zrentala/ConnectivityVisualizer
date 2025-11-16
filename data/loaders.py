# data_loader.py

from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd

from dash import no_update

from data.simulation import Simulation
from utils.braindata import BrainData
from visualization.brainvisualizer import ConnectivityVisualizer
from utils.global_app_state import GlobalAppState


# You can move / edit these presets here instead of in callbacks.py
PRESET_CONFIGS: Dict[str, Dict[str, Any]] = {
    "small_undirected": {"n_elec": 10, "directed": False, "n_mat": 5},
    "medium_directed": {"n_elec": 20, "directed": True, "n_mat": 10},
    "large_undirected": {"n_elec": 64, "directed": False, "n_mat": 20},
}


@dataclass
class SliderState:
    max_idx: int
    marks: Dict[int, str]
    value: int


@dataclass
class DatasetMeta:
    name: str
    source: str   # "uploaded", "preset", "simulated_custom", etc.
    extra: Dict[str, Any]


class DataLoader:
    """
    Encapsulates all logic for constructing BrainData objects and
    updating GlobalAppState when datasets change.

    Responsibilities:
      - Build BrainData from simulated configs
      - Build BrainData from uploaded arrays
      - Build BrainData from preset configs
      - Provide slider metadata (max, marks, value)
      - Provide safe "initial" UI state based on current global_state
    """

    def __init__(
        self,
        global_state: GlobalAppState,
        preset_configs: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        self.global_state = global_state
        self.preset_configs = preset_configs or PRESET_CONFIGS

    # ---------- Internal helpers ----------

    def _brain_data_from_sim(self, cfg: Dict[str, Any]) -> BrainData:
        sim = Simulation(cfg)
        chanlocs = pd.DataFrame(
            {
                "label": [f"E{i}" for i in range(sim.n_elec)],
                "x": sim.locations[:, 0] * 100,
                "y": sim.locations[:, 1] * 100,
                "z": sim.locations[:, 2] * 100,
            }
        )
        brain_mesh = sim.build_brain_mesh()
        return BrainData(sim.conn_matrices, chanlocs, brain_mesh, directed=sim.is_directed)

    def _brain_data_from_uploaded_array(self, arr: np.ndarray) -> BrainData:
        """
        Build a BrainData object from a 3D connectivity array of shape (n_mat, n_elec, n_elec).
        """
        if arr.ndim != 3:
            raise ValueError("Uploaded data must be 3D: (n_mat, n_elec, n_elec)")

        n_mat, n_elec, _ = arr.shape
        angles = np.linspace(0, 2 * np.pi, n_elec, endpoint=False)
        x = np.cos(angles)
        y = np.sin(angles)
        z = np.zeros_like(x)

        chanlocs = pd.DataFrame(
            {
                "label": [f"E{i}" for i in range(n_elec)],
                "x": x * 100,
                "y": y * 100,
                "z": z,
            }
        )
        brain_mesh = None
        return BrainData(arr, chanlocs, brain_mesh, directed=False)

    def _slider_state_from_brain_data(
        self,
        brain_data: BrainData,
        current_value: int | None = 0,
    ) -> SliderState:
        """
        Given a BrainData, compute reasonable slider settings for matrix index.
        """
        conn = brain_data.conn_mat  # your BrainData uses .conn_mat
        n_frames = int(conn.shape[0])
        max_idx = max(n_frames - 1, 0)
        marks = {0: "0", max_idx: str(max_idx)} if n_frames > 1 else {0: "0"}
        value = min(current_value or 0, max_idx)
        return SliderState(max_idx=max_idx, marks=marks, value=value)

    # ---------- Public API used by callbacks ----------

    def initial_ui_state(
        self,
        store_data: Optional[Dict[str, Any]],
        slider_max: Optional[int],
        slider_marks: Optional[Dict[int, str]],
        slider_value: Optional[int],
    ) -> Tuple[str, Dict[str, Any] | None, SliderState]:
        """
        Compute initial label + slider state when the app / component first loads.
        """
        label = (store_data or {}).get("name") or "No dataset loaded"

        try:
            brain_data = self.global_state.brain_data
            slider = self._slider_state_from_brain_data(brain_data, slider_value)
        except Exception:
            # Fallback to whatever Dash already has
            max_idx = slider_max or 0
            marks = slider_marks or {0: "0"}
            value = slider_value or 0
            slider = SliderState(max_idx=max_idx, marks=marks, value=value)

        return label, store_data, slider

    def load_uploaded(
        self,
        contents: str,
        filename: Optional[str],
        previous_store: Optional[Dict[str, Any]],
    ) -> Tuple[DatasetMeta, SliderState]:
        """
        Handle option 1: "Load your own".
        Decodes Dash upload contents, builds BrainData, updates global_state.
        Raises ValueError on failure.
        """
        if contents is None:
            raise ValueError("No upload contents provided.")

        try:
            _, content_string = contents.split(",", 1)
        except ValueError:
            raise ValueError("Upload contents not in expected 'data:...;base64,...' format.")

        decoded = base64.b64decode(content_string)
        buffer = io.BytesIO(decoded)
        loaded = np.load(buffer, allow_pickle=True)

        if isinstance(loaded, np.lib.npyio.NpzFile):
            if "conn" not in loaded:
                raise ValueError("NPZ file must contain a 'conn' array.")
            conn = loaded["conn"]
        else:
            conn = loaded

        brain_data = self._brain_data_from_uploaded_array(conn)
        self.global_state.brain_data = brain_data
        self.global_state.viz = ConnectivityVisualizer(self.global_state.brain_data)

        slider = self._slider_state_from_brain_data(brain_data, current_value=0)
        ds_name = filename or "Uploaded dataset"
        meta = DatasetMeta(
            name=ds_name,
            source="uploaded",
            extra={},
        )
        return meta, slider

    def load_preset(
        self,
        preset_key: str,
    ) -> Tuple[DatasetMeta, SliderState]:
        """
        Handle option 2: "Choose from preset".
        """
        cfg = self.preset_configs.get(preset_key)
        if cfg is None:
            raise KeyError(f"Unknown preset key: {preset_key}")

        brain_data = self._brain_data_from_sim(cfg)
        self.global_state.brain_data = brain_data
        self.global_state.viz = ConnectivityVisualizer(self.global_state.brain_data)

        slider = self._slider_state_from_brain_data(brain_data, current_value=0)
        ds_name = f"Preset: {preset_key}"
        meta = DatasetMeta(
            name=ds_name,
            source="preset",
            extra={"preset": preset_key, "cfg": cfg},
        )
        return meta, slider

    def load_simulated_custom(
        self,
        n_elec: Optional[int],
        n_mat: Optional[int],
        directed: Optional[bool],
    ) -> Tuple[DatasetMeta, SliderState]:
        """
        Handle option 3: "Generate your own".
        Falls back to defaults if values are None.
        """
        n_elec = int(n_elec) if n_elec is not None else 20
        n_mat = int(n_mat) if n_mat is not None else 10
        directed_flag = bool(directed)

        cfg = {"n_elec": n_elec, "directed": directed_flag, "n_mat": n_mat}
        brain_data = self._brain_data_from_sim(cfg)
        self.global_state.brain_data = brain_data
        self.global_state.viz = ConnectivityVisualizer(self.global_state.brain_data)

        slider = self._slider_state_from_brain_data(brain_data, current_value=0)
        ds_name = f"Simulated (n={n_elec}, mats={n_mat}, directed={directed_flag})"
        meta = DatasetMeta(
            name=ds_name,
            source="simulated_custom",
            extra={"cfg": cfg},
        )
        return meta, slider
