from __future__ import annotations
from dataclasses import dataclass, field
from typing import Iterable, Optional, Tuple, Union, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.colors as plc

import analysis.threshold as thresh
from utils.braindata import BrainData
from analysis.threshold import Threshold
from visualization.vizhelpers import VizType, UpdateType

import visualization.vizhelpers as helpers

try:
    import pyvista as pv
except Exception:  # make pv optional
    pv = None




@dataclass
class Channel:
    x: float
    y: float
    label: Optional[str] = None
    # z is optional for 3D; if absent, zeros are assumed
    z: Optional[float] = None


class VizUIManager:
    def __init__(
        self,
        brain_data: BrainData,
        threshold: Threshold,
        conn_idx: int = 0,
        colorscale: str = "Viridis",
        color_min: float = 0.0,
        color_max: float = 1.0,

        show_labels: bool = True,
        viz_type: VizType = VizType.FIG2D,
    ) -> None:

        # -----------------------
        # Core settings
        # -----------------------
        self.conn_idx = conn_idx
        self.colorscale = colorscale
        self.color_min = color_min
        self.color_max = color_max
        self.viz_type = viz_type
        self.show_labels = show_labels

        # coordinates + labels (filled by update_xyz)
        self.xyz: np.ndarray = np.empty((0, 3), dtype=float)     # (n, 3)
        self.xy_topo: np.ndarray = np.empty((0, 2), dtype=float) # (n, 2)

        # Use brain_data ONCE to initialize geometry; do not store it.
        self.update_xyz(brain_data.chanlocs)

        self.viz = ConnectivityView() 
        self.build_figure(brain_data=brain_data, threshold=threshold)
        print("Built")
    # --------- Boilerplate ----------

    # def __repr__(self) -> str:
    #     return (
    #         f"{self.__class__.__name__}("
    #         f"conn_idx={self.conn_idx}, "
    #         f"colorscale={self.colorscale!r}, "
    #         f"color_min={self.color_min}, "
    #         f"color_max={self.color_max}, "
    #         f"node_size={self.node_size}, "
    #         f"show_labels={self.show_labels}, "
    #         f"viz_type={self.viz_type!r}"
    #         f")"
    #     )
    
    # def __eq__(self, other) -> bool:
    #     """Two visualizers are considered equal if all configuration fields match."""
    #     if not isinstance(other, ConnectivityVisualizer):
    #         return False

    #     return (
    #         self.conn_idx == other.conn_idx
    #         and self.colorscale == other.colorscale
    #         and self.color_min == other.color_min
    #         and self.color_max == other.color_max
    #         and self.node_size == other.node_size
    #         and self.show_labels == other.show_labels
    #         and self.default_pos_color == other.default_pos_color
    #         and self.default_neg_color == other.default_neg_color
    #         and self.node_fill == other.node_fill
    #         and self.node_edge == other.node_edge
    #         and self.viz_type == other.viz_type
    #     )
    
    # ---------- Shared helpers ----------
    
    # ---------- Utils ----------
    def update(self, brain_data, threshold, update_type, viz_updates):
        self.conn_idx = viz_updates["conn_idx"]
        self.colorscale = viz_updates["colorscale"]
        self.color_max = viz_updates["color_max"]
        self.color_min = viz_updates["color_min"]

        
        if self.viz_type != viz_updates["viz_type"]:
            self.viz_type = viz_updates["viz_type"]
            self.build_figure(brain_data, threshold)

        # -----------------------
        # 2D settings
        # -----------------------
        if self.viz_type == VizType.FIG2D:
            self.node_size_2d = viz_updates["node_size_2d"]
            self.edge_size_min_2d = viz_updates["edge_min_2d"]
            self.edge_size_max_2d = viz_updates["edge_max_2d"]

        # -----------------------
        # 3D settings
        # -----------------------
        if self.viz_type == VizType.FIG3D:
            self.node_size_3d = viz_updates["node_size_3d"]
            self.edge_size_min_3d = viz_updates["edge_min_3d"]
            self.edge_size_max_3d = viz_updates["edge_max_3d"]
            self.num_arc_points_3d = viz_updates["arc_points_3d"]
            self.show_left_hemi_3d  = viz_updates["show_hemi_left_3d"]
            self.show_right_hemi_3d = viz_updates["show_hemi_right_3d"]
            
        self.update_figure(brain_data=brain_data, threshold=threshold, update_type=update_type)
        # inefficient?? building then updating?

    
    
    def update_xyz(
        self,
        chanlocs: Union[pd.DataFrame, Iterable[Union[Channel, dict, Iterable]]]
    ) -> None:
        """
        Update channel locations and recompute derived fields (xyz, xy_topo, labels)
        from a chanlocs object.

        chanlocs can be:
        - a pandas DataFrame with columns x, y, optional z, optional label
        - an iterable of Channel objects
        - an iterable of dicts with keys 'x', 'y', optional 'z', 'label'
        - a generic iterable of sequences like [x, y], [x, y, z], [x, y, z, label]
        """

        # ---- Parse channel locations into xyz + labels ----
        if isinstance(chanlocs, pd.DataFrame):
            sx = chanlocs["x"].to_numpy()
            sy = chanlocs["y"].to_numpy()
            sz = chanlocs["z"].to_numpy() if "z" in chanlocs.columns else np.zeros_like(sx)
            if "label" in chanlocs.columns:
                labs = chanlocs["label"].astype(str).to_numpy()
            else:
                labs = np.arange(len(sx)).astype(str)
        else:
            # list/ndarray of Channels or rows like [x, y, (z), (label)]
            sx, sy, sz, labs = [], [], [], []
            for row in chanlocs:
                if isinstance(row, Channel):
                    sx.append(row.x)
                    sy.append(row.y)
                    sz.append(row.z if row.z is not None else 0.0)
                    labs.append(row.label or "")
                elif isinstance(row, dict):
                    sx.append(float(row["x"]))
                    sy.append(float(row["y"]))
                    sz.append(float(row.get("z", 0.0)))
                    labs.append(str(row.get("label", "")))
                else:
                    # generic sequence
                    x = float(row[0])
                    y = float(row[1])
                    z = float(row[2]) if len(row) >= 3 and np.isscalar(row[2]) else 0.0
                    lab = (
                        str(row[3])
                        if len(row) >= 4
                        else (str(row[2]) if len(row) >= 3 and not np.isscalar(row[2]) else "")
                    )
                    sx.append(x)
                    sy.append(y)
                    sz.append(z)
                    labs.append(lab)

            sx = np.asarray(sx, dtype=float)
            sy = np.asarray(sy, dtype=float)
            sz = np.asarray(sz, dtype=float)
            labs = np.asarray(labs, dtype=str)

            if labs.size == 0:
                labs = np.arange(len(sx)).astype(str)

        n_ch = len(sx)
        # If you keep self.n as "number of channels", keep it in sync:
        self.n = n_ch

        # labels: ensure length matches n_ch; otherwise fallback to generic labels
        if labs.size == n_ch:
            self.labels = labs
        else:
            self.labels = np.arange(n_ch).astype(str)

        # 3D coordinates
        self.xyz = np.column_stack([sx, sy, sz]).astype(float)

        # ---- Precompute normalized 2D topography (EEG top view: flip x) ----
        xs = sx.copy()
        ys = sy.copy()
        xs = -xs / (np.max(np.abs(xs)) + 1e-12) * 0.9
        ys =  ys / (np.max(np.abs(ys)) + 1e-12) * 0.9
        self.xy_topo = np.column_stack([xs, ys])

    def get_figure(self) -> go.Figure:
        """Get the current figure based on viz_type."""
        if self.viz_type == VizType.FIG2D:
            return self.fig_2d_cache
        elif self.viz_type == VizType.FIG3D:
            return self.fig_3d_cache
        elif self.viz_type == VizType.FIGHEATMAP:
            return self.fig_heatmap_cache
        else:
            return go.Figure()

    def build_figure(self, brain_data: BrainData, threshold: Threshold) -> go.Figure:
        # get current matrix to get scale and range of data for new traces 
        C = helpers._get_mat_at_idx(brain_data.conn_mat, self.conn_idx)
        np.fill_diagonal(C, 0.0)
        new_thresh_mask = threshold.apply_threshold(brain_data.conn_mat, self.conn_idx)
        self._mask_cache = new_thresh_mask

        self.viz.build_figure(C=C, colorscale=self.colorscale, labels=brain_data.labels, directed=brain_data.directed)
        
    def update_figure(self, brain_data: BrainData, threshold: Threshold, update_type: UpdateType) -> go.Figure:
        """Get the current figure based on viz_type."""
        C = helpers._get_mat_at_idx(brain_data.conn_mat, self.conn_idx)
        np.fill_diagonal(C, 0.0)
        new_thresh_mask = threshold.apply_threshold(brain_data.conn_mat, self.conn_idx)
        old_thresh_mask = self._mask_cache
        new_thresh_mask = old_thresh_mask.copy()
        self.viz.update_figure(C=C, colorscale=self.colorscale, labels=brain_data.labels, directed=brain_data.directed, update_type=update_type, new_thresh_mask=new_thresh_mask, old_thresh_mask=old_thresh_mask)