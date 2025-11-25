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
from visualization.vizconn import ConnectivityView2D, ConnectivityView3D, ConnectivityViewHeatmap
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
        self.viz_dict = {VizType.FIG2D: ConnectivityView2D(chanlocs=brain_data.chanlocs, show_labels=show_labels)}
        self.build_figure(brain_data=brain_data, threshold=threshold)

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

        ### SWITCH VIZ NEED TO FIX
        if self.viz_type != viz_updates["viz_type"]:
            self.viz_type = viz_updates["viz_type"]
            BROKEN
            self.viz.build_figure(brain_data, threshold)

        
        self.viz.update_figure()
        # inefficient?? building then updating?

    
    def get_figure(self) -> go.Figure:
        return self.viz.fig

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