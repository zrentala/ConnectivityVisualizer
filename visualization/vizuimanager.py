from __future__ import annotations
from dataclasses import dataclass, field
from typing import Iterable, Optional, Tuple, Union, List, Dict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.colors as plc

import analysis.threshold as thresh
from utils.braindata import BrainData
from analysis.threshold import Threshold
from visualization.vizhelpers import VizType, UpdateType

import visualization.vizhelpers as helpers
from visualization.vizconn import ConnectivityView2D, ConnectivityView3D, ConnectivityViewHeatmap, ConnectivityView
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
    """
    Manages UI state for connectivity visualization and delegates figure
    construction to specialized ConnectivityView classes.

    Does NOT store brain_data or threshold internally.
    """

    def __init__(
        self,
        brain_data: BrainData,
        threshold: Threshold,
        conn_idx: int = 0,
        colorscale: str = "Viridis",
        color_min: float = 0.0,
        color_max: float = 1.0,
        viz_type: VizType = VizType.FIG2D,
        show_title: bool = True
    ) -> None:

        # -----------------------
        # Core UI/settings fields
        # -----------------------
        self.conn_idx = conn_idx
        self.colorscale = colorscale
        self.color_min = color_min
        self.color_max = color_max
        self.viz_type = viz_type
        self.show_title = show_title

        # Internal cached threshold mask
        self._mask_cache = None

        # -----------------------
        # Build visualizers immediately
        # but DO NOT store brain_data
        # -----------------------
        self.viz_dict = {
            VizType.FIG2D: ConnectivityView2D(chanlocs=brain_data.chanlocs),
            VizType.FIG3D: ConnectivityView3D(chanlocs=brain_data.chanlocs),
            VizType.FIGHEATMAP: ConnectivityViewHeatmap(),
        }
        self.build_figure(brain_data=brain_data, threshold=threshold)

    # ------------------------------------------------------------------
    # UI update
    # ------------------------------------------------------------------
    def update_attributes(self, viz_updates: dict):
        self.conn_idx = viz_updates["conn_idx"]
        self.colorscale = viz_updates["colorscale"]
        self.color_max = viz_updates["color_max"]
        self.color_min = viz_updates["color_min"]
        self.viz_type = viz_updates["viz_type"]
        # forward attribute updates to active visualizer
        self.viz_dict[self.viz_type].update_attributes(viz_updates)

    # ------------------------------------------------------------------
    # Build new figure
    # ------------------------------------------------------------------
    def build_figure(self, brain_data: BrainData, threshold: Threshold):
        C = helpers._get_mat_at_idx(brain_data.conn_mat, self.conn_idx)
        np.fill_diagonal(C, 0.0)

        scale, data_min, data_max, zmin, zmax = helpers._get_scale_and_range(C, color_min=self.color_min, color_max=self.color_max)

        color_scale_info = (scale, data_min, data_max, zmin, zmax, self.colorscale)

        # compute and store threshold mask
        _, self._mask_cache = threshold.apply_threshold(
            brain_data.conn_mat, self.conn_idx
        )

        title = brain_data.mat_names[self.conn_idx]

        if self.viz_type == VizType.FIG2D:
            self.viz_dict[self.viz_type].build_figure(
                C=C,
                labels=brain_data.labels,
                directed=brain_data.directed,
                color_scale_info= color_scale_info,
                title=title
            )
        elif self.viz_type == VizType.FIG3D:
            self.viz_dict[self.viz_type].build_figure(
                C=C,
                labels=brain_data.labels,
                directed=brain_data.directed,
                color_scale_info= color_scale_info,
                brain_data=brain_data.brain_mesh,
                title=title
            )
        elif self.viz_type == VizType.FIGHEATMAP:
            self.viz_dict[self.viz_type].build_figure(
                C=C,
                labels=brain_data.labels,
                directed=brain_data.directed,
                color_scale_info= color_scale_info,
                title=title
            )

    # ------------------------------------------------------------------
    # Update figure in place (fast)
    # ------------------------------------------------------------------
    def update_figure(self, brain_data: BrainData, threshold: Threshold, update_type: UpdateType):
        old_mask = self._mask_cache
        C, new_mask = threshold.apply_threshold(brain_data.conn_mat, self.conn_idx)
        self._mask_cache = new_mask.copy()
        np.fill_diagonal(C, 0.0)
        
        scale, data_min, data_max, zmin, zmax = helpers._get_scale_and_range(C, color_min=self.color_min, color_max=self.color_max)
        color_scale_info = (scale, data_min, data_max, zmin, zmax, self.colorscale)
        
        title = brain_data.mat_names[self.conn_idx] if self.show_title else None

        if self.viz_type == VizType.FIG2D:
            self.viz_dict[self.viz_type].update_figure(
            C=C,
            labels=brain_data.labels,
            directed=brain_data.directed,
            update_type=update_type,
            new_thresh_mask=new_mask,
            old_thresh_mask=old_mask,
            color_scale_info=color_scale_info,
            title=title
        )
        elif self.viz_type == VizType.FIG3D:
            self.viz_dict[self.viz_type].update_figure(
            C=C,
            labels=brain_data.labels,
            directed=brain_data.directed,
            update_type=update_type,
            new_thresh_mask=new_mask,
            old_thresh_mask=old_mask,
            color_scale_info=color_scale_info,
            brain_mesh=brain_data.brain_mesh,
            title=title
        )
        elif self.viz_type == VizType.FIGHEATMAP:
            self.viz_dict[self.viz_type].update_figure(
            C=C,
            labels=brain_data.labels,
            directed=brain_data.directed,
            update_type=update_type,
            new_thresh_mask=new_mask,
            old_thresh_mask=old_mask,
            color_scale_info=color_scale_info,
            title=title
        )

    # ------------------------------------------------------------------
    def get_figure(self) -> go.Figure:
        return self.viz_dict[self.viz_type].fig
    
    def get_viz_class(self) -> ConnectivityView:
        return self.viz_dict[self.viz_type]