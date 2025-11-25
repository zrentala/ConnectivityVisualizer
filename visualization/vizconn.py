from __future__ import annotations
from dataclasses import dataclass, field
from typing import Iterable, Optional, Tuple, Union, List

import numpy as np
import plotly.graph_objects as go

from utils.braindata import BrainData
from analysis.threshold import Threshold

# from typing import Protocol
from abc import abstractmethod, ABC
import plotly.graph_objects as go
import analysis.threshold as thresh
from utils.braindata import BrainData
from analysis.threshold import Threshold
from itertools import product
import visualization.vizhelpers as helpers
from visualization.vizhelpers import VizType, UpdateType

class ConnectivityView(ABC):

    def __init__(self):
        self.fig = None
        # self.build_figure(brain_data=brain_data, threshold=threshold)
    """Interface for all connectivity visualizers (2D, 3D, heatmap, etc.)."""
    @abstractmethod
    def build_figure(self, brain_data: BrainData) -> go.Figure:
        pass
    @abstractmethod
    def update_figure(
        self,
        brain_data: BrainData,
        threshold: Threshold,
        update_type: UpdateType,
        new_thresh_mask: Optional[np.ndarray],
        old_thresh_mask: Optional[np.ndarray]
    ) -> go.Figure:
        pass

    @abstractmethod
    def get_figure(self) -> go.Figure:
        pass

# class NodeView()

class ConnectivityViewHeatmap(ConnectivityView):
    def __init__(
        self,
        brain_data: BrainData,
        threshold: Threshold,
    ) -> None:
        # caches
        self.mask_cache: np.ndarray = np.empty((brain_data.n_nodes, brain_data.n_nodes), dtype=bool) 


    def build_figure(
        self,
        *,
        C:np.ndarray, 
        colorscale: str, 
        labels, 
        directed: bool,
    ) -> go.Figure:
        fig = go.Figure()

        # Color range: compute full-data min/max then map color_min/color_max (0..1) into that range
        scale, data_min, data_max, zmin, zmax = helpers._get_scale_and_range(C)

        ### CREATE HEATMAP, WE DO NOT NEED TO CREATE IT'S OWN COLORBAR
        fig.add_trace(go.Heatmap(
            z=C,
            x=labels,
            y=labels,
            colorscale=colorscale,
            zmin=zmin,
            zmax=zmax,
            xgap=0.5,
            ygap=0.5,
            colorbar=dict(title="Conn"),
            showscale=True,
            hovertemplate="From %{y}<br>To %{x}<br>Value=%{z:.3f}<extra></extra>",
            name="main"
        ))

        # Layout styling
        fig.update_layout(
            xaxis=dict(
                title="To",
                tickangle=45,
                showgrid=False,
                zeroline=False,
            ),
            yaxis=dict(
                title="From",
                autorange="reversed",
                showgrid=False,
                zeroline=False,
            ),
            autosize=True,
            margin=dict(l=60, r=20, t=40, b=80),
            plot_bgcolor="white",
        )

        return fig

    def update_figure( self,
        C: np.ndarray,
        labels, 
        directed: bool,
        colorscale: str,
        update_type:UpdateType,
        new_thresh_mask: Optional[np.ndarray],
        old_thresh_mask: Optional[np.ndarray],
    ) -> go.Figure:

        scale, data_min, data_max, zmin, zmax = self._get_scale_and_range(C)

        self.fig.update_traces(
            z=C,
            zmin=zmin,
            zmax=zmax,
            colorscale=self.colorscale,
            selector=dict(name="main"),
        )


        return self.fig
       

class ConnectivityViewNode(ConnectivityView): 
    def __init__(
        self,
        chanlocs,
        show_labels:bool =True,
        # 2D settings
        node_size: float = 10.0,
        edge_size_min: float = 0.4,
        edge_size_max: float = 4.0,
        default_pos_color: str = "red",
        default_neg_color: str = "blue",
        node_fill: str = "lightgreen",
        node_edge: str = "black",
    ) -> None:
        self.show_labels = show_labels
        self.node_size = node_size
        self.edge_size_min = edge_size_min
        self.edge_size_max = edge_size_max
        self.node_fill = node_fill
        self.node_edge = node_edge
        self.default_pos_color = default_pos_color
        self.default_neg_color = default_neg_color
        self.locs: np.ndarray = np.empty((0, 2), dtype=float) # (n, 2)
        self.update_locs(chanlocs)

        # caches
        self._edge_trace_idx = {}
        self._colorbar_trace_idx = -999

    def _get_candidate_edges(self, old_thresh_mask, new_thresh_mask, update_type, directed, n_nodes):

        ij_iter = helpers._get_ij_iter(n_nodes=n_nodes, directed=directed)
        changed_edges = []
        # ---- ALL ----
        if update_type is UpdateType.ALL:
            changed_edges = [
                ((i, j), self._edge_trace_idx[(i, j)])
                for (i, j) in ij_iter
                if (i, j) in self._edge_trace_idx
            ]

        # ---- COLOR ----, only get visible edges
        if update_type is UpdateType.COLOR:
            changed_edges = [
                ((i, j), self._edge_trace_idx[(i, j)])
                for (i, j) in ij_iter
                if new_thresh_mask[i, j] and (i, j) in self._edge_trace_idx
            ]
        # ---- THRESHOLD ----, get edges which differ in their visibility between the old_thresh_mask and new_thresh_mask
        if update_type is UpdateType.THRESHOLD:
            diff = (old_thresh_mask != new_thresh_mask)
            changed_edges = [
                ((i, j), self._edge_trace_idx[(i, j)])
                for (i, j) in ij_iter
                if diff[i, j] and (i, j) in self._edge_trace_idx
            ]
            

        print(f"Number edges: {len(changed_edges)}")
        return changed_edges

    @abstractmethod
    def update_locs(chanlocs):
        pass

    @abstractmethod
    def _get_edge_path(self, i: int, j: int, *args, **kwargs) -> np.ndarray:
        """Return NxD array of coordinates for the edge."""
        pass

    @abstractmethod
    def _make_edge_trace(self, P: np.ndarray, color: str, width: float, i: int, j: int, w: float):
        """Return the Plotly trace representing the edge."""
        pass

    @abstractmethod
    def _collect_arrow(self, P: np.ndarray, w: float):
        """Extract arrow position + direction from path."""
        pass

    @abstractmethod
    def _build_arrowhead_trace(self, arrow_positions, arrow_vectors, arrow_vals, arrow_sizes,
                              zmin, zmax):
        """Return the aggregated arrowhead trace (or None)."""
        pass

    # -------- SHARED LOGIC -------- #

    def _build_edge_traces(self, C: np.ndarray, color_min: float, color_max: float, labels, directed: bool, scale_info):
        """Generic shared implementation used by both 2D and 3D."""
        scale, data_min, data_max, zmin, zmax = helpers._get_scale_and_range(C=C, color_min=color_min, color_max=color_max)

        edge_traces = []
        arrow_positions = []
        arrow_vectors   = []
        arrow_vals      = []
        arrow_sizes     = []

        for i, j in helpers._get_ij_iter(C.shape[0], directed):
            w = float(C[i, j])

            color = helpers._get_edge_color(edge=w, data_min=data_min, data_max=data_max)
            width = helpers._get_edge_width(edge=w, scale=scale,
                                         min_width=self.edge_size_min,
                                         max_width=self.edge_size_max)

            # polymorphic!
            P = self._get_edge_path(i, j)

            # polymorphic!
            trace = self._make_edge_trace(P, color, width, i, j, w)
            edge_traces.append(trace)

            if directed:
                pos_vec = self._collect_arrow(P, w)
                if pos_vec is not None:
                    pos, vec = pos_vec
                    arrow_positions.append(pos)
                    arrow_vectors.append(vec)
                    arrow_vals.append(w)

                    adj = helpers._normalize_weight(w, data_min, data_max)
                    arrow_sizes.append(max(0.6, 0.6 * adj))

        arrow_trace = None
        if directed:
            arrow_trace = self._build_arrowhead_trace(
                arrow_positions, arrow_vectors, arrow_vals, arrow_sizes, zmin, zmax
            )

        return edge_traces, arrow_trace, (zmin, zmax)
    

    def build_figure(
        self,
        C: np.ndarray,
        labels, 
        directed: bool,
        colorscale: str,
    ) -> go.Figure:
        ### GET HEAD, NOSE, NODES (NODES MAY NEED TO BE SEPARATED) 
        fig = go.Figure()
        for tr in self._build_base_traces():
            fig.add_trace(tr)

        ### CREATE EDGES
        edge_traces, (zmin, zmax) = self._build_edge_traces(C=C, 
            directed=directed,
        )
        for tr in edge_traces:
            fig.add_trace(tr)

        ### CACHE EDGES (i, j) --> idx
        labels = labels
        self._edge_trace_idx = helpers._create_cache_edges(labels, fig)

        ### CREATE COLOR BAR
        colorbar_trace_idx = helpers._add_colorbar_trace(fig=fig, colorscale=colorscale,zmin=zmin,zmax=zmax, viz_type=VizType.FIG2D)
        self._colorbar_trace_idx = colorbar_trace_idx

        ### ADD REST OF LAYOUT
        fig.update_layout(
            xaxis=dict(
                visible=False,
                scaleanchor="y",
                scaleratio=1,
            ),
            yaxis=dict(visible=False),
            autosize=True,
            margin=dict(l=0, r=0, t=40, b=0),
            showlegend=False,
            plot_bgcolor="white",
        )

        return fig

    def update_figure( self,
        C: np.ndarray,
        labels, 
        directed: bool,
        colorscale: str,
        update_type:UpdateType,
        new_thresh_mask: Optional[np.ndarray],
        old_thresh_mask: Optional[np.ndarray],
    ) -> go.Figure:
        
        fig = self.fig
        if fig is None:
            # safety: fall back to full build if cache is missing
            return self.build_figure(C=C, labels=labels, directed=directed, colorscale=colorscale)
        
        scale, data_min, data_max, zmin, zmax = helpers._get_scale_and_range(C)

        with fig.batch_update():

            traces_list = self._get_candidate_edges(old_thresh_mask=old_thresh_mask, new_thresh_mask=new_thresh_mask, update_type=update_type, directed=directed, n_nodes=C.shape[0])

            for (i, j), idx in traces_list:
                ### GET EDGE's CONN and MASK (BOOL)
                w = C[i, j]
                m = new_thresh_mask[i, j]

                ### TRIED GETTING RID OF CHECK OF IDX
                # if idx is None:
                #     continue  # should not happen if build/setup is consistent

                ### GET TRACE
                trace = fig.data[idx]
                
                ### HIDE IF MASK == FALSE
                if not m:
                    trace.opacity = 0.0
                    trace.hoverinfo = "skip"
                    trace.text = ""
                    continue

                ### GET COLOR
                color = helpers._get_edge_color(edge=w, data_max=data_max, data_min=data_min)

                ### GET WIDTH
                width = helpers._get_edge_width(edge=w, scale=scale, min_width=self.edge_size_min, max_width=self.edge_size_max)

                ### UPDATE TRACE FOR VISIBLE EDGES. NEED TO FIX OPACITY FOR 2D, LET'S MAKE THIS AN UPDATEABLE VALUE
                helpers._update_edge_trace(trace, w, color, width, 0.75, labels[i], labels[j])

            # UPDATE COLOR BAR (MAKE THIS FUNCTION)
            helpers._update_colorbar(fig, self._colorbar_trace_idx, colorscale, zmin, zmax)
        return fig
    


class ConnectivityView2D(ConnectivityViewNode):
    def __init__(
        self,
        chanlocs,
        show_labels:bool =True,
        node_size: float = 10.0,
        edge_size_min: float = 0.4,
        edge_size_max: float = 4.0,
        default_pos_color: str = "red",
        default_neg_color: str = "blue",
        node_fill: str = "lightgreen",
        node_edge: str = "black",
    ) -> None:
        self.show_labels = show_labels
        super().__init__(chanlocs=chanlocs, 
                         show_labels=show_labels, 
                         node_size=node_size, 
                         edge_size_max=edge_size_max, 
                         edge_size_min=edge_size_min,
                         default_pos_color=default_pos_color,
                         default_neg_color=default_neg_color, 
                         node_fill=node_fill,
                         node_edge=node_edge)
       
    def update_locs(chanlocs):
           # Parse → sx, sy, sz, labels
        sx, sy, sz, labs = helpers.parse_channel_locs(chanlocs)

        # update internal state
        n = len(sx)

        # 2D normalized topography
        return helpers.compute_xy_topo(sx, sy)


    def _build_base_traces(self) -> List[go.Scattergl]:
            """Head outline, nose, and node markers (no edges)."""
            theta = np.linspace(0, 2 * np.pi, 256)
            x, y = self.xy_topo[:, 0], self.xy_topo[:, 1]

            head = go.Scattergl(
                x=np.cos(theta),
                y=np.sin(theta),
                mode="lines",
                line=dict(color="black", width=2),
                hoverinfo="skip",
                name="Head",
            )

            nose = go.Scattergl(
                x=[0.10, 0.00, -0.10],
                y=[1.00, 1.10, 1.00],
                mode="lines",
                line=dict(color="black", width=2),
                name="Nose",
                hoverinfo="skip",
                showlegend=False,
            )

            nodes = go.Scattergl(
                x=x,
                y=y,
                mode="markers+text" if self.show_labels else "markers",
                text=self.labels if self.show_labels else None,
                textposition="middle center",
                marker=dict(
                    size=self.node_size_2d,
                    color=self.node_fill,
                    line=dict(color=self.node_edge, width=2),
                ),
                hovertext=self.labels,
                hoverinfo="text",
                name="Electrodes",
            )

            return [head, nose, nodes]
    
    def _get_edge_path(self, i: int, j: int, use_arcs: bool, curvature: float) -> np.ndarray:
        p0 = self.locs[i]
        p1 = self.locs[j]
        if use_arcs:
            P = helpers._quad_bezier(p0, p1, curvature, m=60)
        else:
            P = np.vstack([p0, p1])
        return P

    def _build_edge_traces(
        self,
        C: np.ndarray,
        directed: bool,
        labels,
        curvature: float,
    ) -> List[go.Scattergl]:
        scale, data_min, data_max, zmin, zmax = helpers._get_scale_and_range(C)
        # set up these traces
        edge_traces: List[go.Scattergl] = []
        n_nodes = C.shape[0]
        #  LOOP OVER ALL POSSIBLE TRACES TO CREATE ALL EDGES
        ij_iter = helpers._get_ij_iter(n_nodes, directed) 
        for i, j in ij_iter:
            # get connection value
            w = C[i, j]

            ### GET EDGE COLOR (MAYBE FUNCTION??)
            # Normalize weight to signed [data_min, data_max] then to [0,1]
            color = helpers._get_edge_color(edge_weight=w, data_max=data_max, data_min=data_min)

            ### GET EDGE WIDTH 
            width = helpers._get_edge_width(edge_weight=w, scale=scale, min_width=self.edge_size_min, max_width=self.edge_size_max)

            ### GET EDGE PATH
            P = self._get_edge_path(i, j, use_arcs=directed, curvature=curvature)

            ### ADD TO EDGE TRACE LIST. CREATES EDGE
            edge_traces.append(
                go.Scattergl(
                    x=P[:, 0],
                    y=P[:, 1],
                    mode="lines",
                    line=dict(color=color, width=width),
                    opacity=0.75,
                    showlegend=False,
                    hoverinfo="text",
                    text=f"{labels[i]} → {labels[j]}<br>Weight: {w:.3f}",
                    name=f"{labels[i]},{labels[j]}",
                )
            )

            ### ADDS ARROWS IF DIRECTED
            if directed and len(P) >= 2:
                q0, q1 = P[-2], P[-1]
                edge_traces.append(
                    go.Scattergl(
                        x=[q0[0], q1[0]],
                        y=[q0[1], q1[1]],
                        mode="lines",
                        line=dict(color=color, width=width / 2),
                        opacity=0.0,  
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )

        return edge_traces, (zmin, zmax)
