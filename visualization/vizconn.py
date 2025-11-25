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
from visualization.vizhelpers import VizType, UpdateType, Channel

class ConnectivityView(ABC):

    def __init__(self):
        self.fig = None
        # self.build_figure(brain_data=brain_data, threshold=threshold)
    """Interface for all connectivity visualizers (2D, 3D, heatmap, etc.)."""
    @abstractmethod
    def build_figure(
        self,
        C: np.ndarray,
        labels, 
        directed: bool,
        colorscale: str,
    ) -> go.Figure:
        pass
    @abstractmethod
    def update_figure(
        self,
        C: np.ndarray,
        labels, 
        directed: bool,
        colorscale: str,
        update_type:UpdateType,
        new_thresh_mask: Optional[np.ndarray],
        old_thresh_mask: Optional[np.ndarray],
    ) -> go.Figure:
        pass

    @abstractmethod
    def update_attributes(self):
        pass

# class NodeView()

class ConnectivityViewHeatmap(ConnectivityView):

    def build_figure(
        self,
        *,
        C:np.ndarray, 
        colorscale: str, 
        labels, 
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

        self.fig = fig
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
        scale, data_min, data_max, zmin, zmax = helpers._get_scale_and_range(C)

        fig.update_traces(
            z=C,
            zmin=zmin,
            zmax=zmax,
            colorscale=self.colorscale,
            selector=dict(name="main"),
        )

        self.fig = fig
        return self.fig

class HandlesGraphs():
    def __init__(self):
        pass

class HandlesNodes(): 
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


    @abstractmethod
    def update_locs(self, chanlocs):
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
    def _build_base_trace(self):
        pass

    @abstractmethod
    def _build_edge_traces(self, C: np.ndarray, color_min: float, color_max: float, labels, directed: bool, scale_info):
        pass


class ConnectivityView2D(ConnectivityView, HandlesNodes):
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
    
    def _make_edge_trace(self, P, color, width, i, j, w):
        return go.Scattergl(
            x=P[:, 0], y=P[:, 1],
            mode="lines",
            line=dict(color=color, width=width),
            opacity=0.75,
            showlegend=False,
            hoverinfo="text",
            text=f"{self.labels[i]} → {self.labels[j]}<br>Weight: {w:.3f}",
            name=f"{self.labels[i]},{self.labels[j]}",
        )
    
    def _collect_arrow(self, P, w):
        if len(P) < 2: 
            return None
        q0, q1 = P[-2], P[-1]
        pos = q1 - 0.05 * (q1 - q0)
        vec = q1 - q0
        L = np.linalg.norm(vec)
        if L < 1e-9:
            return None
        return (pos, vec / L)

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

        self.fig = fig
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

            traces_list = helpers._get_candidate_edges(old_thresh_mask=old_thresh_mask, new_thresh_mask=new_thresh_mask, update_type=update_type, directed=directed, n_nodes=C.shape[0])

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
        self.fig = fig
        return fig


class ConnectivityView3D(ConnectivityView, HandlesNodes):
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
        # need more settings
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

        # 2D normalized topography
        return helpers.compute_locs_3d(sx=sx, sy=sy, sz=sz)

    ### NEED TO FIX INPUTS
    def _get_edge_path(
        self,
        i: int,
        j: int,
        C: np.ndarray,
        arc_radius: Optional[float] = None,
        m: int = 60
    ) -> np.ndarray:
        p0 = self.locs[i]
        p1 = self.locs[j]

        # chord vector and length
        chord = p1 - p0
        L = np.linalg.norm(chord)

        if L < 1e-12:
            return np.vstack([p0, p1])   # fallback straight segment

        d = chord / L

        # curvature direction: choose a perpendicular vector
        perp = np.cross(d, np.array([0.0, 0.0, 1.0]))
        if np.linalg.norm(perp) < 1e-6:
            perp = np.cross(d, np.array([0.0, 1.0, 0.0]))
        perp = perp / (np.linalg.norm(perp) + 1e-12)

        # detect bidirectional edges to offset arcs
        reverse_exists = (np.isfinite(C[j, i]) and abs(C[j, i]) > 1e-12)
        sign = 0
        if reverse_exists:
            sign = 1 if i < j else -1

        # arc amplitude
        if arc_radius is None:
            arc_height = 0.15 * L      # automatic light curvature
        else:
            arc_height = float(arc_radius)

        # parametric t in [0,1]
        t = np.linspace(0.0, 1.0, m)

        # central arc (quadratic "hump")
        base = p0[None, :] + np.outer(t, chord)
        hump = arc_height * np.sin(np.pi * t)

        # primary curvature (adds elevation)
        P = base + np.outer(hump, perp)

        # if bidirectional, offset each arc sideways
        if sign != 0:
            env = np.sin(np.pi * t)
            offset_amt = 0.06 * L * sign
            P += np.outer(env * offset_amt, perp)

        return P


    def _build_base_traces(self, brain_mesh: Optional[pv.PolyData]) -> List[go.Scattergl]:
        if brain_mesh is not None and pv is not None and brain_mesh.n_points > 0:
            pts = np.asarray(brain_mesh.points)
            faces_np = np.asarray(brain_mesh.faces)
            faces = faces_np.reshape(-1, 4)[:, 1:4].astype(int)
            head = go.Mesh3d(
                x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                color="lightgray", opacity=0.25, flatshading=True,
                lighting=dict(ambient=0.6, diffuse=0.6, specular=0.1),
                name="Brain"
            )

        x, y, z = self.xyz[:, 0], self.xyz[:, 1], self.xyz[:, 2]
        nodes = go.Scatter3d(
            x=x, y=y, z=z,
            mode="markers+text" if self.show_labels else "markers",
            text=self.labels if self.show_labels else None,
            textposition="top center",
            textfont=dict(size=10, color="black"),
            marker=dict(size=self.node_size_3d),
            name="Electrodes"
        )
        return [head, nodes]
    
    def _make_edge_trace(self, P, color, width, i, j, w):
        return go.Scattergl(
            x=P[:, 0], y=P[:, 1],
            mode="lines",
            line=dict(color=color, width=width),
            opacity=0.75,
            showlegend=False,
            hoverinfo="text",
            text=f"{self.labels[i]} → {self.labels[j]}<br>Weight: {w:.3f}",
            name=f"{self.labels[i]},{self.labels[j]}",
        )
    
    def _collect_arrow(self, P, w):
        if len(P) < 2:
            return None

        q0, q1 = P[-2], P[-1]
        pos = q1 - 0.05 * (q1 - q0)

        vec = q1 - q0
        L = np.linalg.norm(vec)
        if L < 1e-9:
            return None
        return (pos, vec / L)

    def _build_arrowhead_trace(self, pos_list, vec_list, vals, sizes, zmin, zmax):
        if not pos_list:
            return None

        xs, ys, zs = zip(*pos_list)
        us, vs, ws = zip(*vec_list)

        return go.Cone(
            x=xs, y=ys, z=zs,
            u=us, v=vs, w=ws,
            color=vals,
            sizeref=max(0.5, float(np.nanmax(sizes))),
            sizemode="absolute",
            showscale=False,
            colorscale=self.colorscale,
            cmin=zmin, cmax=zmax,
            anchor="tip",
            name="arrows",
        )

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
    
    # edge_traces = []
    #     arrow_positions = []
    #     arrow_vectors   = []
    #     arrow_vals      = []
    #     arrow_sizes     = []

    #     for i, j in self.get_ij_iter(brain_data.n_nodes, brain_data.directed):
    #         w = float(C[i, j])

    #         color = self._get_edge_color(edge=w, data_min=data_min, data_max=data_max)
    #         width = self._get_edge_width(edge=w, scale=scale,
    #                                      min_width=self.min_width,
    #                                      max_width=self.max_width)

    #         # polymorphic!
    #         P = self.get_edge_path(i, j)

    #         # polymorphic!
    #         trace = self.make_edge_trace(P, color, width, i, j, w)
    #         edge_traces.append(trace)

    #         if brain_data.directed:
    #             pos_vec = self.collect_arrow(P, w)
    #             if pos_vec is not None:
    #                 pos, vec = pos_vec
    #                 arrow_positions.append(pos)
    #                 arrow_vectors.append(vec)
    #                 arrow_vals.append(w)

    #                 adj = self._normalize_weight(w, data_min, data_max)
    #                 arrow_sizes.append(max(0.6, 0.6 * adj))

    #     arrow_trace = None
    #     if brain_data.directed:
    #         arrow_trace = self.build_arrowhead_trace(
    #             arrow_positions, arrow_vectors, arrow_vals, arrow_sizes, zmin, zmax
    #         )

    #     return edge_traces, arrow_trace, (zmin, zmax)

    
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

            traces_list = helpers._get_candidate_edges(old_thresh_mask=old_thresh_mask, new_thresh_mask=new_thresh_mask, update_type=update_type, directed=directed, n_nodes=C.shape[0])

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
