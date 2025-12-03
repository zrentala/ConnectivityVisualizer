from __future__ import annotations
from dataclasses import dataclass, field
from typing import Iterable, Optional, Tuple, Union, List
import pyvista as pv
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

    def __init__(
            self,
            default_pos_color: str = "red",
            default_neg_color: str = "blue"
            ):
        self.fig = None
        self.default_pos_color = default_pos_color
        self.default_neg_color = default_neg_color
        # self.build_figure(brain_data=brain_data, threshold=threshold)
    """Interface for all connectivity visualizers (2D, 3D, heatmap, etc.)."""
    @abstractmethod
    def build_figure(
        self,
        C: np.ndarray,
        labels,
        directed: bool,
        color_scale_info,
    ) -> go.Figure:
        pass
    @abstractmethod
    def update_figure(
        self,
        C: np.ndarray,
        labels, 
        directed: bool,
        update_type:UpdateType,
        color_scale_info,
        new_thresh_mask: Optional[np.ndarray],
        old_thresh_mask: Optional[np.ndarray],
    ) -> go.Figure:
        pass

    @abstractmethod
    def update_attributes(self, viz_updates):
        pass


class HandlesGraphs():
    def __init__(self):
        pass

class HandlesNodes(): 
    def __init__(
        self,
        chanlocs,
        node_size: float = 28.0,
        # node_opacity: float = 0.75,
        edge_width_range: Tuple[float]=(0.4, 5),
        edge_opacity: float = 0.5,
        arc_radius: float=0.5,
        
        node_fill: str = "lightgreen",
        node_edge: str = "black",
    ) -> None:
        self.node_size = node_size
        # self.node_opacity = node_opacity
        self.edge_width_range = edge_width_range
        self.edge_opacity = edge_opacity
        self.node_fill = node_fill
        self.node_edge = node_edge
        self.arc_radius = arc_radius
        self.locs: np.ndarray = np.empty((0, 2), dtype=float) # (n, 2)
        self.update_locs(chanlocs)

        # caches
        self._node_trace_idx = -999
        self._edge_trace_idx = {}
        self._arrow_trace_idx = {}
        self._colorbar_trace_idx = -999


    @abstractmethod
    def update_locs(self, chanlocs):
        pass

    @abstractmethod
    def _get_edge_path(self, i: int, j: int, *args, **kwargs) -> np.ndarray:
        """Return NxD array of coordinates for the edge."""
        pass

    @abstractmethod
    def _make_edge_trace(self, P: np.ndarray, color: str, width: float, i: int, j: int, w: float, labels: bool):
        """Return the Plotly trace representing the edge."""
        pass

    @abstractmethod
    def _collect_arrow(self, P: np.ndarray, w: float):
        """Extract arrow position + direction from path."""
        pass

    @abstractmethod
    def _build_base_trace(self, fig):
        pass

    @abstractmethod
    def _build_node_trace(self, fig, labels):
        pass

    @abstractmethod
    def _build_edge_traces(self, C: np.ndarray, labels, directed: bool, color_scale_info):
        pass


class ConnectivityViewHeatmap(ConnectivityView):
    def __init__(
        self,
        default_pos_color: str = "red",
        default_neg_color: str = "blue",
    ):
        super().__init__(
            default_pos_color=default_pos_color,
            default_neg_color=default_neg_color,
        )


    def build_figure(
        self,
        C: np.ndarray,
        labels,
        directed: bool,
        color_scale_info,
    ):
        fig = go.Figure()

        # Color range: compute full-data min/max then map color_min/color_max (0..1) into that range
        scale, data_min, data_max, zmin, zmax, colorscale = color_scale_info

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

    def update_figure(
        self,
        C: np.ndarray,
        labels, 
        directed: bool,
        update_type:UpdateType,
        color_scale_info,
        new_thresh_mask: Optional[np.ndarray],
        old_thresh_mask: Optional[np.ndarray],
    ) -> go.Figure:
        fig = self.fig
        
        scale, data_min, data_max, zmin, zmax, colorscale = color_scale_info

        if fig is None:
            fig = self.build_figure(
                C=C,
                labels=labels,
                directed=directed,
                color_scale_info=color_scale_info,
            )

        fig.update_traces(
            z=C,
            zmin=zmin,
            zmax=zmax,
            colorscale=colorscale,
            selector=dict(name="main"),
        )

        self.fig = fig
        return self.fig
    
    def update_attributes(self, viz_updates):
        return


class ConnectivityView2D(ConnectivityView, HandlesNodes):
    def __init__(
        self,
        chanlocs,
        node_size: float = 28,
        # node_opacity: float=0.75,
        edge_width_range: Tuple[float] = (0.4, 5),
        edge_opacity: float=0.5,
        default_pos_color: str = "red",
        default_neg_color: str = "blue",
        node_fill: str = "lightgreen",
        node_edge: str = "black",
    ) -> None:
        ConnectivityView.__init__(self, default_pos_color=default_pos_color,
            default_neg_color=default_neg_color,)  # no args
        HandlesNodes.__init__(
            self,
            chanlocs=chanlocs,
            node_size=node_size,
            # node_opacity=node_opacity,
            edge_width_range=edge_width_range,
            edge_opacity=edge_opacity,
            node_fill=node_fill,
            node_edge=node_edge,
        )
       
    def update_attributes(self, viz_updates):
        print(f"{self.node_size=}")
        self.node_size = viz_updates["node_size_2d"]
        print(f"{self.node_size=}")
        self.edge_width_range = viz_updates["edge_width_range_2d"]
        # self.node_opacity = viz_updates["node_opacity_2d"]
        self.edge_opacity = viz_updates["edge_opacity_2d"]
        # print(f"{self.node_opacity=}, {self.edge_opacity=}")
        # self.update_locs(viz_updates["chanlocs"])

    def update_locs(self, chanlocs):
           # Parse → sx, sy, sz, labels
        sx, sy, sz, labs = helpers.parse_channel_locs(chanlocs)

        # 2D normalized topography
        self.locs = helpers.compute_locs_2d_topo(sx, sy)


    def _build_node_trace(self, fig, labels):
            x, y = self.locs[:, 0], self.locs[:, 1]
            nodes = go.Scattergl(
                x=x,
                y=y,
                mode="markers+text",
                text=labels,
                textposition="middle center",
                marker=dict(
                    size=self.node_size,
                    color=self.node_fill,
                    line=dict(color=self.node_edge, width=2),
                ),
                # opacity=self.node_opacity,
                hovertext=labels,
                hoverinfo="text",
                name="Electrodes",
            )
            fig.add_trace(nodes)
            self._node_trace_idx = len(fig.data) - 1

    def _build_base_trace(self, fig):
            """Head outline, nose, and node markers (no edges)."""
            theta = np.linspace(0, 2 * np.pi, 256)
            

            head = go.Scattergl(
                x=np.cos(theta),
                y=np.sin(theta),
                mode="lines",
                line=dict(color="black", width=2),
                hoverinfo="skip",
                name="Head",
            )

            fig.add_trace(head)

            nose = go.Scattergl(
                x=[0.10, 0.00, -0.10],
                y=[1.00, 1.10, 1.00],
                mode="lines",
                line=dict(color="black", width=2),
                name="Nose",
                hoverinfo="skip",
                showlegend=False,
            )

            fig.add_trace(head)
    
    def _get_edge_path(self, i: int, j: int, use_arcs: bool) -> np.ndarray:
        p0 = self.locs[i]
        p1 = self.locs[j]
        if use_arcs:
            P = helpers._quad_bezier(p0, p1, self.arc_radius, m=60)
        else:
            P = np.vstack([p0, p1])
        return P
    
    def _make_edge_trace(self, P, color, width, i, j, w, labels):
        return go.Scattergl(
            x=P[:, 0], y=P[:, 1],
            mode="lines",
            line=dict(color=color, width=width),
            opacity=self.edge_opacity,
            showlegend=False,
            hoverinfo="text",
            text=f"{labels[i]} → {labels[j]}<br>Weight: {w:.3f}",
            name=f"{labels[i]},{labels[j]}",
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
        color_scale_info,
        fig: go.Figure,
    ) -> List[go.Scattergl]:
        scale, data_min, data_max, zmin, zmax, colorscale = color_scale_info
        
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
            color = helpers._get_edge_color(edge_weight=w, zmin=zmin, zmax=zmax, colorscale=colorscale, default_neg_color=self.default_neg_color, default_pos_color=self.default_pos_color)

            ### GET EDGE WIDTH 
            width = helpers._get_edge_width(edge_weight=w, scale=scale, width_range=self.edge_width_range)

            ### GET EDGE PATH
            P = self._get_edge_path(i, j, use_arcs=directed)

            ### ADD TO EDGE TRACE LIST. CREATES EDGE
            fig.add_trace(
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
            self._edge_trace_idx[(i,j)] = len(fig.data) - 1

            ### ADDS ARROWS IF DIRECTED
            if directed and len(P) >= 2:
                q0, q1 = P[-2], P[-1]
                fig.add_annotation(
                    x=q1[0], y=q1[1],
                    ax=q0[0], ay=q0[1],
                    xref="x", yref="y",
                    axref="x", ayref="y",
                    showarrow=True,
                    arrowhead=3,
                    arrowsize=1,
                    arrowwidth=width,
                    arrowcolor=color,
                )
                self._arrow_trace_idx[(i,j)] = len(fig.layout.annotations) - 1
    
    def build_figure(
        self,
        C: np.ndarray,
        labels,
        directed: bool,
        color_scale_info,
    ) -> go.Figure:
        print("built 2d")
        scale, data_min, data_max, zmin, zmax, colorscale = color_scale_info
        ### GET HEAD, NOSE, NODES (NODES MAY NEED TO BE SEPARATED) 
        fig = go.Figure()
        self._build_base_trace(fig)
        # The node trace is always the *last* one in base:
        
        

        ### CREATE EDGES
        self._build_edge_traces(C=C, 
                                labels=labels,
                                directed=directed,
                                color_scale_info=color_scale_info,
                                fig=fig
                                )


        ### CACHE EDGES (i, j) --> idx

        # self._edge_trace_idx = helpers._create_cache_edges(labels, fig)
        self._build_node_trace(fig, labels)

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

    def update_figure(self,
        C: np.ndarray,
        labels, 
        directed: bool,
        update_type:UpdateType,
        color_scale_info,
        new_thresh_mask: Optional[np.ndarray],
        old_thresh_mask: Optional[np.ndarray],
    ) -> go.Figure:
        scale, data_min, data_max, zmin, zmax, colorscale = color_scale_info
        print("updated 2d")
        fig = self.fig

        if fig is None:
            # raise Error
            fig = self.build_figure(
                C=C,
                labels=labels,
                directed=directed,
                color_scale_info=color_scale_info,
            )

        with fig.batch_update():
            if update_type == UpdateType.NODES:

                node_trace = fig.data[self._node_trace_idx]

                helpers._update_node_trace_all(
                    trace=node_trace,
                    labels=labels,
                    size=self.node_size,
                    color=self.node_fill,
                )

                self.fig = fig
                return fig


            traces_list = helpers._get_candidate_edges(edge_trace_idx=self._edge_trace_idx, old_thresh_mask=old_thresh_mask, new_thresh_mask=new_thresh_mask, update_type=update_type, directed=directed, n_nodes=C.shape[0])

            for (i, j), idx in traces_list:
                
                ### GET EDGE's CONN and MASK (BOOL)
                w = C[i, j]
                m = new_thresh_mask[i, j]
                trace = fig.data[idx]
                
                ### HIDE IF MASK == FALSE
                if directed and not m:
                    ann_idx = self._arrow_trace_idx.get((i,j), None)
                    arrow_ann = fig.layout.annotations[ann_idx]
                    arrow_ann.visible = False  # fully hide
                if not m:
                    # trace.opacity = 0.0
                    # trace.hoverinfo = "skip"
                    # trace.text = ""
                    trace.visible = False  # fully hide
                    
                    continue

                ### GET COLOR
                color = helpers._get_edge_color(edge_weight=w, zmin=zmin, zmax=zmax, colorscale=colorscale, default_neg_color=self.default_neg_color, default_pos_color=self.default_pos_color)

                ### GET WIDTH
                width = helpers._get_edge_width(edge_weight=w, scale=scale, width_range=self.edge_width_range)
                # print(f"{width=}")
                ### UPDATE TRACE FOR VISIBLE EDGES. NEED TO FIX OPACITY FOR 2D, LET'S MAKE THIS AN UPDATEABLE VALUE
                helpers._update_edge_trace(trace=trace, edge_weight=w, color=color, width=width, opacity=self.edge_opacity, label1=labels[i], label2=labels[j])

            # UPDATE COLOR BAR (MAKE THIS FUNCTION)
            helpers._update_colorbar(fig, self._colorbar_trace_idx, colorscale, zmin, zmax)
        self.fig = fig
        return fig


class ConnectivityView3D(ConnectivityView, HandlesNodes):
    def __init__(
        self,
        chanlocs,
        node_size: float = 28.0,
        # node_opacity: float=0.75,
        edge_width_range: Tuple[float] = (0.4, 5),
        edge_opacity: float=0.5,
        default_pos_color: str = "red",
        default_neg_color: str = "blue",
        node_fill: str = "lightgreen",
        node_edge: str = "black",
        arc_radius: float= 20.0,
        show_hemi_left: bool=True,
        show_hemi_right: bool=True,
        brain_mesh_opacity: float=0.25
    ) -> None:
        ConnectivityView.__init__(self,default_pos_color=default_pos_color,
            default_neg_color=default_neg_color)
        HandlesNodes.__init__(self,
                              chanlocs=chanlocs, 
                         node_size=node_size, 
                        #  node_opacity=node_opacity,
                         edge_width_range=edge_width_range,
                         edge_opacity=edge_opacity,
                         node_fill=node_fill,
                         node_edge=node_edge,
                         arc_radius=arc_radius
                         )

        self.show_hemi_left = show_hemi_left
        self.show_hemi_right = show_hemi_right
        self.brain_mesh_opacity = brain_mesh_opacity
        self._mesh_trace_idx = {}
       
    def update_locs(self, chanlocs):
           # Parse → sx, sy, sz, labels
        sx, sy, sz, labs = helpers.parse_channel_locs(chanlocs)

        # 2D normalized topography
        self.locs = helpers.compute_locs_3d(sx=sx, sy=sy, sz=sz)

    def _build_node_trace(self, fig, labels):
        # ========= NODES =========
        x, y, z = self.locs[:, 0], self.locs[:, 1], self.locs[:, 2]
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode="markers+text",
            text=labels,
            textposition="top center",
            textfont=dict(size=10, color="black"),
            marker=dict(size=self.node_size, color=self.node_fill),
            name="Electrodes"
        ))
        self._node_trace_idx = len(fig.data) - 1
        print(self._node_trace_idx)


    def _build_base_trace(self, fig, brain_mesh: Optional[pv.PolyData]):
        if brain_mesh is not None and pv is not None and brain_mesh.n_points > 0:
            pts = np.asarray(brain_mesh.points)
            faces_np = np.asarray(brain_mesh.faces)
            faces = faces_np.reshape(-1, 4)[:, 1:4].astype(int)

            # ========= LEFT HEMISPHERE =========
            left_mask = pts[:, 0] < 0
            left_faces_mask = left_mask[faces].all(axis=1)
            lf = faces[left_faces_mask]

            if lf.size > 0:
                # remap indices
                new_idx = np.flatnonzero(left_mask)
                remap = {old: new for new, old in enumerate(new_idx)}
                lf_remap = np.vectorize(remap.get)(lf)

                left_pts = pts[left_mask]
                fig.add_trace(go.Mesh3d(
                    x=left_pts[:, 0], 
                    y=left_pts[:, 1], 
                    z=left_pts[:, 2],
                    i=lf_remap[:, 0], 
                    j=lf_remap[:, 1], 
                    k=lf_remap[:, 2],
                    color="lightgray", 
                    opacity=self.brain_mesh_opacity, 
                    name="Brain_L"
                ))
                self._mesh_trace_idx["left"] = len(fig.data) - 1

            # ========= RIGHT HEMISPHERE =========
            right_mask = pts[:, 0] > 0
            right_faces_mask = right_mask[faces].all(axis=1)
            rf = faces[right_faces_mask]

            if rf.size > 0:
                new_idx = np.flatnonzero(right_mask)
                remap = {old: new for new, old in enumerate(new_idx)}
                rf_remap = np.vectorize(remap.get)(rf)

                right_pts = pts[right_mask]
                fig.add_trace(go.Mesh3d(
                    x=right_pts[:, 0], 
                    y=right_pts[:, 1], 
                    z=right_pts[:, 2],
                    i=rf_remap[:, 0], 
                    j=rf_remap[:, 1], 
                    k=rf_remap[:, 2],
                    color="lightgray", 
                    opacity=self.brain_mesh_opacity, 
                    name="Brain_R"
                ))
                self._mesh_trace_idx["right"] = len(fig.data) - 1

    
    def toggle_hemisphere_visibility(self, fig: go.Figure, left=None, right=None):

        # Coerce to actual booleans (fixes list inputs)
        print(f"{left=}")
        if isinstance(left, list):
            left = left[0]
        if isinstance(right, list):
            right = right[0]

        if left is not None:
            self.show_hemi_left = bool(left)

        if right is not None:
            self.show_hemi_right = bool(right)

        mesh_l_trace = fig.data[self._mesh_trace_idx["left"]]
        mesh_r_trace = fig.data[self._mesh_trace_idx["right"]]
        mesh_l_trace.visible = self.show_hemi_left
        mesh_r_trace.visible = self.show_hemi_right
    

    ### NEED TO FIX INPUTS
    def _get_edge_path(self, i: int, j: int, C: np.ndarray, m: int = 60) -> np.ndarray:
        """
        Build a 3D concave curve from node i to j.
        The curve bulges outward from the origin (shell-like), instead of convex.
        """
        p0 = self.locs[i]
        p1 = self.locs[j]
        chord = p1 - p0
        L = np.linalg.norm(chord)
        if L < 1e-12:
            return np.vstack([p0, p1])

        # Direction of edge
        d = chord / L

        # Perpendicular vector: any vector perpendicular to chord and pointing away from origin
        perp = np.cross(d, p0)  # cross with starting point to push outward
        if np.linalg.norm(perp) < 1e-6:
            perp = np.cross(d, np.array([0.0, 0.0, 1.0]))
        perp /= np.linalg.norm(perp) + 1e-12

        # Arc height
        arc_height = self.arc_radius if self.arc_radius is not None else 0.15 * L

        # Parametric t
        t = np.linspace(0.0, 1.0, m)

        # Base straight line
        base = p0[None, :] + np.outer(t, chord)

        # Concave hump: subtract to point outward from origin
        hump = -arc_height * np.sin(np.pi * t)
        P = base + np.outer(hump, perp)

        # Optional offset for bidirectional edges
        reverse_exists = (np.isfinite(C[j, i]) and abs(C[j, i]) > 1e-12)
        if reverse_exists:
            sign = 1 if i < j else -1
            offset_amt = 0.06 * L * sign
            P += np.outer(np.sin(np.pi * t) * offset_amt, perp)

        return P


    def _collect_arrow(self, P: np.ndarray, w: float):
        """
        Get the position and normalized direction of an arrow for the last segment of P.
        """
        if len(P) < 2:
            return None
        q0, q1 = P[-2], P[-1]
        pos = q1 - 0.05 * (q1 - q0)  # slightly back along edge
        vec = q1 - q0
        L = np.linalg.norm(vec)
        if L < 1e-9:
            return None
        return pos, vec / L


    def _build_arrowhead_trace(self, pos_list, vec_list, sizes, color_scale_info):
        """
        Build a Plotly Cone trace for all collected arrows.
        """
        scale, data_min, data_max, zmin, zmax, colorscale = color_scale_info
        if not pos_list:
            return None
        xs, ys, zs = zip(*pos_list)
        us, vs, ws = zip(*vec_list)
        return go.Cone(
            x=xs, y=ys, z=zs,
            u=us, v=vs, w=ws,
            # value=vals,
            sizeref=float(np.nanmax(sizes)),
            sizemode="absolute",
            showscale=False,
            colorscale=colorscale,
            cmin=zmin,
            cmax=zmax,
            anchor="tip",
            name="arrows"
        )


    def _build_edge_traces(self, C: np.ndarray, directed: bool, labels, color_scale_info, fig: go.Figure):
        """
        Build edge traces with concave curves and arrow cones.
        Add them to the figure and cache their trace indices.
        """
        scale, data_min, data_max, zmin, zmax, colorscale = color_scale_info
        n_nodes = C.shape[0]

        arrow_positions, arrow_vectors, arrow_sizes = [], [], []

        for i, j in helpers._get_ij_iter(n_nodes, directed):
            w = C[i, j]

            # Color and width
            color = helpers._get_edge_color(
                edge_weight=w, zmin=zmin, zmax=zmax, colorscale=colorscale,
                default_neg_color=self.default_neg_color, default_pos_color=self.default_pos_color
            )
            width = helpers._get_edge_width(edge_weight=w, scale=scale, width_range=self.edge_width_range)

            # Edge path
            P = self._get_edge_path(i, j, C)

            # --- Add edge trace ---
            edge_trace = go.Scatter3d(
                x=P[:, 0], y=P[:, 1], z=P[:, 2],
                mode="lines",
                line=dict(color=color, width=width),
                opacity=self.edge_opacity,
                showlegend=False,
                hoverinfo="text",
                text=f"{labels[i]} → {labels[j]}<br>Weight: {w:.3f}",
                name=f"{labels[i]},{labels[j]}"
            )
            fig.add_trace(edge_trace)
            self._edge_trace_idx[(i,j)] = len(fig.data) - 1  # cache index

            # --- Collect arrow ---
            if directed:
                arrow = self._collect_arrow(P, w)
                if arrow:
                    pos, vec = arrow
                    arrow_positions.append(pos)
                    arrow_vectors.append(vec)
                    arrow_sizes.append(width)
                    
                    # Build arrow trace immediately and cache
                    arrow_trace = self._build_arrowhead_trace([pos], [vec], [width], color_scale_info)
                    if arrow_trace:
                        fig.add_trace(arrow_trace)
                        self._arrow_trace_idx[(i,j)] = len(fig.data) - 1  # cache index


    
    def build_figure(
        self,
        C: np.ndarray,
        labels,
        directed: bool,
        color_scale_info,
        brain_mesh: Optional[pv.PolyData]
    ) -> go.Figure:
        scale, data_min, data_max, zmin, zmax, colorscale = color_scale_info
        ### GET HEAD, NOSE, NODES (NODES MAY NEED TO BE SEPARATED) 
        fig = go.Figure()
        self._build_base_trace(fig=fig, brain_mesh=brain_mesh)
        # The node trace is always the *last* one in base:
        

        ### CREATE EDGES
        self._build_edge_traces(C=C, labels=labels,
            directed=directed, color_scale_info=color_scale_info, fig=fig
        )
        self._build_node_trace(fig=fig, labels=labels)

        ### CREATE COLOR BAR
        colorbar_trace_idx = helpers._add_colorbar_trace(fig=fig, colorscale=colorscale,zmin=zmin,zmax=zmax, viz_type=VizType.FIG3D)
        self._colorbar_trace_idx = colorbar_trace_idx

        ### ADD REST OF LAYOUT
        fig.update_layout(
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                aspectmode="data",
            ),
            autosize=True,
            margin=dict(l=0, r=0, t=40, b=0),
            showlegend=False,
            plot_bgcolor="white",
        )

        self.fig = fig
        return fig

    def update_figure(self,
        C: np.ndarray,
        labels, 
        directed: bool,
        update_type:UpdateType,
        color_scale_info,
        new_thresh_mask: Optional[np.ndarray],
        old_thresh_mask: Optional[np.ndarray],
        brain_mesh: Optional[pv.PolyData]
    ) -> go.Figure:
        scale, data_min, data_max, zmin, zmax, colorscale = color_scale_info
        
        fig = self.fig

        if fig is None:
            fig = self.build_figure(
                C=C,
                labels=labels,
                directed=directed,
                color_scale_info=color_scale_info,
                brain_mesh=brain_mesh
            )

        with fig.batch_update():
            self.toggle_hemisphere_visibility(fig, self.show_hemi_left, self.show_hemi_right)
            mesh_l_trace = fig.data[self._mesh_trace_idx["left"]]
            mesh_r_trace = fig.data[self._mesh_trace_idx["right"]]
            mesh_l_trace.opacity = self.brain_mesh_opacity
            mesh_r_trace.opacity = self.brain_mesh_opacity

            print("hemisphere viz")
            if update_type == UpdateType.NODES:
                print(self._node_trace_idx)
                node_trace = fig.data[self._node_trace_idx]

                helpers._update_node_trace_all(
                    trace=node_trace,
                    labels=labels,
                    size=self.node_size,
                    color=self.node_fill,
                    # opacity=self.node_opacity
                )

                self.fig = fig
                return fig
            traces_list = helpers._get_candidate_edges(edge_trace_idx=self._edge_trace_idx, old_thresh_mask=old_thresh_mask, new_thresh_mask=new_thresh_mask, update_type=update_type, directed=directed, n_nodes=C.shape[0])

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
                if directed and not m:
                    arrow_idx = self._arrow_trace_idx[(i, j)]
                    arrow_trace = fig.data[arrow_idx]
                    arrow_trace.visible = False  # fully hide
                if not m:
                    # trace.opacity = 0.0
                    # trace.hoverinfo = "skip"
                    # trace.text = ""
                    trace.visible = False  # fully hide
                    continue

                ### GET COLOR
                color = helpers._get_edge_color(edge_weight=w, zmin=zmin, zmax=zmax, colorscale=colorscale, default_neg_color=self.default_neg_color, default_pos_color=self.default_pos_color)

                ### GET WIDTH
                width = helpers._get_edge_width(edge_weight=w, scale=scale, width_range=self.edge_width_range)

                ### UPDATE TRACE FOR VISIBLE EDGES. NEED TO FIX OPACITY FOR 2D, LET'S MAKE THIS AN UPDATEABLE VALUE
                helpers._update_edge_trace(trace, w, color, width, self.edge_opacity, labels[i], labels[j])

            # UPDATE COLOR BAR (MAKE THIS FUNCTION)
            helpers._update_colorbar(fig, self._colorbar_trace_idx, colorscale, zmin, zmax)
        self.fig = fig
        return fig

    def update_attributes(self, viz_updates):
        self.node_size = viz_updates["node_size_3d"]
        # self.node_opacity = viz_updates["node_opacity_3d"]
        self.edge_opacity = viz_updates["edge_opacity_3d"]
        self.edge_width_range = viz_updates["edge_width_range_3d"]
        self.show_hemi_left = viz_updates["show_hemi_left_3d"]
        self.show_hemi_right = viz_updates["show_hemi_right_3d"]
        self.brain_mesh_opacity = viz_updates["brain_mesh_opacity_3d"]