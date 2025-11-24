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

        # 2D settings
        node_size_2d: float = 10.0,
        edge_size_min_2d: float = 0.4,
        edge_size_max_2d: float = 4.0,

        # 3D settings
        node_size_3d: float = 10.0,
        edge_size_min_3d: float = 0.4,
        edge_size_max_3d: float = 4.0,
        num_arc_points_3d: int = 4,
        show_right_hemi_3d: bool = True,
        show_left_hemi_3d: bool = True,

        show_labels: bool = True,
        default_pos_color: str = "red",
        default_neg_color: str = "blue",
        node_fill: str = "lightgreen",
        node_edge: str = "black",
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

        # -----------------------
        # 2D settings
        # -----------------------
        self.node_size_2d = node_size_2d
        self.edge_size_min_2d = edge_size_min_2d
        self.edge_size_max_2d = edge_size_max_2d

        # -----------------------
        # 3D settings
        # -----------------------
        self.node_size_3d = node_size_3d
        self.edge_size_min_3d = edge_size_min_3d
        self.edge_size_max_3d = edge_size_max_3d
        self.num_arc_points_3d = num_arc_points_3d
        self.show_right_hemi_3d = show_right_hemi_3d
        self.show_left_hemi_3d = show_left_hemi_3d

        # -----------------------
        # Node rendering settings
        # -----------------------
        self.node_fill = node_fill
        self.node_edge = node_edge

        # -----------------------
        # Edge rendering settings
        # -----------------------
        self.default_pos_color = default_pos_color
        self.default_neg_color = default_neg_color
        
        # coordinates + labels (filled by update_xyz)
        self.xyz: np.ndarray = np.empty((0, 3), dtype=float)     # (n, 3)
        self.xy_topo: np.ndarray = np.empty((0, 2), dtype=float) # (n, 2)

        # Use brain_data ONCE to initialize geometry; do not store it.
        self.update_xyz(brain_data.chanlocs)

        # caches
        self.mask_cache: np.ndarray = np.empty((brain_data.n_nodes, brain_data.n_nodes), dtype=bool) 
        self._edge2d_trace_idx = {}
        self._edge3d_trace_idx = {}
        self._colorbar_trace_idx_2d = -999
        self._colorbar_trace_idx_3d = -999
        self._base_2d_traces: Optional[List[go.Scattergl]] = None
        self.fig_2d_cache = None
        self.fig_3d_cache = None
        self.fig_heatmap_cache = None
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
        """Get the current figure based on viz_type."""
        self.fig_2d_cache = None
        self.fig_3d_cache = None
        self.fig_heatmap_cache = None
        if self.viz_type == VizType.FIG2D:
            fig = self.build_figure_2d(
                brain_data=brain_data,
                threshold=threshold,
                curvature=0.25,
            )
            self.fig_2d_cache = fig
            return fig
        elif self.viz_type == VizType.FIG3D:
            fig = self.build_figure_3d(
                brain_data=brain_data,
                threshold=threshold,
            )
            self.fig_3d_cache = fig
            return fig
        elif self.viz_type == VizType.FIGHEATMAP:
            fig = self.build_figure_heatmap(
                brain_data=brain_data,
                threshold=threshold,
            )
            self.fig_heatmap_cache = fig
            return fig
        else:
            return go.Figure()
        
    def update_figure(self, brain_data: BrainData, threshold: Threshold, update_type: UpdateType) -> go.Figure:
        """Get the current figure based on viz_type."""
        if self.viz_type == VizType.FIG2D:
            return self.update_figure_2d(
                brain_data=brain_data,
                threshold=threshold,
                update_type=update_type
            )
        elif self.viz_type == VizType.FIG3D:
            return self.update_figure_3d(
                brain_data=brain_data,
                threshold=threshold,
                update_type=update_type
            )
        elif self.viz_type == VizType.FIGHEATMAP:
            return self.update_figure_heatmap(
                brain_data=brain_data,
                threshold=threshold,
                update_type=update_type
            )
        else:
            return go.Figure()

    # ---------- Visualization methods ----------

    
    # ------------------------------------------------------------------
    # Edges builder
    # ------------------------------------------------------------------
    def _build_base_traces_2d(self) -> List[go.Scattergl]:
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
    
    def _get_edge_path_2d(self, i: int, j: int, use_arcs: bool, curvature: float) -> np.ndarray:
        p0 = self.xy_topo[i]
        p1 = self.xy_topo[j]
        if use_arcs:
            P = self._quad_bezier(p0, p1, curvature, m=60)
        else:
            P = np.vstack([p0, p1])
        return P

    def _build_edge_traces_2d(
        self,
        brain_data: BrainData,
        use_arcs: bool,
        curvature: float,
    ) -> List[go.Scattergl]:
        # get current matrix to get scale and range of data for new traces 
        C = self.get_mat_at_idx(brain_data)
        np.fill_diagonal(C, 0.0)
        scale, data_min, data_max, zmin, zmax = self._get_scale_and_range(C)

        # set up these traces
        edge_traces: List[go.Scattergl] = []
        labels = self.labels

        #  LOOP OVER ALL POSSIBLE TRACES TO CREATE ALL EDGES
        ij_iter = self.get_ij_iter(brain_data.n_nodes, brain_data.directed) 
        for i, j in ij_iter:
            # get connection value
            w = C[i, j]

            ### GET EDGE COLOR (MAYBE FUNCTION??)
            # Normalize weight to signed [data_min, data_max] then to [0,1]
            color = self._get_edge_color(edge=w, data_max=data_max, data_min=data_min)

            ### GET EDGE WIDTH 
            width = self._get_edge_width(edge=w, scale=scale, min_width=self.edge_size_min_2d, max_width=self.edge_size_max_3d)

            ### GET EDGE PATH
            P = self._get_edge_path_2d(i, j, use_arcs=use_arcs, curvature=curvature)

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
            if brain_data.directed and len(P) >= 2:
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
    # ------------------------------------------------------------------
    # Main: figure_2d
    # ------------------------------------------------------------------



    def build_figure_2d(
        self,
        *,
        brain_data: BrainData,
        threshold: Threshold,
        curvature: float = 0.25,
        title: Optional[str] = None,
    ) -> go.Figure:
        ### GET HEAD, NOSE, NODES (NODES MAY NEED TO BE SEPARATED) 
        fig = go.Figure()
        for tr in self._build_base_traces_2d():
            fig.add_trace(tr)

        ### CREATE EDGES
        edge_traces, (zmin, zmax) = self._build_edge_traces_2d(brain_data=brain_data,
            use_arcs=brain_data.directed,
            curvature=curvature,
        )
        for tr in edge_traces:
            fig.add_trace(tr)

        ### CACHE EDGES (i, j) --> idx
        labels = self.labels
        self._edge2d_trace_idx = self._create_cache_edges(labels, fig)

        ### CREATE COLOR BAR
        colorbar_trace_idx = self._add_colorbar_trace(fig=fig, colorscale=self.colorscale,zmin=zmin,zmax=zmax, viz_type=VizType.FIG2D)
        self._colorbar_trace_idx_2d = colorbar_trace_idx

        ### ADD REST OF LAYOUT
        fig.update_layout(
            title=title,
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

    def update_figure_2d( self, *, brain_data: BrainData, threshold: Threshold, update_type:UpdateType) -> go.Figure:
        fig = self.fig_2d_cache
        if fig is None:
            # safety: fall back to full build if cache is missing
            return self.build_figure_2d(brain_data=brain_data, threshold=threshold)
        
        C, mask = threshold.apply_threshold(brain_data, self.conn_idx)
        np.fill_diagonal(C, 0.0)
        scale, data_min, data_max, zmin, zmax = self._get_scale_and_range(C)
        labels = self.labels

        with fig.batch_update():
            old_mask = self.mask_cache
            self.mask_cache = mask.copy()

            traces_list = self._get_candidate_edges(old_mask, mask, update_type, brain_data)

            for (i, j), idx in traces_list:
                ### GET EDGE's CONN and MASK (BOOL)
                w = C[i, j]
                m = mask[i, j]

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
                color = self._get_edge_color(edge=w, data_max=data_max, data_min=data_min)

                ### GET WIDTH
                width = self._get_edge_width(edge=w, scale=scale, min_width=self.edge_size_min_2d, max_width=self.edge_size_max_3d)

                ### UPDATE TRACE FOR VISIBLE EDGES. NEED TO FIX OPACITY FOR 2D, LET'S MAKE THIS AN UPDATEABLE VALUE
                self._update_edge_trace(trace, w, color, width, 0,75, labels[i], labels[j])

            # UPDATE COLOR BAR (MAKE THIS FUNCTION)
            self._update_colorbar(fig, self._colorbar_trace_idx_2d, self.colorscale, zmin, zmax)
        return fig

    
    def _get_edge_path_3d(
        self,
        i: int,
        j: int,
        C: np.ndarray,
        data_min: float,
        data_max: float,
        arc_radius: Optional[float] = None,
        m: int = 60
    ) -> np.ndarray:
        """
        Builds a smooth 3D arc (with optional offset for bidirectional edges).
        Returns P as an (m, 3) NumPy array.
        """

        p0 = self.xyz[i]
        p1 = self.xyz[j]

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



    def _build_edge_traces_3d(
        self,
        brain_data: BrainData,
    ) -> Tuple[List[go.Scatter3d], Optional[go.Cone], Tuple[float, float]]:
        """
        Build all 3D edge traces for the brain network + a single aggregated arrowhead trace.
        Returns (edge_traces, arrow_trace, (zmin, zmax)).
        """

        # ---- GET CURRENT MATRIX AND GLOBAL SCALING INFO ----
        C = self.get_mat_at_idx(brain_data)
        np.fill_diagonal(C, 0.0)
        scale, data_min, data_max, zmin, zmax = self._compute_scale_and_range(C)

        labels = self.labels
        n = brain_data.n_nodes

        # ---- OUTPUT CONTAINERS ----
        edge_traces: List[go.Scatter3d] = []

        # MAYBE TURN ARROWS INTO NUMPYS
        arrow_pos = []   # list of (x, y, z)
        arrow_vec = []   # list of (u, v, w)

        arrow_vals = []
        arrow_size_vals = []

        # ---- LOOP OVER ALL EDGES ----
        for i, j in self.get_ij_iter(n, brain_data.directed):
            w = float(C[i, j])

            # ---- COLOR + WIDTH ----
            color = self._get_edge_color(edge=w, data_max=data_max, data_min=data_min)
            width = self._get_edge_width(
                edge=w,
                scale=scale,
                min_width=self.edge_size_min_3d,
                max_width=self.edge_size_max_3d,
            )

            # ---- COMPUTE EDGE PATH (ARC + OFFSET FOR REVERSE FLOW) ----
            P = self._get_edge_path_3d(i, j, C, data_min, data_max)

            # ---- CREATE EDGE TRACE ---- NEED TO CHANGE OPACITY
            edge_traces.append(
                go.Scatter3d(
                    x=P[:, 0],
                    y=P[:, 1],
                    z=P[:, 2],
                    mode="lines",
                    line=dict(width=width, color=color),
                    opacity=0.75,
                    showlegend=False,
                    hoverinfo="text",
                    text=f"{labels[i]} → {labels[j]}<br>Weight: {w:.3f}",
                    name=f"{i},{j}",
                )
            )

            # ---- COLLECT ARROWHEADS ONLY IF DIRECTED ----
            if brain_data.directed and len(P) >= 2:
                q0 = P[-2]
                q1 = P[-1]

                # Arrowhead position
                pos = q1 - 0.05 * (q1 - q0)
                # Direction vector
                vec = q1 - q0
                L = np.linalg.norm(vec)
                if L < 1e-9:
                    vec = (self.xyz[j] - self.xyz[i])
                    L = np.linalg.norm(vec) + 1e-12
                vec = vec / L

                arrow_pos.append((pos[0], pos[1], pos[2]))
                arrow_vec.append((vec[0], vec[1], vec[2]))


                arrow_vals.append(w)

                # For variable arrow size – scaled by normalized magnitude
                adj = self._normalize_weight(w, data_min, data_max)
                arrow_size_vals.append(max(0.6, 0.6 * adj))

        # ------------------------------------------------------
        #  BUILD THE SINGLE ARROWHEAD CONE TRACE (IF DIRECTED)
        # ------------------------------------------------------

        arrow_trace = None
        if brain_data.directed and arrow_pos:
            xs, ys, zs = zip(*arrow_pos)
            us, vs, ws = zip(*arrow_vec)

            arrow_trace = go.Cone(
                x=xs, y=ys, z=zs,
                u=us, v=vs, w=ws,
                sizemode="absolute",
                sizeref=max(0.5, float(np.nanmax(arrow_size_vals))),
                anchor="tip",
                colorscale=self.colorscale,
                cmin=zmin, cmax=zmax,
                color=arrow_vals,
                showscale=False,
                name="arrows",
            )


        return edge_traces, arrow_trace, (zmin, zmax)


    def _build_base_traces_3d(self, brain_data):
        if brain_data.brain_mesh is not None and pv is not None and brain_data.brain_mesh.n_points > 0:
            pts = np.asarray(brain_data.brain_mesh.points)
            faces_np = np.asarray(brain_data.brain_mesh.faces)
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


    def build_figure_3d(
        self,
        *,
        brain_data: BrainData,
        title: Optional[str] = None,
    ) -> go.Figure:
        fig = go.Figure()
         ### CREATE BRAIN_MESH and NODES
        for tr in self._build_base_traces_3d(brain_data):
            fig.add_trace(tr)

        ### CREATE EDGES
        edge_traces, arrow_trace, (zmin, zmax) = self._build_edge_traces_3d(brain_data=brain_data)
        for tr in edge_traces:
            fig.add_trace(tr)
        for tr in arrow_trace:
            fig.add_trace(tr)

        ### CACHE EDGE IDX
        labels = self.labels
        self._edge23_trace_idx = self._create_cache_edges(labels, fig)

        ### COLOR BAR
        self._add_colorbar_trace(fig=fig, colorscale=self.colorscale,zmin=zmin,zmax=zmax, viz_type=VizType.FIG3D)

        fig.update_layout(
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                aspectmode="data"
            ),
            autosize=True,
            margin=dict(l=0, r=0, t=40, b=0),
            legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.02),
            title=title or "3D Connectivity"
        )
        return fig

    def update_figure_3d(
        self,
        *,
        brain_data: BrainData,
        threshold: Threshold,
        line_width: float = 3.0,
        opacity: float = 0.6,
        update_type:UpdateType
    ) -> go.Figure:
        """Restyle edges and colorbar in the existing 3D figure without rebuilding geometry."""
        fig = self.fig_3d_cache
        if fig is None:
            # safety: fall back to full build if cache is missing
            return self.build_figure_3d(brain_data=brain_data, threshold=threshold)

        C, mask = threshold.apply_threshold(brain_data, self.conn_idx)
        np.fill_diagonal(C, 0.0)
        scale, data_min, data_max, zmin, zmax = self._get_scale_and_range(C)
        labels = self.labels


        with fig.batch_update():
            # restyle per-edge traces
            old_mask = self.mask_cache
            self.mask_cache = mask.copy()

            traces_list = self._get_candidate_edges(old_mask, mask, update_type, brain_data)

            for (i, j), idx in traces_list:
                w = float(C[i, j])
                m = mask[i, j]
                ### GET TRACE
                trace = fig.data[idx]

                ### HIDE IF MASK == FALSE
                if not m:
                    trace.opacity = 0.0
                    trace.hoverinfo = "skip"
                    trace.text = ""
                    continue

                ### GET COLOR
                color = self._get_edge_color(edge=w, data_max=data_max, data_min=data_min)

                ### GET WIDTH
                width = self._get_edge_width(edge=w, scale=scale, min_width=self.edge_size_min_2d, max_width=self.edge_size_max_3d)

                ### UPDATE TRACE FOR VISIBLE EDGES. NEED TO FIX OPACITY FOR 2D, LET'S MAKE THIS AN UPDATEABLE VALUE
                self._update_edge_trace(trace, w, color, width, 0,75, labels[i], labels[j])

            # UPDATE COLOR BAR (MAKE THIS FUNCTION)
            self._update_colorbar(fig, self._colorbar_trace_idx_2d, self.colorscale, zmin, zmax)
        return fig


    def build_figure_heatmap(
        self,
        *,
        threshold: Threshold,
        brain_data: BrainData,
    ) -> go.Figure:
        C, mask = threshold.apply_threshold(brain_data, self.conn_idx)
        self.mask_cache = mask

        fig = go.Figure()

        # Color range: compute full-data min/max then map color_min/color_max (0..1) into that range
        scale, data_min, data_max, zmin, zmax = self._get_scale_and_range(C)

        ### CREATE HEATMAP, WE DO NOT NEED TO CREATE IT'S OWN COLORBAR
        fig.add_trace(go.Heatmap(
            z=C,
            x=self.labels,
            y=self.labels,
            colorscale=self.colorscale,
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

    def update_figure_heatmap(
        self,
        *,
        brain_data: BrainData,
        threshold: Threshold,
        update_type:UpdateType
    ) -> go.Figure:

        C, mask = threshold.apply_threshold(brain_data, self.conn_idx)

        scale, data_min, data_max, zmin, zmax = self._get_scale_and_range(C)

        self.fig_heatmap_cache.update_traces(
            z=C,
            zmin=zmin,
            zmax=zmax,
            colorscale=self.colorscale,
            selector=dict(name="main"),
        )


        return self.fig_heatmap_cache