from __future__ import annotations
from dataclasses import dataclass, field
from typing import Iterable, Optional, Tuple, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.colors as plc

import analysis.threshold as thresh
from utils.braindata import BrainData
try:
    import pyvista as pv
except Exception:  # make pv optional
    pv = None


def _rgba_from_color(col: str, strength: float) -> str:
    """Return an 'rgba(r,g,b,a)' string for a given hex or named color and strength in [0,1].

    Strength controls the alpha channel; for hex colors we parse RGB. For a small set
    of named colors we map to RGB; otherwise we fall back to the original color string
    (which Plotly will accept) and append an alpha via rgba if possible.
    """
    strength = float(np.clip(strength, 0.0, 1.0))
    if not isinstance(col, str):
        return f"rgba(0,0,0,{strength:.3f})"
    c = col.strip()
    # hex form
    if c.startswith("#") and len(c) in (7, 4):
        try:
            if len(c) == 7:
                r = int(c[1:3], 16)
                g = int(c[3:5], 16)
                b = int(c[5:7], 16)
            else:
                r = int(c[1] * 2, 16)
                g = int(c[2] * 2, 16)
                b = int(c[3] * 2, 16)
            return f"rgba({r},{g},{b},{strength:.3f})"
        except Exception:
            return c

    # basic named colors fallback
    named = {
        "red": (255, 0, 0),
        "blue": (0, 0, 255),
        "black": (0, 0, 0),
        "white": (255, 255, 255),
        "lightgreen": (144, 238, 144),
        "green": (0, 128, 0),
    }
    lc = c.lower()
    if lc in named:
        r, g, b = named[lc]
        return f"rgba({r},{g},{b},{strength:.3f})"

    # last resort: return original color (Plotly may accept strings like 'rgba(...)')
    return c


def _color_from_scale(name: str, t: float) -> str:
    """Return an rgb hex string (e.g. '#rrggbb') sampled from the named plotly colorscale at t in [0,1].

    Falls back to Viridis if the named scale isn't found. Performs linear interpolation in RGB space.
    """
    t = float(np.clip(t, 0.0, 1.0))
    if not isinstance(name, str) or not name:
        name = "Viridis"

    seq = getattr(plc.sequential, name, None)
    if seq is None or len(seq) == 0:
        # try diverging
        seq = getattr(plc.diverging, name, None)
    if seq is None or len(seq) == 0:
        # fallback
        seq = plc.sequential.Viridis

    # seq is a list of color strings (hex or rgb). Normalize to hex '#rrggbb'.
    def _to_rgb_tuple(cstr: str):
        s = cstr.strip()
        if s.startswith("#"):
            if len(s) == 7:
                return int(s[1:3], 16), int(s[3:5], 16), int(s[5:7], 16)
            if len(s) == 4:
                return int(s[1]*2, 16), int(s[2]*2, 16), int(s[3]*2, 16)
        # try 'rgb(r,g,b)'
        if s.startswith("rgb"):
            try:
                inside = s[s.find("(")+1:s.find(")")]
                parts = [int(p.strip()) for p in inside.split(",")]
                return tuple(parts[:3])
            except Exception:
                pass
        # otherwise fallback to black
        return (0, 0, 0)

    # position in scale
    n = len(seq)
    if n == 1:
        r, g, b = _to_rgb_tuple(seq[0])
        return f"rgb({r},{g},{b})"

    pos = t * (n - 1)
    i = int(np.floor(pos))
    j = min(i + 1, n - 1)
    frac = pos - i
    c0 = _to_rgb_tuple(seq[i])
    c1 = _to_rgb_tuple(seq[j])
    r = int(round((1 - frac) * c0[0] + frac * c1[0]))
    g = int(round((1 - frac) * c0[1] + frac * c1[1]))
    b = int(round((1 - frac) * c0[2] + frac * c1[2]))
    return f"rgb({r},{g},{b})"


@dataclass
class Channel:
    x: float
    y: float
    label: Optional[str] = None
    # z is optional for 3D; if absent, zeros are assumed
    z: Optional[float] = None


class ConnectivityVisualizer:
    """
    One object to hold data + build both interactive 2D and 3D connectivity figures.
    """
    def __init__(
        self,
        brain_data: BrainData,
        conn_idx: int = 0,
        threshold: float = 0.5,
        threshold_type: Optional[str] = None,
        colorscale: str = "Viridis",
        alpha: float = 5.0,
        conn_min: float = 0.0,
        conn_max: float = 1.0,
        node_size: float = 10.0,
        show_labels: bool = True,
        default_pos_color: str = "red",
        default_neg_color: str = "blue",
        node_fill: str = "lightgreen",
        node_edge: str = "black",
        viz_type: str = "2D",
    ) -> None:
        # ---- config fields ----
        self.conn_idx: int = conn_idx
        self.threshold: float = threshold
        self.threshold_type: Optional[str] = threshold_type
        self.colorscale: str = colorscale
        self.alpha: float = alpha
        self.conn_min: float = conn_min
        self.conn_max: float = conn_max
        self.node_size: float = node_size
        self.show_labels: bool = show_labels
        self.default_pos_color: str = default_pos_color
        self.default_neg_color: str = default_neg_color
        self.node_fill: str = node_fill
        self.node_edge: str = node_edge
        self.viz_type: str = viz_type

        # ---- derived / cached fields ----

        # coordinates + labels (filled by update_xyz)
        self.xyz: np.ndarray = np.empty((0, 3), dtype=float)     # (n, 3)
        self.xy_topo: np.ndarray = np.empty((0, 2), dtype=float) # (n, 2)

        # Use brain_data ONCE to initialize geometry; do not store it.
        self.update_xyz(brain_data.chanlocs)

    # --------- Boilerplate ----------

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"conn_idx={self.conn_idx}, "
            f"threshold={self.threshold}, "
            f"threshold_type={self.threshold_type!r}, "
            f"colorscale={self.colorscale!r}, "
            f"alpha={self.alpha}, "
            f"conn_min={self.conn_min}, "
            f"conn_max={self.conn_max}, "
            f"node_size={self.node_size}, "
            f"show_labels={self.show_labels}, "
            f"viz_type={self.viz_type!r}"
            f")"
        )
    
    def __eq__(self, other) -> bool:
        """Two visualizers are considered equal if all configuration fields match."""
        if not isinstance(other, ConnectivityVisualizer):
            return False

        return (
            self.conn_idx == other.conn_idx
            and self.threshold == other.threshold
            and self.threshold_type == other.threshold_type
            and self.colorscale == other.colorscale
            and self.alpha == other.alpha
            and self.conn_min == other.conn_min
            and self.conn_max == other.conn_max
            and self.node_size == other.node_size
            and self.show_labels == other.show_labels
            and self.default_pos_color == other.default_pos_color
            and self.default_neg_color == other.default_neg_color
            and self.node_fill == other.node_fill
            and self.node_edge == other.node_edge
            and self.viz_type == other.viz_type
        )
    
    # ---------- Shared helpers ----------
    @staticmethod
    def _max_conn_scale(C: np.ndarray) -> float:
        D = C.copy()
        np.fill_diagonal(D, 0.0)
        if np.any(np.isfinite(D)):
            m = np.nanmax(np.abs(D))
            return float(m) if m > 0 else 1.0
        return 1.0

    @staticmethod
    def _quad_bezier(p0: np.ndarray, p1: np.ndarray, curvature: float = 0.25, m: int = 40) -> np.ndarray:
        d = p1 - p0
        L = np.linalg.norm(d)
        if L < 1e-12:
            return np.repeat(p0[None, :], m, axis=0)
        u = d / L
        perp = np.array([-u[1], u[0]])
        c = (p0 + p1) / 2.0 + curvature * L * perp
        t = np.linspace(0, 1, m)[:, None]
        return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * c + t ** 2 * p1

    @staticmethod
    def _arc_points_origin_plane(p0: np.ndarray, p1: np.ndarray, R: Optional[float], m: int = 20) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Arc between p0 and p1 in the plane through {p0, p1, origin}.
        If R is None, an automatic radius slightly larger than the chord’s
        minimum (d/2) is chosen to create a gentle arc.
        """
        p0 = p0.astype(float); p1 = p1.astype(float)
        chord = p1 - p0
        d = np.linalg.norm(chord)
        if d < 1e-12:
            return np.array([p0[0]]), np.array([p0[1]]), np.array([p0[2]])

        n = np.cross(p0, p1)
        nn = np.linalg.norm(n)
        if nn < 1e-12:
            # collinear with origin; fall back to straight
            t = np.linspace(0, 1, m)
            P = p0[None, :] + t[:, None] * chord[None, :]
            return P[:, 0], P[:, 1], P[:, 2]
        n /= nn

        u = chord / d
        v = np.cross(n, u); v /= np.linalg.norm(v)

        # Auto radius: just above the feasibility limit (d/2), scaled by chord length
        if R is None:
            R = 0.55 * d + (d / 2.0)  # = 1.05 * d/2 (gentle bow). Adjust factor if you want more curvature.
        R = max(R, d / 2.0 + 1e-9)

        h = np.sqrt(max(R * R - (d / 2) ** 2, 0.0))
        mid = 0.5 * (p0 + p1)
        center = mid + h * v  # pick +v; flip to -v if you prefer the other bow direction

        # Angles in (u,v) frame
        def angle(vec):
            return np.arctan2(np.dot(vec, v), np.dot(vec, u))

        theta0 = angle(p0 - center)
        theta1 = angle(p1 - center)
        dtheta = (theta1 - theta0 + np.pi) % (2 * np.pi) - np.pi  # shortest arc

        thetas = theta0 + np.linspace(0, dtheta, m)
        cs, ss = np.cos(thetas), np.sin(thetas)
        pts = center[None, :] + R * cs[:, None] * u[None, :] + R * ss[:, None] * v[None, :]
        return pts[:, 0], pts[:, 1], pts[:, 2]

    # ---------- Public API ----------
    def apply_threshold(self, C: np.ndarray) -> np.ndarray:
        conn_normalized = (C - np.min(C)) / (np.max(C) - np.min(C) + 1e-12)
        
        if self.threshold_type == "Basic":
            # Convert percentage to normalized threshold
            norm_threshold = self.threshold / 100.0
            return thresh.get_basic_map(conn_normalized, norm_threshold)
        elif self.threshold_type == "Statistical Test":
            # Convert percentage to normalized threshold
            norm_alpha = self.alpha / 100.0
            return thresh.get_stattest_mask(conn_normalized, norm_alpha)
        elif self.threshold_type == "MST" or self.threshold_type == "Minimum Spanning Tree":  # Minimum Spanning Tree
            return thresh.get_mst_map(conn_normalized)
    

    # ---------- Utils ----------
    def get_conn_matrix(self, brain_data: BrainData) -> np.ndarray:
        C = brain_data.conn_mat[self.conn_idx, :, : ].copy()
        return C
    
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


    def update_fields(
        self,
        *,
        brain_data: Optional[BrainData] = None,
        viz_type: Optional[str] = None,
        conn_idx: Optional[int] = None,
        thresh_type: Optional[str] = None,
        thresh_value: Optional[float] = None,
        color_name: Optional[str] = None,
        conn_min: Optional[float] = None,
        conn_max: Optional[float] = None,
        alpha: Optional[float] = None,
    ) -> None:
        """Update multiple fields of the visualizer at once."""
        if brain_data is not None:
            self.update_xyz(brain_data.chanlocs)

        if viz_type is not None:
            self.viz_type = viz_type
        if conn_idx is not None:
            self.conn_idx = int(conn_idx)
        if thresh_type is not None:
            self.threshold_type = thresh_type
        if thresh_value is not None:
            self.threshold = float(thresh_value)
        if color_name is not None:
            self.colorscale = color_name
        if conn_min is not None:
            self.conn_min = float(conn_min)
        if conn_max is not None:
            self.conn_max = float(conn_max)
        if alpha is not None:
            self.alpha = float(alpha)


    def get_figure(self, brain_data: BrainData) -> go.Figure:
        """Get the current figure based on viz_type."""
        if self.viz_type == "2D":
            return self.figure_2d(
                brain_data=brain_data,
                use_arcs=True,
                curvature=0.25,
                lw_min=0.5,
                lw_max=4.0,
                # title=None,
            )
        elif self.viz_type == "3D":
            return self.figure_3d(
                brain_data=brain_data,
                # title=None,
            )
        elif self.viz_type == "Heatmap":
            return self.figure_heatmap(
                brain_data=brain_data,
                # title=None,
            )
        else:
            return go.Figure()

    # ---------- Visualization methods ----------

    def figure_2d(
        self,
        *,
        brain_data: BrainData,
        use_arcs: bool = True,
        curvature: float = 0.25,
        lw_min: float = 0.5,
        lw_max: float = 4.0,
        title: Optional[str] = None,
    ) -> go.Figure:
        """
        Interactive 2D EEG-style top view.
        
        Args:
            threshold: Basic threshold value (absolute) or percentage (if threshold_type is set)
            threshold_type: If set to "Basic" or "Minimum Spanning Tree", applies that thresholding
        """
        C = self.get_conn_matrix(brain_data)
        np.fill_diagonal(C, 0.0)
        
        # Apply thresholding if specified
        if self.threshold_type:
            # apply_threshold expects threshold as percentage when threshold_type == 'Basic'
            mask = self.apply_threshold(C)
            C = C * mask
        
        scale = self._max_conn_scale(C)
        # data range (signed) for mapping t in [0,1]
        if np.any(np.isfinite(C)):
            data_min = float(np.nanmin(C))
            data_max = float(np.nanmax(C))
        else:
            data_min, data_max = -1.0, 1.0
        # Map normalized conn_min/conn_max (0..1) into actual data range for colorbar limits
        zmin = data_min + float(np.clip(self.conn_min, 0.0, 1.0)) * (data_max - data_min)
        zmax = data_min + float(np.clip(self.conn_max, 0.0, 1.0)) * (data_max - data_min)
        if zmin == zmax:
            zmin, zmax = zmin - 1e-6, zmax + 1e-6
        x, y = self.xy_topo[:, 0], self.xy_topo[:, 1]
        labels = self.labels

        fig = go.Figure()

        # head outline + nose
        theta = np.linspace(0, 2 * np.pi, 256)
        fig.add_trace(go.Scatter(x=np.cos(theta), y=np.sin(theta), mode="lines",
                                 line=dict(color="black", width=2), hoverinfo="skip", name="Head"))
        fig.add_trace(go.Scatter(x=[0.10, 0.00, -0.10], y=[1.00, 1.10, 1.00], mode="lines",
                                 line=dict(color="black", width=2), hoverinfo="skip", showlegend=False))

        # edges
        for i in range(self.n):
            for j in range(self.n):
                if i == j:
                    continue
                w = C[i, j]
                # Skip NaNs and near-zero entries
                if not np.isfinite(w) or abs(w) < 1e-12:
                    continue

                # If threshold_type was not used, the caller may have given an absolute threshold
                if not self.threshold_type and self.threshold > 0 and abs(w) < self.threshold:
                    continue

                # Signed mapping: normalize w into global [data_min,data_max] then map via conn_min/conn_max
                t_global = (w - data_min) / (max((data_max - data_min), 1e-12))
                try:
                    adj = (t_global - self.conn_min) / max((self.conn_max - self.conn_min), 1e-12)
                except Exception:
                    adj = t_global
                adj = float(np.clip(adj, 0.0, 1.0))

                # Color sampled from colorscale using sign-aware adj
                try:
                    color = _color_from_scale(self.colorscale, adj)
                except Exception:
                    base_color = self.default_pos_color if w >= 0 else self.default_neg_color
                    color = _rgba_from_color(base_color, max(0.12, 0.25 + 0.75 * adj))

                # width reflects absolute magnitude relative to max abs
                width = lw_min + (abs(w) / max(scale, 1e-12)) * (lw_max - lw_min)
                p0 = self.xy_topo[i]
                p1 = self.xy_topo[j]
                P = self._quad_bezier(p0, p1, curvature, m=60) if use_arcs else np.vstack([p0, p1])
                fig.add_trace(go.Scatter(
                    x=P[:, 0], y=P[:, 1], mode="lines",
                    line=dict(color=color, width=width),
                    opacity=0.75, showlegend=False,
                    hoverinfo="text",
                    text=f"{labels[i]} → {labels[j]}<br>Weight: {w:.3f}",
                ))
                if brain_data.is_directed:
                    # arrowhead near the end (simple 2-point)
                    if len(P) >= 2:
                        q0, q1 = P[-2], P[-1]
                        fig.add_annotation(
                            x=q1[0], y=q1[1],
                            ax=q0[0], ay=q0[1],
                            xref="x", yref="y", axref="x", ayref="y",
                            showarrow=True, arrowhead=2, arrowsize=1.2, arrowwidth=width/2, arrowcolor=color,
                            opacity=0.8
                        )

        # nodes
        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode="markers+text" if self.show_labels else "markers",
            text=self.labels if self.show_labels else None,
            textposition="middle center",
            marker=dict(size=self.node_size, color=self.node_fill, line=dict(color=self.node_edge, width=2)),
            hovertext=self.labels, hoverinfo="text",
            name="Electrodes"
        ))

        # colorbar: add an invisible marker trace to show colorscale for edges
        try:
            fig.add_trace(go.Scatter(
                x=[None], y=[None], mode="markers",
                marker=dict(colorscale=self.colorscale, cmin=zmin, cmax=zmax, color=[zmin, zmax], showscale=True,
                            colorbar=dict(title="Conn", len=0.45, thickness=12)),
                showlegend=False, hoverinfo="none",
            ))
        except Exception:
            pass

        fig.update_layout(
            xaxis=dict(
                visible=False,
                scaleanchor="y",   # lock aspect ratio
                scaleratio=1,      # ensure equal scaling
            ),
            yaxis=dict(visible=False),
            autosize=True,        # let container scale overall size
            margin=dict(l=0, r=0, t=40, b=0),
            showlegend=False,
            plot_bgcolor="white",
        )
        return fig

    def figure_3d(
        self,
        *,
        brain_data: BrainData,
        arc_radius: Optional[float] = None,   # None -> automatic radius; set a float to force
        arc_samples: int = 4,  # Reduced from 24 for faster rendering
        line_width: float = 3.0,
        opacity: float = 0.6,
        title: Optional[str] = None,
    ) -> go.Figure:
        """
        Interactive 3D connectivity. If edge_style == 'arc', edges curve in the plane
        defined by (p0, p1, origin). If arc_radius is None, a gentle automatic radius
        is chosen per edge; otherwise your fixed radius is used.
        
        Args:
            threshold: Threshold value (absolute) or percentage (if threshold_type is set)
            threshold_type: If set to "Basic" or "Minimum Spanning Tree", applies that thresholding
        """
        
        # Get connectivity matrix (potentially thresholded)
        C = self.get_conn_matrix(brain_data)
        if self.threshold_type:
            mask = self.apply_threshold(C)
            C = C * mask
        
        np.fill_diagonal(C, 0.0)

        fig = go.Figure()

        # mesh (optional)
        if brain_data.brain_mesh is not None and pv is not None and brain_data.brain_mesh.n_points > 0:
            pts = np.asarray(brain_data.brain_mesh.points)
            faces_np = np.asarray(brain_data.brain_mesh.faces)
            faces = faces_np.reshape(-1, 4)[:, 1:4].astype(int)
            fig.add_trace(go.Mesh3d(
                x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                color="lightgray", opacity=0.25, flatshading=True,
                lighting=dict(ambient=0.6, diffuse=0.6, specular=0.1),
                name="Brain"
            ))

        # nodes
        x, y, z = self.xyz[:, 0], self.xyz[:, 1], self.xyz[:, 2]
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode="markers+text" if self.show_labels else "markers",
            text=self.labels if self.show_labels else None,
            textposition="top center",
            textfont=dict(size=10, color="black"),
            marker=dict(size=self.node_size),
            name="Electrodes"
        ))

        # edges (support directed and conn range scaling)
        scale = self._max_conn_scale(C)
        # signed data range for mapping
        if np.any(np.isfinite(C)):
            data_min = float(np.nanmin(C))
            data_max = float(np.nanmax(C))
        else:
            data_min, data_max = -1.0, 1.0
        # Map normalized conn_min/conn_max (0..1) into actual data range for colorbar limits
        zmin = data_min + float(np.clip(self.conn_min, 0.0, 1.0)) * (data_max - data_min)
        zmax = data_min + float(np.clip(self.conn_max, 0.0, 1.0)) * (data_max - data_min)
        if zmin == zmax:
            zmin, zmax = zmin - 1e-6, zmax + 1e-6
        line_xs, line_ys, line_zs = [], [], []
        arrow_x, arrow_y, arrow_z, arrow_adj, arrow_size = [], [], [], [], []
        arrow_dir_u, arrow_dir_v, arrow_dir_w = [], [], []
        arrow_vals = []

        for i in range(self.n):
            p0 = self.xyz[i]
            targets = range(self.n) if brain_data.is_directed else range(i + 1, self.n)
            for j in targets:
                if i == j:
                    continue
                w = float(C[i, j])
                if not np.isfinite(w) or abs(w) < 1e-12:
                    continue

                if not self.threshold_type and self.threshold > 0 and w < self.threshold:
                    continue

                # map signed value to 0..1 over data_min..data_max then apply conn window
                t_global = (w - data_min) / max((data_max - data_min), 1e-12)
                adj = (t_global - self.conn_min) / max((self.conn_max - self.conn_min), 1e-12)
                adj = float(np.clip(adj, 0.0, 1.0))

                p1 = self.xyz[j]
                # sample color from the provided colorscale
                try:
                    edge_col = _color_from_scale(self.colorscale, adj)
                except Exception:
                    edge_col = _rgba_from_color('red' if w >= 0 else 'blue', max(0.12, 0.25 + 0.75 * adj))

                # detect reciprocal edge and compute an offset perpendicular to the chord
                reverse_exists = (np.isfinite(C[j, i]) and abs(C[j, i]) > 1e-12)
                sign = 0
                if reverse_exists:
                    sign = 1 if i < j else -1

                chord = p1 - p0
                L = np.linalg.norm(chord)
                if L < 1e-12:
                    perp = np.array([0.0, 0.0, 0.0])
                else:
                    d = chord / L
                    perp = np.cross(d, np.array([0.0, 0.0, 1.0]))
                    if np.linalg.norm(perp) < 1e-6:
                        perp = np.cross(d, np.array([0.0, 1.0, 0.0]))
                    perp = perp / (np.linalg.norm(perp) + 1e-12)

                offset_amt = 0.06 * L * sign

                X, Y, Z = self._arc_points_origin_plane(p0, p1, arc_radius, m=max(int(arc_samples), 2))
                # apply lateral offset to the arc shape (keep endpoints fixed)
                if sign != 0:
                    tvals = np.linspace(0.0, 1.0, len(X))
                    # envelope zero at endpoints, max at midpoint -> sin(pi*t)
                    env = np.sin(np.pi * tvals)
                    X = np.array(X) + perp[0] * offset_amt * env
                    Y = np.array(Y) + perp[1] * offset_amt * env
                    Z = np.array(Z) + perp[2] * offset_amt * env

                # add each edge as its own trace so we can color it independently
                fig.add_trace(go.Scatter3d(x=list(X), y=list(Y), z=list(Z), mode="lines",
                                            line=dict(width=line_width * (0.6 + 0.8 * adj), color=edge_col),
                                            opacity=opacity, showlegend=False, hoverinfo="text",
                                            text=f"{self.labels[i]} → {self.labels[j]}<br>Weight: {w:.3f}"))

                # add a cone (arrowhead) near the end of the arc
                if brain_data.is_directed and len(X) >= 2:
                    q0 = np.array([X[-2], Y[-2], Z[-2]])
                    q1 = np.array([X[-1], Y[-1], Z[-1]])
                    pos = q1 - 0.05 * (q1 - q0)
                    # store cone base position and direction
                    arrow_x.append(pos[0]); arrow_y.append(pos[1]); arrow_z.append(pos[2])
                    # direction from q0->q1
                    vec = q1 - q0
                    norm = np.linalg.norm(vec)
                    if norm < 1e-9:
                        # fallback to chord direction if segment too small
                        vec = p1 - p0
                        norm = np.linalg.norm(vec) + 1e-12
                    vec = vec / (norm + 1e-12)
                    arrow_adj.append(adj)
                    arrow_vals.append(w)
                    arrow_size.append(max(0.6, 0.6 * adj))
                    arrow_dir_u.append(vec[0]); arrow_dir_v.append(vec[1]); arrow_dir_w.append(vec[2])

        if line_xs:
            fig.add_trace(go.Scatter3d(
                x=line_xs, y=line_ys, z=line_zs,
                mode="lines",
                line=dict(width=line_width),
                opacity=opacity,
                name=("Edges")
            ))

        if brain_data.is_directed and arrow_x:
            # Prefer 3D cone glyphs for arrowheads. We map arrow_adj (0..1) into colorscale for cone coloring.
            try:
                # Color cones using the actual connection values (arrow_vals) and the mapped zmin/zmax
                fig.add_trace(go.Cone(
                    x=arrow_x, y=arrow_y, z=arrow_z,
                    u=arrow_dir_u, v=arrow_dir_v, w=arrow_dir_w,
                    sizemode='absolute', sizeref=max(0.5, float(np.nanmax(arrow_size))),
                    anchor='tip',
                    colorscale=self.colorscale, cmin=zmin, cmax=zmax,
                    color=arrow_vals,
                    showscale=False,
                ))
            except Exception:
                # Fallback: use colored markers sampled from colorscale
                try:
                    # map actual arrow values into normalized [0,1] for sampling the colorscale
                    span = float(zmax - zmin) if zmax != zmin else 1.0
                    marker_colors = [_color_from_scale(self.colorscale, float(np.clip((val - zmin) / span, 0.0, 1.0))) for val in arrow_vals]
                except Exception:
                    marker_colors = ['red' if v >= 0.5 else 'blue' for v in arrow_vals]
                fig.add_trace(go.Scatter3d(
                    x=arrow_x, y=arrow_y, z=arrow_z,
                    mode="markers",
                    marker=dict(size= [max(4, s*6) for s in arrow_size], color=marker_colors),
                    name="Direction",
                    hoverinfo="skip",
                ))

        # colorbar for 3D: invisible marker trace to display colorscale legend
        try:
            fig.add_trace(go.Scatter3d(
                x=[None], y=[None], z=[None], mode="markers",
                marker=dict(colorscale=self.colorscale, cmin=zmin, cmax=zmax, color=[zmin, zmax], showscale=True,
                            colorbar=dict(title="Conn", len=0.45, thickness=12)),
                showlegend=False, hoverinfo="none",
            ))
        except Exception:
            pass

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
    
    def figure_heatmap(
        self,
        *,
        brain_data: BrainData,
    ) -> go.Figure:
        """
        Connectivity heatmap (n x n) with faint empty grid rectangles for missing/thresholded cells.
        
        Args:
            threshold: Threshold value (absolute) or percentage (if threshold_type is set)
            threshold_type: If set to "Basic" or "Minimum Spanning Tree", applies that thresholding
        """
        C = self.get_conn_matrix(brain_data)
        bg_color = "rgba(230,230,230,0.3)"

        mask = self.apply_threshold(C)
        C = C * mask

        # Color range: compute full-data min/max then map conn_min/conn_max (0..1) into that range
        if np.any(np.isfinite(C)):
            data_min = float(np.nanmin(C))
            data_max = float(np.nanmax(C))
        else:
            data_min, data_max = -1.0, 1.0

         # Map normalized conn_min/conn_max (0..1) into actual data range
        zmin = data_min + float(np.clip(self.conn_min, 0.0, 1.0)) * (data_max - data_min)
        zmax = data_min + float(np.clip(self.conn_max, 0.0, 1.0)) * (data_max - data_min)
        if zmin == zmax:
            zmin, zmax = zmin - 1e-6, zmax + 1e-6

        # --- Background layer: faint grid of boxes ---
        bg = np.full_like(C, np.nan)
        bg[np.isnan(C)] = 0  # only fill where data is missing

        fig = go.Figure()

        fig.add_trace(go.Heatmap(
            z=bg,
            x=self.labels,
            y=self.labels,
            colorscale=[[0, bg_color], [1, bg_color]],  # constant faint gray
            showscale=False,
            xgap=0.5,
            ygap=0.5,
            hoverinfo="skip"
        ))

        # --- Main heatmap ---
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
