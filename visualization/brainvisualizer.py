from __future__ import annotations
from dataclasses import dataclass, field
from typing import Iterable, Optional, Tuple, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.colors as plc

import analysis.threshold as thresh

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


@dataclass
class ConnectivityVisualizer:
    """
    One object to hold data + build both interactive 2D and 3D connectivity figures.
    """
    conn: np.ndarray
    chanlocs: Union[pd.DataFrame, Iterable[Channel], np.ndarray, list]
    brain_mesh: Optional[object] = None

    # derived / cached
    n: int = field(init=False)
    labels: np.ndarray = field(init=False)
    xyz: np.ndarray = field(init=False)      # (n, 3)
    xy_topo: np.ndarray = field(init=False)  # (n, 2) normalized to head circle

    def __post_init__(self):
        self.conn = np.asarray(self.conn)
        assert self.conn.ndim == 2 and self.conn.shape[0] == self.conn.shape[1], "conn must be (n,n)"
        self.n = self.conn.shape[0]

        # ---- Parse channel locations into xyz + labels ----
        if isinstance(self.chanlocs, pd.DataFrame):
            sx = self.chanlocs["x"].to_numpy()
            sy = self.chanlocs["y"].to_numpy()
            sz = self.chanlocs["z"].to_numpy() if "z" in self.chanlocs.columns else np.zeros_like(sx)
            if "label" in self.chanlocs.columns:
                labs = self.chanlocs["label"].astype(str).to_numpy()
            else:
                labs = np.arange(self.n).astype(str)
        else:
            # list/ndarray of Channels or rows like [x, y, (z), (label)]
            sx, sy, sz, labs = [], [], [], []
            for row in self.chanlocs:
                if isinstance(row, Channel):
                    sx.append(row.x); sy.append(row.y); sz.append(row.z if row.z is not None else 0.0); labs.append(row.label or "")
                elif isinstance(row, dict):
                    sx.append(float(row["x"])); sy.append(float(row["y"]))
                    sz.append(float(row.get("z", 0.0))); labs.append(str(row.get("label", "")))
                else:
                    # generic sequence
                    x = float(row[0]); y = float(row[1])
                    z = float(row[2]) if len(row) >= 3 and np.isscalar(row[2]) else 0.0
                    lab = str(row[3]) if len(row) >= 4 else (str(row[2]) if len(row) >= 3 and not np.isscalar(row[2]) else "")
                    sx.append(x); sy.append(y); sz.append(z); labs.append(lab)
            sx, sy, sz, labs = map(np.asarray, (sx, sy, sz, labs))
            if labs.size == 0:  # fallback
                labs = np.arange(self.n).astype(str)

        self.labels = labs if labs.size == self.n else np.arange(self.n).astype(str)
        self.xyz = np.column_stack([sx, sy, sz]).astype(float)

        # ---- Precompute normalized 2D topography (EEG top view: flip x) ----
        xs, ys = sx.copy(), sy.copy()
        xs = -xs / (np.max(np.abs(xs)) + 1e-12) * 0.9
        ys =  ys / (np.max(np.abs(ys)) + 1e-12) * 0.9
        self.xy_topo = np.column_stack([xs, ys])

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
    def apply_threshold(self, threshold_type: str = "Basic", threshold_value: float = 50.0) -> np.ndarray:
        """
        Apply thresholding to the connectivity matrix.
        
        Args:
            threshold_type: Either "Basic" or "Minimum Spanning Tree"
            threshold_value: Percentile threshold for basic thresholding (0-100)
        
        Returns:
            Boolean mask of thresholded connections
        """
        conn_normalized = (self.conn - np.min(self.conn)) / (np.max(self.conn) - np.min(self.conn) + 1e-12)
        
        if threshold_type == "Basic":
            # Convert percentage to normalized threshold
            norm_threshold = threshold_value / 100.0
            return thresh.get_basic_map(conn_normalized, norm_threshold)
        else:  # Minimum Spanning Tree
            return thresh.get_mst_map(self.conn)
    
    def figure_2d(
        self,
        *,
        threshold: float = 0.0,
        threshold_type: Optional[str] = None,
        directed: bool = True,
        use_arcs: bool = True,
        curvature: float = 0.25,
        lw_min: float = 0.5,
        lw_max: float = 4.0,
        pos_color: str = "red",
        neg_color: str = "blue",
        node_fill: str = "lightgreen",
        node_edge: str = "black",
        node_size: float = 10.0,
        show_labels: bool = True,
        title: Optional[str] = None,
        conn_min: float = 0.0,
        conn_max: float = 1.0,
        colorscale: str = "Viridis",
    ) -> go.Figure:
        """
        Interactive 2D EEG-style top view.
        
        Args:
            threshold: Basic threshold value (absolute) or percentage (if threshold_type is set)
            threshold_type: If set to "Basic" or "Minimum Spanning Tree", applies that thresholding
        """
        C = self.conn.copy()
        np.fill_diagonal(C, 0.0)
        
        # Apply thresholding if specified
        if threshold_type:
            # apply_threshold expects threshold as percentage when threshold_type == 'Basic'
            mask = self.apply_threshold(threshold_type, threshold)
            C = C * mask
        
        scale = self._max_conn_scale(C)
        # data range (signed) for mapping t in [0,1]
        if np.any(np.isfinite(C)):
            data_min = float(np.nanmin(C))
            data_max = float(np.nanmax(C))
        else:
            data_min, data_max = -1.0, 1.0
        # Map normalized conn_min/conn_max (0..1) into actual data range for colorbar limits
        zmin = data_min + float(np.clip(conn_min, 0.0, 1.0)) * (data_max - data_min)
        zmax = data_min + float(np.clip(conn_max, 0.0, 1.0)) * (data_max - data_min)
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
                if not threshold_type and threshold > 0 and abs(w) < threshold:
                    continue

                # Signed mapping: normalize w into global [data_min,data_max] then map via conn_min/conn_max
                t_global = (w - data_min) / (max((data_max - data_min), 1e-12))
                try:
                    adj = (t_global - conn_min) / max((conn_max - conn_min), 1e-12)
                except Exception:
                    adj = t_global
                adj = float(np.clip(adj, 0.0, 1.0))

                # Color sampled from colorscale using sign-aware adj
                try:
                    color = _color_from_scale(colorscale, adj)
                except Exception:
                    base_color = pos_color if w >= 0 else neg_color
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
                if directed:
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
            mode="markers+text" if show_labels else "markers",
            text=self.labels if show_labels else None,
            textposition="middle center",
            marker=dict(size=node_size, color=node_fill, line=dict(color=node_edge, width=2)),
            hovertext=self.labels, hoverinfo="text",
            name="Electrodes"
        ))

        # colorbar: add an invisible marker trace to show colorscale for edges
        try:
            fig.add_trace(go.Scatter(
                x=[None], y=[None], mode="markers",
                marker=dict(colorscale=colorscale, cmin=zmin, cmax=zmax, color=[zmin, zmax], showscale=True,
                            colorbar=dict(title="Conn", len=0.45, thickness=12)),
                showlegend=False, hoverinfo="none",
            ))
        except Exception:
            pass

        fig.update_layout(
            title=title or "Interactive 2D EEG Connectivity",
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
        threshold: float = 0.9,
        threshold_type: Optional[str] = None,
        show_labels: bool = True,
        electrode_size: float = 4.5,
        edge_style: str = "arc",              # "arc" or "straight"
        arc_radius: Optional[float] = None,   # None -> automatic radius; set a float to force
        arc_samples: int = 4,  # Reduced from 24 for faster rendering
        line_width: float = 3.0,
        opacity: float = 0.6,
        title: Optional[str] = None,
        directed: bool = True,
        conn_min: float = 0.0,
        conn_max: float = 1.0,
        colorscale: str = "Viridis",
    ) -> go.Figure:
        """
        Interactive 3D connectivity. If edge_style == 'arc', edges curve in the plane
        defined by (p0, p1, origin). If arc_radius is None, a gentle automatic radius
        is chosen per edge; otherwise your fixed radius is used.
        
        Args:
            threshold: Threshold value (absolute) or percentage (if threshold_type is set)
            threshold_type: If set to "Basic" or "Minimum Spanning Tree", applies that thresholding
        """
        assert edge_style in ("arc", "straight")
        
        # Get connectivity matrix (potentially thresholded)
        C = self.conn.copy()
        if threshold_type:
            mask = self.apply_threshold(threshold_type, threshold)
            C = C * mask
        
        np.fill_diagonal(C, 0.0)

        fig = go.Figure()

        # mesh (optional)
        if self.brain_mesh is not None and pv is not None and self.brain_mesh.n_points > 0:
            pts = np.asarray(self.brain_mesh.points)
            faces_np = np.asarray(self.brain_mesh.faces)
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
            mode="markers+text" if show_labels else "markers",
            text=self.labels if show_labels else None,
            textposition="top center",
            textfont=dict(size=10, color="black"),
            marker=dict(size=electrode_size),
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
        zmin = data_min + float(np.clip(conn_min, 0.0, 1.0)) * (data_max - data_min)
        zmax = data_min + float(np.clip(conn_max, 0.0, 1.0)) * (data_max - data_min)
        if zmin == zmax:
            zmin, zmax = zmin - 1e-6, zmax + 1e-6
        line_xs, line_ys, line_zs = [], [], []
        arrow_x, arrow_y, arrow_z, arrow_adj, arrow_size = [], [], [], [], []
        arrow_dir_u, arrow_dir_v, arrow_dir_w = [], [], []
        arrow_vals = []

        for i in range(self.n):
            p0 = self.xyz[i]
            targets = range(self.n) if directed else range(i + 1, self.n)
            for j in targets:
                if i == j:
                    continue
                w = float(C[i, j])
                if not np.isfinite(w) or abs(w) < 1e-12:
                    continue

                if not threshold_type and threshold > 0 and w < threshold:
                    continue

                # map signed value to 0..1 over data_min..data_max then apply conn window
                t_global = (w - data_min) / max((data_max - data_min), 1e-12)
                adj = (t_global - conn_min) / max((conn_max - conn_min), 1e-12)
                adj = float(np.clip(adj, 0.0, 1.0))

                p1 = self.xyz[j]
                # sample color from the provided colorscale
                try:
                    edge_col = _color_from_scale(colorscale, adj)
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

                if edge_style == "arc":
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
                    if directed and len(X) >= 2:
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

                else:  # straight edge
                    # straight edge as a short bent segment when reciprocal, otherwise straight line
                    if sign != 0:
                        mid = (p0 + p1) / 2.0 + perp * offset_amt
                        xs_seg = [p0[0], mid[0], p1[0]]
                        ys_seg = [p0[1], mid[1], p1[1]]
                        zs_seg = [p0[2], mid[2], p1[2]]
                    else:
                        xs_seg = [p0[0], p1[0]]
                        ys_seg = [p0[1], p1[1]]
                        zs_seg = [p0[2], p1[2]]

                    fig.add_trace(go.Scatter3d(x=xs_seg, y=ys_seg, z=zs_seg, mode="lines",
                                               line=dict(width=line_width * (0.6 + 0.8 * adj), color=edge_col),
                                               opacity=opacity, showlegend=False, hoverinfo="text",
                                               text=f"{self.labels[i]} → {self.labels[j]}<br>Weight: {w:.3f}"))
                    if directed:
                        pos = np.array([xs_seg[-2], ys_seg[-2], zs_seg[-2]]) if len(xs_seg) > 2 else ((p0 + p1) / 2.0)
                        arrow_x.append(pos[0]); arrow_y.append(pos[1]); arrow_z.append(pos[2])
                        vec = (p1 - p0)
                        vec = vec / (np.linalg.norm(vec) + 1e-12)
                        arrow_adj.append(adj)
                        arrow_vals.append(w)
                        arrow_size.append(max(1.5, 3.0 * adj))
                        arrow_dir_u.append(vec[0]); arrow_dir_v.append(vec[1]); arrow_dir_w.append(vec[2])

        if line_xs:
            fig.add_trace(go.Scatter3d(
                x=line_xs, y=line_ys, z=line_zs,
                mode="lines",
                line=dict(width=line_width),
                opacity=opacity,
                name=("Arcs" if edge_style == "arc" else "Edges")
            ))

        if directed and arrow_x:
            # Prefer 3D cone glyphs for arrowheads. We map arrow_adj (0..1) into colorscale for cone coloring.
            try:
                # Color cones using the actual connection values (arrow_vals) and the mapped zmin/zmax
                fig.add_trace(go.Cone(
                    x=arrow_x, y=arrow_y, z=arrow_z,
                    u=arrow_dir_u, v=arrow_dir_v, w=arrow_dir_w,
                    sizemode='absolute', sizeref=max(0.5, float(np.nanmax(arrow_size))),
                    anchor='tip',
                    colorscale=colorscale, cmin=zmin, cmax=zmax,
                    color=arrow_vals,
                    showscale=False,
                ))
            except Exception:
                # Fallback: use colored markers sampled from colorscale
                try:
                    # map actual arrow values into normalized [0,1] for sampling the colorscale
                    span = float(zmax - zmin) if zmax != zmin else 1.0
                    marker_colors = [_color_from_scale(colorscale, float(np.clip((val - zmin) / span, 0.0, 1.0))) for val in arrow_vals]
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
                marker=dict(colorscale=colorscale, cmin=zmin, cmax=zmax, color=[zmin, zmax], showscale=True,
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
        threshold: float = 0.0,
        threshold_type: Optional[str] = None,
        center_zero: bool = False,
        colorscale: str = "RdBu",
        title: Optional[str] = None,
        showscale: bool = True,
        bg_color: str = "rgba(230,230,230,0.3)"  # faint gray grid background
        ,
        conn_min: float = 0.0,
        conn_max: float = 1.0,
    ) -> go.Figure:
        """
        Connectivity heatmap (n x n) with faint empty grid rectangles for missing/thresholded cells.
        
        Args:
            threshold: Threshold value (absolute) or percentage (if threshold_type is set)
            threshold_type: If set to "Basic" or "Minimum Spanning Tree", applies that thresholding
        """
        C = np.array(self.conn, dtype=float)

        # Apply advanced thresholding if specified
        if threshold_type:
            mask = self.apply_threshold(threshold_type, threshold)
            C = C * mask
        # Otherwise apply basic absolute thresholding
        elif threshold > 0:
            mask = np.abs(C) < threshold
            C = C.copy()
            C[mask] = np.nan

        # Color range: compute full-data min/max then map conn_min/conn_max (0..1) into that range
        if np.any(np.isfinite(C)):
            data_min = float(np.nanmin(C))
            data_max = float(np.nanmax(C))
        else:
            data_min, data_max = -1.0, 1.0

        if center_zero:
            max_abs = np.nanmax(np.abs(C)) if np.any(np.isfinite(C)) else 1.0
            zmin, zmax = -max_abs, max_abs
        else:
            # Map normalized conn_min/conn_max (0..1) into actual data range
            zmin = data_min + float(np.clip(conn_min, 0.0, 1.0)) * (data_max - data_min)
            zmax = data_min + float(np.clip(conn_max, 0.0, 1.0)) * (data_max - data_min)
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
            colorscale=colorscale,
            zmin=zmin,
            zmax=zmax,
            xgap=0.5,
            ygap=0.5,
            colorbar=dict(title="Conn") if showscale else None,
            showscale=showscale,
            hovertemplate="From %{y}<br>To %{x}<br>Value=%{z:.3f}<extra></extra>",
        ))

        # Layout styling
        fig.update_layout(
            title=title or "Connectivity Heatmap",
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




    # ---------- Convenience utilities ----------
    def subset(self, nodes: Iterable[int]) -> "ConnectivityVisualizer":
        """Return a new visualizer with a node subset (by indices)."""
        idx = np.array(list(nodes), dtype=int)
        sub_conn = self.conn[np.ix_(idx, idx)]
        sub_xyz = self.xyz[idx]
        sub_labels = self.labels[idx]
        df = pd.DataFrame(dict(x=sub_xyz[:, 0], y=sub_xyz[:, 1], z=sub_xyz[:, 2], label=sub_labels))
        return ConnectivityVisualizer(sub_conn, df, brain_mesh=self.brain_mesh)

    def set_conn(self, conn: np.ndarray) -> None:
        """Replace connectivity matrix in-place (keeps geometry)."""
        conn = np.asarray(conn)
        assert conn.shape == (self.n, self.n)
        self.conn = conn
