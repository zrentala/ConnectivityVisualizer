from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple, Union, List, Dict

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


# ---------------------------------------------------------------------
# Color utilities
# ---------------------------------------------------------------------


def _rgba_from_color(col: str, strength: float) -> str:
    """Return an 'rgba(r,g,b,a)' string for a given color and alpha in [0,1]."""
    strength = float(np.clip(strength, 0.0, 1.0))
    if not isinstance(col, str):
        return f"rgba(0,0,0,{strength:.3f})"
    c = col.strip()
    # Hex colors
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

    return c


def _color_from_scale(name: str, t: float) -> str:
    """Sample an RGB color from a Plotly colorscale at t in [0,1]."""
    t = float(np.clip(t, 0.0, 1.0))
    if not isinstance(name, str) or not name:
        name = "Viridis"

    seq = getattr(plc.sequential, name, None)
    if seq is None or len(seq) == 0:
        seq = getattr(plc.diverging, name, None)
    if seq is None or len(seq) == 0:
        seq = plc.sequential.Viridis

    def _to_rgb_tuple(cstr: str):
        s = cstr.strip()
        if s.startswith("#"):
            if len(s) == 7:
                return int(s[1:3], 16), int(s[3:5], 16), int(s[5:7], 16)
            if len(s) == 4:
                return int(s[1] * 2, 16), int(s[2] * 2, 16), int(s[3] * 2, 16)
        if s.startswith("rgb"):
            try:
                inside = s[s.find("(") + 1:s.find(")")]
                parts = [int(p.strip()) for p in inside.split(",")]
                return tuple(parts[:3])
            except Exception:
                pass
        return (0, 0, 0)

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


# ---------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------


@dataclass
class Channel:
    x: float
    y: float
    label: Optional[str] = None
    z: Optional[float] = None


# ---------------------------------------------------------------------
# Main visualizer
# ---------------------------------------------------------------------


class ConnectivityVisualizer:
    """
    One object to hold data + build interactive 2D, 3D, and heatmap connectivity figures.

    Thresholding is handled externally via a Threshold object; the visualizer itself
    does not maintain threshold state. Geometry (paths/arcs) and base figures are cached.
    """

    def __init__(
        self,
        brain_data: BrainData,
        conn_idx: int = 0,
        colorscale: str = "Viridis",
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
        self.colorscale: str = colorscale
        self.conn_min: float = conn_min
        self.conn_max: float = conn_max
        self.node_size: float = node_size
        self.show_labels: bool = show_labels
        self.default_pos_color: str = default_pos_color
        self.default_neg_color: str = default_neg_color
        self.node_fill: str = node_fill
        self.node_edge: str = node_edge
        self.viz_type: str = viz_type

        # ---- caches ----
        # Layout/geometry caches
        self._base_2d_traces: Optional[List[go.Scatter]] = None
        self._edge_paths_cache: Dict[Tuple[int, int, bool, float], np.ndarray] = {}
        self._arrow_geom_cache: Dict[Tuple[int, int, bool, float], Tuple[np.ndarray, np.ndarray]] = {}
        self._arc3d_cache: Dict[Tuple[int, int, Optional[float], int], Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

        # Data-dependent caches: keyed by (id(brain_data), conn_idx)
        self._scale_range_cache: Dict[Tuple[int, int], Tuple[float, float, float]] = {}
        self._edge_index_cache: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}

        # Base figure + edge-trace metadata caches
        # 2D: key = (id(brain_data), conn_idx, use_arcs, curvature)
        self._fig_cache_2d: Dict[Tuple[int, int, bool, float], go.Figure] = {}
        self._edge_traces_2d_meta: Dict[Tuple[int, int, bool, float], Tuple[List[Tuple[int, int]], int]] = {}
        # 3D: key = (id(brain_data), conn_idx, arc_radius, arc_samples)
        self._fig_cache_3d: Dict[Tuple[int, int, Optional[float], int], go.Figure] = {}
        self._edge_traces_3d_meta: Dict[Tuple[int, int, Optional[float], int], Tuple[List[Tuple[int, int]], int]] = {}
        # Heatmap: key = (id(brain_data), conn_idx)
        self._fig_cache_heatmap: Dict[Tuple[int, int], go.Figure] = {}
        self._heatmap_meta: Dict[Tuple[int, int], int] = {}  # heatmap trace index

        # ---- derived / cached fields ----
        self.xyz: np.ndarray = np.empty((0, 3), dtype=float)
        self.xy_topo: np.ndarray = np.empty((0, 2), dtype=float)
        self.labels: np.ndarray = np.array([], dtype=str)
        self.n: int = 0

        self.update_xyz(brain_data.chanlocs)

    # ------------------------------------------------------------------
    # Boilerplate
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"conn_idx={self.conn_idx}, "
            f"colorscale={self.colorscale!r}, "
            f"conn_min={self.conn_min}, "
            f"conn_max={self.conn_max}, "
            f"node_size={self.node_size}, "
            f"show_labels={self.show_labels}, "
            f"default_pos_color={self.default_pos_color!r}, "
            f"default_neg_color={self.default_neg_color!r}, "
            f"node_fill={self.node_fill!r}, "
            f"node_edge={self.node_edge!r}, "
            f"viz_type={self.viz_type!r}"
            f")"
        )

    def __eq__(self, other) -> bool:
        if not isinstance(other, ConnectivityVisualizer):
            return False

        return (
            self.conn_idx == other.conn_idx
            and self.colorscale == other.colorscale
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

    # ------------------------------------------------------------------
    # Shared numeric helpers
    # ------------------------------------------------------------------

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
    def _arc_points_origin_plane(
        p0: np.ndarray,
        p1: np.ndarray,
        R: Optional[float],
        m: int = 20,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Arc between p0 and p1 in the plane through {p0, p1, origin}.
        If R is None, choose a gentle automatic radius.
        """
        p0 = p0.astype(float)
        p1 = p1.astype(float)
        chord = p1 - p0
        d = np.linalg.norm(chord)
        if d < 1e-12:
            return np.array([p0[0]]), np.array([p0[1]]), np.array([p0[2]])

        n = np.cross(p0, p1)
        nn = np.linalg.norm(n)
        if nn < 1e-12:
            t = np.linspace(0, 1, m)
            P = p0[None, :] + t[:, None] * chord[None, :]
            return P[:, 0], P[:, 1], P[:, 2]
        n /= nn

        u = chord / d
        v = np.cross(n, u)
        v /= np.linalg.norm(v)

        if R is None:
            R = 0.55 * d + (d / 2.0)
        R = max(R, d / 2.0 + 1e-9)

        h = np.sqrt(max(R * R - (d / 2) ** 2, 0.0))
        mid = 0.5 * (p0 + p1)
        center = mid + h * v

        def angle(vec):
            return np.arctan2(np.dot(vec, v), np.dot(vec, u))

        theta0 = angle(p0 - center)
        theta1 = angle(p1 - center)
        dtheta = (theta1 - theta0 + np.pi) % (2 * np.pi) - np.pi

        thetas = theta0 + np.linspace(0, dtheta, m)
        cs, ss = np.cos(thetas), np.sin(thetas)
        pts = center[None, :] + R * cs[:, None] * u[None, :] + R * ss[:, None] * v[None, :]
        return pts[:, 0], pts[:, 1], pts[:, 2]

    # ------------------------------------------------------------------
    # Data + range helpers (shared across 2D/3D/heatmap)
    # ------------------------------------------------------------------

    @staticmethod
    def _data_key(brain_data: BrainData, conn_idx: int) -> Tuple[int, int]:
        """Stable key for data caches: brain_data object id + current conn_idx."""
        return (id(brain_data), conn_idx)

    def get_mat_at_idx(self, brain_data: BrainData) -> np.ndarray:
        """Direct slice; no caching of the matrix itself."""
        return np.asarray(brain_data.conn_mat[self.conn_idx, :, :], dtype=float)

    def _apply_threshold(self, baseC: np.ndarray, threshold: Threshold) -> np.ndarray:
        """Apply threshold to a copy of baseC and zero the diagonal."""
        C = baseC.copy()
        mask = threshold.apply_threshold(C)
        C = C * mask
        np.fill_diagonal(C, 0.0)
        return C

    def _get_scale_and_data_range(
        self,
        baseC: np.ndarray,
        data_key: Tuple[int, int],
    ) -> Tuple[float, float, float]:
        """
        Shared scale & data range for 2D/3D/heatmap.
        Cached by (id(brain_data), conn_idx).
        Returns (scale, data_min, data_max).
        """
        key = data_key
        if key in self._scale_range_cache:
            return self._scale_range_cache[key]

        scale = self._max_conn_scale(baseC)
        if np.any(np.isfinite(baseC)):
            data_min = float(np.nanmin(baseC))
            data_max = float(np.nanmax(baseC))
        else:
            data_min, data_max = -1.0, 1.0

        self._scale_range_cache[key] = (scale, data_min, data_max)
        return self._scale_range_cache[key]

    def _get_z_limits(self, data_min: float, data_max: float) -> Tuple[float, float]:
        """
        Map normalized conn_min/conn_max (0..1) into actual data range.
        """
        zmin = data_min + float(np.clip(self.conn_min, 0.0, 1.0)) * (data_max - data_min)
        zmax = data_min + float(np.clip(self.conn_max, 0.0, 1.0)) * (data_max - data_min)
        if zmin == zmax:
            zmin, zmax = zmin - 1e-6, zmax + 1e-6
        return zmin, zmax

    def _get_candidate_edges_cached(
        self,
        baseC: np.ndarray,
        data_key: Tuple[int, int],
    ) -> List[Tuple[int, int]]:
        """
        Precompute candidate (i,j) edges (non-trivial entries) on base matrix.
        Cached by (id(brain_data), conn_idx). Thresholding is applied later.
        """
        key = data_key
        if key in self._edge_index_cache:
            return self._edge_index_cache[key]

        mask = np.isfinite(baseC) & (np.abs(baseC) >= 1e-12)
        i_idx, j_idx = np.where(mask)
        edges = list(zip(i_idx.tolist(), j_idx.tolist()))
        self._edge_index_cache[key] = edges
        return edges

    # ------------------------------------------------------------------
    # Geometry caches (2D paths + 2D arrowheads + 3D arcs)
    # ------------------------------------------------------------------

    def _get_edge_path(self, i: int, j: int, use_arcs: bool, curvature: float) -> np.ndarray:
        key = (i, j, use_arcs, float(curvature))
        if key in self._edge_paths_cache:
            return self._edge_paths_cache[key]

        p0 = self.xy_topo[i]
        p1 = self.xy_topo[j]
        if use_arcs:
            P = self._quad_bezier(p0, p1, curvature, m=60)
        else:
            P = np.vstack([p0, p1])

        self._edge_paths_cache[key] = P
        return P

    def _get_arrow_geometry(
        self,
        i: int,
        j: int,
        use_arcs: bool,
        curvature: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Cache arrowhead anchor geometry (q0 -> q1) for edge (i,j) in 2D.
        Depends only on layout + arc settings, not on weights.
        """
        key = (i, j, use_arcs, float(curvature))
        if key in self._arrow_geom_cache:
            return self._arrow_geom_cache[key]

        P = self._get_edge_path(i, j, use_arcs, curvature)
        if len(P) >= 2:
            q0, q1 = P[-2], P[-1]
        else:
            q0 = q1 = P[0]

        self._arrow_geom_cache[key] = (q0, q1)
        return q0, q1

    def _get_arc3d(
        self,
        i: int,
        j: int,
        arc_radius: Optional[float],
        arc_samples: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Cached 3D arc geometry between nodes i and j, before reciprocal offset.
        Depends only on coordinates + radius + samples.
        """
        key = (i, j, arc_radius, int(arc_samples))
        if key in self._arc3d_cache:
            return self._arc3d_cache[key]

        p0 = self.xyz[i]
        p1 = self.xyz[j]
        X, Y, Z = self._arc_points_origin_plane(p0, p1, arc_radius, m=max(int(arc_samples), 2))
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        Z = np.asarray(Z, dtype=float)
        self._arc3d_cache[key] = (X, Y, Z)
        return X, Y, Z

    # ------------------------------------------------------------------
    # Coord / label handling
    # ------------------------------------------------------------------

    def update_xyz(
        self,
        chanlocs: Union[pd.DataFrame, Iterable[Union[Channel, dict, Iterable]]]
    ) -> None:
        """
        Update channel locations and derived coordinates (xyz, xy_topo, labels).
        Invalidates geometry + base figure caches.
        """
        if isinstance(chanlocs, pd.DataFrame):
            sx = chanlocs["x"].to_numpy()
            sy = chanlocs["y"].to_numpy()
            sz = chanlocs["z"].to_numpy() if "z" in chanlocs.columns else np.zeros_like(sx)
            if "label" in chanlocs.columns:
                labs = chanlocs["label"].astype(str).to_numpy()
            else:
                labs = np.arange(len(sx)).astype(str)
        else:
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
        self.n = n_ch
        if labs.size == n_ch:
            self.labels = labs
        else:
            self.labels = np.arange(n_ch).astype(str)

        self.xyz = np.column_stack([sx, sy, sz]).astype(float)

        xs = sx.copy()
        ys = sy.copy()
        xs = -xs / (np.max(np.abs(xs)) + 1e-12) * 0.9
        ys = ys / (np.max(np.abs(ys)) + 1e-12) * 0.9
        self.xy_topo = np.column_stack([xs, ys])

        # Geometry/layout changes invalidate path/arrow/fig caches
        self._base_2d_traces = None
        self._edge_paths_cache.clear()
        self._arrow_geom_cache.clear()
        self._arc3d_cache.clear()
        self._fig_cache_2d.clear()
        self._edge_traces_2d_meta.clear()
        self._fig_cache_3d.clear()
        self._edge_traces_3d_meta.clear()
        self._fig_cache_heatmap.clear()
        self._heatmap_meta.clear()

    # ------------------------------------------------------------------
    # Public dispatch
    # ------------------------------------------------------------------

    def get_figure(self, brain_data: BrainData, threshold: Threshold) -> go.Figure:
        thr = threshold 
        if self.viz_type == "2D":
            return self.figure_2d(brain_data=brain_data, threshold=thr)
        elif self.viz_type == "3D":
            return self.figure_3d(brain_data=brain_data, threshold=thr)
        elif self.viz_type == "Heatmap":
            return self.figure_heatmap(brain_data=brain_data, threshold=thr)
        else:
            return go.Figure()

    # ------------------------------------------------------------------
    # 2D base traces
    # ------------------------------------------------------------------

    def _build_base_2d_traces(self) -> List[go.Scatter]:
        theta = np.linspace(0, 2 * np.pi, 256)
        x, y = self.xy_topo[:, 0], self.xy_topo[:, 1]

        head = go.Scatter(
            x=np.cos(theta),
            y=np.sin(theta),
            mode="lines",
            line=dict(color="black", width=2),
            hoverinfo="skip",
            name="Head",
        )

        nose = go.Scatter(
            x=[0.10, 0.00, -0.10],
            y=[1.00, 1.10, 1.00],
            mode="lines",
            line=dict(color="black", width=2),
            hoverinfo="skip",
            showlegend=False,
        )

        nodes = go.Scatter(
            x=x,
            y=y,
            mode="markers+text" if self.show_labels else "markers",
            text=self.labels if self.show_labels else None,
            textposition="middle center",
            marker=dict(
                size=self.node_size,
                color=self.node_fill,
                line=dict(color=self.node_edge, width=2),
            ),
            hovertext=self.labels,
            hoverinfo="text",
            name="Electrodes",
        )

        return [head, nose, nodes]

    def _get_base_2d_traces(self) -> List[go.Scatter]:
        if self._base_2d_traces is None:
            self._base_2d_traces = self._build_base_2d_traces()
        return self._base_2d_traces

    # ------------------------------------------------------------------
    # 2D base figure + meta (geometry-only)
    # ------------------------------------------------------------------

    def _base2d_key(self, brain_data: BrainData, use_arcs: bool, curvature: float) -> Tuple[int, int, bool, float]:
        return (id(brain_data), self.conn_idx, bool(use_arcs), float(curvature))

    def _build_base_2d_figure(
        self,
        brain_data: BrainData,
        use_arcs: bool,
        curvature: float,
    ) -> go.Figure:
        """
        Build a base 2D figure with:
          - head, nose, nodes
          - one line trace per candidate edge (geometry only, neutral styling)
        """
        baseC = self.get_mat_at_idx(brain_data)
        data_key = self._data_key(brain_data, self.conn_idx)
        edges = self._get_candidate_edges_cached(baseC, data_key)

        fig = go.Figure()
        for tr in self._get_base_2d_traces():
            fig.add_trace(tr)

        edge_order: List[Tuple[int, int]] = []
        edge_start = len(fig.data)

        for (i, j) in edges:
            if i == j:
                continue
            w = baseC[i, j]
            if not np.isfinite(w) or abs(w) < 1e-12:
                continue

            P = self._get_edge_path(i, j, use_arcs=use_arcs, curvature=curvature)
            edge_order.append((i, j))

            fig.add_trace(
                go.Scatter(
                    x=P[:, 0],
                    y=P[:, 1],
                    mode="lines",
                    line=dict(color="rgba(0,0,0,0.3)", width=1.0),
                    opacity=0.75,
                    showlegend=False,
                    hoverinfo="text",
                    text="",  # filled later
                    visible=True,
                )
            )

        fig.update_layout(
            xaxis=dict(visible=False, scaleanchor="y", scaleratio=1),
            yaxis=dict(visible=False),
            autosize=True,
            margin=dict(l=0, r=0, t=40, b=0),
            showlegend=False,
            plot_bgcolor="white",
        )

        key = self._base2d_key(brain_data, use_arcs, curvature)
        self._fig_cache_2d[key] = fig
        self._edge_traces_2d_meta[key] = (edge_order, edge_start)
        return fig

    def _get_base_2d_figure(
        self,
        brain_data: BrainData,
        use_arcs: bool,
        curvature: float,
    ) -> go.Figure:
        key = self._base2d_key(brain_data, use_arcs, curvature)
        if key not in self._fig_cache_2d:
            return self._build_base_2d_figure(brain_data, use_arcs, curvature)
        return self._fig_cache_2d[key]

    # ------------------------------------------------------------------
    # 2D figure (restyling cached traces)
    # ------------------------------------------------------------------

    def figure_2d(
        self,
        *,
        brain_data: BrainData,
        threshold: Threshold,
        use_arcs: bool = True,
        curvature: float = 0.25,
        lw_min: float = 0.5,
        lw_max: float = 4.0,
        title: Optional[str] = None,
    ) -> go.Figure:
        """
        Interactive 2D EEG-style top view using a cached base figure.

        For each cached edge trace:
          - apply threshold
          - set visibility
          - update color and width
          - update hovertext
          - rebuild arrow annotations from cached geometry
        """
        baseC = self.get_mat_at_idx(brain_data)
        data_key = self._data_key(brain_data, self.conn_idx)
        C = self._apply_threshold(baseC, threshold)
        scale, data_min, data_max = self._get_scale_and_data_range(baseC, data_key)
        zmin, zmax = self._get_z_limits(data_min, data_max)

        base_fig = self._get_base_2d_figure(brain_data, use_arcs, curvature)
        fig = go.Figure(base_fig.to_dict())  # clone

        meta_key = self._base2d_key(brain_data, use_arcs, curvature)
        edge_order, edge_start = self._edge_traces_2d_meta[meta_key]

        arrow_annots: List[dict] = []

        for idx, (i, j) in enumerate(edge_order):
            trace_idx = edge_start + idx
            tr = fig.data[trace_idx]

            w = C[i, j]
            if not np.isfinite(w) or abs(w) < 1e-12:
                tr.visible = False
                continue

            tr.visible = True

            t_global = (w - data_min) / max((data_max - data_min), 1e-12)
            try:
                adj = (t_global - self.conn_min) / max((self.conn_max - self.conn_min), 1e-12)
            except Exception:
                adj = t_global
            adj = float(np.clip(adj, 0.0, 1.0))

            try:
                color = _color_from_scale(self.colorscale, adj)
            except Exception:
                base_color = self.default_pos_color if w >= 0 else self.default_neg_color
                color = _rgba_from_color(base_color, max(0.12, 0.25 + 0.75 * adj))

            width = lw_min + (abs(w) / max(scale, 1e-12)) * (lw_max - lw_min)

            tr.line.color = color
            tr.line.width = width
            tr.hovertext = f"{self.labels[i]} → {self.labels[j]}<br>Weight: {w:.3f}"
            tr.hoverinfo = "text"

            if brain_data.directed:
                q0, q1 = self._get_arrow_geometry(i, j, use_arcs=use_arcs, curvature=curvature)
                arrow_annots.append(
                    dict(
                        x=q1[0],
                        y=q1[1],
                        ax=q0[0],
                        ay=q0[1],
                        xref="x",
                        yref="y",
                        axref="x",
                        ayref="y",
                        showarrow=True,
                        arrowhead=2,
                        arrowsize=1.2,
                        arrowwidth=width / 2,
                        arrowcolor=color,
                        opacity=0.8,
                    )
                )

        # Colorbar
        try:
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="markers",
                    marker=dict(
                        colorscale=self.colorscale,
                        cmin=zmin,
                        cmax=zmax,
                        color=[zmin, zmax],
                        showscale=True,
                        colorbar=dict(
                            title="Conn",
                            len=0.45,
                            thickness=12,
                        ),
                    ),
                    showlegend=False,
                    hoverinfo="none",
                )
            )
        except Exception:
            pass

        existing = list(fig.layout.annotations) if fig.layout.annotations else []
        if arrow_annots:
            fig.update_layout(annotations=existing + arrow_annots)

        fig.update_layout(title=title)
        return fig

    # ------------------------------------------------------------------
    # 3D base figure + meta (geometry-only)
    # ------------------------------------------------------------------

    def _base3d_key(
        self,
        brain_data: BrainData,
        arc_radius: Optional[float],
        arc_samples: int,
    ) -> Tuple[int, int, Optional[float], int]:
        return (id(brain_data), self.conn_idx, arc_radius, int(arc_samples))

    def _build_base_3d_figure(
        self,
        brain_data: BrainData,
        arc_radius: Optional[float],
        arc_samples: int,
    ) -> go.Figure:
        """
        Build a base 3D figure with:
          - brain mesh (if available)
          - nodes
          - one line trace per candidate edge (3D arc geometry only, neutral styling)
        """
        baseC = self.get_mat_at_idx(brain_data)
        data_key = self._data_key(brain_data, self.conn_idx)
        edges = self._get_candidate_edges_cached(baseC, data_key)

        fig = go.Figure()

        # Brain mesh
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

        # Nodes
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

        edge_order: List[Tuple[int, int]] = []
        edge_start = len(fig.data)

        for (i, j) in edges:
            if i == j:
                continue

            w = baseC[i, j]
            if not np.isfinite(w) or abs(w) < 1e-12:
                continue

            X, Y, Z = self._get_arc3d(i, j, arc_radius, arc_samples)

            edge_order.append((i, j))
            fig.add_trace(go.Scatter3d(
                x=list(X), y=list(Y), z=list(Z),
                mode="lines",
                line=dict(width=1.0, color="rgba(0,0,0,0.2)"),
                opacity=0.5,
                showlegend=False,
                hoverinfo="text",
                text="",
                visible=True,
            ))

        fig.update_layout(
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                aspectmode="data",
            ),
            autosize=True,
            margin=dict(l=0, r=0, t=40, b=0),
            legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.02),
            title="3D Connectivity",
        )

        key = self._base3d_key(brain_data, arc_radius, arc_samples)
        self._fig_cache_3d[key] = fig
        self._edge_traces_3d_meta[key] = (edge_order, edge_start)
        return fig

    def _get_base_3d_figure(
        self,
        brain_data: BrainData,
        arc_radius: Optional[float],
        arc_samples: int,
    ) -> go.Figure:
        key = self._base3d_key(brain_data, arc_radius, arc_samples)
        if key not in self._fig_cache_3d:
            return self._build_base_3d_figure(brain_data, arc_radius, arc_samples)
        return self._fig_cache_3d[key]

    # ------------------------------------------------------------------
    # 3D figure (restyling cached traces, cones recomputed)
    # ------------------------------------------------------------------

    def figure_3d(
        self,
        *,
        brain_data: BrainData,
        threshold: Threshold,
        arc_radius: Optional[float] = None,
        arc_samples: int = 4,
        line_width: float = 3.0,
        opacity: float = 0.6,
        title: Optional[str] = None,
    ) -> go.Figure:
        """
        Interactive 3D connectivity visualization using cached base figure.
        For each cached edge trace:
          - apply threshold
          - set visibility
          - update color/width
          - update hovertext
        Arrowheads (cones or markers) are built fresh each time.
        """
        baseC = self.get_mat_at_idx(brain_data)
        data_key = self._data_key(brain_data, self.conn_idx)
        C = self._apply_threshold(baseC, threshold)
        _, data_min, data_max = self._get_scale_and_data_range(baseC, data_key)
        zmin, zmax = self._get_z_limits(data_min, data_max)

        base_fig = self._get_base_3d_figure(brain_data, arc_radius, arc_samples)
        fig = go.Figure(base_fig.to_dict())  # clone

        meta_key = self._base3d_key(brain_data, arc_radius, arc_samples)
        edge_order, edge_start = self._edge_traces_3d_meta[meta_key]

        arrow_x, arrow_y, arrow_z = [], [], []
        arrow_adj, arrow_size = [], []
        arrow_dir_u, arrow_dir_v, arrow_dir_w = [], [], []
        arrow_vals = []

        for idx, (i, j) in enumerate(edge_order):
            trace_idx = edge_start + idx
            tr = fig.data[trace_idx]

            # For undirected: keep only one direction (upper triangle)
            if not brain_data.directed and j <= i:
                tr.visible = False
                continue

            w = float(C[i, j])
            if not np.isfinite(w) or abs(w) < 1e-12:
                tr.visible = False
                continue

            tr.visible = True

            t_global = (w - data_min) / max((data_max - data_min), 1e-12)
            adj = (t_global - self.conn_min) / max((self.conn_max - self.conn_min), 1e-12)
            adj = float(np.clip(adj, 0.0, 1.0))

            try:
                edge_col = _color_from_scale(self.colorscale, adj)
            except Exception:
                edge_col = _rgba_from_color('red' if w >= 0 else 'blue', max(0.12, 0.25 + 0.75 * adj))

            tr.line.color = edge_col
            tr.line.width = line_width * (0.6 + 0.8 * adj)
            tr.opacity = opacity
            tr.hoverinfo = "text"
            tr.hovertext = f"{self.labels[i]} → {self.labels[j]}<br>Weight: {w:.3f}"

            if brain_data.directed:
                # reconstruct arrow direction from cached arc geometry
                X = np.array(tr.x, dtype=float)
                Y = np.array(tr.y, dtype=float)
                Z = np.array(tr.z, dtype=float)
                if len(X) < 2:
                    continue
                q0 = np.array([X[-2], Y[-2], Z[-2]])
                q1 = np.array([X[-1], Y[-1], Z[-1]])
                pos = q1 - 0.05 * (q1 - q0)

                arrow_x.append(pos[0]); arrow_y.append(pos[1]); arrow_z.append(pos[2])
                vec = q1 - q0
                norm = np.linalg.norm(vec)
                if norm < 1e-9:
                    p0 = self.xyz[i]
                    p1 = self.xyz[j]
                    vec = p1 - p0
                    norm = np.linalg.norm(vec) + 1e-12
                vec = vec / (norm + 1e-12)
                arrow_adj.append(adj)
                arrow_vals.append(w)
                arrow_size.append(max(0.6, 0.6 * adj))
                arrow_dir_u.append(vec[0]); arrow_dir_v.append(vec[1]); arrow_dir_w.append(vec[2])

        # Arrowheads
        if brain_data.directed and arrow_x:
            try:
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
                try:
                    span = float(zmax - zmin) if zmax != zmin else 1.0
                    marker_colors = [
                        _color_from_scale(
                            self.colorscale,
                            float(np.clip((val - zmin) / span, 0.0, 1.0))
                        )
                        for val in arrow_vals
                    ]
                except Exception:
                    marker_colors = ['red' if v >= 0.5 else 'blue' for v in arrow_vals]
                fig.add_trace(go.Scatter3d(
                    x=arrow_x, y=arrow_y, z=arrow_z,
                    mode="markers",
                    marker=dict(size=[max(4, s * 6) for s in arrow_size], color=marker_colors),
                    name="Direction",
                    hoverinfo="skip",
                ))

        # Colorbar
        try:
            fig.add_trace(go.Scatter3d(
                x=[None], y=[None], z=[None], mode="markers",
                marker=dict(
                    colorscale=self.colorscale,
                    cmin=zmin,
                    cmax=zmax,
                    color=[zmin, zmax],
                    showscale=True,
                    colorbar=dict(title="Conn", len=0.45, thickness=12),
                ),
                showlegend=False, hoverinfo="none",
            ))
        except Exception:
            pass

        fig.update_layout(title=title or "3D Connectivity")
        return fig

    # ------------------------------------------------------------------
    # Heatmap base figure + meta
    # ------------------------------------------------------------------

    def _heatmap_key(self, brain_data: BrainData) -> Tuple[int, int]:
        return (id(brain_data), self.conn_idx)

    def _build_base_heatmap_figure(self, brain_data: BrainData) -> go.Figure:
        """
        Build a base heatmap figure with:
          - background grid trace
          - placeholder main heatmap trace (z filled later)
        """
        baseC = self.get_mat_at_idx(brain_data)
        bg_color = "rgba(230,230,230,0.3)"

        bg = np.full_like(baseC, np.nan)
        bg[np.isnan(baseC)] = 0

        fig = go.Figure()

        # Background grid
        fig.add_trace(go.Heatmap(
            z=bg,
            x=self.labels,
            y=self.labels,
            colorscale=[[0, bg_color], [1, bg_color]],
            showscale=False,
            xgap=0.5,
            ygap=0.5,
            hoverinfo="skip",
        ))

        # Placeholder main heatmap (z updated in figure_heatmap)
        fig.add_trace(go.Heatmap(
            z=np.zeros_like(baseC),
            x=self.labels,
            y=self.labels,
            colorscale=self.colorscale,
            showscale=True,
            xgap=0.5,
            ygap=0.5,
            colorbar=dict(title="Conn"),
            hovertemplate="From %{y}<br>To %{x}<br>Value=%{z:.3f}<extra></extra>",
        ))

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

        key = self._heatmap_key(brain_data)
        self._fig_cache_heatmap[key] = fig
        # main heatmap is trace index 1
        self._heatmap_meta[key] = 1
        return fig

    def _get_base_heatmap_figure(self, brain_data: BrainData) -> go.Figure:
        key = self._heatmap_key(brain_data)
        if key not in self._fig_cache_heatmap:
            return self._build_base_heatmap_figure(brain_data)
        return self._fig_cache_heatmap[key]

    # ------------------------------------------------------------------
    # Heatmap figure (restyling cached traces)
    # ------------------------------------------------------------------

    def figure_heatmap(
        self,
        *,
        threshold: Threshold,
        brain_data: BrainData,
    ) -> go.Figure:
        """
        Connectivity heatmap (n x n) reusing shared base connectivity and color scaling.
        Only z-values and color range are updated; geometry/layout are cached.
        """
        baseC = self.get_mat_at_idx(brain_data)
        data_key = self._data_key(brain_data, self.conn_idx)
        C = self._apply_threshold(baseC, threshold)
        _, data_min, data_max = self._get_scale_and_data_range(baseC, data_key)
        zmin, zmax = self._get_z_limits(data_min, data_max)

        base_fig = self._get_base_heatmap_figure(brain_data)
        fig = go.Figure(base_fig.to_dict())  # clone

        key = self._heatmap_key(brain_data)
        heat_idx = self._heatmap_meta[key]
        main = fig.data[heat_idx]

        main.z = C
        main.colorscale = self.colorscale
        main.zmin = zmin
        main.zmax = zmax
        if hasattr(main, "colorbar") and main.colorbar is not None:
            main.colorbar.title = "Conn"

        return fig
