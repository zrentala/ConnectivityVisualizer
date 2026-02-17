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
from itertools import product
from itertools import product
from enum import Enum, auto

class UpdateType(Enum):
    NONE=auto()
    XYZ=auto()
    THRESHOLD=auto()
    VISIBLE=auto()
    ALL=auto()
    SWITCH_FIG=auto()
    NODES=auto()

class VizType(Enum):
    FIG2D=auto()
    FIG3D=auto()
    FIGHEATMAP=auto()

@dataclass
class Channel:
    x: float
    y: float
    label: Optional[str] = None
    # z is optional for 3D; if absent, zeros are assumed
    z: Optional[float] = None

def set_node_colors(self, node_color_map: dict, labels):
    """
    Set node colors for the current node trace based on a mapping {label: color}.
    Used for graph controls (metric/community shading).
    """
    if self.fig is None or self._node_trace_idx is None:
        return
    node_trace = self.fig.data[self._node_trace_idx]
    # Map label order to color
    colors = [node_color_map.get(l, self.node_fill) for l in labels]
    node_trace.marker.color = colors
    # Force redraw
    self.fig.update()

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

def _get_edge_width(edge_weight:float, scale: float, width_range: Tuple[float])->float:
    min_width = width_range[0]
    max_width = width_range[1]
    return min_width + (abs(edge_weight) / max(scale, 1e-12)) * (max_width - min_width)

def _get_edge_color(
    edge_weight: float,
    zmin: float,
    zmax: float,
    colorscale: str,
    default_pos_color: str,
    default_neg_color: str,
):
    """
    Map edge_weight to a color using a precomputed color range [zmin, zmax].
    zmin/zmax already reflect color_min/color_max cropping.
    """
    # Normalize weight into [0, 1] based on the visible color range
    denom = max((zmax - zmin), 1e-12)
    adj = (edge_weight - zmin) / denom
    adj = float(np.clip(adj, 0.0, 1.0))

    try:
        return _color_from_scale(colorscale, adj)
    except Exception:
        base_color = default_pos_color if edge_weight >= 0 else default_neg_color
        alpha = max(0.12, 0.25 + 0.75 * adj)
        return _rgba_from_color(base_color, alpha)

def _quad_bezier(p0: np.ndarray, p1: np.ndarray, curvature: float = 0.25, m: int = 40) -> np.ndarray:
    d = p1 - p0
    L = np.linalg.norm(d)
    if L < 1e-12:
        return np.repeat(p0[None, :], m, axis=0)
    u = d / L
    # print(f"{curvature=}")
    perp = np.array([-u[1], u[0]])
    c = (p0 + p1) / 2.0 + curvature * L * perp
    t = np.linspace(0, 1, m)[:, None]
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * c + t ** 2 * p1


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


def _add_colorbar_trace(fig, colorscale, zmin, zmax, viz_type: VizType) -> int:
    """
    Add an invisible dummy trace to show a colorbar, depending on VizType.
    Returns the index of the added trace (or None on failure).
    """

    # Select trace class + coordinates based on visualization type
    if viz_type == VizType.FIG2D:
        trace_cls = go.Scattergl
        coords = {"x": [None], "y": [None]}

    elif viz_type == VizType.FIG3D:
        trace_cls = go.Scatter3d
        coords = {"x": [None], "y": [None], "z": [None]}
    else:
        raise ValueError(f"Unsupported viz_type: {viz_type}")

    trace = trace_cls(
        **coords,
        mode="markers",
        marker=dict(
            colorscale=colorscale,
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
        name=f"colorbar-{viz_type.name.lower()}",
    )

    try:
        fig.add_trace(trace)
        return len(fig.data) - 1  # return trace index
    except Exception:
        return None

def _update_colorbar(fig, colorbar_idx, colorscale, zmin, zmax):
    tr = fig.data[colorbar_idx]
    tr.marker.colorscale = colorscale
    tr.marker.cmin = zmin
    tr.marker.cmax = zmax
    tr.marker.color = [zmin, zmax]

def _update_edge_trace(trace, edge_weight, color, width, opacity, label1, label2):    
    trace.visible = True 
    trace.line.color = color
    trace.line.width = width
    trace.opacity = opacity
    trace.hoverinfo = "text"
    trace.text = f"{label1} → {label2}<br>Weight: {edge_weight:.3f}"

# def _update_node_trace_all(
#     trace,
#     labels=None,
#     size=None,
#     color=None,
#     # opacity=None,
# ):
#     """
#     Update all nodes in the node trace.
#     """
#     n = len(labels)

#     # Update marker arrays directly
#     if size is not None:
#         print(f"{size=}")
#         trace.update(marker=dict(size=size))

#     if color is not None:
#         trace.marker.color = color
def _update_node_trace_all(
    trace,
    labels=None,
    size=None,
    node_color_map=None,
    node_fill=None,
):
    """
    Update all nodes in the node trace. If node_color_map is provided, use it to set node colors by label.
    """
    # n = len(labels) if labels is not None else 0

    # Update marker arrays directly
    if size is not None:
        print(f"{size=}")
        trace.update(marker=dict(size=size))

    # node_fill is the default color for nodes not in node_color_map
    if node_color_map is not None and labels is not None:
        colors = [node_color_map.get(l, node_fill) for l in labels]
        trace.marker.color = colors


def _get_mat_at_idx(conn_mat: np.ndarray, idx: int) -> np.ndarray:
        C = conn_mat[idx, :, : ].copy()
        return C

def _get_scale_and_range(C: np.ndarray, color_min: float, color_max: float):
        """
        Compute:
        - scale       : max abs connection (excluding diagonal)
        - data_min    : -1 or 0 depending on sign of C
        - data_max    : 1 or 0 depending on sign of C
        - zmin, zmax  : clipped color scale range
        """

        # ---- SCALE ----
        D = C.copy()
        np.fill_diagonal(D, 0.0)

        if np.any(np.isfinite(D)):
            scale = float(np.nanmax(np.abs(D)))
            if scale <= 0:
                scale = 1.0
        else:
            scale = 1.0

        # ---- DATA RANGE (SIGN-BASED) ----
        data_min = 0.0 if np.nanmin(C) >= 0 else -1.0
        data_max = 0.0 if np.nanmax(C) < 0 else  1.0

        # ---- COLOR RANGE ----
        cmin = float(np.clip(color_min, 0.0, 1.0))
        cmax = float(np.clip(color_max, 0.0, 1.0))

        zmin = data_min + cmin * (data_max - data_min)
        zmax = data_min + cmax * (data_max - data_min)

        if zmin == zmax:
            zmin -= 1e-6
            zmax += 1e-6

        return scale, data_min, data_max, zmin, zmax

def _get_ij_iter(n_nodes: int, directed: bool):
    if directed:
        # All ordered pairs except i == j
        return ((i, j) for i, j in product(range(n_nodes), repeat=2) if i != j)
    else:
        # Only undirected upper-triangle pairs, i < j automatically guarantees i != j
        return ((i, j) for i in range(n_nodes)
                for j in range(i + 1, n_nodes))

def _create_cache_nodes(labels, fig):
    """
    Build a lookup table:
        label -> (trace_index_in_fig, point_index_in_trace)

    Assumes the node trace is the one named 'Electrodes', and that
    markers/text ordering matches `labels`.
    """
    # 1. Find the node trace index in the figure
    node_trace_idx = None
    for k, tr in enumerate(fig.data):
        if getattr(tr, "name", None) == "Electrodes":
            node_trace_idx = k
            break

    if node_trace_idx is None:
        return {}  # No node trace found

    # 2. Map label → (trace_idx, point_idx)
    trace = fig.data[node_trace_idx]
    return {
        lab: (node_trace_idx, i)
        for i, lab in enumerate(labels)
    }

def _create_cache_edges(labels, fig):
    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    trace_idx = {}
    for k, tr in enumerate(fig.data):
        name = getattr(tr, "name", None)
        if not name or "," not in name:
            continue

        try:
            a, b = name.split(",")
            i = label_to_idx.get(a)
            j = label_to_idx.get(b)
            if i is not None and j is not None:
                trace_idx[(i, j)] = k
        except Exception:
            continue

    return trace_idx

def _normalize_weight( w: float, data_min: float, data_max: float) -> float:
        """
        Normalize a weight w → [0,1] based on global min/max.
        Handles signed matrices gracefully.
        """
        rng = max((data_max - data_min), 1e-12)
        t = (w - data_min) / rng
        return float(np.clip(t, 0.0, 1.0))
    
def parse_channel_locs(
    chanlocs: Union[pd.DataFrame, Iterable[Union[Channel, dict, Iterable]]]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    if isinstance(chanlocs, pd.DataFrame):
        sx = chanlocs["x"].to_numpy()
        sy = chanlocs["y"].to_numpy()
        sz = chanlocs["z"].to_numpy() if "z" in chanlocs.columns else np.zeros_like(sx)

        if "label" in chanlocs.columns:
            labs = chanlocs["label"].astype(str).to_numpy()
        else:
            labs = np.arange(len(sx)).astype(str)

        return sx, sy, sz, labs

    # ---- list / dict / channel objects ----
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
            # generic list-like: [x, y, (z), (label)]
            x = float(row[0])
            y = float(row[1])
            z = float(row[2]) if len(row) >= 3 and np.isscalar(row[2]) else 0.0
            lab = (
                str(row[3]) if len(row) >= 4
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

    return sx, sy, sz, labs

def compute_locs_3d(sx: np.ndarray, sy: np.ndarray, sz: np.ndarray) -> np.ndarray:
    return np.column_stack([sx, sy, sz]).astype(float)

# def compute_locs_2d_topo(sx: np.ndarray, sy: np.ndarray) -> np.ndarray:
#     xs = -sx.copy()
#     ys =  sy.copy()

#     xs = xs / (np.max(np.abs(xs)) + 1e-12)
#     ys = ys / (np.max(np.abs(ys)) + 1e-12) * 1.1 + 0.1

#     return np.column_stack([xs, ys])

def compute_locs_2d_topo(sx: np.ndarray, sy: np.ndarray) -> np.ndarray:
    """
    Convert raw x,y EEG positions to 2D topographic coordinates.
    If two nodes lie too close together (overlap), the later one is pushed outward.
    """
    # --- 1) Standard topo projection ---
    xs = -sx.copy()
    ys = sy.copy()

    xs = xs / (np.max(np.abs(xs)) + 1e-12)
    ys = ys / (np.max(np.abs(ys)) + 1e-12) * 1.1 + 0.1

    locs = np.column_stack([xs, ys])

    # --- 2) Sort for sliding-window proximity checking ---
    # Sort by y primarily (top → bottom), then by x
    order = np.lexsort((locs[:, 0], locs[:, 1]))
    sorted_locs = locs[order]

    # --- 3) Sliding-window collision resolution ---
    min_dist = 0.15   # threshold distance for considering "overlap"
    push_amount = 0.09  # how far outward to push

    for i in range(1, len(sorted_locs)):
        xi, yi = sorted_locs[i]

        # Compare only to a window of recent neighbors (e.g., last 8)
        start = max(0, i - 8)
        for j in range(start, i):
            xj, yj = sorted_locs[j]
            dist = np.hypot(xi - xj, yi - yj)

            if dist < min_dist:
                # Push node i outward from the center
                r = np.hypot(xi, yi)
                if r == 0:
                    # If at center, push upward slightly
                    sorted_locs[i] += np.array([0.0, push_amount])
                else:
                    sorted_locs[i] += push_amount * (sorted_locs[i] / r)

                # Update xi, yi for subsequent checks
                xi, yi = sorted_locs[i]

    # --- 4) Restore original ordering ---
    unsorted_locs = np.zeros_like(sorted_locs)
    unsorted_locs[order] = sorted_locs

    return unsorted_locs


def _get_candidate_edges(edge_trace_idx, old_thresh_mask, new_thresh_mask, update_type, directed, n_nodes):

        ij_iter = _get_ij_iter(n_nodes=n_nodes, directed=directed)
        changed_edges = []
        # print(f"{update_type=}")
        # ---- ALL ----
        if update_type is UpdateType.ALL:
            changed_edges = [
                ((i, j), edge_trace_idx[(i, j)])
                for (i, j) in ij_iter
                if (i, j) in edge_trace_idx
            ]

        # ---- VISIBLE ----, only get visible edges
        if update_type is UpdateType.VISIBLE:
            changed_edges = [
                ((i, j), edge_trace_idx[(i, j)])
                for (i, j) in ij_iter
                if new_thresh_mask[i, j] and (i, j) in edge_trace_idx
            ]
        # ---- THRESHOLD ----, get edges which differ in their visibility between the old_thresh_mask and new_thresh_mask
        if update_type is UpdateType.THRESHOLD:
            diff = (old_thresh_mask != new_thresh_mask)
            # print(diff)
            changed_edges = [
                ((i, j), edge_trace_idx[(i, j)])
                for (i, j) in ij_iter
                if diff[i, j] and (i, j) in edge_trace_idx
            ]
            # print(changed_edges)
            
            

        print(f"Number edges: {len(changed_edges)}")
        return changed_edges


def _update_title(fig: go.Figure, title: str=None):
    if title is not None:
        fig.update_layout(
                title={
                "text": title,
                # "color": "black",
                "y": 0.95,              # vertical position
                "x": 0.5,               # centered
                "xanchor": "center",
                "yanchor": "top",
                "font": dict(size=20)
                }   
            )

VIZ_STR_TO_ENUM = {
    "2d": VizType.FIG2D,
    "3d": VizType.FIG3D,
    "heatmap": VizType.FIGHEATMAP,
}

VIZ_ENUM_TO_STR = {v: k.capitalize() for k, v in VIZ_STR_TO_ENUM.items()}

def str_to_viz_type(s: str) -> VizType:
    try:
        return VIZ_STR_TO_ENUM[s.strip().lower()]
    except KeyError:
        raise ValueError(f"Invalid viz type string: {s}")

def viz_type_to_str(viz: VizType) -> str:
    try:
        return VIZ_ENUM_TO_STR[viz]
    except KeyError:
        raise ValueError(f"Invalid VizType: {viz}")
