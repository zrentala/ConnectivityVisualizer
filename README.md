---
title: ConnectivityVisualizer
emoji: 🧠
colorFrom: indigo
colorTo: blue
sdk: docker
sdk_version: "1.0"
app_file: app.py
pinned: true
short_description: Viewer of functional connectivity data
---

# ConnectivityVisualizer

An interactive web application for visualizing and analyzing functional brain connectivity data. Built with Dash and Plotly, it supports 2D topographical layouts, 3D brain surface models, and heatmap views with real-time thresholding and network analysis.

---

## Table of Contents

- [Installation](#installation)
- [Quick Start (UI)](#quick-start-ui)
- [Quick Start (Code)](#quick-start-code)
- [Input File Formats](#input-file-formats)
- [UI Guide](#ui-guide)
- [Programmatic API](#programmatic-api)
- [Deployment](#deployment)

---

## Installation

### From source

```bash
git clone https://github.com/your-username/ConnectivityVisualizer.git
cd ConnectivityVisualizer
pip install -r requirements.txt
```

### Docker

```bash
docker build -t connectivity-visualizer .
docker run -p 7860:7860 connectivity-visualizer
```

**Requirements:** Python 3.11+

---

## Quick Start (UI)

Launch the web application:

```bash
python app.py
```

Open your browser to `http://localhost:7860`. The interface has two panes:

- **Left pane** — Data loading, visualization controls, thresholding, and graph analysis settings
- **Right pane** — Interactive figure and a collapsible statistics panel

To get started quickly, use the **Simulate** mode for both connectivity data and electrode locations, then click the **Submit** button.

---

## Quick Start (Code)

Use the package programmatically to load data, apply thresholds, and compute network metrics without launching the UI:

```python
import numpy as np
from data.simulation import generate_conn, generate_locs, build_brain_mesh
from data.loaders import DataLoader, locs_dict_to_dataframe
from utils.braindata import BrainData
from analysis.threshold import Threshold
from analysis.graph import GraphAnalysis

# 1. Generate synthetic data (or load your own — see "Input File Formats")
conn_mat = generate_conn(n_mat=5, n_elec=64, directed=False)
locs_dict = generate_locs(n_elec=64)
chanlocs = locs_dict_to_dataframe(locs_dict)
brain_mesh = build_brain_mesh()

# 2. Create the core data container
brain_data = BrainData(
    conn_mat=conn_mat,
    chanlocs=chanlocs,
    brain_mesh=brain_mesh,
    directed=False,
)

# 3. Threshold the connectivity matrix
threshold = Threshold(threshold=80.0, threshold_type="Basic")
thresholded_mat, mask = threshold.apply_threshold(conn_mat, idx=0)

# 4. Compute graph metrics
ga = GraphAnalysis(
    conn_mat=conn_mat,
    elec_names=list(brain_data.labels),
    directed=False,
    mask=mask,
)

ga.summary()
# Graph Summary:
#   Nodes: 64
#   Total Edges: 2016
#   Edges: 403
#   Density: 0.1862
#   Global Efficiency: 0.4231
#   Local Efficiency: 0.3912
#   Modularity: 0.1274
```

---

## Input File Formats

### Connectivity Matrix

The connectivity matrix represents pairwise functional connectivity values between electrodes. It must be a **square matrix** (or a stack of square matrices).

#### Supported file types

| Format | Extension | Notes |
|--------|-----------|-------|
| NumPy binary | `.npy` | Saved with `np.save()` |
| NumPy archive | `.npz` | Looks for a key named `"conn"`, then falls back to the first array |
| CSV | `.csv` | Comma-delimited, no headers, no index column |
| MATLAB | `.mat` | Uses the first non-internal variable found |

#### Expected shape

- **Single matrix:** `(n_electrodes, n_electrodes)` — automatically expanded to 3D internally
- **Multiple matrices:** `(n_matrices, n_electrodes, n_electrodes)` — for comparing conditions or time points

#### Requirements

- Must be square (rows = columns = number of electrodes)
- Diagonal values are ignored (set to 0 internally)
- For **undirected** connectivity, the matrix should be symmetric
- For **directed** connectivity, asymmetry is preserved
- Values represent connection weights (e.g., coherence, correlation, PLV)

#### Example: creating and saving a connectivity matrix

```python
import numpy as np

# Single 64-electrode connectivity matrix
conn = np.random.rand(64, 64)
conn = (conn + conn.T) / 2   # make symmetric for undirected
np.fill_diagonal(conn, 0)
np.save("my_connectivity.npy", conn)

# Multiple matrices (e.g., 10 subjects)
conn_stack = np.random.rand(10, 64, 64)
for i in range(10):
    conn_stack[i] = (conn_stack[i] + conn_stack[i].T) / 2
    np.fill_diagonal(conn_stack[i], 0)
np.save("my_connectivity_stack.npy", conn_stack)
```

#### Example: CSV format

A 4-electrode connectivity matrix as CSV (no headers, no row labels):

```
0.0,0.82,0.45,0.31
0.82,0.0,0.67,0.29
0.45,0.67,0.0,0.55
0.31,0.29,0.55,0.0
```

---

### Electrode Locations

Electrode locations define the spatial position of each electrode for 2D and 3D rendering.

#### Option 1: CSV file (custom locations)

A CSV file with the following **required columns**:

| Column | Type | Description |
|--------|------|-------------|
| `label` | string | Electrode name (e.g., `"Cz"`, `"Fz"`, `"E1"`) |
| `x` | float | X coordinate in **millimeters** |
| `y` | float | Y coordinate in **millimeters** |
| `z` | float | Z coordinate in **millimeters** |

Example `locations.csv`:

```csv
label,x,y,z
Fp1,-28.5,83.2,3.1
Fp2,28.5,83.2,3.1
F3,-48.1,56.8,45.3
F4,48.1,56.8,45.3
C3,-63.6,0.0,63.6
C4,63.6,0.0,63.6
P3,-48.1,-56.8,45.3
P4,48.1,-56.8,45.3
O1,-28.5,-83.2,3.1
O2,28.5,-83.2,3.1
Fz,0.0,63.6,63.6
Cz,0.0,0.0,90.0
Pz,0.0,-63.6,63.6
```

The number of rows must match the number of electrodes in the connectivity matrix.

#### Option 2: EEG recording files

The following EEG file formats are supported. Electrode locations are extracted from the file's embedded montage via MNE:

| Format | Extension | Loaded via |
|--------|-----------|------------|
| EEGLAB | `.set` | `mne.io.read_raw_eeglab()` |
| BrainVision | `.vhdr` | `mne.io.read_raw_brainvision()` |
| BioSemi / EDF | `.bdf`, `.edf` | `mne.io.read_raw()` |

#### Option 3: Custom montage files

Standard electrode montage file formats:

| Format | Extension |
|--------|-----------|
| Cambridge Electronics Design | `.ced` |
| Custom locations | `.locs` |
| EasyCap locations | `.elc` |
| Polhemus | `.elp` |
| BESA spherical | `.sfp` |

These are loaded via `mne.channels.read_custom_montage()`. Coordinates are automatically converted from meters to millimeters.

#### Option 4: Preset montages

Instead of uploading a file, select one of 30+ standard montages from the dropdown in the UI. Available presets include:

| Montage | Electrodes | Montage | Electrodes |
|---------|------------|---------|------------|
| `standard_1020` | 94 | `biosemi64` | 64 |
| `standard_1005` | 343 | `biosemi128` | 128 |
| `biosemi16` | 16 | `biosemi256` | 256 |
| `biosemi32` | 32 | `EGI_256` | 256 |
| `GSN-HydroCel-32` | 33 | `GSN-HydroCel-128` | 128 |
| `GSN-HydroCel-64_1.0` | 64 | `GSN-HydroCel-256` | 256 |
| `easycap-M1` | 74 | `mgh60` | 60 |
| `easycap-M10` | 61 | `mgh70` | 70 |

When using a preset, the electrode count of the preset must match the electrode count of your connectivity matrix.

---

## UI Guide

### 1. Loading Data

The **Data Controls** section at the top of the left pane has three tabs for each data type:

**Connectivity Matrix (FC):**

| Mode | Description |
|------|-------------|
| **Upload** | Drag and drop a `.npy`, `.npz`, `.csv`, or `.mat` file |
| **Preset** | Select a built-in demo configuration (small, medium, or large) |
| **Simulate** | Generate random connectivity — set the number of electrodes and matrices |

**Electrode Locations (LOC):**

| Mode | Description |
|------|-------------|
| **Upload** | Drag and drop a location file (see [Electrode Locations](#electrode-locations) for supported formats) |
| **Preset** | Select a standard EEG montage from the dropdown (only shows montages that match your electrode count) |
| **Simulate** | Generate random spherical positions |

**Directed checkbox:** Check this if your connectivity matrix is asymmetric (directed graph).

After configuring both FC and LOC, click the **Submit** button to load the data and render the initial visualization.

If you have multiple connectivity matrices (3D array), use the **Matrix Index slider** to switch between them.

### 2. Visualization Controls

**Figure Type:** Switch between three visualization modes:

- **2D** — Standard EEG topographical layout with a head outline, electrode dots, and curved edges. This is the most common format for publication figures.
- **3D** — Electrodes plotted on a 3D brain surface (fsaverage). Includes toggles for left/right hemisphere visibility and brain mesh opacity.
- **Heatmap** — The raw connectivity matrix as a color-coded grid with electrode labels on both axes.

**Colorscale:** Choose from 24 color palettes (Viridis, RdBu, Plasma, Inferno, etc.).

**Color Range:** Adjust the min/max range of the colorscale to emphasize specific value ranges.

**Node and Edge Properties (2D and 3D only):**

| Control | Range | Description |
|---------|-------|-------------|
| Node Size | 15–50 | Radius of electrode dots |
| Edge Width | 0–10 (range) | Line width mapped to connection weight |
| Edge Opacity | 0–1 | Transparency of edges |
| Arc Radius | 0–1 | Curvature of edges (0 = straight, 1 = maximum curve) |

**3D-specific controls:**

| Control | Description |
|---------|-------------|
| Show Left Hemisphere | Toggle left brain mesh visibility |
| Show Right Hemisphere | Toggle right brain mesh visibility |
| Brain Mesh Opacity | 0 (transparent) to 1 (opaque) |

### 3. Thresholding

Thresholding controls which edges are displayed based on connection strength or statistical significance.

**Basic (percentage):** Keep only edges above the Nth percentile of absolute weight. For example, setting the slider to 80 shows only the top 20% strongest connections.

**Minimum Spanning Tree:** Show only the edges that form the maximum spanning tree of the network. This produces a connected, tree-structured subgraph with the strongest possible edges.

**Statistical Test** (requires a 3D connectivity matrix with multiple samples):

| Test | Description |
|------|-------------|
| t-test | One-sample t-test against zero |
| z-test | Z-score normalization test |
| Wilcoxon | Non-parametric signed-rank test |
| Permutation | Sign-flip permutation test with Bonferroni correction |

The **alpha slider** controls the significance level (0–10%). Only edges with p-values below alpha are shown.

### 4. Graph Analysis Controls

**Metric:** Choose which network metric to highlight on the visualization:

| Metric | Description |
|--------|-------------|
| Degree (In/Out/Bidirectional) | Number of connections per node |
| Connection Strength | Sum of edge weights per node |
| Community | Louvain community detection — colors nodes by community |
| None | No metric highlighting |

**Top-X slider:** When using degree or strength metrics, highlights the top N nodes in gold.

### 5. Statistics Panel

Click the arrow on the right edge of the figure to expand the statistics panel. It displays:

- Total nodes and edges
- Visible edges (after thresholding)
- Connection density
- Global efficiency
- Local efficiency
- Modularity score
- Top nodes by degree and strength

All statistics update in real time as you adjust thresholds or switch matrices.

---

## Programmatic API

### `BrainData` — Core data container

```python
from utils.braindata import BrainData

bd = BrainData(
    conn_mat,     # np.ndarray: (n_nodes, n_nodes) or (n_mat, n_nodes, n_nodes)
    chanlocs,     # pd.DataFrame: columns [label, x, y, z]
    brain_mesh,   # pv.PolyData: 3D brain surface mesh
    directed,     # bool: True for directed graphs
)

# Derived attributes:
bd.n_nodes    # int: number of electrodes
bd.labels     # np.ndarray: electrode label strings
```

### `DataLoader` — Load data from files

```python
from data.loaders import DataLoader

loader = DataLoader()

# Build from configuration dictionaries
brain_data, metadata, slider_meta = loader.build_braindata(
    fc_source={"type": "sim", "n_elec": 64, "n_mat": 10, "directed": False},
    loc_source={"type": "preset", "name": "biosemi64"},
    directed=False,
)

# Or build configs step-by-step:
fc_cfg = loader.make_fc_cfg(
    source="upload",
    upload=(base64_contents, "my_data.npy"),
)

loc_cfg = loader.make_loc_cfg(
    source="preset",
    preset="standard_1020",
)

brain_data, metadata, slider_meta = loader.build_braindata(fc_cfg, loc_cfg, directed=False)
```

### `Threshold` — Apply thresholds

```python
from analysis.threshold import Threshold

# Percentage-based: keep top 30% of edges
thresh = Threshold(threshold=70.0, threshold_type="Basic")

# MST: keep only the maximum spanning tree
thresh = Threshold(threshold_type="Minimum Spanning Tree")

# Statistical: t-test at alpha = 0.05
thresh = Threshold(threshold_type="Statistical Test", stat_test="t", alpha=5.0)

# Apply to a connectivity matrix (3D array, select index)
thresholded_matrix, mask = thresh.apply_threshold(conn_mat, idx=0)
# thresholded_matrix: np.ndarray with sub-threshold values zeroed
# mask: boolean np.ndarray (True = edge survives threshold)
```

Available `stat_test` values: `"t"`, `"z"`, `"wilcoxon"`, `"permutation"`

### `GraphAnalysis` — Compute network metrics

```python
from analysis.graph import GraphAnalysis

ga = GraphAnalysis(
    conn_mat=conn_mat,           # (n_mat, n, n) or (n, n)
    elec_names=labels,           # list of electrode name strings
    directed=False,
    mat_idx=0,                   # which matrix to analyze (if 3D)
    agg=None,                    # "mean", "sum", "median" to aggregate across matrices
    mask=threshold_mask,         # optional boolean mask from Threshold
)

# Scalar metrics
ga.density          # float: fraction of possible edges present
ga.num_nodes        # int
ga.total_num_edges  # int: maximum possible edges
ga.visible_num_edges  # int: edges after masking
ga.global_eff       # float: weighted global efficiency
ga.local_eff        # float: weighted local efficiency
ga.modularity       # float: Louvain modularity score

# Per-node metrics
ga.node_degrees     # dict: {node: {"in_degree", "out_degree", "bidirectional"}}
ga.node_strengths   # dict: {node: {"in_strength", "out_strength", "bidirectional_strength"}}
ga.partition        # dict: {node: community_id}

# Underlying NetworkX graph
ga.graph            # nx.Graph or nx.DiGraph

# Print summary
ga.summary()
```

### `Simulation` — Generate synthetic data

```python
from data.simulation import generate_conn, generate_locs, build_brain_mesh
from data.loaders import locs_dict_to_dataframe

# Random connectivity matrices
conn_mat = generate_conn(n_mat=5, n_elec=32, directed=False)
# Returns: np.ndarray of shape (5, 32, 32), symmetric, zero diagonal

# Random electrode locations on a sphere
locs_dict = generate_locs(n_elec=32)
# Returns: {"E1": np.array([x, y, z]), "E2": ..., ...}

# Convert to DataFrame (meters to millimeters)
chanlocs = locs_dict_to_dataframe(locs_dict)

# Fetch fsaverage brain mesh
brain_mesh = build_brain_mesh()
# Returns: pv.PolyData (downloads fsaverage from MNE on first call)
```

### `VizUIManager` — Render figures programmatically

```python
from visualization.vizuimanager import VizUIManager
from analysis.threshold import Threshold

threshold = Threshold(threshold=70.0, threshold_type="Basic")
viz = VizUIManager(brain_data, threshold)

# Get the current Plotly figure
fig = viz.get_figure()
fig.show()           # open in browser
fig.write_image("connectivity_2d.png")   # save to file
```

---

## Deployment

### Local

```bash
python app.py
# Runs on http://localhost:7860
```

The port can be changed via the `PORT` environment variable.

### Docker

```bash
docker build -t connectivity-visualizer .
docker run -p 7860:7860 connectivity-visualizer
```

### Hugging Face Spaces

The repository includes a GitHub Actions workflow (`.github/workflows/sync_to_hf.yml`) that automatically syncs to Hugging Face Spaces on push.
