# Functional Connectivity Visualization App
### Project Log and Design Retrospective

## Introduction

For my undergraduate senior thesis, I compared the neurological effects of action observation therapy when delivered by human therapists versus our lab's social robot. The primary metric used in this comparison was **functional connectivity (FC)**, which measures statistical dependencies and synchronized activity between spatially separated brain regions.

Functional connectivity analysis is central to network neuroscience, yet I found there were no accessible tools that allowed researchers to easily generate publication-quality FC visualizations while also performing lightweight exploratory analysis.

To address this gap, I set out to build an interactive application that would:

- Generate flexible, visually appealing FC plots suitable for research papers
- Allow researchers to explore thresholding and network statistics in real time
- Provide intuitive controls for customizing visualization parameters

Although I developed a working prototype, early architectural decisions significantly limited performance and maintainability. This document outlines my design goals, the challenges I encountered, the optimizations I attempted, and the lessons I learned.

---

## Design Principles

While planning the application, I centered development around three core principles.

### 1. Flexibility

Users should be able to control every aspect of the visualization, including:

- Node color, size, and position
- Edge color, width, curvature, opacity, and visibility
- Plot type:
  - **2D topographical layouts** with standard EEG head outlines
  - **3D brain surface models** with hemisphere toggles and mesh opacity
  - **Heatmaps** for matrix-level inspection
- Threshold levels, statistical tests, and number of displayed edges
- Colorscale selection (Viridis, RdBu, and other Plotly sequential scales)

The goal was to provide full customization without forcing users to modify code.

### 2. Modularity

Each visualization type and analysis component should be logically separated. I aimed to:

- Decouple visualization logic from analysis logic
- Isolate rendering behavior for each plot type
- Ensure new features could be added without modifying core infrastructure

### 3. Insightful Analysis

The tool was not intended to be purely aesthetic. It needed to support exploratory network analysis, including:

- Thresholding controls (percentage-based, statistical tests, and minimum spanning tree)
- Node and edge counts (total and visible)
- Connection density
- Global and local efficiency (weighted inverse shortest path metrics)
- Modularity (Louvain community detection)
- Node degree (in-degree, out-degree, bidirectional)
- Node connection strength

These metrics reflect common practice in network neuroscience and are often required for early-stage hypothesis generation.

---

## Technology Stack

| Library | Role |
|---------|------|
| **Dash** | Web application framework (reactive callbacks, layout) |
| **Plotly** | Interactive 2D and 3D figure rendering |
| **Dash Bootstrap Components** | UI styling (Cyborg dark theme) |
| **NumPy** | Matrix operations, masking, vectorized computation |
| **Pandas** | Electrode location storage (DataFrames) |
| **SciPy** | Statistical tests (t-test, Wilcoxon, z-test), minimum spanning tree |
| **NetworkX** | Graph construction and network metric computation |
| **python-louvain** | Community detection (modularity-based partition) |
| **PyVista** | 3D brain mesh handling (fsaverage surface) |
| **MNE** | Standard EEG electrode montages (30+ presets) and brain surface data |

I selected Plotly as the graphical engine because it provides interactive plotting, integrates naturally with Dash for web applications, supports both 2D and 3D visualizations, and allows dynamic updates without full redraws.

---

## Architecture

The application is organized into five packages, each with a distinct responsibility:

```
ConnectivityVisualizer/
├── app.py                        # Entry point: Dash app, layout, callback registration
├── visualization/                # Rendering pipeline
│   ├── vizconn.py                # ConnectivityView ABC + 2D, 3D, Heatmap subclasses
│   ├── vizuimanager.py           # VizUIManager: orchestrates figure builds/updates
│   ├── vizhelpers.py             # UpdateType/VizType enums, Bezier curves, color utilities
│   ├── ui.py                     # Dash layout (split-pane: controls left, figure right)
│   └── brainvisualizer.py        # Brain-specific 3D rendering logic
├── interaction/                  # User interaction layer
│   ├── callbacks.py              # Dash callback registration and update dispatch
│   └── ui_controls.py            # Reusable UI component builders (sliders, dropdowns)
├── analysis/                     # Network analysis
│   ├── graph.py                  # GraphAnalysis: builds NetworkX graph, computes metrics
│   └── threshold.py              # Threshold: percentage, MST, and statistical tests
├── data/                         # Data ingestion
│   ├── loaders.py                # DataLoader factory: upload, preset, or simulate
│   └── simulation.py             # Synthetic data generation (random matrices, spherical locs)
└── utils/                        # Shared utilities
    ├── braindata.py              # BrainData dataclass: core data container
    ├── global_app_state.py       # GlobalAppState: app-wide mutable state
    ├── io.py                     # File I/O for matrices and configs
    └── update.py                 # Attribute update helpers
```

### Core Data Container: `BrainData`

The `BrainData` dataclass (`utils/braindata.py`) serves as the central data structure. It holds:

- `conn_mat`: a NumPy array of connectivity values, either 2D `(n_nodes, n_nodes)` for a single matrix or 3D `(n_matrices, n_nodes, n_nodes)` for multiple conditions or time points
- `chanlocs`: a Pandas DataFrame with electrode positions (columns: `label`, `x`, `y`, `z`)
- `brain_mesh`: a PyVista `PolyData` object for 3D brain surface rendering (fsaverage)
- `directed`: boolean flag for directed vs. undirected graphs
- Derived fields computed on initialization: `n_nodes` and `labels`

### Application State: `GlobalAppState`

`GlobalAppState` (`utils/global_app_state.py`) holds four mutable references that persist across callbacks:

- `brain_data`: the active `BrainData` instance
- `threshold`: the current `Threshold` settings
- `viz`: the `VizUIManager` instance managing all three visualizers
- `graph_analysis`: the cached `GraphAnalysis` instance

### Data Pipeline: `DataLoader`

The `DataLoader` class (`data/loaders.py`) is a factory that assembles `BrainData` from three possible sources:

- **Upload**: user-provided files (`.npy`, `.npz`, `.csv`, `.mat` for connectivity; `.csv`, `.set`, `.vhdr`, `.bdf`, `.edf`, `.ced` for electrode locations)
- **Preset**: predefined configurations using MNE standard montages (30+ EEG systems from BioSemi, EGI, HydroCel, EasyCap, etc.)
- **Simulate**: synthetic random connectivity matrices and spherical electrode positions generated via the `Simulation` class

### Visualization Pipeline

The visualization system follows a layered architecture:

1. **`ConnectivityView` (abstract base class)**: defines the interface (`build_figure`, `update_figure`, `update_attributes`) that all visualizers implement.

2. **Concrete visualizers**:
   - `ConnectivityView2D`: renders a standard EEG topographical layout with a head outline, nose indicator, electrode scatter points, and quadratic Bezier curved edges. Supports arrowheads for directed graphs.
   - `ConnectivityView3D`: renders electrodes on a 3D brain surface (left/right hemispheres from fsaverage mesh) with concave 3D edge curves. Supports hemisphere visibility toggles and mesh opacity.
   - `ConnectivityViewHeatmap`: renders the connectivity matrix as a color-mapped heatmap with labeled axes.

3. **`HandlesNodes` (mixin class)**: shared functionality for 2D and 3D views, including node sizing, edge width mapping, opacity, arc radius, and trace index caching for fast in-place updates.

4. **`VizUIManager` (mediator)**: holds instances of all three visualizers in a dictionary keyed by `VizType`. It dispatches `build_figure` for full rebuilds and `update_figure` for fast in-place updates, and caches the threshold mask (`_mask_cache`) to enable differential updates.

### Update Dispatch System

Rather than rebuilding the entire figure on every UI interaction, the callback system classifies each trigger into an `UpdateType` enum:

| UpdateType | Trigger Examples | Behavior |
|---|---|---|
| `THRESHOLD` | Threshold slider, alpha, test type | Diff old vs. new mask; toggle only changed edges |
| `VISIBLE` | Color, edge width, opacity, arc radius | Update visual properties of existing traces |
| `NODES` | Node size, graph metric, community | Update node trace properties |
| `SWITCH_FIG` / `ALL` | Figure type dropdown, matrix index | Full figure rebuild |
| `NONE` | Unrecognized trigger | No-op |

This classification drives selective updates, avoiding expensive full redraws for most interactions.

### Analysis Components

**`Threshold`** (`analysis/threshold.py`) supports three thresholding approaches:

- **Percentage-based**: retains the top X% of edges by absolute weight
- **Statistical tests** (for 3D multi-sample matrices): one-sample t-test, z-test, Wilcoxon signed-rank test, and permutation test (sign-flip with Bonferroni correction)
- **Minimum Spanning Tree (MST)**: retains only edges in the maximum spanning tree via SciPy

**`GraphAnalysis`** (`analysis/graph.py`) constructs a NetworkX graph from the thresholded connectivity matrix and computes:

- Density, node count, total and visible edge counts
- Node degrees (in, out, bidirectional) and connection strengths
- Weighted global efficiency (inverse shortest path via Dijkstra)
- Weighted local efficiency (average subgraph global efficiency)
- Modularity and community partition (Louvain algorithm via `python-louvain`)

### Callback Architecture

All interactivity is driven by Dash callbacks registered in `interaction/callbacks.py`:

- `register_data_callbacks`: handles data source selection (upload/preset/simulate), file decoding, location matching, and `BrainData` assembly
- `register_visualization_callback`: the main rendering callback with 15 inputs (sliders, dropdowns, checklists) that dispatches updates through `VizUIManager` and refreshes the statistics panel
- `register_threshold_callback`: toggles threshold UI visibility between percentage slider and statistical test controls
- `register_viz_control_callback`: shows/hides node controls and 3D-specific settings based on the active figure type
- `register_stat_toggle_callback`: manages the collapsible statistics panel

### Deployment

- **Entry point**: `app.py` creates the Dash app with the Cyborg Bootstrap theme, sets the layout, registers callbacks, and runs on port 7860
- **Docker**: containerized with Python 3.11, non-root user (Hugging Face requirement)
- **CI/CD**: GitHub Actions workflow (`.github/workflows/sync_to_hf.yml`) syncs the repository to Hugging Face Spaces

---

## Performance Problems

### Computational Complexity

EEG montages commonly include 60 to 300 electrodes. The number of potential edges in a fully connected graph grows quadratically:

> Total edges = n(n - 1) / 2

For 128 electrodes, this yields 8,128 edges. For 256 electrodes, over 32,000 edges.

Because I initially allowed full customization of every node and edge property, any update required iterating over all visible graph elements. Many operations became O(n^2).

Even simple UI actions such as:

- Changing color
- Adjusting threshold
- Modifying node size

triggered expensive redraws or property updates across thousands of elements.

---

### Optimization Attempts

To improve performance, I implemented several optimizations:

#### Selective Updates via `UpdateType` Dispatch

Instead of redrawing the entire graph on every callback, I built a classification system (`determine_update_type_from_trigger` in `callbacks.py`) that maps each UI control ID to an `UpdateType` enum. The `VizUIManager` then dispatches only the minimum necessary update:

- If threshold changed, the old mask is compared against the new mask, and only edges whose visibility state changed are toggled
- If color or edge style changed, only visual properties of existing Plotly traces are modified in place
- If node properties changed, only the node scatter trace is updated

#### Trace Index Caching

Each `ConnectivityView` subclass caches the Plotly trace indices for nodes and edges (`_node_trace_idx`, per-edge trace indices). This eliminates the need to search through `fig.data` on every update and enables O(1) access to individual traces.

#### Vectorization

I vectorized matrix operations using NumPy (masking, thresholding, diagonal filling) to reduce Python-level loops for calculations required to update the plot.

#### Batched Updates

Rather than triggering many small Plotly figure mutations, I batched property modifications into grouped updates before returning the figure to the browser.

These changes significantly improved responsiveness but did not eliminate core structural limitations.

---

### Multithreading Limitations

I attempted to parallelize computation using multithreading and multiprocessing (a parallel variant exists in `visualization/vizconn_parallel.py`). However, Plotly rendering is not designed for parallel execution in this context. The bottleneck was not purely computation but rendering and UI synchronization.

Because Plotly does not leverage GPU-accelerated OpenGL rendering for these operations, rendering thousands of edges remained costly.

In retrospect, creating an OpenGL-based rendering framework would likely have improved performance significantly, particularly for large, dynamic graphs. However, this would have come at the cost of accessibility and usability. OpenGL tools are typically optimized for real-time rendering rather than producing production-quality, customizable figures. As a result, the application would have been faster but less practical for researchers who need precise control and paper-ready visual outputs.

---

## Modularity Challenges

My initial modular design separated each visualization type into its own class, each handling drawing, data processing, and backend interfacing independently. While clean in theory, this approach quickly became unmanageable. Adding new features required modifying multiple classes. Shared logic was duplicated, and coupling between visualization and analysis grew unintentionally.

---

## Architectural Refactor

To improve structure, I adopted a more formal object-oriented design approach.

### Abstract Base Class: `ConnectivityView`

I created an abstract base class (`ConnectivityView` in `visualization/vizconn.py`) defining the core interface that all visualizers must implement:

- `build_figure()`: constructs a complete Plotly figure from scratch
- `update_figure()`: performs fast, in-place modifications based on an `UpdateType`
- `update_attributes()`: applies UI control changes to the visualizer's internal state

### Mixin for Shared Behaviors: `HandlesNodes`

Instead of duplicating node and edge rendering logic, I introduced the `HandlesNodes` mixin class. It encapsulates:

- Node sizing and fill color
- Edge width range mapping, opacity, and arc radius
- 2D/3D coordinate normalization
- Trace index caching for efficient updates

`ConnectivityView2D` and `ConnectivityView3D` both inherit from `ConnectivityView` and `HandlesNodes`, while `ConnectivityViewHeatmap` inherits only from `ConnectivityView` since it does not render individual nodes or edges.

### Visualization Manager: `VizUIManager`

The `VizUIManager` class (`visualization/vizuimanager.py`) acts as a mediator between:

- The Dash callback layer (which translates UI events into structured update dictionaries)
- The three `ConnectivityView` instances (which handle actual rendering)
- The threshold mask cache (which enables differential edge updates)

It holds all three visualizers in a dictionary and delegates to the currently active one based on `VizType`. This eliminated the need for callbacks to know about specific visualizer implementations.

---

## What I Learned

### 1. Dashboard Engineering

I gained practical experience building interactive dashboards using Dash and Plotly with state-driven UI updates and reactive callbacks. I built a responsive split-pane layout with collapsible panels, dynamic control visibility (showing 3D controls only for 3D views), and a live statistics panel that recalculates network metrics as the user adjusts thresholds.

### 2. Performance Diagnosis

I learned how to identify algorithmic bottlenecks, reduce unnecessary redraws through update classification, cache intermediate computations (threshold masks, trace indices), vectorize operations with NumPy, and profile runtime performance. The experience reinforced how quickly O(n^2) problems become intractable when applied to interactive UIs.

### 3. Limits of Multithreading in Python

I explored Python concurrency models and learned the limitations of threading for CPU-bound tasks (GIL), the overhead of multiprocessing for rendering-bound work, and the importance of distinguishing rendering bottlenecks from computational bottlenecks.

### 4. Object-Oriented Design Patterns

This project forced me to think more deeply about abstraction, composition (mixins vs. inheritance), separation of concerns, and the mediator pattern. The refactor from duplicated per-class logic to `ConnectivityView` + `HandlesNodes` + `VizUIManager` was a direct application of these principles.

### 5. Scientific Computing Integration

I integrated multiple scientific Python libraries (MNE for electrode montages and brain surfaces, NetworkX for graph metrics, SciPy for statistical tests and MST computation, PyVista for 3D mesh handling) into a cohesive pipeline. This required understanding each library's data formats and coordinate conventions.

### 6. Containerization and Deployment

I containerized the application using Docker (Python 3.11, non-root user) and deployed it via Hugging Face Spaces with GitHub Actions for automated sync. This introduced environment reproducibility practices, dependency management discipline, and lightweight production deployment workflows.

---

## Reflection

This project did not end as a polished production tool, but it was one of the most instructive engineering experiences I have had.

The central mistake was prioritizing maximal flexibility before understanding scaling constraints. By allowing full control over every graphical element without bounding computational cost, I designed myself into a performance ceiling.

However, the failure was productive. It forced me to confront:

- Algorithmic complexity in interactive systems
- UI-performance tradeoffs specific to scientific visualization
- System architecture design under real constraints
- Maintainability considerations when balancing OOP patterns with performance

If rebuilding this project, I would:

- Constrain visualization flexibility early and define clear performance budgets before feature expansion
- Use GPU-accelerated rendering (WebGL or a library like `deck.gl`) for large graphs
- Design around incremental state updates from the beginning, rather than retrofitting them
- Consider a client-side rendering approach for edge-heavy visualizations to avoid server round-trips

Despite its limitations, the project strengthened my understanding of both network neuroscience visualization and scalable software design. I believe the underlying code could still be highly valuable to researchers as one of the first interactive functional connectivity visualization packages built.
