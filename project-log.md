# Functional Connectivity Visualization App  
### Project Log and Design Retrospective

## Introduction

For my undergraduate senior thesis, I compared the neurological effects of action observation therapy when delivered by human therapists versus our lab’s social robot. The primary metric used in this comparison was **functional connectivity (FC)**, which measures statistical dependencies and synchronized activity between spatially separated brain regions.

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
- Edge color, width, curvature, and visibility
- Plot type:
  - 2D topographical layouts
  - 3D brain models
  - Heatmaps
- Threshold levels and number of displayed edges

The goal was to provide full customization without forcing users to modify code.

### 2. Modularity

Each visualization type and analysis component should be logically separated. I aimed to:

- Decouple visualization logic from analysis logic
- Isolate rendering behavior for each plot type
- Ensure new features could be added without modifying core infrastructure

### 3. Insightful Analysis

The tool was not intended to be purely aesthetic. It needed to support exploratory network analysis, including:

- Thresholding controls
- Node and edge counts
- Visible edge counts
- Connection density
- Global efficiency
- Modularity
- Node strength

These metrics reflect common practice in network neuroscience and are often required for early-stage hypothesis generation.

---

## Technology Choice

I selected Plotly as the graphical engine because:

- It provides flexible and interactive plotting tools
- It integrates naturally with Dash for web applications
- It supports both 2D and 3D visualizations
- It allows dynamic updates without full redraws

The app was built using Dash for the frontend and structured backend logic for analysis and visualization coordination.

---

## Architecture

The application architecture separated responsibilities into:

- Visualization components
- Analysis components
- State management
- UI controls

A visualization manager acted as a bridge between frontend controls and backend computation.

Although conceptually clean, the architecture revealed scaling issues during implementation.

---

## Performance Problems

### Computational Complexity

EEG montages commonly include 60 to 300 electrodes. The number of potential edges in a fully connected graph grows quadratically: Total edges = n(n - 1) / 2


For 128 electrodes, this yields 8128 edges. For 256 electrodes, over 32,000 edges.

Because I initially allowed full customization of every node and edge property, any update required iterating over all visible graph elements. Many operations became O(n²).

Even simple UI actions such as:

- Changing color
- Adjusting threshold
- Modifying node size

triggered expensive redraws or property updates across thousands of elements.

---

### Optimization Attempts

To improve performance, I implemented several optimizations:

#### Selective Updates

Instead of redrawing the entire graph:

- If color changed, only visible edges were updated
- If threshold changed, a cached mask was compared against the new mask
- Only edges whose visibility state changed were updated
- If node color changed, only node properties were updated

#### Vectorization

I vectorized matrix operations using NumPy to reduce Python-level loops for any calculations required to update the plot.

#### Batched Updates

Rather than triggering many small updates, I batched property modifications into grouped updates.

These changes significantly improved responsiveness but did not eliminate core structural limitations.

---

### Multithreading Limitations

I attempted to parallelize computation using:

- Multithreading
- Multiprocessing

However, Plotly rendering is not designed for parallel execution in this context. The bottleneck was not purely computation but rendering and UI synchronization.

Because Plotly does not leverage GPU-accelerated OpenGL rendering for these operations, rendering thousands of edges remained costly.

In retrospect, creating an OpenGL-based rendering framework would likely have improved performance significantly, particularly for large, dynamic graphs. However, this would have come at the cost of accessibility and usability. OpenGL tools are typically optimized for real-time rendering rather than producing production-quality, customizable figures. As a result, the application would have been faster but less practical for researchers who need precise control and paper-ready visual outputs.

---

## Modularity Challenges

My initial modular design separated each visualization type into its own class:

- 2D visualization class
- 3D visualization class
- Heatmap class

Each class handled:

- Drawing
- Data processing
- Backend interfacing

While clean in theory, this approach quickly became unmanageable. Adding new features required modifying multiple classes. Shared logic was duplicated, and coupling between visualization and analysis grew unintentionally.

---

## Architectural Refactor

To improve structure, I adopted a more formal object-oriented design approach.

### Abstract Base Class

I created an abstract visualization base class defining core drawing behavior.

### Interfaces for Shared Behaviors

Instead of duplicating logic, I introduced composable behaviors:

- Graph (node and edge) drawing behavior
- Heatmap behavior

Visualization types could inherit only the behaviors they required.

### Visualization Manager

A central visualization manager acted as a mediator between:

- Backend analysis modules
- Visualization classes
- Frontend state

This reduced duplication and improved extensibility, though the foundational performance limitations remained.

---

## What I Learned

### 1. Dashboard Engineering

I gained practical experience building interactive dashboards using:

- Plotly
- Dash
- State-driven UI updates
- Reactive callbacks

I developed a deeper understanding of frontend performance tradeoffs in scientific visualization.

### 2. Performance Diagnosis

I learned how to:

- Identify algorithmic bottlenecks
- Reduce unnecessary redraws
- Cache intermediate computations
- Vectorize operations
- Profile runtime performance

The experience reinforced how quickly O(n²) problems become intractable.

### 3. Limits of Multithreading in Python

I explored Python concurrency models and learned:

- The limitations of threading for CPU-bound tasks
- The overhead of multiprocessing
- The importance of understanding rendering bottlenecks versus computational bottlenecks

### 4. Object-Oriented Design Patterns

This project forced me to think more deeply about:

- Abstraction
- Interfaces
- Separation of concerns
- Architectural scalability

I gained appreciation for designing systems around extensibility rather than immediate functionality.

### 5. Containerization and Deployment

I containerized the application using Docker and deployed it via Hugging Face Spaces. This introduced:

- Environment reproducibility practices
- Dependency management discipline
- Lightweight production deployment workflows

---

## Reflection

This project did not end as a polished production tool, but it was one of the most instructive engineering experiences I have had.

The central mistake was prioritizing maximal flexibility before understanding scaling constraints. By allowing full control over every graphical element without bounding computational cost, I designed myself into a performance ceiling.

However, the failure was productive. It forced me to confront:

- Algorithmic complexity
- UI-performance tradeoffs
- System architecture design
- Maintainability considerations

If rebuilding this project, I would:

- Constrain visualization flexibility early
- Use GPU-accelerated rendering
- Design around incremental state updates from the beginning
- Define clear performance budgets before feature expansion

Despite its limitations, the project strengthened my understanding of both network neuroscience visualization and scalable software design. I believe the underlying code could still be highly valuable to researchers as one of the first functional connectivity visualization packages built.