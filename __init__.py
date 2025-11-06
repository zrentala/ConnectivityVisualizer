"""
Connectivity Visualizer
=======================

A modular package for visualizing and analyzing neural connectivity data.

Modules
--------
- data: Load and preprocess input matrices and electrode locations.
- analysis: Compute thresholds, graph metrics, and statistical properties.
- visualization: Render 3D, 2D, and circular brain connectivity visualizations.
- utils: Helper functions for formatting, color maps, and math utilities.
"""

# --- Version info ---
__version__ = "0.1.0"
__author__ = "Zachary Rentala"

# --- Logging setup (optional but good practice) ---
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info(f"Connectivity Visualizer {__version__} loaded.")

# --- Subpackage imports ---
from . import data
from . import analysis
from . import visualization
from . import utils

# --- Define a clean public API ---
__all__ = [
    "data",
    "analysis",
    "visualization",
    "utils"
]
