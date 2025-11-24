from typing import Protocol
from abc import abstractmethod
import plotly.graph_objects as go
import analysis.threshold as thresh
from utils.braindata import BrainData
from analysis.threshold import Threshold
from itertools import product
from visualization.vizuimanager import UpdateType


class ConnectivityView(Protocol):
    """Interface for all connectivity visualizers (2D, 3D, heatmap, etc.)."""
    @abstractmethod
    def build_figure(self, brain_data: BrainData, threshold: Threshold) -> go.Figure:
        pass
    @abstractmethod
    def update_figure(
        self,
        brain_data: BrainData,
        threshold: Threshold,
        update_type: UpdateType,
    ) -> go.Figure:
        pass

    @abstractmethod
    def get_figure(self) -> go.Figure:
        pass
