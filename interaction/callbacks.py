# callbacks.py
from __future__ import annotations
import numpy as np
from dash import Input, Output
from visualization.ui import viz_tabs_builder

def register_callbacks(app, frame_figs: Dict[str, List[go.Figure]]):
    """
    frame_figs: {
        "2D": [fig2d_0, fig2d_1, ...],
        "3D": [fig3d_0, ...],
        "Heatmap": [fig_hm_0, ...],
    }
    """
    labels = list(frame_figs.keys())
    id_map = viz_tabs_builder.id_map
    n_frames = len(next(iter(frame_figs.values())))

    # One Output per label + one for frame-label
    outputs = [Output(id_map[label], "figure") for label in labels]
    outputs.append(Output("frame-label", "children"))

    @app.callback(
        *outputs,
        Input("mat-idx", "value"),
        prevent_initial_call=False,
    )
    def update_figures(idx):
        idx = int(np.clip(idx or 0, 0, n_frames - 1))

        out_figs = []
        for label in labels:
            fig = frame_figs[label][idx]
            fig.update_layout(uirevision="keep")
            out_figs.append(fig)

        frame_text = f"Frame: {idx+1} / {n_frames}"
        # return all figures plus the label text
        return (*out_figs, frame_text)