# visualization/plotly_3d.py
import numpy as np
import pandas as pd
import plotly.graph_objects as go

def visualize_connectivity_plotly(conn_matrix, electrode_locs, threshold=0.7):
    """
    Create a 3D connectivity visualization as a Plotly figure.
    """
    n = len(electrode_locs)
    assert conn_matrix.shape == (n, n), "Matrix size must match number of electrodes."

    # --- Add electrode nodes ---
    scatter = go.Scatter3d(
        x=electrode_locs['x'],
        y=electrode_locs['y'],
        z=electrode_locs['z'],
        mode='markers+text',
        text=electrode_locs['label'],
        textposition="top center",
        marker=dict(size=5, color='royalblue', opacity=0.8),
        name='Electrodes'
    )

    # --- Add connectivity edges ---
    edge_traces = []
    for i in range(n):
        for j in range(i + 1, n):
            weight = conn_matrix[i, j]
            if weight >= threshold:
                edge_traces.append(go.Scatter3d(
                    x=[electrode_locs.loc[i, 'x'], electrode_locs.loc[j, 'x'], None],
                    y=[electrode_locs.loc[i, 'y'], electrode_locs.loc[j, 'y'], None],
                    z=[electrode_locs.loc[i, 'z'], electrode_locs.loc[j, 'z'], None],
                    mode='lines',
                    line=dict(width=1 + 4 * weight, color='red'),
                    opacity=min(0.2 + weight, 1.0),  # moved here!
                    hoverinfo='none'
                ))

    # --- Combine all traces ---
    fig = go.Figure(data=[scatter] + edge_traces)

    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        showlegend=False
    )

    return fig
