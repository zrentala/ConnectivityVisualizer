import numpy as np
import pandas as pd
import pyvista as pv
import mne

def visualize_connectivity_3d(conn_matrix, electrode_locs, brain_mesh, threshold=0.7, show_labels=True):
    """
    Render a 3D interactive brain connectivity visualization using PyVista.

    Parameters
    ----------
    conn_matrix : np.ndarray
        n x n symmetric connectivity matrix (values between 0 and 1 recommended)
    electrode_locs : pd.DataFrame
        DataFrame with columns ['label', 'x', 'y', 'z']
    brain_mesh : pyvista.PolyData
        Brain surface mesh (PyVista object)
    threshold : float
        Minimum connectivity strength to display an edge
    show_labels : bool
        Whether to show electrode text labels
    """

    n = len(electrode_locs)
    assert conn_matrix.shape == (n, n), "Matrix size must match number of electrodes."

    # Create plotter
    plotter = pv.Plotter()
    plotter.add_mesh(brain_mesh, color="lightgray", opacity=0.2, smooth_shading=True)

    # Add electrodes
    for i, row in electrode_locs.iterrows():
        sphere = pv.Sphere(center=(row.x, row.y, row.z), radius=2.0)
        plotter.add_mesh(sphere, color="royalblue", smooth_shading=True)
        if show_labels:
            plotter.add_point_labels([sphere.center], [row.label],
                                     text_color="black", font_size=10, point_size=0)

    # Add connectivity edges
    for i in range(n):
        for j in range(i + 1, n):
            weight = conn_matrix[i, j]
            if weight >= threshold:
                line = pv.Line(
                    (electrode_locs.loc[i, 'x'], electrode_locs.loc[i, 'y'], electrode_locs.loc[i, 'z']),
                    (electrode_locs.loc[j, 'x'], electrode_locs.loc[j, 'y'], electrode_locs.loc[j, 'z'])
                )
                plotter.add_mesh(
                    line,
                    color=(1.0, 0.0, 0.0),  # red
                    line_width=1 + 3 * weight,
                    opacity=min(0.2 + weight, 1.0)
                )

    plotter.add_axes()
    plotter.show_grid()
    plotter.show()


# Example usage
if __name__ == "__main__":
    # --------------------------
    # Step 1: Load MNE fsaverage brain
    # --------------------------
    # Step 1: Load both hemispheres of fsaverage
    fs_dir = mne.datasets.fetch_fsaverage(verbose=True)
    lh_pial = fs_dir / 'surf' / 'lh.pial'
    rh_pial = fs_dir / 'surf' / 'rh.pial'

    # Load both surfaces from MNE
    coords_lh, faces_lh = mne.read_surface(lh_pial)
    coords_rh, faces_rh = mne.read_surface(rh_pial)

    # Combine into one PyVista mesh
    faces_lh = np.column_stack((np.full(len(faces_lh), 3), faces_lh))
    faces_rh = np.column_stack((np.full(len(faces_rh), 3), faces_rh + len(coords_lh)))  # shift indices

    coords_combined = np.vstack((coords_lh, coords_rh))
    faces_combined = np.vstack((faces_lh, faces_rh))

    brain_mesh = pv.PolyData(coords_combined, faces_combined)

    # --------------------------
    # Step 2: Create sample data
    # --------------------------
    n = 12
    np.random.seed(42)
    conn = np.random.rand(n, n)
    conn = (conn + conn.T) / 2
    np.fill_diagonal(conn, 0)

    locs = pd.DataFrame({
        'label': [f'E{i}' for i in range(n)],
        'x': np.random.uniform(-40, 40, n),
        'y': np.random.uniform(-60, 60, n),
        'z': np.random.uniform(-30, 30, n)
    })

    # --------------------------
    # Step 3: Visualize
    # --------------------------
    visualize_connectivity_3d(conn, locs, brain_mesh, threshold=0.75)
