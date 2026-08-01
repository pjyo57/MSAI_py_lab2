import numpy as np


def pca_via_svd(data, n_components):

    # Center the data
    centered = data - np.mean(data, axis=0)

    # Singular Value Decomposition
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)

    # Principal components
    components = Vt[:n_components]

    # Projection
    projected = centered @ components.T

    return projected, components