import numpy as np
from kneed import KneeLocator


def get_2d_projection(activation_batch):
    # TBD: use pytorch batch svd implementation
    activation_batch[np.isnan(activation_batch)] = 0
    projections = []
    for activations in activation_batch:
        reshaped_activations = (activations).reshape(
            activations.shape[0], -1).transpose()
        # Centering before the SVD seems to be important here,
        # Otherwise the image returned is negative
        reshaped_activations = reshaped_activations - \
            reshaped_activations.mean(axis=0)
        U, S, VT = np.linalg.svd(reshaped_activations, full_matrices=True)
        projection = reshaped_activations @ VT[0, :]
        projection = projection.reshape(activations.shape[1:])
        projections.append(projection)
    return np.float32(projections)


def get_3d_projection(activation_batch, eps=1e-5, n_components=1):
    activation_batch[np.isnan(activation_batch)] = 0
    projections = []

    for activations in activation_batch:
        # activations: (C, H, W, D)
        C, H, W, D = activations.shape
        spatial_shape = (H, W, D)
        reshaped = activations.reshape(C, -1).T  # (N, C)

        # Standardize features: zero mean, unit variance (per feature)
        mean = reshaped.mean(axis=0)
        std = reshaped.std(axis=0) + eps
        normalized = (reshaped - mean) / std  # (N, C)

        # Compute L2 norm per spatial location
        l2_weights = np.linalg.norm(normalized, axis=1)  # (N,)

        # Use kneed to find the cut-off point
        sorted_indices = np.argsort(-l2_weights)  # Sort in descending order
        sorted_weights = l2_weights[sorted_indices]
        kneedle = KneeLocator(range(len(sorted_weights)), sorted_weights, curve="convex", direction="decreasing")
        k = kneedle.knee if kneedle.knee is not None else max(2, int(0.2 * len(sorted_weights)))  # Ensure k >= 2
        k = max(2, k)  # Ensure k is at least 2

        top_idx = sorted_indices[:k]
        focused = normalized[top_idx]  # shape: (k, C)

        # Weight matrix (optional): emphasize high activation locations
        weights = l2_weights[top_idx][:, np.newaxis]  # (k, 1)
        weighted_focused = focused * weights  # (k, C)

        # Perform SVD on weighted activations
        U, S, VT = np.linalg.svd(weighted_focused, full_matrices=False)

        # Use the top `n_components` principal components
        VT_n = - VT[:n_components, :]  # (n_components, C)

        # Project entire volume onto these principal axes
        projection = normalized @ VT_n.T  # (N, n_components)

        # Combine the projections (e.g., sum or mean across components)
        combined_projection = projection.mean(axis=1)  # (N,)
        combined_projection = combined_projection.reshape(spatial_shape)

        # Optional: apply nonlinearity (e.g., ReLU) to suppress noise
        combined_projection = np.maximum(combined_projection, 0)

        projections.append(combined_projection)

    return np.float32(projections)
