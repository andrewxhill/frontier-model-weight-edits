"""Joint, prior-centered ridge fit for island scores and zero camera updates."""

import numpy as np
from scipy.linalg import cho_factor, cho_solve


def _finite_array(value, name):
    array = np.asarray(value)
    if array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numbers")
    array = array.astype(np.float64, copy=False)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def _positive_scalar(value, name):
    array = _finite_array(value, name)
    if array.ndim != 0 or array <= 0:
        raise ValueError(f"{name} must be a positive scalar")
    return float(array)


def _domain_weights(value, rows, name):
    if value is None:
        return np.full(rows, 1.0 / rows)

    weights = _finite_array(value, name)
    if weights.shape != (rows,):
        raise ValueError(f"{name} must have shape ({rows},)")
    maximum = weights.max()
    if np.any(weights < 0) or maximum == 0:
        raise ValueError(f"{name} must be nonnegative with a positive sum")
    scaled = weights / maximum
    return scaled / scaled.sum()


def joint_ridge(Xi, Yi, Xc, W0, camera_weight, lam=0.01, wi=None, wc=None):
    """Return slopes, intercept and diagnostics for the joint weighted fit.

    Xi and Xc are row-by-feature matrices. Yi holds the island target
    scores and W0 is the feature-by-output prior. Camera target scores
    are zero. Normalize row weights within each domain before applying
    the camera-to-island weight ratio. The diagnostics report per-domain
    weighted MSE averaged over outputs and actual combined weight totals.
    """
    Xi = _finite_array(Xi, "Xi")
    Yi = _finite_array(Yi, "Yi")
    Xc = _finite_array(Xc, "Xc")
    W0 = _finite_array(W0, "W0")
    rho = _positive_scalar(camera_weight, "camera_weight")
    lam = _positive_scalar(lam, "lam")

    if Xi.ndim != 2 or not Xi.size:
        raise ValueError("Xi must be a nonempty row-by-feature matrix")
    island_rows, features = Xi.shape
    if Xc.ndim != 2 or not Xc.size or Xc.shape[1] != features:
        raise ValueError("Xc must be a nonempty matrix with the same features as Xi")
    if Yi.ndim != 2 or not Yi.size or Yi.shape[0] != island_rows:
        raise ValueError("Yi must be a nonempty target matrix with one row per Xi row")
    camera_rows = Xc.shape[0]
    outputs = Yi.shape[1]
    if W0.shape != (features, outputs):
        raise ValueError(f"W0 must have shape ({features}, {outputs})")

    wi = _domain_weights(wi, island_rows, "wi")
    wc = _domain_weights(wc, camera_rows, "wc")
    weights = np.concatenate((wi / (1 + rho), (rho / (1 + rho)) * wc))
    X = np.vstack((Xi, Xc))
    Y = np.vstack((Yi, np.zeros((camera_rows, outputs))))
    mean_x = weights @ X
    mean_y = weights @ Y
    root_weights = np.sqrt(weights)[:, None]

    A = root_weights * (X - mean_x)
    B = root_weights * (Y - mean_y)
    gram = A.T @ A
    scale = max(float(np.trace(gram) / features), 1e-12)
    ridge = lam * scale
    factor = cho_factor(gram + ridge * np.eye(features), lower=True)
    W = cho_solve(factor, A.T @ B + ridge * W0)
    b = mean_y - mean_x @ W

    squared_error = ((X @ W + b - Y) ** 2).mean(axis=1)
    diagnostics = {
        "island_mse": float(wi @ squared_error[:island_rows]),
        "camera_mse": float(wc @ squared_error[island_rows:]),
        "scale": scale,
        "island_rows": island_rows,
        "camera_rows": camera_rows,
        "features": features,
        "outputs": outputs,
        "island_weight_total": float(weights[:island_rows].sum()),
        "camera_weight_total": float(weights[island_rows:].sum()),
    }
    return W, b, diagnostics
