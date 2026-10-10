"""Poincare sections interpolated from sampled Boozer trajectories."""

import numpy as np

PERIOD = 2 * np.pi


def section_crossings(path, zeta_section=0.0, direction="positive"):
    """Interpolate crossings of zeta_section modulo 2*pi between saved rows.

    Both Boozer angles are wrapped in GPU output. Unwrapping requires each
    angle to advance by less than pi between samples. Crossings are linear
    interpolations of saved dense-output values, not solver event roots.
    Exclude a segment's left endpoint and include its right endpoint so a
    sample exactly on the section is counted once, excluding the launch.
    """
    path = np.asarray(path, dtype=float)
    if path.ndim != 2 or path.shape[1] != 5:
        raise ValueError("path must have five columns: t, s, theta, zeta, vpar")
    if direction not in ("positive", "negative", "both"):
        raise ValueError("direction must be positive, negative, or both")
    if not np.isfinite(zeta_section) or not np.all(np.isfinite(path)):
        raise ValueError("section and trajectory values must be finite")
    if len(path) < 2:
        return np.empty((0, 5))
    if np.any(np.diff(path[:, 0]) <= 0):
        raise ValueError("saved times must be strictly increasing")
    unwrapped = path.copy()
    unwrapped[:, 2:4] = np.unwrap(path[:, 2:4], axis=0)
    section = zeta_section % PERIOD
    z0, z1 = unwrapped[:-1, 3], unwrapped[1:, 3]
    indices, planes = [], []
    for sign in (1, -1):
        if direction != "both" and (sign == 1) != (direction == "positive"):
            continue
        turns = (z0 - section) / PERIOD
        target = section + PERIOD * (
            np.floor(turns) + 1 if sign == 1 else np.ceil(turns) - 1
        )
        crossed = (
            (z1 >= target) & (z1 > z0) if sign == 1 else (z1 <= target) & (z1 < z0)
        )
        indices.extend(np.flatnonzero(crossed))
        planes.extend(target[crossed])
    if not indices:
        return np.empty((0, 5))
    order = np.argsort(indices)
    indices = np.asarray(indices)[order]
    planes = np.asarray(planes)[order]
    fraction = (planes - z0[indices]) / (z1[indices] - z0[indices])
    crossings = unwrapped[indices] + fraction[:, None] * (
        unwrapped[indices + 1] - unwrapped[indices]
    )
    crossings[:, 3] = section
    # Lost trajectories can contain one endpoint outside the plasma.
    return crossings[(crossings[:, 1] >= 0) & (crossings[:, 1] < 1)]
