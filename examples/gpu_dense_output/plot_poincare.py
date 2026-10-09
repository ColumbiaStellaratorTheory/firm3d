"""Plot approximate toroidal Poincare sections from saved Boozer trajectories."""

import argparse
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input", nargs="?", type=Path, default=HERE / "output/trajectories.npz"
    )
    parser.add_argument(
        "--zeta-section", type=float, default=0.0, help="Angle in radians"
    )
    parser.add_argument(
        "--direction", choices=["positive", "negative", "both"], default="positive"
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if not np.isfinite(args.zeta_section):
        parser.error("zeta-section must be finite")

    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    plt.switch_backend("Agg")
    with np.load(args.input, allow_pickle=False) as saved:
        if str(saved["coordinate_system"]) != "boozer":
            parser.error("the input must contain Boozer trajectories")
        names = sorted(name for name in saved.files if name.startswith("particle_"))
        rows, radii = [], []
        for name in names:
            path = saved[name]
            crossings = section_crossings(path, args.zeta_section, args.direction)
            particle = int(name.removeprefix("particle_"))
            rows.append(np.column_stack((np.full(len(crossings), particle), crossings)))
            radii.extend(np.full(len(crossings), path[0, 1]))
    if not rows or not any(len(row) for row in rows):
        parser.error("no crossings found; increase tmax or select another direction")
    points = np.concatenate(rows)
    radii = np.asarray(radii)
    theta, s = points[:, 3] % PERIOD, points[:, 2]
    output = args.output_dir or args.input.parent
    output.mkdir(parents=True, exist_ok=True)
    csv = output / "poincare.csv"
    np.savetxt(
        csv,
        points,
        delimiter=",",
        header="particle,t_s,s,theta_rad,zeta_rad,vpar_m_per_s",
        comments="",
        fmt=["%d", *(["%.17g"] * 5)],
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), layout="constrained")
    colors = {"c": radii, "cmap": "viridis", "norm": Normalize(0, 1), "s": 3}
    axes[0].scatter(theta, s, **colors)
    axes[0].set(xlabel=r"$\theta$ (rad)", ylabel=r"$s$", xlim=(0, PERIOD), ylim=(0, 1))
    scatter = axes[1].scatter(
        np.sqrt(s) * np.cos(theta), np.sqrt(s) * np.sin(theta), **colors
    )
    angle = np.linspace(0, PERIOD, 200)
    axes[1].plot(np.cos(angle), np.sin(angle), color="black", linewidth=0.8)
    axes[1].set(
        xlabel=r"$\sqrt{s}\cos\theta$", ylabel=r"$\sqrt{s}\sin\theta$", aspect="equal"
    )
    fig.colorbar(scatter, ax=axes, label=r"Initial $s$")
    fig.suptitle(
        rf"$\zeta={args.zeta_section % PERIOD:.2f}$ mod $2\pi$: "
        f"{args.direction} crossings"
    )
    figure = output / "poincare.png"
    fig.savefig(figure, dpi=180)
    plt.close(fig)
    print(f"Interpolated {len(points)} crossings from {len(names)} trajectories")
    print(f"Crossing data: {csv}")
    print(f"Poincare plot: {figure}")


if __name__ == "__main__":
    main()
