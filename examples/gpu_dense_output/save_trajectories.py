"""Save and classify lost fusion-alpha GPU orbits, matching the CPU example."""

import argparse
from pathlib import Path
import time

import numpy as np


HERE = Path(__file__).resolve().parent


def classification_inputs(path, loss_hits):
    """Unwrap angles and interpolate vpar=0 hits from saved dense-output rows.

    Mirror times are sampled crossings, not CPU solver event roots. Reduce
    dt_save to converge them and keep angle increments below pi per sample.
    GPU wall hits retain the first accepted endpoint at or beyond s=1.
    """
    path = np.asarray(path, dtype=float).copy()
    if path.ndim != 2 or path.shape[1] != 5 or not len(path):
        raise ValueError("path must have rows (t, s, theta, zeta, vpar)")
    if not np.all(np.isfinite(path)) or np.any(np.diff(path[:, 0]) <= 0):
        raise ValueError("path must be finite with strictly increasing times")
    path[:, 2:4] = np.unwrap(path[:, 2:4], axis=0)
    v0, v1 = path[:-1, 4], path[1:, 4]
    indices = np.flatnonzero(((v0 < 0) & (v1 >= 0)) | ((v0 > 0) & (v1 <= 0)))
    fraction = -v0[indices] / (v1[indices] - v0[indices])
    mirrors = path[indices] + fraction[:, None] * (path[indices + 1] - path[indices])
    mirrors[:, 4] = 0.0
    hits = np.column_stack((mirrors[:, 0], np.zeros(len(mirrors)), mirrors[:, 1:]))
    if len(loss_hits):
        # The GPU loss hit is the last trajectory row; use its unwrapped angles.
        hits = np.vstack((hits, [path[-1, 0], -1.0, *path[-1, 1:]]))
    return path, hits


def main():
    from firm3d.util.functions import in_github_actions

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--nparticles", type=int, default=50 if in_github_actions else 5000
    )
    parser.add_argument(
        "--tmax", type=float, default=1e-4 if in_github_actions else 1e-2
    )
    parser.add_argument("--dt-save", type=float, default=1e-7)
    parser.add_argument(
        "--resolution", type=int, default=10 if in_github_actions else 48
    )
    parser.add_argument(
        "--tol", type=float, default=1e-4 if in_github_actions else 1e-8
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--input", type=Path, default=HERE.parent / "inputs/boozmn_ariescs.nc"
    )
    parser.add_argument("--output-dir", type=Path, default=HERE / "output")
    args = parser.parse_args()
    if args.nparticles < 1 or args.batch_size < 1 or args.resolution < 4:
        parser.error(
            "particle and batch counts must be positive; resolution must be >=4"
        )
    if any(not np.isfinite(v) or v <= 0 for v in (args.tmax, args.dt_save, args.tol)):
        parser.error("tmax, dt-save, and tol must be finite and positive")

    import matplotlib.pyplot as plt

    from firm3d.catapult.field import CatapultBoozerField
    from firm3d.catapult.tracing import trace_particles_boozer_gpu
    from firm3d.field.boozermagneticfield import InterpolatedBoozerField
    from firm3d.field.tracing_helpers import (
        initialize_position_profile,
        initialize_velocity_uniform,
    )
    from firm3d.plotting.orbit_classification import OrbitClassification
    from firm3d.plotting.plotting_helpers import plot_trajectory_poloidal
    from firm3d.util.constants import (
        ALPHA_PARTICLE_CHARGE as CHARGE,
        ALPHA_PARTICLE_MASS as MASS,
        FUSION_ALPHA_PARTICLE_ENERGY as ENERGY,
    )
    from firm3d.util.functions import sigmav

    started = time.perf_counter()
    plt.switch_backend("Agg")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Same cubic, no-K equilibrium interpolant as the CPU loss example.
    equilibrium = InterpolatedBoozerField.from_booz_xform(
        str(args.input.resolve()),
        degree=3,
        ns=args.resolution,
        ntheta=args.resolution,
        nzeta=args.resolution,
    )

    # nD=nT=1-s**5 and T=11.5*(1-s) keV, with the same seed-zero CPU samplers.
    # Sample the whole ensemble once so changing the batch size changes no launches.
    def reactivity(s):
        return (1 - s**5) ** 2 * sigmav(11.5 * (1 - s))

    stz = initialize_position_profile(equilibrium, args.nparticles, reactivity, seed=0)
    vpar = initialize_velocity_uniform(
        np.sqrt(2 * ENERGY / MASS), args.nparticles, seed=0
    )
    field = CatapultBoozerField(
        equilibrium,
        args.resolution,
        args.resolution,
        args.resolution,
        precision="double",
    )
    classifier = OrbitClassification(equilibrium, ENERGY, MASS, CHARGE, 1, 0)
    lost = np.zeros(args.nparticles, dtype=bool)
    end_times = np.zeros(args.nparticles)
    nbounce = np.zeros(args.nparticles, dtype=int)
    selected, survivors = [], []
    rows_saved = 0
    for first in range(0, args.nparticles, args.batch_size):
        last = min(first + args.batch_size, args.nparticles)
        paths, wall_hits = trace_particles_boozer_gpu(
            field,
            stz[first:last],
            vpar[first:last],
            tmax=args.tmax,
            dt_save=args.dt_save,
            tol=args.tol,
            Ekin=ENERGY,
            mass=MASS,
            charge=CHARGE,
            forget_exact_path=False,
        )
        for i, (path, wall_hit) in enumerate(zip(paths, wall_hits), start=first):
            path, hits = classification_inputs(path, wall_hit)
            end_times[i] = path[-1, 0]
            nbounce[i] = np.count_nonzero(hits[:, 1] == 0)
            lost[i] = bool(len(wall_hit))
            if not lost[i]:
                if len(survivors) < 3:
                    survivors.append((i, path.copy()))
                continue
            # Match the CPU example's loss-only selection and text file names.
            np.savetxt(args.output_dir / f"particle_{i}_traj.txt", path)
            np.savetxt(args.output_dir / f"particle_{i}_hits.txt", hits)
            diagnostics = classifier.classify_orbit(path, hits)
            # Keep numeric diagnostics without the optional object-valued debug_data.
            diagnostics.pop("debug_data")
            np.savez_compressed(
                args.output_dir / f"particle_{i}.npz",
                **diagnostics,
                trajectory=path,
                hits=hits,
            )
            rows_saved += len(path)
            if len(selected) < 3:
                selected.append((i, path.copy()))
        print(
            f"Traced {last}/{args.nparticles}; lost {np.count_nonzero(lost)}",
            flush=True,
        )
        # Release this batch's histories before allocating the next GPU buffer.
        del paths, wall_hits

    np.savez_compressed(
        args.output_dir / "summary.npz",
        columns=np.array(["t", "s", "theta", "zeta", "vpar"]),
        hit_columns=np.array(["t", "type", "s", "theta", "zeta", "vpar"]),
        coordinate_system="boozer",
        initial_positions=stz,
        initial_parallel_speeds=vpar,
        lost=lost,
        end_times=end_times,
        nbounce=nbounce,
        tmax=args.tmax,
        dt_save=args.dt_save,
        tol=args.tol,
        resolution=args.resolution,
        batch_size=args.batch_size,
        nfp=field.nfp,
        input_file=str(args.input.resolve()),
        seed=0,
        helicity_M=1,
        helicity_N=0,
        mass=MASS,
        charge=CHARGE,
        energy=ENERGY,
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for i, path in selected or survivors:
        axes[0].plot(path[:, 0] * 1e3, path[:, 1], label=f"Particle {i}")
        plot_trajectory_poloidal(path, ax=axes[1])
    axes[0].axhline(1, color="black", linestyle="--", linewidth=0.8)
    axes[0].set(xlabel="Time (ms)", ylabel=r"Normalized toroidal flux $s$")
    axes[0].legend()
    axes[1].set_title(
        "Saved lost orbits" if selected else "Surviving orbits (no losses)"
    )
    figure = args.output_dir / "trajectories.png"
    fig.savefig(figure, dpi=180)
    plt.close(fig)
    print(
        f"Saved and classified {np.count_nonzero(lost)} lost trajectories "
        f"({rows_saved} rows)"
    )
    print(f"Results: {args.output_dir.resolve()}")
    print(
        "Total time for tracing, classifying and saving: "
        f"{time.perf_counter() - started:.2f} s"
    )


if __name__ == "__main__":
    main()
