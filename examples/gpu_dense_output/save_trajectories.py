"""Save double-precision GPU guiding-center trajectories and plot three orbits."""

import argparse
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nparticles", type=int, default=32)
    parser.add_argument("--tmax", type=float, default=1e-3)
    parser.add_argument("--dt-save", type=float, default=1e-7)
    parser.add_argument("--resolution", type=int, default=15)
    parser.add_argument("--tol", type=float, default=1e-8)
    parser.add_argument(
        "--input",
        type=Path,
        default=HERE.parent / "inputs/boozmn_aten_rescaled_low_res.nc",
    )
    parser.add_argument("--output-dir", type=Path, default=HERE / "output")
    args = parser.parse_args()
    if args.nparticles < 1 or args.resolution < 4:
        parser.error("nparticles must be positive and resolution must be at least 4")
    if any(not np.isfinite(v) or v <= 0 for v in (args.tmax, args.dt_save, args.tol)):
        parser.error("tmax, dt-save, and tol must be finite and positive")

    import matplotlib.pyplot as plt

    from firm3d.catapult.field import CatapultBoozerField
    from firm3d.catapult.tracing import trace_particles_boozer_gpu
    from firm3d.field.boozermagneticfield import BoozerRadialInterpolant
    from firm3d.plotting.plotting_helpers import plot_trajectory_poloidal
    from firm3d.util.constants import (
        ALPHA_PARTICLE_CHARGE as CHARGE,
        ALPHA_PARTICLE_MASS as MASS,
        FUSION_ALPHA_PARTICLE_ENERGY as ENERGY,
    )

    plt.switch_backend("Agg")
    equilibrium = BoozerRadialInterpolant(
        str(args.input.resolve()), 3, enforce_vacuum=True
    )
    field = CatapultBoozerField(
        equilibrium,
        args.resolution,
        args.resolution,
        args.resolution,
        precision="double",
    )
    # Co-passing alpha particles at different radii, initially off the zeta=0 plane.
    stz = np.zeros((args.nparticles, 3))
    stz[:, 0] = np.linspace(0.15, 0.75, args.nparticles)
    stz[:, 2] = 0.1
    vpar = np.full(args.nparticles, 0.9 * np.sqrt(2 * ENERGY / MASS))
    paths, hits = trace_particles_boozer_gpu(
        field,
        stz,
        vpar,
        tmax=args.tmax,
        dt_save=args.dt_save,
        tol=args.tol,
        Ekin=ENERGY,
        mass=MASS,
        charge=CHARGE,
        forget_exact_path=False,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    archive = args.output_dir / "trajectories.npz"
    # Separate numeric arrays allow different path lengths without object arrays
    # or pickle. Reload with np.load(archive, allow_pickle=False).
    arrays = {f"particle_{i:06d}": path for i, path in enumerate(paths)}
    arrays.update({f"hits_{i:06d}": hit for i, hit in enumerate(hits)})
    np.savez_compressed(
        archive,
        **arrays,
        columns=np.array(["t", "s", "theta", "zeta", "vpar"]),
        coordinate_system="boozer",
        initial_positions=stz,
        initial_parallel_speeds=vpar,
        tmax=args.tmax,
        dt_save=args.dt_save,
        tol=args.tol,
        resolution=args.resolution,
        nfp=field.nfp,
        input_file=str(args.input.resolve()),
        mass=MASS,
        charge=CHARGE,
        energy=ENERGY,
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for i in np.unique([0, args.nparticles // 2, args.nparticles - 1]):
        path = paths[i]
        axes[0].plot(path[:, 0] * 1e3, path[:, 1], label=rf"$s_0={stz[i, 0]:.2f}$")
        plot_trajectory_poloidal(path, ax=axes[1])
    axes[0].set(xlabel="Time (ms)", ylabel=r"Normalized toroidal flux $s$")
    axes[0].legend()
    axes[1].set_title("Projection of saved orbits")
    figure = args.output_dir / "trajectories.png"
    fig.savefig(figure, dpi=180)
    plt.close(fig)
    print(f"Saved {len(paths)} trajectories ({sum(map(len, paths))} rows) to {archive}")
    print(f"Lost particles: {sum(bool(len(hit)) for hit in hits)}")
    print(f"Trajectory plot: {figure}")


if __name__ == "__main__":
    main()
