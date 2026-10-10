"""Kinetic passing Poincare map with CPU or CATAPULT tracing."""

import argparse
from pathlib import Path
import time

import numpy as np

from firm3d.field.boozermagneticfield import InterpolatedBoozerField
from firm3d.trajectory_helpers import PassingPoincare
from firm3d.util.constants import (
    ALPHA_PARTICLE_CHARGE,
    ALPHA_PARTICLE_MASS,
    FUSION_ALPHA_PARTICLE_ENERGY,
)
from firm3d.util.functions import in_github_actions, proc0_print, setup_logging
from firm3d.util.mpi import comm_size, comm_world, verbose

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["cpu", "catapult"], default="cpu")
    parser.add_argument(
        "--resolution", type=int, default=10 if in_github_actions else 48
    )
    parser.add_argument("--ns-poinc", type=int, default=5 if in_github_actions else 120)
    parser.add_argument("--nmaps", type=int, default=5 if in_github_actions else 1000)
    parser.add_argument(
        "--tol", type=float, default=1e-4 if in_github_actions else 1e-8
    )
    parser.add_argument("--tmax", type=float, default=1e-2)
    parser.add_argument("--dt-save", type=float, default=1e-7)
    parser.add_argument(
        "--input", type=Path, default=HERE.parent / "inputs/boozmn_aten_rescaled.nc"
    )
    parser.add_argument("--output-dir", type=Path, default=HERE / "output")
    args = parser.parse_args()
    if args.ns_poinc < 1 or args.nmaps < 1 or args.resolution < 4:
        parser.error("ns-poinc/nmaps must be positive; resolution must be at least 4")
    if any(not np.isfinite(v) or v <= 0 for v in (args.tol, args.tmax, args.dt_save)):
        parser.error("tol, tmax, and dt-save must be finite and positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(str(args.output_dir / f"stdout_{args.backend}_{comm_size}.txt"))
    started = time.perf_counter()
    field = InterpolatedBoozerField.from_booz_xform(
        str(args.input.resolve()),
        degree=3,
        ns=args.resolution,
        ntheta=args.resolution,
        nzeta=args.resolution,
        comm=comm_world,
    )
    if args.backend == "catapult":
        from firm3d.catapult.field import CatapultBoozerField

        field = CatapultBoozerField(
            field, args.resolution, args.resolution, args.resolution
        )

    # Same ATEN equilibrium, alpha energy, pitch, launch grid, equations,
    # tolerance, and momentum/WBA diagnostics for either backend.
    poinc = PassingPoincare(
        field,
        lam=0.0,
        sign_vpar=1.0,
        mass=ALPHA_PARTICLE_MASS,
        charge=ALPHA_PARTICLE_CHARGE,
        Ekin=FUSION_ALPHA_PARTICLE_ENERGY,
        ns_poinc=args.ns_poinc,
        ntheta_poinc=1,
        Nmaps=args.nmaps,
        comm=comm_world,
        tmax=args.tmax,
        dt_save=args.dt_save,
        helicity_N=field.nfp,
        helicity_M=1,
        solver_options={"reltol": args.tol, "abstol": args.tol},
        chaos_detection=True,
    )
    proc0_print("poincare time: ", time.perf_counter() - started)
    if verbose:
        arrays = {}
        for i, (s, theta, vpar, transits, peta, accuracy, steps) in enumerate(
            zip(
                poinc.s_all,
                poinc.thetas_all,
                poinc.vpars_all,
                poinc.t_all,
                poinc.peta_all,
                poinc.DA_all,
                poinc.DA_times,
            )
        ):
            arrays[f"particle_{i:06d}"] = np.column_stack(
                (np.cumsum(transits), s, theta, np.zeros(len(s)), vpar)
            )
            arrays[f"momentum_{i:06d}"] = np.asarray(peta)
            arrays[f"accuracy_{i:06d}"] = np.column_stack((steps, accuracy))
        np.savez_compressed(
            args.output_dir / f"poincare_{args.backend}.npz",
            **arrays,
            backend=args.backend,
            columns=np.array(["t", "s", "theta", "zeta", "vpar"]),
            input_file=str(args.input.resolve()),
            ns_poinc=args.ns_poinc,
            nmaps=args.nmaps,
            resolution=args.resolution,
            tol=args.tol,
            tmax=args.tmax,
            dt_save=args.dt_save,
            energy=FUSION_ALPHA_PARTICLE_ENERGY,
            mass=ALPHA_PARTICLE_MASS,
            charge=ALPHA_PARTICLE_CHARGE,
            pitch=0.0,
            sign_vpar=1.0,
        )
        poinc.plot_poincare(
            filename=str(args.output_dir / f"poincare_{args.backend}.png")
        )
        proc0_print("Return counts: ", [len(path) - 1 for path in poinc.s_all])


if __name__ == "__main__":
    main()
