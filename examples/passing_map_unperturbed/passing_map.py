"""Kinetic passing Poincare map with the original CPU example parameters."""

import argparse
from pathlib import Path
import time

from firm3d.field.boozermagneticfield import InterpolatedBoozerField
from firm3d.trajectory_helpers import PassingPoincare
from firm3d.util.constants import (
    ALPHA_PARTICLE_CHARGE,
    ALPHA_PARTICLE_MASS,
    FUSION_ALPHA_PARTICLE_ENERGY,
)
from firm3d.util.functions import in_github_actions, proc0_print, setup_logging
from firm3d.util.mpi import comm_size, comm_world, verbose


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["cpu", "catapult"], default="cpu")
    args = parser.parse_args()

    # Same equilibrium, launch grid, and solver settings for both backends.
    boozmn_filename = (
        Path(__file__).resolve().parent.parent / "inputs/boozmn_aten_rescaled.nc"
    )
    resolution = 10 if in_github_actions else 48
    ns_poinc = 5 if in_github_actions else 120
    Nmaps = 5 if in_github_actions else 1000
    tol = 1e-4 if in_github_actions else 1e-8

    setup_logging(f"stdout_passing_map_{args.backend}_{resolution}_{comm_size}.txt")
    started = time.perf_counter()
    field = InterpolatedBoozerField.from_booz_xform(
        str(boozmn_filename),
        degree=3,
        ns=resolution,
        ntheta=resolution,
        nzeta=resolution,
        comm=comm_world,
    )
    if args.backend == "catapult":
        from firm3d.catapult.field import CatapultBoozerField

        field = CatapultBoozerField(field, resolution, resolution, resolution)

    poinc = PassingPoincare(
        field,
        lam=0.0,
        sign_vpar=1.0,
        mass=ALPHA_PARTICLE_MASS,
        charge=ALPHA_PARTICLE_CHARGE,
        Ekin=FUSION_ALPHA_PARTICLE_ENERGY,
        ns_poinc=ns_poinc,
        ntheta_poinc=1,
        Nmaps=Nmaps,
        comm=comm_world,
        solver_options={"reltol": tol, "abstol": tol},
    )
    if verbose and not in_github_actions:
        poinc.plot_poincare(filename=f"poincare_{args.backend}.png")
    proc0_print("poincare time: ", time.perf_counter() - started)


if __name__ == "__main__":
    main()
