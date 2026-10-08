"""Measure dense trajectory saving against endpoint-only GPU tracing.

Run from the repository root on a GPU node, for example:
    python examples/benchmark_gpu_saving.py --output gpu-saving.json

Field tabulation and warm-up are excluded. Timings include device allocation,
tracing, output transfer, and assembly of the public CPU-format trajectories.
"""

import argparse
import gc
import json
from pathlib import Path
import platform
import statistics
import time

import numpy as np
import firm3dpp

from firm3d.catapult.field import CatapultBoozerField
from firm3d.catapult.tracing import trace_particles_boozer_gpu
from firm3d.field.boozermagneticfield import BoozerRadialInterpolant
from firm3d.util.constants import (
    ALPHA_PARTICLE_CHARGE as CHARGE,
    ALPHA_PARTICLE_MASS as MASS,
    FUSION_ALPHA_PARTICLE_ENERGY as ENERGY,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--particles", type=int, nargs="+", default=[1000, 10000])
    parser.add_argument("--samples", type=int, nargs="+", default=[0, 10, 100, 1000])
    parser.add_argument("--precision", choices=["single", "double"], default="double")
    parser.add_argument("--tmax", type=float, default=1e-4)
    parser.add_argument("--tol", type=float, default=1e-8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("gpu-saving.json"))
    args = parser.parse_args()
    if min(args.particles) <= 0 or min(args.samples) < 0 or args.repeats < 1:
        parser.error("particles/repeats must be positive and samples nonnegative")
    filename = Path(__file__).parent / "inputs/boozmn_aten_rescaled_low_res.nc"
    equilibrium = BoozerRadialInterpolant(str(filename), 3, enforce_vacuum=True)
    field = CatapultBoozerField(equilibrium, 15, 15, 15, precision=args.precision)
    speed = np.sqrt(2 * ENERGY / MASS)
    results = []
    native = firm3dpp.boozer_gpu_tracing
    native_times = []

    def timed_native(**kwargs):
        start = time.perf_counter()
        result = native(**kwargs)
        native_times.append(time.perf_counter() - start)
        return result

    firm3dpp.boozer_gpu_tracing = timed_native
    try:
        for nparticles in args.particles:
            rng = np.random.default_rng(2170)
            stz = np.column_stack(
                (
                    rng.uniform(0.2, 0.7, nparticles),
                    rng.uniform(0, 2 * np.pi, nparticles),
                    rng.uniform(0, 2 * np.pi, nparticles),
                )
            )
            vpar = rng.uniform(-0.8, 0.8, nparticles) * speed
            reference = None
            baseline = None
            for nsamples in args.samples:
                kwargs = {
                    "tmax": args.tmax,
                    "tol": args.tol,
                    "mass": MASS,
                    "charge": CHARGE,
                    "Ekin": ENERGY,
                    "forget_exact_path": (nsamples == 0),
                }
                if nsamples:
                    kwargs["dt_save"] = args.tmax / nsamples
                # First call initializes CUDA and warms this output size.
                warmup, _ = trace_particles_boozer_gpu(field, stz, vpar, **kwargs)
                terminal = np.array([traj[-1] for traj in warmup])
                if reference is None:
                    reference = terminal
                else:
                    np.testing.assert_array_equal(terminal, reference)
                saved_rows = sum(len(traj) for traj in warmup)
                returned_bytes = sum(traj.nbytes for traj in warmup)
                del warmup
                elapsed = []
                native_times.clear()
                for _ in range(args.repeats):
                    gc.collect()
                    start = time.perf_counter()
                    trajectories, _ = trace_particles_boozer_gpu(
                        field, stz, vpar, **kwargs
                    )
                    elapsed.append(time.perf_counter() - start)
                    del trajectories
                seconds = statistics.median(elapsed)
                if nsamples == 0:
                    baseline = seconds
                record = {
                    "particles": nparticles,
                    "samples": nsamples,
                    "dt_save": (args.tmax / nsamples if nsamples else None),
                    "seconds": seconds,
                    "repeat_seconds": elapsed,
                    "native_seconds": statistics.median(native_times),
                    "slowdown": (seconds / baseline if baseline else None),
                    "saved_rows": saved_rows,
                    "gpu_output_bytes": 7
                    * nparticles
                    * max(nsamples, 1)
                    * field.dtype.itemsize,
                    "returned_bytes": returned_bytes,
                }
                results.append(record)
                print(json.dumps(record), flush=True)
                args.output.write_text(
                    json.dumps(
                        {
                            "config": vars(args) | {"output": str(args.output)},
                            "host": platform.node(),
                            "numpy": np.__version__,
                            "extension": firm3dpp.__file__,
                            "results": results,
                        },
                        indent=2,
                    )
                    + "\n"
                )
    finally:
        firm3dpp.boozer_gpu_tracing = native


if __name__ == "__main__":
    main()
