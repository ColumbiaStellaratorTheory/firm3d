"""Measure dense trajectory saving against endpoint-only GPU tracing.

Run from the repository root on a GPU node, for example:
    python benchmarks/benchmark_gpu_saving.py --output gpu-saving.json

Field tabulation and warm-up are excluded. Timings include device allocation,
tracing, output transfer, and assembly of the public CPU-format trajectories.
The output JSON is accompanied by an NPZ of final states for comparisons.
To measure the old method, put its checkout's src directory in PYTHONPATH
and pass --method existing; this option does not switch the implementation.
"""

import argparse
import gc
import json
from pathlib import Path
import platform
import statistics
import sys
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
    parser.add_argument(
        "--method",
        choices=["dense", "existing"],
        default="dense",
        help="Label the loaded implementation; existing allows endpoint overshoot.",
    )
    args = parser.parse_args()
    if min(args.particles) <= 0 or min(args.samples) < 0 or args.repeats < 1:
        parser.error("particles/repeats must be positive and samples nonnegative")
    native = getattr(firm3dpp, "boozer_gpu_tracing", None)
    if native is None:
        parser.error("a CUDA-enabled firm3dpp is required")
    tracing_module = sys.modules[trace_particles_boozer_gpu.__module__]
    expected_dense = args.method == "dense"
    if ("save_times" in native.__doc__) != expected_dense or hasattr(
        tracing_module, "_save_times"
    ) != expected_dense:
        parser.error("--method must match the native and Python code in PYTHONPATH")
    filename = (
        Path(__file__).resolve().parents[1]
        / "examples/inputs/boozmn_aten_rescaled_low_res.nc"
    )
    equilibrium = BoozerRadialInterpolant(str(filename), 3, enforce_vacuum=True)
    field = CatapultBoozerField(equilibrium, 15, 15, 15, precision=args.precision)
    speed = np.sqrt(2 * ENERGY / MASS)
    results = []
    terminal_states = {}
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
                terminal_states[f"{nparticles}_{nsamples}"] = terminal
                if reference is None:
                    reference = terminal
                elif args.method == "dense":
                    np.testing.assert_array_equal(terminal, reference)
                saved_rows = sum(len(traj) for traj in warmup)
                returned_bytes = sum(traj.nbytes for traj in warmup)
                survivors = terminal[:, 0] >= args.tmax * (1 - 1e-6)
                survivor_rows = np.array(
                    [len(traj) for traj, keep in zip(warmup, survivors) if keep]
                )
                overshoot = terminal[survivors, 0] - args.tmax
                sampling = {}
                if nsamples and len(survivor_rows):
                    # Distance from the requested grid, using survivors so an
                    # off-grid loss endpoint is not counted as a sampling error.
                    times = np.concatenate(
                        [traj[1:, 0] for traj, keep in zip(warmup, survivors) if keep]
                    )
                    residual = np.abs(
                        times / kwargs["dt_save"] - np.rint(times / kwargs["dt_save"])
                    )
                    sampling = {
                        "off_grid_fraction": float(
                            np.mean(residual > 8 * np.finfo(field.dtype).eps * nsamples)
                        ),
                        "grid_distance_seconds_p50_p90_max": (
                            np.percentile(residual, [50, 90, 100]) * kwargs["dt_save"]
                        ).tolist(),
                        "survivor_saved_samples_p10_p50_p90": (
                            np.percentile(survivor_rows - 1, [10, 50, 90])
                        ).tolist(),
                        "survivors_missing_samples": int(
                            np.sum(survivor_rows < nsamples + 1)
                        ),
                    }
                    del times, residual
                del warmup
                elapsed = []
                native_elapsed = []
                launch_counts = []
                for _ in range(args.repeats):
                    gc.collect()
                    native_times.clear()
                    start = time.perf_counter()
                    trajectories, _ = trace_particles_boozer_gpu(
                        field, stz, vpar, **kwargs
                    )
                    elapsed.append(time.perf_counter() - start)
                    native_elapsed.append(sum(native_times))
                    launch_counts.append(len(native_times))
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
                    "native_seconds": statistics.median(native_elapsed),
                    "launches_per_call": launch_counts,
                    "slowdown": (seconds / baseline if baseline else None),
                    "saved_rows": saved_rows,
                    "gpu_output_bytes": 7
                    * nparticles
                    * (max(nsamples, 1) if args.method == "dense" else 1)
                    * field.dtype.itemsize,
                    "returned_bytes": returned_bytes,
                    "survivors": int(np.sum(survivors)),
                    "terminal_overshoot_seconds_p50_p90_max": (
                        np.percentile(overshoot, [50, 90, 100]).tolist()
                        if len(overshoot)
                        else []
                    ),
                    **sampling,
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
                            "tracing_source": sys.modules[
                                trace_particles_boozer_gpu.__module__
                            ].__file__,
                            "results": results,
                        },
                        indent=2,
                    )
                    + "\n"
                )
                np.savez(args.output.with_suffix(".npz"), **terminal_states)
    finally:
        firm3dpp.boozer_gpu_tracing = native


if __name__ == "__main__":
    main()
