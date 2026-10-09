"""Compare forget_exact_path=True with a separately built master checkout.

Run on one GPU node with --baseline /path/to/master --output endpoints.json.
Both implementations stay loaded in separate processes. Warm-up, tabulation,
and garbage collection are excluded; alternating call order reduces timing
bias. The full public call includes transfers and host result assembly.
"""

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
PREFIX = "FIRM3D_ENDPOINTS "
DEFAULT_CASES = [
    "1000:1e-6",
    "10000:1e-6",
    "100000:1e-6",
    "1000:1e-4",
    "10000:1e-4",
    "100000:1e-4",
    "1000:1e-3",
    "10000:1e-3",
    "1000:1e-2",
    "100000:1e-3",
    "10000:1e-2",
]


def respond(value):
    print(PREFIX + json.dumps(value), flush=True)


def worker(args):
    import numpy as np
    import firm3dpp

    from firm3d.catapult.field import CatapultBoozerField
    from firm3d.catapult import tracing
    from firm3d.field.boozermagneticfield import BoozerRadialInterpolant
    from firm3d.util.constants import (
        ALPHA_PARTICLE_CHARGE as CHARGE,
        ALPHA_PARTICLE_MASS as MASS,
        FUSION_ALPHA_PARTICLE_ENERGY as ENERGY,
    )

    native = getattr(firm3dpp, "boozer_gpu_tracing", None)
    if native is None:
        raise RuntimeError("a CUDA-enabled firm3dpp is required")
    if ("save_times" in native.__doc__) != (args.worker == "dense"):
        raise RuntimeError("native build does not match the requested checkout")
    equilibrium = BoozerRadialInterpolant(
        str(ROOT / "examples/inputs/boozmn_aten_rescaled_low_res.nc"),
        3,
        enforce_vacuum=True,
    )
    field = CatapultBoozerField(equilibrium, 15, 15, 15, precision=args.precision)
    native_elapsed = []

    def timed_native(**kwargs):
        start = time.perf_counter()
        result = native(**kwargs)
        native_elapsed.append(time.perf_counter() - start)
        return result

    firm3dpp.boozer_gpu_tracing = timed_native
    source = Path(tracing.__file__)
    respond(
        {
            "extension": firm3dpp.__file__,
            "extension_sha256": hashlib.sha256(
                Path(firm3dpp.__file__).read_bytes()
            ).hexdigest(),
            "tracing_source": str(source),
            "tracing_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "numpy": np.__version__,
        }
    )
    stz = vpar = None
    current_particles = None
    for line in sys.stdin:
        request = json.loads(line)
        if request.get("quit"):
            return
        nparticles = request["particles"]
        if current_particles != nparticles:
            rng = np.random.default_rng(2170)
            stz = np.column_stack(
                (
                    rng.uniform(0.2, 0.7, nparticles),
                    rng.uniform(0, 2 * np.pi, nparticles),
                    rng.uniform(0, 2 * np.pi, nparticles),
                )
            )
            vpar = rng.uniform(-0.8, 0.8, nparticles) * np.sqrt(2 * ENERGY / MASS)
            current_particles = nparticles
        gc.collect()
        native_elapsed.clear()
        start = time.perf_counter()
        trajectories, hits = tracing.trace_particles_boozer_gpu(
            field,
            stz,
            vpar,
            tmax=request["tmax"],
            tol=args.tol,
            mass=MASS,
            charge=CHARGE,
            Ekin=ENERGY,
            forget_exact_path=True,
        )
        seconds = time.perf_counter() - start
        respond(
            {
                "seconds": seconds,
                "native_seconds": sum(native_elapsed),
                "launches": len(native_elapsed),
                "rows": sum(len(path) for path in trajectories),
                "losses": sum(bool(len(hit)) for hit in hits),
            }
        )
        del trajectories, hits


def read_response(process):
    while line := process.stdout.readline():
        if line.startswith(PREFIX):
            return json.loads(line[len(PREFIX) :])
    raise RuntimeError(f"benchmark worker exited with code {process.poll()}")


def request(process, value):
    process.stdin.write(json.dumps(value) + "\n")
    process.stdin.flush()
    return read_response(process)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output", type=Path, default=Path("gpu-endpoints.json"))
    parser.add_argument("--cases", nargs="+", default=DEFAULT_CASES)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--tol", type=float, default=1e-8)
    parser.add_argument("--worker", choices=["master", "dense"], help=argparse.SUPPRESS)
    parser.add_argument(
        "--precision",
        choices=["double", "single"],
        default="double",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return
    if args.baseline is None:
        parser.error("--baseline must name the separately built master checkout")
    if args.repeats < 1 or args.warmups < 1:
        parser.error("repeats and warmups must be positive")
    cases = []
    for case in args.cases:
        particles, tmax = case.split(":")
        cases.append({"particles": int(particles), "tmax": float(tmax)})
        if int(particles) < 1 or float(tmax) <= 0:
            parser.error("case particles and tmax must be positive")
    data = {
        "host": platform.node(),
        "config": {
            key: value
            for key, value in vars(args).items()
            if key not in ("worker", "precision")
        }
        | {
            "baseline": str(args.baseline),
            "output": str(args.output),
            "precisions": ["double", "single"],
        },
        "implementations": {},
        "results": [],
    }
    for precision in ["double", "single"]:
        processes = {}
        try:
            for name, checkout in [("master", args.baseline), ("dense", ROOT)]:
                env = os.environ | {"PYTHONPATH": str(checkout / "src")}
                processes[name] = subprocess.Popen(
                    [
                        sys.executable,
                        "-u",
                        str(Path(__file__).resolve()),
                        "--worker",
                        name,
                        "--precision",
                        precision,
                        "--tol",
                        str(args.tol),
                    ],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    text=True,
                    env=env,
                )
                data["implementations"][f"{name}-{precision}"] = read_response(
                    processes[name]
                )
            for case in cases:
                measurements = {name: [] for name in processes}
                for repeat in range(args.warmups + args.repeats):
                    order = (
                        ["master", "dense"] if repeat % 2 == 0 else ["dense", "master"]
                    )
                    for name in order:
                        result = request(processes[name], case)
                        if (
                            result["launches"] != 1
                            or result["rows"] != 2 * case["particles"]
                        ):
                            raise RuntimeError(
                                "endpoint-only call returned unexpected output"
                            )
                        if repeat >= args.warmups:
                            measurements[name].append(result)
                record = case | {"precision": precision, "measurements": measurements}
                for name in processes:
                    record[f"{name}_seconds"] = statistics.median(
                        item["seconds"] for item in measurements[name]
                    )
                    record[f"{name}_native_seconds"] = statistics.median(
                        item["native_seconds"] for item in measurements[name]
                    )
                record["dense_over_master"] = (
                    record["dense_seconds"] / record["master_seconds"]
                )
                data["results"].append(record)
                print(
                    json.dumps(
                        {k: v for k, v in record.items() if k != "measurements"}
                    ),
                    flush=True,
                )
                args.output.write_text(json.dumps(data, indent=2) + "\n")
        finally:
            for process in processes.values():
                if process.poll() is None:
                    process.stdin.write('{"quit": true}\n')
                    process.stdin.flush()
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        process.terminate()
                        process.wait(timeout=30)


if __name__ == "__main__":
    main()
