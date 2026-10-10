"""Compare GPU sections with the existing unperturbed CPU passing-map example."""

import argparse
import json
from pathlib import Path
import time

import numpy as np

from plot_poincare import section_crossings


HERE = Path(__file__).resolve().parent
PERIOD = 2 * np.pi


class NoKTableSource:
    """Represent a CPU no-K field in the GPU's full-field table with K=0.

    The unperturbed CPU example retains I(s) and G(s), but omits K. Setting
    K and its angular derivatives to zero in the full GPU equations gives
    exactly those no-K equations. All other quantities come from the CPU
    interpolant, sampled at its own cubic interpolation nodes. This adapter
    is used only to construct the GPU table, not for CPU time integration.
    """

    field_type = ""

    def __init__(self, field):
        if field.field_type != "nok":
            raise ValueError("expected the CPU example's no-K field")
        self.field = field

    def __getattr__(self, name):
        return getattr(self.field, name)

    def set_points(self, points):
        # Preserve the CPU interpolant's upper toroidal endpoint. Evaluating
        # exactly at one field period would wrap to its lower endpoint; spline
        # derivatives at those endpoints need not be numerically identical.
        points = points.copy()
        upper = PERIOD / self.field.nfp
        points[points[:, 2] == upper, 2] = np.nextafter(upper, 0.0)
        self.field.set_points(points)

    def K(self):
        return np.zeros((len(self.field.get_points_ref()), 1))

    def K_derivs(self):
        return np.zeros((len(self.field.get_points_ref()), 2))


def map_sections(poincare):
    """Convert CPU map data, excluding launches, to t,s,theta,zeta,vpar."""
    return [
        np.column_stack((np.cumsum(t), s, theta, np.zeros(len(s)), vpar))[1:]
        for t, s, theta, vpar in zip(
            poincare.t_all,
            poincare.s_all,
            poincare.thetas_all,
            poincare.vpars_all,
        )
    ]


def sampled_sections(paths, nmaps):
    """Use the CPU example's s=0.99 boundary and requested return count."""
    sections = []
    for path in paths:
        outside = np.flatnonzero(path[:, 1] >= 0.99)
        if len(outside):
            path = path[: outside[0]]
        sections.append(section_crossings(path)[:nmaps])
    return sections


def differences(actual, reference):
    """Pair returns by particle and return number; report unmatched counts."""
    errors = []
    for a, b in zip(actual, reference):
        n = min(len(a), len(b))
        delta = a[:n] - b[:n]
        delta[:, 2] = (delta[:, 2] + np.pi) % PERIOD - np.pi
        errors.append(delta)
    errors = np.concatenate(errors)
    if not len(errors):
        raise ValueError("no corresponding returns to compare")
    return {
        "paired_returns": len(errors),
        "actual_returns": sum(map(len, actual)),
        "reference_returns": sum(map(len, reference)),
        "particles_with_different_counts": sum(
            len(a) != len(b) for a, b in zip(actual, reference)
        ),
        "max_abs_time_s": float(np.max(np.abs(errors[:, 0]))),
        "max_abs_s": float(np.max(np.abs(errors[:, 1]))),
        "rms_s": float(np.sqrt(np.mean(errors[:, 1] ** 2))),
        "max_abs_theta_rad": float(np.max(np.abs(errors[:, 2]))),
        "rms_theta_rad": float(np.sqrt(np.mean(errors[:, 2] ** 2))),
        "max_abs_vpar_m_per_s": float(np.max(np.abs(errors[:, 4]))),
    }


def save_sections(filename, sections, particle_ids=None):
    if particle_ids is None:
        particle_ids = range(len(sections))
    rows = np.concatenate(
        [
            np.column_stack((np.full(len(p), i), p))
            for i, p in zip(particle_ids, sections)
        ]
    )
    np.savetxt(
        filename,
        rows,
        delimiter=",",
        header="particle,t_s,s,theta_rad,zeta_rad,vpar_m_per_s",
        comments="",
        fmt=["%d", *(["%.17g"] * 5)],
    )


def comparison_plot(filename, sections, initial_s):
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    fig, axes = plt.subplots(1, len(sections), figsize=(15, 4.5), layout="constrained")
    for ax, (label, paths) in zip(axes, sections.items()):
        points = np.concatenate(paths)
        colors = np.concatenate(
            [np.full(len(path), radius) for path, radius in zip(paths, initial_s)]
        )
        scatter = ax.scatter(
            points[:, 2] % PERIOD,
            points[:, 1],
            c=colors,
            cmap="viridis",
            norm=Normalize(0, 1),
            s=0.5,
            rasterized=True,
        )
        ax.set(title=label, xlabel=r"$\theta$ (rad)", xlim=(0, PERIOD), ylim=(0, 1))
    axes[0].set_ylabel(r"Normalized toroidal flux $s$")
    fig.colorbar(scatter, ax=axes, label=r"Initial $s$")
    fig.suptitle(r"ATEN co-passing 3.5 MeV alpha particles, $\lambda=0$, $\zeta=0$")
    fig.savefig(filename, dpi=200)
    plt.close(fig)


def error_plot(filename, comparisons, reference):
    """Show growth of orbit differences instead of only their global maximum."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True, layout="constrained")
    for label, paths in comparisons.items():
        longest = max(min(len(a), len(b)) for a, b in zip(paths, reference))
        sums = np.zeros((longest, 2))
        counts = np.zeros(longest)
        for actual, ref in zip(paths, reference):
            n = min(len(actual), len(ref))
            delta = actual[:n, 1:3] - ref[:n, 1:3]
            delta[:, 1] = (delta[:, 1] + np.pi) % PERIOD - np.pi
            sums[:n] += delta**2
            counts[:n] += 1
        rms = np.sqrt(sums / counts[:, None])
        for column, ax in enumerate(axes):
            ax.semilogy(np.arange(1, longest + 1), rms[:, column], label=label)
    axes[0].set_ylabel(r"RMS $\Delta s$")
    axes[1].set(xlabel="Return number", ylabel=r"RMS circular $\Delta\theta$ (rad)")
    axes[0].legend(fontsize="small")
    fig.savefig(filename, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    # Production defaults from examples/passing_map_unperturbed/passing_map.py.
    parser.add_argument("--ns-poinc", type=int, default=120)
    parser.add_argument(
        "--particle-ids",
        type=int,
        nargs="+",
        help="Optional zero-based IDs to select from the CPU launch grid",
    )
    parser.add_argument("--nmaps", type=int, default=1000)
    parser.add_argument("--resolution", type=int, default=48)
    parser.add_argument("--tol", type=float, default=1e-8)
    parser.add_argument("--dt-save", type=float, default=1e-7)
    parser.add_argument(
        "--input", type=Path, default=HERE.parent / "inputs/boozmn_aten_rescaled.nc"
    )
    parser.add_argument("--output-dir", type=Path, default=HERE / "output/comparison")
    args = parser.parse_args()
    if args.ns_poinc < 1 or args.nmaps < 1 or args.resolution < 4:
        parser.error("ns-poinc/nmaps must be positive; resolution must be at least 4")
    if any(not np.isfinite(v) or v <= 0 for v in (args.tol, args.dt_save)):
        parser.error("tol and dt-save must be finite and positive")
    particle_ids = args.particle_ids or list(range(args.ns_poinc))
    if len(set(particle_ids)) != len(particle_ids) or any(
        i < 0 or i >= args.ns_poinc for i in particle_ids
    ):
        parser.error("particle-ids must be distinct IDs in the CPU launch grid")

    import matplotlib.pyplot as plt
    import firm3dpp

    from firm3d.catapult.field import CatapultBoozerField
    from firm3d.catapult.tracing import trace_particles_boozer_gpu
    from firm3d.field.boozermagneticfield import InterpolatedBoozerField
    from firm3d.field.tracing import (
        MaxToroidalFluxStoppingCriterion,
        trace_particles_boozer,
    )
    from firm3d.trajectory_helpers import PassingPoincare
    from firm3d.util.constants import (
        ALPHA_PARTICLE_CHARGE as CHARGE,
        ALPHA_PARTICLE_MASS as MASS,
        FUSION_ALPHA_PARTICLE_ENERGY as ENERGY,
    )

    plt.switch_backend("Agg")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    field = InterpolatedBoozerField.from_booz_xform(
        str(args.input.resolve()),
        degree=3,
        ns=args.resolution,
        ntheta=args.resolution,
        nzeta=args.resolution,
    )
    setup_cpu = time.perf_counter() - started
    print(f"CPU field constructed in {setup_cpu:.3f} s", flush=True)
    started = time.perf_counter()
    gpu_field = CatapultBoozerField(
        NoKTableSource(field),
        args.resolution,
        args.resolution,
        args.resolution,
        precision="double",
    )
    setup_gpu = time.perf_counter() - started
    print(f"GPU table constructed in {setup_gpu:.3f} s", flush=True)

    # Check the entire non-vacuum table, including stellarator reflections.
    rng = np.random.default_rng(42)
    probe = rng.uniform([0.01, -PERIOD, -PERIOD], [0.98, PERIOD, PERIOD], (1000, 3))
    field.set_points(probe)
    expected = np.hstack(
        (
            field.modB(),
            field.modB_derivs(),
            field.G(),
            field.dGds(),
            field.I(),
            field.dIds(),
            field.iota(),
            np.zeros((len(probe), 3)),
        )
    )
    actual = firm3dpp.test_gpu_interpolation(
        gpu_field.quad_info,
        gpu_field.srange,
        gpu_field.trange,
        gpu_field.zrange,
        probe.copy(),
        "boozer",
        len(probe),
    ).reshape(expected.shape)
    field_error = float(np.max(np.abs(actual - expected) / (1 + np.abs(expected))))
    if field_error > 1e-10:
        row, column = np.unravel_index(
            np.argmax(np.abs(actual - expected) / (1 + np.abs(expected))), actual.shape
        )
        raise RuntimeError(
            f"CPU/GPU field tables differ: {field_error}, column {column}, "
            f"point {probe[row]}, CPU {expected[row]}, GPU {actual[row]}"
        )
    print(f"Field tables agree: max scaled error {field_error:.3g}", flush=True)

    # This is the existing CPU example, including its momentum/WBA diagnostics.
    started = time.perf_counter()
    cpu_map = PassingPoincare(
        field,
        0.0,
        1.0,
        MASS,
        CHARGE,
        ENERGY,
        ns_poinc=args.ns_poinc,
        ntheta_poinc=1,
        s_init=np.linspace(0, 1, args.ns_poinc + 1, endpoint=False)[1:][particle_ids],
        thetas_init=np.zeros(len(particle_ids)),
        Nmaps=args.nmaps,
        helicity_N=field.nfp,
        helicity_M=1,
        solver_options={"reltol": args.tol, "abstol": args.tol},
        chaos_detection=True,
    )
    cpu_map_time = time.perf_counter() - started
    reference = map_sections(cpu_map)
    print(
        f"CPU example: {sum(map(len, reference))} returns in {cpu_map_time:.3f} s",
        flush=True,
    )
    save_sections(args.output_dir / "cpu_map.csv", reference, particle_ids)
    initial = np.column_stack(
        (cpu_map.s_init, cpu_map.thetas_init, np.zeros(len(particle_ids)))
    )
    vpar = np.asarray(cpu_map.vpars_init)
    first_transits = [p[0, 0] for p in reference if len(p)]
    if not first_transits:
        raise RuntimeError("CPU map found no returns")
    # Add two transits so tiny solver differences cannot drop the final return.
    # Trim sections to Nmaps; the horizon is identical for continuous CPU/GPU.
    horizon = max(p[-1, 0] for p in reference if len(p)) + 2 * max(first_transits)

    trace_options = {
        "tmax": horizon,
        "tol": args.tol,
        "Ekin": ENERGY,
        "mass": MASS,
        "charge": CHARGE,
    }
    trace_particles_boozer_gpu(
        gpu_field,
        initial[:1],
        vpar[:1],
        **(trace_options | {"tmax": 1e-6}),
        forget_exact_path=True,
    )
    gpu_sections = []
    gpu_times = []
    gpu_post_times = []
    max_angle_steps = []
    for label, cadence in [("coarse", args.dt_save), ("fine", args.dt_save / 2)]:
        started = time.perf_counter()
        paths, hits = trace_particles_boozer_gpu(
            gpu_field,
            initial,
            vpar,
            **trace_options,
            dt_save=cadence,
            forget_exact_path=False,
        )
        gpu_times.append(time.perf_counter() - started)
        started = time.perf_counter()
        gpu_sections.append(sampled_sections(paths, args.nmaps))
        gpu_post_times.append(time.perf_counter() - started)
        max_angle_steps.append(
            float(
                max(
                    np.max(np.abs(np.diff(np.unwrap(p[:, 2:4], axis=0), axis=0)))
                    for p in paths
                    if len(p) > 1
                )
            )
        )
        if max_angle_steps[-1] >= np.pi:
            raise RuntimeError("angles advance too far between samples; reduce dt-save")
        # Save the fine history for trajectory plots and arbitrary offline sections.
        if label == "fine":
            np.savez_compressed(
                args.output_dir / "trajectories.npz",
                **{f"particle_{i:06d}": p for i, p in zip(particle_ids, paths)},
                **{f"hits_{i:06d}": h for i, h in zip(particle_ids, hits)},
                particle_ids=np.asarray(particle_ids),
                columns=np.array(["t", "s", "theta", "zeta", "vpar"]),
                coordinate_system="boozer",
                initial_positions=initial,
                initial_parallel_speeds=vpar,
                tmax=horizon,
                dt_save=cadence,
                tol=args.tol,
                resolution=args.resolution,
                nfp=field.nfp,
                input_file=str(args.input.resolve()),
                mass=MASS,
                charge=CHARGE,
                energy=ENERGY,
            )
        print(
            f"GPU {label}: {sum(map(len, gpu_sections[-1]))} returns; "
            f"trace {gpu_times[-1]:.3f} s",
            flush=True,
        )
        del paths

    # Continuous CPU tracing isolates differences due to restarting at each
    # return in PassingPoincare. Also compare CPU sampled
    # sections with CPU event roots to measure the postprocessing error alone.
    print(
        "Running continuous CPU control with event roots and fine saved output",
        flush=True,
    )
    started = time.perf_counter()
    paths, roots = trace_particles_boozer(
        field,
        initial,
        vpar,
        **trace_options,
        dt_save=args.dt_save / 2,
        phases=[0.0],
        n_zetas=[1.0],
        m_thetas=[0.0],
        omegas=[0.0],
        stopping_criteria=[MaxToroidalFluxStoppingCriterion(0.99)],
        forget_exact_path=False,
    )
    cpu_continuous_time = time.perf_counter() - started
    continuous = [
        r[r[:, 1] == 0][: args.nmaps][:, [0, 2, 3, 4, 5]]
        if len(r)
        else np.empty((0, 5))
        for r in roots
    ]
    cpu_sampled = sampled_sections(paths, args.nmaps)
    del paths
    sections = {
        "CPU example (event roots)": reference,
        "GPU saved paths": gpu_sections[1],
        "CPU continuous (event roots)": continuous,
    }
    comparison_plot(args.output_dir / "cpu_gpu_poincare.png", sections, initial[:, 0])
    error_plot(
        args.output_dir / "errors_vs_return.png",
        {
            "GPU coarse": gpu_sections[0],
            "GPU fine": gpu_sections[1],
            "CPU continuous": continuous,
        },
        reference,
    )
    for name, data in [
        ("gpu_coarse", gpu_sections[0]),
        ("gpu_fine", gpu_sections[1]),
        ("cpu_continuous", continuous),
    ]:
        save_sections(args.output_dir / f"{name}.csv", data, particle_ids)
    results = {
        "parameters": {
            "input": str(args.input.resolve()),
            "ns_poinc": args.ns_poinc,
            "particle_ids": particle_ids,
            "nmaps": args.nmaps,
            "resolution": args.resolution,
            "tol": args.tol,
            "dt_save_s": args.dt_save,
            "horizon_s": horizon,
            "lambda": 0.0,
            "sign_vpar": 1,
            "cpu_stop_s": 0.99,
            "gpu_stop_s": 1.0,
        },
        "field_table_max_scaled_error": field_error,
        "max_saved_angle_step_rad": max_angle_steps,
        "timings_s": {
            "cpu_field_setup": setup_cpu,
            "gpu_table_setup": setup_gpu,
            "cpu_map_including_momentum_and_WBA": cpu_map_time,
            "cpu_continuous_trace_with_history_and_event_roots": cpu_continuous_time,
            "gpu_trace_coarse_fine": gpu_times,
            "gpu_section_postprocessing_coarse_fine": gpu_post_times,
        },
        "gpu_coarse_vs_cpu_map": differences(gpu_sections[0], reference),
        "gpu_fine_vs_cpu_map": differences(gpu_sections[1], reference),
        "gpu_fine_vs_cpu_continuous": differences(gpu_sections[1], continuous),
        "gpu_coarse_vs_gpu_fine": differences(gpu_sections[0], gpu_sections[1]),
        "cpu_sampled_fine_vs_cpu_event_roots": differences(cpu_sampled, continuous),
        "cpu_continuous_vs_cpu_map": differences(continuous, reference),
    }
    (args.output_dir / "comparison.json").write_text(
        json.dumps(results, indent=2) + "\n"
    )
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
