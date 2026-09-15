__all__ = [
    "trace_particles_boozer_gpu",
    "trace_particles_boozer_perturbed_gpu",
    "trace_particles_cartesian_gpu",
]
import numpy as np

import firm3dpp
from firm3d.catapult.utils import (
    boozer_interpolant,
    boozer_saw_interpolant,
    cartesian_interpolant,
)
from firm3d.field.boozermagneticfield import ShearAlfvenWavesSuperposition


def trace_particles_boozer_gpu(
    field,
    stz_inits,
    parallel_speeds,
    tmax,
    mass,
    charge,
    vtotal,
    tol,
    ns,
    ntheta,
    nzeta,
    dt=None,
):
    """
    Trace particles in Boozer coordinates using CATAPULT
    field: a magnetic field object representing the field in Boozer coordinates
    stz_inits: initial conditions for particles in (s, theta, zeta) coordinates
    parallel_speeds: initial parallel speeds of the particles
    tmax: maximum time to trace particles
    mass: mass of each particle
    charge: charge of each particle
    vtotal: initial total speed shared by all particles, in m/s
    tol: tolerance for the ODE solver
    dt: the initial time step size for the solver (optional)

    For perturbed fields, this preserves the monoenergetic calling convention
    by converting vtotal to Ekin. Use trace_particles_boozer_perturbed_gpu
    directly to supply initial energies in Joules, including an energy array.
    """
    nparticles = stz_inits.shape[0]

    if isinstance(field, ShearAlfvenWavesSuperposition):
        return trace_particles_boozer_perturbed_gpu(
            field,
            stz_inits,
            parallel_speeds,
            Ekin=0.5 * mass * vtotal**2,
            tmax=tmax,
            mass=mass,
            charge=charge,
            tol=tol,
            ns=ns,
            ntheta=ntheta,
            nzeta=nzeta,
            dt=dt,
        )
    else:
        if field.field_type not in ["vac", ""]:
            raise ValueError(
                f"Unsupported field type {field.field_type} for Boozer tracing, \
                     expected 'vac' or ''"
            )
        vacuum = field.field_type == "vac"  # true if vacuum, false if finite beta
        srange, trange, zrange, quad_info, maxJ = boozer_interpolant(
            field, field.nfp, ns, ntheta, nzeta, vacuum=vacuum
        )
        psi0 = field.psi0
        last_time = firm3dpp.boozer_gpu_tracing(
            quad_pts=quad_info,
            srange=srange,
            trange=trange,
            zrange=zrange,
            stz_init=stz_inits,
            m=mass,
            q=charge,
            vtotal=vtotal,
            vtang=parallel_speeds,
            tmax=tmax,
            tol=tol,
            dt_in=-np.ones(nparticles),
            psi0=psi0,
            nparticles=nparticles,
            vacuum=vacuum,
        )

    last_time = np.reshape(last_time, (nparticles, 6))
    return last_time


def trace_particles_boozer_perturbed_gpu(
    field,
    stz_inits,
    parallel_speeds,
    Ekin,
    tmax,
    mass,
    charge,
    tol,
    ns,
    ntheta,
    nzeta,
    dt=None,
):
    """Trace particles in a SAW field from their initial kinetic energies.

    field: ShearAlfvenWavesSuperposition with a vacuum or no-K background
    stz_inits: finite (N, 3) launch coordinates (s, theta, zeta), with N > 0
    parallel_speeds: finite (N,) initial parallel velocities in m/s
    Ekin: nonnegative initial kinetic energy in Joules, a scalar or (N,) array
    tmax: maximum tracing time in seconds
    mass: positive particle mass in kg, shared by the particles
    charge: particle charge in Coulombs, shared by the particles
    tol: tolerance for the ODE solver
    ns, ntheta, nzeta: numbers of interpolation cells in each direction
    dt: optional (N,) array of initial timesteps in seconds

    The wrapper computes mu = (2*Ekin/mass - v_parallel**2)/(2*|B0|)
    at the launch points and passes it to C++ as a fixed moment per unit mass,
    in m^2/(s^2 T), clipping negative values to zero.
    The maximum initial speed is passed as the numerical reference vtotal for
    timestep and tolerance scaling and must be positive.

    Returns an (N, 6) array with columns [time, s, theta, zeta, v_parallel, dt].
    """
    if not isinstance(field, ShearAlfvenWavesSuperposition):
        raise ValueError("field must be a ShearAlfvenWavesSuperposition")
    B0 = field.B0
    if B0.field_type not in ["vac", "nok"]:
        raise ValueError(f"Unsupported field type {B0.field_type} for SAW tracing")

    # C++ overwrites the coordinates; copy them to preserve the caller's input.
    points = np.array(stz_inits, dtype=np.float64, order="C", copy=True)
    vpar = np.ascontiguousarray(parallel_speeds, dtype=np.float64)
    energy = np.asarray(Ekin, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("stz_inits must have shape (N, 3)")
    nparticles = points.shape[0]
    if nparticles == 0 or vpar.shape != (nparticles,):
        raise ValueError("parallel_speeds must have shape (N,), with N > 0")
    if energy.ndim == 0:
        energy = np.full(nparticles, energy.item(), dtype=np.float64)
    elif energy.shape != (nparticles,):
        raise ValueError("Ekin must be a scalar or have shape (N,)")
    if not np.isfinite(energy).all() or np.any(energy < 0):
        raise ValueError("Ekin must be finite and nonnegative")

    speed2 = 2.0 * energy / mass
    vperp2 = speed2 - vpar**2
    if not np.isfinite(speed2).all() or not np.isfinite(vperp2).all():
        raise ValueError("initial squared speeds must be finite")
    vperp2 = np.maximum(vperp2, 0.0)

    B0.set_points(points)
    B_init = np.array(B0.modB()[:, 0], dtype=np.float64, copy=True)
    if (
        B_init.shape != (nparticles,)
        or not np.isfinite(B_init).all()
        or np.any(B_init <= 0)
    ):
        raise ValueError("initial background |B| must be finite and positive")
    mus = np.ascontiguousarray(vperp2 / (2.0 * B_init))
    if not np.isfinite(mus).all():
        raise ValueError("computed mus must be finite")
    speed2_max = np.max(speed2)
    if speed2_max <= 0:
        raise ValueError("max(speed2) must be positive to define v_reference")
    v_reference = float(np.sqrt(speed2_max))

    srange, trange, zrange, quad_info, maxJ = boozer_saw_interpolant(
        B0, B0.nfp, ns, ntheta, nzeta
    )
    saw_nharmonics = len(field)
    saw_omega = field.get_wave(0).omega
    saw_s = field.get_wave(0).phihat.get_s_basis()
    saw_srange = (saw_s[0], saw_s[-1], len(saw_s))
    saw_m = [field.get_wave(i).Phim for i in range(saw_nharmonics)]
    saw_n = [field.get_wave(i).Phin for i in range(saw_nharmonics)]
    saw_phases = np.ascontiguousarray(
        [field.get_wave(i).phase for i in range(saw_nharmonics)], dtype=np.float64
    )
    saw_phihats = np.ascontiguousarray(
        np.column_stack(
            [
                np.array([field.get_wave(i).phihat(s_val) for s_val in saw_s])
                for i in range(saw_nharmonics)
            ]
        )
    )
    if B0.field_type == "vac":
        trace = firm3dpp.boozer_saw_gpu_tracing
    else:
        trace = firm3dpp.boozer_saw_nok_gpu_tracing
    last_time = trace(
        quad_pts=quad_info,
        srange=srange,
        trange=trange,
        zrange=zrange,
        saw_omega=saw_omega,
        saw_srange=saw_srange,
        saw_m=saw_m,
        saw_n=saw_n,
        saw_phihats=saw_phihats,
        saw_phases=saw_phases,
        saw_nharmonics=saw_nharmonics,
        stz_init=points,
        m=mass,
        q=charge,
        vtotal=v_reference,
        vtang=vpar,
        mus=mus,
        tmax=tmax,
        tol=tol,
        dt_in=dt if dt is not None else -np.ones(nparticles),
        psi0=B0.psi0,
        nparticles=nparticles,
    )
    return np.reshape(last_time, (nparticles, 6))


def trace_particles_cartesian_gpu(
    field,
    surface_classifier,
    xyz_inits,
    parallel_speeds,
    tmax,
    mass,
    charge,
    vtotal,
    tol,
    dt=None,
):
    """
    Trace particles in Cartesian coordinates using CATAPULT
    field: a magnetic field object representing the field in Cartesian coordinates
    surface_classifier: a simsopt surface classifier object for detecting a surface
    xyz_inits: initial conditions for particles in (x, y, z) coordinates
    parallel_speeds: initial parallel speeds of the particles
    tmax: maximum time to trace particles
    mass: mass of each particle
    charge: charge of each particle
    vtotal: total velocity of each particle
    tol: tolerance for the ODE solver
    dt: the initial time step size for the solver (optional)
    """
    nparticles = xyz_inits.shape[0]
    r_range, phi_range, z_range, quad_info = cartesian_interpolant(
        field, surface_classifier
    )
    last_time = firm3dpp.cartesian_gpu_tracing(
        quad_pts=quad_info,
        rrange=r_range,
        phirange=phi_range,
        zrange=z_range,
        xyz_init=xyz_inits,
        m=mass,
        q=charge,
        vtotal=vtotal,
        vtang=parallel_speeds,
        tmax=tmax,
        tol=tol,
        dt_in=dt if dt is not None else -np.ones(nparticles),
        nparticles=nparticles,
    )
    last_time = np.reshape(last_time, (nparticles, 6))
    return last_time
