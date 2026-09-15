__all__ = ["trace_particles_boozer_gpu", "trace_particles_cartesian_gpu"]
import numpy as np

import firm3dpp
from firm3d.catapult.utils import (
    boozer_interpolant,
    boozer_saw_interpolant,
    cartesian_interpolant,
)
from firm3d.field.boozermagneticfield import ShearAlfvenWavesSuperposition


def _dt_in(dt, nparticles):
    # one initial step per particle; -1 lets the kernel choose
    if dt is None:
        return -np.ones(nparticles)
    return np.ascontiguousarray(
        np.broadcast_to(np.asarray(dt, dtype=float), nparticles)
    )


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
    vtotal: total velocity of each particle
    tol: tolerance for the ODE solver
    dt: the initial time step size for the solver (optional)
    """
    nparticles = stz_inits.shape[0]
    dt_in = _dt_in(dt, nparticles)

    if isinstance(field, ShearAlfvenWavesSuperposition):
        B0 = field.B0
        srange, trange, zrange, quad_info, maxJ = boozer_saw_interpolant(
            B0, B0.nfp, ns, ntheta, nzeta
        )
        saw_nharmonics = len(field)
        saw_omega = field.get_wave(0).omega
        saw_s = field.get_wave(0).phihat.get_s_basis()
        saw_srange = (saw_s[0], saw_s[-1], len(saw_s))
        saw_m = [field.get_wave(i).Phim for i in range(saw_nharmonics)]
        saw_n = [field.get_wave(i).Phin for i in range(saw_nharmonics)]
        saw_phihats = np.ascontiguousarray(
            np.column_stack(
                [
                    np.array([field.get_wave(i).phihat(s_val) for s_val in saw_s])
                    for i in range(saw_nharmonics)
                ]
            )
        )

        if B0.field_type == "vac":
            last_time = firm3dpp.boozer_saw_gpu_tracing(
                quad_pts=quad_info,
                srange=srange,
                trange=trange,
                zrange=zrange,
                saw_omega=saw_omega,
                saw_srange=saw_srange,
                saw_m=saw_m,
                saw_n=saw_n,
                saw_phihats=saw_phihats,
                saw_nharmonics=saw_nharmonics,
                stz_init=stz_inits,
                m=mass,
                q=charge,
                vtotal=vtotal,
                vtang=parallel_speeds,
                tmax=tmax,
                tol=tol,
                dt_in=dt_in,
                psi0=B0.psi0,
                nparticles=nparticles,
            )
        elif B0.field_type == "nok":
            last_time = firm3dpp.boozer_saw_nok_gpu_tracing(
                quad_pts=quad_info,
                srange=srange,
                trange=trange,
                zrange=zrange,
                saw_omega=saw_omega,
                saw_srange=saw_srange,
                saw_m=saw_m,
                saw_n=saw_n,
                saw_phihats=saw_phihats,
                saw_nharmonics=saw_nharmonics,
                stz_init=stz_inits,
                m=mass,
                q=charge,
                vtotal=vtotal,
                vtang=parallel_speeds,
                tmax=tmax,
                tol=tol,
                dt_in=dt_in,
                psi0=B0.psi0,
                nparticles=nparticles,
            )
        else:
            raise ValueError(f"Unsupported field type {B0.field_type} for SAW tracing")
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
            dt_in=dt_in,
            psi0=psi0,
            nparticles=nparticles,
            vacuum=vacuum,
        )

    last_time = np.reshape(last_time, (nparticles, 6))
    return last_time


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
    dt_in = _dt_in(dt, nparticles)
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
        dt_in=dt_in,
        nparticles=nparticles,
    )
    last_time = np.reshape(last_time, (nparticles, 6))
    return last_time
