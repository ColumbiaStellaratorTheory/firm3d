# import time
import unittest

import numpy as np

import firm3dpp

try:
    from simsopt.field.tracing import (
        IterationStoppingCriterion as SimsoptIterationStoppingCriterion,
    )
    from simsopt.geo import SurfaceRZFourier

    from simsopt.field import (
        BiotSavart,
        InterpolatedField,
        SurfaceClassifier,
        coils_via_symmetries,
        load_coils_from_makegrid_file,
        trace_particles,
    )

    HAS_SIMSOPT = True
except Exception:
    HAS_SIMSOPT = False
    InterpolatedField = type(None)
from firm3d.catapult.utils import (
    boozer_interpolant,
    boozer_saw_interpolant,
    cartesian_interpolant,
)
from firm3d.field.boozermagneticfield import (
    BoozerRadialInterpolant,
    InterpolatedBoozerField,
    ShearAlfvenWavesSuperposition,
)
from firm3d.field.tracing import (
    IterationStoppingCriterion,
    trace_particles_boozer,
    trace_particles_boozer_perturbed,
)
from firm3d.saw.ae3d import AE3DEigenvector
from firm3d.util.constants import (
    ALPHA_PARTICLE_CHARGE as CHARGE,
)
from firm3d.util.constants import (
    ALPHA_PARTICLE_MASS as MASS,
)
from firm3d.util.constants import (
    FUSION_ALPHA_PARTICLE_ENERGY as ENERGY,
)

HAS_CUDA = hasattr(firm3dpp, "test_gpu_interpolation")
n_test_pts = 10000


def get_field(boozmn_filename, n_metagrid_pts, vacuum):
    bri = BoozerRadialInterpolant(boozmn_filename, 3, enforce_vacuum=vacuum)
    nfp = bri.nfp
    degree = 3
    field = InterpolatedBoozerField(
        bri,
        degree,
        ns_interp=n_metagrid_pts,
        ntheta_interp=n_metagrid_pts,
        nzeta_interp=n_metagrid_pts,
    )
    # Even though bri isn't used further in this script, we need to return it,
    # or else it is garbage-collected, resulting in an error.
    return bri, field, nfp


def construct_interpolant(field, nfp, saw_present=False):
    ns, ntheta, nzeta = 15, 15, 15
    if isinstance(field, ShearAlfvenWavesSuperposition):
        field = field.B0
        srange, trange, zrange, quad_info, maxJ = boozer_saw_interpolant(
            field, nfp, ns, ntheta, nzeta
        )
    else:  # the field is an InterpolatedBoozerField (unperturbed)
        if field.field_type == "vac":
            srange, trange, zrange, quad_info, maxJ = boozer_interpolant(
                field, nfp, ns, ntheta, nzeta, vacuum=True
            )
        elif field.field_type == "":  # implies finite beta
            srange, trange, zrange, quad_info, maxJ = boozer_interpolant(
                field, nfp, ns, ntheta, nzeta, vacuum=False
            )

    return srange, trange, zrange, quad_info, maxJ


def sample_test_points(n_test_pts):
    np.random.seed(1865)
    # generate test points
    s = np.random.uniform(low=0, high=1.0, size=(n_test_pts, 1))
    t = np.random.uniform(low=0, high=2 * np.pi, size=(n_test_pts, 1))
    z = np.random.uniform(low=0, high=2 * np.pi, size=(n_test_pts, 1))
    stz = np.hstack((s, t, z))
    return stz


def cartesian_rhs(position, vpar, field, mass, charge, velocity):
    field.set_points_cyl(position.reshape(-1, 3))
    B = field.B()
    GradAbsB = field.GradAbsB()
    AbsB = np.linalg.norm(B[0])

    BcrossGradAbsB = [0] * 3
    BcrossGradAbsB[0] = B[0, 1] * GradAbsB[0, 2] - B[0, 2] * GradAbsB[0, 1]
    BcrossGradAbsB[1] = B[0, 2] * GradAbsB[0, 0] - B[0, 0] * GradAbsB[0, 2]
    BcrossGradAbsB[2] = B[0, 0] * GradAbsB[0, 1] - B[0, 1] * GradAbsB[0, 0]

    v_perp2 = velocity**2 - vpar**2
    mu = v_perp2 / (2 * AbsB)
    fak1 = vpar / AbsB
    fak2 = (mass / (charge * AbsB**3)) * (0.5 * v_perp2 + vpar**2)

    out = [0] * 4
    for i in range(3):
        out[i] = fak1 * B[0, i] + fak2 * BcrossGradAbsB[i]
    out[3] = -mu * np.sum([B[0, i] * GradAbsB[0, i] for i in range(3)]) / AbsB
    return out


def test_interpolant(
    field, nfp, stz, saw_present=False, surf_classifier=None, tol=1e-8
):
    # if in Cartesian coordinates
    if isinstance(field, InterpolatedField):
        rrange, phirange, zrange, quad_info = cartesian_interpolant(
            field, surf_classifier
        )
        field.set_points_cyl(stz)
        # Quantities to interpolate
        B = field.B_cyl()
        GradAbsB = field.GradAbsB_cyl()

        # Compare interpolation of B and GradAbsB
        cpu_interpolation = np.hstack((B, GradAbsB))

        gpu_interpolation_dbl = firm3dpp.test_gpu_interpolation(
            quad_info,
            rrange,
            phirange,
            zrange,
            stz.copy(),
            "cartesian_vacuum",
            stz.shape[0],
        )
        gpu_interpolation_dbl = np.reshape(gpu_interpolation_dbl, (stz.shape[0], -1))
        gpu_interpolation_dbl = gpu_interpolation_dbl[:, 0:6]

        gpu_interpolation_flt = firm3dpp.test_gpu_interpolation(
            quad_info.astype(np.float32),
            rrange,
            phirange,
            zrange,
            stz.copy().astype(np.float32),
            "cartesian_vacuum",
            stz.shape[0],
        )
        gpu_interpolation_flt = np.reshape(gpu_interpolation_flt, (stz.shape[0], -1))
        gpu_interpolation_flt = gpu_interpolation_flt[:, 0:6]

    else:  # Boozer coordinates
        srange, trange, zrange, quad_info, maxJ = construct_interpolant(
            field, nfp, saw_present=saw_present
        )

        # evaluate interpolants
        if isinstance(field, ShearAlfvenWavesSuperposition):
            field = field.B0
            field.set_points(stz)
            modB = field.modB()
            modB_derivs = field.modB_derivs()
            G = field.G()
            dGds = field.dGds()
            I = field.I()
            dIds = field.dIds()
            iota = field.iota()
            diotads = field.diotads()
            cpu_interpolation = np.hstack(
                (modB, modB_derivs, G, dGds, I, dIds, iota, diotads)
            )

            ## evaluate GPU interpolant
            stz = np.ascontiguousarray(stz)
            gpu_interpolation_dbl = firm3dpp.test_gpu_interpolation(
                quad_info,
                srange,
                trange,
                zrange,
                stz.copy(),
                "boozer_saw_vacuum",
                stz.shape[0],
            )

            gpu_interpolation_flt = firm3dpp.test_gpu_interpolation(
                quad_info.astype(np.float32),
                srange,
                trange,
                zrange,
                stz.copy().astype(np.float32),
                "boozer_saw_vacuum",
                stz.shape[0],
            )
        else:
            if field.field_type == "vac":
                # evaluate CPU interpolant
                field.set_points(stz)
                modB = field.modB()
                modB_derivs = field.modB_derivs()
                G = field.G()
                iota = field.iota()
                cpu_interpolation = np.hstack((modB, modB_derivs, G, iota))

                ## evaluate GPU interpolant
                stz = np.ascontiguousarray(stz)
                gpu_interpolation_dbl = firm3dpp.test_gpu_interpolation(
                    quad_info,
                    srange,
                    trange,
                    zrange,
                    stz.copy(),
                    "boozer_vacuum",
                    stz.shape[0],
                )
                # print("computing float interpolants")
                gpu_interpolation_flt = firm3dpp.test_gpu_interpolation(
                    quad_info.astype(np.float32),
                    srange,
                    trange,
                    zrange,
                    stz.copy().astype(np.float32),
                    "boozer_vacuum",
                    stz.shape[0],
                )

            elif field.field_type == "":  # implies finite beta
                # evaluate CPU interpolant
                field.set_points(stz)
                modB = field.modB()
                modB_derivs = field.modB_derivs()
                G = field.G()
                dGds = field.dGds()
                I = field.I()
                dIds = field.dIds()
                iota = field.iota()
                K = field.K()
                K_derivs = field.K_derivs()
                cpu_interpolation = np.hstack(
                    (modB, modB_derivs, G, dGds, I, dIds, iota, K, K_derivs)
                )

                # evaluate GPU interpolant
                stz = np.ascontiguousarray(stz)
                gpu_interpolation_dbl = firm3dpp.test_gpu_interpolation(
                    quad_info,
                    srange,
                    trange,
                    zrange,
                    stz.copy(),
                    "boozer",
                    stz.shape[0],
                )

                gpu_interpolation_flt = firm3dpp.test_gpu_interpolation(
                    quad_info.astype(np.float32),
                    srange,
                    trange,
                    zrange,
                    stz.copy().astype(np.float32),
                    "boozer",
                    stz.shape[0],
                )

        gpu_interpolation_dbl = np.reshape(gpu_interpolation_dbl, (stz.shape[0], -1))
        gpu_interpolation_flt = np.reshape(gpu_interpolation_flt, (stz.shape[0], -1))

    # compute error between cpu and gpu interpolants
    device_error_is_small = np.isclose(
        gpu_interpolation_dbl, cpu_interpolation, rtol=tol, atol=tol
    ).all()
    error = np.abs(cpu_interpolation - gpu_interpolation_dbl) / (
        np.abs(cpu_interpolation) + 1
    )
    if error.max() > tol:
        print("tolerance not satisfied in interpolant")
        row_idx = np.unravel_index(np.argmax(error), error.shape)[0]
        print(row_idx)
        print("stz:", stz[row_idx, :])
        print("cpu:", cpu_interpolation[row_idx, :])
        print("gpu:", gpu_interpolation_dbl[row_idx, :])
        print("error:", error[row_idx, :])

    # compute error between single and double precision on gpu
    error = np.abs(
        (gpu_interpolation_dbl - gpu_interpolation_flt)
        / (np.max(gpu_interpolation_dbl, axis=0) + 1e-16)
    )
    print("max error in interpolant precision comparison on gpu: ", error.max())

    # precision_error_is_small = error.max() <= 1e-4
    # if not precision_error_is_small:
    #     print("tolerance not satisfied in interpolant")
    #     print(error.max())
    #     row_idx = np.unravel_index(np.argmax(error), error.shape)[0]
    #     print("stz:", stz[row_idx, :])
    #     print("flt:", gpu_interpolation_flt[row_idx, :])
    #     print("dbl:", gpu_interpolation_dbl[row_idx, :])
    #     print("diff: ", gpu_interpolation_dbl[row_idx, :] - gpu_interpolation_flt[row_idx, :])
    #     print("prec. error:", error[row_idx, :])
    #     print("rel. error: ", np.abs((gpu_interpolation_dbl[row_idx, :] - gpu_interpolation_flt[row_idx, :]) / gpu_interpolation_dbl[row_idx, :]))

    return device_error_is_small  # and precision_error_is_small


def test_derivatives(
    field,
    nfp,
    stz,
    vpar,
    vtotal,
    psi0=0,
    time=None,
    saw_present=False,
    saw_filename=None,
    surf_classifier=None,
    tol=1e-8,
):
    if isinstance(field, InterpolatedField):  # Cartesian
        rrange, phirange, zrange, quad_info = cartesian_interpolant(
            field, surf_classifier
        )
        gpu_derivs_dbl = firm3dpp.test_derivatives_cartesian(
            quad_info,
            rrange,
            phirange,
            zrange,
            stz,
            vpar,
            vtotal,
            MASS,
            CHARGE,
            stz.shape[0],
        )
        gpu_derivs_dbl = np.reshape(gpu_derivs_dbl, (stz.shape[0], 4))

        gpu_derivs_flt = firm3dpp.test_derivatives_cartesian(
            quad_info.astype(np.float32),
            rrange,
            phirange,
            zrange,
            stz.astype(np.float32),
            vpar.astype(np.float32),
            vtotal.astype(np.float32),
            MASS,
            CHARGE,
            stz.shape[0],
        )
        gpu_derivs_flt = np.reshape(gpu_derivs_flt, (stz.shape[0], 4))

        cpu_derivs = np.empty((stz.shape[0], 4))
        for i in range(stz.shape[0]):
            cpu_derivs[i, :] = cartesian_rhs(
                stz[i, :], vpar[i], field, MASS, CHARGE, vtotal
            )

    else:
        srange, trange, zrange, quad_info, maxJ = construct_interpolant(field, nfp)
        ## evaluate derivatives
        if isinstance(field, ShearAlfvenWavesSuperposition):
            assert time is not None, (
                "time array must be provided when testing derivatives with SAW"
            )
            assert saw_filename is not None, (
                "saw filename must be provided when testing derivatives with SAW"
            )
            # evaluate CPU derivatives
            cpu_derivs = np.empty((stz.shape[0], 4))

            if field.B0.field_type == "vac":
                for i in range(stz.shape[0]):
                    cpu_derivs[i, :] = firm3dpp.simsopt_derivs_saw(
                        field,
                        stz[i, :],
                        MASS,
                        CHARGE,
                        vtotal,
                        vpar[i],
                        time[i],
                        "vacuum_saw",
                    )
            elif field.B0.field_type == "nok":  # NoK tracing
                for i in range(stz.shape[0]):
                    cpu_derivs[i, :] = firm3dpp.simsopt_derivs_saw(
                        field,
                        stz[i, :],
                        MASS,
                        CHARGE,
                        vtotal,
                        vpar[i],
                        time[i],
                        "nok_saw",
                    )
            else:
                ValueError("Field type not recognized for SAW derivatives")

            saw_nharmonics = 5
            ## load saw data as arrays
            saw_data = np.load(saw_filename, allow_pickle=True)
            saw_data = saw_data[()]
            saw_omega = field.get_wave(0).omega
            s = field.get_wave(0).phihat.get_s_basis()
            saw_srange = (s[0], s[-1], len(s))

            saw_m = [field.get_wave(i).Phim for i in range(saw_nharmonics)]
            saw_n = [field.get_wave(i).Phin for i in range(saw_nharmonics)]
            saw_phihats = np.ascontiguousarray(
                np.column_stack(
                    [
                        np.array([field.get_wave(i).phihat(s_val) for s_val in s])
                        for i in range(saw_nharmonics)
                    ]
                )
            )
            ## evaluate GPU interpolant
            stz = np.ascontiguousarray(stz)
            vpar = np.ascontiguousarray(vpar)

            if field.B0.field_type == "vac":
                gpu_derivs_dbl = firm3dpp.test_derivatives_saw(
                    quad_info,
                    srange,
                    trange,
                    zrange,
                    saw_omega,
                    saw_srange,
                    saw_m,
                    saw_n,
                    saw_phihats,
                    saw_nharmonics,
                    stz,
                    vpar,
                    time,
                    vtotal,
                    MASS,
                    CHARGE,
                    psi0,
                    stz.shape[0],
                )

                gpu_derivs_flt = firm3dpp.test_derivatives_saw(
                    quad_info.astype(np.float32),
                    srange,
                    trange,
                    zrange,
                    saw_omega,
                    saw_srange,
                    saw_m,
                    saw_n,
                    saw_phihats.astype(np.float32),
                    saw_nharmonics,
                    stz.astype(np.float32),
                    vpar.astype(np.float32),
                    time.astype(np.float32),
                    vtotal,
                    MASS,
                    CHARGE,
                    psi0,
                    stz.shape[0],
                )
            elif field.B0.field_type == "nok":
                gpu_derivs_dbl = firm3dpp.test_derivatives_saw_nok(
                    quad_info,
                    srange,
                    trange,
                    zrange,
                    saw_omega,
                    saw_srange,
                    saw_m,
                    saw_n,
                    saw_phihats,
                    saw_nharmonics,
                    stz,
                    vpar,
                    time,
                    vtotal,
                    MASS,
                    CHARGE,
                    psi0,
                    stz.shape[0],
                )
                gpu_derivs_flt = firm3dpp.test_derivatives_saw_nok(
                    quad_info.astype(np.float32),
                    srange,
                    trange,
                    zrange,
                    saw_omega,
                    saw_srange,
                    saw_m,
                    saw_n,
                    saw_phihats.astype(np.float32),
                    saw_nharmonics,
                    stz.astype(np.float32),
                    vpar.astype(np.float32),
                    time.astype(np.float32),
                    vtotal,
                    MASS,
                    CHARGE,
                    psi0,
                    stz.shape[0],
                )
            else:
                ValueError("Field type not recognized for SAW derivatives")
        else:
            if field.field_type == "vac":
                # evaluate CPU derivatives
                cpu_derivs = np.empty((stz.shape[0], 4))
                for i in range(stz.shape[0]):
                    cpu_derivs[i, :] = firm3dpp.simsopt_derivs_boozer(
                        field, stz[i, :], MASS, CHARGE, vtotal, vpar[i], vacuum=True
                    )

                ## evaluate GPU interpolant
                stz = np.ascontiguousarray(stz)
                vpar = np.ascontiguousarray(vpar)
                gpu_derivs_dbl = firm3dpp.test_derivatives_boozer(
                    quad_info,
                    srange,
                    trange,
                    zrange,
                    stz.copy(),
                    vpar,
                    vtotal,
                    MASS,
                    CHARGE,
                    psi0,
                    stz.shape[0],
                    vacuum=True,
                )

                gpu_derivs_flt = firm3dpp.test_derivatives_boozer(
                    quad_info.astype(np.float32),
                    srange,
                    trange,
                    zrange,
                    stz.copy().astype(np.float32),
                    vpar.astype(np.float32),
                    vtotal,
                    MASS,
                    CHARGE,
                    psi0,
                    stz.shape[0],
                    vacuum=True,
                )

            elif field.field_type == "":  # implies finite beta
                # evaluate CPU derivatives
                cpu_derivs = np.empty((stz.shape[0], 4))
                # start_time = time.time()
                for i in range(stz.shape[0]):
                    cpu_derivs[i, :] = firm3dpp.simsopt_derivs_boozer(
                        field, stz[i, :], MASS, CHARGE, vtotal, vpar[i], vacuum=False
                    )

                ## evaluate GPU interpolant
                stz = np.ascontiguousarray(stz)
                vpar = np.ascontiguousarray(vpar)
                gpu_derivs_dbl = firm3dpp.test_derivatives_boozer(
                    quad_info,
                    srange,
                    trange,
                    zrange,
                    stz.copy(),
                    vpar,
                    vtotal,
                    MASS,
                    CHARGE,
                    psi0,
                    stz.shape[0],
                    vacuum=False,
                )

                gpu_derivs_flt = firm3dpp.test_derivatives_boozer(
                    quad_info.astype(np.float32),
                    srange,
                    trange,
                    zrange,
                    stz.copy().astype(np.float32),
                    vpar.astype(np.float32),
                    vtotal,
                    MASS,
                    CHARGE,
                    psi0,
                    stz.shape[0],
                    vacuum=False,
                )
        gpu_derivs_dbl = np.reshape(gpu_derivs_dbl, (stz.shape[0], 4))
        gpu_derivs_flt = np.reshape(gpu_derivs_flt, (stz.shape[0], 4))

    device_error_is_small = np.isclose(
        gpu_derivs_dbl, cpu_derivs, rtol=tol, atol=tol
    ).all()
    error = np.abs(cpu_derivs - gpu_derivs_dbl) / (np.abs(cpu_derivs) + 1)

    if not device_error_is_small:
        row_idx = np.unravel_index(np.argmax(error), error.shape)[0]
        print("stz:", stz[row_idx, :])
        print("cpu:", cpu_derivs[row_idx, :])
        print("gpu:", gpu_derivs_dbl[row_idx, :])
        print("rel error:", error[row_idx, :])

    # compute error between single and double precision on gpu
    input_diff = (vpar - vpar.astype(np.float32)) / (vpar + 1e-16)
    precision_error_is_small = np.isclose(
        gpu_derivs_dbl, gpu_derivs_flt, rtol=tol, atol=1e3 * np.max(input_diff)
    ).all()
    error = np.abs(gpu_derivs_dbl - gpu_derivs_flt) / (np.abs(gpu_derivs_dbl) + 1)
    print("max error in derivative precision comparison: ", error.max())
    # if error.max() > 1e3*np.max(input_diff):
    #     print("precision tolerance not satisfied in derivatives")
    #     row_idx = np.unravel_index(np.argmax(error), error.shape)[0]
    #     print(row_idx)
    #     print("stz:", stz[row_idx, :])
    #     print("cpu: ", cpu_derivs[row_idx, :])
    #     print("flt:", gpu_derivs_flt[row_idx, :])
    #     print("dbl:", gpu_derivs_dbl[row_idx, :])
    #     print("error:", error[row_idx, :])
    #     print("max input diff:", np.max(input_diff))

    return device_error_is_small  # and precision_error_is_small


def test_timestep(
    field,
    nfp,
    stz,
    vpar,
    vtotal,
    psi0=0,
    time=None,
    saw_filename=None,
    tol=1e-8,
    surf_classifier=None,
):
    if isinstance(field, InterpolatedField):  # Cartesian
        rrange, phirange, zrange, quad_info = cartesian_interpolant(
            field, surf_classifier
        )
        last_time = firm3dpp.test_timestep_cartesian(
            quad_pts=quad_info,
            rrange=rrange,
            phirange=phirange,
            zrange=zrange,
            loc_init=stz,
            m=MASS,
            q=CHARGE,
            vtotal=np.sqrt(2 * ENERGY / MASS),
            vtang=vpar,
            tol=1e-9,
            nparticles=stz.shape[0],
        )
        gpu_final_positions = np.reshape(last_time, (stz.shape[0], 5))

        rphiz = stz
        r = rphiz[:, 0].reshape(-1, 1)
        phi = rphiz[:, 1].reshape(-1, 1)
        z = rphiz[:, 2].reshape(-1, 1)
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        xyz = np.hstack((x, y, z))
        gc_tys, gc_zeta_hits = trace_particles(
            field,
            xyz,
            vpar,
            tmax=1e-2,
            mass=MASS,
            charge=CHARGE,
            Ekin=ENERGY,
            tol=1e-9,
            stopping_criteria=[SimsoptIterationStoppingCriterion(1)],
            forget_exact_path=True,
        )
        cpu_positions = np.array([x[-1] for x in gc_tys])

    else:
        srange, trange, zrange, quad_info, maxJ = construct_interpolant(field, nfp)

        if isinstance(field, ShearAlfvenWavesSuperposition):
            assert saw_filename is not None, (
                "saw filename must be provided when testing timesteps with SAW"
            )
            # evaluate CPU timestep
            field.B0.set_points(stz)
            mu_init = (vtotal**2 - vpar**2) / (2 * field.B0.modB()[:, 0])

            gc_tys, gc_zeta_hits = trace_particles_boozer_perturbed(
                field,
                stz,
                vpar,
                mu_init,
                tmax=1e-2,
                mass=MASS,
                charge=CHARGE,
                tol=1e-9,
                stopping_criteria=[IterationStoppingCriterion(1)],
                forget_exact_path=True,
            )

            saw_nharmonics = 5
            ## load saw data as arrays
            saw_data = np.load(saw_filename, allow_pickle=True)
            saw_data = saw_data[()]
            saw_omega = field.get_wave(0).omega
            s = field.get_wave(0).phihat.get_s_basis()
            saw_srange = (s[0], s[-1], len(s))

            saw_m = [field.get_wave(i).Phim for i in range(saw_nharmonics)]
            saw_n = [field.get_wave(i).Phin for i in range(saw_nharmonics)]
            saw_phihats = np.ascontiguousarray(
                np.column_stack(
                    [
                        np.array([field.get_wave(i).phihat(s_val) for s_val in s])
                        for i in range(saw_nharmonics)
                    ]
                )
            )
            stz = np.ascontiguousarray(stz)
            psi0 = field.B0.psi0

            if field.B0.field_type == "vac":
                last_time = firm3dpp.test_timestep_saw(
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
                    stz_init=stz,
                    m=MASS,
                    q=CHARGE,
                    vtotal=np.sqrt(2 * ENERGY / MASS),
                    vtang=vpar,
                    time=time,
                    tol=1e-9,
                    psi0=psi0,
                    nparticles=stz.shape[0],
                )
            elif field.B0.field_type == "nok":
                last_time = firm3dpp.test_timestep_saw_nok(
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
                    stz_init=stz,
                    m=MASS,
                    q=CHARGE,
                    vtotal=np.sqrt(2 * ENERGY / MASS),
                    vtang=vpar,
                    time=time,
                    tol=1e-9,
                    psi0=psi0,
                    nparticles=stz.shape[0],
                )
            last_time = np.reshape(last_time, (stz.shape[0], 5))
        else:
            if field.field_type == "vac":
                gc_tys, gc_zeta_hits = trace_particles_boozer(
                    field,
                    stz,
                    vpar,
                    tmax=1e-2,
                    mass=MASS,
                    charge=CHARGE,
                    Ekin=ENERGY,
                    tol=1e-9,
                    stopping_criteria=[IterationStoppingCriterion(1)],
                    forget_exact_path=True,
                )

                stz = np.ascontiguousarray(stz)
                psi0 = field.psi0
                last_time = firm3dpp.test_timestep_boozer(
                    quad_pts=quad_info,
                    srange=srange,
                    trange=trange,
                    zrange=zrange,
                    stz_init=stz,
                    m=MASS,
                    q=CHARGE,
                    vtotal=np.sqrt(2 * ENERGY / MASS),
                    vtang=vpar,
                    tol=1e-9,
                    psi0=psi0,
                    nparticles=stz.shape[0],
                    vacuum=True,
                )
                last_time = np.reshape(last_time, (stz.shape[0], 5))
            elif field.field_type == "":  # implies finite beta
                gc_tys, gc_zeta_hits = trace_particles_boozer(
                    field,
                    stz,
                    vpar,
                    tmax=1e-2,
                    mass=MASS,
                    charge=CHARGE,
                    Ekin=ENERGY,
                    tol=1e-9,
                    stopping_criteria=[IterationStoppingCriterion(1)],
                    forget_exact_path=True,
                )

                stz = np.ascontiguousarray(stz)
                psi0 = field.psi0
                last_time = firm3dpp.test_timestep_boozer(
                    quad_pts=quad_info,
                    srange=srange,
                    trange=trange,
                    zrange=zrange,
                    stz_init=stz,
                    m=MASS,
                    q=CHARGE,
                    vtotal=np.sqrt(2 * ENERGY / MASS),
                    vtang=vpar,
                    tol=1e-9,
                    psi0=psi0,
                    nparticles=stz.shape[0],
                    vacuum=False,
                )
                last_time = np.reshape(last_time, (stz.shape[0], 5))

        # map to pseudo-cylindrical coordinates
        cpu_positions = np.array([x[-1] for x in gc_tys])
        cpu_positions = np.array(
            [
                [
                    x[0],
                    x[1] * np.cos(x[2]),
                    x[1] * np.sin(x[2]),
                    np.fmod(x[3], zrange[1]),
                    x[4],
                ]
                for x in cpu_positions
            ]
        )
        gpu_final_positions = np.array(
            [
                [x[0], x[1] * np.cos(x[2]), x[1] * np.sin(x[2]), x[3], x[4]]
                for x in last_time
            ]
        )
    error_is_small = np.isclose(
        gpu_final_positions, cpu_positions, rtol=tol, atol=tol
    ).all()
    error = np.abs(cpu_positions - gpu_final_positions) / (np.abs(cpu_positions) + 1)

    if error.max() > tol:
        row_idx = np.unravel_index(np.argmax(error), error.shape)[0]
        print("stz:", stz[row_idx, :])
        print("cpu:", cpu_positions[row_idx, :])
        print("gpu:", gpu_final_positions[row_idx, :])
        print("error:", error[row_idx, :])
        print("zrange:", zrange)

    return error_is_small


@unittest.skipUnless(HAS_CUDA, "CUDA support not available")
class TestGPUTracing(unittest.TestCase):
    def test_boozer_vacuum(self):
        print("test_boozer_vacuum")
        n_metagrid_pts = 15

        ### Vacuum case
        boozmn_filename = "examples/inputs/boozmn_aten_rescaled_low_res.nc"
        vacuum = True
        bri, field, nfp = get_field(boozmn_filename, n_metagrid_pts, vacuum)

        stz = sample_test_points(n_test_pts)

        tol = 1e-8

        # stz = stz[[21360], :]

        ### test interpolant
        is_small = test_interpolant(field, nfp, stz, tol=tol)
        self.assertTrue(is_small)

        ### test derivatives
        VELOCITY = np.sqrt(2 * ENERGY / MASS)
        vpar_init = np.random.uniform(-VELOCITY, VELOCITY, (n_test_pts,))

        # vpar_init = [vpar_init[21360]]

        is_small = test_derivatives(
            field, nfp, stz, vpar_init, VELOCITY, field.psi0, tol
        )
        self.assertTrue(is_small)

        ### test timesteps
        is_small = test_timestep(field, nfp, stz, vpar_init, VELOCITY, field.psi0, tol)
        self.assertTrue(is_small)

    def test_boozer_finite_beta(self):
        print("test_boozer_finite_beta")
        n_metagrid_pts = 15

        ### Vacuum case
        boozmn_filename = "examples/inputs/boozmn_aten_rescaled_low_res.nc"
        vacuum = False
        bri, field, nfp = get_field(boozmn_filename, n_metagrid_pts, vacuum)

        stz = sample_test_points(n_test_pts)

        tol = 1e-8

        ### test interpolant
        is_small = test_interpolant(field, nfp, stz, tol)
        self.assertTrue(is_small)

        ### test derivatives
        VELOCITY = np.sqrt(2 * ENERGY / MASS)
        vpar_init = np.random.uniform(-VELOCITY, VELOCITY, (n_test_pts,))
        is_small = test_derivatives(
            field, nfp, stz, vpar_init, VELOCITY, field.psi0, tol
        )
        self.assertTrue(is_small)

        ### test timesteps
        is_small = test_timestep(field, nfp, stz, vpar_init, VELOCITY, field.psi0, tol)
        self.assertTrue(is_small)

    def test_boozer_vacuum_saw(self):
        print("test_boozer_vacuum_saw")
        n_metagrid_pts = 15

        ### Vacuum case
        boozmn_filename = "examples/inputs/boozmn_aten_rescaled_low_res.nc"
        vacuum = True
        bri, field, nfp = get_field(boozmn_filename, n_metagrid_pts, vacuum)

        ### set up SAW
        saw_filename = "./examples/tracing_with_AE/ae.npy"
        saw = ShearAlfvenWavesSuperposition.from_ae3d(
            eigenvector=AE3DEigenvector.load_from_numpy(
                filename=saw_filename,
            ),
            B0=field,
            max_dB_normal_by_B0=5e-3,
            minor_radius_meters=1.7,
        )

        stz = sample_test_points(n_test_pts)
        tol = 1e-8

        ### test interpolant
        is_small = test_interpolant(saw, nfp, stz, saw_present=True, tol=tol)
        self.assertTrue(is_small)

        ## test derivatives
        VELOCITY = np.sqrt(2 * ENERGY / MASS)
        vpar_init = np.random.uniform(-VELOCITY, VELOCITY, (n_test_pts,))
        time = np.random.uniform(low=0, high=1e-3, size=(n_test_pts,))
        is_small = test_derivatives(
            saw,
            nfp,
            stz,
            vpar_init,
            VELOCITY,
            field.psi0,
            time=time,
            saw_present=True,
            saw_filename=saw_filename,
            tol=tol,
        )
        self.assertTrue(is_small)

        ### test timesteps
        is_small = test_timestep(
            saw,
            nfp,
            stz,
            vpar_init,
            VELOCITY,
            field.psi0,
            time=time,
            saw_filename=saw_filename,
            tol=tol,
        )
        self.assertTrue(is_small)

    def test_boozer_nok_saw(self):
        print("test_boozer_nok_saw")
        n_metagrid_pts = 15

        ### Vacuum case
        boozmn_filename = "examples/inputs/boozmn_aten_rescaled_low_res.nc"
        vacuum = True
        bri, field, nfp = get_field(boozmn_filename, n_metagrid_pts, vacuum)

        ### set up SAW
        saw_filename = "./examples/tracing_with_AE/ae.npy"
        saw = ShearAlfvenWavesSuperposition.from_ae3d(
            eigenvector=AE3DEigenvector.load_from_numpy(
                filename=saw_filename,
            ),
            B0=field,
            max_dB_normal_by_B0=5e-3,
            minor_radius_meters=1.7,
        )

        stz = sample_test_points(n_test_pts)
        tol = 1e-8

        ### test interpolant
        is_small = test_interpolant(saw, nfp, stz, saw_present=True, tol=tol)
        self.assertTrue(is_small)

        ## test derivatives
        VELOCITY = np.sqrt(2 * ENERGY / MASS)
        vpar_init = np.random.uniform(-VELOCITY, VELOCITY, (n_test_pts,))
        time = np.random.uniform(low=0, high=1e-3, size=(n_test_pts,))
        is_small = test_derivatives(
            saw,
            nfp,
            stz,
            vpar_init,
            VELOCITY,
            field.psi0,
            time=time,
            saw_present=True,
            saw_filename=saw_filename,
            tol=tol,
        )
        self.assertTrue(is_small)

        ### test timesteps
        is_small = test_timestep(
            saw,
            nfp,
            stz,
            vpar_init,
            VELOCITY,
            field.psi0,
            time=time,
            saw_filename=saw_filename,
            tol=tol,
        )
        self.assertTrue(is_small)

    @unittest.skipUnless(HAS_SIMSOPT, "simsopt not available")
    def test_cartesian_vacuum(self):
        np.random.seed(0)
        degree = 3  # degree of interpolant
        n = 16  # resolution of interpolant
        order = 12  # order of coil curves

        filename = "examples/inputs/coils.curves_22_7_21"
        wout_filename = "examples/inputs/wout_aten_rescaled.nc"

        surf = SurfaceRZFourier.from_wout(wout_filename)

        coils = load_coils_from_makegrid_file(filename, order, ppp=20, group_names=None)

        curves = []
        currents = []
        for _i, coil in enumerate(coils):
            curves.append(coil.curve)
            currents.append(coil.current)

        coils_full = coils_via_symmetries(curves, currents, surf.nfp, True)
        bs = BiotSavart(coils_full)

        sc_particle = SurfaceClassifier(surf, h=0.1, p=2)
        rs = np.linalg.norm(surf.gamma()[:, :, 0:2], axis=2)
        zs = surf.gamma()[:, :, 2]

        rrange = (np.min(rs), np.max(rs), n)
        phirange = (0, 2 * np.pi / surf.nfp, n * 2)
        # exploit stellarator symmetry and only consider positive z values:
        zrange = (0, np.max(zs), n // 2)
        bsh = InterpolatedField(
            bs, degree, rrange, phirange, zrange, True, nfp=surf.nfp, stellsym=True
        )

        # rejection sample points inside the surface uniformly
        rphiz = np.empty((n_test_pts, 3))
        for i in range(n_test_pts):
            pt = np.random.uniform(low=0, high=1, size=(1, 3))
            pt[0, 0] = pt[0, 0] * (rrange[1] - rrange[0]) + rrange[0]
            pt[0, 1] *= 2 * np.pi
            pt[0, 2] = (pt[0, 2] - 0.5) * 2 * zrange[1]

            # particle is outside the surface or too close to the surface
            max_iters = 1000
            for _ in range(max_iters):
                if sc_particle.evaluate_rphiz(pt) > 0.2:
                    break
                pt = np.random.uniform(low=0, high=1, size=(1, 3))
                pt[0, 0] = pt[0, 0] * (rrange[1] - rrange[0]) + rrange[0]
                pt[0, 1] *= 2 * np.pi
                pt[0, 2] = (pt[0, 2] - 0.5) * 2 * zrange[1]
            else:
                raise RuntimeError("Could not sample a valid point inside the surface")
            rphiz[i, :] = pt
        xyz = np.empty((n_test_pts, 3))
        xyz[:, 0] = rphiz[:, 0] * np.cos(rphiz[:, 1])
        xyz[:, 1] = rphiz[:, 0] * np.sin(rphiz[:, 1])
        xyz[:, 2] = rphiz[:, 2]

        # test interpolant
        is_small = test_interpolant(
            bsh, surf.nfp, rphiz, surf_classifier=sc_particle, tol=1e-8
        )
        self.assertTrue(is_small)

        # test rhs
        VELOCITY = np.sqrt(2 * ENERGY / MASS)
        vpar_init = np.random.uniform(-VELOCITY, VELOCITY, (n_test_pts,))
        is_small = test_derivatives(
            bsh,
            surf.nfp,
            rphiz,
            vpar_init,
            VELOCITY,
            surf_classifier=sc_particle,
            tol=1e-8,
        )
        self.assertTrue(is_small)

        # test timestep
        is_small = test_timestep(
            bsh,
            surf.nfp,
            rphiz,
            vpar_init,
            VELOCITY,
            surf_classifier=sc_particle,
            tol=1e-8,
        )
        self.assertTrue(is_small)


if __name__ == "__main__":
    print("Running GPU tracing tests...")
    unittest.main()
