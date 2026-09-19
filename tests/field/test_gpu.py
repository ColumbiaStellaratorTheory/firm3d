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
from firm3d.catapult.field import (
    CatapultBoozerField,
    CatapultPerturbedBoozerField,
)
from firm3d.catapult.tracing import (
    advance_particles_boozer_gpu,
    advance_particles_boozer_perturbed_gpu,
    save_trajectories_boozer_gpu,
    trace_particles_boozer_gpu,
)
from firm3d.field.tracing import MaxToroidalFluxStoppingCriterion
from firm3d.trajectory_helpers import compute_loss_fraction

HAS_CUDA = hasattr(firm3dpp, "test_gpu_interpolation")
n_test_pts = 10000


def sample_test_points(n_test_pts):
    np.random.seed(1)
    # generate test points
    s = np.random.uniform(low=0, high=1.1, size=(n_test_pts, 1))
    t = np.random.uniform(low=0, high=2 * np.pi, size=(n_test_pts, 1))
    z = np.random.uniform(low=0, high=2 * np.pi, size=(n_test_pts, 1))
    stz = np.hstack((s, t, z))
    return stz


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


class CATAPULTField:
    def __init__(
        self, field, ns, ntheta, nzeta, nfp, saw_filename=None, sc_classifier=None
    ):

        ### Set up interpolant grid
        self.field_type = None
        # if this is a SAW, get the underlying field
        if isinstance(field, ShearAlfvenWavesSuperposition):
            assert saw_filename is not None, (
                "SAW filename must be provided when testing derivatives with SAW"
            )

            self.saw_nharmonics = 5
            ## load saw data as arrays
            saw_data = np.load(saw_filename, allow_pickle=True)
            saw_data = saw_data[()]
            self.saw_omega = field.get_wave(0).omega
            s = field.get_wave(0).phihat.get_s_basis()
            self.saw_srange = (s[0], s[-1], len(s))

            self.saw_m = [field.get_wave(i).Phim for i in range(self.saw_nharmonics)]
            self.saw_n = [field.get_wave(i).Phin for i in range(self.saw_nharmonics)]
            self.saw_phihats = np.ascontiguousarray(
                np.column_stack(
                    [
                        np.array([field.get_wave(i).phihat(s_val) for s_val in s])
                        for i in range(self.saw_nharmonics)
                    ]
                )
            )
            self.saw_field = field
            field = field.B0
            range0, range1, range2, quad_info, maxJ = boozer_saw_interpolant(
                field, nfp, ns, ntheta, nzeta
            )
            self.field_type = (
                "boozer_saw_vacuum" if field.field_type == "vac" else "boozer_saw_nok"
            )
        elif isinstance(field, InterpolatedField):  # cartesian field
            range0, range1, range2, quad_info = cartesian_interpolant(
                field, sc_classifier
            )
            self.field_type = "cartesian_vacuum"
        else:  # the field is an InterpolatedBoozerField (unperturbed)
            if field.field_type == "vac":
                range0, range1, range2, quad_info, maxJ = boozer_interpolant(
                    field, nfp, ns, ntheta, nzeta, vacuum=True
                )
                self.field_type = "boozer_vacuum"
            elif field.field_type == "":  # implies finite beta
                range0, range1, range2, quad_info, maxJ = boozer_interpolant(
                    field, nfp, ns, ntheta, nzeta, vacuum=False
                )
                self.field_type = "boozer"
            else:
                raise ValueError("Field type not recognized")

        # set psi0 if in Boozer coordinates
        self.psi0 = None
        if self.field_type != "cartesian_vacuum":
            self.psi0 = field.psi0

        self.field = field
        self.ns = ns
        self.ntheta = ntheta
        self.nzeta = nzeta
        self.nfp = nfp

        self.range0 = range0
        self.range1 = range1
        self.range2 = range2

        self.quad_info = quad_info  # record interpolant data

    def compute_gpu_interpolant(self, stz, dtype=np.float64):
        # dtype selects the single or double precision kernel; the interpolant
        # data and evaluation points must both carry it (noconvert bindings).
        # test_gpu_interpolation converts (s, theta) to pseudo-Cartesian IN
        # PLACE in the array it is given, so always pass a copy.
        gpu_interpolation = firm3dpp.test_gpu_interpolation(
            self.quad_info.astype(dtype),
            self.range0,
            self.range1,
            self.range2,
            np.array(stz, dtype=dtype, order="C"),
            self.field_type,
            stz.shape[0],
        )
        gpu_interpolation = gpu_interpolation.reshape((stz.shape[0], -1))

        # remove surface classifier column
        if self.field_type == "cartesian_vacuum":
            gpu_interpolation = gpu_interpolation[:, 0:6]

        return gpu_interpolation

    def compute_cpu_interpolant(self, stz):
        self.field.set_points(stz)
        if self.field_type == "cartesian_vacuum":
            self.field.set_points_cyl(stz)
            cpu_interpolation = np.hstack(
                (self.field.B_cyl(), self.field.GradAbsB_cyl())
            )
        elif self.field_type in ["boozer_saw_vacuum", "boozer_saw_nok"]:
            cpu_interpolation = np.hstack(
                (
                    self.field.modB(),
                    self.field.modB_derivs(),
                    self.field.G(),
                    self.field.dGds(),
                    self.field.I(),
                    self.field.dIds(),
                    self.field.iota(),
                    self.field.diotads(),
                )
            )
        elif self.field_type == "boozer_vacuum":
            cpu_interpolation = np.hstack(
                (
                    self.field.modB(),
                    self.field.modB_derivs(),
                    self.field.G(),
                    self.field.iota(),
                )
            )
        elif self.field_type == "boozer":
            cpu_interpolation = np.hstack(
                (
                    self.field.modB(),
                    self.field.modB_derivs(),
                    self.field.G(),
                    self.field.dGds(),
                    self.field.I(),
                    self.field.dIds(),
                    self.field.iota(),
                    self.field.K(),
                    self.field.K_derivs(),
                )
            )
        else:
            raise ValueError("Field type not recognized in cpu interpolant")

        return cpu_interpolation

    def test_interpolant(self, stz, tol):
        gpu_interpolation = self.compute_gpu_interpolant(stz)
        cpu_interpolation = self.compute_cpu_interpolant(stz)

        gpu_error_is_small = np.allclose(
            gpu_interpolation, cpu_interpolation, rtol=tol, atol=tol
        )
        error = np.abs(cpu_interpolation - gpu_interpolation) / (
            np.abs(cpu_interpolation) + 1
        )
        if error.max() > tol:
            print("tolerance not satisfied in interpolant")
            row_idx = np.unravel_index(np.argmax(error), error.shape)[0]
            print(row_idx)
            print("stz:", stz[row_idx, :])
            print("cpu:", cpu_interpolation[row_idx, :])
            print("gpu:", gpu_interpolation[row_idx, :])
            print("error:", error[row_idx, :])
        return gpu_error_is_small

    def compute_gpu_derivatives(self, stz, vpar, vtotal, time=None, dtype=np.float64):
        # dtype selects the single or double precision kernel; every T-typed
        # array (interpolant data, positions, vpar, time, SAW phihats) must
        # carry it, since the bindings are noconvert
        quad_info = self.quad_info.astype(dtype)
        # np.array (not ascontiguousarray) so a float64 call never aliases the
        # caller's array: the derivative kernels convert positions in place
        stz = np.array(stz, dtype=dtype, order="C")
        vpar = np.ascontiguousarray(vpar, dtype=dtype)
        if time is not None:
            time = np.ascontiguousarray(time, dtype=dtype)
        saw_phihats = (
            self.saw_phihats.astype(dtype) if hasattr(self, "saw_phihats") else None
        )
        if self.field_type == "boozer_vacuum" or self.field_type == "boozer":
            gpu_derivs_dbl = firm3dpp.test_derivatives_boozer(
                quad_info,
                self.range0,
                self.range1,
                self.range2,
                stz.copy(),
                vpar,
                vtotal,
                MASS,
                CHARGE,
                self.psi0,
                stz.shape[0],
                vacuum=(self.field_type == "boozer_vacuum"),
            )
        elif self.field_type == "boozer_saw_vacuum":
            gpu_derivs_dbl = firm3dpp.test_derivatives_saw(
                quad_info,
                self.range0,
                self.range1,
                self.range2,
                self.saw_omega,
                self.saw_srange,
                self.saw_m,
                self.saw_n,
                saw_phihats,
                self.saw_nharmonics,
                stz,
                vpar,
                time,
                vtotal,
                MASS,
                CHARGE,
                self.psi0,
                stz.shape[0],
            )
        elif self.field_type == "boozer_saw_nok":
            gpu_derivs_dbl = firm3dpp.test_derivatives_saw_nok(
                quad_info,
                self.range0,
                self.range1,
                self.range2,
                self.saw_omega,
                self.saw_srange,
                self.saw_m,
                self.saw_n,
                saw_phihats,
                self.saw_nharmonics,
                stz,
                vpar,
                time,
                vtotal,
                MASS,
                CHARGE,
                self.psi0,
                stz.shape[0],
            )
        elif self.field_type == "cartesian_vacuum":
            gpu_derivs_dbl = firm3dpp.test_derivatives_cartesian(
                quad_info,
                self.range0,
                self.range1,
                self.range2,
                stz.copy(),
                vpar,
                vtotal,
                MASS,
                CHARGE,
                stz.shape[0],
            )
        else:
            raise ValueError(
                f"GPU derivative computation not implemented for this field \
                type: {self.field_type}"
            )

        return gpu_derivs_dbl.reshape((stz.shape[0], 4))

    def compute_cpu_derivatives(self, stz, vpar, vtotal, time=None):
        if self.field_type == "boozer_vacuum" or self.field_type == "boozer":
            cpu_derivs = np.empty((stz.shape[0], 4))
            for i in range(stz.shape[0]):
                cpu_derivs[i, :] = firm3dpp.simsopt_derivs_boozer(
                    self.field,
                    stz[i, :],
                    MASS,
                    CHARGE,
                    vtotal,
                    vpar[i],
                    vacuum=(self.field_type == "boozer_vacuum"),
                )
        elif self.field_type in ["boozer_saw_vacuum", "boozer_saw_nok"]:
            assert time is not None, (
                "time array must be provided when testing derivatives with SAW"
            )
            cpu_derivs = np.empty((stz.shape[0], 4))
            for i in range(stz.shape[0]):
                cpu_derivs[i, :] = firm3dpp.simsopt_derivs_saw(
                    self.saw_field,
                    stz[i, :],
                    MASS,
                    CHARGE,
                    vtotal,
                    vpar[i],
                    time[i],
                    "vacuum_saw"
                    if self.field_type == "boozer_saw_vacuum"
                    else "nok_saw",
                )
        elif self.field_type == "cartesian_vacuum":
            cpu_derivs = np.empty((stz.shape[0], 4))
            for i in range(stz.shape[0]):
                cpu_derivs[i, :] = cartesian_rhs(
                    stz[i, :], vpar[i], self.field, MASS, CHARGE, vtotal
                )
        else:
            raise ValueError(
                f"CPU derivative computation not implemented for this \
                field type: {self.field_type}"
            )

        return cpu_derivs

    def test_derivatives(self, stz, vpar, vtotal, tol, time=None):
        gpu_derivs = self.compute_gpu_derivatives(stz, vpar, vtotal, time=time)
        cpu_derivs = self.compute_cpu_derivatives(stz, vpar, vtotal, time=time)

        gpu_error_is_small = np.allclose(gpu_derivs, cpu_derivs, rtol=tol, atol=tol)
        error = np.abs(cpu_derivs - gpu_derivs) / (np.abs(cpu_derivs) + 1)
        if not gpu_error_is_small:
            row_idx = np.unravel_index(np.argmax(error), error.shape)[0]
            print("stz:", stz[row_idx, :])
            print("cpu:", cpu_derivs[row_idx, :])
            print("gpu:", gpu_derivs[row_idx, :])
            print("rel error:", error[row_idx, :])

        return gpu_error_is_small

    def test_precision(self, stz, vpar, vtotal, time=None, tol=1e-3):
        """Single vs double precision kernels, column-scale normalized so
        sign-changing components do not inflate the relative error at their
        zero crossings. Measured on A100/CUDA 12.9: interpolant 2e-6 to 1e-5,
        derivatives 4e-6 across all RHS paths, against float32 eps 1.2e-7."""
        ok = True
        for name, f64, f32 in (
            (
                "interpolant",
                self.compute_gpu_interpolant(stz),
                self.compute_gpu_interpolant(stz, dtype=np.float32),
            ),
            (
                "derivatives",
                self.compute_gpu_derivatives(stz, vpar, vtotal, time=time),
                self.compute_gpu_derivatives(
                    stz, vpar, vtotal, time=time, dtype=np.float32
                ),
            ),
        ):
            # a float64 result here means the float32 inputs were silently
            # upcast by a double-only binding, which is what this guards
            if f32.dtype != np.float32:
                print(f"{name}: float32 call returned {f32.dtype}, bindings upcast")
                ok = False
            scale = np.max(np.abs(f64), axis=0) + 1e-16
            error = np.abs(f64 - f32) / scale
            print(f"max error in {name} precision comparison: {error.max()}")
            if error.max() > tol:
                row_idx = np.unravel_index(np.argmax(error), error.shape)[0]
                print("stz:", stz[row_idx, :])
                print("f64:", f64[row_idx, :])
                print("f32:", f32[row_idx, :])
                ok = False
        return ok

    def compute_gpu_timestep(self, stz, vpar, vtotal, time, psi0):
        if self.field_type == "boozer_vacuum" or self.field_type == "boozer":
            last_time = firm3dpp.test_timestep_boozer(
                quad_pts=self.quad_info,
                srange=self.range0,
                trange=self.range1,
                zrange=self.range2,
                stz_init=stz,
                m=MASS,
                q=CHARGE,
                vtotal=vtotal,
                vtang=vpar,
                tol=1e-9,
                psi0=psi0,
                nparticles=stz.shape[0],
                vacuum=(self.field_type == "boozer_vacuum"),
            )
        elif self.field_type == "boozer_saw_vacuum":
            assert time is not None, (
                "time array must be provided when testing timesteps with SAW"
            )
            last_time = firm3dpp.test_timestep_saw(
                quad_pts=self.quad_info,
                srange=self.range0,
                trange=self.range1,
                zrange=self.range2,
                saw_omega=self.saw_omega,
                saw_srange=self.saw_srange,
                saw_m=self.saw_m,
                saw_n=self.saw_n,
                saw_phihats=self.saw_phihats,
                saw_nharmonics=self.saw_nharmonics,
                stz_init=stz,
                m=MASS,
                q=CHARGE,
                vtotal=vtotal,
                vtang=vpar,
                time=time,
                tol=1e-9,
                psi0=psi0,
                nparticles=stz.shape[0],
            )
        elif self.field_type == "boozer_saw_nok":
            assert time is not None, (
                "time array must be provided when testing timesteps with SAW"
            )
            last_time = firm3dpp.test_timestep_saw_nok(
                quad_pts=self.quad_info,
                srange=self.range0,
                trange=self.range1,
                zrange=self.range2,
                saw_omega=self.saw_omega,
                saw_srange=self.saw_srange,
                saw_m=self.saw_m,
                saw_n=self.saw_n,
                saw_phihats=self.saw_phihats,
                saw_nharmonics=self.saw_nharmonics,
                stz_init=stz,
                m=MASS,
                q=CHARGE,
                vtotal=vtotal,
                vtang=vpar,
                time=time,
                tol=1e-9,
                psi0=psi0,
                nparticles=stz.shape[0],
            )
        elif self.field_type == "cartesian_vacuum":
            last_time = firm3dpp.test_timestep_cartesian(
                quad_pts=self.quad_info,
                rrange=self.range0,
                phirange=self.range1,
                zrange=self.range2,
                loc_init=stz,
                m=MASS,
                q=CHARGE,
                vtotal=vtotal,
                vtang=vpar,
                tol=1e-9,
                nparticles=stz.shape[0],
            )
        else:
            raise ValueError(
                f"GPU timestep computation not implemented for this \
                field type: {self.field_type}"
            )

        last_time = np.reshape(last_time, (stz.shape[0], 5))
        if self.field_type != "cartesian_vacuum":
            # transform to pseudocylindrical coordinates for comparison with CPU results
            last_time = np.array(
                [
                    [x[0], x[1] * np.cos(x[2]), x[1] * np.sin(x[2]), x[3], x[4]]
                    for x in last_time
                ]
            )
        return last_time

    def compute_cpu_timesteps(self, stz, vpar, vtotal, time, psi0):
        if self.field_type == "boozer_vacuum" or self.field_type == "boozer":
            cpu_positions = np.empty((stz.shape[0], 5))
            gc_tys, gc_zeta_hits = trace_particles_boozer(
                self.field,
                stz,
                vpar,
                mass=MASS,
                charge=CHARGE,
                Ekin=ENERGY,
                tol=1e-9,
                stopping_criteria=[IterationStoppingCriterion(1)],
                forget_exact_path=True,
            )
        elif self.field_type in ["boozer_saw_vacuum", "boozer_saw_nok"]:
            cpu_positions = np.empty((stz.shape[0], 5))
            self.field.set_points(stz)
            mu_init = (vtotal**2 - vpar**2) / (2 * self.field.modB()[:, 0])
            gc_tys, gc_zeta_hits = trace_particles_boozer_perturbed(
                self.saw_field,
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
        elif self.field_type == "cartesian_vacuum":
            # convert r, phi, z to x, y, z for CPU tracing
            rphiz = stz
            r = rphiz[:, 0].reshape(-1, 1)
            phi = rphiz[:, 1].reshape(-1, 1)
            z = rphiz[:, 2].reshape(-1, 1)
            x = r * np.cos(phi)
            y = r * np.sin(phi)
            xyz = np.hstack((x, y, z))
            gc_tys, gc_zeta_hits = trace_particles(
                self.field,
                xyz,
                vpar,
                mass=MASS,
                charge=CHARGE,
                Ekin=ENERGY,
                tol=1e-9,
                stopping_criteria=[SimsoptIterationStoppingCriterion(1)],
                forget_exact_path=True,
            )
        else:
            raise ValueError(
                f"CPU timestep computation not implemented for this \
                field type: {self.field_type}"
            )

        cpu_positions = np.array([x[-1] for x in gc_tys])

        if self.field_type != "cartesian_vacuum":
            # transform to pseudocylindrical coordinates for comparison with GPU results
            cpu_positions = np.array(
                [
                    [
                        x[0],
                        x[1] * np.cos(x[2]),
                        x[1] * np.sin(x[2]),
                        np.fmod(x[3], 2 * np.pi) + 2 * np.pi * (x[3] < 0),
                        x[4],
                    ]
                    for x in cpu_positions
                ]
            )
        return cpu_positions

    def test_timestep(self, stz, vpar, vtotal, time, psi0, tol):
        gpu_final_positions = self.compute_gpu_timestep(stz, vpar, vtotal, time, psi0)

        cpu_positions = self.compute_cpu_timesteps(stz, vpar, vtotal, time, psi0)

        gpu_error_is_small = np.allclose(
            gpu_final_positions, cpu_positions, rtol=tol, atol=tol
        )
        error = np.abs(cpu_positions - gpu_final_positions) / (
            np.abs(cpu_positions) + 1
        )
        if not gpu_error_is_small:
            row_idx = np.unravel_index(np.argmax(error), error.shape)[0]
            print("stz:", stz[row_idx, :])
            print("cpu:", cpu_positions[row_idx, :])
            print("gpu:", gpu_final_positions[row_idx, :])
            print("error:", error[row_idx, :])

        return gpu_error_is_small

    def compute_gpu_trajectories(self, stz, vpar, vtotal, tmax, dt_save, psi0):

        if self.field_type == "boozer_vacuum":
            trajectories = save_trajectories_boozer_gpu(
                field=self.field,
                stz_inits=stz.copy(),
                parallel_speeds=vpar.copy(),
                tmax=tmax,
                dt_save=dt_save,
                mass=MASS,
                charge=CHARGE,
                vtotal=vtotal,
                tol=1e-9,
                ns=self.ns,
                ntheta=self.ntheta,
                nzeta=self.nzeta,
            )
        else:
            raise NotImplementedError(
                f"GPU trajectory computation not implemented for \
            this field type: {self.field_type}"
            )
        return trajectories

    def compute_gpu_final_pos(self, stz, vpar, vtotal, tmax, psi0):

        if self.field_type == "boozer_vacuum":
            s = stz[:, 0]
            theta = stz[:, 1]
            x1 = s * np.cos(theta)
            x2 = s * np.sin(theta)
            stz[:, 0] = x1
            stz[:, 1] = x2
            final_pos = firm3dpp.boozer_gpu_tracing(
                quad_pts=self.quad_info,
                srange=self.range0,
                trange=self.range1,
                zrange=self.range2,
                stz_init=stz.copy(),
                m=MASS,
                q=CHARGE,
                vtotal=vtotal,
                vtang=vpar.copy(),
                tmax=[tmax] * stz.shape[0],
                tol=1e-9,
                psi0=psi0,
                dt_in=-np.ones(stz.shape[0]),
                mu_in=-np.ones(stz.shape[0]),
                nparticles=stz.shape[0],
                vacuum=True,
            )
        else:
            raise NotImplementedError(
                f"GPU final position computation not implemented\
             for this field type: {self.field_type}"
            )
        final_pos = np.reshape(final_pos, (stz.shape[0], 7))
        return final_pos

    def test_trajectory_saving(self, stz, vpar, vtotal, tmax, dt_save, psi0):
        gpu_trajectories = self.compute_gpu_trajectories(
            stz, vpar, vtotal, tmax, dt_save, psi0
        )
        final_pos_traj = np.array([trajectory[-1] for trajectory in gpu_trajectories])

        gpu_final_pos = self.compute_gpu_final_pos(stz, vpar, vtotal, tmax, psi0)
        # save_trajectories_boozer_gpu returns Boozer coordinates, while
        # boozer_gpu_tracing returns pseudo-Cartesian (x1, x2): compare in Boozer
        x1, x2 = gpu_final_pos[:, 1].copy(), gpu_final_pos[:, 2].copy()
        gpu_final_pos[:, 1] = np.hypot(x1, x2)
        gpu_final_pos[:, 2] = np.arctan2(x2, x1)

        gpu_error_is_small = np.allclose(
            final_pos_traj[:, 0:5], gpu_final_pos[:, 0:5], rtol=1e-9, atol=1e-9
        )
        error = np.abs(final_pos_traj[:, 0:5] - gpu_final_pos[:, 0:5]) / (
            np.abs(gpu_final_pos[:, 0:5]) + 1
        )

        if not gpu_error_is_small:
            row_idx = np.unravel_index(np.argmax(error), error.shape)[0]
            print("stz:", stz[row_idx, :])
            print("vpar:", vpar[row_idx])
            print("gpu final pos:", gpu_final_pos[row_idx, :])
            print("gpu traj final pos:", final_pos_traj[row_idx, :])
            print("error:", error[row_idx, :])

        return gpu_error_is_small


@unittest.skipUnless(HAS_CUDA, "CUDA support not available")
class TestGPUTracingBoozerVacuum(unittest.TestCase):
    def setUp(self):
        self.n_metagrid_pts = 15
        self.filename = "examples/inputs/boozmn_aten_rescaled_low_res.nc"
        self.vacuum = True
        self.bri, self.field, self.nfp = get_field(
            self.filename, self.n_metagrid_pts, self.vacuum
        )
        self.stz = sample_test_points(n_test_pts)

        self.VELOCITY = np.sqrt(2 * ENERGY / MASS)
        self.vpar_init = np.random.uniform(-self.VELOCITY, self.VELOCITY, (n_test_pts,))

        self.tol = 1e-8

        self.field = CATAPULTField(
            self.field,
            ns=self.n_metagrid_pts,
            ntheta=self.n_metagrid_pts,
            nzeta=self.n_metagrid_pts,
            nfp=self.nfp,
        )

    def test_interpolant(self):
        is_small = self.field.test_interpolant(self.stz, 1e-8)
        self.assertTrue(is_small)

    def test_derivatives(self):
        is_small = self.field.test_derivatives(
            self.stz, self.vpar_init, self.VELOCITY, 1e-8
        )
        self.assertTrue(is_small)

    def test_precision(self):
        self.assertTrue(
            self.field.test_precision(self.stz, self.vpar_init, self.VELOCITY)
        )

    def test_timestep(self):
        is_small = self.field.test_timestep(
            self.stz, self.vpar_init, self.VELOCITY, None, self.field.psi0, 1e-8
        )
        self.assertTrue(is_small)

    def test_trajectory_saving(self):
        tmax = 2e-8
        dt_save = 1e-8

        is_small = self.field.test_trajectory_saving(
            self.stz, self.vpar_init, self.VELOCITY, tmax, dt_save, self.field.psi0
        )
        self.assertTrue(is_small)

    def test_catapult_field(self):
        # a prebuilt CatapultBoozerField traces exactly as a bare field with
        # the resolution passed, and its precision replaces casting the inputs
        field = self.field.field
        res = self.n_metagrid_pts
        stz = self.stz[:200]
        vpar = self.vpar_init[:200]
        kwargs = {
            "tmax": 1e-6,
            "mass": MASS,
            "charge": CHARGE,
            "vtotal": self.VELOCITY,
            "tol": 1e-8,
        }
        bare = advance_particles_boozer_gpu(
            field, stz, vpar, ns=res, ntheta=res, nzeta=res, **kwargs
        )
        cfield = CatapultBoozerField(field, res, res, res)
        np.testing.assert_array_equal(
            advance_particles_boozer_gpu(cfield, stz, vpar, **kwargs), bare
        )

        single = CatapultBoozerField(field, res, res, res, precision="single")
        out = advance_particles_boozer_gpu(single, stz, vpar, **kwargs)
        self.assertEqual(out.dtype, np.float32)
        legacy = advance_particles_boozer_gpu(
            field,
            stz.astype(np.float32),
            vpar.astype(np.float32),
            ns=res,
            ntheta=res,
            nzeta=res,
            **kwargs,
        )
        np.testing.assert_array_equal(out, legacy)

    def test_cpu_format(self):
        # trace_particles_boozer_gpu returns what trace_particles_boozer
        # does: per particle, (t, s, theta, zeta, vpar) rows from t = 0 to
        # the final state, and a (t, -1, s, theta, zeta, vpar) hit when lost
        field = self.field.field
        res = self.n_metagrid_pts
        stz = self.stz[:500]
        vpar = self.vpar_init[:500]
        tmax = 1e-5
        kwargs = {
            "mass": MASS,
            "charge": CHARGE,
            "vtotal": self.VELOCITY,
            "tol": 1e-8,
            "stopping_criteria": [MaxToroidalFluxStoppingCriterion(1.0)],
        }
        cfield = CatapultBoozerField(field, res, res, res)
        state = advance_particles_boozer_gpu(
            cfield, stz, vpar, tmax, MASS, CHARGE, self.VELOCITY, 1e-8
        )
        lost = state[:, 0] < tmax

        res_tys, res_hits = trace_particles_boozer_gpu(
            cfield, stz, vpar, tmax, forget_exact_path=True, **kwargs
        )
        self.assertEqual(len(res_tys), len(stz))
        for i in range(len(stz)):
            self.assertEqual(res_tys[i].shape, (2, 5))
            self.assertEqual(res_tys[i].dtype, np.float64)
            np.testing.assert_array_equal(res_tys[i][0], [0, *stz[i], vpar[i]])
            np.testing.assert_array_equal(res_tys[i][1], state[i, :5])
            if lost[i]:
                self.assertEqual(res_hits[i].shape, (1, 6))
                self.assertEqual(res_hits[i][0, 1], -1)
                np.testing.assert_array_equal(res_hits[i][0, 2:], state[i, 1:5])
            else:
                self.assertEqual(res_hits[i].shape, (0,))
        self.assertTrue(lost.any() and not lost.all())
        # and the CPU post-processing takes it as is
        times, loss_frac = compute_loss_fraction(res_tys, tmin=1e-7, tmax=tmax)
        self.assertAlmostEqual(loss_frac[-1], lost.mean())

        # with trajectories: the same initial row, rows at successive save
        # times, and the same final state
        dt_save = 2e-6
        res_tys, res_hits = trace_particles_boozer_gpu(
            cfield, stz, vpar, tmax, dt_save=dt_save, **kwargs
        )
        for i in range(len(stz)):
            t = res_tys[i][:, 0]
            self.assertEqual(t[0], 0)
            self.assertTrue(np.all(np.diff(t) > 0))
            self.assertEqual(res_hits[i].shape, (1, 6) if lost[i] else (0,))
            if not lost[i]:
                self.assertGreaterEqual(t[-1], tmax)
                self.assertEqual(len(t), 1 + round(tmax / dt_save))

        with self.assertRaises(NotImplementedError):
            trace_particles_boozer_gpu(
                cfield,
                stz,
                vpar,
                tmax,
                stopping_criteria=[MaxToroidalFluxStoppingCriterion(0.9)],
            )


@unittest.skipUnless(HAS_CUDA, "CUDA support not available")
class TestGPUTracingBoozerFiniteBeta(unittest.TestCase):
    def setUp(self):
        self.n_metagrid_pts = 15
        self.filename = "examples/inputs/boozmn_aten_rescaled_low_res.nc"
        self.vacuum = False
        self.bri, self.field, self.nfp = get_field(
            self.filename, self.n_metagrid_pts, self.vacuum
        )
        self.stz = sample_test_points(n_test_pts)

        self.VELOCITY = np.sqrt(2 * ENERGY / MASS)
        self.vpar_init = np.random.uniform(-self.VELOCITY, self.VELOCITY, (n_test_pts,))

        self.tol = 1e-8

        self.field = CATAPULTField(
            self.field,
            ns=self.n_metagrid_pts,
            ntheta=self.n_metagrid_pts,
            nzeta=self.n_metagrid_pts,
            nfp=self.nfp,
        )

    def test_interpolant(self):
        is_small = self.field.test_interpolant(self.stz, 1e-8)
        self.assertTrue(is_small)

    def test_derivatives(self):
        is_small = self.field.test_derivatives(
            self.stz, self.vpar_init, self.VELOCITY, 1e-8
        )
        self.assertTrue(is_small)

    def test_precision(self):
        self.assertTrue(
            self.field.test_precision(self.stz, self.vpar_init, self.VELOCITY)
        )

    def test_timestep(self):
        is_small = self.field.test_timestep(
            self.stz, self.vpar_init, self.VELOCITY, None, self.field.psi0, 1e-8
        )
        self.assertTrue(is_small)


@unittest.skipUnless(HAS_CUDA, "CUDA support not available")
class TestGPUTracingBoozerVacuumSAW(unittest.TestCase):
    def setUp(self):
        self.n_metagrid_pts = 15
        self.filename = "examples/inputs/boozmn_aten_rescaled_low_res.nc"
        self.vacuum = True
        self.bri, self.field, self.nfp = get_field(
            self.filename, self.n_metagrid_pts, self.vacuum
        )
        self.stz = sample_test_points(n_test_pts)

        self.VELOCITY = np.sqrt(2 * ENERGY / MASS)
        self.vpar_init = np.random.uniform(-self.VELOCITY, self.VELOCITY, (n_test_pts,))

        self.time = np.random.uniform(low=0, high=1e-3, size=(n_test_pts,))
        self.tol = 1e-8

        ### set up SAW
        saw_filename = "./examples/tracing_with_AE/ae.npy"
        self.saw = ShearAlfvenWavesSuperposition.from_ae3d(
            eigenvector=AE3DEigenvector.load_from_numpy(
                filename=saw_filename,
            ),
            B0=self.field,
            max_dB_normal_by_B0=5e-3,
            minor_radius_meters=1.7,
        )

        self.field = CATAPULTField(
            self.saw,
            ns=self.n_metagrid_pts,
            ntheta=self.n_metagrid_pts,
            nzeta=self.n_metagrid_pts,
            nfp=self.nfp,
            saw_filename="./examples/tracing_with_AE/ae.npy",
        )

    def test_interpolant(self):
        is_small = self.field.test_interpolant(self.stz, 1e-8)
        self.assertTrue(is_small)

    def test_derivatives(self):
        is_small = self.field.test_derivatives(
            self.stz, self.vpar_init, self.VELOCITY, 1e-8, self.time
        )
        self.assertTrue(is_small)

    def test_precision(self):
        self.assertTrue(
            self.field.test_precision(
                self.stz, self.vpar_init, self.VELOCITY, time=self.time
            )
        )

    def test_timestep(self):
        is_small = self.field.test_timestep(
            self.stz, self.vpar_init, self.VELOCITY, self.time, self.field.psi0, 1e-8
        )
        self.assertTrue(is_small)

    def test_perturbed_tracer(self):
        # tracing through a prebuilt CatapultPerturbedBoozerField matches
        # tracing with the superposition and a resolution, and the given
        # magnetic moments are the ones the kernel uses
        res = self.n_metagrid_pts
        stz = self.stz[:200]
        vpar = self.vpar_init[:200]
        self.saw.B0.set_points(stz)
        mus = (self.VELOCITY**2 - vpar**2) / (2 * self.saw.B0.modB()[:, 0])
        kwargs = {"tmax": 1e-6, "mass": MASS, "charge": CHARGE, "tol": 1e-8}
        bare = advance_particles_boozer_perturbed_gpu(
            self.saw, stz, vpar, mus, ns=res, ntheta=res, nzeta=res, **kwargs
        )
        cfield = CatapultPerturbedBoozerField(self.saw, res, res, res)
        out = advance_particles_boozer_perturbed_gpu(cfield, stz, vpar, mus, **kwargs)
        np.testing.assert_array_equal(out, bare)
        np.testing.assert_array_equal(out[:, 6], mus)


@unittest.skipUnless(HAS_CUDA, "CUDA support not available")
class TestGPUTracingBoozerNoKSAW(unittest.TestCase):
    def setUp(self):
        self.n_metagrid_pts = 15
        self.filename = "examples/inputs/boozmn_aten_rescaled_low_res.nc"
        self.vacuum = False
        self.bri, self.field, self.nfp = get_field(
            self.filename, self.n_metagrid_pts, self.vacuum
        )
        self.stz = sample_test_points(n_test_pts)

        self.VELOCITY = np.sqrt(2 * ENERGY / MASS)
        self.vpar_init = np.random.uniform(-self.VELOCITY, self.VELOCITY, (n_test_pts,))

        self.time = np.random.uniform(low=0, high=1e-3, size=(n_test_pts,))
        self.tol = 1e-8

        ### set up SAW
        saw_filename = "./examples/tracing_with_AE/ae.npy"
        self.saw = ShearAlfvenWavesSuperposition.from_ae3d(
            eigenvector=AE3DEigenvector.load_from_numpy(
                filename=saw_filename,
            ),
            B0=self.field,
            max_dB_normal_by_B0=5e-3,
            minor_radius_meters=1.7,
        )

        self.field = CATAPULTField(
            self.saw,
            ns=self.n_metagrid_pts,
            ntheta=self.n_metagrid_pts,
            nzeta=self.n_metagrid_pts,
            nfp=self.nfp,
            saw_filename="./examples/tracing_with_AE/ae.npy",
        )

    def test_interpolant(self):
        is_small = self.field.test_interpolant(self.stz, 1e-8)
        self.assertTrue(is_small)

    def test_derivatives(self):
        is_small = self.field.test_derivatives(
            self.stz, self.vpar_init, self.VELOCITY, 1e-8, self.time
        )
        self.assertTrue(is_small)

    def test_precision(self):
        self.assertTrue(
            self.field.test_precision(
                self.stz, self.vpar_init, self.VELOCITY, time=self.time
            )
        )

    def test_timestep(self):
        is_small = self.field.test_timestep(
            self.stz, self.vpar_init, self.VELOCITY, self.time, self.field.psi0, 1e-8
        )
        self.assertTrue(is_small)


@unittest.skipUnless(HAS_SIMSOPT and HAS_CUDA, "simsopt or CUDA not available")
class TestGPUTracingCartesian(unittest.TestCase):
    def setUp(self):
        degree = 3  # degree of interpolant
        self.n_metagrid_pts = 16  # resolution of interpolant
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

        self.range0 = (np.min(rs), np.max(rs), self.n_metagrid_pts)
        self.range1 = (0, 2 * np.pi / surf.nfp, self.n_metagrid_pts * 2)
        # exploit stellarator symmetry and only consider positive z values:
        self.range2 = (0, np.max(zs), self.n_metagrid_pts // 2)
        bsh = InterpolatedField(
            bs,
            degree,
            self.range0,
            self.range1,
            self.range2,
            True,
            nfp=surf.nfp,
            stellsym=True,
        )

        ### rejection sample points inside the loss surface
        rphiz = np.empty((n_test_pts, 3))
        for i in range(n_test_pts):
            pt = np.random.uniform(low=0, high=1, size=(1, 3))
            pt[0, 0] = pt[0, 0] * (self.range0[1] - self.range0[0]) + self.range0[0]
            pt[0, 1] *= 2 * np.pi
            pt[0, 2] = (pt[0, 2] - 0.5) * 2 * self.range2[1]

            # particle is outside the surface or too close to the surface
            max_iters = 1000
            for _ in range(max_iters):
                if sc_particle.evaluate_rphiz(pt) > 0.2:
                    break
                pt = np.random.uniform(low=0, high=1, size=(1, 3))
                pt[0, 0] = pt[0, 0] * (self.range0[1] - self.range0[0]) + self.range0[0]
                pt[0, 1] *= 2 * np.pi
                pt[0, 2] = (pt[0, 2] - 0.5) * 2 * self.range2[1]
            else:
                raise RuntimeError("Could not sample a valid point inside the surface")
            rphiz[i, :] = pt

        self.stz = rphiz

        self.field = CATAPULTField(
            bsh,
            ns=self.n_metagrid_pts,
            ntheta=self.n_metagrid_pts * 2,
            nzeta=self.n_metagrid_pts // 2,
            nfp=surf.nfp,
            sc_classifier=sc_particle,
        )

    def test_interpolant(self):
        is_small = self.field.test_interpolant(self.stz, tol=1e-8)
        self.assertTrue(is_small)

    def test_derivatives(self):
        VELOCITY = np.sqrt(2 * ENERGY / MASS)
        vpar_init = np.random.uniform(-VELOCITY, VELOCITY, (n_test_pts,))
        is_small = self.field.test_derivatives(self.stz, vpar_init, VELOCITY, tol=1e-8)
        self.assertTrue(is_small)

    def test_precision(self):
        VELOCITY = np.sqrt(2 * ENERGY / MASS)
        vpar_init = np.random.uniform(-VELOCITY, VELOCITY, (n_test_pts,))
        self.assertTrue(self.field.test_precision(self.stz, vpar_init, VELOCITY))

    def test_timestep(self):
        VELOCITY = np.sqrt(2 * ENERGY / MASS)
        vpar_init = np.random.uniform(-VELOCITY, VELOCITY, (n_test_pts,))
        is_small = self.field.test_timestep(
            self.stz, vpar_init, VELOCITY, None, 0, tol=1e-8
        )
        self.assertTrue(is_small)


if __name__ == "__main__":
    print("Running GPU tracing tests...")
    unittest.main()
