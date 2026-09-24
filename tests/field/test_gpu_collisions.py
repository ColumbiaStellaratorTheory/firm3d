"""
GPU Coulomb collision tests.

These live apart from test_gpu.py because they need fields the collisionless
tests do not: an equilibrium and a coil set built from the same wout file, so
that a Boozer trace and a Cartesian trace can be compared particle for
particle, and a flux label tabulated alongside the Cartesian field.
"""

import functools
import unittest

import numpy as np
import firm3dpp

try:
    from simsopt.geo import SurfaceRZFourier

    from simsopt.field import (
        BiotSavart,
        InterpolatedField,
        SurfaceClassifier,
        coils_via_symmetries,
        load_coils_from_makegrid_file,
    )

    HAS_SIMSOPT = True
except Exception:
    HAS_SIMSOPT = False

from firm3d.catapult.field import CatapultBoozerField, CatapultCartesianField
from firm3d.catapult.tracing import (
    trace_particles_boozer_gpu,
    trace_particles_boozer_with_collisions_gpu,
    trace_particles_cartesian_gpu,
    trace_particles_cartesian_with_collisions_gpu,
)
from firm3d.field.boozermagneticfield import (
    BoozerRadialInterpolant,
    InterpolatedBoozerField,
)
from firm3d.field.collisions import (
    ThermalBackground,
    trace_particles_boozer_with_collisions,
)
from firm3d.util.constants import (
    ALPHA_PARTICLE_CHARGE as CHARGE,
)
from firm3d.util.constants import (
    ALPHA_PARTICLE_MASS as MASS,
)
from firm3d.util.constants import (
    ELECTRON_MASS,
    ELEMENTARY_CHARGE,
    PROTON_MASS,
)
from firm3d.util.constants import (
    FUSION_ALPHA_PARTICLE_ENERGY as ENERGY,
)

HAS_CUDA = hasattr(firm3dpp, "test_gpu_interpolation")

RES = 15  # interpolant resolution shared by the tests


def final_states(res_tys):
    """The last row of each trajectory, (t, x1, x2, x3, vpar)."""
    return np.array([traj[-1] for traj in res_tys])


@functools.lru_cache(maxsize=1)
def build_wout_boozer_field():
    """
    Boozer field for the cross-coordinate tests, built from the same wout
    file as the coil surface; the boozmn files in examples/inputs are
    differently rescaled equilibria whose boundaries do not match the coils.
    """
    bri = BoozerRadialInterpolant(
        "examples/inputs/wout_aten_rescaled.nc", 3, enforce_vacuum=True
    )
    bfield = InterpolatedBoozerField(
        bri, 3, ns_interp=RES, ntheta_interp=RES, nzeta_interp=RES
    )
    return bri, bfield, bri.nfp


@functools.lru_cache(maxsize=1)
def build_cartesian_field():
    """
    Coil field, boundary classifier, and grid ranges for the Cartesian tests.
    """
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

    # coils.curves_22_7_21 holds the stellarator-symmetric half of a full-torus
    # 40-coil set, so only stellsym is applied here.
    coils_full = coils_via_symmetries(curves, currents, 1, True)
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
    return surf, sc_particle, bsh, rrange, phirange, zrange


def sample_rphiz_inside(nparticles, rrange, zrange, sc_particle, lo=0.2, hi=np.inf):
    """
    Rejection sample cylindrical points whose signed distance to the plasma
    boundary lies in (lo, hi); the interior is positive.  The band form lets
    the collisional tests place ensembles at chosen depths in the flux label.
    """
    rphiz = np.empty((nparticles, 3))
    for i in range(nparticles):
        pt = np.random.uniform(low=0, high=1, size=(1, 3))
        pt[0, 0] = pt[0, 0] * (rrange[1] - rrange[0]) + rrange[0]
        pt[0, 1] *= 2 * np.pi
        pt[0, 2] = (pt[0, 2] - 0.5) * 2 * zrange[1]

        # particle is outside the surface or too close to the surface
        max_iters = 1000
        for _ in range(max_iters):
            if lo < sc_particle.evaluate_rphiz(pt) < hi:
                break
            pt = np.random.uniform(low=0, high=1, size=(1, 3))
            pt[0, 0] = pt[0, 0] * (rrange[1] - rrange[0]) + rrange[0]
            pt[0, 1] *= 2 * np.pi
            pt[0, 2] = (pt[0, 2] - 0.5) * 2 * zrange[1]
        else:
            raise RuntimeError("Could not sample a valid point inside the surface")
        rphiz[i, :] = pt
    return rphiz


def rphiz_to_xyz(rphiz):
    xyz = np.empty_like(rphiz)
    xyz[:, 0] = rphiz[:, 0] * np.cos(rphiz[:, 1])
    xyz[:, 1] = rphiz[:, 0] * np.sin(rphiz[:, 1])
    xyz[:, 2] = rphiz[:, 2]
    return xyz


def distance_flux_label(sc_particle, scale=0.3):
    """
    Flux-label stand-in built from the signed boundary distance d:
    s = 1 - d/scale, clipped to [0, 2].  It is 1 on the boundary, falls
    toward the core, and exceeds 1 outside, mimicking a normalized flux
    without needing an equilibrium; the coil-field tests use it because a
    Biot-Savart field carries no flux surfaces of its own.
    """

    def label(points_rphiz):
        d = np.asarray(sc_particle.evaluate_rphiz(points_rphiz)).reshape(-1)
        return np.clip(1.0 - d / scale, 0.0, 2.0)

    return label


def equilibrium_flux_label(bfield, nfp, n_s=48, n_theta=48, n_zeta_per_period=48):
    """
    Flux label s(r, phi, z) built by forward-mapping a dense Boozer grid
    through the equilibrium and answering queries with the nearest mapped
    point.
    """
    from scipy.spatial import cKDTree

    from firm3d.field.coordinates import boozer_to_cylindrical

    s = np.linspace(0.02, 1.0, n_s)
    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    zeta = np.linspace(0, 2 * np.pi, nfp * n_zeta_per_period, endpoint=False)
    grid = np.array(np.meshgrid(s, theta, zeta, indexing="ij")).reshape(3, -1).T
    samples_xyz = rphiz_to_xyz(boozer_to_cylindrical(bfield, grid))
    tree = cKDTree(samples_xyz)
    s_samples = grid[:, 0]

    def label(points_rphiz):
        _, idx = tree.query(rphiz_to_xyz(np.asarray(points_rphiz)))
        return s_samples[idx]

    return label


@functools.lru_cache(maxsize=1)
def build_equilibrium_label():
    """Flux label for the wout-built equilibrium, cached across tests."""
    bri, bfield, nfp = build_wout_boozer_field()
    return equilibrium_flux_label(bfield, nfp)


@functools.lru_cache(maxsize=1)
def build_boozmn_fields():
    """
    The low-resolution vacuum equilibrium the Boozer-only tests trace in, as
    the CPU tracer wants it and as CATAPULT wants it.
    """
    bri = BoozerRadialInterpolant(
        "examples/inputs/boozmn_aten_rescaled_low_res.nc", 3, enforce_vacuum=True
    )
    field = InterpolatedBoozerField(
        bri, 3, ns_interp=RES, ntheta_interp=RES, nzeta_interp=RES
    )
    return field, CatapultBoozerField(field, RES, RES, RES)


def build_boozmn_catapult_field():
    return build_boozmn_fields()[1]


def zero_background():
    """A background whose every collision coefficient vanishes."""
    return ThermalBackground(
        n_profile=lambda s: 0.0,
        T_profile=lambda s: 1e3,
        mass=2 * PROTON_MASS,
        charge=ELEMENTARY_CHARGE,
    )


@unittest.skipUnless(HAS_CUDA, "CUDA support not available")
class TestGPUCollisionsBoozer(unittest.TestCase):
    def test_zero_density_reproduces_the_collisionless_path(self):
        """
        A zero-density background makes every collision coefficient zero, so
        the collisional entry point must return the collisionless answer.
        """
        cfield = build_boozmn_catapult_field()
        VELOCITY = np.sqrt(2 * ENERGY / MASS)
        n = 256
        rng = np.random.default_rng(0)
        stz = np.column_stack(
            [
                np.full(n, 0.3),
                rng.uniform(0, 2 * np.pi, n),
                rng.uniform(0, 2 * np.pi, n),
            ]
        )
        vpar = 0.5 * VELOCITY * np.ones(n)
        kw = {
            "tmax": 2e-6,
            "mass": MASS,
            "charge": CHARGE,
            "Ekin": ENERGY,
            "tol": 1e-8,
        }
        res_tys, _ = trace_particles_boozer_gpu(
            cfield, stz.copy(), vpar, forget_exact_path=True, **kw
        )
        without = final_states(res_tys)
        no_kick = trace_particles_boozer_with_collisions_gpu(
            cfield, stz.copy(), vpar, backgrounds=zero_background(), rng_seed=0, **kw
        )

        self.assertEqual(no_kick.shape, (n, 7))
        differing = np.mean(np.abs(no_kick[:, 4] - without[:, 4]) > 1e-6 * VELOCITY)
        self.assertLess(
            differing,
            0.1,
            f"{differing:.2f} of particles differ from the collisionless run "
            f"with every coefficient zero; the kick is not a no-op there",
        )
        np.testing.assert_allclose(
            no_kick[:, 5],
            VELOCITY,
            rtol=1e-3,
            err_msg=(
                "zero-density run does not return the launch speed in column "
                "5; the speed column is not what is being written"
            ),
        )

    def _collisional_ensemble(self, background, Ekin, xi0, tmax, n=256, seed=0):
        """
        Trace n particles with collisions on the GPU; return their final
        (xi, v).  Shared by the two physics tests below, which differ only in
        the background and the launch pitch.
        """
        cfield = build_boozmn_catapult_field()
        vtotal = np.sqrt(2 * Ekin / MASS)
        rng = np.random.default_rng(seed)
        stz = np.column_stack(
            [
                np.full(n, 0.3),
                rng.uniform(0, 2 * np.pi, n),
                rng.uniform(0, 2 * np.pi, n),
            ]
        )
        out = trace_particles_boozer_with_collisions_gpu(
            cfield,
            stz,
            xi0 * vtotal * np.ones(n),
            backgrounds=background,
            tmax=tmax,
            mass=MASS,
            charge=CHARGE,
            Ekin=Ekin,
            tol=1e-8,
            rng_seed=seed,
        )
        self.assertTrue(np.all(np.isfinite(out)), "non-finite GPU results")
        np.testing.assert_allclose(out[:, 0], tmax, rtol=1e-12)
        v, vpar = out[:, 5], out[:, 4]
        self.assertTrue(
            np.all(v >= np.abs(vpar) - 1e-6 * vtotal),
            "speed is below |v_par|, so the recovered mu would be negative",
        )
        return vpar / v, v

    def test_collisions_isotropize_the_pitch(self):
        """
        An ion background scatters pitch: an ensemble launched at xi = 0.9
        must relax toward isotropy, <xi> -> 0 and <xi^2> -> 1/3.
        """
        deuterium = ThermalBackground(
            n_profile=lambda s: 1e25,
            T_profile=lambda s: 1e3,
            mass=2 * PROTON_MASS,
            charge=ELEMENTARY_CHARGE,
        )
        # 0.1 v_alpha, so the same energy the original test launched at
        xi, _ = self._collisional_ensemble(deuterium, 0.01 * ENERGY, xi0=0.9, tmax=2e-6)
        mean_xi, mean_xi2 = np.mean(xi), np.mean(xi**2)
        # Targets are the stationary values of the pitch SDE, whose
        # distribution is xi ~ U(-1, 1): <xi> = 0 and <xi^2> = 1/3
        self.assertLess(abs(mean_xi), 0.15, f"<xi> = {mean_xi:.3f}, launched at 0.9")
        self.assertGreater(mean_xi2, 0.24, f"<xi^2> = {mean_xi2:.3f}, isotropic is 1/3")
        self.assertLess(mean_xi2, 0.43, f"<xi^2> = {mean_xi2:.3f}, isotropic is 1/3")

    def test_cpu_and_gpu_agree_at_tmax(self):
        r"""
        The CPU and the GPU must stop a collisional trace at the same time and
        in the same state distribution.

        Regression test for the endpoint asymmetry reported on PR #67, where
        one tracer applied the collision kick over the step that reaches tmax
        and the other did not. It has since broken in both directions: the CPU
        once reported the pre-kick interpolated state, and the GPU once ran a
        full step past tmax, so both halves are checked here.

        tmax is a tenth of one orbit step, which makes the whole trace an
        endpoint test -- a tracer that mishandles the final step has nowhere
        to hide. The failure modes are far outside the tolerances below: an
        uncapped GPU step carries the ensemble to <v>/v0 ~ 0.19 rather than
        ~0.92, and a missing kick leaves it at exactly 1. Across CPU seeds the
        mean holds to 4e-4 and the KS statistic to 0.012, so the tolerances
        are loose by more than an order of magnitude either way.
        """
        from scipy.stats import ks_2samp

        field, cfield = build_boozmn_fields()
        Ekin = 0.01 * ENERGY  # 0.1 v_alpha
        v0 = np.sqrt(2 * Ekin / MASS)
        n = 256
        rng = np.random.default_rng(0)
        stz = np.column_stack(
            [
                np.full(n, 0.3),
                rng.uniform(0, 2 * np.pi, n),
                rng.uniform(0, 2 * np.pi, n),
            ]
        )
        vpar = 0.5 * v0 * np.ones(n)
        # dense and cold, so one step of collisional evolution is unmissable
        deuterium = ThermalBackground(
            n_profile=lambda s: 1e25,
            T_profile=lambda s: 1e3,
            mass=2 * PROTON_MASS,
            charge=ELEMENTARY_CHARGE,
        )
        tmax = 1e-8
        kw = {
            "backgrounds": deuterium,
            "tmax": tmax,
            "mass": MASS,
            "charge": CHARGE,
            "Ekin": Ekin,
            "tol": 1e-8,
            "rng_seed": 0,
        }

        res_tys, _ = trace_particles_boozer_with_collisions(
            field, stz.copy(), vpar.copy(), **kw
        )
        cpu = np.array([traj[-1] for traj in res_tys])
        gpu = trace_particles_boozer_with_collisions_gpu(
            cfield, stz.copy(), vpar.copy(), **kw
        )

        for who, t in (("CPU", cpu[:, 0]), ("GPU", gpu[:, 0])):
            np.testing.assert_allclose(
                t,
                tmax,
                rtol=1e-12,
                err_msg=(
                    f"{who} did not stop at tmax; the kick window then covers "
                    f"a different interval than the other tracer's"
                ),
            )

        # the kick must reach the step that ends the trace, on both sides
        for who, v in (("CPU", cpu[:, 5]), ("GPU", gpu[:, 5])):
            moved = np.mean(np.abs(v - v0) > 1e-9 * v0)
            self.assertGreater(
                moved,
                0.99,
                f"only {moved:.3f} of the {who} ensemble changed speed; the "
                f"final step is being reported before its kick",
            )

        v_cpu, v_gpu = cpu[:, 5], gpu[:, 5]
        d_mean = abs(np.mean(v_cpu) - np.mean(v_gpu)) / v0
        self.assertLess(
            d_mean,
            0.01,
            f"<v>/v0 disagrees by {d_mean:.4f}: CPU {np.mean(v_cpu) / v0:.4f} "
            f"vs GPU {np.mean(v_gpu) / v0:.4f}",
        )
        d_std = abs(np.std(v_cpu) - np.std(v_gpu)) / np.std(v_cpu)
        self.assertLess(
            d_std,
            0.10,
            f"the speed spread disagrees by {d_std:.3f} relative; the two "
            f"tracers are diffusing over different windows",
        )
        ks = ks_2samp(v_cpu, v_gpu).statistic
        self.assertLess(ks, 0.15, f"speed distributions disagree at tmax (KS {ks:.3f})")

    def test_electron_drag_slows_without_scattering_pitch(self):
        """
        An electron background drags but barely scatters: <v> must fall while
        <xi> stays where it was launched.
        """
        electrons = ThermalBackground(
            n_profile=lambda s: 1e25,
            T_profile=lambda s: 10e3,
            mass=ELECTRON_MASS,
            charge=-ELEMENTARY_CHARGE,
        )
        v0 = np.sqrt(2 * ENERGY / MASS)
        xi, v = self._collisional_ensemble(electrons, ENERGY, xi0=0.5, tmax=1e-6)
        ratio = np.mean(v) / v0
        # Integrating dv/dt = K(v) over tmax gives v/v0 = 0.825
        self.assertGreater(ratio, 0.80, f"<v>/v0 = {ratio:.3f}; too little drag")
        self.assertLess(ratio, 0.87, f"<v>/v0 = {ratio:.3f}; too much drag")
        self.assertAlmostEqual(
            np.mean(xi),
            0.5,
            delta=0.05,
            msg=f"<xi> = {np.mean(xi):.3f}; electrons should barely scatter pitch",
        )


@unittest.skipUnless(HAS_SIMSOPT and HAS_CUDA, "simsopt or CUDA not available")
class TestGPUCollisionsCartesian(unittest.TestCase):
    def test_zero_density_matches_the_collisionless_cartesian_path(self):
        """
        A zero-density background makes every collision coefficient zero, so
        the collisional tracer must reproduce the collisionless one.
        """
        np.random.seed(0)
        surf, sc_particle, bsh, rrange, phirange, zrange = build_cartesian_field()

        n = 128
        # Launch deep so the ensemble reaches tmax rather than the boundary.
        rphiz = sample_rphiz_inside(n, rrange, zrange, sc_particle, lo=0.2)
        xyz = rphiz_to_xyz(rphiz)

        VELOCITY = np.sqrt(2 * ENERGY / MASS)
        vpar = 0.5 * VELOCITY * np.ones(n)
        kw = {
            "tmax": 1e-6,
            "mass": MASS,
            "charge": CHARGE,
            "Ekin": ENERGY,
            "tol": 1e-8,
        }

        plain = CatapultCartesianField(bsh, sc_particle)
        labelled = CatapultCartesianField(
            bsh, sc_particle, flux_label=distance_flux_label(sc_particle)
        )
        res_tys, _ = trace_particles_cartesian_gpu(
            plain, xyz.copy(), vpar, forget_exact_path=True, **kw
        )
        without = final_states(res_tys)
        no_kick = trace_particles_cartesian_with_collisions_gpu(
            labelled,
            xyz.copy(),
            vpar,
            backgrounds=zero_background(),
            rng_seed=0,
            **kw,
        )

        self.assertEqual(no_kick.shape, (n, 7))
        self.assertTrue(np.all(np.isfinite(no_kick)), "non-finite GPU results")

        done = np.isclose(without[:, 0], kw["tmax"], rtol=1e-12) & np.isclose(
            no_kick[:, 0], kw["tmax"], rtol=1e-12
        )
        self.assertGreater(np.mean(done), 0.9, "too many particles lost to compare")

        scale = np.linalg.norm(xyz, axis=1)[done]
        same_position = (
            np.linalg.norm(no_kick[done, 1:4] - without[done, 1:4], axis=1)
            < 1e-6 * scale
        )
        same_vpar = np.abs(no_kick[done, 4] - without[done, 4]) < 1e-6 * VELOCITY
        agree = np.mean(same_position & same_vpar)
        self.assertGreater(
            agree,
            0.9,
            f"only {agree:.2f} of the zero-density ensemble matches the "
            f"collisionless run; the flux-label column is perturbing the orbit",
        )

        # v >= |v_par| is required for mu = (v^2 - v_par^2)/(2|B|) >= 0.
        self.assertTrue(
            np.all(no_kick[:, 5] >= np.abs(no_kick[:, 4]) - 1e-6 * VELOCITY),
            "speed is below |v_par|, so the recovered mu would be negative",
        )

    def test_cartesian_collisions_read_the_flux_label(self):
        """
        A background confined to small flux-label values must leave particles
        at large label values collisionless: the end-to-end check that the
        label is interpolated at the particle position and fed to the kick.

        Launching at 0.1 v_alpha keeps drift orbits within millimeters of
        their surface; at full energy the shallow ensemble grazes the dense
        zone.  The shallow group is compared against a zero-density control
        run at the same seed.
        """
        np.random.seed(0)
        surf, sc_particle, bsh, rrange, phirange, zrange = build_cartesian_field()

        # The density cut at s = 0.5 sits at boundary distance d = 0.15,
        # between the deep band (collides) and the shallow band (must not).
        label = distance_flux_label(sc_particle, scale=0.3)
        cfield = CatapultCartesianField(bsh, sc_particle, flux_label=label)
        n_deep, n_shallow = 64, 64
        deep = sample_rphiz_inside(n_deep, rrange, zrange, sc_particle, lo=0.2)
        shallow = sample_rphiz_inside(
            n_shallow, rrange, zrange, sc_particle, lo=0.04, hi=0.10
        )
        xyz = rphiz_to_xyz(np.vstack((deep, shallow)))
        n = n_deep + n_shallow

        bg = ThermalBackground(
            n_profile=lambda s: 1e21 if s < 0.5 else 0.0,
            T_profile=lambda s: 1e3,
            mass=2 * PROTON_MASS,
            charge=ELEMENTARY_CHARGE,
        )

        Ekin = 0.01 * ENERGY  # 0.1 v_alpha
        vtotal = np.sqrt(2 * Ekin / MASS)
        kw = {
            "xyz_inits": xyz,
            "parallel_speeds": 0.5 * vtotal * np.ones(n),
            "tmax": 1e-6,
            "mass": MASS,
            "charge": CHARGE,
            "Ekin": Ekin,
            "tol": 1e-8,
            "rng_seed": 0,
        }
        out = trace_particles_cartesian_with_collisions_gpu(
            cfield, backgrounds=bg, **kw
        )
        control = trace_particles_cartesian_with_collisions_gpu(
            cfield, backgrounds=zero_background(), **kw
        )
        self.assertTrue(np.all(np.isfinite(out)), "non-finite GPU results")

        moved = np.mean(np.abs(out[:n_deep, 5] - vtotal) > 1e-6 * vtotal)
        self.assertGreater(
            moved,
            0.9,
            f"only {moved:.2f} of the deep ensemble changed speed; the kick "
            f"is not seeing the dense region of the profile",
        )
        row_identical = np.all(
            np.isclose(out[n_deep:, :], control[n_deep:, :], rtol=1e-12, atol=0.0),
            axis=1,
        )
        self.assertGreater(
            np.mean(row_identical),
            0.9,
            f"only {np.mean(row_identical):.2f} of the zero-density ensemble "
            f"matches the zero-density control run; the flux label reaching "
            f"the kick is wrong",
        )

    def test_cross_coordinate_alpha_relaxation(self):
        """
        The same alpha ensemble in the same ATEN configuration must relax the
        same way whether traced in Boozer coordinates (equilibrium field; the
        state carries s) or Cartesian coordinates (coil field; s arrives
        through the interpolated flux label).

        Full-energy alphas in a DT + electron background relax in two stages:
        electron drag slows them with almost no pitch scattering, then ion
        scattering isotropizes the pitch as the speed approaches the critical
        velocity.

        A third trace with a deliberately constant label is the control: it
        must miss the Boozer answer by far more than the two tracers miss
        each other, which is what makes their agreement evidence rather than
        insensitivity.
        """
        from scipy.stats import ks_2samp

        from firm3d.field.coordinates import boozer_to_cylindrical

        bri, bfield, nfp = build_wout_boozer_field()
        surf, sc_particle, bsh, rrange, phirange, zrange = build_cartesian_field()
        label = build_equilibrium_label()

        cbooz = CatapultBoozerField(bfield, RES, RES, RES)
        ccart = CatapultCartesianField(bsh, sc_particle, flux_label=label)
        cwrong = CatapultCartesianField(
            bsh, sc_particle, flux_label=lambda pts: np.zeros(len(pts))
        )

        # 50/50 DT with electrons, reactor profile shapes, density boosted
        # 500x so full slowing down fits in a sub-millisecond trace.
        def ne(s):
            return 5e22 * (1.0 - 0.8 * s**2)

        def Te(s):
            return 10e3 * (1.0 - 0.8 * s) + 1e3

        bgs = [
            ThermalBackground(
                n_profile=lambda s: 0.5 * ne(s),
                T_profile=Te,
                mass=2 * PROTON_MASS,
                charge=ELEMENTARY_CHARGE,
            ),
            ThermalBackground(
                n_profile=lambda s: 0.5 * ne(s),
                T_profile=Te,
                mass=3 * PROTON_MASS,
                charge=ELEMENTARY_CHARGE,
            ),
            ThermalBackground(
                n_profile=ne,
                T_profile=Te,
                mass=ELECTRON_MASS,
                charge=-ELEMENTARY_CHARGE,
            ),
        ]

        n = 512
        xi0 = 0.9
        v0 = np.sqrt(2 * ENERGY / MASS)
        rng = np.random.default_rng(0)
        stz = np.column_stack(
            [
                rng.uniform(0.05, 0.6, n),
                rng.uniform(0, 2 * np.pi, n),
                rng.uniform(0, 2 * np.pi, n),
            ]
        )
        xyz = rphiz_to_xyz(boozer_to_cylindrical(bfield, stz.copy()))
        vpar = xi0 * v0 * np.ones(n)
        kw = {
            "backgrounds": bgs,
            "mass": MASS,
            "charge": CHARGE,
            "Ekin": ENERGY,
            "tol": 1e-8,
            "rng_seed": 0,
        }

        def trace_boozer(tmax):
            return trace_particles_boozer_with_collisions_gpu(
                cbooz, stz.copy(), vpar.copy(), tmax=tmax, **kw
            )

        def trace_cartesian(tmax, cf):
            return trace_particles_cartesian_with_collisions_gpu(
                cf, xyz.copy(), vpar.copy(), tmax=tmax, **kw
            )

        def confined(out, tmax, who):
            mask = np.isclose(out[:, 0], tmax, rtol=1e-12)
            self.assertGreater(np.mean(mask), 0.95, f"{who} losses at {tmax:g}")
            return mask

        def moments(out, mask):
            v = out[mask, 5]
            return np.mean(v**2) / v0**2, np.mean(out[mask, 4] / v)

        # Drag stage: a quarter of the energy is gone, the pitch is not.
        t_early = 1e-4
        booz = trace_boozer(t_early)
        cart = trace_cartesian(t_early, ccart)
        E_b, xi_b = moments(booz, confined(booz, t_early, "Boozer"))
        E_c, xi_c = moments(cart, confined(cart, t_early, "Cartesian"))

        self.assertLess(E_b, 0.85, "drag has not started; the regime is wrong")
        self.assertLess(
            abs(E_b - E_c),
            0.02,
            f"<E>/E0 disagrees in the drag stage: {E_b:.4f} vs {E_c:.4f}",
        )
        for who, xi in (("Boozer", xi_b), ("Cartesian", xi_c)):
            self.assertGreater(
                xi,
                xi0 - 0.03,
                f"pitch scattered during the drag stage ({who} <xi>={xi:.3f}); "
                f"electron drag must not scatter pitch",
            )

        # Scattering stage: near the critical energy the pitch must have
        # decayed, by the same amount in both tracers.
        t_late = 6e-4
        booz = trace_boozer(t_late)
        cart = trace_cartesian(t_late, ccart)
        wrong = trace_cartesian(t_late, cwrong)
        mask_b = confined(booz, t_late, "Boozer")
        mask_c = confined(cart, t_late, "Cartesian")
        mask_w = confined(wrong, t_late, "control")
        E_b, xi_b = moments(booz, mask_b)
        E_c, xi_c = moments(cart, mask_c)
        E_w, _ = moments(wrong, mask_w)

        self.assertLess(
            abs(E_b - E_c),
            0.02,
            f"<E>/E0 disagrees in the scattering stage: {E_b:.4f} vs {E_c:.4f}",
        )
        for who, xi in (("Boozer", xi_b), ("Cartesian", xi_c)):
            self.assertLess(
                xi,
                xi0 - 0.03,
                f"pitch did not decay near the critical energy ({who} <xi>={xi:.3f})",
            )
        self.assertLess(
            abs(xi_b - xi_c),
            0.05,
            f"isotropization rates disagree: <xi> {xi_b:.3f} vs {xi_c:.3f}",
        )
        ks = ks_2samp(booz[mask_b, 5], cart[mask_c, 5]).statistic
        self.assertLess(
            ks, 0.10, f"confined speed distributions disagree (KS {ks:.3f})"
        )
        self.assertGreater(
            abs(E_b - E_w),
            0.04,
            f"a constant flux label of 0 gives <E>/E0 = {E_w:.4f} against the "
            f"Boozer {E_b:.4f}; the agreement above is insensitive to the "
            f"label and proves nothing",
        )

        # The ensembles must also sit on the same surfaces, so the agreement
        # above is not two errors cancelling.
        final_rphiz = np.column_stack(
            [
                np.hypot(cart[mask_c, 1], cart[mask_c, 2]),
                np.arctan2(cart[mask_c, 2], cart[mask_c, 1]),
                cart[mask_c, 3],
            ]
        )
        s_cart = np.mean(label(final_rphiz))
        s_booz = np.mean(booz[mask_b, 1])
        self.assertLess(
            abs(s_booz - s_cart),
            0.05,
            f"mean flux label disagrees: Boozer {s_booz:.3f}, Cartesian {s_cart:.3f}",
        )


if __name__ == "__main__":
    unittest.main()
