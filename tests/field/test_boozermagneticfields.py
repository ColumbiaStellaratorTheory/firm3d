import functools
import unittest
from pathlib import Path

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.io import netcdf_file
from scipy.optimize import minimize

from firm3d._core.util import align_and_pad, allocate_aligned_and_padded_array
from firm3d.field.boozermagneticfield import (
    BoozerAnalytic,
    BoozerRadialInterpolant,
    BoozerSplineField,
    InterpolatedBoozerField,
    InterpolatedShearAlfvenWave,
    ShearAlfvenHarmonic,
    ShearAlfvenWavesSuperposition,
)
from firm3d.saw.ae3d import AE3DEigenvector
from firm3dpp import (
    RegularGridInterpolant3D,
    UniformInterpolationRule,
    inverse_fourier_transform_even,
    inverse_fourier_transform_odd,
)

TEST_DIR = (Path(__file__).parent / ".." / "test_files").resolve()
filename_vac = str((TEST_DIR / "boozmn_LandremanPaul2021_QA_lowres.nc").resolve())
filename_vac_wout = str((TEST_DIR / "wout_LandremanPaul2021_QA_lowres.nc").resolve())

filename_mhd = str((TEST_DIR / "boozmn_n3are_R7.75B5.7.nc").resolve())
filename_mhd_wout = str((TEST_DIR / "wout_n3are_R7.75B5.7.nc").resolve())
filename_mhd_reduced = str((TEST_DIR / "boozmn_n3are_R7.75B5.7_reduced.nc").resolve())
filename_mhd_reordered = str(
    (TEST_DIR / "boozmn_n3are_R7.75B5.7_reordered.nc").resolve()
)
filename_mhd_lasym = str(
    (TEST_DIR / "boozmn_LandremanSenguptaPlunk_section5p3.nc").resolve()
)
filename_mhd_lasym_wout = str(
    (TEST_DIR / "wout_LandremanSenguptaPlunk_section5p3.nc").resolve()
)

try:
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
except ImportError:
    comm = None


class TestingAnalytic(unittest.TestCase):
    def test_boozeranalytic(self):
        etabar = 1.1
        B0 = 1.0
        N = 0
        G0 = 1.1
        psi0 = 0.8
        iota0 = 0.4
        K1 = 1.8
        ba = BoozerAnalytic(etabar, B0, N, G0, psi0, iota0, K1)

        ntheta = 101
        nzeta = 100
        thetas = np.linspace(0, 2 * np.pi, ntheta, endpoint=False)
        zetas = np.linspace(0, 2 * np.pi, nzeta, endpoint=False)
        [zetas, thetas] = np.meshgrid(zetas, thetas)
        points = np.zeros((len(thetas.flatten()), 3))
        points[:, 0] = 0.5 * np.ones_like(thetas.flatten())
        points[:, 1] = thetas.flatten()
        points[:, 2] = zetas.flatten()
        ba.set_points(points)

        # Check that get_points returns correct points
        points_get = ba.get_points()
        thetas_get = points_get[:, 1]
        assert np.allclose(thetas_get, thetas.flatten())

        # Check that angular derivatives integrate to zero
        assert np.allclose(
            np.sum(ba.dmodBdtheta().reshape(np.shape(thetas)), axis=0), 0, rtol=1e-12
        )
        assert np.allclose(
            np.sum(ba.dmodBdzeta().reshape(np.shape(thetas)), axis=1), 0, rtol=1e-12
        )

        # Check that zeta derivatives are small since we are QA
        assert np.allclose(ba.dmodBdzeta(), 0, atol=1e-12)

        # Check that (theta + zeta) derivatives are small since we are QH
        ba.set_N(1)
        assert np.allclose(ba.dmodBdtheta() + ba.dmodBdzeta(), 0, atol=1e-12)

        # Check that G = G0
        assert np.allclose(ba.G(), G0, atol=1e-12)

        # Check that dGds = 0 since G1 = 0
        assert np.allclose(ba.dGds(), 0, atol=1e-12)

        # Check that I = 0 and I'=0 since I0 = I1 = 0
        assert np.allclose(ba.I(), 0, atol=1e-12)
        assert np.allclose(ba.dIds(), 0, atol=1e-12)

        # Check that iota = iota0
        assert np.allclose(ba.iota(), iota0, atol=1e-12)

        # Check that diotads = 0
        assert np.allclose(ba.diotads(), 0, atol=1e-12)

        # Check that if etabar = 0, dBdtheta = dBdzeta = dBds = 0
        ba.set_etabar(0.0)
        assert np.allclose(ba.dmodBdtheta(), 0, atol=1e-12)
        assert np.allclose(ba.dmodBdzeta(), 0, atol=1e-12)
        assert np.allclose(ba.dmodBds(), 0, atol=1e-12)

        # Check that K_derivs matches dKdtheta and dKdzeta
        K_derivs = ba.K_derivs()
        assert np.allclose(ba.dKdtheta(), K_derivs[:, 0])
        assert np.allclose(ba.dKdzeta(), K_derivs[:, 1])

        # Check that modB_derivs matches (dmodBds,dmodBdtheta,dmodBdzeta)
        modB_derivs = ba.modB_derivs()
        assert np.allclose(ba.dmodBds(), modB_derivs[:, 0])
        assert np.allclose(ba.dmodBdtheta(), modB_derivs[:, 1])
        assert np.allclose(ba.dmodBdzeta(), modB_derivs[:, 2])

        # Check other set_ functions
        ba.set_B0(1.3)
        assert ba.B0 == 1.3
        ba.set_Bbar(3.0)
        assert ba.Bbar == 3.0
        ba.set_G0(3.1)
        assert ba.G0 == 3.1
        ba.set_I0(3.2)
        assert ba.I0 == 3.2
        ba.set_G1(3.3)
        assert ba.G1 == 3.3
        ba.set_I1(3.4)
        assert ba.I1 == 3.4
        ba.set_iota0(3.5)
        assert ba.iota0 == 3.5
        ba.set_psi0(3.6)
        assert ba.psi0 == 3.6
        ba.set_K1(3.7)
        assert ba.K1 == 3.7


class TestingFiniteBeta(unittest.TestCase):
    def test_boozerradialinterpolant_finite_beta(self):
        """
        This first loop tests a finite-beta equilibria for both
        BoozerRadialInterpolant and BoozerSplineField
        """
        order = 3
        ntheta = 96
        nzeta = 96

        # Test both BoozerRadialInterpolant and BoozerSplineField
        field_classes = [
            ("BoozerSplineField", BoozerSplineField),
            ("BoozerRadialInterpolant", BoozerRadialInterpolant),
        ]

        for field_name, FieldClass in field_classes:
            # The following tests different initializations
            filename = filename_mhd
            filename_wout = filename_mhd_wout
            filename_reduced = filename_mhd_reduced
            filename_reordered = filename_mhd_reordered

            # First, initialize correctly-sized grid (booz_xform)
            if field_name == "BoozerSplineField":
                field = FieldClass(filename, ntheta=ntheta, nzeta=nzeta, comm=comm)
            else:
                field = FieldClass(filename, order, comm=comm)

            thetas = np.linspace(0, 2 * np.pi, ntheta, endpoint=False)
            zetas = np.linspace(0, 2 * np.pi / field.nfp, nzeta, endpoint=False)

            thetas, zetas = np.meshgrid(thetas, zetas)
            thetas_flat = thetas.flatten()
            zetas_flat = zetas.flatten()

            s_0 = np.copy(field.s_half_ext)
            G_0 = field.G_spline(0.5)

            # Next, initialize with wout file and check for consistency
            if field_name == "BoozerSplineField":
                field = FieldClass(
                    filename_wout,
                    ntheta=ntheta,
                    nzeta=nzeta,
                    comm=comm,
                )
            else:
                field = FieldClass(filename_wout, order, comm=comm)

            s_1 = np.copy(field.s_half_ext)
            G_1 = field.G_spline(0.5)

            assert np.allclose(s_0, s_1)
            assert G_0 == G_1

            # Next, initialize wrong size of radial grid
            with self.assertRaises(ValueError):
                if field_name == "BoozerSplineField":
                    field = FieldClass(
                        filename_reduced,
                        ntheta=ntheta,
                        nzeta=nzeta,
                        comm=comm,
                    )
                else:
                    field = FieldClass(filename_reduced, order, comm=comm)

            # Next, initialize grid with incorrect order or s points
            # Note: BoozerSplineField doesn't check for reordered files
            # in the same way, so we only test this for BoozerRadialInterpolant
            if field_name == "BoozerRadialInterpolant":
                with self.assertRaises(ValueError):
                    field = FieldClass(filename_reordered, order, comm=comm)

        # These tests require higher resolution equilibrium, since they check
        # for consistency of the Jacobian and satisfying the magnetic
        # differential equation
        for field_name, FieldClass in field_classes:
            for asym in [True, False]:
                filename = filename_mhd_lasym if asym else filename_mhd
                filename_wout = filename_mhd_lasym_wout if asym else filename_mhd_wout

                if field_name == "BoozerSplineField":
                    field = FieldClass(
                        filename,
                        ntheta=ntheta,
                        nzeta=nzeta,
                        comm=comm,
                        no_K=False,
                        spline_deriv=True,
                    )
                else:
                    field = FieldClass(filename, order, comm=comm, no_K=False)

                isurf = round(0.75 * len(field.s_half_ext))

                """
                These evaluation points test that the Jacobian sqrtg = (G + iota I)/B^2
                matches sqrt(det(g_ij)).
                """
                # Perform interpolation from full grid
                points = np.zeros((len(thetas_flat), 3))
                points[:, 0] = field.s_half_ext[isurf]
                points[:, 1] = thetas_flat
                points[:, 2] = zetas_flat
                field.set_points(points)

                G = field.G()[:, 0]
                I = field.I()[:, 0]
                iota = field.iota()[:, 0]
                B = field.modB()[:, 0]
                sqrtg = (G + iota * I) / (B * B)

                detg = np.abs(
                    np.sqrt(np.abs(field.get_covariant_metric().det())) / field.psi0
                )

                assert np.allclose(
                    detg / np.mean(np.abs(sqrtg)),
                    np.abs(sqrtg) / np.mean(np.abs(sqrtg)),
                    atol=1e-2,
                )

                # Asym equilibria is at p = 0
                if not asym:
                    """
                    These evaluation points test that K() satisfies the magnetic
                    differential equation: iota dK/dtheta + dK/dzeta = sqrt(g)
                    mu0*p'(psi) + G'(psi) + iota*I'(psi)
                    """
                    isurf = round(0.3 * len(field.s_half_ext))
                    points = np.zeros((len(thetas_flat), 3))
                    # For BoozerSplineField, use s_full from the field
                    # For BoozerRadialInterpolant, compute from bx.ns_b
                    if field_name == "BoozerSplineField":
                        s_full = field.s_full
                    else:
                        s_full = np.linspace(0, 1, field.bx.ns_b + 1)
                    points[:, 0] = s_full[isurf]
                    points[:, 1] = thetas_flat
                    points[:, 2] = zetas_flat

                    field.set_points(points)

                    K = field.K()[:, 0]
                    dKdtheta = field.dKdtheta()[:, 0]
                    dKdzeta = field.dKdzeta()[:, 0]
                    K_derivs = field.K_derivs()

                    assert np.allclose(K_derivs[:, 0], dKdtheta)
                    assert np.allclose(K_derivs[:, 1], dKdzeta)

                    I = field.I()[:, 0]
                    G = field.G()[:, 0]
                    iota = field.iota()[:, 0]
                    modB = field.modB()[:, 0]
                    sqrtg = (G + iota * I) / (modB * modB)
                    dGdpsi = field.dGds()[:, 0] / field.psi0
                    dIdpsi = field.dIds()[:, 0] / field.psi0

                    f = netcdf_file(filename_wout, mmap=False)
                    pres = f.variables["pres"][()]
                    ds = field.s_half_ext[2] - field.s_half_ext[1]
                    dpdpsi = (pres[isurf + 1] - pres[isurf - 1]) / (2 * ds * field.psi0)
                    mu0 = 4 * np.pi * 1e-7
                    rhs = mu0 * dpdpsi * sqrtg + dGdpsi + iota * dIdpsi

                    K = K.reshape(np.shape(thetas))
                    dKdtheta = dKdtheta.reshape(np.shape(thetas))
                    dKdzeta = dKdzeta.reshape(np.shape(zetas))
                    lhs = iota.reshape(np.shape(thetas)) * dKdtheta + dKdzeta

                    assert np.allclose(rhs, lhs.flatten(), atol=1e-2)

    def test_boozerradialinterpolant_vacuum(self):
        """
        The next loop tests a vacuum equilibria
        """
        order = 3
        for asym in [True, False]:
            if asym:
                filename = filename_mhd_lasym
                filename_wout = filename_mhd_lasym_wout
            else:
                filename = filename_vac
                filename_wout = filename_vac_wout
            bri = BoozerRadialInterpolant(
                filename,
                order,
                mpol=20,
                ntor=18,
                no_K=True,
                comm=comm,
            )

            """
            These evaluation points test G(), iota(), modB(), R(), and
            associated derivatives by comparing with linear interpolation
            onto vmec full grid.
            """
            # Perform interpolation from full grid
            points = np.zeros((len(bri.s_half_ext) - 3, 3))
            s_full = np.linspace(0, 1, bri.bx.ns_b + 1)
            points[:, 0] = s_full[1:-1]
            bri.set_points(points)

            # Check with linear interpolation from half grid
            f = netcdf_file(filename_wout, mmap=False)
            bvco = f.variables["bvco"][()]
            iotas = f.variables["iotas"][()]
            G_full = (bvco[1:-1] + bvco[2::]) / 2.0
            iota_full = (iotas[1:-1] + iotas[2::]) / 2.0
            # magnitude of B at theta = 0, zeta = 0
            if comm is not None:
                modB00 = np.sum(bri.bx.bmnc_b, axis=0) if comm.rank == 0 else None
                modB00 = comm.bcast(modB00, root=0)
            else:
                modB00 = np.sum(bri.bx.bmnc_b, axis=0)
            modB_full = (modB00[0:-1] + modB00[1::]) / 2

            # Compare splines of derivatives with spline derivatives
            f = netcdf_file(filename_wout, mmap=False)

            bvco = f.variables["bvco"][()][1::]
            iotas = f.variables["iotas"][()][1::]
            G_spline = InterpolatedUnivariateSpline(bri.s_half_ext[1:-1], bvco)
            iota_spline = InterpolatedUnivariateSpline(bri.s_half_ext[1:-1], iotas)
            modB00_spline = InterpolatedUnivariateSpline(bri.s_half_ext[1:-1], modB00)

            if comm is not None:
                if comm.rank == 0:
                    rmnc_half = bri.bx.rmnc_b
                    rmnc_full = 0.5 * (bri.bx.rmnc_b[:, 0:-1] + bri.bx.rmnc_b[:, 1::])
                else:
                    rmnc_half = None
                    rmnc_full = None
                rmnc_half = comm.bcast(rmnc_half, root=0)
                rmnc_full = comm.bcast(rmnc_full, root=0)
            else:
                rmnc_half = bri.bx.rmnc_b
                rmnc_full = 0.5 * (bri.bx.rmnc_b[:, 0:-1] + bri.bx.rmnc_b[:, 1::])
            # major radius at theta = 0, zeta = 0
            R00_half = np.sum(rmnc_half, axis=0)
            R00_full = np.sum(rmnc_full, axis=0)
            R00_spline = InterpolatedUnivariateSpline(bri.s_half_ext[1:-1], R00_half)

            assert np.allclose(bri.G()[:, 0], G_full, rtol=1e-4)
            assert np.allclose(bri.iota()[:, 0], iota_full, rtol=1e-2)
            assert np.allclose(bri.modB()[:, 0], modB_full, rtol=1e-2)
            assert np.allclose(bri.R()[:, 0], R00_full, rtol=1e-2)

            # Only compare away from axis since inacurracies are introduced
            # through spline due to r ~ sqrt(s) behavior

            s_full = np.linspace(0, 1, bri.bx.ns_b + 1)
            assert np.allclose(
                bri.dGds()[5::, 0],
                G_spline.derivative()(s_full[6:-1]),
                atol=1e-5,
            )
            mean_diotads = np.mean(np.abs(bri.diotads()[5::, 0]))
            assert np.allclose(
                bri.diotads()[5::, 0] / mean_diotads,
                iota_spline.derivative()(s_full[6:-1]) / mean_diotads,
                atol=1e-2,
            )
            assert np.allclose(
                bri.dmodBds()[5::, 0],
                modB00_spline.derivative()(s_full[6:-1]),
                rtol=1e-2,
            )
            mean_dRds = np.mean(np.abs(bri.dRds()))
            assert np.allclose(
                bri.dRds()[5::, 0] / mean_dRds,
                R00_spline.derivative()(s_full[6:-1]) / mean_dRds,
                atol=1e-2,
            )

            """
            The next evaluation points test Z() and nu()
            """
            points = np.zeros((len(bri.s_half_ext[1:-1]), 3))
            points[:, 0] = bri.s_half_ext[1:-1]
            points[:, 1] = 0.0
            points[:, 2] = np.pi / 3
            bri.set_points(points)

            nu = bri.nu()
            iota = bri.iota()

            # zmns/zmnc on full grid
            # lmnc/lmns on half grid
            f = netcdf_file(filename_wout, mmap=False)

            zmns = f.variables["zmns"][()].T
            lmns_half = (f.variables["lmns"][()].T)[:, 1::]
            zmns_half = 0.5 * (zmns[:, 0:-1] + zmns[:, 1::])
            if bri.asym:
                lmnc_half = (f.variables["lmnc"][()].T)[:, 1::]
                zmnc = f.variables["zmnc"][()].T
                zmnc_half = 0.5 * (zmnc[:, 0:-1] + zmnc[:, 1::])
            else:
                lmnc_half = np.zeros_like(lmns_half)
                zmnc_half = np.zeros_like(zmns_half)

            # Determine the vmec theta/phi corresponding to theta_b = 0,
            # zeta_b = pi/3
            # Here zeta_b = phi + nu
            # theta + lambda - iota * phi = theta_b - iota * zeta_b
            # theta + lambda - iota * (pi/3 - nu) = - iota * pi/3
            # theta + lambda + iota * nu = 0

            xm = f.variables["xm"][()]
            xn = f.variables["xn"][()]

            def theta_diff(theta, isurf, lmns_half, xm, xn, nu, lmnc_half, iota):
                lam = np.sum(
                    lmns_half[:, isurf]
                    * np.sin(xm * theta - xn * (np.pi / 3 - nu[isurf, 0]))
                    + lmnc_half[:, isurf]
                    * np.cos(xm * theta - xn * (np.pi / 3 - nu[isurf, 0])),
                    axis=0,
                )
                return ((theta + lam) + iota[isurf, 0] * (nu[isurf, 0])) ** 2

            s_half_grid = bri.s_half_ext[1:-1]
            thetas_vmec = np.zeros((len(s_half_grid),))
            for isurf in range(len(s_half_grid)):
                theta_diff_bound = functools.partial(
                    theta_diff,
                    isurf=isurf,
                    lmns_half=lmns_half,
                    xm=xm,
                    xn=xn,
                    nu=nu,
                    lmnc_half=lmnc_half,
                    iota=iota,
                )
                opt = minimize(theta_diff_bound, 0)
                thetas_vmec[isurf] = opt.x[0]

            # Compute Z at theta_b = 0, zeta_b = pi/2  and compare with vmec result
            Z0pi = np.sum(
                zmns_half
                * np.sin(
                    xm[:, None] * thetas_vmec[None, :]
                    - xn[:, None] * (np.pi / 3 - nu[None, :, 0])
                )
                + zmnc_half
                * np.cos(
                    xm[:, None] * thetas_vmec[None, :]
                    - xn[:, None] * (np.pi / 3 - nu[None, :, 0])
                ),
                axis=0,
            )
            Z0pi_spline = InterpolatedUnivariateSpline(s_half_grid, Z0pi)

            mean_dZds = np.mean(np.abs(bri.dZds()[5::, 0]))

            assert np.allclose(bri.Z()[:, 0], Z0pi, atol=1e-2)

            if not asym:
                assert np.allclose(
                    bri.dZds()[5::, 0] / mean_dZds,
                    Z0pi_spline.derivative()(s_half_grid[5::]) / mean_dZds,
                    atol=1e-2,
                )
            # For asymmetric case, the dZds = 0
            else:
                assert np.allclose(bri.dZds()[5::, 0], 0, atol=1e-2)
                assert np.allclose(
                    Z0pi_spline.derivative()(s_half_grid[5::]), 0, atol=1e-2
                )

            """
            The next evaluation points test the derivatives of modB, R, Z, and nu
            """
            points = np.zeros((len(s_half_grid), 3))
            points[:, 0] = s_half_grid
            bri.set_points(points)

            # Check that angular derivatives integrate to zero
            ntheta = 101
            nzeta = 100
            thetas = np.linspace(0, 2 * np.pi, ntheta, endpoint=False)
            zetas = np.linspace(0, 2 * np.pi, nzeta, endpoint=False)
            [zetas, thetas] = np.meshgrid(zetas, thetas)
            points = np.zeros((len(thetas.flatten()), 3))
            points[:, 0] = 0.5 * np.ones_like(thetas.flatten())
            points[:, 1] = thetas.flatten()
            points[:, 2] = zetas.flatten()
            bri.set_points(points)
            # Check that get_points returns correct points
            points_get = bri.get_points()
            thetas_get = points_get[:, 1]
            assert np.allclose(thetas_get, thetas.flatten())
            assert np.allclose(
                np.sum(bri.dmodBdtheta().reshape(np.shape(thetas)), axis=0),
                0,
                rtol=1e-12,
            )
            assert np.allclose(
                np.sum(bri.dmodBdzeta().reshape(np.shape(thetas)), axis=1),
                0,
                rtol=1e-12,
            )
            assert np.allclose(
                np.sum(bri.dRdtheta().reshape(np.shape(thetas)), axis=0),
                0,
                rtol=1e-12,
            )
            assert np.allclose(
                np.sum(bri.dRdzeta().reshape(np.shape(thetas)), axis=1),
                0,
                rtol=1e-12,
            )
            assert np.allclose(
                np.sum(bri.dZdtheta().reshape(np.shape(thetas)), axis=0),
                0,
                rtol=1e-12,
            )
            assert np.allclose(
                np.sum(bri.dZdzeta().reshape(np.shape(thetas)), axis=1),
                0,
                rtol=1e-12,
            )
            assert np.allclose(
                np.sum(bri.dnudtheta().reshape(np.shape(thetas)), axis=0),
                0,
                rtol=1e-12,
            )
            assert np.allclose(
                np.sum(bri.dnudzeta().reshape(np.shape(thetas)), axis=1),
                0,
                rtol=1e-12,
            )
            assert np.allclose(
                np.sum(bri.dKdtheta().reshape(np.shape(thetas)), axis=0),
                0,
                rtol=1e-12,
            )
            assert np.allclose(
                np.sum(bri.dKdzeta().reshape(np.shape(thetas)), axis=1),
                0,
                rtol=1e-12,
            )

            # Check that zeta derivatives are small since we are close to QA
            if not asym:
                assert np.allclose(bri.dmodBdzeta(), 0, atol=1e-2)

            assert np.allclose(bri.R_derivs()[:, 0], bri.dRds()[:, 0])
            assert np.allclose(bri.R_derivs()[:, 1], bri.dRdtheta()[:, 0])
            assert np.allclose(bri.R_derivs()[:, 2], bri.dRdzeta()[:, 0])
            assert np.allclose(bri.Z_derivs()[:, 0], bri.dZds()[:, 0])
            assert np.allclose(bri.Z_derivs()[:, 1], bri.dZdtheta()[:, 0])
            assert np.allclose(bri.Z_derivs()[:, 2], bri.dZdzeta()[:, 0])
            assert np.allclose(bri.nu_derivs()[:, 0], bri.dnuds()[:, 0])
            assert np.allclose(bri.nu_derivs()[:, 1], bri.dnudtheta()[:, 0])
            assert np.allclose(bri.nu_derivs()[:, 2], bri.dnudzeta()[:, 0])
            assert np.allclose(bri.modB_derivs()[:, 0], bri.dmodBds()[:, 0])
            assert np.allclose(bri.modB_derivs()[:, 1], bri.dmodBdtheta()[:, 0])
            assert np.allclose(bri.modB_derivs()[:, 2], bri.dmodBdzeta()[:, 0])

    def test_interpolatedboozerfield_sym(self):
        """
        Here we perform 3D interpolation on a random set of points. Compare
        BoozerRadialInterpolant with InterpolatedBoozerField. We enforce
        nfp and stellarator symmetry in the 3D interpolant.
        """

        order = 3
        bri = BoozerRadialInterpolant(filename_vac, order, comm=comm)

        n = 8
        smin = 0.4
        smax = 0.6
        ssteps = n
        thetamin = 0
        thetamax = np.pi
        thetasteps = n
        zetamin = 0
        zetamax = 2 * np.pi / (bri.nfp)
        zetasteps = n * 2
        bsh = InterpolatedBoozerField(
            bri,
            4,
            [smin, smax, ssteps],
            [thetamin, thetamax, thetasteps],
            [zetamin, zetamax, zetasteps],
            True,
            stellsym=True,
            nfp=bri.nfp,
        )

        # Compute points outside of interpolation range
        N = 10
        np.random.seed(2)
        points = np.random.uniform(size=(N, 3))
        thetamin = -np.pi
        thetamax = 2 * np.pi
        zetamin = -2 * np.pi / bri.nfp
        zetamax = 4 * np.pi / bri.nfp
        points[:, 0] = points[:, 0] * (smax - smin) + smin
        points[:, 1] = points[:, 1] * (thetamax - thetamin) + thetamin
        points[:, 2] = points[:, 2] * (zetamax - zetamin) + zetamin

        bri.set_points(points)
        modB = bri.modB()
        R = bri.R()
        dRdtheta = bri.dRdtheta()
        dRdzeta = bri.dRdzeta()
        dRds = bri.dRds()
        Z = bri.Z()
        dZdtheta = bri.dZdtheta()
        dZdzeta = bri.dZdzeta()
        dZds = bri.dZds()
        dmodBds = bri.dmodBds()
        dmodBdtheta = bri.dmodBdtheta()
        dmodBdzeta = bri.dmodBdzeta()
        nu = bri.nu()
        dnudtheta = bri.dnudtheta()
        dnudzeta = bri.dnudzeta()
        dnuds = bri.dnuds()
        G = bri.G()
        I = bri.I()
        iota = bri.iota()
        diotads = bri.diotads()
        dGds = bri.dGds()
        dIds = bri.dIds()
        K = bri.K()
        dKdtheta = bri.dKdtheta()
        dKdzeta = bri.dKdzeta()

        bsh.set_points(points)
        modBh = bsh.modB()
        Rh = bsh.R()
        dRdthetah = bsh.dRdtheta()
        dRdzetah = bsh.dRdzeta()
        dRdsh = bsh.dRds()
        Zh = bsh.Z()
        dZdthetah = bsh.dZdtheta()
        dZdzetah = bsh.dZdzeta()
        dZdsh = bsh.dZds()
        nuh = bsh.nu()
        dnudthetah = bsh.dnudtheta()
        dnudzetah = bsh.dnudzeta()
        dnudsh = bsh.dnuds()
        dmodBdsh = bsh.dmodBds()
        dmodBdthetah = bsh.dmodBdtheta()
        dmodBdzetah = bsh.dmodBdzeta()
        Gh = bsh.G()
        Ih = bsh.I()
        iotah = bsh.iota()
        diotadsh = bsh.diotads()
        dGdsh = bsh.dGds()
        dIdsh = bsh.dIds()
        Kh = bsh.K()
        dKdthetah = bsh.dKdtheta()
        dKdzetah = bsh.dKdzeta()

        assert np.allclose((K - Kh) / np.mean(np.abs(K)), 0, atol=1e-2)
        assert np.allclose(
            (dKdtheta - dKdthetah) / np.mean(np.abs(dKdtheta)), 0, atol=1e-2
        )
        assert np.allclose(
            (dKdzeta - dKdzetah) / np.mean(np.abs(dKdzeta)), 0, atol=5e-2
        )

        assert np.allclose(modB, modBh, rtol=1e-3)
        assert np.allclose(
            (dmodBds - dmodBdsh) / np.mean(np.abs(dmodBds)), 0, atol=1e-2
        )
        assert np.allclose(
            (dmodBdtheta - dmodBdthetah) / np.mean(np.abs(dmodBdtheta)), 0, atol=1e-2
        )
        assert np.allclose(dmodBdzeta - dmodBdzetah, 0, atol=1e-3)

        assert np.allclose(R, Rh, rtol=1e-3)
        assert np.allclose(dRds, dRdsh, rtol=1e-3)
        assert np.allclose(dRdtheta, dRdthetah, rtol=1e-3)
        assert np.allclose(dRdzeta, dRdzetah, rtol=1e-3)

        assert np.allclose(Z, Zh, rtol=1e-3)
        assert np.allclose(dZds, dZdsh, rtol=1e-3)
        assert np.allclose(dZdtheta, dZdthetah, rtol=1e-3)
        assert np.allclose(dZdzeta, dZdzetah, rtol=1e-3)

        assert np.allclose(nu, nuh, rtol=1e-3)
        assert np.allclose(dnuds, dnudsh, rtol=1e-3)
        assert np.allclose(dnudtheta, dnudthetah, rtol=1e-3)
        assert np.allclose(dnudzeta, dnudzetah, rtol=1e-3)

        assert np.allclose(iota, iotah, rtol=1e-3)
        assert np.allclose(G, Gh, rtol=1e-3)
        assert np.allclose(I, Ih, rtol=1e-3)

        assert np.allclose(diotads, diotadsh, rtol=1e-3)
        assert np.allclose(dGds, dGdsh, rtol=1e-3)
        assert np.allclose(dIds, dIdsh, rtol=1e-3)

        assert np.allclose(bsh.K_derivs()[:, 0], bsh.dKdtheta()[:, 0])
        assert np.allclose(bsh.K_derivs()[:, 1], bsh.dKdzeta()[:, 0])
        assert np.allclose(bsh.R_derivs()[:, 0], bsh.dRds()[:, 0])
        assert np.allclose(bsh.R_derivs()[:, 1], bsh.dRdtheta()[:, 0])
        assert np.allclose(bsh.R_derivs()[:, 2], bsh.dRdzeta()[:, 0])
        assert np.allclose(bsh.Z_derivs()[:, 0], bsh.dZds()[:, 0])
        assert np.allclose(bsh.Z_derivs()[:, 1], bsh.dZdtheta()[:, 0])
        assert np.allclose(bsh.Z_derivs()[:, 2], bsh.dZdzeta()[:, 0])
        assert np.allclose(bsh.nu_derivs()[:, 0], bsh.dnuds()[:, 0])
        assert np.allclose(bsh.nu_derivs()[:, 1], bsh.dnudtheta()[:, 0])
        assert np.allclose(bsh.nu_derivs()[:, 2], bsh.dnudzeta()[:, 0])
        assert np.allclose(bsh.modB_derivs()[:, 0], bsh.dmodBds()[:, 0])
        assert np.allclose(bsh.modB_derivs()[:, 1], bsh.dmodBdtheta()[:, 0])
        assert np.allclose(bsh.modB_derivs()[:, 2], bsh.dmodBdzeta()[:, 0])

    def test_interpolatedboozerfield_no_sym(self):
        """
        Here we perform 3D interpolation on a random set of points. Compare
        BoozerRadialInterpolant with InterpolatedBoozerField. We don't enforce
        nfp and stellarator symmetry in the 3D interpolant.
        """

        order = 3
        bri = BoozerRadialInterpolant(filename_vac, order, comm=comm)

        n = 16
        smin = 0.4
        smax = 0.6
        ssteps = n
        thetamin = 0
        thetamax = 2 * np.pi
        thetasteps = n
        zetamin = 0
        zetamax = 2 * np.pi
        zetasteps = n * 2

        with self.assertRaises(ValueError):
            bsh = InterpolatedBoozerField(
                bri,
                4,
                [smin, smax, ssteps],
                [-np.pi, 0, thetasteps],
                [zetamin, zetamax, zetasteps],
                True,
                stellsym=False,
            )
        with self.assertRaises(ValueError):
            bsh = InterpolatedBoozerField(
                bri,
                4,
                [smin, smax, ssteps],
                [thetamin, thetamax, thetasteps],
                [-np.pi, 0, zetasteps],
                True,
                stellsym=False,
            )

        bsh = InterpolatedBoozerField(
            bri,
            4,
            [smin, smax, ssteps],
            [thetamin, thetamax, thetasteps],
            [zetamin, zetamax, zetasteps],
            True,
            stellsym=False,
        )

        # Compute points outside of interpolation range
        N = 10
        np.random.seed(2)
        points = np.random.uniform(size=(N, 3))
        thetamin = -2 * np.pi
        thetamax = 4 * np.pi
        zetamin = -2 * np.pi
        zetamax = 4 * np.pi
        points[:, 0] = points[:, 0] * (smax - smin) + smin
        points[:, 1] = points[:, 1] * (thetamax - thetamin) + thetamin
        points[:, 2] = points[:, 2] * (zetamax - zetamin) + zetamin

        bri.set_points(points)
        modB = bri.modB()
        R = bri.R()
        dRdtheta = bri.dRdtheta()
        dRdzeta = bri.dRdzeta()
        dRds = bri.dRds()
        Z = bri.Z()
        dZdtheta = bri.dZdtheta()
        dZdzeta = bri.dZdzeta()
        dZds = bri.dZds()
        dmodBds = bri.dmodBds()
        dmodBdtheta = bri.dmodBdtheta()
        dmodBdzeta = bri.dmodBdzeta()
        nu = bri.nu()
        dnudtheta = bri.dnudtheta()
        dnudzeta = bri.dnudzeta()
        dnuds = bri.dnuds()
        G = bri.G()
        I = bri.I()
        iota = bri.iota()
        diotads = bri.diotads()
        dGds = bri.dGds()
        dIds = bri.dIds()
        K = bri.K()
        dKdtheta = bri.dKdtheta()
        dKdzeta = bri.dKdzeta()

        bsh.set_points(points)
        modBh = bsh.modB()
        Rh = bsh.R()
        dRdthetah = bsh.dRdtheta()
        dRdzetah = bsh.dRdzeta()
        dRdsh = bsh.dRds()
        Zh = bsh.Z()
        dZdthetah = bsh.dZdtheta()
        dZdzetah = bsh.dZdzeta()
        dZdsh = bsh.dZds()
        nuh = bsh.nu()
        dnudthetah = bsh.dnudtheta()
        dnudzetah = bsh.dnudzeta()
        dnudsh = bsh.dnuds()
        dmodBdsh = bsh.dmodBds()
        dmodBdthetah = bsh.dmodBdtheta()
        dmodBdzetah = bsh.dmodBdzeta()
        Gh = bsh.G()
        Ih = bsh.I()
        iotah = bsh.iota()
        diotadsh = bsh.diotads()
        dGdsh = bsh.dGds()
        dIdsh = bsh.dIds()
        Kh = bsh.K()
        dKdthetah = bsh.dKdtheta()
        dKdzetah = bsh.dKdzeta()

        assert np.allclose((K - Kh) / np.mean(np.abs(K)), 0, atol=1e-2)
        assert np.allclose(
            (dKdtheta - dKdthetah) / np.mean(np.abs(dKdtheta)), 0, atol=1e-2
        )
        assert np.allclose(
            (dKdzeta - dKdzetah) / np.mean(np.abs(dKdzeta)), 0, atol=5e-2
        )

        assert np.allclose(modB, modBh, rtol=1e-3)
        assert np.allclose(
            (dmodBds - dmodBdsh) / np.mean(np.abs(dmodBds)), 0, atol=1e-2
        )
        assert np.allclose(
            (dmodBdtheta - dmodBdthetah) / np.mean(np.abs(dmodBdtheta)), 0, atol=1e-2
        )
        assert np.allclose(dmodBdzeta - dmodBdzetah, 0, atol=1e-3)

        assert np.allclose(R, Rh, rtol=1e-3)
        assert np.allclose(dRds, dRdsh, rtol=1e-3)
        assert np.allclose(dRdtheta, dRdthetah, rtol=1e-3)
        assert np.allclose(dRdzeta, dRdzetah, rtol=1e-3)

        assert np.allclose(Z, Zh, rtol=1e-3)
        assert np.allclose(dZds, dZdsh, rtol=1e-3)
        assert np.allclose(dZdtheta, dZdthetah, rtol=1e-3)
        assert np.allclose(dZdzeta, dZdzetah, rtol=1e-3)

        assert np.allclose(nu, nuh, rtol=1e-3)
        assert np.allclose(dnuds, dnudsh, rtol=1e-3)
        assert np.allclose(dnudtheta, dnudthetah, rtol=1e-3)
        assert np.allclose(dnudzeta, dnudzetah, rtol=1e-3)

        assert np.allclose(iota, iotah, rtol=1e-3)
        assert np.allclose(G, Gh, rtol=1e-3)
        assert np.allclose(I, Ih, rtol=1e-3)

        assert np.allclose(diotads, diotadsh, rtol=1e-3)
        assert np.allclose(dGds, dGdsh, rtol=1e-3)
        assert np.allclose(dIds, dIdsh, rtol=1e-3)

        assert np.allclose(bsh.K_derivs()[:, 0], bsh.dKdtheta()[:, 0])
        assert np.allclose(bsh.K_derivs()[:, 1], bsh.dKdzeta()[:, 0])
        assert np.allclose(bsh.R_derivs()[:, 0], bsh.dRds()[:, 0])
        assert np.allclose(bsh.R_derivs()[:, 1], bsh.dRdtheta()[:, 0])
        assert np.allclose(bsh.R_derivs()[:, 2], bsh.dRdzeta()[:, 0])
        assert np.allclose(bsh.Z_derivs()[:, 0], bsh.dZds()[:, 0])
        assert np.allclose(bsh.Z_derivs()[:, 1], bsh.dZdtheta()[:, 0])
        assert np.allclose(bsh.Z_derivs()[:, 2], bsh.dZdzeta()[:, 0])
        assert np.allclose(bsh.nu_derivs()[:, 0], bsh.dnuds()[:, 0])
        assert np.allclose(bsh.nu_derivs()[:, 1], bsh.dnudtheta()[:, 0])
        assert np.allclose(bsh.nu_derivs()[:, 2], bsh.dnudzeta()[:, 0])
        assert np.allclose(bsh.modB_derivs()[:, 0], bsh.dmodBds()[:, 0])
        assert np.allclose(bsh.modB_derivs()[:, 1], bsh.dmodBdtheta()[:, 0])
        assert np.allclose(bsh.modB_derivs()[:, 2], bsh.dmodBdzeta()[:, 0])

    def test_interpolatedboozerfield_convergence_rate(self):
        """
        Here we test the convergence rate of modB, R, Z, nu, K, G, I, and iota from
        InterpolatedBoozerField.
        """
        order = 3
        bri = BoozerRadialInterpolant(filename_mhd, order, mpol=10, ntor=10, comm=comm)

        # Perform interpolation from full grid
        points = np.zeros((bri.bx.ns_b - 1, 3))
        s_full = np.linspace(0, 1, bri.bx.ns_b + 1)
        points[:, 0] = s_full[1:-1]
        bri.set_points(points)

        nfp = bri.nfp
        smin = 0.1
        smax = 0.9
        thetamin = np.pi * (1 / 4)
        thetamax = np.pi * (3 / 4)
        zetamin = 2 * np.pi / (4 * nfp)
        zetamax = 2 * np.pi * 3 / (4 * nfp)
        old_err_modB = 1e6
        old_err_I = 1e6
        old_err_G = 1e6
        old_err_iota = 1e6
        old_err_R = 1e6
        old_err_Z = 1e6
        old_err_nu = 1e6
        old_err_K = 1e6
        for n in [4, 8, 16]:
            ssteps = n
            thetasteps = n
            zetasteps = n
            bsh = InterpolatedBoozerField(
                bri,
                1,
                [smin, smax, ssteps],
                [thetamin, thetamax, thetasteps],
                [zetamin, zetamax, zetasteps],
                True,
                nfp=nfp,
                stellsym=True,
            )
            err_modB = np.mean(bsh.estimate_error_modB(1000))
            err_I = np.mean(bsh.estimate_error_I(1000))
            err_G = np.mean(bsh.estimate_error_G(1000))
            err_iota = np.mean(bsh.estimate_error_iota(1000))
            err_R = np.mean(bsh.estimate_error_R(1000))
            err_Z = np.mean(bsh.estimate_error_Z(1000))
            err_nu = np.mean(bsh.estimate_error_nu(1000))
            err_K = np.mean(bsh.estimate_error_K(1000))

            assert err_modB < 0.6**2 * old_err_modB
            assert err_I < 0.6**2 * old_err_I
            assert err_G < 0.6**2 * old_err_G
            assert err_iota < 0.6**2 * old_err_iota
            assert err_R < 0.6**2 * old_err_R
            assert err_Z < 0.6**2 * old_err_Z
            assert err_nu < 0.6**2 * old_err_nu
            assert err_K < 0.6**2 * old_err_K

            old_err_modB = err_modB
            old_err_I = err_I
            old_err_G = err_G
            old_err_iota = err_iota
            old_err_R = err_R
            old_err_Z = err_Z
            old_err_nu = err_nu
            old_err_K = err_K


class TestingRegularGridInterpolantStorage(unittest.TestCase):
    """
    Tests for the flat per-cell storage of RegularGridInterpolant3D: kernel
    dispatch per value_size, out-of-domain lookups, skip-cell compaction, and
    the serialisation round-trip that rebuilds the transient ``vals`` array.
    """

    RANGE = (0.0, 1.0, 8)

    def _build(
        self, f, value_size, degree=3, out_of_bounds_ok=True, skip=None, grid=None
    ):
        grid = self.RANGE if grid is None else grid
        skip = skip or (lambda xs, ys, zs: [False] * len(xs))
        interp = RegularGridInterpolant3D(
            UniformInterpolationRule(degree),
            grid,
            grid,
            grid,
            value_size,
            out_of_bounds_ok,
            skip,
        )
        interp.interpolate_batch(
            lambda x, y, z: (
                f(np.asarray(x), np.asarray(y), np.asarray(z)).flatten().tolist()
            )
        )
        return interp

    @staticmethod
    def _f(value_size):
        # linear in each argument, so every rule reproduces it exactly
        def f(x, y, z):
            return np.stack([(l + 1) * (x + y + z) + l for l in range(value_size)], -1)

        return f

    def test_evaluate_exact_for_each_value_size(self):
        """value_size 1-4 use per-size kernels and 5+ the general one."""
        pts = np.random.default_rng(3).uniform(0.02, 0.98, (200, 3))
        for value_size in (1, 2, 3, 4, 5):
            with self.subTest(value_size=value_size):
                f = self._f(value_size)
                interp = self._build(f, value_size)
                out = np.zeros((200, value_size))
                interp.evaluate_batch(pts, out)
                np.testing.assert_allclose(
                    out, f(pts[:, 0], pts[:, 1], pts[:, 2]), atol=1e-10
                )

    def test_out_of_domain(self):
        """
        out_of_bounds_ok=True clamps the cell indices into range and keeps the
        true local coordinate, extrapolating the boundary cell's polynomial -
        exact for the linear test function, and what lets the flat storage be
        indexed unchecked. out_of_bounds_ok=False raises instead.
        """
        f = self._f(2)
        interp = self._build(f, 2)
        pts = np.array([[-0.4] * 3, [1.4] * 3, [-0.4, 1.4, 0.5]])
        out = np.zeros((3, 2))
        interp.evaluate_batch(pts, out)
        # far-corner extrapolation cancels catastrophically: ~1e-7 attainable
        np.testing.assert_allclose(
            out, f(pts[:, 0], pts[:, 1], pts[:, 2]), rtol=0.0, atol=1e-5
        )
        strict = self._build(self._f(1), 1, out_of_bounds_ok=False)
        strict.evaluate_batch(np.array([[0.5] * 3]), np.zeros((1, 1)))  # in range: fine
        for bad in ([-0.5, 0.5, 0.5], [0.5, 1.5, 0.5], [0.5, 0.5, -0.5]):
            with self.assertRaisesRegex(RuntimeError, "not within", msg=str(bad)):
                strict.evaluate_batch(np.array([bad]), np.zeros((1, 1)))

    def test_skipped_cells_and_roundtrip(self):
        """
        Cells whose corners all satisfy skip get no block: kept blocks are
        compacted and cell_offsets holds -1 for them. The round-trip has to
        rebuild the reduced ``vals`` from that compacted storage.
        """
        vs, R = 2, self.RANGE
        f = self._f(vs)

        def skip(xs, ys, zs):
            # with R's 8 cells this skips the three cells with x >= 0.625
            return [x > 0.5 for x in xs]

        interp = self._build(f, vs, skip=skip)
        kept = np.random.default_rng(5).uniform(
            [0.02] * 3, [0.60, 0.98, 0.98], (100, 3)
        )
        out = np.zeros((100, vs))
        interp.evaluate_batch(kept, out)
        np.testing.assert_allclose(
            out, f(kept[:, 0], kept[:, 1], kept[:, 2]), atol=1e-10
        )

        # a skipped cell leaves the output untouched, or raises in strict mode
        skipped = np.array([[0.7, 0.5, 0.5]])
        sentinel = np.full((1, vs), 7.0)
        interp.evaluate_batch(skipped, sentinel)
        np.testing.assert_array_equal(sentinel, 7.0)
        strict = self._build(f, vs, out_of_bounds_ok=False, skip=skip)
        with self.assertRaisesRegex(RuntimeError, "outside the interpolation domain"):
            strict.evaluate_batch(skipped, np.zeros((1, vs)))

        dst = RegularGridInterpolant3D(
            UniformInterpolationRule(3), R, R, R, vs, True, skip
        )
        data = interp.get_interpolant_data()
        dst.set_interpolant_data(data)
        np.testing.assert_array_equal(data["vals"], dst.get_interpolant_data()["vals"])
        out_dst = np.zeros_like(out)
        dst.evaluate_batch(kept, out_dst)
        np.testing.assert_array_equal(out, out_dst)

    def test_degree_cap(self):
        """MAX_NODES = 16 stack buffers: degree 15 works, degree 16 throws."""
        tiny = (0.0, 1.0, 2)
        interp = self._build(self._f(1), 1, degree=15, grid=tiny)
        pts = np.random.default_rng(7).uniform(0.02, 0.98, (20, 3))
        out = np.zeros((20, 1))
        interp.evaluate_batch(pts, out)
        np.testing.assert_allclose(out[:, 0], pts.sum(axis=1), atol=1e-8)
        with self.assertRaises(ValueError):
            RegularGridInterpolant3D(
                UniformInterpolationRule(16), tiny, tiny, tiny, 1, True
            )

    def test_set_interpolant_data_validates_input(self):
        interp = self._build(self._f(1), 1)
        with self.assertRaisesRegex(RuntimeError, "no 'vals' entry"):
            interp.set_interpolant_data({"nx": [8.0]})
        with self.assertRaisesRegex(RuntimeError, "'vals' has size"):
            interp.set_interpolant_data({"vals": [1.0, 2.0]})


class TestingInverseFourier(unittest.TestCase):
    def test_inverse_fourier(self):
        thetas = np.linspace(0, 2 * np.pi, 131)
        zetas = np.linspace(0, 2 * np.pi, 131)
        [thetas, zetas] = np.meshgrid(thetas, zetas)
        thetas, zetas = thetas.flatten(), zetas.flatten()
        num_points = len(thetas)
        padded_thetas = align_and_pad(thetas)
        padded_zetas = align_and_pad(zetas)

        if comm is not None:
            size = comm.size
            rank = comm.rank
        else:
            size = 1
            rank = 0

        num_modes = [1, 20, 48]
        for mpol in num_modes:
            for ntor in num_modes:
                nfp = np.random.randint(1, 8) if rank == 0 else None
                kmns = (
                    np.random.random(mpol * (ntor * 2 + 1) - ntor) * 2 - 1
                    if rank == 0
                    else None
                )
                if comm is not None:
                    nfp = comm.bcast(nfp, root=0)
                    kmns = comm.bcast(kmns, root=0)

                xm = np.repeat(np.array(range(mpol)), ntor * 2 + 1)[ntor:]
                xn = np.tile(np.array(range(-ntor * nfp, ntor * nfp + 1, nfp)), mpol)[
                    ntor:
                ]
                assert len(kmns) == len(xn)
                assert len(kmns) == len(xm)

                even_K = sum(
                    kmns[:, np.newaxis]
                    * np.cos(np.outer(xm, thetas) - np.outer(xn, zetas))
                )
                odd_K = sum(
                    kmns[:, np.newaxis]
                    * np.sin(np.outer(xm, thetas) - np.outer(xn, zetas))
                )

                mn_idxs = np.array([i * len(xm) // size for i in range(size + 1)])
                first_mn, last_mn = mn_idxs[rank], mn_idxs[rank + 1]
                padded_kmns = align_and_pad(
                    np.tile(kmns[first_mn:last_mn, np.newaxis], (1, num_points))
                )
                padded_even_output = allocate_aligned_and_padded_array(thetas.shape)
                padded_odd_output = allocate_aligned_and_padded_array(thetas.shape)

                inverse_fourier_transform_even(
                    padded_even_output,
                    padded_kmns,
                    xm[first_mn:last_mn],
                    xn[first_mn:last_mn],
                    padded_thetas,
                    padded_zetas,
                    ntor,
                    nfp,
                    False,
                )
                inverse_fourier_transform_odd(
                    padded_odd_output,
                    padded_kmns,
                    xm[first_mn:last_mn],
                    xn[first_mn:last_mn],
                    padded_thetas,
                    padded_zetas,
                    ntor,
                    nfp,
                    False,
                )

                if comm is not None:
                    comm.Allreduce(
                        MPI.IN_PLACE,
                        [padded_even_output[:num_points], MPI.DOUBLE],
                        op=MPI.SUM,
                    )
                    comm.Allreduce(
                        MPI.IN_PLACE,
                        [padded_odd_output[:num_points], MPI.DOUBLE],
                        op=MPI.SUM,
                    )

                assert np.allclose(
                    even_K, padded_even_output[:num_points], rtol=1e-12, atol=1e-11
                )
                assert np.allclose(
                    odd_K, padded_odd_output[:num_points], rtol=1e-12, atol=1e-11
                )

                # a single point
                padded_kmns = align_and_pad(kmns[first_mn:last_mn])
                padded_xm = align_and_pad(xm[first_mn:last_mn])
                padded_xn = align_and_pad(xn[first_mn:last_mn])
                for i in range(num_points):
                    theta = np.array([thetas[i]])
                    zeta = np.array([zetas[i]])
                    even_output = np.zeros(1)
                    odd_output = np.zeros(1)

                    inverse_fourier_transform_even(
                        even_output,
                        padded_kmns,
                        padded_xm,
                        padded_xn,
                        theta,
                        zeta,
                        ntor,
                        nfp,
                        False,
                    )
                    inverse_fourier_transform_odd(
                        odd_output,
                        padded_kmns,
                        padded_xm,
                        padded_xn,
                        theta,
                        zeta,
                        ntor,
                        nfp,
                        False,
                    )

                    if comm is not None:
                        comm.Allreduce(
                            MPI.IN_PLACE, [even_output, MPI.DOUBLE], op=MPI.SUM
                        )
                        comm.Allreduce(
                            MPI.IN_PLACE, [odd_output, MPI.DOUBLE], op=MPI.SUM
                        )

                    assert np.allclose(
                        even_K[i], even_output[0], rtol=1e-12, atol=1e-11
                    )
                    assert np.allclose(odd_K[i], odd_output[0], rtol=1e-12, atol=1e-11)


class TestingBoozerRadialInterpolantSums(unittest.TestCase):
    """
    Check the inverse Fourier sum of BoozerRadialInterpolant against a direct
    evaluation of the harmonic series, for point sets that exercise each of the
    evaluation paths: a single point, a repeated small point set, scattered
    points, and a tensor-product lattice.
    """

    @staticmethod
    def _scattered(rng, npoints):
        return np.column_stack(
            [
                rng.uniform(0.1, 0.9, npoints),
                rng.uniform(0, 2 * np.pi, npoints),
                rng.uniform(0, 2 * np.pi, npoints),
            ]
        )

    @staticmethod
    def _modB_directly(bri, points):
        """|B| summed one point and one mode at a time."""
        modB = np.zeros(len(points))
        for i, (s, theta, zeta) in enumerate(points):
            angle = bri.xm_b * theta - bri.xn_b * zeta
            modB[i] = np.sum(bri.bmnc_splines(np.array([s]))[0] * np.cos(angle))
            if bri.asym:
                modB[i] += np.sum(bri.bmns_splines(np.array([s]))[0] * np.sin(angle))
        return modB

    def test_modB_matches_direct_sum(self):
        for asym in [True, False]:
            if asym:
                bri = BoozerRadialInterpolant(
                    filename_mhd_lasym, 3, mpol=20, ntor=18, no_K=True, comm=comm
                )
            else:
                bri = BoozerRadialInterpolant(filename_vac, 3, no_K=True, comm=comm)
            self.assertEqual(bri.asym, asym)

            rng = np.random.default_rng(17)
            scattered = functools.partial(self._scattered, rng)

            # A lattice of (s, theta, zeta) values, the shape used when
            # building an InterpolatedBoozerField.
            thetas = np.linspace(0, 2 * np.pi, 21)
            zetas = np.linspace(0, 2 * np.pi / bri.nfp, 23)
            grid = np.meshgrid(np.array([0.3, 0.7]), thetas, zetas, indexing="ij")
            lattice = np.column_stack([g.ravel() for g in grid])

            point_sets = [scattered(n) for n in (1, 2, 3, 200)]
            # Evaluate each point set twice, at slightly shifted positions the
            # second time, so that any state cached from the first call has to
            # be invalidated.
            point_sets += [p + 0.01 for p in point_sets]
            point_sets.append(lattice)

            for points in point_sets:
                bri.set_points(points)
                modB = np.asarray(bri.modB()).ravel()
                assert np.allclose(
                    modB, self._modB_directly(bri, points), rtol=1e-12, atol=1e-11
                )


class TestingBoozerSplineField(unittest.TestCase):
    def test_boozersplinefield_initialization(self):
        """
        Test BoozerSplineField initialization with boozmn and wout files.
        """
        ntheta = 21
        nzeta = 20

        for asym in [True, False]:
            if asym:
                filename = filename_mhd_lasym
                filename_wout = filename_mhd_lasym_wout
            else:
                filename = filename_vac
                filename_wout = filename_vac_wout

            # Test initialization with boozmn file
            bsf = BoozerSplineField(filename, ntheta=ntheta, nzeta=nzeta, comm=comm)

            # Test initialization with wout file
            bsf_wout = BoozerSplineField(
                filename_wout,
                ntheta=ntheta,
                nzeta=nzeta,
                mpol=20,
                ntor=18,
                comm=comm,
            )

            # Check that basic attributes are set
            assert bsf.psi0 is not None
            assert bsf.nfp is not None
            assert bsf.ntheta == ntheta
            assert bsf.nzeta == nzeta
            assert bsf_wout.psi0 is not None
            assert bsf_wout.nfp is not None

            # Test error cases
            with self.assertRaises(ValueError):
                BoozerSplineField(
                    filename_mhd_reduced,
                    ntheta=ntheta,
                    nzeta=nzeta,
                    comm=comm,
                )

    def test_boozersplinefield_angular_derivatives_integrate_to_zero(self):
        """
        Test that angular derivatives integrate to zero over their periods.
        """
        ntheta = 72
        nzeta = 72

        filename = filename_vac
        for spline_deriv in [True, False]:
            bsf = BoozerSplineField(
                filename,
                ntheta=ntheta,
                nzeta=nzeta,
                comm=comm,
                spline_deriv=spline_deriv,
                no_K=False,
            )
            tol = 1e-6

            ntheta_test = 48
            nzeta_test = 48
            thetas = np.linspace(0, 2 * np.pi, ntheta_test, endpoint=False)
            zetas = np.linspace(0, 2 * np.pi / bsf.nfp, nzeta_test, endpoint=False)
            [zetas, thetas] = np.meshgrid(zetas, thetas)
            points = np.zeros((len(thetas.flatten()), 3))
            points[:, 0] = 0.5 * np.ones_like(thetas.flatten())
            points[:, 1] = thetas.flatten()
            points[:, 2] = zetas.flatten()
            bsf.set_points(points)

            # Check that angular derivatives integrate to zero
            assert np.allclose(
                np.mean(bsf.dmodBdtheta().reshape(np.shape(thetas)), axis=0),
                0,
                atol=tol,
            )
            assert np.allclose(
                np.mean(bsf.dmodBdzeta().reshape(np.shape(thetas)), axis=1),
                0,
                atol=tol,
            )
            assert np.allclose(
                np.mean(bsf.dRdtheta().reshape(np.shape(thetas)), axis=0),
                0,
                atol=tol,
            )
            assert np.allclose(
                np.mean(bsf.dRdzeta().reshape(np.shape(thetas)), axis=1),
                0,
                atol=tol,
            )
            assert np.allclose(
                np.mean(bsf.dZdtheta().reshape(np.shape(thetas)), axis=0),
                0,
                atol=tol,
            )
            assert np.allclose(
                np.mean(bsf.dZdzeta().reshape(np.shape(thetas)), axis=1),
                0,
                atol=tol,
            )
            assert np.allclose(
                np.mean(bsf.dnudtheta().reshape(np.shape(thetas)), axis=0),
                0,
                atol=tol,
            )
            assert np.allclose(
                np.mean(bsf.dnudzeta().reshape(np.shape(thetas)), axis=1),
                0,
                atol=tol,
            )

    def test_boozersplinefield_consistency_with_radial_interpolant(self):
        """
        Test that BoozerSplineField gives consistent results with
        BoozerRadialInterpolant.
        """
        ntheta = 101
        nzeta = 100

        filename = filename_vac
        bri = BoozerRadialInterpolant(filename, order=3, comm=comm, no_K=False)
        for spline_deriv in [True, False]:
            tol = 1e-4

            bsf = BoozerSplineField(
                filename,
                ntheta=ntheta,
                nzeta=nzeta,
                comm=comm,
                spline_deriv=spline_deriv,
                no_K=False,
            )

            # Test at a set of points
            npoints = 10
            np.random.seed(42)
            points = np.random.uniform(size=(npoints, 3))
            points[:, 0] = points[:, 0] * 0.6 + 0.2  # s in [0.2, 0.8]
            points[:, 1] = points[:, 0] * np.pi  # theta in [0, pi]
            points[:, 2] = points[:, 0] * 2 * np.pi / bsf.nfp  # zeta in [0, 2*pi/nfp]

            bri.set_points(points)
            bsf.set_points(points)

            for field in [
                "modB",
                "R",
                "Z",
                "G",
                "iota",
                "nu",
                "dmodBdtheta",
                "dmodBdzeta",
                "dRdtheta",
                "dRdzeta",
                "dZdtheta",
                "dZdzeta",
                "dnudtheta",
                "dnudzeta",
                "dnuds",
                "diotads",
                "dGds",
                "dIds",
                "K",
            ]:
                field_bri = getattr(bri, field)()
                field_bsf = getattr(bsf, field)()
                assert np.allclose(field_bri, field_bsf, atol=tol)

    def test_boozersplinefield_derivs_methods(self):
        """
        Test that _derivs methods return consistent results.
        """
        ntheta = 21
        nzeta = 20

        filename = filename_vac
        bsf = BoozerSplineField(
            filename,
            ntheta=ntheta,
            nzeta=nzeta,
            comm=comm,
            no_K=False,
        )

        npoints = 10
        points = np.zeros((npoints, 3))
        points[:, 0] = np.linspace(0.2, 0.8, npoints)
        points[:, 1] = np.linspace(0, np.pi, npoints)
        points[:, 2] = np.linspace(0, 2 * np.pi / bsf.nfp, npoints)
        bsf.set_points(points)

        # Test modB_derivs
        modB_derivs = bsf.modB_derivs()
        assert np.allclose(modB_derivs[:, 0], bsf.dmodBds()[:, 0])
        assert np.allclose(modB_derivs[:, 1], bsf.dmodBdtheta()[:, 0])
        assert np.allclose(modB_derivs[:, 2], bsf.dmodBdzeta()[:, 0])

        # Test R_derivs
        R_derivs = bsf.R_derivs()
        assert np.allclose(R_derivs[:, 0], bsf.dRds()[:, 0])
        assert np.allclose(R_derivs[:, 1], bsf.dRdtheta()[:, 0])
        assert np.allclose(R_derivs[:, 2], bsf.dRdzeta()[:, 0])

        # Test Z_derivs
        Z_derivs = bsf.Z_derivs()
        assert np.allclose(Z_derivs[:, 0], bsf.dZds()[:, 0])
        assert np.allclose(Z_derivs[:, 1], bsf.dZdtheta()[:, 0])
        assert np.allclose(Z_derivs[:, 2], bsf.dZdzeta()[:, 0])

        # Test nu_derivs
        nu_derivs = bsf.nu_derivs()
        assert np.allclose(nu_derivs[:, 0], bsf.dnuds()[:, 0])
        assert np.allclose(nu_derivs[:, 1], bsf.dnudtheta()[:, 0])
        assert np.allclose(nu_derivs[:, 2], bsf.dnudzeta()[:, 0])


class TestingShearAlfvenWavesSuperposition(unittest.TestCase):
    @staticmethod
    def _harmonics(field, n):
        """
        A set of n harmonics on `field`, differing in mode numbers, frequency,
        phase and amplitude.

        """
        s = np.linspace(0.0, 1.0, 32)
        harmonics = []
        for i in range(n):
            m = 1 + (i % 5)
            phihat = 1e3 * s ** (m / 2) * (1 - s) / (i + 1)
            harmonics.append(
                ShearAlfvenHarmonic(
                    (s.tolist(), phihat.tolist()),
                    m,
                    1 + (i % 3),
                    136041.0 * (1 + 0.01 * i),
                    0.1 * i,
                    field,
                )
            )
        return harmonics

    def test_superposition_equals_sum_of_harmonics(self):
        """
        A superposition evaluates all of its harmonics in one pass rather than
        wave by wave, so check the result against summing the harmonics
        individually.
        """
        rng = np.random.default_rng(0)
        npts = 64
        points = np.ascontiguousarray(
            np.column_stack(
                [
                    rng.uniform(0.05, 0.95, npts),
                    rng.uniform(0.0, 2 * np.pi, npts),
                    rng.uniform(0.0, 2 * np.pi, npts),
                    rng.uniform(0.0, 1e-4, npts),
                ]
            )
        )

        for vacuum in (True, False):
            kw = {
                "etabar": 1.2,
                "B0": 5.0,
                "N": 0,
                "G0": 3.0,
                "psi0": 0.8,
                "iota0": 0.4,
            }
            if not vacuum:
                kw.update(I0=0.3, I1=0.05)
            field = BoozerAnalytic(**kw)
            if not vacuum:
                field.field_type = "nok"

            harmonics = self._harmonics(field, 6)
            saw = ShearAlfvenWavesSuperposition(harmonics)

            quantities = [
                "Phi",
                "dPhidpsi",
                "dPhidtheta",
                "dPhidzeta",
                "Phidot",
                "alpha",
                "alphadot",
                "dalphadpsi",
                "dalphadtheta",
                "dalphadzeta",
            ]

            saw.set_points(points)
            fused = {q: np.asarray(getattr(saw, q)()).copy() for q in quantities}

            expected = dict.fromkeys(quantities, 0.0)
            for harmonic in harmonics:
                harmonic.set_points(points)
                for q in quantities:
                    expected[q] = expected[q] + np.asarray(getattr(harmonic, q)())

            for q in quantities:
                np.testing.assert_allclose(
                    fused[q],
                    expected[q],
                    rtol=1e-12,
                    atol=0,
                    err_msg=f"{q} (vacuum={vacuum})",
                )

    def test_add_wave_refreshes_stored_quantities(self):
        """
        The superposition accumulates its harmonics into one set of arrays when
        set_points is called, and every accessor hands back one of those arrays
        directly. Adding a harmonic afterwards therefore has to re-evaluate
        them: without that, a read taken after add_wave silently reports the
        total from before the addition.
        """
        field = BoozerAnalytic(
            etabar=1.2, B0=5.0, N=0, G0=3.0, psi0=0.8, iota0=0.4, I0=0.3, I1=0.05
        )
        field.field_type = "nok"
        points = np.ascontiguousarray(np.array([[0.62, 0.3, 0.7, 1e-5]]))

        quantities = [
            "Phi",
            "dPhidpsi",
            "dPhidtheta",
            "dPhidzeta",
            "Phidot",
            "alpha",
            "alphadot",
            "dalphadpsi",
            "dalphadtheta",
            "dalphadzeta",
        ]

        first, second = self._harmonics(field, 2)
        saw = ShearAlfvenWavesSuperposition([first])
        saw.set_points(points)
        before = {q: np.asarray(getattr(saw, q)()).copy() for q in quantities}

        # No set_points between the addition and the reads below.
        saw.add_wave(second)
        after = {q: np.asarray(getattr(saw, q)()).copy() for q in quantities}

        expected = dict.fromkeys(quantities, 0.0)
        for h in (first, second):
            h.set_points(points)
            for q in quantities:
                expected[q] = expected[q] + np.asarray(getattr(h, q)())

        for q in quantities:
            np.testing.assert_allclose(
                after[q], expected[q], rtol=1e-12, atol=0, err_msg=q
            )
            # Guard against the assertion above passing on a quantity that the
            # second harmonic happens not to move.
            assert not np.array_equal(after[q], before[q]), (
                f"{q} is unchanged by add_wave, so it cannot distinguish a "
                f"refreshed total from a stale one"
            )


class TestInterpolatedShearAlfvenWave(unittest.TestCase):
    """
    Test InterpolatedShearAlfvenWave against original ShearAlfvenWavesSuperposition.

    This test compares all field quantities from InterpolatedShearAlfvenWave
    with the original ShearAlfvenWavesSuperposition to ensure interpolation
    accuracy.
    """

    def test_interpolated_shear_alfven_wave_accuracy(self):
        """Test that InterpolatedShearAlfvenWave matches original SAW."""
        # Test parameters
        resolution = 32  # Lower resolution for faster tests
        degree = 3
        boozmn_filename = str((TEST_DIR / "boozmn_beta2.5_QA.nc").resolve())
        ns_interp = resolution
        ntheta_interp = resolution
        nzeta_interp = resolution

        # Check if ae.npy exists in examples directory, otherwise skip test
        example_dir = (
            Path(__file__).parent / ".." / ".." / "examples" / "tracing_with_AE"
        )
        saw_filename = example_dir / "ae.npy"
        if not saw_filename.exists():
            self.skipTest(f"Test data file {saw_filename} not found")

        # Setup radial interpolation
        bri = BoozerRadialInterpolant(boozmn_filename, order=3, no_K=True, comm=comm)

        # Setup 3D interpolation for equilibrium field
        field = InterpolatedBoozerField(
            bri,
            degree,
            ns_interp=ns_interp,
            ntheta_interp=ntheta_interp,
            nzeta_interp=nzeta_interp,
        )

        # Load and setup original shear alfven wave
        saw = ShearAlfvenWavesSuperposition.from_ae3d(
            eigenvector=AE3DEigenvector.load_from_numpy(
                filename=str(saw_filename),
            ),
            B0=field,
            max_dB_normal_by_B0=5e-3,
            minor_radius_meters=1.7,
        )

        # Create interpolated version
        saw_interp = InterpolatedShearAlfvenWave(
            saw,
            degree,
            ns_interp=ns_interp,
            ntheta_interp=ntheta_interp,
            nzeta_interp=nzeta_interp,
            comm=comm,
        )

        # Create evaluation grid
        theta_grid = np.linspace(0, 2 * np.pi, 50)
        zeta_grid = np.linspace(0, 2 * np.pi, 50)
        THETA, ZETA = np.meshgrid(theta_grid, zeta_grid, indexing="ij")
        Npoints = THETA.shape[0] * THETA.shape[1]
        point = np.zeros((Npoints, 4))
        point[:, 0] = 0.5  # s = 0.5
        point[:, 1] = THETA.flatten()
        point[:, 2] = ZETA.flatten()
        # Random times
        np.random.seed(42)  # For reproducibility
        point[:, 3] = np.random.uniform(0, 1, Npoints)

        # Set points for both original and interpolated
        saw.set_points(point)
        saw_interp.set_points(point)

        # Test all field quantities
        field_names = [
            "Phi",
            "dPhidpsi",
            "dPhidtheta",
            "dPhidzeta",
            "Phidot",
            "alpha",
            "alphadot",
            "dalphadtheta",
            "dalphadpsi",
            "dalphadzeta",
        ]

        for field_name in field_names:
            # Evaluate interpolated version
            field_interp = getattr(saw_interp, field_name)()

            # Evaluate original version
            field_saw = getattr(saw, field_name)()

            # Compute normalized error
            mean = np.sqrt(np.mean(field_interp**2))
            if mean < 1e-12:
                # Field is essentially zero, check absolute error
                error = np.sqrt(np.mean((field_interp - field_saw) ** 2))
                assert error < 1e-6, (
                    f"{field_name}: absolute error = {error} "
                    f"(field is near zero, mean = {mean})"
                )
            else:
                # Normal field, check relative error
                error = np.sqrt(np.mean((field_interp - field_saw) ** 2))
                normalized_error = error / mean
                assert normalized_error < 1e-3, (
                    f"{field_name}: normalized error = {normalized_error} "
                    f"(error = {error}, mean = {mean})"
                )


if __name__ == "__main__":
    unittest.main()
