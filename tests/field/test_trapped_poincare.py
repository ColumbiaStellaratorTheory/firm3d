import os
import tempfile
import unittest
import warnings

import numpy as np

from firm3d.field.boozermagneticfield import BoozerAnalytic
from firm3d.trajectory_helpers import TrappedPoincare
from firm3d.trajectory_helpers._utils import chi_eta_to_theta_zeta
from firm3d.trajectory_helpers.poincare import TRAPPED_MAP_OUTCOMES
from firm3d.util.constants import (
    ALPHA_PARTICLE_CHARGE,
    ALPHA_PARTICLE_MASS,
    FUSION_ALPHA_PARTICLE_ENERGY,
)

try:
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
except ImportError:
    comm = None

B0 = 5.7
NS = 4
NETA = 2


def qh_field(perturbation):
    """Analytic QH field (N=4) with an m=1, n=0 symmetry-breaking mode."""
    return BoozerAnalytic(
        etabar=0.12,
        B0=B0,
        N=4,
        G0=8 * B0,
        psi0=B0 * 1.7**2 / 2,
        iota0=1.2,
        Bbar=B0,
        B0z=[perturbation * B0],
        n=[0],
        m=[1],
    )


def trapped_map(field, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return TrappedPoincare(
            field,
            1,
            4,
            ALPHA_PARTICLE_MASS,
            ALPHA_PARTICLE_CHARGE,
            FUSION_ALPHA_PARTICLE_ENERGY,
            lam=1 / (B0 * (1 + 0.12 * 1.2 * 0.9)),
            ns_poinc=NS,
            neta_poinc=NETA,
            comm=comm,
            tmax=1e-4,
            solver_options={"reltol": 1e-6, "abstol": 1e-6, "axis": 0},
            **{"Nmaps": 2, **kwargs},
        )


class TrappedPoincareOutcomeTests(unittest.TestCase):
    def test_outcomes_quasisymmetric(self):
        poinc = trapped_map(qh_field(0.0))
        reasons = poinc.outcomes["reason"]
        self.assertEqual(len(reasons), NS * NETA)
        self.assertTrue(set(reasons) <= set(TRAPPED_MAP_OUTCOMES))
        # Bcrit exceeds max |B| on the inner surfaces
        inner = poinc.outcomes["s_init"] < 0.5
        self.assertTrue(np.all(reasons[inner] == "surface_B_below_Bcrit"))
        self.assertTrue(np.all(reasons[~inner] == "completed"))
        self.assertEqual(poinc.outcome_counts()["completed"], len(poinc.s_all))
        completed = reasons == "completed"
        self.assertTrue(np.all(poinc.outcomes["nmaps"][completed] == 2))

    def test_outcomes_symmetry_breaking(self):
        poinc = trapped_map(qh_field(0.5))
        counts = poinc.outcome_counts()
        self.assertEqual(sum(counts.values()), NS * NETA)
        self.assertGreater(counts["lost_inner"], 0)
        self.assertGreater(counts["lost_outer"], 0)
        self.assertEqual(counts.get("completed", 0), len(poinc.s_all))
        with tempfile.TemporaryDirectory() as tmp:
            filename = os.path.join(tmp, "trapped_poincare.png")
            ax = poinc.plot_poincare(filename=filename)
            labels = [text.get_text() for text in ax.get_legend().get_texts()]
            self.assertEqual(len(labels), len(counts) - ("completed" in counts))
            self.assertTrue(os.path.exists(filename))

    def test_modB_range(self):
        poinc = trapped_map(qh_field(0.0), Nmaps=1)
        for s in [0.1, 0.5, 0.9]:
            r = 1.7 * np.sqrt(s)
            Bmin, Bmax = poinc.modB_range(s)
            self.assertAlmostEqual(Bmin, B0 * (1 - 0.12 * r), places=6)
            self.assertAlmostEqual(Bmax, B0 * (1 + 0.12 * r), places=6)
        with tempfile.TemporaryDirectory() as tmp:
            filename = os.path.join(tmp, "modB_range.png")
            poinc.plot_modB_range(filename=filename, ns=5)
            self.assertTrue(os.path.exists(filename))

    def test_field_line_well_minimum(self):
        poinc = trapped_map(qh_field(0.0), Nmaps=1)
        for s in [0.2, 0.6]:
            for alpha in [0.0, 1.0, 4.0]:
                theta, zeta, Bmin = poinc.field_line_well_minimum(s, alpha)
                self.assertAlmostEqual(
                    Bmin, B0 * (1 - 0.12 * 1.7 * np.sqrt(s)), places=6
                )
                self.assertAlmostEqual(np.cos(theta - 4 * zeta), -1.0, places=6)

    def test_field_line_wells(self):
        poinc = trapped_map(qh_field(0.0), Nmaps=1)
        with tempfile.TemporaryDirectory() as tmp:
            wells = poinc.plot_field_line_wells(
                filename=os.path.join(tmp, "wells.png"), ns=4, nalpha=3
            )
        r = 1.7 * np.sqrt(wells["s"])[:, None]
        self.assertTrue(np.allclose(wells["Bmin"], B0 * (1 - 0.12 * r), atol=1e-6))
        for key in ("Bmax_low", "Bmax_high"):
            self.assertTrue(np.allclose(wells[key], B0 * (1 + 0.12 * r), atol=1e-4))
        # QS: every field line on a surface is alike, trapped above s ~ 0.4
        expected = np.where(wells["s"] < 0.5, 3, 1)[:, None]
        self.assertTrue(np.all(wells["category"] == expected))

    def test_one_sided_fraction(self):
        poinc = trapped_map(qh_field(0.0), Nmaps=1)
        with tempfile.TemporaryDirectory() as tmp:
            Bcrits, fractions = poinc.plot_one_sided_fraction(
                filename=os.path.join(tmp, "one_sided.png"), ns=4, nalpha=3
            )
        self.assertTrue(np.allclose(fractions.sum(axis=1), 100))
        # QS: the two maxima bounding each well are equal, so nothing is one-sided
        self.assertTrue(np.allclose(fractions[:, 2], 0))

    def test_trace_mirror_init(self):
        poinc = trapped_map(qh_field(0.0), mirror_init="trace")
        reasons = poinc.outcomes["reason"]
        self.assertEqual(len(reasons), NS * NETA)
        # Bcrit exceeds max |B| within an orbit width of s = 0.2
        self.assertEqual(
            poinc.outcome_counts(), {"completed": 6, "surface_B_below_Bcrit": 2}
        )
        below = reasons == "surface_B_below_Bcrit"
        self.assertTrue(np.allclose(poinc.outcomes["s_launch"][below], 0.2))
        completed = reasons == "completed"
        self.assertTrue(
            np.allclose(
                sorted(poinc.outcomes["s_launch"][completed]),
                [0.4] * 2 + [0.6] * 2 + [0.8] * 2,
            )
        )
        theta, zeta = chi_eta_to_theta_zeta(
            np.array(poinc.chis_init), np.array(poinc.etas_init), 1, 4, 0, 1
        )
        for s, th, ze in zip(poinc.s_init, theta, zeta):
            poinc.field.set_points(np.array([[s, th, ze]]))
            self.assertAlmostEqual(
                poinc.field.modB()[0, 0] / poinc.modBcrit, 1.0, places=5
            )


if __name__ == "__main__":
    unittest.main()
