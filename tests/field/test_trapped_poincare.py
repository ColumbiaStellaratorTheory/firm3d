import os
import tempfile
import unittest
import warnings

import numpy as np

from firm3d.field.boozermagneticfield import BoozerAnalytic
from firm3d.trajectory_helpers import TrappedPoincare
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


def trapped_map(field):
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
            Nmaps=2,
            comm=comm,
            tmax=1e-4,
            solver_options={"reltol": 1e-6, "abstol": 1e-6, "axis": 0},
        )


class TrappedPoincareOutcomeTests(unittest.TestCase):
    def test_outcomes_quasisymmetric(self):
        poinc = trapped_map(qh_field(0.0))
        reasons = poinc.outcomes["reason"]
        self.assertEqual(len(reasons), NS * NETA)
        self.assertTrue(set(reasons) <= set(TRAPPED_MAP_OUTCOMES))
        # Bcrit exceeds max |B| on the inner surfaces
        inner = poinc.outcomes["s_init"] < 0.5
        self.assertTrue(np.all(reasons[inner] == "no_mirror_B_below"))
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


if __name__ == "__main__":
    unittest.main()
