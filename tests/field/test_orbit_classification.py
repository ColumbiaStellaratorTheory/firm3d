import os
import tempfile
import unittest

import numpy as np

from firm3d.field.boozermagneticfield import BoozerAnalytic
from firm3d.plotting.orbit_classification import OrbitClassification
from firm3d.util.constants import (
    ALPHA_PARTICLE_CHARGE,
    ALPHA_PARTICLE_MASS,
    FUSION_ALPHA_PARTICLE_ENERGY,
)

B0 = 5.7
ETABAR = 0.12
A = 1.7


class PhaseSpaceFractionTests(unittest.TestCase):
    def test_quasisymmetric(self):
        field = BoozerAnalytic(
            etabar=ETABAR,
            B0=B0,
            N=4,
            G0=8 * B0,
            psi0=B0 * A**2 / 2,
            iota0=1.2,
            Bbar=B0,
        )
        oc = OrbitClassification(
            field,
            FUSION_ALPHA_PARTICLE_ENERGY,
            ALPHA_PARTICLE_MASS,
            ALPHA_PARTICLE_CHARGE,
            1,
            4,
        )
        s_grid = [0.2, 0.6]
        with tempfile.TemporaryDirectory() as tmp:
            data = os.path.join(tmp, "fractions.txt")
            fr = oc.phase_space_fractions(
                s_grid,
                nalpha=2,
                nb=200,
                filename=os.path.join(tmp, "fractions.png"),
                data_filename=data,
            )
            saved = np.loadtxt(data)
        self.assertTrue(np.allclose(saved[:, 1], fr["banana"]))
        total = fr["banana"] + fr["barely"] + fr["ripple"] + fr["passing"]
        self.assertTrue(np.allclose(total, 1))
        # QS: one well per period, no ripple or barely trapped phase space
        self.assertTrue(np.allclose(fr["ripple"], 0))
        self.assertTrue(np.allclose(fr["barely"], 0))
        # Trapped fraction <sqrt(1 - B/Bmax)> with weight 1/B^2
        chi = np.linspace(0, 2 * np.pi, 4001)
        for s, banana in zip(s_grid, fr["banana"]):
            B = B0 * (1 + ETABAR * A * np.sqrt(s) * np.cos(chi))
            w = 1 / B**2
            expected = np.trapezoid(w * np.sqrt(1 - B / B.max()), chi) / np.trapezoid(
                w, chi
            )
            self.assertAlmostEqual(banana, expected, delta=0.01)


if __name__ == "__main__":
    unittest.main()
