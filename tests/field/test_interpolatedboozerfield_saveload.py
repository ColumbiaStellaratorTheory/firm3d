"""
Tests for InterpolatedBoozerField JSON save/load functionality.

Verifies that:
- Field attributes (psi0, nfp, stellsym, grid ranges) are preserved
- Interpolation rule (degree, nodes, scalings) is preserved
- Field evaluations (modB, G, iota, etc.) match after load
- Status flags are correctly set to prevent recomputation
- Tracing is preserved: test_tracing_identical_after_load and
  test_tracing_perturbed_identical_after_load run the same particle tracing
  with the original field and with a field loaded from JSON, then assert the
  trajectories match (unperturbed and SAW-perturbed, as in the fusion_distribution
  examples).
Test configurations cover vacuum and MHD equilibria with different symmetries.

Many tests use try/finally when writing to a temporary JSON file: the try block
runs the test (save field, load field, run assertions); the finally block
deletes the temp file. The finally runs even when the test fails or raises,
so we never leave temporary JSON files on disk.

Pass/fail: Assertions (assertEqual, assert_allclose, etc.) in the try block
determine whether the test passes. If one fails, it raises AssertionError;
that exception propagates after the finally block runs (cleanup only). We do
not catch exceptions, so the test runner correctly reports FAILED.
"""

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from firm3d.field.boozermagneticfield import (
    BoozerRadialInterpolant,
    InterpolatedBoozerField,
    ShearAlfvenHarmonic,
)
from firm3d.field.tracing import (
    MaxToroidalFluxStoppingCriterion,
    trace_particles_boozer,
    trace_particles_boozer_perturbed,
)
from firm3d.field.tracing_helpers import (
    initialize_position_profile,
    initialize_velocity_uniform,
)
from firm3d.util.constants import (
    ALPHA_PARTICLE_CHARGE,
    ALPHA_PARTICLE_MASS,
    FUSION_ALPHA_PARTICLE_ENERGY,
)

# Path to test_files directory containing equilibrium data files
TEST_DIR = (Path(__file__).parent / ".." / "test_files").resolve()

TEST_CONFIGS = {
    "vac_qa": {
        "file": str((TEST_DIR / "boozmn_LandremanPaul2021_QA_lowres.nc").resolve()),
        "wout_file": str((TEST_DIR / "wout_LandremanPaul2021_QA_lowres.nc").resolve()),
        "nfp": 4,
        "stellsym": True,
        "order": 3,
    },
    "mhd_sym": {
        "file": str((TEST_DIR / "boozmn_n3are_R7.75B5.7.nc").resolve()),
        "wout_file": str((TEST_DIR / "wout_n3are_R7.75B5.7.nc").resolve()),
        "reduced_file": str((TEST_DIR / "boozmn_n3are_R7.75B5.7_reduced.nc").resolve()),
        "reordered_file": str(
            (TEST_DIR / "boozmn_n3are_R7.75B5.7_reordered.nc").resolve()
        ),
        "nfp": 3,
        "stellsym": True,
        "order": 3,
    },
    "mhd_asym": {
        "file": str((TEST_DIR / "boozmn_ITERModel_reference.nc").resolve()),
        "wout_file": str((TEST_DIR / "wout_ITERModel_reference.nc").resolve()),
        "reduced_file": str(
            (TEST_DIR / "boozmn_ITERModel_reference_reduced.nc").resolve()
        ),
        "reordered_file": str(
            (TEST_DIR / "boozmn_ITERModel_reference_reordered.nc").resolve()
        ),
        "nfp": 3,
        "stellsym": True,  # Note: Despite name, file is actually symmetric
        "order": 3,
    },
}

# MPI communicator (if available)
try:
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
except ImportError:
    comm = None


class TestInterpolatedBoozerFieldSaveLoad(unittest.TestCase):
    """Test JSON save/load for InterpolatedBoozerField."""

    def _create_field(self, config, degree=3, grid_size=6):
        """Create an InterpolatedBoozerField from config."""
        bri = BoozerRadialInterpolant(config["file"], config["order"], comm=comm)

        n = config["nfp"]
        smin, smax, ssteps = 0.2, 0.8, grid_size
        thetamin, thetamax, thetasteps = 0, np.pi, grid_size
        zetamin, zetamax, zetasteps = 0, 2 * np.pi / n, grid_size * 2

        field = InterpolatedBoozerField(
            bri,
            degree,
            [smin, smax, ssteps],
            [thetamin, thetamax, thetasteps],
            [zetamin, zetamax, zetasteps],
            True,  # extrapolate
        )

        grid_ranges = {
            "smin": smin,
            "smax": smax,
            "thetamin": thetamin,
            "thetamax": thetamax,
            "zetamin": zetamin,
            "zetamax": zetamax,
        }

        return bri, field, grid_ranges

    def _verify_scalar_attributes(self, field1, field2, msg_prefix=""):
        """Verify that scalar attributes match between two fields."""
        self.assertAlmostEqual(
            field1.psi0, field2.psi0, places=12, msg=f"{msg_prefix}psi0 mismatch"
        )
        self.assertEqual(
            field1.get_nfp(), field2.get_nfp(), msg=f"{msg_prefix}nfp mismatch"
        )
        self.assertEqual(
            field1.get_stellsym(),
            field2.get_stellsym(),
            msg=f"{msg_prefix}stellsym mismatch",
        )
        self.assertEqual(
            field1.get_extrapolate(),
            field2.get_extrapolate(),
            msg=f"{msg_prefix}extrapolate mismatch",
        )
        self.assertEqual(
            field1.field_type, field2.field_type, msg=f"{msg_prefix}field_type mismatch"
        )

    def _verify_rule_attributes(self, field1, field2, msg_prefix=""):
        """Verify that interpolation rule attributes match."""
        self.assertEqual(
            field1.rule.degree,
            field2.rule.degree,
            msg=f"{msg_prefix}rule.degree mismatch",
        )
        np.testing.assert_allclose(
            field1.rule.nodes,
            field2.rule.nodes,
            rtol=1e-12,
            atol=1e-14,
            err_msg=f"{msg_prefix}rule.nodes mismatch",
        )
        np.testing.assert_allclose(
            field1.rule.scalings,
            field2.rule.scalings,
            rtol=1e-12,
            atol=1e-14,
            err_msg=f"{msg_prefix}rule.scalings mismatch",
        )

    # Every status flag exposed by InterpolatedBoozerField.  The callable
    # method name is the flag name with the "status_" prefix stripped.
    ALL_STATUS_FLAGS = [
        "status_modB",
        "status_dmodBdtheta",
        "status_dmodBdzeta",
        "status_dmodBds",
        "status_modB_derivs",
        "status_G",
        "status_I",
        "status_iota",
        "status_psip",
        "status_dGds",
        "status_dIds",
        "status_diotads",
        "status_K",
        "status_dKdtheta",
        "status_dKdzeta",
        "status_K_derivs",
        "status_nu",
        "status_dnudtheta",
        "status_dnudzeta",
        "status_dnuds",
        "status_nu_derivs",
        "status_R",
        "status_dRdtheta",
        "status_dRdzeta",
        "status_dRds",
        "status_R_derivs",
        "status_Z",
        "status_dZdtheta",
        "status_dZdzeta",
        "status_dZds",
        "status_Z_derivs",
    ]

    def _verify_evaluations(
        self,
        field1,
        field2,
        grid_ranges,
        num_points=100,
        msg_prefix="",
    ):
        """Verify field evaluations match at random points.

        Discovers which quantities were actually computed by checking every
        status flag on field1.  For each True flag, asserts the loaded field
        also has it True, then compares evaluated values.  This way the test
        automatically covers exactly the quantities that were computed and
        saved, with no hardcoded list to keep in sync.
        """
        # Discover which quantities were computed on the original field
        computed = []
        for flag in self.ALL_STATUS_FLAGS:
            if getattr(field1, flag, False):
                self.assertTrue(
                    getattr(field2, flag, False),
                    f"{msg_prefix}{flag} is True on original but False on loaded field",
                )
                computed.append(flag[len("status_") :])  # strip prefix -> method name

        self.assertGreater(
            len(computed), 0, f"{msg_prefix}No quantities were computed on field1"
        )

        np.random.seed(42)
        margin = 0.05
        s_vals = np.random.uniform(
            grid_ranges["smin"] + margin, grid_ranges["smax"] - margin, num_points
        )
        theta_vals = np.random.uniform(
            grid_ranges["thetamin"] + margin,
            grid_ranges["thetamax"] - margin,
            num_points,
        )
        zeta_vals = np.random.uniform(
            grid_ranges["zetamin"] + margin, grid_ranges["zetamax"] - margin, num_points
        )

        points = np.column_stack([s_vals, theta_vals, zeta_vals])
        field1.set_points(points)
        field2.set_points(points)

        for qty in computed:
            val1 = getattr(field1, qty)()
            val2 = getattr(field2, qty)()
            np.testing.assert_allclose(
                val1,
                val2,
                rtol=1e-12,
                atol=1e-14,
                err_msg=f"{msg_prefix}{qty} values differ",
            )

    def _verify_status_flags(self, field1, field2, msg_prefix=""):
        """Verify that every status flag matches between two fields."""
        for flag in self.ALL_STATUS_FLAGS:
            val1 = getattr(field1, flag)
            val2 = getattr(field2, flag)
            self.assertEqual(
                val1, val2, msg=f"{msg_prefix}{flag} mismatch: {val1} vs {val2}"
            )

    def test_saveload_vac_qa(self):
        """Test save/load with vacuum QA equilibrium."""
        config = TEST_CONFIGS["vac_qa"]
        if not os.path.exists(config["file"]):
            self.skipTest(f"Test file not found: {config['file']}")

        json_path = None
        try:
            bri, field, grid_ranges = self._create_field(config)

            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
                json_path = f.name
            field.to_json(json_path)
            loaded_field = InterpolatedBoozerField.from_json(json_path)

            self._verify_scalar_attributes(field, loaded_field, "vac_qa: ")
            self._verify_rule_attributes(field, loaded_field, "vac_qa: ")
            self._verify_evaluations(
                field, loaded_field, grid_ranges, msg_prefix="vac_qa: "
            )
        finally:
            if json_path and os.path.exists(json_path):
                os.remove(json_path)

    def test_saveload_mhd_sym(self):
        """Test save/load with MHD symmetric equilibrium."""
        config = TEST_CONFIGS["mhd_sym"]
        if not os.path.exists(config["file"]):
            self.skipTest(f"Test file not found: {config['file']}")

        json_path = None
        try:
            bri, field, grid_ranges = self._create_field(config)

            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
                json_path = f.name
            field.to_json(json_path)

            loaded_field = InterpolatedBoozerField.from_json(json_path)

            self._verify_scalar_attributes(field, loaded_field, "mhd_sym: ")
            self._verify_evaluations(
                field, loaded_field, grid_ranges, msg_prefix="mhd_sym: "
            )

        finally:
            if json_path and os.path.exists(json_path):
                os.remove(json_path)

    def test_all_field_quantities(self):
        """Test that every computed quantity is preserved for each config.

        Iterates over all TEST_CONFIGS so that vac, nok, and general field
        types are each verified with their full set of computed quantities.
        _verify_evaluations discovers the set automatically via status flags.
        """
        for config_name, config in TEST_CONFIGS.items():
            if not os.path.exists(config["file"]):
                continue  # skip unavailable equilibria

            json_path = None
            try:
                bri, field, grid_ranges = self._create_field(config)

                with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
                    json_path = f.name
                field.to_json(json_path)

                loaded_field = InterpolatedBoozerField.from_json(json_path)

                self._verify_evaluations(
                    field,
                    loaded_field,
                    grid_ranges,
                    num_points=50,
                    msg_prefix=f"all_quantities({config_name}): ",
                )

            finally:
                if json_path and os.path.exists(json_path):
                    os.remove(json_path)

    def test_tracing_identical_after_load(self):
        """
        Unperturbed tracing: trace one particle with original field, save
        trajectory; repeat with field loaded from JSON; compare trajectories.
        Uses the full s range [0, 1] (like the examples) so that
        initialize_position_profile can sample the whole domain.
        Small resolution and grid sizes keep field construction and
        rejection sampling fast.
        """
        config = TEST_CONFIGS["vac_qa"]
        if not os.path.exists(config["file"]):
            self.skipTest(f"Test file not found: {config['file']}")

        n_particles = 1
        tmax = 1e-2
        reactivity = lambda s: (1 - s**5) ** 2

        # Build field with full s range [0, 1] and higher resolution
        bri = BoozerRadialInterpolant(config["file"], config["order"], comm=comm)
        field = InterpolatedBoozerField(
            bri,
            degree=3,
            ns_interp=24,
            ntheta_interp=24,
            nzeta_interp=24,
        )

        # Small grid sizes (10x10x10 = 1000 pts) instead of default 100^3
        points = initialize_position_profile(
            field,
            n_particles,
            reactivity,
            ns_max=10,
            ntheta_max=10,
            nzeta_max=10,
            comm=comm,
            seed=42,
        )
        Ekin = FUSION_ALPHA_PARTICLE_ENERGY
        mass = ALPHA_PARTICLE_MASS
        charge = ALPHA_PARTICLE_CHARGE
        v0 = np.sqrt(2 * Ekin / mass)
        vpar_init = initialize_velocity_uniform(
            v0,
            n_particles,
            comm=comm,
            seed=42,
        )

        res_tys_orig, res_hits_orig = trace_particles_boozer(
            field,
            points,
            vpar_init,
            tmax=tmax,
            mass=mass,
            charge=charge,
            comm=comm,
            Ekin=Ekin,
            stopping_criteria=[MaxToroidalFluxStoppingCriterion(1.0)],
            forget_exact_path=True,
            abstol=1e-10,
            reltol=1e-10,
        )

        json_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
                json_path = f.name
            field.to_json(json_path)
            loaded_field = InterpolatedBoozerField.from_json(json_path)

            res_tys_loaded, res_hits_loaded = trace_particles_boozer(
                loaded_field,
                points,
                vpar_init,
                tmax=tmax,
                mass=mass,
                charge=charge,
                comm=comm,
                Ekin=Ekin,
                stopping_criteria=[MaxToroidalFluxStoppingCriterion(1.0)],
                forget_exact_path=True,
                abstol=1e-10,
                reltol=1e-10,
            )

            self.assertEqual(
                len(res_tys_orig),
                len(res_tys_loaded),
                msg="Tracing with loaded field must produce same trajectories",
            )
            for i, (ty_orig, ty_loaded) in enumerate(zip(res_tys_orig, res_tys_loaded)):
                np.testing.assert_allclose(
                    ty_orig,
                    ty_loaded,
                    rtol=1e-11,
                    atol=1e-13,
                    err_msg=f"Tracing trajectory {i} differs after load",
                )
        finally:
            if json_path and os.path.exists(json_path):
                os.remove(json_path)

    def test_tracing_perturbed_identical_after_load(self):
        """
        Perturbed tracing: trace one particle with original SAW field, then
        with field loaded from JSON; compare trajectories.
        Uses the full s range [0, 1] (like the examples) so that
        initialize_position_profile can sample the whole domain.
        Small resolution and grid sizes keep field construction and
        rejection sampling fast.
        """
        config = TEST_CONFIGS["vac_qa"]
        if not os.path.exists(config["file"]):
            self.skipTest(f"Test file not found: {config['file']}")

        n_particles = 1
        tmax = 1e-2
        Phihat = -1.5e3
        Phim, Phin, omega, phase = 1, 1, 1e5, 0.0
        reactivity = lambda s: (1 - s**5) ** 2

        # Build field with full s range [0, 1] and higher resolution
        bri = BoozerRadialInterpolant(config["file"], config["order"], comm=comm)
        field = InterpolatedBoozerField(
            bri,
            degree=3,
            ns_interp=24,
            ntheta_interp=24,
            nzeta_interp=24,
        )

        # Small grid sizes (10x10x10 = 1000 pts) instead of default 100^3
        points = initialize_position_profile(
            field,
            n_particles,
            reactivity,
            ns_max=10,
            ntheta_max=10,
            nzeta_max=10,
            comm=comm,
            seed=42,
        )
        Ekin = FUSION_ALPHA_PARTICLE_ENERGY
        mass = ALPHA_PARTICLE_MASS
        charge = ALPHA_PARTICLE_CHARGE
        v0 = np.sqrt(2 * Ekin / mass)
        vpar_init = initialize_velocity_uniform(
            v0,
            n_particles,
            comm=comm,
            seed=42,
        )
        field.set_points(points)
        mu_init = (v0**2 - vpar_init**2) / (2 * field.modB()[:, 0])

        saw_orig = ShearAlfvenHarmonic(Phihat, Phim, Phin, omega, phase, field)
        res_tys_orig, res_hits_orig = trace_particles_boozer_perturbed(
            saw_orig,
            points,
            vpar_init,
            mu_init,
            mass=mass,
            charge=charge,
            comm=comm,
            stopping_criteria=[MaxToroidalFluxStoppingCriterion(1.0)],
            forget_exact_path=True,
            abstol=1e-10,
            reltol=1e-10,
            tmax=tmax,
        )

        json_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
                json_path = f.name
            field.to_json(json_path)
            loaded_field = InterpolatedBoozerField.from_json(json_path)
            loaded_field.set_points(points)
            mu_loaded = (v0**2 - vpar_init**2) / (2 * loaded_field.modB()[:, 0])
            saw_loaded = ShearAlfvenHarmonic(
                Phihat, Phim, Phin, omega, phase, loaded_field
            )

            res_tys_loaded, res_hits_loaded = trace_particles_boozer_perturbed(
                saw_loaded,
                points,
                vpar_init,
                mu_loaded,
                mass=mass,
                charge=charge,
                comm=comm,
                stopping_criteria=[MaxToroidalFluxStoppingCriterion(1.0)],
                forget_exact_path=True,
                abstol=1e-10,
                reltol=1e-10,
                tmax=tmax,
            )

            self.assertEqual(
                len(res_tys_orig),
                len(res_tys_loaded),
                msg="Perturbed loaded field tracing must produce same trajectories",
            )
            for i, (ty_orig, ty_loaded) in enumerate(zip(res_tys_orig, res_tys_loaded)):
                np.testing.assert_allclose(
                    ty_orig,
                    ty_loaded,
                    rtol=1e-11,
                    atol=1e-13,
                    err_msg=f"Perturbed tracing trajectory {i} differs after load",
                )
        finally:
            if json_path and os.path.exists(json_path):
                os.remove(json_path)

    def test_status_flags_preserved(self):
        """Test that status flags are correctly preserved through save/load."""
        config = TEST_CONFIGS["vac_qa"]
        if not os.path.exists(config["file"]):
            self.skipTest(f"Test file not found: {config['file']}")

        json_path = None
        try:
            bri, field, grid_ranges = self._create_field(config)

            # Trigger computation of some quantities
            points = np.array([[0.5, 1.0, 0.5]])
            field.set_points(points)
            _ = field.modB()  # This sets status_modB = True
            _ = field.G()  # This sets status_G = True
            _ = field.iota()  # This sets status_iota = True

            # Verify status flags are set
            self.assertTrue(field.status_modB, "status_modB True after modB()")
            self.assertTrue(field.status_G, "status_G True after G()")
            self.assertTrue(field.status_iota, "status_iota True after iota()")

            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
                json_path = f.name
            field.to_json(json_path)

            loaded_field = InterpolatedBoozerField.from_json(json_path)

            # Verify status flags are preserved
            self.assertTrue(
                loaded_field.status_modB, "status_modB should be True after loading"
            )
            self.assertTrue(
                loaded_field.status_G, "status_G should be True after loading"
            )
            self.assertTrue(
                loaded_field.status_iota, "status_iota should be True after loading"
            )

        finally:
            if json_path and os.path.exists(json_path):
                os.remove(json_path)

    def test_different_grid_resolutions(self):
        """Test save/load with different grid resolutions."""
        config = TEST_CONFIGS["vac_qa"]
        if not os.path.exists(config["file"]):
            self.skipTest(f"Test file not found: {config['file']}")

        json_path = None
        for grid_size in [4, 8]:
            try:
                bri, field, grid_ranges = self._create_field(
                    config, degree=3, grid_size=grid_size
                )

                with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
                    json_path = f.name
                field.to_json(json_path)

                loaded_field = InterpolatedBoozerField.from_json(json_path)

                self._verify_scalar_attributes(
                    field, loaded_field, f"grid_{grid_size}: "
                )
                self._verify_evaluations(
                    field,
                    loaded_field,
                    grid_ranges,
                    num_points=30,
                    msg_prefix=f"grid_{grid_size}: ",
                )

            finally:
                if json_path and os.path.exists(json_path):
                    os.remove(json_path)

    def test_different_degrees(self):
        """Test save/load with different polynomial degrees."""
        config = TEST_CONFIGS["vac_qa"]
        if not os.path.exists(config["file"]):
            self.skipTest(f"Test file not found: {config['file']}")

        json_path = None
        for degree in [2, 4]:
            try:
                bri, field, grid_ranges = self._create_field(
                    config, degree=degree, grid_size=5
                )

                with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
                    json_path = f.name
                field.to_json(json_path)

                loaded_field = InterpolatedBoozerField.from_json(json_path)

                # Verify degree is preserved
                self.assertEqual(
                    field.rule.degree,
                    loaded_field.rule.degree,
                    f"degree_{degree}: rule.degree mismatch",
                )

                self._verify_evaluations(
                    field,
                    loaded_field,
                    grid_ranges,
                    num_points=30,
                    msg_prefix=f"degree_{degree}: ",
                )

            finally:
                if json_path and os.path.exists(json_path):
                    os.remove(json_path)

    def test_json_file_exists_and_readable(self):
        """Test that JSON file is created and has non-zero size."""
        config = TEST_CONFIGS["vac_qa"]
        if not os.path.exists(config["file"]):
            self.skipTest(f"Test file not found: {config['file']}")

        json_path = None
        try:
            bri, field, _ = self._create_field(config, grid_size=4)

            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
                json_path = f.name

            field.to_json(json_path)

            # Verify file exists and has content
            self.assertTrue(os.path.exists(json_path), "JSON file should exist")
            file_size = os.path.getsize(json_path)
            self.assertGreater(file_size, 0, "JSON file should have non-zero size")

        finally:
            if json_path and os.path.exists(json_path):
                os.remove(json_path)

    def test_load_nonexistent_file(self):
        """Test that loading from non-existent file raises an error."""
        with self.assertRaises(RuntimeError):
            InterpolatedBoozerField.from_json("/nonexistent/path/field.json")

    def test_multiple_save_load_cycles(self):
        """Test that multiple save/load cycles preserve data integrity."""
        config = TEST_CONFIGS["vac_qa"]
        if not os.path.exists(config["file"]):
            self.skipTest(f"Test file not found: {config['file']}")

        json_path1 = None
        json_path2 = None
        try:
            # Create original field
            bri, field, grid_ranges = self._create_field(config, grid_size=4)

            # First save/load cycle
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
                json_path1 = f.name
            field.to_json(json_path1)
            loaded_field1 = InterpolatedBoozerField.from_json(json_path1)

            # Second save/load cycle
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
                json_path2 = f.name
            loaded_field1.to_json(json_path2)
            loaded_field2 = InterpolatedBoozerField.from_json(json_path2)

            # Verify original and twice-loaded fields match
            self._verify_scalar_attributes(field, loaded_field2, "multi_cycle: ")
            self._verify_evaluations(
                field,
                loaded_field2,
                grid_ranges,
                num_points=50,
                msg_prefix="multi_cycle: ",
            )

        finally:
            for path in [json_path1, json_path2]:
                if path and os.path.exists(path):
                    os.remove(path)

    def test_saveload_with_wout_file(self):
        """Test save/load when using wout file as input."""
        config = TEST_CONFIGS["vac_qa"]
        wout_file = config.get("wout_file")
        if not wout_file or not os.path.exists(wout_file):
            self.skipTest(f"Wout file not found: {wout_file}")

        json_path = None
        try:
            # Create field from wout file
            bri = BoozerRadialInterpolant(wout_file, config["order"], comm=comm)

            n = config["nfp"]
            field = InterpolatedBoozerField(
                bri,
                3,
                [0.2, 0.8, 5],
                [0, np.pi, 5],
                [0, 2 * np.pi / n, 8],
                True,
            )

            grid_ranges = {
                "smin": 0.2,
                "smax": 0.8,
                "thetamin": 0,
                "thetamax": np.pi,
                "zetamin": 0,
                "zetamax": 2 * np.pi / n,
            }

            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
                json_path = f.name
            field.to_json(json_path)

            loaded_field = InterpolatedBoozerField.from_json(json_path)

            self._verify_scalar_attributes(field, loaded_field, "wout: ")
            self._verify_evaluations(
                field, loaded_field, grid_ranges, num_points=30, msg_prefix="wout: "
            )

        finally:
            if json_path and os.path.exists(json_path):
                os.remove(json_path)

    def test_saveload_mhd_asym(self):
        """Test save/load with MHD asymmetric equilibrium."""
        config = TEST_CONFIGS["mhd_asym"]
        if not os.path.exists(config["file"]):
            self.skipTest(f"Test file not found: {config['file']}")

        json_path = None
        try:
            bri, field, grid_ranges = self._create_field(config)

            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
                json_path = f.name
            field.to_json(json_path)

            loaded_field = InterpolatedBoozerField.from_json(json_path)

            self._verify_scalar_attributes(field, loaded_field, "mhd_asym: ")
            self._verify_evaluations(
                field, loaded_field, grid_ranges, msg_prefix="mhd_asym: "
            )

        finally:
            if json_path and os.path.exists(json_path):
                os.remove(json_path)

    def test_saveload_with_wout_mhd_sym(self):
        """Test save/load when using mhd_sym wout file as input."""
        config = TEST_CONFIGS["mhd_sym"]
        wout_file = config.get("wout_file")
        if not wout_file or not os.path.exists(wout_file):
            self.skipTest(f"Wout file not found: {wout_file}")

        json_path = None
        try:
            bri = BoozerRadialInterpolant(wout_file, config["order"], comm=comm)

            n = config["nfp"]
            field = InterpolatedBoozerField(
                bri,
                3,
                [0.2, 0.8, 5],
                [0, np.pi, 5],
                [0, 2 * np.pi / n, 8],
                True,
            )

            grid_ranges = {
                "smin": 0.2,
                "smax": 0.8,
                "thetamin": 0,
                "thetamax": np.pi,
                "zetamin": 0,
                "zetamax": 2 * np.pi / n,
            }

            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
                json_path = f.name
            field.to_json(json_path)

            loaded_field = InterpolatedBoozerField.from_json(json_path)

            self._verify_scalar_attributes(field, loaded_field, "wout_mhd_sym: ")
            self._verify_evaluations(
                field,
                loaded_field,
                grid_ranges,
                num_points=30,
                msg_prefix="wout_mhd_sym: ",
            )

        finally:
            if json_path and os.path.exists(json_path):
                os.remove(json_path)

    def test_saveload_with_wout_mhd_asym(self):
        """Test save/load when using mhd_asym wout file as input."""
        config = TEST_CONFIGS["mhd_asym"]
        wout_file = config.get("wout_file")
        if not wout_file or not os.path.exists(wout_file):
            self.skipTest(f"Wout file not found: {wout_file}")

        json_path = None
        try:
            bri = BoozerRadialInterpolant(wout_file, config["order"], comm=comm)

            n = config["nfp"]
            field = InterpolatedBoozerField(
                bri,
                3,
                [0.2, 0.8, 5],
                [0, np.pi, 5],
                [0, 2 * np.pi / n, 8],
                True,
            )

            grid_ranges = {
                "smin": 0.2,
                "smax": 0.8,
                "thetamin": 0,
                "thetamax": np.pi,
                "zetamin": 0,
                "zetamax": 2 * np.pi / n,
            }

            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
                json_path = f.name
            field.to_json(json_path)

            loaded_field = InterpolatedBoozerField.from_json(json_path)

            self._verify_scalar_attributes(field, loaded_field, "wout_mhd_asym: ")
            self._verify_evaluations(
                field,
                loaded_field,
                grid_ranges,
                num_points=30,
                msg_prefix="wout_mhd_asym: ",
            )

        finally:
            if json_path and os.path.exists(json_path):
                os.remove(json_path)

    def test_loaded_field_rejects_uncomputed_quantity(self):
        """Test where an uncomputed quantity on a loaded field raises RuntimeError.

        A loaded field has no BoozerRadialInterpolant, so it cannot compute
        quantities that were not saved.  The C++ require_field() guard should
        throw a descriptive RuntimeError instead of segfaulting.
        """
        config = TEST_CONFIGS["vac_qa"]
        if not os.path.exists(config["file"]):
            self.skipTest(f"Test file not found: {config['file']}")

        json_path = None
        try:
            bri, field, grid_ranges = self._create_field(config)

            # diotads is never in any default initialize list
            self.assertFalse(
                field.status_diotads,
                "status_diotads should be False by default",
            )

            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
                json_path = f.name
            field.to_json(json_path)
            loaded_field = InterpolatedBoozerField.from_json(json_path)

            # Calling an uncomputed quantity must raise, not segfault
            loaded_field.set_points(np.array([[0.5, 1.0, 0.5]]))
            with self.assertRaises(RuntimeError):
                loaded_field.diotads()

        finally:
            if json_path and os.path.exists(json_path):
                os.remove(json_path)

    # These files have incorrect radial grids or ordering, so BoozerRadialInterpolant
    # should raise ValueError when trying to load them.
    def test_invalid_reduced_files(self):
        """
        Test that reduced files fail to create BoozerRadialInterpolant.
        These are error-testing files (#7 and #9 in test_files directory).
        Reduced files have incomplete radial grids and should raise ValueError.
        """
        for config_name in ["mhd_sym", "mhd_asym"]:
            config = TEST_CONFIGS[config_name]
            reduced_file = config.get("reduced_file")

            if not reduced_file:
                continue
            if not os.path.exists(reduced_file):
                continue

            with self.assertRaises(
                ValueError, msg=f"{config_name} reduced file should raise ValueError"
            ):
                BoozerRadialInterpolant(reduced_file, config["order"], comm=comm)

    def test_invalid_reordered_files(self):
        """
        Test that reordered files fail to create BoozerRadialInterpolant.
        These are error-testing files (#8 and #10 in test_files directory).
        Reordered files have incorrectly ordered radial grids; raises ValueError.
        """
        for config_name in ["mhd_sym", "mhd_asym"]:
            config = TEST_CONFIGS[config_name]
            reordered_file = config.get("reordered_file")

            if not reordered_file:
                continue
            if not os.path.exists(reordered_file):
                continue

            with self.assertRaises(
                ValueError, msg=f"{config_name} reordered file should raise ValueError"
            ):
                BoozerRadialInterpolant(reordered_file, config["order"], comm=comm)


if __name__ == "__main__":
    unittest.main()
