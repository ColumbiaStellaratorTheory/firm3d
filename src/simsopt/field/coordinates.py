import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.io import netcdf_file
from scipy.optimize import root
from scipy.spatial import KDTree

__all__ = [
    "boozer_to_cylindrical",
    "cylindrical_to_boozer",
    "boozer_to_vmec",
    "vmec_to_boozer",
    "vmec_to_cylindrical",
    "cylindrical_to_vmec",
    "BoozerCoordinateTransformer",
    "VMECCoordinateTransformer",
]


class BoozerCoordinateTransformer:
    """
    A class for efficient coordinate transformations between Boozer and cylindrical
    coordinates with reusable grid-based initialization.

    This class builds a coordinate grid once and reuses it for multiple transformations.

    Args:
        field: The BoozerMagneticField instance used for field evaluation
        grid_resolution: Tuple of (n_s, n_theta, n_zeta) for grid resolution

    Example:
        transformer = BoozerCoordinateTransformer(field, grid_resolution=(15, 30, 30))
        s, theta, zeta = transformer.cylindrical_to_boozer(R, phi, Z)
        # Grid is reused for subsequent calls
        s2, theta2, zeta2 = transformer.cylindrical_to_boozer(R2, phi2, Z2)
    """

    def __init__(self, field, grid_resolution=(10, 20, 20)):
        self.field = field
        self.grid_resolution = grid_resolution
        self._grid_coords = None
        self._grid_cylindrical = None
        self._grid_built = False

    def _build_coordinate_grid(self, n_s, n_theta, n_zeta):
        """
        Build a grid of Boozer coordinates and their corresponding cylindrical
        coordinates.
        Takes advantage of nfp symmetry to reduce grid size.

        Args:
            n_s: Number of s grid points
            n_theta: Number of theta grid points
            n_zeta: Number of zeta grid points

        Returns:
            boozer_coords: Array of shape (n_points, 3) with (s, theta, zeta)
                coordinates
            cylindrical_coords: Array of shape (n_points, 3) with (R, phi, Z)
                coordinates
        """
        # Get nfp from field if available
        nfp = getattr(self.field, "nfp", 1)

        # Create coordinate grids
        s_grid = np.linspace(0, 1, n_s)
        theta_grid = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)

        # Take advantage of nfp symmetry - only need one field period
        zeta_grid = np.linspace(0, 2 * np.pi / nfp, n_zeta, endpoint=False)

        # Create meshgrid
        s_mesh, theta_mesh, zeta_mesh = np.meshgrid(
            s_grid, theta_grid, zeta_grid, indexing="ij"
        )

        s_mesh = s_mesh.flatten()
        theta_mesh = theta_mesh.flatten()
        zeta_mesh = zeta_mesh.flatten()

        # Remove duplicative points on the magnetic axis
        bool_mask = (s_mesh == 0) * (theta_mesh > 0)
        s_mesh = np.delete(s_mesh, bool_mask)
        theta_mesh = np.delete(theta_mesh, bool_mask)
        zeta_mesh = np.delete(zeta_mesh, bool_mask)

        # Flatten to create coordinate arrays
        n_points = len(s_mesh)
        boozer_coords = np.zeros((n_points, 3))
        boozer_coords[:, 0] = s_mesh
        boozer_coords[:, 1] = theta_mesh
        boozer_coords[:, 2] = zeta_mesh

        # Convert to cylindrical coordinates
        self.field.set_points(boozer_coords)
        R = self.field.R()[:, 0]
        Z = self.field.Z()[:, 0]
        nu = self.field.nu()[:, 0]
        phi = zeta_mesh.flatten() - nu

        cylindrical_coords = np.zeros((n_points, 3))
        cylindrical_coords[:, 0] = R
        cylindrical_coords[:, 1] = phi
        cylindrical_coords[:, 2] = Z

        return boozer_coords, cylindrical_coords

    def _ensure_grid_built(self):
        """Build the coordinate grid if not already built."""
        if not self._grid_built:
            try:
                n_s, n_theta, n_zeta = self.grid_resolution
                self._grid_coords, self._grid_cylindrical = self._build_coordinate_grid(
                    n_s, n_theta, n_zeta
                )
                self._grid_built = True
            except Exception as e:
                raise RuntimeError(f"Failed to build coordinate grid: {e}") from e

    def cylindrical_to_boozer(self, R, phi, Z, n_guesses=4, ftol=1e-6):
        """
        Convert from cylindrical coordinates to Boozer coordinates.
        All initial guesses are generated from the coordinate grid.

        Args:
            R: Radial coordinate(s)
            phi: Azimuthal angle(s)
            Z: Vertical coordinate(s)
            n_guesses: Number of grid-based initial guesses to try per point
            ftol: Tolerance for root finding convergence

        Returns:
            s, theta, zeta: Boozer coordinates
        """
        # Ensure grid is built
        self._ensure_grid_built()

        # Convert inputs to arrays
        R = np.asarray(R)
        phi = np.asarray(phi)
        Z = np.asarray(Z)

        # Handle scalar inputs
        input_scalar = np.isscalar(R) or np.isscalar(phi) or np.isscalar(Z)

        # Ensure all arrays have the same shape
        if R.shape != phi.shape or R.shape != Z.shape:
            raise ValueError("R, phi, and Z must have the same shape")

        npoints = R.size
        if npoints == 0:
            raise ValueError("Input arrays cannot be empty")

        s = np.zeros(npoints)
        theta = np.zeros(npoints)
        zeta = np.zeros(npoints)

        def objective_function(x, R_target, phi_target, Z_target):
            """Objective function for root finding."""
            s_val, theta_val, zeta_val = x
            s_val = np.clip(s_val, 0.0, 1.0)

            points = np.zeros((1, 3))
            points[0, 0] = s_val
            points[0, 1] = theta_val
            points[0, 2] = zeta_val

            self.field.set_points(points)

            R_computed = self.field.R()[0, 0]
            Z_computed = self.field.Z()[0, 0]
            nu_computed = self.field.nu()[0, 0]
            phi_computed = zeta_val - nu_computed

            return [
                R_computed - R_target,
                np.arctan2(
                    np.sin(phi_computed - phi_target), np.cos(phi_computed - phi_target)
                ),
                Z_computed - Z_target,
            ]

        def get_grid_guesses(target_point, n_guesses):
            """Get multiple grid-based initial guesses using k-nearest neighbors."""
            # Build KDTree for efficient nearest neighbor search
            tree = KDTree(self._grid_cylindrical)

            # Map target phi to fundamental domain [0, 2*pi/nfp)
            nfp = getattr(self.field, "nfp", 1)
            target_mapped = target_point.copy()
            phi_period = 2 * np.pi / nfp
            target_mapped[1] = target_point[1] % phi_period

            # Find k nearest neighbors (more than n_guesses to have options)
            n_guesses = min(n_guesses * 2, len(self._grid_coords))
            distances, indices = tree.query(target_mapped, k=n_guesses)

            selected_guesses = []
            for idx in indices:
                selected_guesses.append(self._grid_coords[idx])

            return selected_guesses

        for i in range(npoints):
            success = False

            # Get multiple grid-based guesses
            target_point = np.array([R.flat[i], phi.flat[i], Z.flat[i]])
            initial_guesses = get_grid_guesses(target_point, n_guesses)

            for x0 in initial_guesses:
                sol = root(
                    objective_function,
                    x0,
                    args=(R.flat[i], phi.flat[i], Z.flat[i]),
                    method="lm",
                    options={"ftol": ftol},
                )
                if sol.success and np.all(np.abs(sol.fun) < ftol):
                    s[i] = np.clip(sol.x[0], 0.0, 1.0)
                    theta[i] = sol.x[1]
                    zeta[i] = sol.x[2]
                    success = True
                    break

            if not success:
                raise RuntimeError(
                    f"Root finding failed for point {i} with coordinates "
                    f"R={R.flat[i]}, phi={phi.flat[i]}, Z={Z.flat[i]}"
                )

        # Return scalars for scalar inputs
        if input_scalar:
            return s[0], theta[0], zeta[0]
        else:
            return s, theta, zeta

    def boozer_to_cylindrical(self, s, theta, zeta):
        """
        Convert from Boozer coordinates to cylindrical coordinates.

        Args:
            s: Normalized toroidal flux
            theta: Boozer poloidal angle
            zeta: Boozer toroidal angle

        Returns:
            R, phi, Z: Cylindrical coordinates
        """
        return boozer_to_cylindrical(self.field, s, theta, zeta)


class VMECCoordinateTransformer:
    """
    A class for efficient coordinate transformations between VMEC and cylindrical
    coordinates with reusable grid-based initialization.

    This class builds a coordinate grid once and reuses it for multiple transformations,
    providing better performance and robustness than the standalone functions.

    Args:
        wout_filename: Path to VMEC wout file
        grid_resolution: Tuple of (n_s, n_theta, n_phi) for grid resolution

    Example:
        transformer = VMECCoordinateTransformer("wout.nc", grid_resolution=(15, 30, 30))
        s, theta, phi = transformer.cylindrical_to_vmec(R, phi_cyl, Z)
        # Grid is reused for subsequent calls
        s2, theta2, phi2 = transformer.cylindrical_to_vmec(R2, phi_cyl2, Z2)
    """

    def __init__(self, wout_filename, grid_resolution=(10, 20, 20)):
        self.wout_filename = wout_filename
        self.grid_resolution = grid_resolution
        self._grid_coords = None
        self._grid_cylindrical = None
        self._grid_built = False
        self._nfp = None

    def _build_coordinate_grid(self, n_s, n_theta, n_phi):
        """
        Build a grid of VMEC coordinates and their corresponding cylindrical
        coordinates.
        Takes advantage of nfp symmetry to reduce grid size.

        Args:
            n_s: Number of s grid points
            n_theta: Number of theta grid points
            n_phi: Number of phi grid points

        Returns:
            vmec_coords: Array of shape (n_points, 3) with (s, theta, phi) coordinates
            cylindrical_coords: Array of shape (n_points, 3) with (R, phi_cyl, Z)
                coordinates
        """
        # Get nfp from VMEC file
        with netcdf_file(self.wout_filename, "r") as f:
            self._nfp = int(f.variables["nfp"][()])

        # Create coordinate grids
        s_grid = np.linspace(0, 1, n_s)
        theta_grid = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)

        # Take advantage of nfp symmetry - only need one field period
        phi_grid = np.linspace(0, 2 * np.pi / self._nfp, n_phi, endpoint=False)

        # Create meshgrid
        s_mesh, theta_mesh, phi_mesh = np.meshgrid(
            s_grid, theta_grid, phi_grid, indexing="ij"
        )
        # Flatten to create coordinate arrays
        s_mesh = s_mesh.flatten()
        theta_mesh = theta_mesh.flatten()
        phi_mesh = phi_mesh.flatten()

        # Remove duplicative points on the magnetic axis
        bool_mask = (s_mesh == 0) * (theta_mesh > 0)
        s_mesh = np.delete(s_mesh, bool_mask)
        theta_mesh = np.delete(theta_mesh, bool_mask)
        phi_mesh = np.delete(phi_mesh, bool_mask)

        n_points = len(s_mesh)
        vmec_coords = np.zeros((n_points, 3))
        vmec_coords[:, 0] = s_mesh
        vmec_coords[:, 1] = theta_mesh
        vmec_coords[:, 2] = phi_mesh

        # Convert to cylindrical coordinates using vmec_to_cylindrical
        R, phi_cyl, Z = vmec_to_cylindrical(
            self.wout_filename, vmec_coords[:, 0], vmec_coords[:, 1], vmec_coords[:, 2]
        )

        cylindrical_coords = np.zeros((n_points, 3))
        cylindrical_coords[:, 0] = R
        cylindrical_coords[:, 1] = phi_cyl
        cylindrical_coords[:, 2] = Z

        return vmec_coords, cylindrical_coords

    def _ensure_grid_built(self):
        """Build the coordinate grid if not already built."""
        if not self._grid_built:
            try:
                n_s, n_theta, n_phi = self.grid_resolution
                self._grid_coords, self._grid_cylindrical = self._build_coordinate_grid(
                    n_s, n_theta, n_phi
                )
                self._grid_built = True
            except Exception as e:
                raise RuntimeError(f"Failed to build coordinate grid: {e}") from e

    def cylindrical_to_vmec(self, R, phi, Z, n_guesses=4, ftol=1e-6):
        """
        Convert from cylindrical coordinates to VMEC coordinates using robust
        pseudo-Cartesian coordinates x = sqrt(s)*cos(theta), y = sqrt(s)*sin(theta).
        All initial guesses are generated from the coordinate grid.

        Args:
            R: Radial coordinate(s)
            phi: Azimuthal angle(s)
            Z: Vertical coordinate(s)
            n_guesses: Number of grid-based initial guesses to try per point
            ftol: Tolerance for root finding convergence

        Returns:
            s_vmec, theta_vmec, phi_vmec: VMEC coordinates
        """
        # Ensure grid is built
        self._ensure_grid_built()

        # Convert inputs to arrays
        R = np.asarray(R)
        phi = np.asarray(phi)
        Z = np.asarray(Z)

        # Handle scalar inputs
        input_scalar = np.isscalar(R) or np.isscalar(phi) or np.isscalar(Z)

        # Ensure all arrays have the same shape
        if R.shape != phi.shape or R.shape != Z.shape:
            raise ValueError("R, phi, and Z must have the same shape")

        npoints = R.size
        if npoints == 0:
            raise ValueError("Input arrays cannot be empty")

        # Load VMEC data for objective function
        with netcdf_file(self.wout_filename, "r") as f:
            rmnc = f.variables["rmnc"][:]
            zmns = f.variables["zmns"][:]
            xm = f.variables["xm"][:]
            xn = f.variables["xn"][:]
            ns = int(f.variables["ns"][()])
            s_full = np.linspace(0, 1, ns)

        s_vmec = np.zeros(npoints)
        theta_vmec = np.zeros(npoints)
        phi_vmec = np.zeros(npoints)

        def objective_function(x_norm, R_target, phi_target, Z_target):
            """
            Objective function using normalized coordinates
            x = sqrt(s)*cos(theta), y = sqrt(s)*sin(theta).
            This avoids singularity issues at s=0.
            """
            x_coord, y_coord = x_norm

            # Convert normalized coordinates back to s, theta
            s_i = x_coord**2 + y_coord**2
            s_i = np.clip(s_i, 0, 1)

            theta_i = np.arctan2(y_coord, x_coord)

            # Interpolate harmonics
            rmnc_s = np.zeros_like(rmnc[0, :])
            zmns_s = np.zeros_like(zmns[0, :])

            for j in range(rmnc.shape[1]):
                rmnc_s[j] = np.interp(s_i, s_full, rmnc[:, j])
                zmns_s[j] = np.interp(s_i, s_full, zmns[:, j])

            # Compute R and Z
            R_computed = 0.0
            Z_computed = 0.0
            for j in range(len(xm)):
                angle = xm[j] * theta_i - xn[j] * phi_target
                R_computed += rmnc_s[j] * np.cos(angle)
                Z_computed += zmns_s[j] * np.sin(angle)

            return [R_computed - R_target, Z_computed - Z_target]

        def convert_to_normalized(s, theta):
            """Convert (s, theta, phi) to normalized (x, y, phi) coordinates."""
            sqrt_s = np.sqrt(max(s, 0))
            x = sqrt_s * np.cos(theta)
            y = sqrt_s * np.sin(theta)
            return [x, y]

        def convert_from_normalized(x, y):
            """Convert normalized (x, y, phi) to (s, theta, phi) coordinates."""
            s = x**2 + y**2
            s = np.clip(s, 0, 1)
            theta = np.arctan2(y, x)
            return s, theta

        def get_grid_guesses(target_point, n_guesses):
            """Get multiple grid-based initial guesses using k-nearest neighbors."""
            # Build KDTree for efficient nearest neighbor search
            tree = KDTree(self._grid_cylindrical)

            # Map target phi to fundamental domain [0, 2*pi/nfp)
            target_mapped = target_point.copy()
            phi_period = 2 * np.pi / self._nfp
            target_mapped[1] = target_point[1] % phi_period

            # Find k nearest neighbors (more than n_guesses to have options)
            distances, indices = tree.query(target_mapped, k=n_guesses)

            # Convert to list if single neighbor
            if n_guesses == 1:
                indices = [indices]

            selected_guesses = []
            for idx in indices:
                grid_coords = self._grid_coords[idx]
                guess_norm = convert_to_normalized(grid_coords[0], grid_coords[1])
                selected_guesses.append(guess_norm)

            return selected_guesses

        for i in range(npoints):
            success = False

            # Get multiple grid-based guesses
            target_point = np.array([R.flat[i], phi.flat[i], Z.flat[i]])
            initial_guesses_normalized = get_grid_guesses(target_point, n_guesses)

            for x0_norm in initial_guesses_normalized:
                try:
                    sol = root(
                        objective_function,
                        x0_norm,
                        args=(R.flat[i], phi.flat[i], Z.flat[i]),
                        method="lm",
                        options={"ftol": ftol},
                    )

                    if sol.success and np.all(np.abs(sol.fun) < ftol):
                        # Convert solution back to (s, theta, phi)
                        s_result, theta_result = convert_from_normalized(
                            sol.x[0], sol.x[1]
                        )
                        s_vmec[i] = s_result
                        theta_vmec[i] = theta_result
                        phi_vmec[i] = phi.flat[i]
                        success = True
                        break
                except Exception:
                    continue

            if not success:
                raise RuntimeError(
                    f"Root finding failed for point {i} with coordinates "
                    f"R={R.flat[i]}, phi={phi.flat[i]}, Z={Z.flat[i]}"
                )

        # Return scalars for scalar inputs
        if input_scalar:
            return s_vmec[0], theta_vmec[0], phi_vmec[0]
        else:
            return s_vmec, theta_vmec, phi_vmec

    def vmec_to_cylindrical(self, s_vmec, theta_vmec, phi_vmec):
        """
        Convert from VMEC coordinates to cylindrical coordinates.

        Args:
            s_vmec: Normalized toroidal flux
            theta_vmec: VMEC poloidal angle
            phi_vmec: VMEC cylindrical angle

        Returns:
            R, phi_cyl, Z: Cylindrical coordinates
        """
        return vmec_to_cylindrical(self.wout_filename, s_vmec, theta_vmec, phi_vmec)


def boozer_to_cylindrical(field, s, theta, zeta):
    r"""
    Convert from Boozer coordinates to cylindrical coordinates.

    Args:
        field : The :class:`BoozerMagneticField` instance used for field evaluation.
        s : A scalar or a numpy array of shape (npoints,) containing the
            normalized toroidal flux.
        theta : A scalar or a numpy array of shape (npoints,) containing the
            Boozer poloidal angle.
        zeta : A scalar or a numpy array of shape (npoints,) containing the
            Boozer toroidal angle.

    Returns:
        R : A scalar or a numpy array of shape (npoints,) containing the
            radial coordinate.
        phi : A scalar or a numpy array of shape (npoints,) containing the
            azimuthal angle.
        Z : A scalar or a numpy array of shape (npoints,) containing the
            vertical coordinate.
    """
    if not isinstance(s, np.ndarray):
        s = np.asarray(s)
    if not isinstance(theta, np.ndarray):
        theta = np.asarray(theta)
    if not isinstance(zeta, np.ndarray):
        zeta = np.asarray(zeta)

    # Handle scalar inputs - return scalars if any input is a scalar
    input_scalar = np.isscalar(s) or np.isscalar(theta) or np.isscalar(zeta)

    # Ensure all arrays have the same shape
    if s.shape != theta.shape or s.shape != zeta.shape:
        raise ValueError("s, theta, and zeta must have the same shape")

    npoints = s.size

    # Validate that arrays are not empty
    if npoints == 0:
        raise ValueError("Input arrays cannot be empty")

    points = np.zeros((npoints, 3))
    points[:, 0] = s.flatten()
    points[:, 1] = theta.flatten()
    points[:, 2] = zeta.flatten()

    field.set_points(points)

    R = field.R()[:, 0]
    Z = field.Z()[:, 0]
    nu = field.nu()[:, 0]
    phi = zeta - nu

    # Return scalars for scalar inputs, arrays for array inputs
    if input_scalar:
        return R[0], phi[0], Z[0]
    else:
        return R, phi, Z


def cylindrical_to_boozer(
    field,
    R,
    phi,
    Z,
    n_guesses=4,
    ftol=1e-6,
    grid_resolution=(10, 20, 20),
):
    r"""
    Convert from cylindrical coordinates to Boozer coordinates using root finding.

    Args:
        field : The :class:`BoozerMagneticField` instance used for field evaluation.
        R : A scalar or a numpy array of shape (npoints,) containing the
            radial coordinate.
        phi : A scalar or a numpy array of shape (npoints,) containing the
            azimuthal angle.
        Z : A scalar or a numpy array of shape (npoints,) containing the
            vertical coordinate.
        n_guesses : int, optional
            Number of grid-based initial guesses to try for each point (default: 4).
            Must be a positive integer.
        ftol : float, optional
            Tolerance for root finding convergence (default: 1e-6).
        grid_resolution : tuple of int, optional
            Grid resolution as (n_s, n_theta, n_zeta) (default: (10, 20, 20)).

    Returns:
        s : A scalar or a numpy array of shape (npoints,) containing the
            normalized toroidal flux.
        theta : A scalar or a numpy array of shape (npoints,) containing the
            Boozer poloidal angle.
        zeta : A scalar or a numpy array of shape (npoints,) containing the
            Boozer toroidal angle.
    """
    if not isinstance(R, np.ndarray):
        R = np.asarray(R)
    if not isinstance(phi, np.ndarray):
        phi = np.asarray(phi)
    if not isinstance(Z, np.ndarray):
        Z = np.asarray(Z)

    # Ensure all arrays have the same shape
    if R.shape != phi.shape or R.shape != Z.shape:
        raise ValueError("R, phi, and Z must have the same shape")

    npoints = R.size

    # Validate that arrays are not empty
    if npoints == 0:
        raise ValueError("Input arrays cannot be empty")

    transformer = BoozerCoordinateTransformer(field, grid_resolution)
    return transformer.cylindrical_to_boozer(R, phi, Z, n_guesses=n_guesses, ftol=ftol)


def vmec_to_boozer(wout_filename, field, s_vmec, theta_vmec, phi_vmec, ftol=1e-6):
    r"""
    Convert from VMEC coordinates to Boozer coordinates.

    Args:
        wout_filename : str
            The name of the VMEC wout file.
        field : The :class:`BoozerMagneticField` instance used for field evaluation.
        s_vmec : A scalar or a numpy array of shape (npoints,) containing the
            normalized toroidal flux.
        theta_vmec : A scalar or a numpy array of shape (npoints,) containing
            the VMEC poloidal angle.
        phi_vmec : A scalar or a numpy array of shape (npoints,) containing
            the VMEC cylindrical angle.
        ftol : float, optional
            Tolerance for root finding convergence (default: 1e-6).

    Returns:
        theta_b : A numpy array of shape (npoints,) containing the Boozer
            poloidal angle.
        zeta_b : A numpy array of shape (npoints,) containing the Boozer toroidal angle.
    """
    # Validate that arrays are not empty
    if len(s_vmec) == 0:
        raise ValueError("Input arrays cannot be empty")

    # Load VMEC and booz_xform data
    f = netcdf_file(wout_filename, mmap=False)
    lmns = f.variables["lmns"][()]
    mnmax = f.variables["mnmax"][()]
    ns = f.variables["ns"][()]
    xm = f.variables["xm"][()]
    xn = f.variables["xn"][()]
    f.close()

    s_full_grid = np.linspace(0, 1, ns)
    s_half_grid = (s_full_grid[0:-1] + s_full_grid[1::]) / 2.0

    # Create splines for lmns
    lmns_splines = []
    for jmn in range(mnmax):
        lmns_splines.append(InterpolatedUnivariateSpline(s_half_grid, lmns[1::, jmn]))

    def vartheta_vmec(s, theta_vmec, phi_vmec):
        """Compute vartheta from VMEC data."""
        lmns = np.zeros((1, mnmax))
        for jmn in range(mnmax):
            lmns[:, jmn] = lmns_splines[jmn](s)

        angle = xm * theta_vmec - xn * phi_vmec
        sinangle = np.sin(angle)

        lambd = np.sum(lmns * sinangle)
        vartheta = theta_vmec + lambd
        return vartheta

    def vartheta_phi_vmec(s, theta_b, zeta_b):
        """Compute PEST angles from Boozer coordinates."""
        points = np.zeros((1, 3))
        points[:, 0] = s
        points[:, 1] = theta_b
        points[:, 2] = zeta_b
        field.set_points(points)
        nu = field.nu()[0, 0]
        iota = field.iota()[0, 0]
        vartheta = theta_b - iota * nu
        phi = zeta_b - nu
        return vartheta, phi

    def func_root(x, s, vartheta_target, phi_target):
        """Root finding function."""
        theta_b = x[0]
        zeta_b = x[1]
        vartheta, phi = vartheta_phi_vmec(s, theta_b, zeta_b)
        vartheta_diff = np.arctan2(
            np.sin(vartheta - vartheta_target), np.cos(vartheta - vartheta_target)
        )
        phi_diff = np.arctan2(np.sin(phi - phi_target), np.cos(phi - phi_target))
        return [vartheta_diff, phi_diff]

    theta_b = []
    zeta_b = []
    for i in range(len(s_vmec)):
        vartheta = vartheta_vmec(s_vmec[i], theta_vmec[i], phi_vmec[i])
        sol = root(
            func_root,
            [vartheta, phi_vmec[i]],
            args=(s_vmec[i], vartheta, phi_vmec[i]),
            method="lm",
            options={"ftol": ftol},
        )
        if sol.success and np.all(np.abs(sol.fun) < ftol):
            theta_b.append(sol.x[0])
            zeta_b.append(sol.x[1])
        else:
            raise RuntimeError(
                f"Root finding failed for point {i} with coordinates "
                f"s={s_vmec[i]}, theta_vmec={theta_vmec[i]}, phi_vmec={phi_vmec[i]}. "
            )

    return np.array(theta_b), np.array(zeta_b)


def boozer_to_vmec(wout_filename, field, s, theta_b, zeta_b, ftol=1e-6):
    r"""
    Convert from Boozer coordinates to VMEC coordinates.

    Args:
        wout_filename : str
            The name of the VMEC wout file.
        field : The :class:`BoozerMagneticField` instance used for field evaluation.
        s : A scalar or a numpy array of shape (npoints,) containing the
            normalized toroidal flux.
        theta_b : A scalar or a numpy array of shape (npoints,) containing
            the Boozer poloidal angle.
        zeta_b : A scalar or a numpy array of shape (npoints,) containing
            the Boozer toroidal angle.
        ftol : float, optional
            Tolerance for root finding convergence (default: 1e-6).

    Returns:
        theta_vmec : A scalar or a numpy array of shape (npoints,) containing
            the VMEC poloidal angle.
        phi_vmec : A scalar or a numpy array of shape (npoints,) containing
            the VMEC cylindrical angle.
    """
    # Handle scalar inputs - return scalars if any input is a scalar
    input_scalar = np.isscalar(s) or np.isscalar(theta_b) or np.isscalar(zeta_b)

    # Convert to arrays if needed
    s = np.asarray(s)
    theta_b = np.asarray(theta_b)
    zeta_b = np.asarray(zeta_b)

    # Ensure all inputs have the same shape
    if s.shape != theta_b.shape or s.shape != zeta_b.shape:
        raise ValueError("s, theta_b, and zeta_b must have the same shape")

    # Validate that arrays are not empty
    if s.size == 0:
        raise ValueError("Input arrays cannot be empty")

    # Load VMEC and booz_xform data
    f = netcdf_file(wout_filename, mmap=False)
    lmns = f.variables["lmns"][()]
    mnmax = f.variables["mnmax"][()]
    ns = f.variables["ns"][()]
    xm = f.variables["xm"][()]
    xn = f.variables["xn"][()]
    f.close()

    s_full_grid = np.linspace(0, 1, ns)
    s_half_grid = (s_full_grid[0:-1] + s_full_grid[1::]) / 2.0

    # Create splines for lmns
    lmns_splines = []
    for jmn in range(mnmax):
        lmns_splines.append(InterpolatedUnivariateSpline(s_half_grid, lmns[1::, jmn]))

    def vartheta_vmec(s, theta_vmec, phi_vmec):
        """Compute vartheta from VMEC data."""
        lmns = np.zeros((1, mnmax))
        for jmn in range(mnmax):
            lmns[:, jmn] = lmns_splines[jmn](s)

        angle = xm * theta_vmec - xn * phi_vmec
        sinangle = np.sin(angle)

        lambd = np.sum(lmns * sinangle)
        vartheta = theta_vmec + lambd
        return vartheta

    def vartheta_phi_vmec(s, theta_b, zeta_b):
        """Compute PEST angles from Boozer coordinates."""
        points = np.zeros((1, 3))
        points[:, 0] = s
        points[:, 1] = theta_b
        points[:, 2] = zeta_b
        field.set_points(points)
        nu = field.nu()[0, 0]
        iota = field.iota()[0, 0]
        vartheta = theta_b - iota * nu
        phi = zeta_b - nu
        return vartheta, phi

    def func_root(x, s, vartheta_boozer, zeta_boozer):
        """Root finding function."""
        theta_vmec = x[0]
        # Compute PEST angles from desired Boozer coordinates
        vartheta_target, phi_target = vartheta_phi_vmec(s, vartheta_boozer, zeta_boozer)
        # Compute PEST angles from VMEC coordinates
        vartheta = vartheta_vmec(s, theta_vmec, phi_target)
        vartheta_diff = np.arctan2(
            np.sin(vartheta - vartheta_target), np.cos(vartheta - vartheta_target)
        )
        return [vartheta_diff]

    theta_vmec = np.zeros_like(s)
    phi_vmec = np.zeros_like(s)
    for i in range(s.size):
        sol = root(
            func_root,
            [theta_b[i]],
            args=(s[i], theta_b[i], zeta_b[i]),
            method="lm",
            options={"ftol": ftol},
        )
        if sol.success and np.all(np.abs(sol.fun) < ftol):
            theta_vmec[i] = sol.x[0]
            vartheta, phi_vmec[i] = vartheta_phi_vmec(s[i], theta_b[i], zeta_b[i])
        else:
            raise RuntimeError(
                f"Root finding failed for point {i} with coordinates "
                f"s={s[i]}, theta_b={theta_b[i]}, zeta_b={zeta_b[i]}. "
            )

    # Return scalars for scalar inputs, arrays for array inputs
    if input_scalar:
        return theta_vmec[0], phi_vmec[0]
    else:
        return theta_vmec, phi_vmec


def vmec_to_cylindrical(wout_filename, s_vmec, theta_vmec, phi_vmec):
    r"""
    Convert from VMEC coordinates to cylindrical coordinates.

    Args:
        wout_filename : str
            The name of the VMEC wout file.
        s_vmec : A scalar or a numpy array of shape (npoints,) containing the
            normalized toroidal flux.
        theta_vmec : A scalar or a numpy array of shape (npoints,) containing
            the VMEC poloidal angle.
        phi_vmec : A scalar or a numpy array of shape (npoints,) containing
            the VMEC cylindrical angle.

    Returns:
        R : A scalar or a numpy array of shape (npoints,) containing the
            radial coordinate.
        phi_cyl : A scalar or a numpy array of shape (npoints,) containing
            the azimuthal angle.
        Z : A scalar or a numpy array of shape (npoints,) containing the
            vertical coordinate.
    """
    # Handle scalar inputs - return scalars if any input is a scalar
    input_scalar = (
        np.isscalar(s_vmec) or np.isscalar(theta_vmec) or np.isscalar(phi_vmec)
    )

    # Convert to arrays for processing
    s_vmec = np.asarray(s_vmec)
    theta_vmec = np.asarray(theta_vmec)
    phi_vmec = np.asarray(phi_vmec)

    # Validate that arrays are not empty
    if s_vmec.size == 0:
        raise ValueError("Input arrays cannot be empty")

    # Load VMEC data
    with netcdf_file(wout_filename, "r") as f:
        rmnc = f.variables["rmnc"][:]  # R harmonics (cos)
        zmns = f.variables["zmns"][:]  # Z harmonics (sin)
        xm = f.variables["xm"][:]  # poloidal mode numbers
        xn = f.variables["xn"][:]  # toroidal mode numbers
        ns = int(f.variables["ns"][()])  # number of radial surfaces (scalar)
        s_full = np.linspace(0, 1, ns)  # full radial grid

    # Initialize R and Z arrays
    R = np.zeros_like(s_vmec)
    Z = np.zeros_like(s_vmec)

    # For each point, compute R and Z using VMEC Fourier harmonics
    for i in range(s_vmec.size):
        s_i = s_vmec[i]
        theta_i = theta_vmec[i]
        phi_i = phi_vmec[i]

        # Interpolate harmonics to the desired s value
        rmnc_s = np.zeros_like(rmnc[0, :])
        zmns_s = np.zeros_like(zmns[0, :])

        for j in range(rmnc.shape[1]):  # Loop over mode numbers
            # Interpolate rmnc and zmns to s_i
            rmnc_s[j] = np.interp(s_i, s_full, rmnc[:, j])
            zmns_s[j] = np.interp(s_i, s_full, zmns[:, j])

        # Compute R and Z using Fourier series
        R[i] = 0.0
        Z[i] = 0.0

        for j in range(len(xm)):
            angle = xm[j] * theta_i - xn[j] * phi_i
            R[i] += rmnc_s[j] * np.cos(angle)
            Z[i] += zmns_s[j] * np.sin(angle)

    # phi_cyl is the same as phi_vmec
    phi_cyl = phi_vmec

    # Return scalars for scalar inputs, arrays for array inputs
    if input_scalar:
        return (
            R[0] if hasattr(R, "__len__") else R,
            phi_cyl[0] if hasattr(phi_cyl, "__len__") else phi_cyl,
            Z[0] if hasattr(Z, "__len__") else Z,
        )
    else:
        return R, phi_cyl, Z


def cylindrical_to_vmec(
    wout_filename,
    R,
    phi,
    Z,
    n_guesses=4,
    ftol=1e-6,
    grid_resolution=(10, 20, 20),
):
    r"""
    Convert from cylindrical coordinates to VMEC coordinates using robust
    pseudo-Cartesian coordinates x = sqrt(s)*cos(theta), y = sqrt(s)*sin(theta).

    Args:
        wout_filename : str
            The name of the VMEC wout file.
        R : A scalar or a numpy array of shape (npoints,) containing the
            radial coordinate.
        phi : A scalar or a numpy array of shape (npoints,) containing the
            azimuthal angle.
        Z : A scalar or a numpy array of shape (npoints,) containing the
            vertical coordinate.
        n_guesses : int, optional
            Number of grid-based initial guesses to try for each point (default: 4).
            Must be a positive integer.
        ftol : float, optional
            Tolerance for root finding convergence (default: 1e-6).
        grid_resolution : tuple of int, optional
            Grid resolution as (n_s, n_theta, n_phi) (default: (10, 20, 20)).

    Returns:
        s_vmec : A scalar or a numpy array of shape (npoints,) containing the
            normalized toroidal flux.
        theta_vmec : A scalar or a numpy array of shape (npoints,) containing
            the VMEC poloidal angle.
        phi_vmec : A scalar or a numpy array of shape (npoints,) containing
            the VMEC cylindrical angle.
    """
    # Convert to arrays for processing
    R = np.asarray(R)
    phi = np.asarray(phi)
    Z = np.asarray(Z)

    # Validate that arrays are not empty
    if R.size == 0:
        raise ValueError("Input arrays cannot be empty")

    # Ensure all arrays have the same shape
    if R.shape != phi.shape or R.shape != Z.shape:
        raise ValueError("R, phi, and Z must have the same shape")

    transformer = VMECCoordinateTransformer(wout_filename, grid_resolution)
    return transformer.cylindrical_to_vmec(R, phi, Z, n_guesses=n_guesses, ftol=ftol)
