import os
from copy import deepcopy
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
import plotly.graph_objects as go

__all__ = ["Continuum", "Harmonic", "ModeContinuum", "AlfvenSpecData"]


class Continuum:
    r"""Configure a cosine-basis shear Alfvén continuum calculation.

    Inputs are validated and the coordinate splines are copied for local
    sampling. FFT matrix assembly and a direct-quadrature reference are
    available, with per-surface angular convergence checks. Eigensolves will follow.

    Args:
        field (BoozerRadialInterpolant): Stellarator-symmetric equilibrium with
            coordinate splines of degree at least two and continuous first
            derivatives at interior knots, so the tangents are continuous.
        surfaces (array-like): Nonempty one-dimensional normalized flux values
            ``s = psi / psi0`` within the field's radial interpolation interval
            and ``0 < s <= 1``.
        modes (array-like): Nonempty integer array of shape ``(N, 2)`` with
            rows ``(m, n)`` for ``cos(m*theta - n*zeta)``. Toroidal indices are
            actual mode numbers, not divided by ``nfp``. All modes must belong
            to one family: ``n_i = +/- n_j`` modulo ``nfp``. Duplicate cosine
            functions, including ``(m, n)`` and ``(-m, -n)``, are rejected.
            The constant basis function is 1; other cosines have a factor of
            ``sqrt(2)`` for unit norm under the full-torus angular average.
        density (float or callable, optional): Mass density in kg/m^3. A callable
            ``density(s)`` must return
            a finite positive scalar for each requested surface; it is stored
            without evaluation here and will be checked before solving.
            ``None`` requests density-independent results only.

    Raises:
        ValueError: If the field, surfaces, modes, or scalar density are invalid.

    Surface and mode arrays are copied without reordering. ``mode_family`` is
    the smaller of the two equivalent toroidal residues, ``n`` and ``-n``.
    Controls for angular grids, convergence limits, the reference magnetic
    field, and eigenvector storage will accompany the numerical methods.
    """

    def __init__(self, field, surfaces, modes, density=None):
        # The Boozer field module also imports the readers in this module.
        from ..field.boozermagneticfield import BoozerRadialInterpolant

        if not isinstance(field, BoozerRadialInterpolant):
            raise ValueError(
                "field must be a BoozerRadialInterpolant; "
                f"got {type(field).__name__}."
            )
        if not field.stellsym:
            raise ValueError("field must be stellarator-symmetric for a cosine basis.")
        if (
            isinstance(field.nfp, (bool, np.bool_))
            or not isinstance(field.nfp, Integral)
            or field.nfp <= 0
        ):
            raise ValueError(f"field.nfp must be a positive integer; got {field.nfp}.")
        if (
            not isinstance(field.psi0, Real)
            or not np.isfinite(field.psi0)
            or field.psi0 == 0
        ):
            raise ValueError(
                f"field.psi0 must be finite and nonzero; got {field.psi0}."
            )

        self.field = field
        self.surfaces = self._validate_surfaces(surfaces)
        self.modes, self.mode_family = self._validate_modes(modes)

        if density is not None and not callable(density):
            if (
                isinstance(density, (bool, np.bool_))
                or not isinstance(density, Real)
                or not np.isfinite(density)
                or density <= 0
            ):
                raise ValueError(
                    "density must be None, a callable, or a finite positive scalar "
                    f"in kg/m^3; got {density!r}."
                )
            density = float(density)
        self.density = density
        self._copy_geometry(field)

    def _copy_geometry(self, field):
        """Copy the field data needed for local, consistent coordinate sampling."""
        self._nfp = int(field.nfp)
        self._geometry_m = np.array(field.xm_b, copy=True)
        self._geometry_n = np.array(field.xn_b, copy=True)
        mode_pairs = np.column_stack((self._geometry_m, self._geometry_n))
        if len(np.unique(mode_pairs, axis=0)) != len(mode_pairs):
            raise ValueError("The equilibrium Fourier table contains duplicate modes.")

        self._coordinate_splines = {
            "R": deepcopy(field.rmnc_splines),
            "Z": deepcopy(field.zmns_splines),
            "nu": deepcopy(field.numns_splines),
        }
        self._radial_splines = {}
        self._surface_min = max(0.0, field.s_half_ext[0])
        self._surface_max = min(1.0, field.s_half_ext[-1])
        for name, spline in self._coordinate_splines.items():
            lower = spline.t[spline.k]
            upper = spline.t[-spline.k - 1]
            knots, multiplicities = np.unique(spline.t, return_counts=True)
            interior = (knots > lower) & (knots < upper)
            # At a knot of multiplicity p, a degree-k spline is C^(k-p).
            if spline.k < 2 or np.any(multiplicities[interior] >= spline.k):
                raise ValueError(
                    f"{name} coordinate spline must have degree >= 2 and "
                    "continuous first derivatives at interior knots."
                )
            if spline.axis != 0 or spline.c.shape[1:] != (len(mode_pairs),):
                raise ValueError(
                    f"{name} spline must have one column per Fourier mode."
                )
            spline.extrapolate = False
            # Derive tangents from the same splines as the coordinates.
            self._radial_splines[name] = spline.derivative()
            self._surface_min = max(self._surface_min, lower)
            self._surface_max = min(self._surface_max, upper)

        self._flux_splines = {
            "iota": deepcopy(field.iota_spline),
            "G": deepcopy(field.G_spline),
            "I": deepcopy(field.I_spline),
        }
        for spline in self._flux_splines.values():
            spline.extrapolate = False
            self._surface_min = max(self._surface_min, spline.t[spline.k])
            self._surface_max = min(self._surface_max, spline.t[-spline.k - 1])
        if np.any(self.surfaces < self._surface_min) or np.any(
            self.surfaces > self._surface_max
        ):
            raise ValueError(
                "surfaces must lie in the common coordinate/flux spline interval "
                f"[{self._surface_min}, {self._surface_max}]."
            )
        self._psi0 = float(field.psi0)

        self._poloidal_modes, self._m_indices = np.unique(
            self._geometry_m, return_inverse=True
        )
        self._toroidal_modes, self._n_indices = np.unique(
            self._geometry_n, return_inverse=True
        )
        self._theta_grid = None
        self._zeta_grid = None

    def _set_angular_grid(self, theta, zeta):
        """Cache separable angular factors for just the current tensor grid."""
        grids = []
        for name, grid in (("theta", theta), ("zeta", zeta)):
            grid = np.asarray(grid)
            if (
                grid.ndim != 1
                or grid.size == 0
                or not np.issubdtype(grid.dtype, np.number)
                or np.iscomplexobj(grid)
                or not np.all(np.isfinite(grid))
            ):
                raise ValueError(f"{name} must be a finite nonempty real 1D array.")
            grids.append(grid)
        theta, zeta = grids
        if np.array_equal(theta, self._theta_grid) and np.array_equal(
            zeta, self._zeta_grid
        ):
            return

        self._theta_grid = np.array(theta, dtype=float, copy=True)
        self._zeta_grid = np.array(zeta, dtype=float, copy=True)
        theta_phase = np.outer(self._theta_grid, self._poloidal_modes)
        zeta_phase = np.outer(self._toroidal_modes, self._zeta_grid)
        self._cos_theta = np.cos(theta_phase)
        self._sin_theta = np.sin(theta_phase)
        self._cos_zeta = np.cos(zeta_phase)
        self._sin_zeta = np.sin(zeta_phase)

    def _sum_harmonics(self, coefficients, parity):
        """Sum cos(m*theta - n*zeta) or sin(m*theta - n*zeta) on the grid.

        Separate theta and zeta factors using the angle-difference identities.
        The coefficient rectangle has one row per unique m and column per n;
        this avoids allocating a grid-points-by-harmonics array.
        """
        rectangle = np.zeros((len(self._poloidal_modes), len(self._toroidal_modes)))
        rectangle[self._m_indices, self._n_indices] = coefficients
        cosine_zeta_sum = rectangle @ self._cos_zeta
        sine_zeta_sum = rectangle @ self._sin_zeta
        if parity == "cos":
            return (
                self._cos_theta @ cosine_zeta_sum + self._sin_theta @ sine_zeta_sum
            )
        return self._sin_theta @ cosine_zeta_sum - self._cos_theta @ sine_zeta_sum

    def _sample_coordinates(self, surface, theta, zeta):
        """Sample one surface locally without using the field's evaluation API.

        Args:
            surface (float): Normalized flux inside the copied spline interval,
                excluding the axis.
            theta (array-like): One-dimensional poloidal angles in radians.
            zeta (array-like): One-dimensional Boozer toroidal angles in radians.

        Returns:
            dict: ``R``, ``Z``, ``nu``, and ``phi = zeta - nu``, each with
            shape ``(len(theta), len(zeta))``. Derivative keys append ``_s``,
            ``_theta``, or ``_zeta``. R and Z are in meters; nu and phi are in
            radians. The scalar entries ``iota``, ``G``, ``I``, and ``psi0``
            retain the field's flux and magnetic-component conventions.

        Raises:
            ValueError: If the surface or angular arrays are invalid.
        """
        if (
            not isinstance(surface, Real)
            or not np.isfinite(surface)
            or surface <= 0
            or surface < self._surface_min
            or surface > self._surface_max
        ):
            raise ValueError(
                f"surface must lie in [{self._surface_min}, {self._surface_max}] "
                f"with s > 0; got {surface!r}."
            )
        self._set_angular_grid(theta, zeta)
        values = {}
        for name, spline in self._coordinate_splines.items():
            coefficients = spline(surface)
            radial_coefficients = self._radial_splines[name](surface)
            if name == "R":
                values[name] = self._sum_harmonics(coefficients, "cos")
                values[name + "_s"] = self._sum_harmonics(radial_coefficients, "cos")
                values[name + "_theta"] = self._sum_harmonics(
                    -self._geometry_m * coefficients, "sin"
                )
                values[name + "_zeta"] = self._sum_harmonics(
                    self._geometry_n * coefficients, "sin"
                )
            else:
                values[name] = self._sum_harmonics(coefficients, "sin")
                values[name + "_s"] = self._sum_harmonics(radial_coefficients, "sin")
                values[name + "_theta"] = self._sum_harmonics(
                    self._geometry_m * coefficients, "cos"
                )
                values[name + "_zeta"] = self._sum_harmonics(
                    -self._geometry_n * coefficients, "cos"
                )

        values["phi"] = self._zeta_grid[None, :] - values["nu"]
        values["phi_s"] = -values["nu_s"]
        values["phi_theta"] = -values["nu_theta"]
        values["phi_zeta"] = 1 - values["nu_zeta"]
        for name, spline in self._flux_splines.items():
            values[name] = float(spline(surface))
        values["psi0"] = self._psi0
        return values

    def _sample_geometry(self, surface, theta, zeta, expected_orientation=None):
        """Calculate geometric continuum weights and diagnostics on one surface.

        Args:
            surface (float): Normalized flux inside the copied spline interval,
                excluding the axis.
            theta (array-like): One-dimensional poloidal angles in radians.
            zeta (array-like): One-dimensional Boozer toroidal angles in radians.
            expected_orientation (int, optional): Required sign of the Jacobian,
                +1 or -1, for comparison with another surface. If omitted, either
                orientation is accepted provided it is constant on this grid.

        Returns:
            dict: Arrays of shape (len(theta), len(zeta)): the signed ``Jg``
            in m^3, ``H`` in m^4, ``S`` in m^2, and the positive weights
            ``A = H / (abs(Jg) * S)`` and ``W0 = H * abs(Jg) / (psi0**2 * S)``.
            Their units are 1/m and m/T^2, respectively, with psi0 in T*m^2.
            ``jacobian_quality`` is abs(Jg) divided by the product of the three
            tangent lengths. Scalar entries are ``min_jacobian_quality``,
            ``orientation`` (+1 or -1), and ``iota``.

        Raises:
            ValueError: If coordinates or computed quantities are nonfinite,
                geometry is singular, the sampled orientation is inconsistent,
                or expected_orientation is not +1 or -1.
        """
        if expected_orientation is not None and (
            isinstance(expected_orientation, (bool, np.bool_))
            or not isinstance(expected_orientation, Integral)
            or expected_orientation not in (-1, 1)
        ):
            raise ValueError("expected_orientation must be +1, -1, or None.")

        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                values = self._sample_coordinates(surface, theta, zeta)
                for coordinate in ("R", "Z", "nu"):
                    for suffix in ("", "_s", "_theta", "_zeta"):
                        name = coordinate + suffix
                        if not np.all(np.isfinite(values[name])):
                            raise ValueError(f"{name} must be finite at s={surface}.")
                iota = values["iota"]
                if not np.isfinite(iota):
                    raise ValueError(f"iota must be finite at s={surface}.")

                radius = values["R"]
                r_s = np.stack(
                    (values["R_s"], -radius * values["nu_s"], values["Z_s"]),
                    axis=-1,
                )
                r_theta = np.stack(
                    (
                        values["R_theta"],
                        -radius * values["nu_theta"],
                        values["Z_theta"],
                    ),
                    axis=-1,
                )
                r_zeta = np.stack(
                    (
                        values["R_zeta"],
                        radius * (1 - values["nu_zeta"]),
                        values["Z_zeta"],
                    ),
                    axis=-1,
                )
                area_vector = np.cross(r_theta, r_zeta)
                field_line_tangent = r_zeta + iota * r_theta
                H = np.sum(area_vector * area_vector, axis=-1)
                S = np.sum(field_line_tangent * field_line_tangent, axis=-1)
                Jg = np.sum(r_s * area_vector, axis=-1)
                for name, array in (("H", H), ("S", S)):
                    if np.any(array <= 0):
                        raise ValueError(f"{name} must be positive at s={surface}.")
                if np.any(Jg == 0):
                    raise ValueError(f"Jg must be nonzero at s={surface}.")

                orientation = int(np.sign(Jg.flat[0]))
                if np.any(orientation * Jg < 0):
                    raise ValueError(f"Jg changes sign on the grid at s={surface}.")
                if (
                    expected_orientation is not None
                    and orientation != expected_orientation
                ):
                    raise ValueError(
                        f"Jg orientation {orientation} at s={surface} differs "
                        f"from expected orientation {expected_orientation}."
                    )

                absolute_jacobian = np.abs(Jg)
                quality = absolute_jacobian / np.linalg.norm(r_s, axis=-1)
                quality /= np.linalg.norm(r_theta, axis=-1)
                quality /= np.linalg.norm(r_zeta, axis=-1)
                ratio = H / S
                A = ratio / absolute_jacobian
                W0 = ratio * absolute_jacobian / np.square(self._psi0)
                for name, array in (
                    ("jacobian_quality", quality), ("A", A), ("W0", W0)
                ):
                    if not np.all(np.isfinite(array)) or np.any(array <= 0):
                        raise ValueError(
                            f"{name} must be finite and positive at s={surface}."
                        )
        except FloatingPointError as error:
            raise ValueError(
                f"Geometry calculation failed at s={surface}: {error}."
            ) from error

        return {
            "Jg": Jg,
            "H": H,
            "S": S,
            "A": A,
            "W0": W0,
            "jacobian_quality": quality,
            "min_jacobian_quality": float(quality.min()),
            "orientation": orientation,
            "iota": iota,
        }

    def _assemble_quadrature(self, theta, zeta, geometry):
        """Assemble reference matrices using a uniform full-torus angular average.

        Args:
            theta (array-like): Uniform poloidal grid on [0, 2*pi), starting at
                zero and excluding the upper endpoint.
            zeta (array-like): Uniform toroidal grid on [0, 2*pi), also starting
                at zero and excluding the upper endpoint. Use the full torus.
            geometry (dict): Positive finite real ``A`` and ``W0`` arrays of
                shape (len(theta), len(zeta)) and a finite scalar ``iota``, as
                returned by ``_sample_geometry`` on these grids.

        Returns:
            tuple: Real (N, N) stiffness K and density-independent mass M0.
            K uses the derivative d/dzeta + iota*d/dtheta of each normalized
            cosine basis function. Both matrices use the average 1/Nq, where
            Nq is the number of grid points; their units are 1/m and m/T^2.

        Raises:
            ValueError: If grids, weights, or iota are invalid, or arithmetic
                produces nonfinite values.

        This reference for small validation cases allocates Nq-by-N arrays.
        An undersampled basis can give a singular mass matrix.
        """
        grids = []
        for name, angles in (("theta", theta), ("zeta", zeta)):
            angles = np.asarray(angles)
            if (
                angles.ndim != 1
                or angles.size == 0
                or not np.issubdtype(angles.dtype, np.number)
                or np.iscomplexobj(angles)
                or not np.all(np.isfinite(angles))
            ):
                raise ValueError(f"{name} must be a finite nonempty real 1D array.")
            expected = 2 * np.pi * np.arange(angles.size) / angles.size
            if not np.allclose(angles, expected, rtol=0, atol=1e-12):
                raise ValueError(
                    f"{name} must be a uniform full-torus grid on [0, 2*pi)."
                )
            grids.append(angles)
        theta, zeta = grids
        shape = (len(theta), len(zeta))
        weights = {}
        for name in ("A", "W0"):
            array = np.asarray(geometry[name])
            if (
                array.shape != shape
                or not np.issubdtype(array.dtype, np.number)
                or np.iscomplexobj(array)
                or not np.all(np.isfinite(array))
                or np.any(array <= 0)
            ):
                raise ValueError(
                    f"{name} must be finite and positive with shape {shape}."
                )
            weights[name] = array
        iota = geometry["iota"]
        if not isinstance(iota, Real) or not np.isfinite(iota):
            raise ValueError("iota must be a finite real scalar.")

        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                theta_mesh, zeta_mesh = np.meshgrid(theta, zeta, indexing="ij")
                phase = theta_mesh.ravel()[:, None] * self.modes[:, 0]
                phase -= zeta_mesh.ravel()[:, None] * self.modes[:, 1]
                normalization = np.full(len(self.modes), np.sqrt(2.0))
                normalization[np.all(self.modes == 0, axis=1)] = 1.0
                parallel_numbers = iota * self.modes[:, 0] - self.modes[:, 1]
                basis_values = np.cos(phase) * normalization
                basis_derivatives = -np.sin(phase) * (normalization * parallel_numbers)

                weighted_basis = np.sqrt(weights["W0"].ravel())[:, None] * basis_values
                weighted_derivatives = (
                    np.sqrt(weights["A"].ravel())[:, None] * basis_derivatives
                )
                M0 = (weighted_basis.T @ weighted_basis) / theta_mesh.size
                K = (weighted_derivatives.T @ weighted_derivatives) / theta_mesh.size
                if not np.all(np.isfinite(K)) or not np.all(np.isfinite(M0)):
                    raise ValueError("Direct quadrature produced nonfinite matrices.")
        except FloatingPointError as error:
            raise ValueError(f"Direct quadrature failed: {error}.") from error
        return K, M0

    def _plan_basis(self):
        """Plan cosine normalization and all pairwise Fourier moment indices.

        Returns:
            dict: ``normalization`` has length N, with 1 for the constant and
            sqrt(2) otherwise. ``difference_modes`` and ``sum_modes`` have
            shape (N, N, 2), with entry (i, j) equal to k_i - k_j or k_i + k_j.
            The corresponding ``difference_allowed`` and ``sum_allowed`` masks
            have shape (N, N). A moment is allowed only if its toroidal index
            is divisible by nfp; all other full-torus moments are exactly zero.

        Raises:
            ValueError: If pairwise indices could overflow signed 64-bit integers.

        This allocates O(N**2) lookup arrays on demand, keeping initialization
        inexpensive. It preserves the input mode order and retains all allowed
        moments, including those beyond the equilibrium Fourier cutoff.
        """
        largest_index = max(abs(int(self.modes.min())), abs(int(self.modes.max())))
        if largest_index > np.iinfo(np.int64).max // 2:
            raise ValueError(
                "Mode indices are too large for signed 64-bit sums and differences."
            )

        normalization = np.full(len(self.modes), np.sqrt(2.0))
        constant = np.all(self.modes == 0, axis=1)
        normalization[constant] = 1.0
        difference_modes = self.modes[:, None, :] - self.modes[None, :, :]
        sum_modes = self.modes[:, None, :] + self.modes[None, :, :]
        return {
            "normalization": normalization,
            "difference_modes": difference_modes,
            "sum_modes": sum_modes,
            "difference_allowed": difference_modes[:, :, 1] % self._nfp == 0,
            "sum_allowed": sum_modes[:, :, 1] % self._nfp == 0,
        }

    def _plan_angular_grid(self, basis, shape=None, *, max_shape=None):
        """Plan one-field-period sampling and checked FFT moment lookups.

        Args:
            basis (dict): The result of ``_plan_basis()`` for this calculation.
            shape (tuple, optional): Positive integer counts (Ntheta, Nzeta).
                Defaults to the smallest odd counts that keep the coordinate
                harmonics and allowed moments strictly below Nyquist.
            max_shape (tuple, optional): Maximum counts in each direction,
                checked before allocating angular grids or FFT lookups.

        Returns:
            dict: Endpoint-excluded arrays ``theta`` on [0, 2*pi) and ``zeta``
            on [0, 2*pi/nfp), the selected ``shape``, and ``minimum_shape``.
            ``difference_indices`` and ``sum_indices`` are pairs of 1D arrays
            indexing only the corresponding True entries in the basis masks,
            in NumPy's boolean-indexing order.

        Raises:
            ValueError: If equilibrium indices are not integers with toroidal
                period nfp, or the grid is invalid, too small, or above its limit.

        For ``F = fft2(weight) / weight.size``, the quadrature estimate of the
        cosine moment (m, n) is ``F[m % Ntheta, (-n // nfp) % Nzeta].real`` when
        n is divisible by nfp.
        The FFT uses exp(-i * (p*theta + q*nfp*zeta)), whereas our Fourier phase
        is m*theta - n*zeta. Bounds are checked before applying modulo indices.

        Nonlinear geometric weights can contain arbitrarily higher harmonics.
        """
        for indices in (self._geometry_m, self._geometry_n):
            if not np.issubdtype(indices.dtype, np.integer):
                raise ValueError("Equilibrium Fourier indices must be integers.")
        if np.any(self._geometry_n % self._nfp != 0):
            raise ValueError(
                "Equilibrium toroidal indices must be multiples of nfp "
                "for one-field-period sampling."
            )

        poloidal_bound = max(
            abs(int(self._geometry_m.min())), abs(int(self._geometry_m.max()))
        )
        toroidal_bound = max(
            abs(int(self._geometry_n.min())), abs(int(self._geometry_n.max()))
        ) // self._nfp
        for kind in ("difference", "sum"):
            selected = basis[kind + "_modes"][basis[kind + "_allowed"]]
            if selected.size:
                poloidal_bound = max(poloidal_bound, int(np.abs(selected[:, 0]).max()))
                toroidal_bound = max(
                    toroidal_bound, int(np.abs(selected[:, 1]).max()) // self._nfp
                )
        minimum_shape = (2 * poloidal_bound + 1, 2 * toroidal_bound + 1)

        if shape is None:
            shape = minimum_shape
        else:
            shape = np.asarray(shape)
            if (
                shape.shape != (2,)
                or not np.issubdtype(shape.dtype, np.integer)
                or np.any(shape <= 0)
            ):
                raise ValueError("grid shape must contain two positive integer counts.")
            shape = tuple(int(count) for count in shape)
            if shape[0] < minimum_shape[0] or shape[1] < minimum_shape[1]:
                raise ValueError(
                    f"grid shape {shape} must be at least {minimum_shape} to keep "
                    "coordinate harmonics and required moments below Nyquist."
                )

        if max_shape is not None:
            limit = np.asarray(max_shape)
            if (
                limit.shape != (2,)
                or not np.issubdtype(limit.dtype, np.integer)
                or np.any(limit <= 0)
            ):
                raise ValueError("max_shape must contain two positive integer counts.")
            if shape[0] > limit[0] or shape[1] > limit[1]:
                raise ValueError(
                    f"grid shape {shape} exceeds max_shape {tuple(limit.tolist())}."
                )

        ntheta, nzeta = shape
        grid = {"shape": shape, "minimum_shape": minimum_shape}
        grid["theta"] = 2 * np.pi * np.arange(ntheta) / ntheta
        grid["zeta"] = 2 * np.pi * np.arange(nzeta) / (self._nfp * nzeta)
        for kind in ("difference", "sum"):
            selected = basis[kind + "_modes"][basis[kind + "_allowed"]]
            theta_indices = selected[:, 0] % ntheta
            zeta_indices = (-selected[:, 1] // self._nfp) % nzeta
            grid[kind + "_indices"] = (theta_indices, zeta_indices)
        return grid

    def _fourier_moments(self, basis, grid, geometry):
        """Extract the cosine moments needed for the retained basis.

        Args:
            basis (dict): The result of ``_plan_basis()`` for this calculation.
            grid (dict): The corresponding ``_plan_angular_grid()`` result.
            geometry (dict): Positive finite real ``A`` and ``W0`` arrays,
                sampled on this grid over one field period.

        Returns:
            dict: Real (N, N) arrays ``A_difference``, ``A_sum``,
            ``W0_difference``, and ``W0_sum``. Each entry is the normalized
            full-torus average of the weight times cos(m*theta - n*zeta),
            at the indicated pairwise mode. Forbidden moments are zero.

        Raises:
            ValueError: If weights have invalid values or shapes, or the
                transform produces nonfinite values.
        """
        moments = {}
        for name in ("A", "W0"):
            weight = np.asarray(geometry[name])
            if (
                weight.shape != grid["shape"]
                or not np.issubdtype(weight.dtype, np.number)
                or np.iscomplexobj(weight)
                or not np.all(np.isfinite(weight))
                or np.any(weight <= 0)
            ):
                raise ValueError(
                    f"{name} must be finite and positive with shape {grid['shape']}."
                )
            try:
                with np.errstate(over="raise", invalid="raise", divide="raise"):
                    coefficients = np.fft.fft2(weight / weight.size)
            except FloatingPointError as error:
                raise ValueError(
                    f"{name} Fourier transform failed: {error}."
                ) from error
            if not np.all(np.isfinite(coefficients)):
                raise ValueError(f"{name} Fourier transform produced nonfinite values.")
            for kind in ("difference", "sum"):
                allowed = basis[kind + "_allowed"]
                values = np.zeros(allowed.shape)
                values[allowed] = coefficients.real[grid[kind + "_indices"]]
                moments[name + "_" + kind] = values
            del coefficients
        return moments

    def _assemble_moments(self, basis, moments, iota):
        """Assemble real matrices from cosine sum and difference moments.

        Args:
            basis (dict): The result of ``_plan_basis()`` for this calculation.
            moments (dict): The corresponding ``_fourier_moments()`` result.
            iota (float): Finite rotational transform on the sampled surface.

        Returns:
            tuple: Real (N, N) stiffness K and density-independent mass M0,
            with units 1/m and m/T^2 and normalized full-torus averaging.

        Raises:
            ValueError: If moments or iota are invalid, or arithmetic produces
                nonfinite matrices.

        The cosine product gives M0_ij = a_i*a_j*(W0_difference + W0_sum)/2.
        The sine derivative product gives
        K_ij = a_i*a_j*kappa_i*kappa_j*(A_difference - A_sum)/2,
        where kappa_j = iota*m_j - n_j. Both triangles use these formulas.
        """
        if not isinstance(iota, Real) or not np.isfinite(iota):
            raise ValueError("iota must be a finite real scalar.")
        shape = (len(self.modes), len(self.modes))
        arrays = {}
        for name in ("A_difference", "A_sum", "W0_difference", "W0_sum"):
            values = np.asarray(moments[name])
            if (
                values.shape != shape
                or not np.issubdtype(values.dtype, np.number)
                or np.iscomplexobj(values)
                or not np.all(np.isfinite(values))
            ):
                raise ValueError(
                    f"{name} must be a finite real array of shape {shape}."
                )
            arrays[name] = values.astype(float, copy=False)
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                normalization = basis["normalization"]
                factors = np.outer(normalization, normalization) / 2
                parallel_numbers = iota * self.modes[:, 0] - self.modes[:, 1]
                M0 = arrays["W0_difference"] + arrays["W0_sum"]
                M0 *= factors
                K = arrays["A_difference"] - arrays["A_sum"]
                K *= factors
                K *= np.outer(parallel_numbers, parallel_numbers)
        except FloatingPointError as error:
            raise ValueError(f"Fourier matrix assembly failed: {error}.") from error
        if not np.all(np.isfinite(K)) or not np.all(np.isfinite(M0)):
            raise ValueError("Fourier matrix assembly produced nonfinite matrices.")
        return K, M0

    def _sample_moments(self, surface, basis, grid, expected_orientation):
        """Return moments and scalar geometry diagnostics for one surface/grid."""
        try:
            geometry = self._sample_geometry(
                surface, grid["theta"], grid["zeta"], expected_orientation
            )
            moments = self._fourier_moments(basis, grid, geometry)
        except ValueError as error:
            raise ValueError(
                f"Quadrature sampling failed at s={surface}, "
                f"grid {grid['shape']}: {error}"
            ) from error
        diagnostics = {
            "iota": geometry["iota"],
            "orientation": geometry["orientation"],
            "min_jacobian_quality": geometry["min_jacobian_quality"],
        }
        return moments, diagnostics

    def _compare_moments(self, coarse, fine, rtol, atol):
        """Compare every moment using its weight's zero-moment scale.

        For each weight, divide both grids' moments by the larger zero moment.
        Require abs(fine - coarse) <= atol + rtol*max(abs(coarse), abs(fine))
        in these scaled units. Return the worst tolerance ratio and mode pair;
        a ratio <= 1 passes. Zero error with zero tolerance also passes.
        """
        worst_ratio = -1.0
        for weight in ("A", "W0"):
            scale = max(
                coarse[weight + "_difference"][0, 0],
                fine[weight + "_difference"][0, 0],
            )
            if not np.isfinite(scale) or scale <= 0:
                raise ValueError(
                    f"{weight} zero-moment scale must be finite and positive."
                )
            for kind in ("difference", "sum"):
                name = weight + "_" + kind
                old = coarse[name] / scale
                new = fine[name] / scale
                change = np.abs(new - old)
                tolerance = atol + rtol * np.maximum(np.abs(old), np.abs(new))
                ratio = np.zeros_like(change)
                np.divide(change, tolerance, out=ratio, where=tolerance > 0)
                ratio[(tolerance == 0) & (change > 0)] = np.inf
                index = np.unravel_index(np.argmax(ratio), ratio.shape)
                if ratio[index] > worst_ratio:
                    worst_ratio = float(ratio[index])
                    worst_name = name
                    worst_pair = tuple(int(i) for i in index)
        return {
            "converged": worst_ratio <= 1,
            "max_tolerance_ratio": worst_ratio,
            "worst_moment": worst_name,
            "worst_pair": worst_pair,
        }

    def _converge_surface(
        self, surface, basis, *, shape=None, rtol=1e-8, atol=1e-10,
        max_shape=(1024, 1024), expected_orientation=None,
    ):
        """This tests angular integration of the supplied interpolant, 
        and assemble matrices on one surface.

        Args:
            surface (float): Normalized flux within the copied spline interval.
            basis (dict): The result of ``_plan_basis()`` for this calculation.
            shape (tuple, optional): Fixed one-period grid to check against a
                doubled grid. If omitted, start with the planned minimum grid
                and double both dimensions until the moment comparison passes.
            rtol (float): Nonnegative relative tolerance for each moment.
            atol (float): Nonnegative absolute tolerance after scaling moments
                by their weight's mean. At least one tolerance must be positive.
            max_shape (tuple): Maximum grid counts, including verification grids.
                Defaults to (1024, 1024); this is not a total memory limit.
            expected_orientation (int, optional): Required Jacobian sign.

        Returns:
            dict: Matrices ``K`` and ``M0``, scalar ``iota``, ``orientation``,
            and ``min_jacobian_quality`` on the returned grid. ``quadrature``
            records the surface, returned and verification grid shapes,
            tolerances, convergence status, and each moment comparison.
            Automatic mode returns the finer passing grid; fixed mode returns
            the requested grid after checking it against the finer grid.

        Raises:
            ValueError: If settings or sampled geometry are invalid.
            RuntimeError: If refinement would exceed max_shape, or a fixed
                grid fails the moment comparison.

        """
        for name, value in (("rtol", rtol), ("atol", atol)):
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, Real)
                or not np.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{name} must be a finite nonnegative real scalar.")
        if rtol == 0 and atol == 0:
            raise ValueError("At least one of rtol and atol must be positive.")
        if max_shape is None:
            raise ValueError("max_shape is required for bounded quadrature refinement.")
        try:
            grid = self._plan_angular_grid(basis, shape, max_shape=max_shape)
        except ValueError as error:
            raise ValueError(
                f"Quadrature setup failed at s={surface}: {error}"
            ) from error
        history = []
        moments, diagnostics = self._sample_moments(
            surface, basis, grid, expected_orientation
        )
        while True:
            fine_shape = tuple(2 * count for count in grid["shape"])
            try:
                fine_grid = self._plan_angular_grid(
                    basis, fine_shape, max_shape=max_shape
                )
            except ValueError as error:
                detail = "No coarse/fine comparison completed."
                if history:
                    last = history[-1]
                    detail = (
                        f"Worst moment {last['worst_moment']}, "
                        f"pair {last['worst_pair']}, "
                        f"tolerance ratio {last['max_tolerance_ratio']:.3e} > 1."
                    )
                raise RuntimeError(
                    f"Angular quadrature did not converge at s={surface}, "
                    f"last grid {grid['shape']}, rtol={rtol}, atol={atol}. "
                    f"{detail} {error}"
                ) from error
            fine_moments, fine_diagnostics = self._sample_moments(
                surface, basis, fine_grid, diagnostics["orientation"]
            )
            try:
                comparison = self._compare_moments(moments, fine_moments, rtol, atol)
            except ValueError as error:
                raise ValueError(
                    f"Quadrature comparison failed at s={surface}, "
                    f"grids {grid['shape']} and {fine_shape}: {error}"
                ) from error
            comparison["coarse_shape"] = grid["shape"]
            comparison["fine_shape"] = fine_shape
            history.append(comparison)
            if comparison["converged"]:
                if shape is None:
                    moments = fine_moments
                    diagnostics = fine_diagnostics
                    grid = fine_grid
                K, M0 = self._assemble_moments(basis, moments, diagnostics["iota"])
                return {
                    "K": K,
                    "M0": M0,
                    "iota": diagnostics["iota"],
                    "orientation": diagnostics["orientation"],
                    "min_jacobian_quality": diagnostics["min_jacobian_quality"],
                    "quadrature": {
                        "surface": float(surface),
                        "converged": True,
                        "fixed_grid": shape is not None,
                        "shape": grid["shape"],
                        "verification_shape": fine_shape,
                        "rtol": float(rtol),
                        "atol": float(atol),
                        "history": history,
                        "basis_convergence": "unverified",
                        "equilibrium_convergence": "unverified",
                    },
                }
            if shape is not None:
                raise RuntimeError(
                    f"Fixed-grid angular quadrature failed at s={surface}: "
                    f"grid {grid['shape']} versus {fine_shape}, "
                    f"rtol={rtol}, atol={atol}; "
                    f"worst moment {comparison['worst_moment']}, "
                    f"pair {comparison['worst_pair']}, "
                    f"tolerance ratio {comparison['max_tolerance_ratio']:.3e} > 1."
                )
            moments, diagnostics, grid = fine_moments, fine_diagnostics, fine_grid

    def _validate_surfaces(self, surfaces):
        """Return a copy of the requested surfaces without extrapolating."""
        surfaces = np.asarray(surfaces)
        if surfaces.ndim != 1 or surfaces.size == 0:
            raise ValueError("surfaces must be a nonempty one-dimensional array.")
        if not np.issubdtype(surfaces.dtype, np.number) or np.iscomplexobj(surfaces):
            raise ValueError("surfaces must contain real numbers.")
        surfaces = surfaces.astype(float, copy=True)
        invalid = surfaces[~np.isfinite(surfaces)]
        if invalid.size:
            raise ValueError(f"surfaces must be finite; got {invalid[0]}.")

        lower = max(0.0, self.field.s_half_ext[0])
        upper = min(1.0, self.field.s_half_ext[-1])
        invalid = surfaces[(surfaces <= 0) | (surfaces < lower) | (surfaces > upper)]
        if invalid.size:
            raise ValueError(
                f"surfaces must lie in [{lower}, {upper}] with s > 0; "
                f"got {invalid[0]}."
            )
        return surfaces

    def _validate_modes(self, modes):
        """Return the integer mode array and its common cosine-mode family."""
        modes = np.asarray(modes)
        if modes.ndim != 2 or modes.shape[0] == 0 or modes.shape[1] != 2:
            raise ValueError("modes must be a nonempty array of (m, n) pairs.")
        if not np.issubdtype(modes.dtype, np.integer):
            raise ValueError("modes must contain integer mode numbers.")
        try:
            # Converting through Python integers catches unsigned values above int64.
            modes = np.array(modes.tolist(), dtype=np.int64)
        except OverflowError as error:
            raise ValueError(
                "mode numbers must fit in signed 64-bit integers."
            ) from error

        nfp = int(self.field.nfp)
        first_residue = int(modes[0, 1]) % nfp
        mode_family = min(first_residue, nfp - first_residue)
        seen = set()
        for index, (m, n) in enumerate(modes.tolist()):
            if (m, n) in seen or (-m, -n) in seen:
                raise ValueError(
                    f"modes[{index}] = ({m}, {n}) duplicates a cosine basis function."
                )
            seen.add((m, n))

            residue = n % nfp
            family = min(residue, nfp - residue)
            if family != mode_family:
                raise ValueError(
                    f"modes[{index}] = ({m}, {n}) belongs to family {family}; "
                    f"expected family {mode_family} for nfp = {nfp}."
                )
        return modes, mode_family


@dataclass
class Harmonic:
    """
    Represents a harmonic in the Fourier decomposition of an eigenvector.

    Attributes:
        m (int): Poloidal mode number.
        n (int): Toroidal mode number.
        amplitudes (np.ndarray): Array of amplitudes corresponding to radial points.
    """

    m: int
    n: int
    amplitudes: np.ndarray


class ModeContinuum:
    _n: int
    _m: int
    _s: np.array
    _freq: np.array
    r"""
    A class to handle the parsing and storage of continuum modes, which includes
    poloidal and toroidal mode numbers, flux surfaces, and frequencies.
    This class is used to represent the continuum modes extracted from AE3D and STELLGAP
    simulations.
    """

    def __init__(self, m: int, n: int, s=None, freq=None):
        r"""
        Initialize a ModeContinuum instance. s and frequencies can be specified
        but are not necessary to initialize.

        Args:
            m (int): Poloidal mode number.
            n (int): Toroidal mode number.
            s (np.array, optional): Array of flux surfaces. Defaults to None.
            freq (np.array, optional): Array of frequencies corresponding to
                                      the flux surfaces. Defaults to None.
        Raises:
            Exception: If a negative flux label is provided.
        """
        self._m = m
        self._n = n
        self._s = s
        self._freq = freq

    def _check_negative_s(self):
        r"""
        Check if any flux label is negative.
        Raises an exception if a negative flux label is found.
        """
        for s in self._s:
            if s < 0:
                self._negative_exception()

    def _negative_exception(self):
        r"""
        Raises an exception indicating that a negative flux label was provided.
        The flux label must be positive.
        """
        raise Exception(
            "A negative flux label was provided. The flux label must be positive."
        )

    def set_poloidal_mode(self, m: int):
        r"""
        Set the poloidal mode number.
        Args:
            m (int): Poloidal mode number.
        """
        self._m = m

    def set_toroidal_mode(self, n: int):
        r"""
        Set the toroidal mode number.
        Args:
            n (int): Toroidal mode number.
        """
        self._n = n

    def set_points(self, s: np.array, freq: np.array):
        r"""
        Set the flux surfaces and frequencies.
        Args:
            s (np.array): Array of flux surfaces.
            freq (np.array): Array of frequencies corresponding to the flux surfaces.
        """
        self._s = s
        self._freq = freq

        self._check_matching_freqs()

    def get_poloidal_mode(self):
        r"""
        Get the poloidal mode number.

        Returns:
            int: Poloidal mode number.
        """
        return self._m

    def get_toroidal_mode(self):
        r"""
        Get the toroidal mode number.
        Returns:
            int: Toroidal mode number.
        """
        return self._n

    def get_flux_surfaces(self):
        r"""
        Get the flux surfaces.

        Returns:
            np.array: Array of flux surfaces.
        """
        return self._s

    def get_frequencies(self):
        r"""
        Get the frequencies.

        Returns:
            np.array: Array of frequencies.
        """
        return self._freq

    def add_point(self, s: float, freq: float):
        r"""
        Add a point to the flux surfaces and frequencies.
        Args:
            s (float): Flux surface value to add.
            freq (float): Frequency value to add.
        Raises:
            Exception: If the flux surface value is negative.
        """
        if s < 0:
            self._negative_exception()

        self._s = np.append(self._s, s)
        self._freq = np.append(self._freq, freq)


class AlfvenSpecData(np.ndarray):
    r"""
    Subclass of numpy.ndarray with dtype specific to STELLGAP output in
    alfven_spec files.
    """

    def __new__(cls, filenames: list[str]):
        r"""
        Create a new instance of AlfvenSpecData from a list of filenames.

        Args:
            filenames (List[str]): List of filenames containing alfven_spec data.

        Returns:
            AlfvenSpecData: An instance of AlfvenSpecData containing the loaded data.
        """
        if not filenames:
            raise ValueError("No filenames provided")

        data = np.vstack(
            [
                np.loadtxt(
                    fname,
                    dtype=[
                        ("s", float),
                        ("ar", float),
                        ("ai", float),
                        ("beta", float),
                        ("m", int),
                        ("n", int),
                    ],
                )
                for fname in filenames
            ]
        )
        obj = np.asarray(data).view(cls)
        return obj

    @classmethod
    def from_dir(cls, directory: str):
        r"""
        Load all alfven_spec data from a specified directory.

        Args:
            directory (str): Path to the directory containing alfven_spec files.

        Returns:
            AlfvenSpecData: An instance of AlfvenSpecData containing the loaded data.
        """
        files = [
            os.path.join(directory, fname)
            for fname in os.listdir(directory)
            if fname.startswith("alfven_spec")
        ]
        if not files:
            raise ValueError(f"No alfven_spec files found in the directory {directory}")
        return cls(files)

    def nonzero_beta(self):
        r"""
        Filter entries where beta is not zero.

        Returns:
            AlfvenSpecData: An instance of AlfvenSpecData containing the filtered data.
        """
        return self[self["beta"] != 0]

    def sort_by_s(self):
        r"""
        Sort the array based on the 's' field.

        Returns:
            AlfvenSpecData: An instance of AlfvenSpecData sorted by the 's'
            field.
        """
        return self[np.argsort(self["s"])]

    def get_modes(self) -> list[ModeContinuum]:
        r"""
        Extract modes from the AlfvenSpecData, creating ModeContinuum instances
        for each unique combination of poloidal (n) and toroidal (m) mode numbers.

        Returns:
            List[ModeContinuum]: A list of ModeContinuum instances, each representing
            a unique mode with its corresponding flux surfaces (s) and frequencies.
        """
        data = self.nonzero_beta()
        modes = [
            ModeContinuum(
                n=n,
                m=m,
                s=(
                    filtered_data := np.sort(
                        data[(data["n"] == n) & (data["m"] == m)], order="s"
                    )
                )["s"],
                freq=np.sqrt(np.abs(filtered_data["ar"] / filtered_data["beta"])),
            )
            for n, m in {(a["n"], a["m"]) for a in data}
        ]
        return modes

    def condition_number(self):
        r"""
        For each s, compute the condition number as the ratio of largest
        to smallest eigenvalue return the array of s and corresponding
        condition numbers.

        Returns:
            tuple: A tuple containing:
                - s (np.array): Unique flux surface values.
                - condition_numbers (np.array): Condition numbers for each
                  unique flux surface.
        """
        data = self.nonzero_beta().sort_by_s()
        s = np.unique(data["s"])
        condition_numbers = np.array(
            [
                np.max(np.abs(data[data["s"] == s_]["ar"]))
                / np.min(np.abs(data[data["s"] == s_]["ar"]))
                if np.min(np.abs(data[data["s"] == s_]["ar"])) != 0
                else np.inf
                for s_ in s
            ]
        )
        return s, condition_numbers


def plot_continuum_modes(
    overlays: list[list[ModeContinuum]],
    show_legend: bool = False,
    normalized_modes: bool = False,
    yrange: list = None,
) -> go.Figure:
    r"""
    Plot the continuum modes using Plotly. Several overlays can be provided.
    This is useful, for example, in comparing AE3D and STELLGAP results.

    Args:
        overlays: list[list[ModeContinuum]]:
            A list of lists, where each inner list contains ModeContinuum
            instances representing different overlays.
        show_legend (bool, optional): Whether to show the legend in the plot.
                                      Defaults to False.
        normalized_modes (bool, optional): If True, normalize the frequencies
                                          by the Alfven frequency. Defaults to False.
        yrange (list, optional): Custom y-axis range for the plot. If None,
                                 defaults are used based on normalized_modes.
    """
    fig = go.Figure()

    colors = ["blue", "red", "green", "purple"]
    markers = ["circle", "square", "diamond", "cross"]

    for idx, modes in enumerate(overlays):
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]

        for md in modes:
            fig.add_trace(
                go.Scatter(
                    x=md.get_flux_surfaces(),
                    y=md.get_frequencies(),
                    mode="markers",
                    name=(
                        f"m={md.get_poloidal_mode()}, "
                        + f"n={md.get_toroidal_mode()} (Overlay {idx + 1})"
                    ),
                    marker={"size": 3, "symbol": marker, "color": color},
                    line={"width": 0.5, "color": color},
                    text=[
                        (
                            f"m={md.get_poloidal_mode()}, "
                            + f"n={md.get_toroidal_mode()} (Overlay {idx + 1})"
                        ),
                    ],
                    hoverinfo="text+x+y",
                    hoverlabel={
                        "font_size": 16,
                        "bgcolor": "white",
                        "bordercolor": color,
                    },
                )
            )

    if normalized_modes:
        yaxis_title = r"$\text{normalized frequency }\omega/\omega_A$"
        yaxis_range = [0, 5]
    else:
        yaxis_title = r"$\text{Frequency }\omega\text{ [kHz]}$"
        yaxis_range = [0, 600]

    if yrange is not None:
        yaxis_range = yrange

    fig.update_layout(
        autosize=True,
        title=r"$\text{Continuum: }$",
        xaxis_title=r"$\text{Normalized flux }s$",
        yaxis_title=yaxis_title,
        xaxis={
            "range": [
                np.min(
                    [
                        np.min(md.get_flux_surfaces())
                        for modes in overlays
                        for md in modes
                    ]
                ),
                np.max(
                    [
                        np.max(md.get_flux_surfaces())
                        for modes in overlays
                        for md in modes
                    ]
                ),
            ]
        },
        yaxis={"range": yaxis_range},
        legend={
            "title": r"$\text{Mode: }$",
            "yanchor": "top",
            "y": 1.4,
            "xanchor": "center",
            "x": 0.5,
            "orientation": "h",
        },
        showlegend=show_legend,
    )

    return fig
