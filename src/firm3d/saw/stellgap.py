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
    sampling. Matrix assembly and eigensolves will be added in subsequent steps.

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
