__all__ = [
    "CatapultBoozerField",
    "CatapultPerturbedBoozerField",
    "CatapultCartesianField",
]

import numpy as np

from firm3d.catapult.utils import (
    boozer_interpolant,
    boozer_saw_interpolant,
    cartesian_interpolant,
)
from firm3d.field.boozermagneticfield import (
    ShearAlfvenWave,
    ShearAlfvenWavesSuperposition,
)

_PRECISIONS = {"double": np.float64, "single": np.float32}


def _dtype_from_precision(precision):
    """
    Map a precision name, "double" or "single", or a numpy float dtype, onto
    np.float64 or np.float32.
    """
    if isinstance(precision, str) and precision.lower() in _PRECISIONS:
        return np.dtype(_PRECISIONS[precision.lower()])
    dtype = np.dtype(precision)
    if dtype not in (np.float32, np.float64):
        raise ValueError(f"precision must be 'double' or 'single', got {precision!r}")
    return dtype


class _CatapultBoozerTable:
    """What the Boozer field objects share: the precision and the equilibrium."""

    def __init__(self, B0, ns, ntheta, nzeta, precision, field_types):
        self.dtype = _dtype_from_precision(precision)
        self.ns, self.ntheta, self.nzeta = ns, ntheta, nzeta
        if B0.field_type not in field_types:
            raise ValueError(
                f"Unsupported field type {B0.field_type!r}, expected one of "
                f"{field_types}"
            )
        self.field_type = B0.field_type
        self.vacuum = B0.field_type == "vac"
        self.nfp = B0.nfp
        self.psi0 = B0.psi0

    @property
    def precision(self):
        return "single" if self.dtype == np.float32 else "double"


class CatapultBoozerField(_CatapultBoozerTable):
    r"""
    An equilibrium magnetic field in Boozer coordinates tabulated for CATAPULT.

    The GPU kernels read the field from a table of :math:`|B|`, its
    derivatives, :math:`G`, :math:`I` and :math:`\iota` on a grid in
    :math:`(s, \theta, \zeta)`. This class builds that table once, at a chosen
    resolution and precision, so that it can be reused across tracing calls;
    it is the GPU counterpart of :class:`InterpolatedBoozerField`. Pass it as
    the ``field`` argument of :func:`trace_particles_boozer_gpu` or
    :func:`save_trajectories_boozer_gpu`. For a perturbed field use
    :class:`CatapultPerturbedBoozerField`.

    Args:
        field: A :class:`BoozerMagneticField` of type ``"vac"`` or ``""``.
        ns, ntheta, nzeta: The number of interpolation cells in each
            coordinate.
        precision: ``"double"`` (the default) or ``"single"``: the precision
            the table is stored in and the kernels run in.

    Attributes:
        dtype: The numpy dtype matching ``precision``.
        srange, trange, zrange: ``(start, end, npoints)`` of the grid in each
            coordinate.
        quad_info: The tabulated field, in ``dtype``.
        maxJ: The largest Jacobian seen on the grid, for rejection sampling
            of positions.
        field_type, vacuum, nfp, psi0: Taken from the field.
    """

    def __init__(self, field, ns, ntheta, nzeta, precision="double"):
        if isinstance(field, ShearAlfvenWave):
            raise TypeError(
                "CatapultBoozerField tabulates equilibrium fields; use "
                "CatapultPerturbedBoozerField for a ShearAlfvenWavesSuperposition"
            )
        super().__init__(field, ns, ntheta, nzeta, precision, ("vac", ""))
        self.field = field
        self.srange, self.trange, self.zrange, self.quad_info, self.maxJ = (
            boozer_interpolant(
                field,
                field.nfp,
                ns,
                ntheta,
                nzeta,
                vacuum=self.vacuum,
                dtype=self.dtype,
            )
        )


class CatapultPerturbedBoozerField(_CatapultBoozerTable):
    r"""
    A magnetic field in Boozer coordinates with shear Alfven waves, tabulated
    for CATAPULT.

    Builds the table the perturbed GPU kernels read, once, from the equilibrium
    under the waves, and keeps the waves' harmonics alongside it. Pass it as
    the ``perturbed_field`` argument of
    :func:`trace_particles_boozer_perturbed_gpu`.

    Args:
        perturbed_field: A :class:`ShearAlfvenWavesSuperposition` whose
            equilibrium is of type ``"vac"`` or ``"nok"``.
        ns, ntheta, nzeta: The number of interpolation cells in each
            coordinate.
        precision: ``"double"`` (the default) or ``"single"``: the precision
            the table is stored in and the kernels run in.

    Attributes:
        dtype: The numpy dtype matching ``precision``.
        srange, trange, zrange: ``(start, end, npoints)`` of the grid in each
            coordinate.
        quad_info: The tabulated equilibrium, in ``dtype``.
        maxJ: The largest Jacobian seen on the grid, for rejection sampling
            of positions.
        field_type, vacuum, nfp, psi0: Taken from the equilibrium.
        saw_omega, saw_srange, saw_m, saw_n, saw_phihats, saw_nharmonics: The
            waves, as the kernels take them.
    """

    def __init__(self, perturbed_field, ns, ntheta, nzeta, precision="double"):
        if not isinstance(perturbed_field, ShearAlfvenWavesSuperposition):
            raise TypeError(
                "CatapultPerturbedBoozerField needs a ShearAlfvenWavesSuperposition; "
                "use CatapultBoozerField for an equilibrium field"
            )
        B0 = perturbed_field.B0
        super().__init__(B0, ns, ntheta, nzeta, precision, ("vac", "nok"))
        self.perturbed_field = perturbed_field
        self.B0 = B0
        self.srange, self.trange, self.zrange, self.quad_info, self.maxJ = (
            boozer_saw_interpolant(B0, B0.nfp, ns, ntheta, nzeta, dtype=self.dtype)
        )
        waves = [perturbed_field.get_wave(i) for i in range(len(perturbed_field))]
        self.saw_nharmonics = len(waves)
        self.saw_omega = waves[0].omega
        saw_s = waves[0].phihat.get_s_basis()
        self.saw_srange = (saw_s[0], saw_s[-1], len(saw_s))
        self.saw_m = [wave.Phim for wave in waves]
        self.saw_n = [wave.Phin for wave in waves]
        self.saw_phihats = np.ascontiguousarray(
            np.column_stack(
                [np.array([wave.phihat(s_val) for s_val in saw_s]) for wave in waves]
            ),
            dtype=self.dtype,
        )


class CatapultCartesianField:
    r"""
    A magnetic field in Cartesian coordinates tabulated for CATAPULT.

    Builds the table the Cartesian GPU kernel reads, once, so that it can be
    reused across tracing calls. Pass it as the ``field`` argument of
    :func:`trace_particles_cartesian_gpu` or
    :func:`save_trajectories_cartesian_gpu`, with ``surface_classifier=None``.

    Args:
        field: A simsopt :class:`InterpolatedField` in cylindrical coordinates,
            with ``r_range``, ``phi_range`` and ``z_range`` attributes.
        surface_classifier: A simsopt :class:`SurfaceClassifier`. Its signed
            distance to the plasma boundary is tabulated alongside the field
            and is what the kernel tests to decide that a particle is lost.
        precision: ``"double"`` (the default) or ``"single"``: the precision
            the table is stored in and the kernel runs in.

    Attributes:
        dtype: The numpy dtype matching ``precision``.
        rrange, phirange, zrange: ``(start, end, npoints)`` of the grid in
            each coordinate.
        quad_info: The tabulated field and distance function, in ``dtype``.
    """

    def __init__(self, field, surface_classifier, precision="double"):
        self.dtype = _dtype_from_precision(precision)
        self.field = field
        self.surface_classifier = surface_classifier
        self.rrange, self.phirange, self.zrange, self.quad_info = cartesian_interpolant(
            field, surface_classifier, dtype=self.dtype
        )

    @property
    def precision(self):
        return "single" if self.dtype == np.float32 else "double"
