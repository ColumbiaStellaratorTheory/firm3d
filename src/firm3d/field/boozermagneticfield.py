import os.path
import warnings
from typing import Union

import numpy as np
from booz_xform import Booz_xform
from scipy.interpolate import CubicSpline, make_interp_spline

import firm3dpp as sopp

from .._core.util import (
    align_and_pad,
    allocate_aligned_and_padded_array,
    parallel_loop_bounds,
)
from ..saw.ae3d import AE3DEigenvector
from ..saw.far3d import FAR3DEigenvector
from ..util.constants import (
    VACUUM_PERMEABILITY as MU0,
)
from ..util.constants import (
    VACUUM_PERMITTIVITY as EPS0,
)

__all__ = [
    "BoozerMagneticField",
    "BoozerAnalytic",
    "BoozerRadialInterpolant",
    "BoozerSplineField",
    "InterpolatedBoozerField",
    "ShearAlfvenWave",
    "ShearAlfvenHarmonic",
    "ShearAlfvenWavesSuperposition",
    "InterpolatedShearAlfvenWave",
]

try:
    from mpi4py import MPI
except ImportError:
    MPI = None


class BoozerMetric:
    r"""
     A generic class representing the metric tensor in normalized Boozer coordinates
     :math:`(s, \theta, \zeta)`, where:

     - :math:`s` is the normalized toroidal flux, :math:`s = \psi / \psi_0`.
     - :math:`\theta` is the poloidal Boozer angle.
     - :math:`\zeta` is the toroidal Boozer angle.

     The metric tensor defines the local geometry of the magnetic field in
     these coordinates.
     Given the symmetry of the metric tensor, the components stored are:

     .. math::
         g_{ij} = \begin{pmatrix}
         g_{ss} & g_{s\theta} & g_{s\zeta} \\
         g_{s\theta} & g_{\theta\theta} & g_{\theta\zeta} \\
         g_{s\zeta} & g_{\theta\zeta} & g_{\zeta\zeta}
         \end{pmatrix},

     where the metric components :math:`g_{ij}` are functions of
     :math:`(s, \theta, \zeta)`.

     **Attributes:**

     - `gss` : Co-(Counter-)variant metric component :math:`g_{ss}`
       (`g^{ss}`)).
     - `gst` : Co-(Counter-)variant metric component :math:`g_{s\theta}`
       (`g^{s\theta}`).
     - `gsz` : Co-(Counter-)variant metric component :math:`g_{s\zeta}`
       (`g^{s\zeta}`).
     - `gtt` : Co-(Counter-)variant metric component :math:`g_{\theta\theta}`
       (`g^{\theta\theta}`).
     - `gtz` : Co-(Counter-)variant metric component :math:`g_{\theta\zeta}`
       (`g^{\theta\theta}`).
     - `gzz` : Co-(Counter-)variant metric component :math:`g_{\zeta\zeta}`
       (`g^{\zeta\zeta}`).

     **Usage Example:**

     .. code-block:: python

         # Use BoozerMagneticField object (named `bfield` here)
         # to obtain the covariant metric tensor
         covariant_metric = bfield.get_covariant_metric()

         # Convert covariant to contravariant metric tensor
         contravariant_metric = covariant_metric.to_contravariant()

         # Access specific metric components
         gss_component = covariant_metric.ss
         gst_component = covariant_metric.st

         # Convert to matrix form for a single point
         matrix_form = covariant_metric[0].as_matrix

         # Compute determinant
         determinant = covariant_metric.det
    """

    def __init__(self, gss, gst, gsz, gtt, gtz, gzz):
        self.ss = np.asarray(gss)
        self.st = np.asarray(gst)
        self.sz = np.asarray(gsz)
        self.tt = np.asarray(gtt)
        self.tz = np.asarray(gtz)
        self.zz = np.asarray(gzz)

        shape = self.ss.shape
        for g in [self.st, self.sz, self.tt, self.tz, self.zz]:
            if g.shape != shape:
                raise ValueError("All metric components must have the same shape")
        for g in [self.ss, self.tt, self.zz]:
            if not (g > 0).all():
                raise ValueError("All diagonal metric components must be positive")

    def as_matrix(self, idx=None):
        """
        Return the metric tensor as a 3x3 matrix for a given point.

        Parameters
        ----------
        idx : int, optional
            Index of the point to get the matrix for.
            If None and there's only one point, return that point's matrix.

        Returns
        -------
            numpy.ndarray
            3x3 matrix representing the metric tensor
        """
        if idx is None:
            if self.ss.size == 1:
                idx = 0
            else:
                raise ValueError("Must specify idx for multi-point metric")

        return np.array(
            [
                [self.ss[idx], self.st[idx], self.sz[idx]],
                [self.st[idx], self.tt[idx], self.tz[idx]],
                [self.sz[idx], self.tz[idx], self.zz[idx]],
            ]
        )

    def det(self):
        """
        Compute the determinant of the metric tensor at each point.

        Returns
        -------
        numpy.ndarray
            Array of determinant values
        """
        return (
            self.ss * (self.tt * self.zz - self.tz**2)
            - self.st * (self.st * self.zz - self.tz * self.sz)
            + self.sz * (self.st * self.tz - self.sz * self.tt)
        )


class CovariantBoozerMetric(BoozerMetric):
    r"""
    Represents the covariant metric tensor for normalized Boozer coordinates
    :math:`(s, \theta, \zeta)` in a magnetic field. The covariant metric defines the
    local geometry of the space with respect to the basis vectors
    :math:`(\nabla s, \nabla \theta, \nabla \zeta)`.

    The covariant metric tensor :math:`g_{ij}` in Boozer coordinates is given by:

    .. math::
        g_{ij} = \begin{pmatrix}
        g_{ss} & g_{s\theta} & g_{s\zeta} \\
        g_{s\theta} & g_{\theta\theta} & g_{\theta\zeta} \\
        g_{s\zeta} & g_{\theta\zeta} & g_{\zeta\zeta}
        \end{pmatrix},

    where each component :math:`g_{ij}` is a function of the Boozer
    coordinates :math:`(s, \theta, \zeta)`.

    **Methods:**

    - `to_contravariant()`: Converts the covariant metric to its contravariant
        form by inverting the metric tensor. This operation is mathematically
        equivalent to computing the inverse of the covariant metric matrix.

    **Usage Example:**

    .. code-block:: python

        # Given a BoozerMagneticField instance named `bfield`
        covariant_metric = bfield.get_covariant_metric()

        # Convert to contravariant metric
        contravariant_metric = covariant_metric.to_contravariant()

        # Access specific metric components
        gss_component = covariant_metric.ss

        # Compute the determinant of the metric tensor
        determinant = covariant_metric.det

    **Returns:**

    An instance of :class:`ContravariantBoozerMetric` representing the
    contravariant form of the metric.

    """

    def to_contravariant(self):
        """
        Converts the covariant metric to its contravariant form by inverting
        the metric tensor.

        Returns
        -------
        ContravariantBoozerMetric
            The contravariant form of the metric.

        Raises
        -------
        `LinAlgError`: If the matrix inversion fails, due to the matrix being singular.
        """
        # Vectorized matrix inversion for better performance
        n = len(self.ss)
        matrices = np.zeros((n, 3, 3))
        matrices[:, 0, 0] = self.ss
        matrices[:, 0, 1] = matrices[:, 1, 0] = self.st
        matrices[:, 0, 2] = matrices[:, 2, 0] = self.sz
        matrices[:, 1, 1] = self.tt
        matrices[:, 1, 2] = matrices[:, 2, 1] = self.tz
        matrices[:, 2, 2] = self.zz

        inv_matrices_full = np.linalg.inv(matrices)
        inv_matrices = np.zeros((n, 6))
        inv_matrices[:, 0] = inv_matrices_full[:, 0, 0]  # gss
        inv_matrices[:, 1] = inv_matrices_full[:, 0, 1]  # gst
        inv_matrices[:, 2] = inv_matrices_full[:, 0, 2]  # gsz
        inv_matrices[:, 3] = inv_matrices_full[:, 1, 1]  # gtt
        inv_matrices[:, 4] = inv_matrices_full[:, 1, 2]  # gtz
        inv_matrices[:, 5] = inv_matrices_full[:, 2, 2]  # gzz
        return ContravariantBoozerMetric(
            inv_matrices[:, 0],
            inv_matrices[:, 1],
            inv_matrices[:, 2],
            inv_matrices[:, 3],
            inv_matrices[:, 4],
            inv_matrices[:, 5],
        )


class ContravariantBoozerMetric(BoozerMetric):
    r"""
    Represents the contravariant metric tensor for normalized Boozer
    coordinates :math:`(s, \theta, \zeta)` in a magnetic field. The
    contravariant metric is associated with the basis vectors
    :math:`(\partial_s, \partial_\theta, \partial_\zeta)`.

    The contravariant metric tensor :math:`g^{ij}` in Boozer coordinates is given by:

    .. math::
        g^{ij} = \begin{pmatrix}
        g^{ss} & g^{s\theta} & g^{s\zeta} \\
        g^{s\theta} & g^{\theta\theta} & g^{\theta\zeta} \\
        g^{s\zeta} & g^{\theta\zeta} & g^{\zeta\zeta}
        \end{pmatrix},

    where each component :math:`g^{ij}` is a function of the Boozer
    coordinates :math:`(s, \theta, \zeta)`.

    **Methods:**

    - `to_covariant()`: Converts the contravariant metric to its covariant
        form by inverting the metric tensor. This operation is mathematically
        equivalent to computing the inverse of the contravariant metric matrix.

    **Usage Example:**

    .. code-block:: python

        # Given a BoozerMagneticField instance named `bfield`
        contravariant_metric = bfield.get_contravariant_metric()

        # Convert to covariant metric
        covariant_metric = contravariant_metric.to_covariant()

        # Access specific metric components
        gss_component = contravariant_metric.ss

        # Compute the determinant of the metric tensor
        determinant = contravariant_metric.det

    **Returns:**

    An instance of :class:`CovariantBoozerMetric` representing the covariant
    form of the metric.

    """

    def to_covariant(self):
        """
        Converts the contravariant metric to its covariant form by inverting
        the metric tensor.

        Returns
        -------
        CovariantBoozerMetric
            The covariant form of the metric.

        Raises
        -------
        `LinAlgError`: If the matrix inversion fails, due to the matrix being singular.
        """
        # Vectorized matrix inversion for better performance
        n = len(self.ss)
        matrices = np.zeros((n, 3, 3))
        matrices[:, 0, 0] = self.ss
        matrices[:, 0, 1] = matrices[:, 1, 0] = self.st
        matrices[:, 0, 2] = matrices[:, 2, 0] = self.sz
        matrices[:, 1, 1] = self.tt
        matrices[:, 1, 2] = matrices[:, 2, 1] = self.tz
        matrices[:, 2, 2] = self.zz

        inv_matrices_full = np.linalg.inv(matrices)
        inv_matrices = np.zeros((n, 6))
        inv_matrices[:, 0] = inv_matrices_full[:, 0, 0]  # gss
        inv_matrices[:, 1] = inv_matrices_full[:, 0, 1]  # gst
        inv_matrices[:, 2] = inv_matrices_full[:, 0, 2]  # gsz
        inv_matrices[:, 3] = inv_matrices_full[:, 1, 1]  # gtt
        inv_matrices[:, 4] = inv_matrices_full[:, 1, 2]  # gtz
        inv_matrices[:, 5] = inv_matrices_full[:, 2, 2]  # gzz
        return CovariantBoozerMetric(
            inv_matrices[:, 0],
            inv_matrices[:, 1],
            inv_matrices[:, 2],
            inv_matrices[:, 3],
            inv_matrices[:, 4],
            inv_matrices[:, 5],
        )


class BoozerMagneticField(sopp.BoozerMagneticField):
    r"""
    Generic class that represents a magnetic field in Boozer coordinates
    :math:`(s,\theta,\zeta)`. Here :math:`s = \psi/\psi_0` is the normalized
    toroidal flux where :math:`2\pi\psi_0` is the toroidal flux at the
    boundary.
    The magnetic field in the covariant form is,

    .. math::
        \textbf B(s,\theta,\zeta) = G(s) \nabla \zeta + I(s) \nabla \theta +
        K(s,\theta,\zeta) \nabla \psi,

    and the contravariant form is,

    .. math::
        \textbf B(s,\theta,\zeta) = \frac{1}{\sqrt{g}} \left(\mathbf r_\zeta +
        \iota(s)\mathbf r_\theta\right),

    where,

    .. math::
        \sqrt{g}(s,\theta,\zeta) = \frac{G(s) + \iota(s)I(s)}{B^2}.

    Here :math:`\iota(s) = \psi_P'(\psi)` where :math:`2\pi\psi_P` is the
    poloidal flux and :math:`2\pi\psi` is the toroidal flux. Each subclass of
    :class:`BoozerMagneticField` implements functions to compute
    :math:`B`, :math:`G`, :math:`I`, :math:`\iota`, :math:`\psi_P`, and their
    derivatives. The cylindrical coordinates :math:`R(s,\theta,\zeta)` and
    :math:`Z(s,\theta,\zeta)` in addition to :math:`K(s,\theta,\zeta)` and
    :math:`\nu` where :math:`\zeta = \phi + \nu(s,\theta,\zeta)` and
    :math:`\phi` is the cylindrical azimuthal angle are also implemented by
    :class:`BoozerRadialInterpolant` and :class:`InterpolatedBoozerField`.
    The usage is similar to the :class:`MagneticField` class.

    The usage of :class:`BoozerMagneticField`` is as follows:

    .. code-block::

        booz = BoozerAnalytic(etabar,B0,N,G0,psi0,iota0) # An instance of
        BoozerMagneticField
        points = ... # points is a (n, 3) numpy array defining
        :math:`(s,\theta,\zeta)`
        booz.set_points(points)
        modB = bfield.modB() # returns the magnetic field strength at `points`

    Args:
        psi0: The enclosed toroidal flux divided by 2*pi
        field_type: A string identifying additional assumptions made on the
            magnetic field. Can be 'vac', 'nok', or ''.
    """

    def __init__(self, psi0, field_type="vac", nfp=1, stellsym=True):
        self.psi0 = psi0
        self.nfp = nfp
        self.stellsym = stellsym
        field_type = field_type.lower()
        assert field_type in ["vac", "nok", ""]
        self.field_type = field_type
        sopp.BoozerMagneticField.__init__(self, psi0, field_type)

    def set_points(self, points):
        """
        Set the points where the field should be evaluated in Boozer
        coordinates `(s,theta,zeta)`.

        Args:
            points: A (n, 3) array-like of `(s,theta,zeta)` points.
        """
        points = np.ascontiguousarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(
                "points must have shape (n, 3), corresponding to "
                f"(s, theta, zeta) for each of n points; got shape "
                f"{points.shape}."
            )
        return sopp.BoozerMagneticField.set_points(self, points)

    def _modB_derivs_impl(self, modB_derivs):
        self._dmodBds_impl(modB_derivs[:, 0:1])
        self._dmodBdtheta_impl(modB_derivs[:, 1:2])
        self._dmodBdzeta_impl(modB_derivs[:, 2:3])

    def _K_derivs_impl(self, K_derivs):
        self._dKdtheta_impl(K_derivs[:, 0:1])
        self._dKdzeta_impl(K_derivs[:, 1:2])

    def _nu_derivs_impl(self, nu_derivs):
        self._dnuds_impl(nu_derivs[:, 0:1])
        self._dnudtheta_impl(nu_derivs[:, 1:2])
        self._dnudzeta_impl(nu_derivs[:, 2:3])

    def _R_derivs_impl(self, R_derivs):
        self._dRds_impl(R_derivs[:, 0:1])
        self._dRdtheta_impl(R_derivs[:, 1:2])
        self._dRdzeta_impl(R_derivs[:, 2:3])

    def _Z_derivs_impl(self, Z_derivs):
        self._dZds_impl(Z_derivs[:, 0:1])
        self._dZdtheta_impl(Z_derivs[:, 1:2])
        self._dZdzeta_impl(Z_derivs[:, 2:3])

    def get_covariant_metric(self):
        r"""
        Computes and returns the covariant metric tensor for normalized
        Boozer coordinates :math:`(s, \theta, \zeta)`.

        In normalized Boozer coordinates, the metric tensor defines the local
        geometry of space with respect to the covariant basis vectors
        :math:`(\nabla s, \nabla \theta, \nabla \zeta)`.

        The metric components are computed by evaluating the derivatives of
        the cylindrical coordinates and the Boozer angle with respect to
        :math:`s`, :math:`\theta`, and :math:`\zeta`. The determinant of
        the metric tensor is computed and compared to the inverse Jacobian
        for consistency. If the discrepancy exceeds 0.1%, a warning is issued.

        Returns
        -------
        CovariantBoozerMetric
            The covariant metric tensor.

        Raises
        ------
        AssertionError
            If the metric is singular on the magnetic axis s=0.
        RuntimeWarning
            If there is a large discrepancy (>0.1%) between the computed determinant
            of the covariant metric and the inverse Jacobian.

        **Usage Example:**

        .. code-block:: python

            # Given a BoozerMagneticField instance named `bfield`
            covariant_metric = bfield.get_covariant_metric()

            # Access specific metric components
            gss_component = covariant_metric.ss
            gst_component = covariant_metric.st

            # Convert to matrix form for a single point
            matrix_form = covariant_metric[0].as_matrix
        """
        points = self.get_points_ref()
        s = points[:, 0]
        assert np.all(s > 0), (
            "Metric is singular on magnetic axis s=0, can not compute. "
            "Choose different point."
        )
        zetas = points[:, 2]
        R = self.R()[:, 0]
        dRdtheta = self.dRdtheta()[:, 0]
        dRdzeta = self.dRdzeta()[:, 0]
        dRds = self.dRds()[:, 0]
        dZdtheta = self.dZdtheta()[:, 0]
        dZdzeta = self.dZdzeta()[:, 0]
        dZds = self.dZds()[:, 0]
        nu = self.nu()[:, 0]
        dnudtheta = self.dnudtheta()[:, 0]
        dnudzeta = self.dnudzeta()[:, 0]
        dnuds = self.dnuds()[:, 0]

        phi = zetas - nu
        dphids = -dnuds
        dphidtheta = -dnudtheta
        dphidzeta = 1 - dnudzeta

        dXdtheta = dRdtheta * np.cos(phi) - R * np.sin(phi) * dphidtheta
        dYdtheta = dRdtheta * np.sin(phi) + R * np.cos(phi) * dphidtheta
        dXds = dRds * np.cos(phi) - R * np.sin(phi) * dphids
        dYds = dRds * np.sin(phi) + R * np.cos(phi) * dphids
        dXdzeta = dRdzeta * np.cos(phi) - R * np.sin(phi) * dphidzeta
        dYdzeta = dRdzeta * np.sin(phi) + R * np.cos(phi) * dphidzeta

        gss = dXds**2 + dYds**2 + dZds**2
        gstheta = dXds * dXdtheta + dYds * dYdtheta + dZds * dZdtheta
        gszeta = dXds * dXdzeta + dYds * dYdzeta + dZds * dZdzeta
        gthetatheta = dXdtheta**2 + dYdtheta**2 + dZdtheta**2
        gthetazeta = dXdtheta * dXdzeta + dYdtheta * dYdzeta + dZdtheta * dZdzeta
        gzetazeta = dXdzeta**2 + dYdzeta**2 + dZdzeta**2

        # Test that determinant of covariant Boozer metric matches inverse Jacobian
        detg = (
            gss * (gthetatheta * gzetazeta - gthetazeta**2)
            - gstheta * (gstheta * gzetazeta - gthetazeta * gszeta)
            + gszeta * (gstheta * gthetazeta - gszeta * gthetatheta)
        )
        G = self.G()[:, 0]
        I = self.I()[:, 0]
        iota = self.iota()[:, 0]
        B = self.modB()[:, 0]
        sqrtg = (G + iota * I) * self.psi0 / (B * B)
        assert np.all(sqrtg != 0), "Jacobian must be non-zero"

        relative_error = np.abs(np.sqrt(np.abs(detg)) - np.abs(sqrtg)) / np.abs(sqrtg)
        max_relative_error_percent = np.max(relative_error) * 100
        if max_relative_error_percent > 0.1:
            # Find the location of maximum error
            max_error_idx = np.argmax(relative_error)
            s_error = s[max_error_idx]
            theta_error = points[max_error_idx, 1]
            zeta_error = points[max_error_idx, 2]

            # Get metric values at error location
            metric_at_error = np.array(
                [
                    [gss[max_error_idx], gstheta[max_error_idx], gszeta[max_error_idx]],
                    [
                        gstheta[max_error_idx],
                        gthetatheta[max_error_idx],
                        gthetazeta[max_error_idx],
                    ],
                    [
                        gszeta[max_error_idx],
                        gthetazeta[max_error_idx],
                        gzetazeta[max_error_idx],
                    ],
                ]
            )

            warnings.warn(
                f"\nLarge maximum relative error ({max_relative_error_percent:.2f}%) "
                f"between metric determinant and Jacobian at:\n"
                f"  (s, theta, zeta) = ({s_error:.3f}, {theta_error:.3f}, "
                f"{zeta_error:.3f})\n"
                f"  sqrt(detg) = {np.sqrt(detg[max_error_idx]):.6e}\n"
                f"  sqrtg     = {sqrtg[max_error_idx]:.6e}\n"
                f"Metric tensor at this point:\n"
                f"  [[ {metric_at_error[0, 0]:.6e}  {metric_at_error[0, 1]:.6e}  "
                f"{metric_at_error[0, 2]:.6e} ]\n"
                f"   [ {metric_at_error[1, 0]:.6e}  {metric_at_error[1, 1]:.6e}  "
                f"{metric_at_error[1, 2]:.6e} ]\n"
                f"   [ {metric_at_error[2, 0]:.6e}  {metric_at_error[2, 1]:.6e}  "
                f"{metric_at_error[2, 2]:.6e} ]]\n"
                "exceeds 0.1% tolerance.",
                RuntimeWarning,
                stacklevel=2,
            )

        return CovariantBoozerMetric(
            gss=gss,
            gst=gstheta,
            gsz=gszeta,
            gtt=gthetatheta,
            gtz=gthetazeta,
            gzz=gzetazeta,
        )

    def get_contravariant_metric(self):
        r"""
        Computes and returns the contravariant metric tensor for normalized
        Boozer coordinates
        :math:`(s, \theta, \zeta)`.

        In normalized Boozer coordinates, the contravariant metric tensor
        defines the local geometry
        of space with respect to the contravariant basis vectors
        :math:`(\partial_s, \partial_\theta, \partial_\zeta)`.

        The contravariant metric is computed by inverting the covariant metric tensor.

        Returns
        -------
        ContravariantBoozerMetric
            The contravariant metric tensor.

        Raises
        ------
        AssertionError
            If the metric is singular on the magnetic axis s=0.
        LinAlgError
            If the covariant metric tensor cannot be inverted.

        **Usage Example:**

        .. code-block:: python

            # Given a BoozerMagneticField instance named `bfield`
            contravariant_metric = bfield.get_contravariant_metric()

            # Access specific metric components
            gss_component = contravariant_metric.ss
            gst_component = contravariant_metric.st

            # Convert to matrix form for a single point
            matrix_form = contravariant_metric[0].as_matrix
        """
        return self.get_covariant_metric().to_contravariant()


class BoozerAnalytic(BoozerMagneticField):
    r"""
    Computes a :class:`BoozerMagneticField` based on a first-order expansion in
    distance from the magnetic axis (Landreman & Sengupta, Journal of Plasma
    Physics 2018). A possibility to include QS-breaking perturbation is added,
    so the magnetic field strength is expressed as,

    .. math::
        B(s,\theta,\zeta) = B_0 \left(1 + \overline{\eta} \sqrt{2s\psi_0/\overline{B}}
        \cos(\theta - N \zeta)\right) + B_{0z}\cos{m\theta-n\zeta},

    the covariant components of equilibrium field are,

    .. math::
        G(s) = G_0 + s G_1

        I(s) = I_0 + s I_1

        K(s,\theta,\zeta) = \sqrt{2s\psi_0/\overline{B}} K_1 \sin(\theta - N \zeta),

    and the rotational transform is,

    .. math::
        \iota(s) = \iota_0 + s \iota_1.

    While formally :math:`I_0 = I_1 = G_1 = K_1 = \iota_1 = 0`, these terms have
    been included in order to test the guiding center equations at finite beta.

    Args:
        etabar: magnitude of first order correction to magnetic field strength
        B0: magnetic field strength on the axis
        N: helicity of symmetry (integer)
        G0: lowest order toroidal covariant component
        psi0: (toroidal flux)/ (2*pi) on the boundary
        iota0: lowest order rotational transform
        Bbar: normalizing magnetic field strength (defaults to 1)
        I0: lowest order poloidal covariant component (defaults to 0)
        G1: first order correction to toroidal covariant component (defaults to 0)
        I1: first order correction to poloidal covariant component (defaults to 0)
        K1: first order correction to radial covariant component (defaults to 0)
        iota1: first order correction to rotational transform (defaults to 0)
        B0z: amplitude of symmetry-breaking perturbation mode
        n: toroidal mode number for the perturbation
        m: poloidal mode bumber for the perturbation
    """

    def __init__(
        self,
        etabar,
        B0,
        N,
        G0,
        psi0,
        iota0,
        Bbar=1.0,
        I0=0.0,
        G1=0.0,
        I1=0.0,
        K1=0.0,
        iota1=0.0,
        B0z=None,
        n=None,
        m=None,
    ):
        if m is None:
            m = [2]
        if n is None:
            n = [1]
        if B0z is None:
            B0z = [0.0]
        assert len(B0z) == len(n)
        assert len(m) == len(n)
        self.etabar = etabar
        self.B0 = B0
        self.B0z = np.array(B0z)
        self.m = np.array(m, dtype="float")
        self.n = np.array(n, dtype="float")
        self.Bbar = Bbar
        self.N = N
        self.G0 = G0
        self.I0 = I0
        self.I1 = I1
        self.G1 = G1
        self.K1 = K1
        self.iota0 = iota0
        self.psi0 = psi0
        self.iota1 = iota1
        self.set_field_type()
        BoozerMagneticField.__init__(self, psi0, self.field_type)

    def set_field_type(self):
        if self.I0 == 0 and self.I1 == 0 and self.G1 == 0 and self.K1 == 0:
            self.field_type = "vac"
        elif self.K1 == 0:
            self.field_type = "nok"
        else:
            self.field_type = ""

    def set_etabar(self, etabar):
        self.etabar = etabar
        self.set_points(self.get_points_ref())  # Force cache invalidation

    def set_B0(self, B0):
        self.B0 = B0
        self.set_points(self.get_points_ref())  # Force cache invalidation

    def set_B0z(self, B0z):
        self.B0z = B0z
        self.set_points(self.get_points_ref())  # Force cache invalidation

    def set_Bbar(self, Bbar):
        self.Bbar = Bbar
        self.set_points(self.get_points_ref())  # Force cache invalidation

    def set_N(self, N):
        self.N = N
        self.set_points(self.get_points_ref())  # Force cache invalidation

    def set_G0(self, G0):
        self.G0 = G0
        self.set_points(self.get_points_ref())  # Force cache invalidation

    def set_I0(self, I0):
        self.I0 = I0
        self.set_field_type()
        self.set_points(self.get_points_ref())  # Force cache invalidation

    def set_G1(self, G1):
        self.G1 = G1
        self.set_field_type()
        self.set_points(self.get_points_ref())  # Force cache invalidation

    def set_I1(self, I1):
        self.I1 = I1
        self.set_field_type()
        self.set_points(self.get_points_ref())  # Force cache invalidation

    def set_K1(self, K1):
        self.K1 = K1
        self.set_field_type()
        self.set_points(self.get_points_ref())  # Force cache invalidation

    def set_iota0(self, iota0):
        self.iota0 = iota0
        self.set_points(self.get_points_ref())  # Force cache invalidation

    def set_iota1(self, iota1):
        self.iota1 = iota1
        self.set_points(self.get_points_ref())  # Force cache invalidation

    def set_psi0(self, psi0):
        self.psi0 = psi0
        self.set_points(self.get_points_ref())  # Force cache invalidation

    def _psip_impl(self, psip):
        points = self.get_points_ref()
        s = points[:, 0]
        psip[:, 0] = self.psi0 * (s * self.iota0 + s**2 * self.iota1 / 2)

    def _iota_impl(self, iota):
        points = self.get_points_ref()
        s = points[:, 0]
        iota[:, 0] = self.iota0 + self.iota1 * s

    def _diotads_impl(self, diotads):
        diotads[:, 0] = self.iota1

    def _G_impl(self, G):
        points = self.get_points_ref()
        s = points[:, 0]
        G[:, 0] = self.G0 + s * self.G1

    def _dGds_impl(self, dGds):
        dGds[:, 0] = self.G1

    def _I_impl(self, I):
        points = self.get_points_ref()
        s = points[:, 0]
        I[:, 0] = self.I0 + s * self.I1

    def _dIds_impl(self, dIds):
        dIds[:, 0] = self.I1

    def _modB_impl(self, modB):
        points = self.get_points_ref()
        s = points[:, 0]
        thetas = points[:, 1]
        zetas = points[:, 2]
        psi = s * self.psi0
        r = np.sqrt(np.abs(2 * psi / self.Bbar))
        modB[:, 0] = self.B0 * (
            1 + self.etabar * r * np.cos(thetas - self.N * zetas)
        ) + np.sum(
            self.B0z[:, None]
            * np.cos(
                self.m[:, None] * thetas[None, :]
                - self.n[:, None] * self.N * zetas[None, :]
            )
        )

    def _dmodBds_impl(self, dmodBds):
        points = self.get_points_ref()
        s = points[:, 0]
        thetas = points[:, 1]
        zetas = points[:, 2]
        psi = s * self.psi0
        # drds = np.zeros_like(s)
        r = np.sqrt(np.abs(2 * psi / self.Bbar))
        # drds[s!=0] = 0.5*r[s!=0]*self.psi0/psi[s!=0]
        if self.etabar != 0:
            drds = 0.5 * r * self.psi0 / psi
            dmodBds[:, 0] = (
                self.B0 * self.etabar * drds * np.cos(thetas - self.N * zetas)
            )
        else:
            dmodBds[:, 0] = 0

    def _dmodBdtheta_impl(self, dmodBdtheta):
        points = self.get_points_ref()
        s = points[:, 0]
        thetas = points[:, 1]
        zetas = points[:, 2]
        psi = s * self.psi0
        r = np.sqrt(np.abs(2 * psi / self.Bbar))
        dmodBdtheta[:, 0] = -self.B0 * self.etabar * r * np.sin(
            thetas - self.N * zetas
        ) - np.sum(
            self.B0z[:, None]
            * self.m[:, None]
            * np.sin(
                self.m[:, None] * thetas[None, :]
                - self.n[:, None] * self.N * zetas[None, :]
            )
        )

    def _dmodBdzeta_impl(self, dmodBdzeta):
        points = self.get_points_ref()
        s = points[:, 0]
        thetas = points[:, 1]
        zetas = points[:, 2]
        psi = s * self.psi0
        r = np.sqrt(np.abs(2 * psi / self.Bbar))
        dmodBdzeta[:, 0] = self.N * self.B0 * self.etabar * r * np.sin(
            thetas - self.N * zetas
        ) + np.sum(
            self.B0z[:, None]
            * self.n[:, None]
            * self.N
            * np.sin(
                self.m[:, None] * thetas[None, :]
                - self.n[:, None] * self.N * zetas[None, :]
            )
        )

    def _K_impl(self, K):
        points = self.get_points_ref()
        s = points[:, 0]
        thetas = points[:, 1]
        zetas = points[:, 2]
        psi = s * self.psi0
        r = np.sqrt(np.abs(2 * psi / self.Bbar))
        K[:, 0] = self.K1 * r * np.sin(thetas - self.N * zetas)

    def _dKdtheta_impl(self, dKdtheta):
        points = self.get_points_ref()
        s = points[:, 0]
        thetas = points[:, 1]
        zetas = points[:, 2]
        psi = s * self.psi0
        r = np.sqrt(np.abs(2 * psi / self.Bbar))
        dKdtheta[:, 0] = self.K1 * r * np.cos(thetas - self.N * zetas)

    def _dKdzeta_impl(self, dKdzeta):
        points = self.get_points_ref()
        s = points[:, 0]
        thetas = points[:, 1]
        zetas = points[:, 2]
        psi = s * self.psi0
        r = np.sqrt(np.abs(2 * psi / self.Bbar))
        dKdzeta[:, 0] = -self.N * self.K1 * r * np.cos(thetas - self.N * zetas)


class BoozerSplineField(BoozerMagneticField):
    def __init__(
        self,
        equil,
        mpol=32,
        ntor=32,
        ntheta=48,
        nzeta=48,
        helicity_M=None,
        helicity_N=None,
        enforce_vacuum=False,
        no_K=True,
        write_boozmn=True,
        boozmn_name="boozmn.nc",
        field_type=None,
        comm=None,
        spline_deriv=True,
    ):
        r"""
        Initialize a BoozerSplineField object. The field is interpolated on the
        VMEC half grid, with the number of angular grid points specified by ntheta
        and nzeta, using :mod:`ndsplines` for 3D fields and
        :class:`scipy.interpolate.CubicSpline` for 1D flux functions.

        Args:
            equil: instance of :class:`Booz_xform` or string containing the
                filename of a boozmn_*.nc file (produced with booz_xform) or
                wout_*.nc file (produced with VMEC). If a :class:`Booz_xform`
                instance or boozmn_*.nc file is passed, the `compute_surfs` needs
                to include all of the grid points in the half-radius grid of the
                corresponding Vmec equilibrium. Otherwise, a ValueError is raised.
            mpol: (int) number of poloidal mode numbers for BOOZXFORM (defaults to
                32). Only used if a wout_*.nc file is passed.
            ntor: (int) number of toroidal mode numbers for BOOZXFORM (defaults to
                32). Only used if a wout_*.nc file is passed.
            ntheta: (int) number of poloidal grid points.
            nzeta: (int) number of toroidal grid points.
            helicity_M : Poloidal helicity coefficient for enforcing field
                quasi-symmetry If specified, then the non-symmetric Fourier
                harmonics of :math:`B` and :math:`K` are filtered out, so the
                field is a function of `chi = helicity_M*theta - helicity_N*zeta`.
                If helicity is unspecified, all harmonics are kept.
                (defaults to ``None``)
            helicity_N : Toroidal helicity coefficient for enforcing field
                quasi-symmetry If specified, then the non-symmetric Fourier
                harmonics of :math:`B` and :math:`K` are filtered out, so the
                field is a function of `chi = helicity_M*theta - helicity_N*zeta`.
                If helicity is unspecified, all harmonics are kept.
            enforce_vacuum: If True, a vacuum field is assumed, :math:`G` is
                set to its mean value, :math:`I = 0`, and :math:`K = 0`.
            no_K: (bool) If ``True``, the Boozer :math:`K` will not be computed or
                interpolated.
            write_boozmn: (bool) If ``True``, save the booz_xform transformation in
                a filename specified by ``boozmn_name``. (defaults to ``True``)
            boozmn_name: (string) Filename to save booz_xform transformation if
                ``write_boozmn`` is ``True``.
            field_type: A string identifying additional assumptions made on the
                magnetic field. Can be
                ``'vac'``, ``'nok'``, or ``''``.  By default, this is determined
                from the options ``enforce_vacuum``
                and ``no_K``.
            comm: A MPI communicator to parallelize over, from which
                the worker groups will be used for spline calculations. If ``comm`` is
                ``None``, each MPI process will compute splines independently.
            spline_deriv: (bool) If ``True``, field derivatives will be evaluated by
                differentiating the spline field. Otherwise, Fourier derivatives are
                evaluated for the angles, and centered differences are used for the
                radial derivative. While ``False`` is more accurate, ``True`` is faster
                due to a reduction in the number of spline evaluations.

        Returns:
            :class:`BoozerSplineField` object.
        """

        self.comm = comm
        self.verbose = self.comm is None or self.comm.rank == 0
        self.spline_deriv = spline_deriv

        if field_type is not None:
            field_type = field_type.lower()
            assert field_type in ["vac", "nok", ""]
            if self.verbose:
                if enforce_vacuum != (field_type == "vac"):
                    warnings.warn(
                        f"Prescribed field_type is inconsistent with enforce_vacuum. "
                        f"Proceeding with field_type={field_type}.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                if no_K != (field_type == "nok"):
                    warnings.warn(
                        f"Prescribed field_type is inconsistent with no_K. "
                        f"Proceeding with field_type={field_type}.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
            self.field_type = field_type
        else:
            if enforce_vacuum:
                self.field_type = "vac"
            elif no_K:
                self.field_type = "nok"
            else:
                self.field_type = ""

        if isinstance(equil, str):
            if self.verbose:
                basename = os.path.basename(equil)
                if basename[:4] == "wout":
                    booz = Booz_xform()
                    booz.read_wout(equil, True)
                    booz.verbose = False
                    booz.mboz = mpol
                    booz.nboz = ntor
                    booz.run()
                    if write_boozmn:
                        booz.write_boozmn(boozmn_name)
                    self.bx = booz
                elif basename[:4] == "booz":
                    booz = Booz_xform()
                    booz.verbose = False
                    booz.read_boozmn(equil)
                    self.bx = booz
                    # Check if grid does not have correct size
                    if self.bx.ns_in != len(self.bx.s_b):
                        raise ValueError("booz filename has incorrect s grid!")
                    # Check if grid does not match Vmec half grid
                    s_in_full = np.linspace(0, 1, self.bx.ns_in + 1)
                    s_in = 0.5 * (s_in_full[1::] + s_in_full[0:-1])
                    if not np.allclose(s_in, self.bx.s_b):
                        raise ValueError("booz filename has incorrect s grid!")
                else:
                    raise ValueError("Invalid filename")
        elif isinstance(equil, Booz_xform):
            if self.verbose:
                self.bx = equil
        else:
            raise ValueError("Incorrect equil type passed to BoozerRadialInterpolant.")

        self.helicity_M = helicity_M
        self.helicity_N = helicity_N
        self.enforce_qs = False
        self.enforce_vacuum = enforce_vacuum
        self.no_K = no_K
        if (helicity_M is not None) and (helicity_N is not None):
            if helicity_M % 1 != 0:
                raise ValueError(
                    "helicity_M must be an integer for field to be 2π-periodic "
                    "in Boozer poloidal angle."
                )

            if helicity_N % 1 != 0:
                raise ValueError(
                    "helicity_N must be an integer for field to be 2π-periodic "
                    "in Boozer toroidal angle."
                )

            self.helicity_M = helicity_M
            self.helicity_N = helicity_N
            self.enforce_qs = True
        elif (helicity_M is not None) or (helicity_N is not None):
            raise ValueError(
                "Both helicity_M and helicity_N must be specified when "
                "enforcing field symmetry."
            )

        # Only proc0 reads from file
        if self.verbose:
            self.asym = self.bx.asym  # Bool for stellarator asymmetry
            self.psi0 = -self.bx.phi[-1] / (
                2 * np.pi
            )  # Sign flip to account for VMEC convention.
            # See https://terpconnect.umd.edu/~mattland/assets/notes/vmec_signs.pdf
            # for phiedge definition
            self.nfp = self.bx.nfp
            self.mpol = self.bx.mboz
            self.ntor = self.bx.nboz
            self.s_half_ext = np.zeros(self.bx.ns_b + 2)
            self.s_half_ext[1:-1] = self.bx.s_b
            self.s_half_ext[-1] = 1
            self.s_half = self.bx.s_b
            self.ns_half = self.bx.ns_b

            self.xm_b = self.bx.xm_b
            self.xn_b = self.bx.xn_b
            self.ns_b = self.bx.ns_b
            self.s_b = self.bx.s_b
            self.bmnc_b = self.bx.bmnc_b
            self.bmns_b = self.bx.bmns_b
            # Define quantities on full grid

            self.ds = self.bx.s_b[1] - self.bx.s_b[0]

            self.s_full = np.linspace(0, 1, self.bx.ns_b + 1)
            self.s_full_reduced = self.s_full[1:-1]
            self.ds = self.s_full[1] - self.s_full[0]

            # Defined on full grid
            self.psip_grid = self.bx.chi / (2 * np.pi)

            # Evaluate iota on s_half_ext grid
            self.iota_grid = np.zeros(self.bx.ns_b)
            self.iota_grid = self.bx.iota

            sign_psip = np.sign(
                ((self.psip_grid[1] - self.psip_grid[0]) / self.psi0)
                / np.sign(self.iota_grid[1])
            )
            self.psip_grid *= sign_psip

            # dGds and dIds are evaluated on the full grid
            if not self.spline_deriv:
                self.dGds_grid = np.zeros((self.ns_b - 1,))
                self.dIds_grid = np.zeros((self.ns_b - 1,))

            # G and I are evaluated on the s_half grid
            if self.field_type == "vac":
                self.G_grid = np.ones(self.bx.ns_b) * np.mean(self.bx.Boozer_G_all)
                self.I_grid = np.zeros(self.bx.ns_b)
            else:
                self.G_grid = np.zeros(self.bx.ns_b)
                self.G_grid = self.bx.Boozer_G_all
                self.I_grid = self.bx.Boozer_I_all
                if not self.spline_deriv:
                    self.dGds_grid = (
                        self.bx.Boozer_G_all[1::] - self.bx.Boozer_G_all[0:-1]
                    ) / self.ds
                    self.dIds_grid = (
                        self.bx.Boozer_I_all[1::] - self.bx.Boozer_I_all[0:-1]
                    ) / self.ds

            bmnc_filtered = self.bx.bmnc_b.copy()
            if self.enforce_qs:
                bmnc_filtered[
                    self.helicity_M * self.xn_b != self.helicity_N * self.xm_b, :
                ] = 0
            if self.asym:
                bmns_filtered = self.bx.bmns_b.copy()
                if self.enforce_qs:
                    bmns_filtered[
                        self.helicity_M * self.xn_b != self.helicity_N * self.xm_b, :
                    ] = 0

            # Evaluate quantities on the s_half grid
            self.bmnc = bmnc_filtered
            self.numns = self.bx.numns_b
            self.rmnc = self.bx.rmnc_b
            self.zmns = self.bx.zmns_b

            # Evaluate derivatives on the full grid
            if not self.spline_deriv:
                self.diotads_grid = (self.bx.iota[1::] - self.bx.iota[0:-1]) / self.ds
                self.dbmncds = (
                    self.bx.bmnc_b[:, 1::] - self.bx.bmnc_b[:, 0:-1]
                ) / self.ds
                self.dnumnsds = (
                    self.bx.numns_b[:, 1::] - self.bx.numns_b[:, 0:-1]
                ) / self.ds
                self.drmncds = (
                    self.bx.rmnc_b[:, 1::] - self.bx.rmnc_b[:, 0:-1]
                ) / self.ds
                self.dzmnsds = (
                    self.bx.zmns_b[:, 1::] - self.bx.zmns_b[:, 0:-1]
                ) / self.ds

            if self.asym:
                self.bmns = bmns_filtered
                self.numnc = self.bx.numnc_b
                self.rmns = self.bx.rmns_b
                self.zmnc = self.bx.zmnc_b

                if not self.spline_deriv:
                    self.dbmnsds = (
                        self.bx.bmns_b[:, 1::] - self.bx.bmns_b[:, 0:-1]
                    ) / self.ds
                    self.dnumncds = (
                        self.bx.numnc_b[:, 1::] - self.bx.numnc_b[:, 0:-1]
                    ) / self.ds
                    self.drmnsds = (
                        self.bx.rmns_b[:, 1::] - self.bx.rmns_b[:, 0:-1]
                    ) / self.ds
                    self.dzmncds = (
                        self.bx.zmnc_b[:, 1::] - self.bx.zmnc_b[:, 0:-1]
                    ) / self.ds
        else:
            self.xm_b = None
            self.xn_b = None
            self.psi0 = None
            self.nfp = None
            self.mpol = None
            self.ntor = None
            self.asym = None
            self.s_half_ext = None
            self.s_half = None
            self.s_full_reduced = None
            self.s_full = None
            self.ns_half = None
            self.ns_b = None
            self.bmnc = None
            self.rmnc = None
            self.zmns = None
            self.numns = None
            self.G_grid = None
            self.I_grid = None
            self.iota_grid = None
            self.psip_grid = None

            if not self.spline_deriv:
                self.dbmncds = None
                self.dnumnsds = None
                self.drmncds = None
                self.dzmnsds = None
                self.diotads_grid = None
                self.dGds_grid = None
                self.dIds_grid = None
            # Initialize asymmetry-related attributes to None for non-proc0 processes
            # They will be set to actual values after broadcasting from proc0
            self.bmns = None
            self.rmns = None
            self.zmnc = None
            self.numnc = None
            if not self.spline_deriv:
                self.dbmnsds = None
                self.dnumnsds = None
                self.drmnsds = None
                self.dzmncds = None
        if self.comm is not None:
            self.xm_b = self.comm.bcast(self.xm_b, root=0)
            self.xn_b = self.comm.bcast(self.xn_b, root=0)
            self.psi0 = self.comm.bcast(self.psi0, root=0)
            self.nfp = self.comm.bcast(self.nfp, root=0)
            self.mpol = self.comm.bcast(self.mpol, root=0)
            self.ntor = self.comm.bcast(self.ntor, root=0)
            self.asym = self.comm.bcast(self.asym, root=0)
            self.s_half_ext = self.comm.bcast(self.s_half_ext, root=0)
            self.s_half = self.comm.bcast(self.s_half, root=0)
            self.s_full = self.comm.bcast(self.s_full, root=0)
            self.s_full_reduced = self.comm.bcast(self.s_full_reduced, root=0)
            self.ns_half = self.comm.bcast(self.ns_half, root=0)
            self.ns_b = self.comm.bcast(self.ns_b, root=0)
            self.bmnc = self.comm.bcast(self.bmnc, root=0)
            self.rmnc = self.comm.bcast(self.rmnc, root=0)
            self.zmns = self.comm.bcast(self.zmns, root=0)
            self.numns = self.comm.bcast(self.numns, root=0)
            self.G_grid = self.comm.bcast(self.G_grid, root=0)
            self.I_grid = self.comm.bcast(self.I_grid, root=0)
            self.iota_grid = self.comm.bcast(self.iota_grid, root=0)
            self.psip_grid = self.comm.bcast(self.psip_grid, root=0)
            if not self.spline_deriv:
                self.dGds_grid = self.comm.bcast(self.dGds_grid, root=0)
                self.dIds_grid = self.comm.bcast(self.dIds_grid, root=0)
                self.diotads_grid = self.comm.bcast(self.diotads_grid, root=0)
                self.dbmncds = self.comm.bcast(self.dbmncds, root=0)
                self.dnumnsds = self.comm.bcast(self.dnumnsds, root=0)
                self.drmncds = self.comm.bcast(self.drmncds, root=0)
                self.dzmnsds = self.comm.bcast(self.dzmnsds, root=0)
            if self.asym:
                self.bmns = self.comm.bcast(self.bmns, root=0)
                self.rmns = self.comm.bcast(self.rmns, root=0)
                self.zmnc = self.comm.bcast(self.zmnc, root=0)
                self.numnc = self.comm.bcast(self.numnc, root=0)
                if not self.spline_deriv:
                    self.dbmnsds = self.comm.bcast(self.dbmnsds, root=0)
                    self.dnumncds = self.comm.bcast(self.dnumncds, root=0)
                    self.drmnsds = self.comm.bcast(self.drmnsds, root=0)
                    self.dzmncds = self.comm.bcast(self.dzmncds, root=0)
        self.ntheta = ntheta
        self.nzeta = nzeta
        if self.asym:
            self.theta_grid = np.linspace(0, 2 * np.pi, ntheta, endpoint=False)
        else:
            self.theta_grid = np.linspace(0, np.pi, ntheta, endpoint=False)
        self.zeta_grid = np.linspace(0, 2 * np.pi / self.nfp, nzeta, endpoint=False)

        # Augment grid in either direction to ensure periodic boundary conditions
        # np.insert returns a new array, so we must assign the result
        dtheta = self.theta_grid[1] - self.theta_grid[0]
        dzeta = self.zeta_grid[1] - self.zeta_grid[0]

        # For theta: grid is [0, π), so add points before 0 and after π
        # For zeta: grid is [0, 2π/nfp), so add points before 0 and after 2π/nfp
        theta_max = np.max(self.theta_grid)
        zeta_max = np.max(self.zeta_grid)

        # Insert boundary points to enforce periodicity
        self.theta_grid = np.insert(
            self.theta_grid, len(self.theta_grid), theta_max + dtheta
        )
        self.theta_grid = np.insert(
            self.theta_grid, len(self.theta_grid), theta_max + 2 * dtheta
        )
        self.theta_grid = np.insert(
            self.theta_grid, len(self.theta_grid), theta_max + 3 * dtheta
        )
        self.theta_grid = np.insert(self.theta_grid, 0, -dtheta)
        self.theta_grid = np.insert(self.theta_grid, 0, -2 * dtheta)
        self.theta_grid = np.insert(self.theta_grid, 0, -3 * dtheta)
        self.zeta_grid = np.insert(
            self.zeta_grid, len(self.zeta_grid), zeta_max + dzeta
        )
        self.zeta_grid = np.insert(
            self.zeta_grid, len(self.zeta_grid), zeta_max + 2 * dzeta
        )
        self.zeta_grid = np.insert(
            self.zeta_grid, len(self.zeta_grid), zeta_max + 3 * dzeta
        )
        self.zeta_grid = np.insert(self.zeta_grid, 0, -dzeta)
        self.zeta_grid = np.insert(self.zeta_grid, 0, -2 * dzeta)
        self.zeta_grid = np.insert(self.zeta_grid, 0, -3 * dzeta)

        self.ntheta_ext = len(self.theta_grid)
        self.nzeta_ext = len(self.zeta_grid)

        self.theta_grid_2d, self.zeta_grid_2d = np.meshgrid(
            self.theta_grid, self.zeta_grid, indexing="ij"
        )
        self.s_grid_3d, self.theta_grid_3d, self.zeta_grid_3d = np.meshgrid(
            self.s_full, self.theta_grid, self.zeta_grid, indexing="ij"
        )

        if not self.asym:
            self.modB_spline = self.compute_spline(self.bmnc, "even", "half", "modB")
            if not self.spline_deriv:
                self.dmodBds_spline = self.compute_spline(
                    self.dbmncds, "even", "full_reduced", "dmodBds"
                )
                self.dmodBdtheta_spline = self.compute_spline(
                    -self.bmnc * self.xm_b[:, None], "odd", "half", "dmodBdtheta"
                )
                self.dmodBdzeta_spline = self.compute_spline(
                    +self.bmnc * self.xn_b[:, None], "odd", "half", "dmodBdzeta"
                )
            self.R_spline = self.compute_spline(self.rmnc, "even", "half", "R")
            if not self.spline_deriv:
                self.dRds_spline = self.compute_spline(
                    self.drmncds, "even", "full_reduced", "dRds"
                )
                self.dRdtheta_spline = self.compute_spline(
                    -self.rmnc * self.xm_b[:, None], "odd", "half", "dRdtheta"
                )
                self.dRdzeta_spline = self.compute_spline(
                    +self.rmnc * self.xn_b[:, None], "odd", "half", "dRdzeta"
                )
            self.Z_spline = self.compute_spline(self.zmns, "odd", "half", "Z")
            if not self.spline_deriv:
                self.dZds_spline = self.compute_spline(
                    self.dzmnsds, "odd", "full_reduced", "dZds"
                )
                self.dZdtheta_spline = self.compute_spline(
                    self.zmns * self.xm_b[:, None], "even", "half", "dZdtheta"
                )
                self.dZdzeta_spline = self.compute_spline(
                    -self.zmns * self.xn_b[:, None], "even", "half", "dZdzeta"
                )
            self.nu_spline = self.compute_spline(self.numns, "odd", "half", "nu")
            if not self.spline_deriv:
                self.dnuds_spline = self.compute_spline(
                    self.dnumnsds, "odd", "full_reduced", "dnuds"
                )
                self.dnudtheta_spline = self.compute_spline(
                    self.numns * self.xm_b[:, None], "even", "half", "dnudtheta"
                )
                self.dnudzeta_spline = self.compute_spline(
                    -self.numns * self.xn_b[:, None], "even", "half", "dnudzeta"
                )
        else:
            self.modB_spline = self.compute_spline(
                (self.bmnc, self.bmns), "even_odd", "half", "modB"
            )
            if not self.spline_deriv:
                self.dmodBds_spline = self.compute_spline(
                    (self.dbmncds, self.dbmnsds), "even_odd", "full_reduced", "dmodBds"
                )
                self.dmodBdtheta_spline = self.compute_spline(
                    (self.bmns * self.xm_b[:, None], -self.bmnc * self.xm_b[:, None]),
                    "even_odd",
                    "half",
                    "dmodBdtheta",
                )
                self.dmodBdzeta_spline = self.compute_spline(
                    (-self.bmns * self.xn_b[:, None], self.bmnc * self.xn_b[:, None]),
                    "even_odd",
                    "half",
                    "dmodBdzeta",
                )
            self.R_spline = self.compute_spline(
                (self.rmnc, self.rmns), "even_odd", "half", "R"
            )
            self.Z_spline = self.compute_spline(
                (self.zmnc, self.zmns), "even_odd", "half", "Z"
            )
            if not self.spline_deriv:
                self.dRds_spline = self.compute_spline(
                    (self.drmncds, self.drmnsds), "even_odd", "full_reduced", "dRds"
                )
                self.dRdtheta_spline = self.compute_spline(
                    (self.rmns * self.xm_b[:, None], -self.rmnc * self.xm_b[:, None]),
                    "even_odd",
                    "half",
                    "dRdtheta",
                )
                self.dRdzeta_spline = self.compute_spline(
                    (-self.rmns * self.xn_b[:, None], +self.rmnc * self.xn_b[:, None]),
                    "even_odd",
                    "half",
                    "dRdzeta",
                )
                self.dZds_spline = self.compute_spline(
                    (self.dzmncds, self.dzmnsds), "even_odd", "full_reduced", "dZds"
                )
                self.dZdtheta_spline = self.compute_spline(
                    (+self.zmns * self.xm_b[:, None], -self.zmnc * self.xm_b[:, None]),
                    "even_odd",
                    "half",
                    "dZdtheta",
                )
                self.dZdzeta_spline = self.compute_spline(
                    (-self.zmns * self.xn_b[:, None], +self.zmnc * self.xn_b[:, None]),
                    "even_odd",
                    "half",
                    "dZdzeta",
                )
            self.nu_spline = self.compute_spline(
                (self.numnc, self.numns), "even_odd", "half", "nu"
            )
            if not self.spline_deriv:
                self.dnuds_spline = self.compute_spline(
                    (self.dnumncds, self.dnumnsds), "even_odd", "full_reduced", "dnuds"
                )
                self.dnudtheta_spline = self.compute_spline(
                    (
                        +self.numns * self.xm_b[:, None],
                        -self.numnc * self.xm_b[:, None],
                    ),
                    "even_odd",
                    "half",
                    "dnudtheta",
                )
                self.dnudzeta_spline = self.compute_spline(
                    (
                        -self.numns * self.xn_b[:, None],
                        +self.numnc * self.xn_b[:, None],
                    ),
                    "even_odd",
                    "half",
                    "dnudzeta",
                )

        self.G_spline = self.compute_spline(self.G_grid, "flux", "half", "G")
        if not self.spline_deriv:
            self.dGds_spline = self.compute_spline(
                self.dGds_grid, "flux", "full_reduced", "dGds"
            )
            self.dIds_spline = self.compute_spline(
                self.dIds_grid, "flux", "full_reduced", "dIds"
            )
            self.diotads_spline = self.compute_spline(
                self.diotads_grid, "flux", "full_reduced", "diotads"
            )
        self.I_spline = self.compute_spline(self.I_grid, "flux", "half", "I")
        self.iota_spline = self.compute_spline(self.iota_grid, "flux", "half", "iota")
        self.psip_spline = self.compute_spline(self.psip_grid, "flux", "full", "psip")

        BoozerMagneticField.__init__(
            self, self.psi0, self.field_type, self.nfp, self.asym == 0
        )

        if self.field_type == "":
            self.K_grid = np.zeros(
                (len(self.s_full), len(self.theta_grid), len(self.zeta_grid))
            )
            self.compute_K(self.K_grid)
            self.K_spline = self._build_3d_interpolator(self.s_full, self.K_grid)

    def compute_spline(self, harmonics, even_odd, half_full="half", field_name=None):
        r"""
        Compute the spline field for the given harmonics.
        Args:
            harmonics: array-like (n_harmonics, n_s) or list of arrays with this
                shape. If stellaratory symmetry is present, only one array is needed,
                while both even and odd harmonics are present if no symmetry is
                present.
            even_odd: string, "even", "odd", "even_odd", or "flux". If stellaratory
                symmetry is present, the harmonics are either even or odd (sin or cos).
                Otherwise, both even and odd harmonics are present. Use "flux" for
                flux functions that depend only on s.
            half_full: string, "half", "full", or "full_reduced". Whether to use
                the half grid, full grid, or full reduced grid (excluding boundary
                points).
            field_name: string, optional. If provided, the grid data will be saved
                for later use in direct interpolation.
        Returns:
            spline: :class:`ndsplines.NDSpline` object for 3D fields or
                :class:`scipy.interpolate.CubicSpline` object for flux functions.
        """
        if half_full == "half":
            s_grid = self.s_half
        elif half_full == "full":
            s_grid = self.s_full
        elif half_full == "full_reduced":
            s_grid = self.s_full_reduced
        else:
            raise ValueError("half_full must be 'half', 'full', or 'full_reduced'")

        if even_odd == "flux":
            field_grid = harmonics
        else:
            field_grid = np.zeros((len(s_grid), self.ntheta_ext, self.nzeta_ext))
            if even_odd == "even_odd":
                harmonics_even = harmonics[0]
                self.compute_full_grid_IFT(
                    harmonics_even, "even", field_grid, half_full
                )
                harmonics_odd = harmonics[1]
                self.compute_full_grid_IFT(harmonics_odd, "odd", field_grid, half_full)
            else:
                self.compute_full_grid_IFT(harmonics, even_odd, field_grid, half_full)

        if even_odd == "flux":
            spline = CubicSpline(s_grid, harmonics)
        else:
            spline = self._build_3d_interpolator(s_grid, field_grid)

        return spline

    def _build_3d_interpolator(self, s_grid, field_grid):
        import ndsplines as nds

        return nds.make_interp_spline(
            [s_grid, self.theta_grid, self.zeta_grid], field_grid, degrees=3
        )

    def _eval_3d_spline(self, spline, points):
        return spline(points)

    def _eval_3d_spline_deriv(self, spline, points, nus):
        nus_arr = np.asarray(nus, dtype=int)
        return spline(points, nus=nus_arr)

    def compute_full_grid_IFT(self, harmonics, even_odd, field_grid, half_full="half"):
        r"""
        Compute the full grid inverse Fourier transform for the given harmonics.
        Args:
            harmonics: array-like (n_harmonics, n_s) field harmonics to transform.
            even_odd: string, "even" or "odd". Whether to perform inverse
                cos or sin transform.
            field_grid: array-like (n_s, n_theta, n_zeta) output array to store the
                transformed field values.
            half_full: string, "half", "full", or "full_reduced". Whether to use the
                half grid, full grid, or full reduced grid (excluding boundary points).
        Returns:
            field_grid: array-like (n_s, n_theta, n_zeta)
                The field grid (modified in place).
        """
        if even_odd == "even":
            inverse_fourier = sopp.inverse_fourier_transform_even
        elif even_odd == "odd":
            inverse_fourier = sopp.inverse_fourier_transform_odd
        else:
            raise ValueError("even_odd must be 'even' or 'odd'")

        # Parallelize over surfaces
        if half_full == "half":
            first, last = parallel_loop_bounds(self.comm, self.ns_half)
        elif half_full == "full":
            first, last = parallel_loop_bounds(self.comm, self.ns_half + 1)
        elif half_full == "full_reduced":
            first, last = parallel_loop_bounds(self.comm, self.ns_half - 1)
        else:
            raise ValueError("half_full must be 'half' or 'full' or 'full_reduced'")

        # Pre-flatten theta and zeta grids for efficiency
        # meshgrid with indexing="ij" creates (ntheta, nzeta) arrays
        # flatten() preserves C-order (row-major), so theta varies fastest
        theta_flat = self.theta_grid_2d.flatten(order="C")
        zeta_flat = self.zeta_grid_2d.flatten(order="C")
        ntheta = self.ntheta_ext
        nzeta = self.nzeta_ext
        npoints = ntheta * nzeta

        # Pre-allocate aligned buffer and prepare aligned arrays for C++ function
        # Align arrays once outside the loop for better performance
        buffer = allocate_aligned_and_padded_array((npoints,))
        xm_aligned = align_and_pad(self.xm_b)
        xn_aligned = align_and_pad(self.xn_b)
        theta_aligned = align_and_pad(theta_flat)
        zeta_aligned = align_and_pad(zeta_flat)
        # Pre-allocate aligned buffer for harmonics (reused for each surface)
        harmonics_buffer = allocate_aligned_and_padded_array((len(self.xm_b),))

        # Each process computes its assigned surfaces
        for isurf in range(first, last):
            # Copy harmonics for this surface to aligned buffer
            harmonics_buffer[: len(self.xm_b)] = harmonics[:, isurf]
            # Clear output buffer for this surface
            buffer.fill(0)
            # Call optimized C++ inverse Fourier transform
            inverse_fourier(
                buffer,
                harmonics_buffer,
                xm_aligned,
                xn_aligned,
                theta_aligned,
                zeta_aligned,
                self.ntor,
                self.nfp,
                True,
            )
            # Extract only the actual data (excluding padding) and reshape
            # buffer has padding, so we must slice to npoints before reshaping
            # Reshape to (ntheta, nzeta) matching the original grid structure
            field_grid[isurf, :, :] += buffer[:npoints].reshape(
                (ntheta, nzeta), order="C"
            )
        # Accumulate all field_grid data to all processes
        if self.comm is not None:
            # Ensure field_grid is contiguous for MPI operations
            if not field_grid.flags["C_CONTIGUOUS"]:
                field_grid_contig = np.ascontiguousarray(field_grid)
            else:
                field_grid_contig = field_grid
            # Use Allreduce to sum contributions from all processes
            recv_buffer = np.empty_like(field_grid_contig)
            self.comm.Allreduce(
                [field_grid_contig, MPI.DOUBLE], recv_buffer, op=MPI.SUM
            )
            field_grid[:, :, :] = recv_buffer

    def _modB_impl(self, modB):
        points = self.get_points_ref()
        points_sym, flip = self.map_points_symmetries(points)
        modB[:, 0] = self._eval_3d_spline(self.modB_spline, points_sym)

    def _dmodBds_impl(self, dmodBds):
        points = self.get_points_ref()
        points_sym, flip = self.map_points_symmetries(points)
        if not self.spline_deriv:
            dmodBds[:, 0] = self._eval_3d_spline(self.dmodBds_spline, points_sym)
        else:
            dmodBds[:, 0] = self._eval_3d_spline_deriv(
                self.modB_spline, points_sym, (1, 0, 0)
            )

    def _dmodBdtheta_impl(self, dmodBdtheta):
        points = self.get_points_ref()
        points_sym, flip = self.map_points_symmetries(points)
        if not self.spline_deriv:
            dmodBdtheta[:, 0] = self._eval_3d_spline(
                self.dmodBdtheta_spline, points_sym
            )
        else:
            dmodBdtheta[:, 0] = self._eval_3d_spline_deriv(
                self.modB_spline, points_sym, (0, 1, 0)
            )
        dmodBdtheta[flip, 0] = -dmodBdtheta[flip, 0]

    def _dmodBdzeta_impl(self, dmodBdzeta):
        points = self.get_points_ref()
        points_sym, flip = self.map_points_symmetries(points)
        if not self.spline_deriv:
            dmodBdzeta[:, 0] = self._eval_3d_spline(self.dmodBdzeta_spline, points_sym)
        else:
            dmodBdzeta[:, 0] = self._eval_3d_spline_deriv(
                self.modB_spline, points_sym, (0, 0, 1)
            )
        dmodBdzeta[flip, 0] = -dmodBdzeta[flip, 0]

    def _R_impl(self, R):
        points = self.get_points_ref()
        points_sym, flip = self.map_points_symmetries(points)
        R[:, 0] = self._eval_3d_spline(self.R_spline, points_sym)

    def _dRds_impl(self, dRds):
        points = self.get_points_ref()
        points_sym, flip = self.map_points_symmetries(points)
        if not self.spline_deriv:
            dRds[:, 0] = self._eval_3d_spline(self.dRds_spline, points_sym)
        else:
            dRds[:, 0] = self._eval_3d_spline_deriv(
                self.R_spline, points_sym, (1, 0, 0)
            )

    def _dRdtheta_impl(self, dRdtheta):
        points = self.get_points_ref()
        points_sym, flip = self.map_points_symmetries(points)
        if not self.spline_deriv:
            dRdtheta[:, 0] = self._eval_3d_spline(self.dRdtheta_spline, points_sym)
        else:
            dRdtheta[:, 0] = self._eval_3d_spline_deriv(
                self.R_spline, points_sym, (0, 1, 0)
            )
        dRdtheta[flip, 0] = -dRdtheta[flip, 0]

    def _dRdzeta_impl(self, dRdzeta):
        points = self.get_points_ref()
        points_sym, flip = self.map_points_symmetries(points)
        if not self.spline_deriv:
            dRdzeta[:, 0] = self._eval_3d_spline(self.dRdzeta_spline, points_sym)
        else:
            dRdzeta[:, 0] = self._eval_3d_spline_deriv(
                self.R_spline, points_sym, (0, 0, 1)
            )
        dRdzeta[flip, 0] = -dRdzeta[flip, 0]

    def _Z_impl(self, Z):
        points = self.get_points_ref()
        points_sym, flip = self.map_points_symmetries(points)
        Z[:, 0] = self._eval_3d_spline(self.Z_spline, points_sym)
        Z[flip, 0] = -Z[flip, 0]

    def _dZds_impl(self, dZds):
        points = self.get_points_ref()
        points_sym, flip = self.map_points_symmetries(points)
        if not self.spline_deriv:
            dZds[:, 0] = self._eval_3d_spline(self.dZds_spline, points_sym)
        else:
            dZds[:, 0] = self._eval_3d_spline_deriv(
                self.Z_spline, points_sym, (1, 0, 0)
            )
        dZds[flip, 0] = -dZds[flip, 0]

    def _dZdtheta_impl(self, dZdtheta):
        points = self.get_points_ref()
        points_sym, flip = self.map_points_symmetries(points)
        if not self.spline_deriv:
            dZdtheta[:, 0] = self._eval_3d_spline(self.dZdtheta_spline, points_sym)
        else:
            dZdtheta[:, 0] = self._eval_3d_spline_deriv(
                self.Z_spline, points_sym, (0, 1, 0)
            )

    def _dZdzeta_impl(self, dZdzeta):
        points = self.get_points_ref()
        points_sym, flip = self.map_points_symmetries(points)
        if not self.spline_deriv:
            dZdzeta[:, 0] = self._eval_3d_spline(self.dZdzeta_spline, points_sym)
        else:
            dZdzeta[:, 0] = self._eval_3d_spline_deriv(
                self.Z_spline, points_sym, (0, 0, 1)
            )

    def _nu_impl(self, nu):
        points = self.get_points_ref()
        points_sym, flip = self.map_points_symmetries(points)
        nu[:, 0] = self._eval_3d_spline(self.nu_spline, points_sym)
        nu[flip, 0] = -nu[flip, 0]

    def _dnuds_impl(self, dnuds):
        points = self.get_points_ref()
        points_sym, flip = self.map_points_symmetries(points)
        if not self.spline_deriv:
            dnuds[:, 0] = self._eval_3d_spline(self.dnuds_spline, points_sym)
        else:
            dnuds[:, 0] = self._eval_3d_spline_deriv(
                self.nu_spline, points_sym, (1, 0, 0)
            )
        dnuds[flip, 0] = -dnuds[flip, 0]

    def _dnudtheta_impl(self, dnudtheta):
        points = self.get_points_ref()
        points_sym, flip = self.map_points_symmetries(points)
        if not self.spline_deriv:
            dnudtheta[:, 0] = self._eval_3d_spline(self.dnudtheta_spline, points_sym)
        else:
            dnudtheta[:, 0] = self._eval_3d_spline_deriv(
                self.nu_spline, points_sym, (0, 1, 0)
            )

    def _dnudzeta_impl(self, dnudzeta):
        points = self.get_points_ref()
        points_sym, flip = self.map_points_symmetries(points)
        if not self.spline_deriv:
            dnudzeta[:, 0] = self._eval_3d_spline(self.dnudzeta_spline, points_sym)
        else:
            dnudzeta[:, 0] = self._eval_3d_spline_deriv(
                self.nu_spline, points_sym, (0, 0, 1)
            )

    def _G_impl(self, G):
        points = self.get_points_ref()
        G[:, 0] = self.G_spline(points[:, 0])

    def _dGds_impl(self, dGds):
        points = self.get_points_ref()
        if not self.spline_deriv:
            dGds[:, 0] = self.dGds_spline(points[:, 0])
        else:
            dGds[:, 0] = self.G_spline.derivative()(points[:, 0])

    def _iota_impl(self, iota):
        points = self.get_points_ref()
        iota[:, 0] = self.iota_spline(points[:, 0])

    def _diotads_impl(self, diotads):
        points = self.get_points_ref()
        if not self.spline_deriv:
            diotads[:, 0] = self.diotads_spline(points[:, 0])
        else:
            diotads[:, 0] = self.iota_spline.derivative()(points[:, 0])

    def _I_impl(self, I):
        points = self.get_points_ref()
        I[:, 0] = self.I_spline(points[:, 0])

    def _dIds_impl(self, dIds):
        points = self.get_points_ref()
        if not self.spline_deriv:
            dIds[:, 0] = self.dIds_spline(points[:, 0])
        else:
            dIds[:, 0] = self.I_spline.derivative()(points[:, 0])

    def _psip_impl(self, psip):
        points = self.get_points_ref()
        psip[:, 0] = self.psip_spline(points[:, 0])

    def _K_impl(self, K):
        points = self.get_points_ref()
        points_sym, flip = self.map_points_symmetries(points)
        K[:, 0] = self._eval_3d_spline(self.K_spline, points_sym)
        K[flip, 0] = -K[flip, 0]

    def _dKdtheta_impl(self, dKdtheta):
        points = self.get_points_ref()
        points_sym, flip = self.map_points_symmetries(points)
        dKdtheta[:, 0] = self._eval_3d_spline_deriv(
            self.K_spline, points_sym, (0, 1, 0)
        )

    def _dKdzeta_impl(self, dKdzeta):
        points = self.get_points_ref()
        points_sym, flip = self.map_points_symmetries(points)
        dKdzeta[:, 0] = self._eval_3d_spline_deriv(self.K_spline, points_sym, (0, 0, 1))

    def map_points_symmetries(self, points):
        points_sym = points.copy()

        points_sym[:, 1] = np.mod(points[:, 1], 2 * np.pi)
        points_sym[:, 2] = np.mod(points[:, 2], 2 * np.pi / self.nfp)
        # Apply stellarator symmetry
        if self.stellsym:
            mask = np.where(points_sym[:, 1] > np.pi)
            points_sym[mask, 1] = 2 * np.pi - points_sym[mask, 1]
            points_sym[mask, 2] = 2 * np.pi / self.nfp - points_sym[mask, 2]
        else:
            mask = np.zeros(points.shape[0], dtype=bool)

        return points_sym, mask

    def compute_K(self, K):
        r"""
        Compute the Boozer K field on the full grid.

        Args:
            K: array-like (n_s, n_theta, n_zeta) output array to store the computed
                K values. The array should be pre-allocated with the correct shape
                matching (len(self.s_full), len(self.theta_grid),
                len(self.zeta_grid)).
        """
        points = np.zeros((len(self.theta_grid_3d.flatten()), 3))
        points[:, 0] = self.s_grid_3d.flatten()
        points[:, 1] = self.theta_grid_3d.flatten()
        points[:, 2] = self.zeta_grid_3d.flatten()
        self.set_points(points)
        zeta = points[:, 2]
        nu = self.nu()[:, 0]
        dnuds = self.dnuds()[:, 0]
        dnudtheta = self.dnudtheta()[:, 0]
        dnudzeta = self.dnudzeta()[:, 0]
        dRdtheta = self.dRdtheta()[:, 0]
        dRds = self.dRds()[:, 0]
        dRdzeta = self.dRdzeta()[:, 0]
        dZdtheta = self.dZdtheta()[:, 0]
        dZds = self.dZds()[:, 0]
        dZdzeta = self.dZdzeta()[:, 0]
        R = self.R()[:, 0]
        G = self.G()[:, 0]
        iota = self.iota()[:, 0]
        I = self.I()[:, 0]
        modB = self.modB()[:, 0]

        phi = zeta - nu
        dphids = -dnuds
        dphidtheta = -dnudtheta
        dphidzeta = 1 - dnudzeta
        dXdtheta = dRdtheta * np.cos(phi) - R * np.sin(phi) * dphidtheta
        dYdtheta = dRdtheta * np.sin(phi) + R * np.cos(phi) * dphidtheta
        dXds = dRds * np.cos(phi) - R * np.sin(phi) * dphids
        dYds = dRds * np.sin(phi) + R * np.cos(phi) * dphids
        dXdzeta = dRdzeta * np.cos(phi) - R * np.sin(phi) * dphidzeta
        dYdzeta = dRdzeta * np.sin(phi) + R * np.cos(phi) * dphidzeta
        gstheta = dXdtheta * dXds + dYdtheta * dYds + dZdtheta * dZds
        gszeta = dXdzeta * dXds + dYdzeta * dYds + dZdzeta * dZds
        sqrtg = (G + iota * I) / (modB * modB)
        K_flat = (gszeta + iota * gstheta) / (sqrtg * self.psi0)
        # Reshape to match the 3D grid shape
        K[:, :, :] = K_flat.reshape(
            (len(self.s_full), len(self.theta_grid), len(self.zeta_grid))
        )


class BoozerRadialInterpolant(BoozerMagneticField):
    r"""
     The magnetic field can be computed at any point in Boozer coordinates
     using radial spline interpolation
           (``scipy.interpolate.make_interp_spline``) and an inverse Fourier
      transform in the two angles.
           If given a `VMEC` output file, performs a Boozer coordinate
      transformation using ``BOOZXFORM``.
           If given a ``BOOZXFORM`` output file, the Boozer transformation must
      be performed with all surfaces on the VMEC
     half grid, and with `phip`, `chi`, `pres`, and `phi` saved in the file.

    Args:
        equil: instance of :class:`Booz_xform` or string containing the
            filename of a boozmn_*.nc file (produced with booz_xform) or
            wout_*.nc file (produced with VMEC). If a :class:`Booz_xform`
            instance or boozmn_*.nc file is passed, the `compute_surfs` needs to
            include all of the grid points in the half-radius grid of the corresponding
            Vmec equilibrium. Otherwise, a ValueError is raised.
        order: (int) order for radial interpolation. Must satisfy 1 <= order <=
            5.
        mpol: (int) number of poloidal mode numbers for BOOZXFORM (defaults to
            32). Only used if a wout_*.nc file is passed.
        ntor: (int) number of toroidal mode numbers for BOOZXFORM (defaults to
            32). Only used if a wout_*.nc file is passed.
        helicity_M : Poloidal helicity coefficient for enforcing field
            quasi-symmetry If specified, then the non-symmetric Fourier
            harmonics of :math:`B` and :math:`K` are filtered out, so the
            field is a function of `chi = helicity_M*theta - helicity_N*zeta`.
            If helicity is unspecified, all harmonics are kept.
            (defaults to ``None``)
        helicity_N : Toroidal helicity coefficient for enforcing field
            quasi-symmetry If specified, then the non-symmetric Fourier
            harmonics of :math:`B` and :math:`K` are filtered out, so the
            field is a function of `chi = helicity_M*theta - helicity_N*zeta`.
            If helicity is unspecified, all harmonics are kept.
        enforce_vacuum: If True, a vacuum field is assumed, :math:`G` is
            set to its mean value, :math:`I = 0`, and :math:`K = 0`.
        no_K: (bool) If ``True``, the Boozer :math:`K` will not be computed or
            interpolated.
        write_boozmn: (bool) If ``True``, save the booz_xform transformation in
            a filename specified by ``boozmn_name``. (defaults to ``True``)
        comm: A MPI communicator to parallelize over, from which
          the worker groups will be used for spline calculations. If ``comm`` is
          ``None``, each MPI process will compute splines independently.
        boozmn_name: (string) Filename to save booz_xform transformation if
            ``write_boozmn`` is ``True``.
        verbose: If True, additional output is written by booz_xform. (defaults
            to False).
        no_shear: If True, the shear in the rotational transform will be
            eliminated, and iota will be taken to be the mean value.
            (defaults to False).
        field_type: A string identifying additional assumptions made on the
            magnetic field. Can be
            ``'vac'``, ``'nok'``, or ``''``.  By default, this is determined
            from the options ``enforce_vacuum``
            and ``no_K``.
    """

    def __init__(
        self,
        equil,
        order,
        mpol=32,
        ntor=32,
        helicity_M=None,
        helicity_N=None,
        enforce_vacuum=False,
        no_K=False,
        write_boozmn=True,
        comm=None,
        boozmn_name="boozmn.nc",
        verbose=0,
        no_shear=False,
        field_type=None,
    ):
        self.comm = comm

        if self.comm is not None:
            self.proc0 = False
            if self.comm.rank == 0:
                self.proc0 = True
        else:
            self.proc0 = True

        if field_type is not None:
            field_type = field_type.lower()
            assert field_type in ["vac", "nok", ""]
            if self.proc0:
                if enforce_vacuum != (field_type == "vac"):
                    warnings.warn(
                        f"Prescribed field_type is inconsistent with enforce_vacuum. "
                        f"Proceeding with field_type={field_type}.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                if no_K != (field_type == "nok"):
                    warnings.warn(
                        f"Prescribed field_type is inconsistent with no_K. "
                        f"Proceeding with field_type={field_type}.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
            self.field_type = field_type
        else:
            if enforce_vacuum:
                self.field_type = "vac"
            elif no_K:
                self.field_type = "nok"
            else:
                self.field_type = ""

        if isinstance(equil, str):
            if self.proc0:
                basename = os.path.basename(equil)
                if basename[:4] == "wout":
                    booz = Booz_xform()
                    booz.read_wout(equil, True)
                    booz.verbose = verbose
                    booz.mboz = mpol
                    booz.nboz = ntor
                    booz.run()
                    if write_boozmn:
                        booz.write_boozmn(boozmn_name)
                    self.bx = booz
                elif basename[:4] == "booz":
                    booz = Booz_xform()
                    booz.verbose = verbose
                    booz.read_boozmn(equil)
                    self.bx = booz
                    # Check if grid does not have correct size
                    if self.bx.ns_in != len(self.bx.s_b):
                        raise ValueError("booz filename has incorrect s grid!")
                    # Check if grid does not match Vmec half grid
                    s_in_full = np.linspace(0, 1, self.bx.ns_in + 1)
                    s_in = 0.5 * (s_in_full[1::] + s_in_full[0:-1])
                    if not np.allclose(s_in, self.bx.s_b):
                        raise ValueError("booz filename has incorrect s grid!")
                else:
                    raise ValueError("Invalid filename")
        elif isinstance(equil, Booz_xform):
            if self.proc0:
                self.bx = equil
        else:
            raise ValueError("Incorrect equil type passed to BoozerRadialInterpolant.")

        self.no_shear = no_shear
        self.order = order
        self.helicity_M = helicity_M
        self.helicity_N = helicity_N
        self.enforce_qs = False
        self.enforce_vacuum = enforce_vacuum
        self.no_K = no_K
        if self.enforce_vacuum:
            self.no_K = True
        if (helicity_M is not None) and (helicity_N is not None):
            if helicity_M % 1 != 0:
                raise ValueError(
                    "helicity_M must be an integer for field to be 2π-periodic "
                    "in Boozer poloidal angle."
                )

            if helicity_N % 1 != 0:
                raise ValueError(
                    "helicity_N must be an integer for field to be 2π-periodic "
                    "in Boozer toroidal angle."
                )

            self.helicity_M = helicity_M
            self.helicity_N = helicity_N
            self.enforce_qs = True
        elif (helicity_M is not None) or (helicity_N is not None):
            raise ValueError(
                "Both helicity_M and helicity_N must be specified when "
                "enforcing field symmetry."
            )

        if self.proc0:
            self.asym = self.bx.asym  # Bool for stellarator asymmetry
            self.psi0 = -self.bx.phi[-1] / (
                2 * np.pi
            )  # Sign flip to account for VMEC convention.
            # See https://terpconnect.umd.edu/~mattland/assets/notes/vmec_signs.pdf
            # for phiedge definition
            self.nfp = self.bx.nfp
            self.mpol = self.bx.mboz
            self.ntor = self.bx.nboz
            self.s_half_ext = np.zeros(self.bx.ns_b + 2)
            self.s_half_ext[1:-1] = self.bx.s_b
            self.s_half_ext[-1] = 1
            self.init_splines()
        else:
            self.psip_spline = None
            self.G_spline = None
            self.I_spline = None
            self.dGds_spline = None
            self.dIds_spline = None
            self.iota_spline = None
            self.diotads_spline = None
            self.numns_splines = None
            self.rmnc_splines = None
            self.zmns_splines = None
            self.dnumnsds_splines = None
            self.drmncds_splines = None
            self.dzmnsds_splines = None
            self.bmnc_splines = None
            self.dbmncds_splines = None
            self.xm_b = None
            self.xn_b = None
            self.numnc_splines = None
            self.rmns_splines = None
            self.zmnc_splines = None
            self.dnumncds_splines = None
            self.drmnsds_splines = None
            self.dzmncds_splines = None
            self.bmns_splines = None
            self.dbmnsds_splines = None
            self.kmns_splines = None
            self.kmnc_splines = None
            self.asym = None
            self.psi0 = None
            self.nfp = None
            self.mpol = None
            self.ntor = None
            self.s_half_ext = None
        if self.comm is not None:
            self.psi0 = self.comm.bcast(self.psi0, root=0)
            self.nfp = self.comm.bcast(self.nfp, root=0)
            self.mpol = self.comm.bcast(self.mpol, root=0)
            self.ntor = self.comm.bcast(self.ntor, root=0)
            self.asym = self.comm.bcast(self.asym, root=0)
            self.psip_spline = self.comm.bcast(self.psip_spline, root=0)
            self.G_spline = self.comm.bcast(self.G_spline, root=0)
            self.I_spline = self.comm.bcast(self.I_spline, root=0)
            self.dGds_spline = self.comm.bcast(self.dGds_spline, root=0)
            self.dIds_spline = self.comm.bcast(self.dIds_spline, root=0)
            self.iota_spline = self.comm.bcast(self.iota_spline, root=0)
            self.diotads_spline = self.comm.bcast(self.diotads_spline, root=0)
            self.numns_splines = self.comm.bcast(self.numns_splines, root=0)
            self.rmnc_splines = self.comm.bcast(self.rmnc_splines, root=0)
            self.zmns_splines = self.comm.bcast(self.zmns_splines, root=0)
            self.dnumnsds_splines = self.comm.bcast(self.dnumnsds_splines, root=0)
            self.drmncds_splines = self.comm.bcast(self.drmncds_splines, root=0)
            self.dzmnsds_splines = self.comm.bcast(self.dzmnsds_splines, root=0)
            self.bmnc_splines = self.comm.bcast(self.bmnc_splines, root=0)
            self.dbmncds_splines = self.comm.bcast(self.dbmncds_splines, root=0)
            self.xm_b = self.comm.bcast(self.xm_b, root=0)
            self.xn_b = self.comm.bcast(self.xn_b, root=0)
            self.s_half_ext = self.comm.bcast(self.s_half_ext, root=0)
            if self.asym:
                self.numnc_splines = self.comm.bcast(self.numnc_splines, root=0)
                self.rmns_splines = self.comm.bcast(self.rmns_splines, root=0)
                self.zmnc_splines = self.comm.bcast(self.zmnc_splines, root=0)
                self.dnumncds_splines = self.comm.bcast(self.dnumncds_splines, root=0)
                self.drmnsds_splines = self.comm.bcast(self.drmnsds_splines, root=0)
                self.dzmncds_splines = self.comm.bcast(self.dzmncds_splines, root=0)
                self.bmns_splines = self.comm.bcast(self.bmns_splines, root=0)
                self.dbmnsds_splines = self.comm.bcast(self.dbmnsds_splines, root=0)

        if not self.no_K:
            self.compute_K()

        BoozerMagneticField.__init__(
            self, self.psi0, self.field_type, self.nfp, self.asym == 0
        )

    def init_splines(self):
        self.xm_b = self.bx.xm_b
        self.xn_b = self.bx.xn_b

        # Define quantities on extended half grid
        iota = np.zeros(self.bx.ns_b + 2)
        G = np.zeros(self.bx.ns_b + 2)
        I = np.zeros(self.bx.ns_b + 2)

        ds = self.bx.s_b[1] - self.bx.s_b[0]

        s_full = np.linspace(0, 1, self.bx.ns_b + 1)

        psip = self.bx.chi / (2 * np.pi)
        iota[1:-1] = self.bx.iota
        sign_psip = np.sign(((psip[1] - psip[0]) / self.psi0) / np.sign(iota[1]))
        psip *= sign_psip
        G[1:-1] = self.bx.Boozer_G_all
        I[1:-1] = self.bx.Boozer_I_all

        s_half_mn = self.s_half_ext
        # Cache size for efficiency
        nm_b = len(self.xm_b)
        ns_b2 = self.bx.ns_b + 2

        bmnc = np.zeros((nm_b, ns_b2))
        bmnc[:, 1:-1] = self.bx.bmnc_b
        bmnc[:, 0] = 1.5 * bmnc[:, 1] - 0.5 * bmnc[:, 2]
        bmnc[:, -1] = 1.5 * bmnc[:, -2] - 0.5 * bmnc[:, -3]

        numns = np.zeros((nm_b, ns_b2))
        rmnc = np.zeros((nm_b, ns_b2))
        zmns = np.zeros((nm_b, ns_b2))
        numns[:, 1:-1] = self.bx.numns_b
        numns[:, 0] = 1.5 * numns[:, 1] - 0.5 * numns[:, 2]
        numns[:, -1] = 1.5 * numns[:, -2] - 0.5 * numns[:, -3]
        rmnc[:, 1:-1] = self.bx.rmnc_b
        rmnc[:, 0] = 1.5 * rmnc[:, 1] - 0.5 * rmnc[:, 2]
        rmnc[:, -1] = 1.5 * rmnc[:, -2] - 0.5 * rmnc[:, -3]
        zmns[:, 1:-1] = self.bx.zmns_b
        zmns[:, 0] = 1.5 * zmns[:, 1] - 0.5 * zmns[:, 2]
        zmns[:, -1] = 1.5 * zmns[:, -2] - 0.5 * zmns[:, -3]

        if self.asym:
            bmns = np.zeros((nm_b, ns_b2))
            bmns[:, 1:-1] = self.bx.bmns_b
            bmns[:, 0] = 1.5 * bmns[:, 1] - 0.5 * bmns[:, 2]
            bmns[:, -1] = 1.5 * bmns[:, -2] - 0.5 * bmns[:, -3]

            numnc = np.zeros((nm_b, ns_b2))
            rmns = np.zeros((nm_b, ns_b2))
            zmnc = np.zeros((nm_b, ns_b2))
            numnc[:, 1:-1] = self.bx.numnc_b
            numnc[:, 0] = 1.5 * numnc[:, 1] - 0.5 * numnc[:, 2]
            numnc[:, -1] = 1.5 * numnc[:, -2] - 0.5 * numnc[:, -3]
            rmns[:, 1:-1] = self.bx.rmns_b
            rmns[:, 0] = 1.5 * rmns[:, 1] - 0.5 * rmns[:, 2]
            rmns[:, -1] = 1.5 * rmns[:, -2] - 0.5 * rmns[:, -3]
            zmnc[:, 1:-1] = self.bx.zmnc_b
            zmnc[:, 0] = 1.5 * zmnc[:, 1] - 0.5 * zmnc[:, 2]
            zmnc[:, -1] = 1.5 * zmnc[:, -2] - 0.5 * zmnc[:, -3]

        dbmncds = (bmnc[:, 2:-1] - bmnc[:, 1:-2]) / ds
        drmncds = (rmnc[:, 2:-1] - rmnc[:, 1:-2]) / ds
        dzmnsds = (zmns[:, 2:-1] - zmns[:, 1:-2]) / ds
        dnumnsds = (numns[:, 2:-1] - numns[:, 1:-2]) / ds
        if self.asym:
            dbmnsds = (bmns[:, 2:-1] - bmns[:, 1:-2]) / ds
            drmnsds = (rmns[:, 2:-1] - rmns[:, 1:-2]) / ds
            dzmncds = (zmnc[:, 2:-1] - zmnc[:, 1:-2]) / ds
            dnumncds = (numnc[:, 2:-1] - numnc[:, 1:-2]) / ds

        # Extrapolate to get points at s = 0 and s = 1
        iota[0] = 1.5 * iota[1] - 0.5 * iota[2]
        G[0] = 1.5 * G[1] - 0.5 * G[2]
        I[0] = 1.5 * I[1] - 0.5 * I[2]
        iota[-1] = 1.5 * iota[-2] - 0.5 * iota[-3]
        G[-1] = 1.5 * G[-2] - 0.5 * G[-3]
        I[-1] = 1.5 * I[-2] - 0.5 * I[-3]
        # Compute first derivatives - on full grid points in [1,ns-1]
        dGds = (G[2:-1] - G[1:-2]) / ds
        dIds = (I[2:-1] - I[1:-2]) / ds
        diotads = (iota[2:-1] - iota[1:-2]) / ds

        self.psip_spline = make_interp_spline(s_full, psip, k=self.order)
        if not self.enforce_vacuum:
            self.G_spline = make_interp_spline(self.s_half_ext, G, k=self.order)
            self.I_spline = make_interp_spline(self.s_half_ext, I, k=self.order)
            self.dGds_spline = make_interp_spline(s_full[1:-1], dGds, k=self.order)
            self.dIds_spline = make_interp_spline(s_full[1:-1], dIds, k=self.order)
        else:
            self.G_spline = make_interp_spline(
                self.s_half_ext,
                np.mean(G) * np.ones_like(self.s_half_ext),
                k=self.order,
            )
            self.I_spline = make_interp_spline(
                self.s_half_ext, np.zeros_like(self.s_half_ext), k=self.order
            )
            self.dGds_spline = make_interp_spline(
                s_full[1:-1], np.zeros_like(s_full[1:-1]), k=self.order
            )
            self.dIds_spline = make_interp_spline(
                s_full[1:-1], np.zeros_like(s_full[1:-1]), k=self.order
            )
        if not self.no_shear:
            self.iota_spline = make_interp_spline(self.s_half_ext, iota, k=self.order)
            self.diotads_spline = make_interp_spline(
                s_full[1:-1], diotads, k=self.order
            )
        else:
            self.iota_spline = make_interp_spline(
                self.s_half_ext,
                np.mean(iota) * np.ones_like(self.s_half_ext),
                k=self.order,
            )
            self.diotads_spline = make_interp_spline(
                s_full[1:-1], np.zeros_like(s_full[1:-1]), k=self.order
            )

        self.numns_splines = make_interp_spline(
            s_half_mn, numns.T, k=self.order, axis=0
        )
        self.rmnc_splines = make_interp_spline(s_half_mn, rmnc.T, k=self.order, axis=0)
        self.zmns_splines = make_interp_spline(s_half_mn, zmns.T, k=self.order, axis=0)

        if self.enforce_qs:
            bmnc_filtered = bmnc.copy()
            bmnc_filtered[
                self.helicity_M * self.xn_b != self.helicity_N * self.xm_b
            ] = 0
            self.bmnc_splines = make_interp_spline(
                s_half_mn, bmnc_filtered.T, k=self.order, axis=0
            )
            dbmncds_filtered = dbmncds.copy()
            dbmncds_filtered[
                self.helicity_M * self.xn_b != self.helicity_N * self.xm_b
            ] = 0
            self.dbmncds_splines = make_interp_spline(
                s_full[1:-1], (dbmncds_filtered).T, k=self.order, axis=0
            )
        else:
            self.bmnc_splines = make_interp_spline(
                s_half_mn, bmnc.T, k=self.order, axis=0
            )
            self.dbmncds_splines = make_interp_spline(
                s_full[1:-1], dbmncds.T, k=self.order, axis=0
            )
        self.dnumnsds_splines = make_interp_spline(
            s_full[1:-1], dnumnsds.T, k=self.order, axis=0
        )
        self.drmncds_splines = make_interp_spline(
            s_full[1:-1], drmncds.T, k=self.order, axis=0
        )
        self.dzmnsds_splines = make_interp_spline(
            s_full[1:-1], dzmnsds.T, k=self.order, axis=0
        )

        if self.asym:
            self.numnc_splines = make_interp_spline(
                s_half_mn, numnc.T, k=self.order, axis=0
            )
            self.rmns_splines = make_interp_spline(
                s_half_mn, rmns.T, k=self.order, axis=0
            )
            self.zmnc_splines = make_interp_spline(
                s_half_mn, zmnc.T, k=self.order, axis=0
            )
            if self.enforce_qs:
                bmns_filtered = bmns.copy()
                bmns_filtered[
                    self.helicity_M * self.xn_b != self.helicity_N * self.xm_b
                ] = 0
                self.bmns_splines = make_interp_spline(
                    s_half_mn, bmns_filtered.T, k=self.order, axis=0
                )
                dbmnsds_filtered = dbmnsds.copy()
                dbmnsds_filtered[
                    self.helicity_M * self.xn_b != self.helicity_N * self.xm_b
                ] = 0
                self.dbmnsds_splines = make_interp_spline(
                    s_full[1:-1], dbmnsds_filtered.T, k=self.order, axis=0
                )
            else:
                self.bmns_splines = make_interp_spline(
                    s_half_mn, bmns.T, k=self.order, axis=0
                )
                self.dbmnsds_splines = make_interp_spline(
                    s_full[1:-1], dbmnsds.T, k=self.order, axis=0
                )

            self.dnumncds_splines = make_interp_spline(
                s_full[1:-1], dnumncds.T, k=self.order, axis=0
            )
            self.drmnsds_splines = make_interp_spline(
                s_full[1:-1], drmnsds.T, k=self.order, axis=0
            )
            self.dzmncds_splines = make_interp_spline(
                s_full[1:-1], dzmncds.T, k=self.order, axis=0
            )

    def compute_K(self):
        ntheta = 2 * (2 * self.mpol + 1)
        nzeta = 2 * (2 * self.ntor + 1)
        thetas = np.linspace(0, 2 * np.pi, ntheta, endpoint=False)
        dtheta = thetas[1] - thetas[0]
        zetas = np.linspace(0, 2 * np.pi / self.nfp, nzeta, endpoint=False)
        dzeta = zetas[1] - zetas[0]
        thetas, zetas = np.meshgrid(thetas, zetas)
        thetas = thetas.flatten()
        zetas = zetas.flatten()

        # Cache frequently used sizes
        ns_half = len(self.s_half_ext)
        nm_b = len(self.xm_b)
        array_shape = (ns_half, nm_b)

        if self.comm is not None:
            size = self.comm.size
            rank = self.comm.rank

            angle_idxs = np.array([i * len(thetas) // size for i in range(size + 1)])
            first, last = angle_idxs[rank], angle_idxs[rank + 1]

            if self.asym:
                kmnc_buffer = allocate_aligned_and_padded_array(array_shape)
            kmns_buffer = allocate_aligned_and_padded_array(array_shape)
            thetas = thetas[first:last]
            zetas = zetas[first:last]

        dzmnsds_half = allocate_aligned_and_padded_array(array_shape)
        drmncds_half = allocate_aligned_and_padded_array(array_shape)
        dnumnsds_half = allocate_aligned_and_padded_array(array_shape)
        bmnc_half = allocate_aligned_and_padded_array(array_shape)
        rmnc_half = allocate_aligned_and_padded_array(array_shape)
        zmns_half = allocate_aligned_and_padded_array(array_shape)
        numns_half = allocate_aligned_and_padded_array(array_shape)
        kmns = allocate_aligned_and_padded_array(array_shape)
        if self.asym:
            dzmncds_half = allocate_aligned_and_padded_array(array_shape)
            drmnsds_half = allocate_aligned_and_padded_array(array_shape)
            dnumncds_half = allocate_aligned_and_padded_array(array_shape)
            bmns_half = allocate_aligned_and_padded_array(array_shape)
            rmns_half = allocate_aligned_and_padded_array(array_shape)
            zmnc_half = allocate_aligned_and_padded_array(array_shape)
            numnc_half = allocate_aligned_and_padded_array(array_shape)
            kmnc = allocate_aligned_and_padded_array(array_shape)

        # Fill pre-allocated arrays to maintain alignment
        # Only fill the actual data portion, not the padding
        n_modes = len(self.xm_b)
        bmnc_half[:, :n_modes] = self.bmnc_splines(self.s_half_ext)
        rmnc_half[:, :n_modes] = self.rmnc_splines(self.s_half_ext)
        zmns_half[:, :n_modes] = self.zmns_splines(self.s_half_ext)
        numns_half[:, :n_modes] = self.numns_splines(self.s_half_ext)

        # For derivatives, use the pre-computed values directly
        dnumnsds_vals = self.dnumnsds_splines(self.s_half_ext)
        drmncds_vals = self.drmncds_splines(self.s_half_ext)
        dzmnsds_vals = self.dzmnsds_splines(self.s_half_ext)

        dnumnsds_half[:, :n_modes] = dnumnsds_vals
        drmncds_half[:, :n_modes] = drmncds_vals
        dzmnsds_half[:, :n_modes] = dzmnsds_vals
        if self.asym:
            bmns_half[:, :n_modes] = self.bmns_splines(self.s_half_ext)
            rmns_half[:, :n_modes] = self.rmns_splines(self.s_half_ext)
            zmnc_half[:, :n_modes] = self.zmnc_splines(self.s_half_ext)
            numnc_half[:, :n_modes] = self.numnc_splines(self.s_half_ext)

            # For derivatives, use the pre-computed values directly
            dnumncds_vals = self.dnumncds_splines(self.s_half_ext)
            drmnsds_vals = self.drmnsds_splines(self.s_half_ext)
            dzmncds_vals = self.dzmncds_splines(self.s_half_ext)

            dnumncds_half[:, :n_modes] = dnumncds_vals
            drmnsds_half[:, :n_modes] = drmnsds_vals
            dzmncds_half[:, :n_modes] = dzmncds_vals

        G_half = self.G_spline(self.s_half_ext)
        I_half = self.I_spline(self.s_half_ext)
        iota_half = self.iota_spline(self.s_half_ext)

        xm_b = align_and_pad(self.xm_b)
        xn_b = align_and_pad(self.xn_b)

        if self.asym:
            sopp.compute_kmnc_kmns(
                kmnc,
                kmns,
                rmnc_half,
                drmncds_half,
                zmns_half,
                dzmnsds_half,
                numns_half,
                dnumnsds_half,
                bmnc_half,
                rmns_half,
                drmnsds_half,
                zmnc_half,
                dzmncds_half,
                numnc_half,
                dnumncds_half,
                bmns_half,
                iota_half,
                G_half,
                I_half,
                xm_b,
                xn_b,
                thetas,
                zetas,
            )

            kmnc = kmnc * dtheta * dzeta * self.nfp / self.psi0
        else:
            sopp.compute_kmns(
                kmns,
                rmnc_half,
                drmncds_half,
                zmns_half,
                dzmnsds_half,
                numns_half,
                dnumnsds_half,
                bmnc_half,
                iota_half,
                G_half,
                I_half,
                xm_b,
                xn_b,
                thetas,
                zetas,
            )
        kmns = kmns * dtheta * dzeta * self.nfp / self.psi0
        if self.comm is not None:
            if self.asym:
                self.comm.Allreduce([kmnc, MPI.DOUBLE], kmnc_buffer, op=MPI.SUM)
                kmnc = kmnc_buffer
            self.comm.Allreduce([kmns, MPI.DOUBLE], kmns_buffer, op=MPI.SUM)
            kmns = kmns_buffer
        if self.proc0:
            n_modes_actual = len(self.xm_b)  # Get actual number of modes
            self.kmns_splines = []
            if self.enforce_qs:
                kmns_filtered = kmns.copy()
                kmns_filtered[
                    self.helicity_M * self.xn_b != self.helicity_N * self.xm_b
                ] = 0
                self.kmns_splines = make_interp_spline(
                    self.s_half_ext,
                    kmns_filtered[:, :n_modes_actual],
                    k=self.order,
                    axis=0,
                )
            else:
                self.kmns_splines = make_interp_spline(
                    self.s_half_ext,
                    kmns[:, :n_modes_actual],
                    k=self.order,
                    axis=0,
                )

            if self.asym:
                if self.enforce_qs:
                    kmnc_filtered = kmnc.copy()
                    kmnc_filtered[
                        self.helicity_M * self.xn_b != self.helicity_N * self.xm_b
                    ] = 0
                    self.kmnc_splines = make_interp_spline(
                        self.s_half_ext,
                        kmnc_filtered[:, :n_modes_actual],
                        k=self.order,
                        axis=0,
                    )
                else:
                    self.kmnc_splines = make_interp_spline(
                        self.s_half_ext,
                        kmnc[:, :n_modes_actual],
                        k=self.order,
                        axis=0,
                    )
        if self.comm is not None:
            self.kmns_splines = self.comm.bcast(self.kmns_splines, root=0)
            if self.asym:
                self.kmnc_splines = self.comm.bcast(self.kmnc_splines, root=0)

    def _K_impl(self, K):
        K[:, 0] = 0.0
        if self.no_K:
            return

        self._compute_impl(K[:, 0], self.kmns_splines, "odd")

        if self.asym:
            self._compute_impl(K[:, 0], self.kmnc_splines, "even")

    def _dKdtheta_impl(self, dKdtheta):
        dKdtheta[:, 0] = 0.0
        if self.no_K:
            return

        self._compute_impl(
            dKdtheta[:, 0], lambda s: self.kmns_splines(s) * self.xm_b, "even"
        )

        if self.asym:
            self._compute_impl(
                dKdtheta[:, 0], lambda s: -self.kmnc_splines(s) * self.xm_b, "odd"
            )

    def _dKdzeta_impl(self, dKdzeta):
        dKdzeta[:, 0] = 0.0
        if self.no_K:
            return

        self._compute_impl(
            dKdzeta[:, 0], lambda s: -self.kmns_splines(s) * self.xn_b, "even"
        )

        if self.asym:
            self._compute_impl(
                dKdzeta[:, 0], lambda s: self.kmnc_splines(s) * self.xn_b, "odd"
            )

    def _nu_impl(self, nu):
        nu[:, 0] = 0.0

        self._compute_impl(nu[:, 0], self.numns_splines, "odd")

        if self.asym:
            self._compute_impl(nu[:, 0], self.numnc_splines, "even")

    def _dnudtheta_impl(self, dnudtheta):
        dnudtheta[:, 0] = 0.0

        self._compute_impl(
            dnudtheta[:, 0], lambda s: self.numns_splines(s) * self.xm_b, "even"
        )

        if self.asym:
            self._compute_impl(
                dnudtheta[:, 0], lambda s: -self.numnc_splines(s) * self.xm_b, "odd"
            )

    def _dnudzeta_impl(self, dnudzeta):
        dnudzeta[:, 0] = 0.0

        self._compute_impl(
            dnudzeta[:, 0], lambda s: -self.numns_splines(s) * self.xn_b, "even"
        )

        if self.asym:
            self._compute_impl(
                dnudzeta[:, 0], lambda s: self.numnc_splines(s) * self.xn_b, "odd"
            )

    def _dnuds_impl(self, dnuds):
        dnuds[:, 0] = 0.0

        self._compute_impl(dnuds[:, 0], self.dnumnsds_splines, "odd")

        if self.asym:
            self._compute_impl(dnuds[:, 0], self.dnumncds_splines, "even")

    def _dRdtheta_impl(self, dRdtheta):
        dRdtheta[:, 0] = 0.0

        self._compute_impl(
            dRdtheta[:, 0], lambda s: -self.rmnc_splines(s) * self.xm_b, "odd"
        )

        if self.asym:
            self._compute_impl(
                dRdtheta[:, 0], lambda s: self.rmns_splines(s) * self.xm_b, "even"
            )

    def _dRdzeta_impl(self, dRdzeta):
        dRdzeta[:, 0] = 0.0

        self._compute_impl(
            dRdzeta[:, 0], lambda s: self.rmnc_splines(s) * self.xn_b, "odd"
        )

        if self.asym:
            self._compute_impl(
                dRdzeta[:, 0], lambda s: -self.rmns_splines(s) * self.xn_b, "even"
            )

    def _dRds_impl(self, dRds):
        dRds[:, 0] = 0.0

        self._compute_impl(dRds[:, 0], self.drmncds_splines, "even")

        if self.asym:
            self._compute_impl(dRds[:, 0], self.drmnsds_splines, "odd")

    def _R_impl(self, R):
        R[:, 0] = 0.0

        self._compute_impl(R[:, 0], self.rmnc_splines, "even")

        if self.asym:
            self._compute_impl(R[:, 0], self.rmns_splines, "odd")

    def _dZdtheta_impl(self, dZdtheta):
        dZdtheta[:, 0] = 0.0

        self._compute_impl(
            dZdtheta[:, 0], lambda s: self.zmns_splines(s) * self.xm_b, "even"
        )

        if self.asym:
            self._compute_impl(
                dZdtheta[:, 0], lambda s: -self.zmnc_splines(s) * self.xm_b, "odd"
            )

    def _dZdzeta_impl(self, dZdzeta):
        dZdzeta[:, 0] = 0.0

        self._compute_impl(
            dZdzeta[:, 0], lambda s: -self.zmns_splines(s) * self.xn_b, "even"
        )

        if self.asym:
            self._compute_impl(
                dZdzeta[:, 0], lambda s: self.zmnc_splines(s) * self.xn_b, "odd"
            )

    def _dZds_impl(self, dZds):
        dZds[:, 0] = 0.0

        self._compute_impl(dZds[:, 0], self.dzmnsds_splines, "odd")

        if self.asym:
            self._compute_impl(dZds[:, 0], self.dzmncds_splines, "even")

    def _Z_impl(self, Z):
        Z[:, 0] = 0.0

        self._compute_impl(Z[:, 0], self.zmns_splines, "odd")

        if self.asym:
            self._compute_impl(Z[:, 0], self.zmnc_splines, "even")

    def _psip_impl(self, psip):
        points = self.get_points_ref()
        s = points[:, 0]
        us, inv = np.unique(s, return_inverse=True)
        psip[:] = self.psip_spline(us)[inv][:, None]

    def _G_impl(self, G):
        points = self.get_points_ref()
        s = points[:, 0]
        us, inv = np.unique(s, return_inverse=True)
        G[:] = self.G_spline(us)[inv][:, None]

    def _I_impl(self, I):
        points = self.get_points_ref()
        s = points[:, 0]
        us, inv = np.unique(s, return_inverse=True)
        I[:] = self.I_spline(us)[inv][:, None]

    def _iota_impl(self, iota):
        points = self.get_points_ref()
        s = points[:, 0]
        us, inv = np.unique(s, return_inverse=True)
        iota[:] = self.iota_spline(us)[inv][:, None]

    def _dGds_impl(self, dGds):
        points = self.get_points_ref()
        s = points[:, 0]
        us, inv = np.unique(s, return_inverse=True)
        dGds[:] = self.dGds_spline(us)[inv][:, None]

    def _dIds_impl(self, dIds):
        points = self.get_points_ref()
        s = points[:, 0]
        us, inv = np.unique(s, return_inverse=True)
        dIds[:] = self.dIds_spline(us)[inv][:, None]

    def _diotads_impl(self, diotads):
        points = self.get_points_ref()
        s = points[:, 0]
        us, inv = np.unique(s, return_inverse=True)
        diotads[:] = self.diotads_spline(us)[inv][:, None]

    def _modB_impl(self, modB):
        modB[:, 0] = 0.0

        self._compute_impl(modB[:, 0], self.bmnc_splines, "even")

        if self.asym:
            self._compute_impl(modB[:, 0], self.bmns_splines, "odd")

    def _dmodBdtheta_impl(self, dmodBdtheta):
        dmodBdtheta[:, 0] = 0.0

        self._compute_impl(
            dmodBdtheta[:, 0], lambda s: -self.xm_b * self.bmnc_splines(s), "odd"
        )

        if self.asym:
            self._compute_impl(
                dmodBdtheta[:, 0], lambda s: self.xm_b * self.bmns_splines(s), "even"
            )

    def _dmodBdzeta_impl(self, dmodBdzeta):
        dmodBdzeta[:, 0] = 0.0

        self._compute_impl(
            dmodBdzeta[:, 0], lambda s: self.xn_b * self.bmnc_splines(s), "odd"
        )

        if self.asym:
            self._compute_impl(
                dmodBdzeta[:, 0], lambda s: -self.xn_b * self.bmns_splines(s), "even"
            )

    def _dmodBds_impl(self, dmodBds):
        dmodBds[:, 0] = 0.0

        self._compute_impl(dmodBds[:, 0], self.dbmncds_splines, "even")

        if self.asym:
            self._compute_impl(dmodBds[:, 0], self.dbmnsds_splines, "odd")

    def _compute_impl(self, output, coeffs, parity):
        r"""
        Add to ``output`` the inverse Fourier transform

        .. math::
            \sum_{mn} c_{mn}(s) \times \begin{cases}
                \cos(m\theta - n\zeta) & \text{parity = "even"} \\
                \sin(m\theta - n\zeta) & \text{parity = "odd"}
            \end{cases}

        evaluated at the current points.

        Args:
            output: 1d array of length ``npoints`` that the result is added to.
            coeffs: callable mapping an array of ``s`` values to the
                ``(len(s), nmodes)`` array of radial coefficients
                :math:`c_{mn}(s)` for every Boozer mode. It is called once,
                on the unique ``s`` values of the current point set.
            parity: ``"even"`` or ``"odd"``, selecting cosine or sine.
        """
        assert parity in ("even", "odd")
        # Fast path for single point evaluation
        points = self.get_points_ref()
        if len(points) == 1:
            return self._compute_single_point(output, coeffs, parity)

        if self.comm is not None:
            size = self.comm.size
            rank = self.comm.rank

            mn_idxs = np.array([i * len(self.xm_b) // size for i in range(size + 1)])
            first_mn, last_mn = mn_idxs[rank], mn_idxs[rank + 1]

            # Pre-allocate and reuse buffers if possible
            if not hasattr(self, "_compute_buffers"):
                self._compute_buffers = {}

            # Reuse recv_buffer if shape matches
            recv_key = ("recv", output.shape)
            if recv_key not in self._compute_buffers:
                self._compute_buffers[recv_key] = np.zeros(output.shape)
            recv_buffer = self._compute_buffers[recv_key]
            recv_buffer.fill(0)  # Clear the buffer
        else:
            first_mn, last_mn = 0, len(self.xm_b)

        points = self.get_points_ref()
        s = points[:, 0]
        thetas = points[:, 1]
        zetas = points[:, 2]
        us, inv = np.unique(s, return_inverse=True)

        # Evaluate the radial coefficients for every mode this rank owns. The
        # splines are vector valued over the mode index, so this is a single
        # call for all modes, on the unique s values only.
        cmn = coeffs(us)[:, first_mn:last_mn]

        result = self._sum_harmonics_separable(
            cmn, inv, thetas, zetas, parity, first_mn, last_mn
        )
        if result is None:
            result = self._sum_harmonics_pointwise(
                cmn, inv, thetas, zetas, parity, first_mn, last_mn
            )

        if self.comm is not None:
            # In place reduce is slightly slower
            # self.mpi.comm_world.Allreduce(MPI.IN_PLACE,
            # [result, MPI.DOUBLE], op=MPI.SUM)
            self.comm.Allreduce([result, MPI.DOUBLE], recv_buffer, op=MPI.SUM)
            output += recv_buffer
        else:
            output += result

    def _sum_harmonics_pointwise(
        self, cmn, inv, thetas, zetas, parity, first_mn, last_mn
    ):
        """
        Sum the Fourier series one point at a time, in C++. Cost is
        ``npoints * nmodes``; used for scattered points, where the point set
        has no exploitable structure.
        """
        npoints = len(inv)

        # Pre-allocate and reuse buffers if possible
        if not hasattr(self, "_compute_buffers"):
            self._compute_buffers = {}

        # align_and_pad returns its argument untouched when it is already
        # aligned and padded, so there is nothing to cache here.
        padded_thetas = align_and_pad(thetas)
        padded_zetas = align_and_pad(zetas)

        # Optimize buffer allocation with better key strategy
        buffer_key = ("padded", npoints)
        if buffer_key not in self._compute_buffers:
            self._compute_buffers[buffer_key] = allocate_aligned_and_padded_array(
                (npoints,)
            )
        padded_buffer = self._compute_buffers[buffer_key]

        # Use memset-like operation for faster clearing
        if padded_buffer.size > 0:
            padded_buffer.fill(0)

        # Optimize chunk allocation with better sizing
        chunk_key = ("chunk", (last_mn - first_mn, npoints))
        if chunk_key not in self._compute_buffers:
            self._compute_buffers[chunk_key] = allocate_aligned_and_padded_array(
                (last_mn - first_mn, npoints)
            )
        chunk_mn = self._compute_buffers[chunk_key]
        # Broadcast the per-unique-s coefficients out to one column per point.
        # take(..., out=) writes straight into the buffer; assigning
        # cmn.T[:, inv] instead would materialize a second array of the same
        # (nmodes, npoints) size.
        np.take(cmn.T, inv, axis=1, out=chunk_mn[:, :npoints])

        inverse_fourier = (
            sopp.inverse_fourier_transform_even
            if parity == "even"
            else sopp.inverse_fourier_transform_odd
        )
        inverse_fourier(
            padded_buffer,
            chunk_mn,
            self.xm_b[first_mn:last_mn],
            self.xn_b[first_mn:last_mn],
            padded_thetas,
            padded_zetas,
            self.ntor,
            self.nfp,
            False,
        )
        return padded_buffer[:npoints]

    def _sum_harmonics_separable(
        self, cmn, inv, thetas, zetas, parity, first_mn, last_mn
    ):
        r"""
        Sum the Fourier series by partial summation over a tensor-product
        lattice of the distinct :math:`(s, \theta, \zeta)` values, using

        .. math::
            \sum_{mn} c_{mn}\cos(m\theta - n\zeta)
              = \sum_m \left[\cos(m\theta) A_m(\zeta)
                             + \sin(m\theta) B_m(\zeta)\right],

        with :math:`A_m(\zeta) = \sum_n c_{mn}\cos(n\zeta)` and
        :math:`B_m(\zeta) = \sum_n c_{mn}\sin(n\zeta)`. This replaces the
        ``npoints * nmodes`` mode loop with four matrix products, which for the
        grids used to build an :class:`InterpolatedBoozerField` is one to two
        orders of magnitude cheaper.

        Returns ``None`` when the point set is not separable enough to pay off
        (e.g. scattered points), leaving the caller to fall back to
        :meth:`_sum_harmonics_pointwise`.
        """
        npoints = len(inv)
        nmodes = last_mn - first_mn
        if npoints < 64 or nmodes == 0:
            return None

        rect = self._mode_rectangle(first_mn, last_mn)
        if rect is None:
            return None
        m_idx, n_idx, ms, ns_ = rect

        uth, tinv = np.unique(thetas, return_inverse=True)
        uze, zinv = np.unique(zetas, return_inverse=True)
        nus, nth, nze = cmn.shape[0], len(uth), len(uze)
        M, N = len(ms), len(ns_)

        # Only worth it if the lattice of distinct values is small compared to
        # summing over every mode at every point.
        lattice = nus * nth * nze
        cost_lattice = nus * (2 * M * N * nze + 4 * nth * M * nze)
        cost_pointwise = 4 * npoints * nmodes
        if lattice > 8 * npoints or cost_lattice > 0.5 * cost_pointwise:
            return None

        cos_mt = np.cos(np.outer(uth, ms))
        sin_mt = np.sin(np.outer(uth, ms))
        angle_z = np.outer(ns_ * self.nfp, uze)
        cos_nz = np.cos(angle_z)
        sin_nz = np.sin(angle_z)

        crect = np.zeros((nus, M, N))
        crect[:, m_idx, n_idx] = cmn

        a_ = crect @ cos_nz  # (nus, M, nze)
        b_ = crect @ sin_nz
        if parity == "even":
            # cos(m theta - n zeta) = cos(m theta) cos(n zeta)
            #                       + sin(m theta) sin(n zeta)
            lattice_vals = cos_mt @ a_ + sin_mt @ b_
        else:
            # sin(m theta - n zeta) = sin(m theta) cos(n zeta)
            #                       - cos(m theta) sin(n zeta)
            lattice_vals = sin_mt @ a_ - cos_mt @ b_
        return np.ascontiguousarray(lattice_vals[inv, tinv, zinv])

    def _mode_rectangle(self, first_mn, last_mn):
        """
        Index the modes ``[first_mn, last_mn)`` into a dense ``(m, n)``
        rectangle, which is what makes partial summation possible. Boozer mode
        tables are rectangular up to a missing half-row at ``m = 0``, so the
        padding this introduces is negligible. Returns ``None`` if the toroidal
        mode numbers are not multiples of ``nfp``, or if a mode is repeated, in
        which case no such rectangle exists.
        """
        if not hasattr(self, "_mode_rect_cache"):
            self._mode_rect_cache = {}
        key = (first_mn, last_mn)
        if key not in self._mode_rect_cache:
            xm = self.xm_b[first_mn:last_mn]
            xn = self.xn_b[first_mn:last_mn]
            duplicated = len(np.unique(np.stack([xm, xn], axis=1), axis=0)) != len(xm)
            if np.any(np.mod(xn, self.nfp) != 0) or duplicated:
                self._mode_rect_cache[key] = None
            else:
                m = np.rint(xm).astype(int)
                n = np.rint(xn / self.nfp).astype(int)
                ms = np.arange(m.min(), m.max() + 1)
                ns_ = np.arange(n.min(), n.max() + 1)
                self._mode_rect_cache[key] = (m - m.min(), n - n.min(), ms, ns_)
        return self._mode_rect_cache[key]

    def _compute_single_point(self, output, coeffs, parity):
        """
        Fast path for single point evaluation - avoids unnecessary allocations
        """
        points = self.get_points_ref()
        s = points[0, 0]
        theta = points[0, 1]
        zeta = points[0, 2]

        # Create minimal arrays for single point computation
        us = np.array([s])

        chunk_mn = align_and_pad(coeffs(us).reshape(-1))
        xm = align_and_pad(self.xm_b)
        xn = align_and_pad(self.xn_b)

        # Create minimal padded arrays
        padded_thetas = np.array([theta])
        padded_zetas = np.array([zeta])

        # Allocate minimal output buffer
        padded_buffer = np.zeros((1, 1))

        inverse_fourier = (
            sopp.inverse_fourier_transform_even
            if parity == "even"
            else sopp.inverse_fourier_transform_odd
        )
        inverse_fourier(
            padded_buffer,
            chunk_mn.reshape(-1, 1),
            xm,
            xn,
            padded_thetas,
            padded_zetas,
            self.ntor,
            self.nfp,
            False,
        )

        # Copy result to output
        output[0] += padded_buffer[0, 0]
        return output


class InterpolatedBoozerField(sopp.InterpolatedBoozerField, BoozerMagneticField):
    r"""
    This field takes an existing :class:`BoozerMagneticField` and interpolates it on a
    regular grid in :math:`s,\theta,\zeta`. the field is represented as a piecewise
    polynomial in (s,theta,zeta) of a given degree. The number of nodes in
    each direction
    are defined by ns_interp, ntheta_interp, and nzeta_interp. It is recommended to use
    this field representation in the tracing loop due to its speed in comparison to
    :class:`BoozerRadialInterpolant`.
    """

    def __init__(
        self,
        field,
        degree,
        srange=None,
        thetarange=None,
        zetarange=None,
        ns_interp=48,
        ntheta_interp=48,
        nzeta_interp=48,
        extrapolate=True,
        nfp=None,
        stellsym=None,
        initialize=None,
        comm=None,
    ):
        r"""
        Args:
            field: the underlying :class:`firm3d.field.boozermagneticfield.
                BoozerMagneticField` to be interpolated.
            degree: the degree of the piecewise polynomial interpolant.
            ns_interp: number of grid points in the :math:`s` direction.
            ntheta_interp: number of grid points in the :math:`\theta` direction.
            nzeta_interp: number of grid points in the :math:`\zeta` direction
            srange: a 3-tuple of the form ``(smin, smax, ns)``. This mean that
                the interval ``[smin, smax]`` is split into ``ns`` many subintervals.
            thetarange: a 3-tuple of the form ``(thetamin, thetamax, ntheta)``.
                thetamin must be >= 0, and thetamax must be <=2*pi.
            zetarange: a 3-tuple of the form ``(zetamin, zetamax, nzeta)``.
                zetamin must be >= 0, and thetamax must be <=2*pi.
            extrapolate: whether to extrapolate the field when evaluate outside
                         the integration domain or to throw an error.
            nfp: Whether to exploit rotational symmetry. In this case any toroidal angle
                 is always mapped into the interval :math:`[0, 2\pi/\mathrm{nfp})`,
                 hence it makes sense to use ``zetamin=0`` and
                 ``zetamax=2*np.pi/nfp``. By default this is obtained from field.nfp.
            stellsym: Whether to exploit stellarator symmetry. In this case
                      ``theta`` is always mapped to the interval :math:`[0, \pi]`,
                      hence it makes sense to use ``thetamin=0`` and
                      ``thetamax=np.pi``. By default
                      this is obtained from field.stellsym.
            initialize: A list of strings, each of which is the name of a
                field quantity, e.g., `modB`, to be initialized when the
                interpolant is created.
                By default, this list is determined by field.field_type.
            comm: (MPI.Comm, optional) MPI communicator for parallel interpolation.
                If provided, interpolation operations will be parallelized across
                MPI processes. Default is None (sequential).
        """
        if initialize is None:
            initialize = []
        field_type = field.field_type.lower()
        assert field_type in ["", "vac", "nok"]
        self.field_type = field.field_type

        initialize = sorted(initialize)
        initialize_vac = sorted(["modB", "psip", "G", "iota", "modB_derivs"])
        initialize_nok = sorted(
            ["modB", "psip", "G", "I", "dGds", "dIds", "iota", "modB_derivs"]
        )
        initialize_gen = sorted(
            [
                "modB",
                "psip",
                "G",
                "I",
                "dGds",
                "dIds",
                "iota",
                "modB_derivs",
                "K",
                "K_derivs",
            ]
        )
        if initialize == []:
            if field_type == "vac":
                initialize = initialize_vac
            elif field_type == "nok":
                initialize = initialize_nok
            elif field_type == "":
                initialize = initialize_gen
        else:
            if (
                (field_type == "vac" and (initialize != initialize_vac))
                or (field_type == "nok" and (initialize != initialize_nok))
                or (field_type == "" and (initialize != initialize_gen))
            ):
                warnings.warn(
                    f"initialize list does not match field_type={field_type}. "
                    f"Proceeding with initialize={initialize}",
                    RuntimeWarning,
                    stacklevel=2,
                )
        if nfp is None:
            nfp = field.nfp
        if stellsym is None:
            stellsym = field.stellsym

        if srange is None:
            srange = (0, 1, ns_interp)
        if thetarange is None:
            if stellsym:
                thetarange = (0, np.pi, ntheta_interp)
            else:
                thetarange = (0, 2 * np.pi, ntheta_interp)
        if zetarange is None:
            zetarange = (0, 2 * np.pi / nfp, nzeta_interp)

        BoozerMagneticField.__init__(self, field.psi0, self.field_type, nfp)
        if np.any(np.asarray(thetarange[0:2]) < 0) or np.any(
            np.asarray(thetarange[0:2]) > 2 * np.pi
        ):
            raise ValueError("thetamin and thetamax must be in [0,2*pi]")
        if np.any(np.asarray(zetarange[0:2]) < 0) or np.any(
            np.asarray(zetarange[0:2]) > 2 * np.pi
        ):
            raise ValueError("zetamin and zetamax must be in [0,2*pi]")
        if stellsym and (
            np.any(np.asarray(thetarange[0:2]) < 0)
            or np.any(np.asarray(thetarange[0:2]) > np.pi)
        ):
            warnings.warn(
                rf"Sure about thetarange=[{thetarange[0]},{thetarange[1]}]? "
                rf"When exploiting stellarator symmetry, the interpolant is "
                rf"only evaluated for theta in [0,pi].",
                RuntimeWarning,
                stacklevel=2,
            )
        if nfp > 1 and (
            np.any(np.asarray(zetarange[0:2]) < 0)
            or np.any(np.asarray(zetarange[0:2]) > 2 * np.pi / nfp)
        ):
            warnings.warn(
                rf"Sure about zetarange=[{zetarange[0]},{zetarange[1]}]? "
                rf"When exploiting rotational symmetry, the interpolant is "
                rf"only evaluated for zeta in [0,2\pi/nfp].",
                RuntimeWarning,
                stacklevel=2,
            )

        sopp.InterpolatedBoozerField.__init__(
            self,
            field,
            degree,
            srange,
            thetarange,
            zetarange,
            extrapolate,
            nfp,
            stellsym,
            field_type,
        )

        if comm is not None:
            # Convert mpi4py communicator to Fortran handle using py2f()
            comm_fortran = comm.py2f()
            self.set_mpi_comm(comm_fortran)

        if initialize:
            for item in initialize:
                getattr(self, item)()

    @classmethod
    def from_json(cls, json_file_path):
        """
        Load an InterpolatedBoozerField from a JSON file.

        This class method creates a new InterpolatedBoozerField instance by loading
        pre-computed interpolant data from JSON file, avoiding recomputation.

        Args:
            json_file_path: Path to the JSON file containing the saved field data.

        Returns:
            InterpolatedBoozerField: A new instance loaded from the JSON file.

        Example:
            # First, create and save a field
            field = InterpolatedBoozerField(bri, 4, srange, thetarange, zetarange, True)
            field.to_json("field_data.json")
            # Later, reload without recomputation
            loaded_field = InterpolatedBoozerField.from_json("field_data.json")
        """
        # InterpolatedBoozerField inherits from both
        # sopp.InterpolatedBoozerField (C++) and BoozerMagneticField (Python).
        # We must initialize each parent separately since the normal __init__
        # requires a BoozerRadialInterpolant which we don't have when loading.
        instance = cls.__new__(cls)

        # C++ side: reads JSON and restores all interpolants + status flags
        sopp.InterpolatedBoozerField.__init__(instance, json_file_path)

        # Python side needs psi0, field_type, nfp, stellsym. C++ keeps them private,
        # so we use getters instead of attributes.
        BoozerMagneticField.__init__(
            instance,
            instance.get_psi0(),
            instance.get_field_type(),
            instance.get_nfp(),
            instance.get_stellsym(),
        )

        return instance

    @classmethod
    def from_booz_xform(
        cls,
        equil,
        degree=3,
        mpol=32,
        ntor=32,
        ns=None,
        ntheta=48,
        nzeta=48,
        helicity_M=None,
        helicity_N=None,
        enforce_vacuum=False,
        no_K=True,
        write_boozmn=True,
        boozmn_name="boozmn.nc",
        field_type=None,
        comm=None,
        spline_deriv=True,
        extrapolate=True,
        initialize=None,
    ):
        r"""
        Create an InterpolatedBoozerField from a Booz_xform equilibrium.

        This is a convenience class method that creates a :class:`BoozerSplineField`
        from the equilibrium and then wraps it with an :class:`InterpolatedBoozerField`.
        The resulting field is represented as a piecewise polynomial interpolant
        on a regular grid in :math:`(s, \theta, \zeta)` coordinates.

        Args:
            equil: Instance of :class:`Booz_xform` or string containing the
                filename of a boozmn_*.nc file (produced with booz_xform) or
                wout_*.nc file (produced with VMEC). If a :class:`Booz_xform`
                instance or boozmn_*.nc file is passed, the `compute_surfs` needs
                to include all of the grid points in the half-radius grid of the
                corresponding Vmec equilibrium. Otherwise, a ValueError is raised.
            degree: The degree of the piecewise polynomial interpolant (default: 3).
            mpol: Number of poloidal mode numbers for BOOZXFORM (default: 32).
                Only used if a wout_*.nc file is passed.
            ntor: Number of toroidal mode numbers for BOOZXFORM (default: 32).
                Only used if a wout_*.nc file is passed.
            ns: Number of grid points in the :math:`s` direction for interpolation.
                If None, uses the number of flux surfaces from the equilibrium
                (default: None).
            ntheta: Number of grid points in the :math:`\theta` direction (default: 48).
            nzeta: Number of grid points in the :math:`\zeta` direction (default: 48).
            helicity_M: Poloidal helicity coefficient for enforcing field
                quasi-symmetry. If specified, then the non-symmetric Fourier
                harmonics of :math:`B` and :math:`K` are filtered out, so the
                field is a function of :math:`\chi = M\theta - N\zeta`.
                If helicity is unspecified, all harmonics are kept (default: None).
            helicity_N: Toroidal helicity coefficient for enforcing field
                quasi-symmetry. Must be specified together with helicity_M
                (default: None).
            enforce_vacuum: If True, a vacuum field is assumed, :math:`G` is
                set to its mean value, :math:`I = 0`, and :math:`K = 0`
                (default: False).
            no_K: If True, the Boozer :math:`K` will not be computed or
                interpolated (default: True).
            write_boozmn: If True, save the booz_xform transformation in
                a file specified by ``boozmn_name`` (default: True).
            boozmn_name: Filename to save booz_xform transformation if
                ``write_boozmn`` is True (default: "boozmn.nc").
            field_type: A string identifying additional assumptions made on the
                magnetic field. Can be ``'vac'``, ``'nok'``, or ``''``.
                By default, this is determined from the options ``enforce_vacuum``
                and ``no_K`` (default: None).
            comm: MPI communicator for parallelization. If provided, the field
                interpolation will be parallelized across MPI processes.
                Should be an mpi4py communicator object (e.g., MPI.COMM_WORLD)
                (default: None).
            spline_deriv: If True, field derivatives will be evaluated by
                differentiating the spline field. Otherwise, Fourier derivatives are
                evaluated for the angles, and centered differences are used for the
                radial derivative. While False is more accurate, True is faster
                due to a reduction in the number of spline evaluations (default: True).
            extrapolate: Whether to extrapolate the field when evaluating outside
                the interpolation domain or to throw an error (default: True).
            initialize: List of strings, each of which is the name of a field
                quantity (e.g., ``'modB'``, ``'G'``, ``'I'``) to be initialized
                when the interpolant is created. If None, the list is determined
                by ``field_type`` (default: None).

        Returns:
            :class:`InterpolatedBoozerField`: An interpolated field instance
            created from the specified equilibrium.

        Examples:
            .. code-block:: python

                # Create from a boozmn file
                field = InterpolatedBoozerField.from_booz_xform(
                    "boozmn_aten.nc",
                    degree=3,
                    ns=48,
                    ntheta=48,
                    nzeta=48,
                    comm=MPI.COMM_WORLD
                )

                # Create with vacuum field assumptions
                field = InterpolatedBoozerField.from_booz_xform(
                    "wout_aten.nc",
                    enforce_vacuum=True,
                    comm=comm
                )
        """
        bsf = BoozerSplineField(
            equil,
            mpol=mpol,
            ntor=ntor,
            ntheta=ntheta,
            nzeta=nzeta,
            helicity_M=helicity_M,
            helicity_N=helicity_N,
            enforce_vacuum=enforce_vacuum,
            no_K=no_K,
            write_boozmn=write_boozmn,
            boozmn_name=boozmn_name,
            field_type=field_type,
            comm=comm,
            spline_deriv=spline_deriv,
        )
        if ns is None:
            ns = bsf.ns_b
        return cls(
            bsf,
            ns_interp=ns,
            ntheta_interp=ntheta,
            nzeta_interp=nzeta,
            degree=degree,
            extrapolate=extrapolate,
            nfp=bsf.nfp,
            stellsym=bsf.stellsym,
            initialize=initialize,
            comm=comm,
        )


class ShearAlfvenWave(sopp.ShearAlfvenWave):
    r"""
    Class representing a generic Shear Alfvén Wave (SAW).

    The Shear Alfvén Wave (SAW) propagates in an equilibrium magnetic field
    `B0` and is represented by
    the scalar potential `Phi` and vector potential parameter `alpha`. The SAW
    magnetic field is defined
    as the curl of `(alpha * B0)`.

    This class provides a framework for representing SAWs in Boozer
    coordinates with attributes for computing
    the scalar and vector potentials and their derivatives: `Phi`, `dPhidpsi`,
    `Phidot`, etc.

    This class is designed to be a base class that can be extended to implement
    specific behaviors or
    variations of Shear Alfvén Waves.

    The usage of :class:`ShearAlfvenWave` is as follows:

    .. code-block:: python

        # Create an instance of a Boozer magnetic field
        B0 = sopp.BoozerAnalytic(etabar, B0, N, G0, psi0, iota0)

        # Create an instance of ShearAlfvenWave using the equilibrium field B0
        saw = ShearAlfvenWave(B0)

        # Points is a (n, 4) numpy array defining
        # :math:`(s, \theta, \zeta, \text{time})`
        points = ...
        saw.set_points(points)

        # Compute scalar potential Phi at the specified points
        Phi = saw.Phi()

    Attributes:
    ----------
    Phi : function
        Computes the scalar potential `Phi` of the shear Alfvén wave perturbation.
    dPhidpsi : function
        Computes the derivative of the scalar potential `Phi` with respect to `psi`.
    Phidot : function
        Computes the time derivative of the scalar potential `Phi`.
    dPhidtheta : function
        Computes the derivative of the scalar potential `Phi` with respect to `theta`.
    dPhidzeta : function
        Computes the derivative of the scalar potential `Phi` with respect to `zeta`.
    alpha : function
        Computes the vector potential parameter `alpha`.
    alphadot : function
        Computes the time derivative of the vector potential parameter `alpha`.
    dalphadtheta : function
        Computes the derivative of the vector potential parameter `alpha` with
        respect to `theta`.
    dalphadpsi : function
        Computes the derivative of the vector potential parameter `alpha` with
        respect to `psi`.
    dalphadzeta : function
        Computes the derivative of the vector potential parameter `alpha` with
        respect to `zeta`.

    For further details, see Paul et al., JPP (2023; 89(5): 905890515.
    doi:10.1017/S0022377823001095)
    and references therein.

    Parameters
    ----------
    B0 : BoozerMagneticField
        Instance of a magnetic field in Boozer coordinates that provides the
        equilibrium field `B0`.

    Raises
    ------
    TypeError
        If `B0` is not an instance of `BoozerMagneticField`.

    """

    def __init__(self, B0, omega):
        if not isinstance(B0, sopp.BoozerMagneticField):
            raise TypeError("B0 must be an instance of BoozerMagneticField.")

        self.omega = omega
        # Call the constructor of the base C++ class
        super().__init__(B0)

        # Initialize additional field attributes for perturbed tracing
        if self.B0.field_type == "nok":
            initialize = ["diotads"]
            for item in initialize:
                getattr(self.B0, item)()


class ShearAlfvenHarmonic(sopp.ShearAlfvenHarmonic, ShearAlfvenWave):
    r"""
    Class representing a single harmonic Shear Alfvén Wave (SAW) in a given
    equilibrium magnetic field.

    This class initializes a Shear Alfvén Wave with a scalar potential of the form:

    .. math::
        \Phi(s, \theta, \zeta, t) = \hat{\Phi}(s) \sin(m \theta - n \zeta +
        \omega t + \text{phase}),

    where :math:`\hat{\Phi}(s)` is a radial profile, :math:`m` is the poloidal
    mode number, :math:`n` is the toroidal mode number,
    :math:`\omega` is the frequency, and `phase` is the phase shift. The vector
    potential `\alpha` is determined by the ideal
    Ohm's law (i.e., zero electric field along the field line). This
    representation is used to describe SAWs propagating in
    an equilibrium magnetic field :math:`B_0`.

    Attributes
    ----------
    Phihat_value_or_tuple : Union[float, Tuple[List[float], List[float]]]
        The radial profile of the scalar potential `\hat{\Phi}(s)`. It can be either:
        - A constant value (float) that represents a uniform `\hat{\Phi}`.
        - A tuple of two lists: `s_values` and `Phihat_values`, defining a
          varying profile.
    Phim : int
        Poloidal mode number `m`.
    Phin : int
        Toroidal mode number `n`.
    omega : float
        Frequency of the harmonic wave.
    phi_0 : float
        Phase of the harmonic wave.
    """

    def __init__(
        self,
        Phihat_value_or_tuple,
        Phim: int,
        Phin: int,
        omega: float,
        phase: float,
        B0: sopp.BoozerMagneticField,
    ):
        r"""
        Initialize a single harmonic Shear Alfvén Wave (SAW) in a given
        equilibrium magnetic field.

        Parameters
        ----------
        Phihat_value_or_tuple : Union[float, int, Tuple[List[float], List[float]]]
            The radial profile of the scalar potential `\hat{\Phi}(s)`.
            It can be either:
            - A constant value (float or int) that represents a uniform `\hat{\Phi}`.
            - A tuple of two lists: `s_values` and `Phihat_values`, defining a
              varying profile.
        Phim : int
            Poloidal mode number `m`.
        Phin : int
            Toroidal mode number `n`.
        omega : float
            Frequency of the harmonic wave.
        phase : float
            Phase of the harmonic wave.
        B0 : BoozerMagneticField
            Instance of a magnetic field in Boozer coordinates that provides
            the equilibrium field `B_0`.

        Raises
        ------
        TypeError
            If `B0` is not an instance of `BoozerMagneticField`.
            If `Phihat_value_or_tuple` is not a float, int, or a tuple of lists.
            If the tuple does not contain lists of floats.
        """

        # Validate B0 type
        if not isinstance(B0, sopp.BoozerMagneticField):
            raise TypeError("B0 must be an instance of BoozerMagneticField.")

        # Determine how to initialize Phihat
        if isinstance(Phihat_value_or_tuple, tuple):
            if len(Phihat_value_or_tuple) != 2:
                raise TypeError(
                    "Phihat_value_or_tuple must be a tuple of two lists: "
                    "(s_values, Phihat_values)."
                )

            s_vals, Phihat_vals = Phihat_value_or_tuple

            # Ensure both s_vals and Phihat_vals are lists of floats
            if not (
                all(isinstance(x, float) for x in s_vals)
                and all(isinstance(x, float) for x in Phihat_vals)
            ):
                raise TypeError("s_values and Phihat_values must be lists of floats.")

            indices = np.argsort(s_vals)
            Phihat_vals = [Phihat_vals[i] for i in indices]
            s_vals = [s_vals[i] for i in indices]

            if s_vals[0] < 0 or s_vals[-1] > 1:
                raise ValueError("s_values must be in the range [0, 1].")

            # Add on s = 0 boundary condition
            if s_vals[0] > 0:
                if Phim == 0:
                    s_vals.insert(0, 0)
                    Phihat_vals.insert(0, Phihat_vals[0])
                else:
                    s_vals.insert(0, 0)
                    Phihat_vals.insert(0, 0)

            phihat_object = sopp.Phihat(s_vals, Phihat_vals)
        else:
            # Try to convert Phihat_value_or_tuple to a float if possible
            try:
                Phihat_value = float(Phihat_value_or_tuple)
                # If Phihat_value_or_tuple can be converted to a float, use it
                # as a constant value
                phihat_object = sopp.Phihat([0, 1], [Phihat_value, Phihat_value])
            except (TypeError, ValueError) as err:
                raise TypeError(
                    "Phihat_value_or_tuple must be either a float, an int, or "
                    "a tuple of (s_values, Phihat_values)."
                ) from err

        # Call the constructor of the base C++ class
        sopp.ShearAlfvenHarmonic.__init__(
            self, phihat_object, Phim, Phin, omega, phase, B0
        )
        ShearAlfvenWave.__init__(self, B0, omega)

    def get_energy(self, grid_factor=10):
        r"""
        Calculates total electromagnetic energy of the perturbed field.

        The perturbed electric and magnetic fields are given by:

        .. math::

            \delta\mathbf{E} = -\nabla\varphi
            - \frac{\partial \alpha \mathbf{B}_0}{\partial t}

            \delta\mathbf{B} = \nabla \times (\alpha \mathbf{B}_0)

        The method evaluates components of the wave field in Boozer coordinates:

        .. math::

            \delta B^s &= \frac{1}{\sqrt{g}}\left(
            G\frac{\partial\alpha}{\partial\theta}
            - I\frac{\partial \alpha}{\partial \zeta}\right)

            \delta B^\theta &= \frac{1}{\sqrt{g}}\left(
            K\psi_0\frac{\partial \alpha}{\partial\zeta}
            - G\frac{\partial\alpha}{\partial s}\right)
            - \frac{\alpha}{\sqrt{g}}\frac{dG}{ds}
            + \frac{\alpha\psi_0}{\sqrt{g}}\frac{\partial K}{\partial \zeta}

            \delta B^\zeta &= \frac{1}{\sqrt{g}}\left(
            I\frac{\partial \alpha}{\partial s}
            - K\frac{\partial \alpha \psi_0}{\partial \theta}\right)
            + \frac{\alpha}{\sqrt{g}}\frac{dI}{ds}
            - \frac{\alpha\psi_0}{\sqrt{g}}\frac{\partial K}{\partial\theta}

            \delta E_s &= -\frac{\partial\varphi}{\partial s}
            - K\psi_0\frac{\partial \alpha}{\partial t}

            \delta E_\theta &= -\frac{\partial\varphi}{\partial \theta}
            - I\frac{\partial \alpha}{\partial t}

            \delta E_\zeta &= -\frac{\partial\varphi}{\partial \zeta}
            - G\frac{\partial\alpha}{\partial t}

        and numerically integrates the field energy density:

        .. math::

            \delta u = \frac{1}{2}\left(\varepsilon_0 g^{ij}\delta E_i \delta E_j
            + \frac{1}{\mu_0}g_{ij}\delta B^i \delta B^j\right)

        Parameters
        ----------
        grid_factor : int, optional
            Controls the grid resolution for numerical integration. The grid
            resolution is set to grid_factor times the maximum poloidal and
            toroidal Fourier harmonic numbers in the superposition. Default is 10.

        Returns
        -------
        tuple
            A tuple containing:

            - dE_energy (float): Electric energy of the wave perturbation in Joules
            - dB_energy (float): Magnetic energy of the wave perturbation in Joules
            - B0_energy (float): Magnetic energy of the background field in Joules
        """
        saws = ShearAlfvenWavesSuperposition([self])
        dE_energy, dB_energy, B0_energy = saws.get_energy(grid_factor)
        return dE_energy, dB_energy, B0_energy


class ShearAlfvenWavesSuperposition(
    sopp.ShearAlfvenWavesSuperposition, ShearAlfvenWave
):
    r"""
    Class representing a superposition of multiple Shear Alfvén Waves (SAWs).
    This class models the superposition of :class:`ShearAlfvenHarmonic` instances
    in the same equilibrium field, combining their scalar potential `Phi`,
    vector potential `alpha`, and their respective derivatives to represent a more
    complex wave structure in the equilibrium field `B0`.

    The superposition of waves is initialized with a base wave, which defines
    the reference
    equilibrium field `B0` for all subsequent waves added to the superposition.
    All added waves
    must have the same `B0` field.

    See Paul et al., JPP (2023; 89(5):905890515.
    doi:10.1017/S0022377823001095) for more details.

    Parameters
    ----------
    SAWs : list of ShearAlfvenHarmonic
        A list of ShearAlfvenHarmonic objects to be superposed. The first wave
        in the list is used
        as the base wave and defines the reference `B0` field for the
        superposition. All other
        waves in the list must have the same `B0`.

    Raises
    ------
    TypeError
        If `SAWs` is not a list of `ShearAlfvenWave` objects.
        If the base wave is not provided or if the waves have different `B0` fields.
    ValueError
        If any wave is not a `ShearAlfvenHarmonic`.

    Examples
    --------
    .. code-block:: python

        # Create a list of ShearAlfvenWave objects
        wave1 = ShearAlfvenHarmonic(...)  # Initialize a harmonic wave
        wave2 = ShearAlfvenHarmonic(...)  # Initialize another harmonic wave

        # Create a superposition of these waves
        superposition = ShearAlfvenWavesSuperposition([wave1, wave2])

        # Set points for evaluation
        points = ...  # Define points (s, theta, zeta, time)
        superposition.set_points(points)

    """

    def __init__(self, SAWs: list):
        if not isinstance(SAWs, list) or not all(
            isinstance(SAW, sopp.ShearAlfvenWave) for SAW in SAWs
        ):
            raise TypeError("SAWs must be a list of ShearAlfvenWave objects.")

        if len(SAWs) == 0:
            raise ValueError("At least one ShearAlfvenWave object must be provided.")

        # Initialize the base C++ class with the first wave as the base wave
        sopp.ShearAlfvenWavesSuperposition.__init__(self, SAWs[0])
        ShearAlfvenWave.__init__(self, SAWs[0].B0, SAWs[0].omega)

        # Add subsequent waves to the superposition
        for SAW in SAWs[1:]:
            self.add_wave(SAW)

    @classmethod
    def from_ae3d(
        cls,
        eigenvector: AE3DEigenvector,
        B0: BoozerMagneticField,
        max_dB_normal_by_B0: Union[float, None] = None,
        minor_radius_meters=1.7,
        phase=0.0,
    ):
        """
        Converts AE3DEigenvector harmonics into ShearAlfvenHarmonics submerged
        in the given BoozerMagneticField.

        Args:
            eigenvector (AE3DEigenvector): The eigenvector object containing
                harmonics from the AE3D simulation.
            B0 (BoozerMagneticField): The background magnetic field
                (computed separately), in Tesla
            max_dB_normal_by_B0 (float | None): Desired ratio of maximum
                normal B from SAW mode over B0 field. If None, no rescaling is
                done so the mode is returned with the ampitudes as they appear in
                AE3DEigenvector.
            minor_radius_meters (float): Stellarator's minor radius, in meters.
                User can get this from VMEC wout equilibrium

        Returns:
            ShearAlfvenWavesSuperposition: A superposition of ShearAlfvenHarmonics.
        """
        harmonic_list = []
        m_list = []
        n_list = []
        s_list = []
        omega = np.sqrt(eigenvector.eigenvalue) * 1000

        if eigenvector.eigenvalue <= 0:
            raise ValueError("The eigenvalue must be positive to compute omega.")

        for harmonic in eigenvector.harmonics:
            sbump = eigenvector.s_coords
            bump = harmonic.amplitudes

            sah = ShearAlfvenHarmonic(
                Phihat_value_or_tuple=(sbump, bump),
                Phim=harmonic.m,
                Phin=harmonic.n,
                omega=omega,
                phase=phase,
                B0=B0,
            )
            m_list.append(harmonic.m)
            n_list.append(harmonic.n)
            s_list += list(sbump)
            harmonic_list.append(sah)
        # start with arbitrary magnitude SAW, then rescale it:
        unscaled_SAW = ShearAlfvenWavesSuperposition(harmonic_list)
        if max_dB_normal_by_B0 is None:
            return unscaled_SAW
        # Make radial grid that captures all unique radial values for all harmonic:
        s_unique = sorted(set(s_list))
        # Make angle grids that resolve maxima of highest harmonics
        thetas = np.linspace(0, 2 * np.pi, 5 * np.max(np.abs(m_list)))
        zetas = np.linspace(0, 2 * np.pi, 5 * np.max(np.abs(n_list)))
        # Create 3D mesh grids:
        thetas2d, zetas2d, s2d = np.meshgrid(thetas, zetas, s_unique, indexing="ij")
        points = np.zeros((len(thetas2d.flatten()), 4))  # s theta zeta time
        points[:, 0] = s2d.flatten()  # s values
        points[:, 1] = thetas2d.flatten()  # theta values
        points[:, 2] = zetas2d.flatten()  # zeta values
        unscaled_SAW.set_points(points)
        G = unscaled_SAW.B0.G()
        iota = unscaled_SAW.B0.iota()
        I = unscaled_SAW.B0.I()
        Bpsi_default = (
            1
            / ((iota * I + G) * minor_radius_meters)
            * (G * unscaled_SAW.dalphadtheta() - I * unscaled_SAW.dalphadzeta())
        )
        max_index = np.argmax(np.abs(Bpsi_default))
        _max_s, _max_theta, _max_zeta = (
            points[max_index, 0],
            points[max_index, 1],
            points[max_index, 2],
        )

        Phihat_scale_factor = max_dB_normal_by_B0 / np.max(np.abs(Bpsi_default))

        # Having determined the scale factor, initialize harmonics with
        # corrected amplitudes:
        harmonic_list = []
        for harmonic in eigenvector.harmonics:
            sbump = eigenvector.s_coords
            bump = harmonic.amplitudes
            sah = ShearAlfvenHarmonic(
                Phihat_value_or_tuple=(sbump, bump * Phihat_scale_factor),
                Phim=harmonic.m,
                Phin=harmonic.n,
                omega=omega,
                phase=phase,
                B0=B0,
            )
            harmonic_list.append(sah)
        return ShearAlfvenWavesSuperposition(harmonic_list)

    @classmethod
    def from_far3d(
        cls,
        eigenvector: FAR3DEigenvector,
        B0: BoozerMagneticField,
        max_dB_normal_by_B0: Union[float, None] = None,
        minor_radius_meters=1.7,
        phase=0.0,
    ):
        """
        Converts FAR3DEigenvector harmonics into ShearAlfvenHarmonics submerged
        in the given BoozerMagneticField.

        Args:
            eigenvector (FAR3DEigenvector): The eigenvector object containing
                harmonics from the FAR3D initial value solver.
            B0 (BoozerMagneticField): The background magnetic field
                (computed separately), in Tesla
            max_dB_normal_by_B0 (float | None): Desired ratio of maximum
            normal B from SAW mode over B0 field. If None, no rescaling is
            done.
            minor_radius_meters (float): Stellarator's minor radius, in meters.
                User can get this from VMEC wout equilibrium
            phase (float): Phase to add to the harmonic. Adds to an existing
                phase rather than replacing it.

        Returns:
            ShearAlfvenWavesSuperposition: A superposition of ShearAlfvenHarmonics.
        """
        harmonic_list = []
        m_list = []
        n_list = []
        s_list = []
        omega = np.sqrt(eigenvector.eigenvalue) * 1000

        if eigenvector.eigenvalue <= 0:
            raise ValueError("The eigenvalue must be positive to compute omega.")

        for harmonic in eigenvector.harmonics:
            sbump = eigenvector.s_coords
            bump = harmonic.amplitudes

            sah = ShearAlfvenHarmonic(
                Phihat_value_or_tuple=(sbump, bump),
                Phim=harmonic.m,
                Phin=harmonic.n,
                omega=omega,
                phase=harmonic.phase + phase,
                B0=B0,
            )
            m_list.append(harmonic.m)
            n_list.append(harmonic.n)
            s_list += list(sbump)
            harmonic_list.append(sah)
        # start with arbitrary magnitude SAW, then rescale it:
        unscaled_SAW = ShearAlfvenWavesSuperposition(harmonic_list)
        if max_dB_normal_by_B0 is None:
            return unscaled_SAW
        # Make radial grid that captures all unique radial values for all harmonic:
        s_unique = sorted(set(s_list))
        # Make angle grids that resolve maxima of highest harmonics
        thetas = np.linspace(0, 2 * np.pi, 5 * np.max(np.abs(m_list)))
        zetas = np.linspace(0, 2 * np.pi, 5 * np.max(np.abs(n_list)))
        # Create 3D mesh grids:
        thetas2d, zetas2d, s2d = np.meshgrid(thetas, zetas, s_unique, indexing="ij")
        points = np.zeros((len(thetas2d.flatten()), 4))  # s theta zeta time
        points[:, 0] = s2d.flatten()  # s values
        points[:, 1] = thetas2d.flatten()  # theta values
        points[:, 2] = zetas2d.flatten()  # zeta values
        unscaled_SAW.set_points(points)
        G = unscaled_SAW.B0.G()
        iota = unscaled_SAW.B0.iota()
        I = unscaled_SAW.B0.I()
        Bpsi_default = (
            1
            / ((iota * I + G) * minor_radius_meters)
            * (G * unscaled_SAW.dalphadtheta() - I * unscaled_SAW.dalphadzeta())
        )
        max_index = np.argmax(np.abs(Bpsi_default))
        _max_s, _max_theta, _max_zeta = (
            points[max_index, 0],
            points[max_index, 1],
            points[max_index, 2],
        )

        Phihat_scale_factor = max_dB_normal_by_B0 / np.max(np.abs(Bpsi_default))

        # Having determined the scale factor, initialize harmonics with
        # corrected amplitudes:
        harmonic_list = []
        for harmonic in eigenvector.harmonics:
            sbump = eigenvector.s_coords
            bump = harmonic.amplitudes
            sah = ShearAlfvenHarmonic(
                Phihat_value_or_tuple=(sbump, bump * Phihat_scale_factor),
                Phim=harmonic.m,
                Phin=harmonic.n,
                omega=omega,
                phase=harmonic.phase,
                B0=B0,
            )
            harmonic_list.append(sah)
        return ShearAlfvenWavesSuperposition(harmonic_list)

    def __getitem__(self, index):
        """
        Get a wave by index.

        Args:
            index : (int) Index of the ShearAlfvenHarmonic to retrieve

        Returns:
            ShearAlfvenHarmonic at the specified index
        """
        return self.get_wave(index)

    def __len__(self):
        """
        Get the number of waves in the superposition.

        Returns:
            Number of waves in the superposition
        """
        return self.size()

    def get_energy(self, grid_factor=10, sgridpoints=100):
        r"""
        Calculates total electromagnetic energy of the perturbed field.

        The perturbed electric and magnetic fields are given by:

        .. math::

            \delta\mathbf{E} = -\nabla\varphi
            - \frac{\partial \alpha \mathbf{B}_0}{\partial t}

            \delta\mathbf{B} = \nabla \times (\alpha \mathbf{B}_0)

        The method evaluates components of the wave field in Boozer coordinates:

        .. math::

            \delta B^s &= \frac{1}{\sqrt{g}}\left(
            G\frac{\partial\alpha}{\partial\theta}
            - I\frac{\partial \alpha}{\partial \zeta}\right)

            \delta B^\theta &= \frac{1}{\sqrt{g}}\left(
            K\psi_0\frac{\partial \alpha}{\partial\zeta}
            - G\frac{\partial\alpha}{\partial s}\right)
            - \frac{\alpha}{\sqrt{g}}\frac{dG}{ds}
            + \frac{\alpha\psi_0}{\sqrt{g}}\frac{\partial K}{\partial \zeta}

            \delta B^\zeta &= \frac{1}{\sqrt{g}}\left(
            I\frac{\partial \alpha}{\partial s}
            - K\frac{\partial \alpha \psi_0}{\partial \theta}\right)
            + \frac{\alpha}{\sqrt{g}}\frac{dI}{ds}
            - \frac{\alpha\psi_0}{\sqrt{g}}\frac{\partial K}{\partial\theta}

            \delta E_s &= -\frac{\partial\varphi}{\partial s}
            - K\psi_0\frac{\partial \alpha}{\partial t}

            \delta E_\theta &= -\frac{\partial\varphi}{\partial \theta}
            - I\frac{\partial \alpha}{\partial t}

            \delta E_\zeta &= -\frac{\partial\varphi}{\partial \zeta}
            - G\frac{\partial\alpha}{\partial t}

        and numerically integrates the field energy density:

        .. math::

            \delta u = \frac{1}{2}\left(\varepsilon_0 g^{ij}\delta E_i \delta E_j
            + \frac{1}{\mu_0}g_{ij}\delta B^i \delta B^j\right)

        Parameters
        ----------
        grid_factor : int, optional
            Controls the grid resolution for numerical integration. The grid
            resolution is set to grid_factor times the maximum poloidal and
            toroidal Fourier harmonic numbers in the superposition. Default is 10.

        Returns
        -------
        tuple
            A tuple containing:

            - dE_energy (float): Electric energy of the wave perturbation in Joules
            - dB_energy (float): Magnetic energy of the wave perturbation in Joules
            - B0_energy (float): Magnetic energy of the background field in Joules
        """
        if not isinstance(grid_factor, int) or grid_factor < 1:
            raise ValueError("grid_factor must be an integer >= 1")
        # remember the points if set already, to set back after calculation
        original_points = self.get_points()

        max_Phim = 0
        max_Phin = 0

        for sah in self:
            max_Phim = max(np.abs(sah.Phim), max_Phim)
            max_Phin = max(np.abs(sah.Phin), max_Phin)

        s_list = np.linspace(0, 1, sgridpoints + 1)[1:]

        for s in s_list:
            assert 0.0 <= s <= 1, f"{s=} is out of allowed [0,1] range"

        # ignore the point on axis since metric is singular there
        if s_list[0] == 0.0:
            s_list[0] = s_list[1] / 2

        theta_list = np.linspace(
            0, 2 * np.pi, num=grid_factor * max(1, max_Phim), endpoint=False
        )
        zeta_list = np.linspace(
            0, 2 * np.pi, num=grid_factor * max(1, max_Phin), endpoint=False
        )

        thetas2d, zetas2d, s2d = np.meshgrid(
            theta_list, zeta_list, s_list, indexing="ij"
        )

        points = np.zeros((len(thetas2d.flatten()), 4))  # s theta zeta time
        points[:, 0] = s2d.flatten()
        points[:, 1] = thetas2d.flatten()
        points[:, 2] = zetas2d.flatten()

        self.set_points(points)
        g_cov = self.B0.get_covariant_metric()

        # Magnetic perturbation energy:
        dads = self.dalphadpsi()[:, 0] * self.B0.psi0
        dadth = self.dalphadtheta()[:, 0]
        dadzt = self.dalphadzeta()[:, 0]
        a = self.alpha()[:, 0]
        G = self.B0.G()[:, 0]
        I = self.B0.I()[:, 0]
        K = self.B0.K()[:, 0]
        dGds = self.B0.dGds()[:, 0]
        dKdzt = self.B0.dKdzeta()[:, 0]
        dKdth = self.B0.dKdtheta()[:, 0]
        dIds = self.B0.dIds()[:, 0]
        iota = self.B0.iota()[:, 0]
        B = self.B0.modB()[:, 0]
        det = B * B / (iota * I + G) / self.B0.psi0

        Bs = det * (G * dadth - I * dadzt)
        Bth = det * (
            self.B0.psi0 * K * dadzt - G * dads - a * dGds + a * self.B0.psi0 * dKdzt
        )
        Bzt = det * (
            I * dads - self.B0.psi0 * K * dadth + a * dIds - a * self.B0.psi0 * dKdth
        )

        B2 = (
            g_cov.ss * Bs * Bs
            + g_cov.tt * Bth * Bth
            + g_cov.zz * Bzt * Bzt
            + 2 * g_cov.st * Bs * Bth
            + 2 * g_cov.sz * Bs * Bzt
            + 2 * g_cov.tz * Bth * Bzt
        )

        B2_grid = (np.abs(det) * B2).reshape(
            len(theta_list), len(zeta_list), len(s_list)
        )

        integrated_s = np.trapz(B2_grid, x=s_list, axis=2)
        integrated_sz = np.trapz(integrated_s, x=zeta_list, axis=1)
        dB_energy = np.trapz(integrated_sz, x=theta_list) / MU0 / 2

        # Electric perturbation energy:
        dphids = self.dPhidpsi()[:, 0] * self.B0.psi0
        dphidth = self.dPhidtheta()[:, 0]
        dphidzt = self.dPhidzeta()[:, 0]
        dadt = self.alphadot()[:, 0]
        Es = -dphids - K * self.B0.psi0 * dadt
        Eth = -dphidth - I * dadt
        Ezt = -dphidzt - G * dadt

        g_cont = g_cov.to_contravariant()

        E2 = (
            g_cont.ss * Es * Es
            + g_cont.tt * Eth * Eth
            + g_cont.zz * Ezt * Ezt
            + 2 * g_cont.st * Es * Eth
            + 2 * g_cont.sz * Es * Ezt
            + 2 * g_cont.tz * Eth * Ezt
        )

        E2_grid = (np.abs(det) * E2).reshape(
            len(theta_list), len(zeta_list), len(s_list)
        )

        integrated_E_s = np.trapz(E2_grid, x=s_list, axis=2)
        integrated_E_sz = np.trapz(integrated_E_s, x=zeta_list, axis=1)
        dE_energy = EPS0 * np.trapz(integrated_E_sz, x=theta_list) / 2

        # Equilibrium magnetic energy:
        B02 = (
            g_cont.ss * (self.B0.psi0 * K) ** 2
            + g_cont.tt * I * I
            + g_cont.zz * G * G
            + 2 * g_cont.st * (self.B0.psi0 * K) * I
            + 2 * g_cont.sz * (self.B0.psi0 * K) * G
            + 2 * g_cont.tz * I * G
        )

        B02_grid = (np.abs(det) * B02).reshape(
            len(theta_list), len(zeta_list), len(s_list)
        )
        integrated_B0_s = np.trapz(B02_grid, x=s_list, axis=2)
        integrated_B0_sz = np.trapz(integrated_B0_s, x=zeta_list, axis=1)
        B0_energy = np.trapz(integrated_B0_sz, x=theta_list) / MU0 / 2

        self.set_points(original_points)

        return dE_energy, dB_energy, B0_energy


class InterpolatedShearAlfvenWave(ShearAlfvenWave):
    r"""
    This class takes an existing :class:`ShearAlfvenWave` and interpolates it on a
    regular grid in :math:`s, \theta, \zeta` using the C++
    :class:`RegularGridInterpolant3D`.
    The field is represented as a piecewise polynomial in (s, theta, zeta) of a given
    degree. The number of nodes in each direction are defined by ns_interp,
    ntheta_interp, and nzeta_interp. All quantities are time-dependent; the
    interpolation is performed at t=0. When evaluating at t≠0, the underlying field
    is used to get the correct time dependence. Note that waves generally break
    rotational symmetry (nfp) and stellarator symmetry (stellsym), so these are not
    automatically inherited from the field. It is recommended to use this field
    representation in the tracing loop due to its speed in comparison to computing
    the wave quantities directly from the underlying :class:`ShearAlfvenWave`.

    This class implements all the methods of :class:`ShearAlfvenWave`:
    `Phi`, `dPhidpsi`, `Phidot`, `dPhidtheta`, `dPhidzeta`, `alpha`, `alphadot`,
    `dalphadtheta`, `dalphadpsi`, and `dalphadzeta`.
    """

    def __init__(
        self,
        field,
        degree,
        srange=None,
        thetarange=None,
        zetarange=None,
        ns_interp=48,
        ntheta_interp=48,
        nzeta_interp=48,
        extrapolate=True,
        initialize=None,
        comm=None,
    ):
        r"""
        Args:
            field: the underlying :class:`ShearAlfvenWave` to be interpolated.
            degree: the degree of the piecewise polynomial interpolant.
            ns_interp: number of grid points in the :math:`s` direction.
            ntheta_interp: number of grid points in the :math:`\theta` direction.
            nzeta_interp: number of grid points in the :math:`\zeta` direction.
            srange: a 3-tuple of the form ``(smin, smax, ns)``. This means that
                the interval ``[smin, smax]`` is split into ``ns`` many
                subintervals.
            thetarange: a 3-tuple of the form ``(thetamin, thetamax, ntheta)``.
                thetamin must be >= 0, and thetamax must be <=2*pi.
            zetarange: a 3-tuple of the form ``(zetamin, zetamax, nzeta)``.
                zetamin must be >= 0, and zetamax must be <=2*pi.
            extrapolate: whether to extrapolate the field when evaluated outside
                         the integration domain or to throw an error.
            initialize: A list of strings, each of which is the name of a
                field quantity, e.g., `Phi`, to be initialized when the
                interpolant is created. By default, all quantities are initialized.
            comm: MPI communicator for parallelizing grid evaluation.
                If None, evaluation is done serially.
        """
        if not isinstance(field, ShearAlfvenWave):
            raise TypeError("field must be an instance of ShearAlfvenWave.")

        # Determine field_type from the underlying B0 field
        field_type = None
        if hasattr(field, "B0") and hasattr(field.B0, "field_type"):
            field_type = field.B0.field_type.lower()
            assert field_type in ["", "vac", "nok"]

        # Set default initialize list based on field_type if not provided
        if initialize is None:
            if field_type == "nok":
                # For noK: dPhidpsi, dPhidtheta, dPhidzeta, alpha, alphadot,
                #          dalphadpsi, dalphadtheta, dalphadzeta
                initialize = [
                    "dPhidpsi",
                    "dPhidtheta",
                    "dPhidzeta",
                    "alpha",
                    "alphadot",
                    "dalphadpsi",
                    "dalphadtheta",
                    "dalphadzeta",
                ]
            elif field_type == "vac":
                # For vacuum: dPhidpsi, dPhidtheta, dPhidzeta, alphadot,
                #             dalphadpsi, dalphadtheta
                initialize = [
                    "dPhidpsi",
                    "dPhidtheta",
                    "dPhidzeta",
                    "alphadot",
                    "dalphadpsi",
                    "dalphadtheta",
                ]
            else:
                # Default: initialize all quantities (for general field_type)
                initialize = [
                    "Phi",
                    "dPhidpsi",
                    "Phidot",
                    "dPhidtheta",
                    "dPhidzeta",
                    "alpha",
                    "alphadot",
                    "dalphadtheta",
                    "dalphadpsi",
                    "dalphadzeta",
                ]

        self.field = field
        self.field_type = field_type
        self.degree = degree
        self.extrapolate = extrapolate
        self.comm = comm

        # Set up grid ranges
        # Note: Waves generally break rotational and stellarator symmetry,
        # so use explicit ranges rather than relying on symmetry.
        if srange is None:
            srange = (0, 1, ns_interp)
        if thetarange is None:
            # Default to full theta range [0, 2*pi] since waves break stellarator
            # symmetry
            thetarange = (0, 2 * np.pi, ntheta_interp)
        if zetarange is None:
            # Default to full zeta range for one field period
            zetarange = (0, 2 * np.pi, nzeta_interp)

        self.srange = srange
        self.thetarange = thetarange
        self.zetarange = zetarange

        # Validate ranges
        if np.any(np.asarray(thetarange[0:2]) < 0) or np.any(
            np.asarray(thetarange[0:2]) > 2 * np.pi
        ):
            raise ValueError("thetamin and thetamax must be in [0,2*pi]")
        if np.any(np.asarray(zetarange[0:2]) < 0) or np.any(
            np.asarray(zetarange[0:2]) > 2 * np.pi
        ):
            raise ValueError("zetamin and zetamax must be in [0,2*pi]")

        # Initialize the base class
        ShearAlfvenWave.__init__(self, field.B0, field.omega)

        # Create interpolation rule
        self.rule = sopp.UniformInterpolationRule(degree)
        self.B0 = field.B0
        # Initialize interpolants dictionary
        # For each quantity, we store two interpolants: one for the cos(omega*t)
        # coefficient and one for the sin(omega*t) coefficient
        self._interpolants_cos = {}  # Coefficient of cos(omega*t)
        self._interpolants_sin = {}  # Coefficient of sin(omega*t)
        self._status = {}

        # Try to extract omega from the field if it's a ShearAlfvenHarmonic
        self.omega = None
        if hasattr(field, "omega"):
            self.omega = field.omega
        elif (
            hasattr(field, "waves")
            and len(field.waves) > 0
            and hasattr(field.waves[0], "omega")
        ):
            # For superposition, use omega from first wave (assuming same omega)
            self.omega = field.waves[0].omega

        # Initialize requested quantities
        if initialize:
            for item in initialize:
                self._initialize_quantity(item, srange, thetarange, zetarange)

    def _initialize_quantity(self, quantity_name, srange, thetarange, zetarange):
        """Initialize interpolation for a specific quantity using C++ 3D interpolant.

        Different quantities have different parities with respect to the phase:
        - sin-like: Phi, dPhidpsi, alpha, dalphadpsi
          Q = Q_spatial * sin(m*theta - n*zeta + omega*t + phase)
          At t=0: Q(0) = Q_spatial * sin(phase) = Q_sin_coeff
          At t=pi/(2*omega): Q(pi/(2*omega)) = Q_spatial * cos(phase) = Q_cos_coeff

        - cos-like: Phidot, dPhidtheta, dPhidzeta, alphadot, dalphadtheta, dalphadzeta
          Q = Q_spatial * cos(m*theta - n*zeta + omega*t + phase)
          At t=0: Q(0) = Q_spatial * cos(phase) = Q_cos_coeff
          At t=pi/(2*omega): Q(pi/(2*omega)) = -Q_spatial * sin(phase) = -Q_sin_coeff

        The time dependence is restored using:
        Q(t) = Q_cos_coeff * cos(omega*t) + Q_sin_coeff * sin(omega*t)
        """
        if quantity_name in self._status and self._status[quantity_name]:
            return

        # Determine parity: sin-like or cos-like quantities
        sin_like_quantities = ["Phi", "dPhidpsi", "alpha", "dalphadpsi"]
        is_sin_like = quantity_name in sin_like_quantities

        # Store original points
        original_points = self.field.get_points()

        # Create two interpolants: one for cos coefficient, one for sin
        interpolant_cos = sopp.RegularGridInterpolant3D(
            self.rule, srange, thetarange, zetarange, 1, self.extrapolate
        )
        interpolant_sin = sopp.RegularGridInterpolant3D(
            self.rule, srange, thetarange, zetarange, 1, self.extrapolate
        )

        t_quarter_period = np.pi / (2.0 * self.omega)

        def _fbatch(time_value, sign=1.0):
            def fbatch(s_vals, theta_vals, zeta_vals):
                pts = np.column_stack(
                    (
                        np.asarray(s_vals),
                        np.asarray(theta_vals),
                        np.asarray(zeta_vals),
                        np.full_like(s_vals, time_value, dtype=float),
                    )
                )
                self.field.set_points(pts)
                values = getattr(self.field, quantity_name)()[:, 0]
                return (sign * np.ascontiguousarray(values)).ravel()

            return fbatch

        if is_sin_like:
            f_sin = _fbatch(0.0, 1.0)
            f_cos = _fbatch(t_quarter_period, 1.0)
        else:
            f_cos = _fbatch(0.0, 1.0)
            f_sin = _fbatch(t_quarter_period, -1.0)

        if self.comm is not None:
            comm_handle = self.comm.py2f()
            interpolant_sin.interpolate_batch(f_sin, comm_handle)
            interpolant_cos.interpolate_batch(f_cos, comm_handle)
        else:
            interpolant_sin.interpolate_batch(f_sin)
            interpolant_cos.interpolate_batch(f_cos)

        # Store both interpolants
        self._interpolants_cos[quantity_name] = interpolant_cos
        self._interpolants_sin[quantity_name] = interpolant_sin

        self._status[quantity_name] = True

        # Restore original points
        self.field.set_points(original_points)

    def _map_points_to_interpolation_range(self, stz):
        """Map points to fall within [0, 2*pi] using periodic wrapping.

        Since waves break rotational and stellarator symmetries, we only do
        periodic wrapping to map theta and zeta to [0, 2*pi], which is their
        natural periodic range.

        Args:
            stz: Array of shape (npoints, 3) with (s, theta, zeta) coordinates.

        Returns:
            stz_mapped: Array of shape (npoints, 3) with mapped coordinates.
        """
        stz_mapped = stz.copy()
        theta = stz[:, 1]
        zeta = stz[:, 2]

        # Map theta to [0, 2*pi] using periodic wrapping
        theta = np.mod(theta, 2 * np.pi)
        stz_mapped[:, 1] = theta

        # Map zeta to [0, 2*pi] using periodic wrapping
        zeta = np.mod(zeta, 2 * np.pi)
        stz_mapped[:, 2] = zeta

        return stz_mapped

    def _evaluate_interpolant(self, quantity_name):
        """Evaluate an interpolated quantity at the current points.

        Uses angle sum identities to restore time dependence, accounting for parity:
        - For sin-like quantities (Phi, dPhidpsi, alpha, dalphadpsi):
          Q = Q_spatial * sin(φ + omega*t) where φ = m*theta - n*zeta + phase
          Using sin(α+β) = sin(α)cos(β) + cos(α)sin(β):
          Q(t) = Q_sin_coeff * cos(omega*t) + Q_cos_coeff * sin(omega*t)
        - For cos-like quantities (Phidot, dPhidtheta, dPhidzeta,
          alphadot, dalphadtheta, dalphadzeta):
          Q = Q_spatial * cos(φ + omega*t) where φ = m*theta - n*zeta + phase
          Using cos(α+β) = cos(α)cos(β) - sin(α)sin(β):
          Q(t) = Q_cos_coeff * cos(omega*t) - Q_sin_coeff * sin(omega*t)
        """
        if quantity_name not in self._status or not self._status[quantity_name]:
            # Need to initialize this quantity
            self._initialize_quantity(
                quantity_name, self.srange, self.thetarange, self.zetarange
            )

        # Get current points
        points = np.asarray(self.get_points())
        npoints = points.shape[0]

        # Extract s, theta, zeta coordinates and times
        stz = points[:, [0, 1, 2]]
        current_times = points[:, 3]

        # Map points to interpolation range using periodic wrapping
        stz_mapped = self._map_points_to_interpolation_range(stz)

        # Evaluate cos and sin coefficients
        result_cos = np.zeros((npoints, 1))
        result_sin = np.zeros((npoints, 1))
        self._interpolants_cos[quantity_name].evaluate_batch(stz_mapped, result_cos)
        self._interpolants_sin[quantity_name].evaluate_batch(stz_mapped, result_sin)
        # Determine parity and restore time dependence accordingly
        sin_like_quantities = ["Phi", "dPhidpsi", "alpha", "dalphadpsi"]
        is_sin_like = quantity_name in sin_like_quantities

        cos_omega_t = np.cos(self.omega * current_times)[:, np.newaxis]
        sin_omega_t = np.sin(self.omega * current_times)[:, np.newaxis]

        if is_sin_like:
            # For sin-like: Q(t) = Q_sin_coeff * cos(omega*t) +
            # Q_cos_coeff * sin(omega*t)
            # where Q_sin_coeff = Q_spatial * sin(φ) and
            # Q_cos_coeff = Q_spatial * cos(φ)
            result = result_sin * cos_omega_t + result_cos * sin_omega_t
        else:
            # For cos-like: Q(t) = Q_cos_coeff * cos(omega*t) -
            # Q_sin_coeff * sin(omega*t)
            # where Q_cos_coeff = Q_spatial * cos(φ) and
            # Q_sin_coeff = Q_spatial * sin(φ)
            result = result_cos * cos_omega_t - result_sin * sin_omega_t

        return result

    def _Phi_impl(self, Phi):
        """Implementation of Phi using interpolation."""
        Phi[:] = self._evaluate_interpolant("Phi")

    def _dPhidpsi_impl(self, dPhidpsi):
        """Implementation of dPhidpsi using interpolation."""
        dPhidpsi[:] = self._evaluate_interpolant("dPhidpsi")

    def _Phidot_impl(self, Phidot):
        """Implementation of Phidot using interpolation."""
        Phidot[:] = self._evaluate_interpolant("Phidot")

    def _dPhidtheta_impl(self, dPhidtheta):
        """Implementation of dPhidtheta using interpolation."""
        dPhidtheta[:] = self._evaluate_interpolant("dPhidtheta")

    def _dPhidzeta_impl(self, dPhidzeta):
        """Implementation of dPhidzeta using interpolation."""
        dPhidzeta[:] = self._evaluate_interpolant("dPhidzeta")

    def _alpha_impl(self, alpha):
        """Implementation of alpha using interpolation."""
        alpha[:] = self._evaluate_interpolant("alpha")

    def _alphadot_impl(self, alphadot):
        """Implementation of alphadot using interpolation."""
        alphadot[:] = self._evaluate_interpolant("alphadot")

    def _dalphadtheta_impl(self, dalphadtheta):
        """Implementation of dalphadtheta using interpolation."""
        dalphadtheta[:] = self._evaluate_interpolant("dalphadtheta")

    def _dalphadpsi_impl(self, dalphadpsi):
        """Implementation of dalphadpsi using interpolation."""
        dalphadpsi[:] = self._evaluate_interpolant("dalphadpsi")

    def _dalphadzeta_impl(self, dalphadzeta):
        """Implementation of dalphadzeta using interpolation."""
        dalphadzeta[:] = self._evaluate_interpolant("dalphadzeta")
