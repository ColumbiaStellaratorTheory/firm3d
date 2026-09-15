Shear Alfvén Wave Field Classes
===============================

Given an equilibrium field :math:`\textbf{B}_0`, a shear Alfvén wave is modeled through the perturbed electrostatic potential, :math:`\Phi`, and parameter :math:`\alpha` defining the perturbed vector potential :math:`\delta \textbf{A} = \alpha \textbf{B}_0`. The perturbed electric and magnetic fields then satisfy:

.. math::

   \delta \textbf{E} = -\nabla \Phi - B_0 \frac{\partial \alpha}{\partial t}

   \delta \textbf{B} = \nabla \times \left(\alpha \textbf{B}_0 \right)

The parameter :math:`\alpha` is related to :math:`\Phi` given ideal Ohm's law:

.. math::

   \nabla_{||} \Phi = -B_0 \frac{\partial \alpha}{\partial t}

In the ``ShearAlfvenWave`` classes, the equilibrium field is prescribed as a ``BoozerMagneticField`` class in addition to input parameters determining :math:`\Phi`.

ShearAlfvenHarmonic
-------------------

This class initializes a Shear Alfvén Wave with a scalar potential of the form:

.. math::

   \Phi(s, \theta, \zeta, t) = \hat{\Phi}(s) \sin(m \theta - n \zeta + \omega t + \phi_0)

where :math:`\hat{\Phi}(s)` is a radial profile, :math:`m` is the poloidal mode number, :math:`n` is the toroidal mode number, :math:`\omega` is the frequency, and :math:`\phi_0` is the phase shift. The perturbed parallel vector potential parameter, :math:`\alpha`, is then determined by ideal Ohm's law. This representation is used to describe SAWs propagating in an equilibrium magnetic field :math:`\textbf{B}_0`.

See Paul et al., JPP (2023; 89(5):905890515. doi:10.1017/S0022377823001095) for more details.

Radial Profiles
---------------

The radial profile :math:`\hat{\Phi}(s)` can be specified in two ways:

**1. Constant Profile (Uniform)**
A single float value representing a uniform amplitude across all radial positions:

.. code-block:: python

   # Uniform amplitude across all s
   Phihat = 0.01
   saw = ShearAlfvenHarmonic(Phihat, Phim=2, Phin=1, omega=1.0, phase=0.0, B0=field)

**2. Varying Profile (Tabulated)**
A tuple of two lists defining the radial dependence: `(s_values, Phihat_values)`:

.. code-block:: python

   # Define varying radial profile
   s_values = [0.0, 0.3, 0.5, 0.7, 1.0]
   Phihat_values = [0.0, 0.005, 0.01, 0.005, 0.0]

   # Create wave with varying profile
   saw = ShearAlfvenHarmonic(
       (s_values, Phihat_values),
       Phim=2, Phin=1, omega=1.0, phase=0.0, B0=field
   )

.. note::
   The `s_values` must be in the range [0, 1] and will be automatically sorted.
   For non-zero poloidal mode numbers (m ≠ 0), if the profile does not already
   include a point at s = 0, one is inserted with :math:`\hat{\Phi}=0` (the
   expected boundary condition at the axis for m ≠ 0). A value explicitly
   provided at s = 0 is not overridden.

Usage Example
~~~~~~~~~~~~~

.. code-block:: python

   from firm3d.field import BoozerAnalytic, ShearAlfvenHarmonic
   import numpy as np

   # Create equilibrium field
   eq_field = BoozerAnalytic(etabar=0.1, B0=1.0, N=0, G0=1.0, psi0=0.1, iota0=0.5)

   # Define radial profile (constant amplitude)
   Phihat = 0.01

   # Create shear Alfvén wave
   saw = ShearAlfvenHarmonic(
       Phihat,  # radial profile amplitude
       Phim=2,     # poloidal mode number
       Phin=1,     # toroidal mode number
       omega=1.0,  # frequency
       phase=0.0,  # phase shift
       B0=eq_field  # equilibrium field
   )

ShearAlfvenWavesSuperposition
-----------------------------

Class representing a superposition of multiple Shear Alfvén Waves (SAWs).

This class models the superposition of multiple Shear Alfvén waves, combining their scalar potentials :math:`\Phi`, vector potential parameters :math:`\alpha`, and their respective derivatives to represent a more complex wave structure in the equilibrium field :math:`\textbf{B}_0`.

The superposition of waves is initialized with a base wave, which defines the reference equilibrium field :math:`\textbf{B}_0` for all subsequent waves added to the superposition. All added waves must have the same :math:`\textbf{B}_0` field.

Usage Example
~~~~~~~~~~~~~

.. code-block:: python

   from firm3d.field import ShearAlfvenWavesSuperposition
   import numpy as np

   # Create base wave
   Phihat1 = 0.01

   wave1 = ShearAlfvenHarmonic(
       Phihat1, Phim=2, Phin=1, omega=1.0, phase=0.0, B0=eq_field
   )

   # Create superposition (constructor takes a list of waves)
   saw_super = ShearAlfvenWavesSuperposition([wave1])

   # Add additional waves
   Phihat2 = 0.005

   wave2 = ShearAlfvenHarmonic(
       Phihat2, Phim=3, Phin=1, omega=1.5, phase=0.0, B0=eq_field
   )

   saw_super.add_wave(wave2)

InterpolatedShearAlfvenWave
---------------------------

``InterpolatedShearAlfvenWave`` accelerates evaluation of wave quantities by
interpolating a ``ShearAlfvenWave`` or ``ShearAlfvenWavesSuperposition`` on a
regular grid in :math:`(s,\theta,\zeta)` using the C++ ``RegularGridInterpolant3D``.
The interpolant is built by sampling each quantity at two reference times
(:math:`t=0` and a quarter-period offset) and reconstructing the full time
dependence at evaluation time via trigonometric identities. Use it in tracing loops where repeated wave evaluations dominate
runtime, especially for a large number of harmonics.

Usage Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from firm3d.field.boozermagneticfield import (
       InterpolatedBoozerField,
       ShearAlfvenWavesSuperposition,
       InterpolatedShearAlfvenWave,
   )
   from firm3d.saw.ae3d import AE3DEigenvector
   from firm3d.util.mpi import comm_world

   # Equilibrium from Boozer transform file
   field = InterpolatedBoozerField.from_booz_xform(
       "boozmn.nc",
       degree=3,
       ns=48,
       ntheta=48,
       nzeta=48,
       comm=comm_world,
   )

   # Wave from AE3D eigenvector
   saw = ShearAlfvenWavesSuperposition.from_ae3d(
       eigenvector=AE3DEigenvector.load_from_numpy("ae.npy"),
       B0=field,
       max_dB_normal_by_B0=5e-3,
       minor_radius_meters=1.7,
   )

   # Interpolant for fast tracing
   saw_interp = InterpolatedShearAlfvenWave(
       saw,
       degree=3,
       ns_interp=48,
       ntheta_interp=48,
       nzeta_interp=48,
       extrapolate=True,
       comm=comm_world,
   )

Wave Evaluation
---------------

ShearAlfvenWave classes provide methods to evaluate the perturbed fields. First, set the evaluation points, then evaluate the wave quantities:

.. code-block:: python

   # Set evaluation points - must be in shape (npoints, 4)
   points = np.array([
       [0.5, 0.0, 0.0, 0.0],  # [s, theta, zeta, t]
       [0.6, 0.1, 0.1, 0.1],
       [0.7, 0.2, 0.2, 0.2]
   ])
   saw.set_points(points)

   # Now evaluate wave quantities
   phi = saw.Phi()  # scalar potential
   alpha = saw.alpha()  # vector potential parameter

   # Get derivatives
   dphi_dpsi = saw.dPhidpsi()  # derivative with respect to psi
   dphi_dtheta = saw.dPhidtheta()  # derivative with respect to theta
   dphi_dzeta = saw.dPhidzeta()  # derivative with respect to zeta
   dphi_dt = saw.Phidot()  # time derivative of phi
   dalpha_dt = saw.alphadot()  # time derivative of alpha

   # For single point evaluation
   single_point = np.array([[0.5, 0.0, 0.0, 0.0]])  # shape (1, 4)
   saw.set_points(single_point)
   phi_single = saw.Phi()[0]  # get first (and only) value
