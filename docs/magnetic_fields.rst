Magnetic Field Classes
======================

The equilibrium magnetic field in Boozer coordinates, an instance of ``BoozerMagneticField``, can be represented using different approaches depending on the requirements of your simulation.

Overview
--------

An important attribute of ``BoozerMagneticField`` classes is ``field_type``, which can be:

- ``vac`` (vacuum field)
- ``nok`` (does not retain the radial covariant component)
- Empty string (no assumptions made)

By default, the ``field_type`` will determine the ``mode`` used for guiding center tracing: ``gc_vac``, ``gc_noK``, or ``gc``.

BoozerAnalytic
--------------

Computes a ``BoozerMagneticField`` based on a first-order expansion in distance from the magnetic axis (Landreman & Sengupta, Journal of Plasma Physics 2018). A possibility to include a QS-breaking perturbation is added.

The magnetic field strength is expressed as:

.. math::

   B(s,\theta,\zeta) = B_0 \left(1 + \overline{\eta} \sqrt{2s\psi_0/\overline{B}}\cos(\theta - N \zeta)\right) + B_{0z}\cos(m\theta-n\zeta)

The covariant components of equilibrium field are:

.. math::

   G(s) = G_0 + s G_1

   I(s) = I_0 + s I_1

   K(s,\theta,\zeta) = \sqrt{2s\psi_0/\overline{B}} K_1 \sin(\theta - N \zeta)

And the rotational transform is:

.. math::

   \iota(s) = \iota_0 + s \iota_1

While formally :math:`I_0 = I_1 = G_1 = K_1 = \iota_1 = 0`, these terms have been included in order to test the guiding center equations at finite beta.

Usage Example
~~~~~~~~~~~~~

.. code-block:: python

   from firm3d.field import BoozerAnalytic

   # Create an analytic Boozer field
   field = BoozerAnalytic(
       etabar=0.1,
       B0=1.0,
       N=3,
       G0=1.0,
       psi0=0.1,
       iota0=0.5,
       I0=0.0,
       G1=0.0,
       I1=0.0,
       K1=0.0,
       B0z=[0.01],
       m=[2],
       n=[1],
   )

BoozerRadialInterpolant
-----------------------

The magnetic field can be computed at any point in Boozer coordinates using radial spline interpolation (``scipy.interpolate.make_interp_spline``) and an inverse Fourier transform in the two angles. While the Fourier representation is more accurate, typically :ref:`InterpolatedBoozerField` is used inside the tracing loop due to its efficiency.

If given a ``VMEC`` output file, performs a Boozer coordinate transformation using ``booz_xform``. If given a ``booz_xform`` output file, the Boozer transformation must be performed with all surfaces on the VMEC half grid, and with ``phip``, ``chi``, ``pres``, and ``phi`` saved in the file.

The one-time setup (computing Fourier coefficients of ``K`` and building splines) is parallelized over evaluation-grid points across CPUs, given the communicator ``comm``, with the resulting splines then broadcast to all ranks. Field evaluations themselves are parallelized over points with OpenMP. Because the guiding center tracing routines are also parallelized over CPUs with MPI, we don't recommend passing ``comm`` to ``BoozerRadialInterpolant`` if it is being passed to a tracing routine.

.. note::
   For most use cases, it is recommended to use ``InterpolatedBoozerField.from_booz_xform()`` instead of creating a ``BoozerRadialInterpolant``. The ``from_booz_xform()`` method provides a more convenient interface (through the ``BoozerSplineField`` class) and is the preferred approach in most examples.

Usage Example
~~~~~~~~~~~~~

.. code-block:: python

   from firm3d.field import BoozerRadialInterpolant

   # Create field from booz_xform output
   field = BoozerRadialInterpolant("boozmn_file.nc")

Preparing booz_xform Equilibrium
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As stated above, the ``booz_xform`` equilibrium must be performed with all surfaces on the VMEC half grid, and with ``phip``, ``chi``, ``pres``, and ``phi`` saved in the file. This can be done using the `C++ implementation <https://github.com/hiddenSymmetries/booz_xform>`_ with the main branch, by passing ``flux=True`` to ``read_wout()``:

.. code-block:: python

   import booz_xform as bx

   b = bx.Booz_xform()
   b.read_wout(wout_filename, True)
   b.mboz = mboz
   b.nboz = nboz
   b.run()
   b.write_boozmn(boozmn_filename)

Equilibria produced with the `STELLOPT implementation <https://github.com/PrincetonUniversity/STELLOPT>`_ can also be used.

BoozerSplineField
-----------------

``BoozerSplineField`` interpolates the magnetic field on the VMEC grid using spline interpolation. It uses :mod:`ndsplines` for 3D fields and :class:`scipy.interpolate.CubicSpline` for 1D flux functions. This class is typically used internally by ``InterpolatedBoozerField.from_booz_xform()`` to handle the Boozer coordinate transformation and initial interpolation.

The field can be created directly from a VMEC output file (``wout_*.nc``), a ``booz_xform`` output file (``boozmn_*.nc``), or a ``Booz_xform`` instance. When given a VMEC file, it automatically performs the Boozer coordinate transformation using ``booz_xform``.

Key features:

- Interpolates on the VMEC grid with user-specified angular resolution
- Supports quasisymmetry enforcement through ``helicity_M`` and ``helicity_N`` parameters
- Can enforce vacuum field assumptions or exclude the K component; note that ``no_K`` defaults to ``True`` (unlike ``BoozerRadialInterpolant``, where it defaults to ``False``), so the K component is excluded unless ``no_K=False`` is passed explicitly
- Supports parallel computation via MPI communicator

Usage Example
~~~~~~~~~~~~~~

.. code-block:: python

   from firm3d.field import BoozerSplineField
   from firm3d.util.mpi import comm_world

   # Create field from booz_xform output
   field = BoozerSplineField(
       "boozmn_file.nc",
       ntheta=48,  # number of poloidal grid points
       nzeta=48,  # number of toroidal grid points
       comm=comm_world  # optional MPI communicator
   )

   # Create field from VMEC output (performs booz_xform automatically)
   field = BoozerSplineField(
       "wout_file.nc",
       mpol=32,  # number of poloidal mode numbers for booz_xform
       ntor=32,  # number of toroidal mode numbers for booz_xform
       ntheta=48,
       nzeta=48,
       comm=comm_world
   )

   # With quasisymmetry explicitly enforced
   field = BoozerSplineField(
       "boozmn_file.nc",
       ntheta=48,
       nzeta=48,
       helicity_M=1,  # poloidal helicity coefficient
       helicity_N=0,  # toroidal helicity coefficient
       comm=comm_world
   )

   # Vacuum field (enforces G = mean(G), I = 0, K = 0)
   field = BoozerSplineField(
       "boozmn_file.nc",
       ntheta=48,
       nzeta=48,
       enforce_vacuum=True,
       comm=comm_world
   )

.. note::
   For most use cases, it is recommended to use ``InterpolatedBoozerField.from_booz_xform()`` instead of creating a ``BoozerSplineField`` directly. The ``from_booz_xform()`` method creates a ``BoozerSplineField`` internally and wraps it with an ``InterpolatedBoozerField`` for optimal performance in particle tracing.

.. _interpolatedboozerfield:

InterpolatedBoozerField
--------------------------

This field interpolates a magnetic field on a regular grid in :math:`(s,\theta,\zeta)`. This resulting interpolant can then be evaluated very quickly inside the tracing loop.

The recommended way to create an ``InterpolatedBoozerField`` from a ``booz_xform`` output file is using the ``from_booz_xform`` class method, which automatically handles the Boozer coordinate transformation and interpolation setup. Note that ``from_booz_xform`` also defaults to ``no_K=True``, excluding the K component unless ``no_K=False`` is passed explicitly.

Usage Example (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from firm3d.field import InterpolatedBoozerField
   from firm3d.util.mpi import comm_world

   # Create the interpolated field directly from booz_xform output
   field = InterpolatedBoozerField.from_booz_xform(
       "boozmn_file.nc",
       degree=3,  # degree of piecewise polynomial interpolant
       ns=48,  # number of radial points
       ntheta=48,  # number of poloidal points
       nzeta=48,  # number of toroidal points
       comm=comm_world  # optional MPI communicator
   )

   # With quasisymmetry explicitly enforced
   field = InterpolatedBoozerField.from_booz_xform(
       "boozmn_file.nc",
       degree=3,
       ns=48,
       ntheta=48,
       nzeta=48,
       helicity_M=1,  # helicity M
       helicity_N=0,  # helicity N
       comm=comm_world
   )

Alternative Usage (Legacy)
~~~~~~~~~~~~~~~~~~~~~~~~~~

For advanced use cases, you can also create an ``InterpolatedBoozerField`` from an existing ``BoozerMagneticField`` instance:

.. code-block:: python

   from firm3d.field import BoozerRadialInterpolant, InterpolatedBoozerField

   # Create the base field
   base_field = BoozerRadialInterpolant("boozmn_file.nc")

   # Create the interpolated field for fast evaluation
   field = InterpolatedBoozerField(
       base_field,
       degree=3,
       ns_interp=48,  # number of radial points
       ntheta_interp=48,  # number of poloidal points
       nzeta_interp=48   # number of toroidal points
   )

Field Evaluation
----------------

All magnetic field classes provide methods to evaluate the field at given points. First, set the evaluation points, then evaluate the field quantities:

.. code-block:: python

   # Set evaluation points - must be in shape (npoints, 3)
   points = np.array([
       [0.5, 0.0, 0.0],  # [s, theta, zeta]
       [0.6, 0.1, 0.1],
       [0.7, 0.2, 0.2]
   ])
   field.set_points(points)

   # Now evaluate field quantities
   B = field.modB()  # magnetic field magnitude
   G = field.G()  # G component
   I = field.I()  # I component
   K = field.K()  # K component

   # For single point evaluation
   single_point = np.array([[0.5, 0.0, 0.0]])  # shape (1, 3)
   field.set_points(single_point)
   B_single = field.modB()[0]  # get first (and only) value
