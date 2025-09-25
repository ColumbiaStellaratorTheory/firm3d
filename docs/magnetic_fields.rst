Magnetic Field Classes
=====================

The equilibrium magnetic field in Boozer coordinates, an instance of ``BoozerMagneticField``, can be represented using different approaches depending on the requirements of your simulation.

Overview
--------

An important attribute of ``BoozerMagneticField`` classes is ``field_type``, which can be:

- ``vac`` (vacuum field)
- ``no_k`` (does not retain the radial covariant component)
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

   G(s) = G_0 + \sqrt{2s\psi_0/\overline{B}} G_1

   I(s) = I_0 + \sqrt{2s\psi_0/\overline{B}} I_1

   K(s,\theta,\zeta) = \sqrt{2s\psi_0/\overline{B}} K_1 \sin(\theta - N \zeta)

And the rotational transform is:

.. math::

   \iota(s) = \iota_0

While formally :math:`I_0 = I_1 = G_1 = K_1 = 0`, these terms have been included in order to test the guiding center equations at finite beta.

Usage Example
~~~~~~~~~~~~

.. code-block:: python

   from firm3d.field import BoozerAnalytic

   # Create an analytic Boozer field
   field = BoozerAnalytic(
       B0=1.0,
       eta_bar=0.1,
       N=3,
       B0z=0.01,
       m=2,
       n=1,
       G0=1.0,
       G1=0.0,
       I0=0.0,
       I1=0.0,
       K1=0.0,
       iota0=0.5
   )

BoozerRadialInterpolant
-----------------------

The magnetic field can be computed at any point in Boozer coordinates using radial spline interpolation (``scipy.interpolate.make_interp_spline``) and an inverse Fourier transform in the two angles. While the Fourier representation is more accurate, typically :ref:`InterpolatedBoozerField` is used inside the tracing loop due to its efficiency.

If given a ``VMEC`` output file, performs a Boozer coordinate transformation using ``booz_xform``. If given a ``booz_xform`` output file, the Boozer transformation must be performed with all surfaces on the VMEC half grid, and with ``phip``, ``chi``, ``pres``, and ``phi`` saved in the file.

Field evaluations are parallelized over the number of Fourier harmonics over CPUs, given the communicator ``comm``. In addition, the evaluations are parallelized over threads with OpenMP. Because the guiding center tracing routines are also parallelized over CPUs with MPI, we don't recommend passing ``comm`` to ``BoozerRadialInterpolant`` if it is being passed to a tracing routine.

Usage Example
~~~~~~~~~~~~

.. code-block:: python

   from firm3d.field import BoozerRadialInterpolant

   # Create field from booz_xform output
   field = BoozerRadialInterpolant("boozmn_file.nc")

Preparing booz_xform Equilibrium
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

InterpolatedBoozerField
-----------------------

This field takes an existing ``BoozerMagneticField`` instance, such as :ref:`BoozerRadialInterpolant`, and interpolates it on a regular grid in :math:`(s,\theta,\zeta)`. This resulting interpolant can then be evaluated very quickly inside the tracing loop.

Usage Example
~~~~~~~~~~~~

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
---------------

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
   B = field.B()  # magnetic field magnitude
   G = field.G()  # G component
   I = field.I()  # I component
   K = field.K()  # K component

   # For single point evaluation
   single_point = np.array([[0.5, 0.0, 0.0]])  # shape (1, 3)
   field.set_points(single_point)
   B_single = field.B()[0]  # get first (and only) value
