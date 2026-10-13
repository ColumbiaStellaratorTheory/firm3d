GPU Tracing (CATAPULT)
======================

The guiding-center equations can be integrated on an NVIDIA GPU with the
CATAPULT kernels in ``firm3d.catapult``, for equilibrium fields in Boozer
coordinates, fields with shear Alfvén waves, and fields in Cartesian
coordinates. The interface follows the CPU tracers as far as the kernels
allow, so that a script written for one runs on the other with the field
line changed.

Field objects
-------------

The kernels read the field from a table on a grid in :math:`(s, \theta,
\zeta)` (or :math:`(r, \phi, z)`). The table is built once, by an object that
plays the role :class:`~firm3d.field.boozermagneticfield.InterpolatedBoozerField`
plays on the CPU:

.. code-block:: python

    from firm3d.catapult.field import CatapultBoozerField
    field_gpu = CatapultBoozerField(field, ns=48, ntheta=48, nzeta=48)

``CatapultPerturbedBoozerField`` does the same for a
``ShearAlfvenWavesSuperposition``, and ``CatapultCartesianField`` for a simsopt
``InterpolatedField`` together with the ``SurfaceClassifier`` that defines the
boundary. Building the table is the expensive step (minutes at production
resolution), so build it once and trace as often as needed.

Tracing
-------

``trace_particles_boozer_gpu`` and ``trace_particles_boozer_perturbed_gpu``
take the arguments of ``trace_particles_boozer`` and
``trace_particles_boozer_perturbed`` that the kernels support, with the same
defaults, and return the same ``(res_tys, res_hits)``:

.. code-block:: python

    from firm3d.catapult.tracing import trace_particles_boozer_gpu
    res_tys, res_hits = trace_particles_boozer_gpu(
        field_gpu, stz_inits, vpar_inits, tmax=1e-2, Ekin=Ekin, mass=mass,
        charge=charge, forget_exact_path=True,
    )
    times, loss_fraction = compute_loss_fraction(res_tys)

The kernels stop a particle at :math:`s = 1` (or at the surface of the
classifier). This is equivalent to the CPU tracer's
``MaxToroidalFluxStoppingCriterion(1.0)``, and ``res_hits`` records it with
index ``-1`` as the CPU does for its first criterion.

Saving trajectories
-------------------

Set ``forget_exact_path=False`` and choose a saving interval with ``dt_save``:

.. code-block:: python

    res_tys, res_hits = trace_particles_boozer_gpu(
        field_gpu, stz_inits, vpar_inits, tmax=1e-2, Ekin=Ekin, mass=mass,
        charge=charge, forget_exact_path=False, dt_save=1e-6,
    )

The GPU uses the same Dormand–Prince continuous extension as the CPU's
default Boost solver. Paths contain the initial state, every reached saving
time, and the final state at ``tmax``. Changing ``dt_save`` does not change
the adaptive step sequence. Endpoint-only tracing also returns the state at
``tmax``. Each particle may have its own ``tmax``; perturbed tracing preserves
the waves' absolute phase throughout the integration.

Lost particles stop at the first accepted endpoint beyond the boundary.
GPU boundary detection does not locate crossings with the CPU's root finder.

The public ``trace_particles_*_gpu`` functions return five-column paths
``(t, coordinate_1, coordinate_2, coordinate_3, vpar)`` in ``float64``.
The lower-level ``save_trajectories_boozer_gpu`` and
``save_trajectories_cartesian_gpu`` omit the initial state and return
seven-column rows with the enclosing step size and magnetic moment appended.
Native ``firm3dpp.*_gpu_tracing`` arrays use the field's precision.
An interpolated sample is not an integrator checkpoint.

Trajectory storage grows with particles times requested samples. Allow for
both GPU buffers and host copies; use smaller batches for large histories.
Set ``forget_exact_path=True`` when only endpoints are needed.

See :ref:`gpu_dense_output_examples` for saving, reloading, and Poincaré
plotting.

Sections and stopping
---------------------

Request ``zetas=[0.0]`` to save toroidal-section crossings directly in
``res_hits``, even with ``forget_exact_path=True``. General CPU phase planes
are supported with ``phases``, ``n_zetas``, ``m_thetas`` and ``omegas``.
Request ``vpars=[0.0], vpars_stop=True`` to stop at a mirror point, or
``phases_stop=True`` to stop at the first phase crossing. These roots use the
accepted step's dense output and are independent of ``dt_save``. The launch
is excluded; a crossing at a step's right endpoint is included once.

Hits have rows ``(t, index, s, theta, zeta, vpar)``. Phase indices start at
zero, velocity indices follow them, and criterion indices are ``-1-i``.
The enforced boundary follows requested criteria in the index sequence.
Trajectory zeta and both hit angles retain accumulated turns. ``max_hits`` bounds
storage per particle (default 1024). Overflow raises an error; increase the
capacity or shorten the trace. ``max_phase_interval`` optionally limits the
time to the next phase hit, measured from launch or the preceding phase hit.
``max_phase_hits`` optionally stops after a
chosen number of phase hits.

Boozer tracing accepts CPU ``MaxToroidalFluxStoppingCriterion``,
``MinToroidalFluxStoppingCriterion``, ``ToroidalTransitStoppingCriterion``,
``IterationStoppingCriterion`` and ``StepSizeStoppingCriterion`` objects.
These checks use accepted endpoints, as on the CPU. Cartesian tracing
supports iteration and step-size criteria and velocity hits. Field boundaries
remain enforced; custom Python stopping callbacks cannot run on the GPU.

Single precision
----------------

Each field object takes ``precision="single"``, which stores the table in
``float32`` and runs the kernels in single precision; the initial conditions
are cast to match. This is faster and halves the memory.

What single precision does and does not reproduce should be understood before
using it. The field and its derivatives agree with double precision to about
:math:`10^{-5}` relative, and loss fractions agree within their statistical
uncertainty. Individual orbits do not agree: after a fraction of a transit,
single and double precision states of the same particle differ by about as
much as two double precision runs at tolerances ``1e-8`` and ``1e-9`` differ
from each other. That is the orbits' own sensitivity to any small
perturbation, and it is the level at which single precision differs. Use
single precision for statistics over an ensemble, such as loss fractions and
confinement times, and not for following a particular particle, for orbit
classification of individual markers, or for anything that compares one
trajectory to another.
