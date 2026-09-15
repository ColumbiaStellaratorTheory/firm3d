Stopping Criteria
=================

Guiding center integration is continued until the maximum integration time, ``tmax``, is reached, or until one of the ``StoppingCriteria`` is hit. Stopping criteria are essential for controlling the integration process and avoiding numerical issues.

Available Stopping Criteria
---------------------------

MaxToroidalFluxStoppingCriterion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Stop when trajectory reaches a maximum value of normalized toroidal flux (e.g., :math:`s=1` indicates the plasma boundary).

.. code-block:: python

   from firm3d.field.tracing import MaxToroidalFluxStoppingCriterion

   # Stop when s >= 1.0 (plasma boundary)
   stopping_criteria = [MaxToroidalFluxStoppingCriterion(1.0)]

MinToroidalFluxStoppingCriterion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Stop when trajectory reaches a minimum value of normalized toroidal flux. When ``axis=0`` a point close to the axis, e.g. :math:`s = 10^{-3}`, is chosen to avoid numerical issues associated with the coordinate singularity.

.. code-block:: python

   from firm3d.field.tracing import MinToroidalFluxStoppingCriterion

   # Stop when s <= 0.001 (close to magnetic axis)
   stopping_criteria = [MinToroidalFluxStoppingCriterion(0.001)]

ToroidalTransitStoppingCriterion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Stop when the toroidal angle increases by an integer multiple of :math:`2\pi`. Useful for resonance detection.

.. code-block:: python

   from firm3d.field.tracing import ToroidalTransitStoppingCriterion

   # Stop after 5 toroidal transits
   stopping_criteria = [ToroidalTransitStoppingCriterion(5)]

IterationStoppingCriterion
~~~~~~~~~~~~~~~~~~~~~~~~~~

Stop when a number of iterations is reached. This is useful for terminating long integrations.

.. code-block:: python

   from firm3d.field.tracing import IterationStoppingCriterion

   # Stop after 10000 integration steps
   stopping_criteria = [IterationStoppingCriterion(10000)]

StepSizeStoppingCriterion
~~~~~~~~~~~~~~~~~~~~~~~~~

Stop when the step size gets too small. When using adaptive timestepping, can avoid particles getting "stuck" due to a small step size.

.. code-block:: python

   from firm3d.field.tracing import StepSizeStoppingCriterion

   # Stop when step size < 1e-10
   stopping_criteria = [StepSizeStoppingCriterion(1e-10)]

.. warning::
   The Python binding for ``StepSizeStoppingCriterion`` currently declares its
   argument as a C++ ``double`` while the underlying class stores it as an
   ``int``, so any fractional value (e.g. ``1e-10``) is silently truncated to
   ``0`` and the criterion never triggers. This is a known bug, tracked
   separately from this documentation.

Usage Examples
--------------

Multiple Stopping Criteria
~~~~~~~~~~~~~~~~~~~~~~~~~~

You can combine multiple stopping criteria to create robust integration conditions:

.. code-block:: python

   from firm3d.field.tracing import (
       trace_particles_boozer,
       MaxToroidalFluxStoppingCriterion,
       MinToroidalFluxStoppingCriterion,
       IterationStoppingCriterion,
   )

   # Combine multiple criteria
   stopping_criteria = [
       MaxToroidalFluxStoppingCriterion(1.0),    # Stop at boundary
       MinToroidalFluxStoppingCriterion(0.001),  # Stop near axis
       IterationStoppingCriterion(50000)         # Stop after max iterations
   ]

   # Use in tracing. Stopping on a parallel-velocity crossing (e.g. mirror
   # points) is not a StoppingCriterion subclass -- pass the target value(s)
   # via `vpars` and set `vpars_stop=True` instead.
   res_tys, res_hits = trace_particles_boozer(
       field=field,
       stz_inits=points,
       parallel_speeds=parallel_speeds,
       tmax=1e-3,
       stopping_criteria=stopping_criteria,
       vpars=[0.0],       # stop when v_parallel crosses zero (mirroring)
       vpars_stop=True,
   )

Interpreting Results
--------------------

When stopping criteria are hit, the information is returned in the ``res_hits`` array. See :ref:`trajectory_saving` for more details. Each row contains:

- **time**: Time when the criterion was hit
- **idx**: Index indicating which criterion was hit

  - If ``idx >= 0`` and ``idx < len(phases)``: the ``phases[idx]`` plane was hit,
    i.e. ``n_zetas[idx]*zeta + m_thetas[idx]*theta - omegas[idx]*t`` crossed
    ``phases[idx]``
  - If ``len(vpars)+len(phases) > idx >= len(phases)``: the
    ``vpars[idx-len(phases)]`` value was crossed
  - If ``idx < 0``: ``stopping_criteria[int(-idx)-1]`` was hit
- **state**: The state vector ``[s, theta, zeta, v_parallel]``

.. code-block:: python

   # Analyze which stopping criteria were hit
   for i, hits in enumerate(res_hits):
       print(f"Particle {i}:")
       for hit in hits:
           time, idx, s, theta, zeta, vpar = hit
           if idx < 0:
               criterion_idx = int(-idx) - 1
               print(f"  Hit stopping criterion {criterion_idx} at t={time:.3f}")
           else:
               print(f"  Hit coordinate plane {idx} at t={time:.3f}")
