Stopping Criteria
================

Guiding center integration is continued until the maximum integration time, ``tmax``, is reached, or until one of the ``StoppingCriteria`` is hit. Stopping criteria are essential for controlling the integration process and avoiding numerical issues.

Available Stopping Criteria
--------------------------

MaxToroidalFluxStoppingCriterion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Stop when trajectory reaches a maximum value of normalized toroidal flux (e.g., :math:`s=1` indicates the plasma boundary).

.. code-block:: python

   from firm3d.field.trajectory_helpers import MaxToroidalFluxStoppingCriterion

   # Stop when s >= 1.0 (plasma boundary)
   stopping_criteria = [MaxToroidalFluxStoppingCriterion(1.0)]

MinToroidalFluxStoppingCriterion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Stop when trajectory reaches a minimum value of normalized toroidal flux. When ``axis=0`` a point close to the axis, e.g. :math:`s = 10^{-3}`, is chosen to avoid numerical issues associated with the coordinate singularity.

.. code-block:: python

   from firm3d.field.trajectory_helpers import MinToroidalFluxStoppingCriterion

   # Stop when s <= 0.001 (close to magnetic axis)
   stopping_criteria = [MinToroidalFluxStoppingCriterion(0.001)]

ZetaStoppingCriterion
~~~~~~~~~~~~~~~~~~~~~

Stop when the toroidal angle reaches a given value (modulus :math:`2\pi`).

.. code-block:: python

   from firm3d.field.trajectory_helpers import ZetaStoppingCriterion

   # Stop when zeta reaches pi/2 (mod 2*pi)
   stopping_criteria = [ZetaStoppingCriterion(np.pi/2)]

VparStoppingCriterion
~~~~~~~~~~~~~~~~~~~~~

Stop when the parallel velocity reaches a given value. For example, can be used to terminate tracing when a particle mirrors.

.. code-block:: python

   from firm3d.field.trajectory_helpers import VparStoppingCriterion

   # Stop when v_parallel changes sign (mirroring)
   stopping_criteria = [VparStoppingCriterion(0.0)]

ToroidalTransitStoppingCriterion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Stop when the toroidal angle increases by an integer multiple of :math:`2\pi`. Useful for resonance detection.

.. code-block:: python

   from firm3d.field.trajectory_helpers import ToroidalTransitStoppingCriterion

   # Stop after 5 toroidal transits
   stopping_criteria = [ToroidalTransitStoppingCriterion(5)]

IterationStoppingCriterion
~~~~~~~~~~~~~~~~~~~~~~~~~~

Stop when a number of iterations is reached. This is useful for terminating long integrations.

.. code-block:: python

   from firm3d.field.trajectory_helpers import IterationStoppingCriterion

   # Stop after 10000 integration steps
   stopping_criteria = [IterationStoppingCriterion(10000)]

StepSizeStoppingCriterion
~~~~~~~~~~~~~~~~~~~~~~~~~

Stop when the step size gets too small. When using adaptive timestepping, can avoid particles getting "stuck" due to a small step size.

.. code-block:: python

   from firm3d.field.trajectory_helpers import StepSizeStoppingCriterion

   # Stop when step size < 1e-10
   stopping_criteria = [StepSizeStoppingCriterion(1e-10)]

Usage Examples
--------------

Multiple Stopping Criteria
~~~~~~~~~~~~~~~~~~~~~~~~~~

You can combine multiple stopping criteria to create robust integration conditions:

.. code-block:: python

   from firm3d.field.tracing import (
       MaxToroidalFluxStoppingCriterion,
       MinToroidalFluxStoppingCriterion,
       VparStoppingCriterion,
       IterationStoppingCriterion
   )

   # Combine multiple criteria
   stopping_criteria = [
       MaxToroidalFluxStoppingCriterion(1.0),    # Stop at boundary
       MinToroidalFluxStoppingCriterion(0.001),  # Stop near axis
       VparStoppingCriterion(0.0),               # Stop at mirror points
       IterationStoppingCriterion(50000)         # Stop after max iterations
   ]

   # Use in tracing
   res_tys, res_hits = trace_particles_boozer(
       field=field,
       stz_inits=points,
       parallel_speeds=vpars,
       tmax=1e-3,
       stopping_criteria=stopping_criteria
   )

Interpreting Results
-------------------

When stopping criteria are hit, the information is returned in the ``res_hits`` array. See :ref:`trajectory_saving` for more details. Each row contains:

- **time**: Time when the criterion was hit
- **idx**: Index indicating which criterion was hit

  - If ``idx >= 0`` and ``idx < len(zetas)``: the ``zetas[idx]`` plane was hit
  - If ``len(vpars)+len(zetas)>idx>=len(zetas)``: the ``vpars[idx-len(zetas)]`` plane was hit
  - If ``idx >= len(vpars)+len(zetas)``: the ``thetas[idx-len(vpars)-len(zetas)]`` plane was hit
  - If ``idx < 0``: ``stopping_criteria[int(-idx)-1]`` was hit
- **state**: The state vector ``[t, s, theta, zeta, v_parallel]``

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
