Magnetic Axis Handling
=====================

The coordinate singularity at the magnetic axis can be handled in several ways using the keyword argument ``axis`` passed to ``trace_particles_boozer`` and ``trace_particles_boozer_perturbed``.

Standard Boozer Coordinates (axis=0)
------------------------------------

If ``axis=0``, the trajectory will be integrated in standard Boozer coordinates :math:`(s,\theta,\zeta)`. If this is used, it is recommended that one passes a ``MinToroidalFluxStoppingCriterion`` to prevent particles from passing to :math:`s < 0`.

.. code-block:: python

   from firm3d.field.trajectory_helpers import MinToroidalFluxStoppingCriterion

   # Use standard Boozer coordinates
   res_tys, res_hits = trace_particles_boozer(
       field=field,
       stz_inits=points,
       parallel_speeds=vpars,
       tmax=1e-3,
       axis=0,  # Standard Boozer coordinates
       stopping_criteria=[MinToroidalFluxStoppingCriterion(0.001),MaxToroidalFluxStoppingCriterion(1)]
   )

Pseudo-Cartesian Coordinates (axis=1)
------------------------------------

If ``axis=1``, the trajectory will be integrated in the pseudo-Cartesian coordinates :math:`(\sqrt{s}\cos(\theta),\sqrt{s}\sin(\theta),\zeta)`, but all trajectory information will be saved in the standard Boozer coordinates :math:`(s,\theta,\zeta)`. This option prevents particles from passing to :math:`s < 0`. Because the equations of motion are mapped from :math:`(s,\theta,\zeta)` to :math:`(\sqrt{s}\cos(\theta),\sqrt{s},\sin(\theta),\zeta)`, a division by :math:`\sqrt{s}` is performed. Thus this option may be ill-behaved near the axis.

.. code-block:: python

   # Use pseudo-Cartesian coordinates with sqrt(s) scaling
   res_tys, res_hits = trace_particles_boozer(
       field=field,
       stz_inits=points,
       parallel_speeds=vpars,
       tmax=1e-3,
       axis=1,  # Pseudo-Cartesian coordinates
       stopping_criteria=[MaxToroidalFluxStoppingCriterion(1)]
   )

Pseudo-Cartesian Coordinates (axis=2)
------------------------------------

If ``axis=2``, the trajectory will be integrated in the pseudo-Cartesian coordinates :math:`(s\cos(\theta),s\sin(\theta),\zeta)`, but all trajectory information will be saved in the standard Boozer coordinates :math:`(s,\theta,\zeta)`. This option prevents particles from passing to :math:`s < 0`. No division by :math:`s` is required to map to this coordinate system. This option (default) is recommended if one would like to integrate near the magnetic axis.

.. code-block:: python

   # Use pseudo-Cartesian coordinates with s scaling (recommended)
   res_tys, res_hits = trace_particles_boozer(
       field=field,
       stz_inits=points,
       parallel_speeds=vpars,
       tmax=1e-3,
       axis=2,  # Pseudo-Cartesian coordinates (recommended)
       stopping_criteria=[MaxToroidalFluxStoppingCriterion(1)]
   )
