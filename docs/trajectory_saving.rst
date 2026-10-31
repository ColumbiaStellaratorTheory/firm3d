.. _trajectory_saving:

Trajectory Saving
=================

There are two ways the trajectory information can be saved: by recording "hits" of user-defined coordinate planes (e.g., Poincaré sections), or by recording uniform time intervals of the trajectory. The routines ``trace_particles_boozer`` or ``trace_particles_boozer_perturbed`` return this information in the tuple ``(res_tys,res_hits)``.

The CATAPULT ``trace_particles_*_gpu`` functions use the same saving options
and return format. See :doc:`gpu_tracing` for GPU field setup.

Trajectory Data (res_tys)
-------------------------

If ``forget_exact_path=False``, the parameter ``dt_save`` determines the time interval for trajectory saving. (Note that if this parameter is made too small, one may run into memory issues.) This trajectory information is returned in ``res_tys``, which is a list (length = number of particles) of numpy arrays with shape ``(nsave,5)``. Here ``nsave`` is the number of timesteps saved. Each row contains the time and the state, ``[t, s, theta, zeta, v_par]``. If ``forget_exact_path=True``, only the state at the initial and final time will be returned.

GPU tracing uses the same Dormand–Prince continuous extension as the CPU's
default Boost solver. Public GPU paths are returned in ``float64``. Native
``firm3dpp.*_gpu_tracing`` arrays use the field's precision and append the
enclosing step size and magnetic moment to their seven-column rows.

.. code-block:: python

   # Trace with trajectory saving
   res_tys, res_hits = trace_particles_boozer(
       field=field,
       stz_inits=points,
       parallel_speeds=vpars,
       tmax=1e-3,
       dt_save=1e-6,
       forget_exact_path=False
   )

   # Access trajectory data
   trajectory = res_tys[0]  # First particle
   times = trajectory[:, 0]
   s_values = trajectory[:, 1]
   theta_values = trajectory[:, 2]
   zeta_values = trajectory[:, 3]
   vpar_values = trajectory[:, 4]

   print(f"Saved {len(times)} trajectory points")
   print(f"Time range: {times[0]:.3f} to {times[-1]:.3f}")

Hits Data (res_hits)
--------------------

The "hits" are defined through the input lists ``phases``, ``n_zetas``, ``m_thetas``, ``omegas``, and ``vpars``.

**Hit Planes:**

- **vpars**: If specified, the trajectory will be recorded when the parallel velocity hits a given value. For example, the Poincaré map for trapped particles is defined by recording the points with :math:`v_{||} = 0`.

- **phases**, **n_zetas**, **m_thetas** and **omegas**: If ``phases`` is specified, the trajectory will be recorded when :math:`n_\zeta \zeta + m_\theta \theta - \omega t` hits the values given in the ``phases`` array, with :math:`n_\zeta`, :math:`m_\theta`, and :math:`\omega` given elementwise by the ``n_zetas``, ``m_thetas``, and ``omegas`` arrays. All four lists must have the same length.

Phase and velocity crossings use Boost TOMS748 on the accepted step's dense output and are
independent of ``dt_save``. They are recorded even with
``forget_exact_path=True``. The launch is excluded; a crossing at a step's
right endpoint is included once. CPU and GPU tracers select at most one
crossing per requested level per accepted step, using the step endpoints
to bracket it. Additional crossings inside a step can be missed. To stop
at these crossings, see :ref:`stopping_criteria`.

The Boozer GPU tracers also accept ``zetas=[0.0]`` as a shortcut for a
toroidal section, equivalent to ``phases=[0.0]``, ``n_zetas=[1.0]``,
``m_thetas=[0.0]`` and ``omegas=[0.0]``.


**Hit Data Structure:**

The hits are returned in ``res_hits``, which is a list (length = number of particles) of numpy arrays with shape ``(nhits,6)``, where ``nhits`` is the number of hits of a coordinate plane or stopping criteria. Each row of the array contains ``[time] + [idx] + state]``, where ``idx`` tells us which of the hit planes or stopping criteria was hit:

- If ``idx >= 0`` and ``idx < len(phases)``: the ``phases[idx]`` plane was hit
- If ``len(phases) <= idx < len(phases) + len(vpars)``: the ``vpars[idx-len(phases)]`` value was hit
- If ``idx < 0``: ``stopping_criteria[int(-idx)-1]`` was hit

The state vector is ``[s, theta, zeta, v_par]``.

On the GPU, trajectory zeta and both hit angles retain accumulated turns.
``max_hits`` bounds hit storage per particle (default 1024); overflow raises
an error. Increase the capacity or shorten the trace if needed.

Multiple Hit Planes
-------------------

For custom analysis beyond the standard Poincaré maps, you can specify multiple hit planes manually:

.. code-block:: python

   # Record hits when zeta - omega*t crosses each of these 4 values
   phases = [0.0, np.pi/2, np.pi, 3*np.pi/2]
   n_zetas = [1.0, 1.0, 1.0, 1.0]
   m_thetas = [0.0, 0.0, 0.0, 0.0]
   omegas = [0.0, 0.0, 0.0, 0.0]

   # Record hits at multiple parallel velocities
   vpar_hits = [0.0, 0.5, -0.5]

   res_tys, res_hits = trace_particles_boozer(
       field=field,
       stz_inits=points,
       parallel_speeds=vpars,
       tmax=1e-3,
       phases=phases,
       n_zetas=n_zetas,
       m_thetas=m_thetas,
       omegas=omegas,
       vpars=vpar_hits
   )

   # Analyze hits
   hits = res_hits[0]
   for hit in hits:
       time, idx, s, theta, zeta, vpar = hit
       if idx < 0:
           print(f"Hit stopping criterion at t={time:.3f}")
       elif idx < len(phases):
           print(f"Hit phase plane {int(idx)} at t={time:.3f}")
       else:
           vpar_idx = int(idx) - len(phases)
           print(f"Hit v_parallel plane {vpar_idx} at t={time:.3f}")

Memory Management
-----------------

For long integrations, memory usage can become an issue. Use ``forget_exact_path=True`` to save only initial and final states:

.. code-block:: python

   # Save only initial and final states to save memory
   res_tys, res_hits = trace_particles_boozer(
       field=field,
       stz_inits=points,
       parallel_speeds=vpars,
       tmax=1e-3,
       forget_exact_path=True,  # Only save initial/final states
       vpars=[0.0]  # Still record hits
   )

   # res_tys will contain only 2 points per particle
   trajectory = res_tys[0]
   print(f"Saved {len(trajectory)} points (initial and final only)")

Trajectory storage grows with particles times requested samples. On the GPU,
allow for both GPU buffers and host copies; use smaller batches for large
histories. See :ref:`gpu_dense_output_examples` for GPU saving, reloading,
and Poincaré plotting examples.
