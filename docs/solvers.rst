Solvers and Solver Options
==========================

By default the Runge-Kutta Dormand-Prince 5(4) method implemented in `Boost <https://www.boost.org/doc/libs/1_54_0/boost/numeric/odeint/stepper/runge_kutta_dopri5.hpp>`_ is used to integrate the ODEs. Adaptive time stepping is performed to satisfy the user-prescribed relative and absolute error tolerance parameters, ``reltol`` and ``abstol``.

Default Solver (Runge-Kutta Dormand-Prince 5(4))
------------------------------------------------

The default solver uses the Runge-Kutta Dormand-Prince 5(4) method with adaptive timestepping:

.. code-block:: python

   from firm3d.field.tracing import trace_particles_boozer

   # Use default solver with custom tolerances
   res_tys, res_hits = trace_particles_boozer(
       field=field,
       stz_inits=points,
       parallel_speeds=vpars,
       tmax=1e-3,
       reltol=1e-8,  # Relative tolerance
       abstol=1e-10  # Absolute tolerance
   )

Symplectic Solver
-----------------

If ``solveSympl=True``, a symplectic solver is used with step size ``dt``. The semi-implicit Euler scheme described in Albert, C. G., et al. (2020). Symplectic integration with non-canonical quadrature for guiding-center orbits in magnetic confinement devices. Journal of computational physics, 403, 109065 is implemented. A root solve is performed to map from non-canonical to canonical variables, with tolerance given by ``roottol``. If ``predictor_step=True``, the initial guess for the next step is improved using first derivative information.

.. code-block:: python

   # Use symplectic solver
   res_tys, res_hits = trace_particles_boozer(
       field=field,
       stz_inits=points,
       parallel_speeds=vpars,
       tmax=1e-3,
       solveSympl=True,      # Enable symplectic solver
       dt=1e-8,              # Fixed step size
       roottol=1e-10,        # Root finding tolerance
       predictor_step=True    # Use predictor step
   )

Solver Options Reference
------------------------

Default Solver Options
~~~~~~~~~~~~~~~~~~~~~~

The default solver (``solveSympl=False``) uses adaptive timestepping with these parameters:

- **reltol**: Relative tolerance for adaptive timestepping (default: 1e-9)
- **abstol**: Absolute tolerance for adaptive timestepping (default: 1e-9)

Symplectic Solver Options
~~~~~~~~~~~~~~~~~~~~~~~~~

For the symplectic solver (``solveSympl=True``), use these parameters:

- **dt**: Fixed step size (required, typically 1e-8)
- **roottol**: Root finding tolerance (typically 1e-10)
- **predictor_step**: Use predictor step for better convergence (True)
