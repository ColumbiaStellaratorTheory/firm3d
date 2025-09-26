Guiding Center Integration
==========================

Guiding center integration in Boozer coordinates is performed using equations of motion obtained from the Littlejohn Lagrangian:

.. math::

   L(\psi,\theta,\zeta,\rho_{||})  = q\left(\left[\psi + I  \rho_{||}\right] \dot{\theta} + \left[- \chi + G \rho_{||} \right] \dot{\zeta} + \rho_{||} K \dot{\psi}\right)  - \frac{\rho_{||}^2 B_0^2q^2}{2m} - \mu B_0

where :math:`2\pi \psi` is the toroidal flux, :math:`2\pi \chi` is the poloidal flux, :math:`q` is the charge, :math:`m` is the mass, :math:`\rho_{||} = q v_{||}/(m B)` and the covariant form of the magnetic field is:

.. math::

   \textbf{B} = G(\psi) \nabla \zeta + I(\psi) \nabla \theta + K(\psi,\theta,\zeta) \nabla \psi.

See R. White, Theory of Tokamak Plasmas, Sec. 3.2.

The trajectory information is saved as :math:`(s,\theta,\zeta,v_{||})`, where :math:`s = \psi_0` is the toroidal flux normalized to its value on the boundary, :math:`2\pi\psi_0`.

Unperturbed Guiding Center Integration
-------------------------------------

The primary routine for unperturbed guiding center integration is ``trace_particles_boozer``. The mode can be specified as ``mode='gc_vac'``, ``mode='gc_noK'``, or ``mode='gc'``. Be default, the mode is determined from the ``BoozerMagneticField`` object.

Vacuum Mode (gc_vac)
~~~~~~~~~~~~~~~~~~~~

In the case of ``mode='gc_vac'`` we solve the guiding center equations under the vacuum assumption, i.e :math:`G =` const. and :math:`I = 0`:

.. math::

   \dot s = -B_{,\theta} \frac{m\left(\frac{v_{||}^2}{B} + \mu \right)}{q \psi_0}

   \dot \theta = B_{,s} m\frac{\frac{v_{||}^2}{B} + \mu}{q \psi_0} + \frac{\iota v_{||} B}{G}

   \dot \zeta = \frac{v_{||}B}{G}

   \dot v_{||} = -(\iota B_{,\theta} + B_{,\zeta})\frac{\mu B}{G}

where :math:`q` is the charge, :math:`m` is the mass, and :math:`v_\perp^2 = 2\mu B`.

General Mode (gc)
~~~~~~~~~~~~~~~~~

In the case of ``mode='gc'`` we solve the general guiding center equations for an MHD equilibrium:

.. math::

   \dot s = (I B_{,\zeta} - G B_{,\theta})\frac{m\left(\frac{v_{||}^2}{B} + \mu\right)}{\iota D \psi_0}

   \dot \theta = \left((G B_{,\psi} - K B_{,\zeta}) m\left(\frac{v_{||}^2}{B} + \mu\right) - C v_{||} B\right)\frac{1}{\iota D}

   \dot \zeta = \left(F v_{||} B - (B_{,\psi} I - B_{,\theta} K) m\left(\frac{v_{||}^2}{B}+ \mu\right)\right) \frac{1}{\iota D}

   \dot v_{||} = (CB_{,\theta} - FB_{,\zeta})\frac{\mu B}{\iota D}

   C = - \frac{m v_{||} K_{,\zeta}}{B}  - q \iota + \frac{m v_{||}G'}{B}

   F = - \frac{m v_{||} K_{,\theta}}{B} + q + \frac{m v_{||}I'}{B}

   D = (F G - C I)/\iota

where primes indicate differentiation wrt :math:`\psi`. In the case ``mode='gc_noK'``, the above equations are used with :math:`K=0`.

Perturbed Guiding Center Integration
-----------------------------------

The primary routine for perturbed guiding center integration is ``trace_particles_boozer_perturbed``.

Vacuum Mode (gc_vac)
~~~~~~~~~~~~~~~~~~~~

In the case of ``mode='gc_vac'`` we solve the guiding center equations under the vacuum assumption, i.e. :math:`G =` const. and :math:`I = 0`:

.. math::

   \dot s      = \left(-B_{,\theta} m \frac{\frac{v_{||}^2}{B} + \mu}{q} + \alpha_{,\theta}B v_{||} - \Phi_{,\theta}\right)\frac{1}{\psi_0}

   \dot \theta = B_{,\psi} m \frac{\frac{v_{||}^2}{B} + \mu}{q} + (\iota - \alpha_{,\psi} G) \frac{v_{||}B}{G} + \Phi_{,\psi}

   \dot \zeta  = \frac{v_{||}B}{G}

   \dot v_{||} = -\frac{B}{Gm} \Bigg(m\mu \left(B_{,\zeta} + \alpha_{,\theta}B_{,\psi}G + B_{,\theta}(\iota - \alpha_{,\psi}G)\right)

               + q \left(\dot\alpha G + \alpha_{,\theta}G\Phi_{,\psi} + (\iota - \alpha_{,\psi}G)\Phi_{,\theta} + \Phi_{,\zeta}\right)\Bigg)

               + \frac{v_{||}}{B} (B_{,\theta}\Phi_{,\psi} - B_{,\psi} \Phi_{,\theta})

where :math:`q` is the charge, :math:`m` is the mass, and :math:`v_\perp^2 = 2\mu B`.

General Mode (gc)
~~~~~~~~~~~~~~~~~

In the case of ``mode='gc'`` we solve the general guiding center equations for an MHD equilibrium:

.. math::

   \dot{s} = \Bigg(-G \Phi_{,\theta}q + I\Phi_{,\zeta}q
                    + B qv_{||}(\alpha_{,\theta}G-\alpha_{,\zeta}I)
                    + (-B_{,\theta}G + B_{,\zeta}I)
                    \left(\frac{mv_{||}^2}{B} + m\mu\right)\Bigg)\frac{1}{D \psi_0}

   \dot{\theta} = \Bigg(G q \Phi_{,\psi}
                   + B q v_{||} (-\alpha_{,\psi} G - \alpha G_{,\psi} + \iota)
                   - G_{,\psi} m v_{||}^2 + B_{,\psi} G \left(\frac{mv_{||}^2}{B} + m\mu\right)\Bigg)\frac{1}{D}

   \dot{\zeta} =  \Bigg(-I (B_{,\psi} m \mu + \Phi_{,\psi} q)
                    + B q v_{||} (1 + \alpha_{,\psi} I + \alpha I'(\psi))
                    + \frac{m v_{||}^2}{B} (B I'(\psi) - B_{,\psi} I)\Bigg)\frac{1}{D}

   \dot v_{||} = \Bigg(\frac{Bq}{m} \Big(-m \mu (B_{,\zeta}(1 + \alpha_{,\psi} I + \alpha I'(\psi))
                     + B_{,\psi} (\alpha_{,\theta} G - \alpha_{,\zeta} I)
                     + B_{,\theta} (\iota - \alpha G'(\psi) - \alpha_{,\psi} G)) \\
                     - q \Big(\dot{\alpha} \left(G + I (\iota - \alpha G'(\psi)) + \alpha G I'(\psi)\right)
                     + \left(\alpha_{,\theta} G - \alpha_{,\zeta} I\right) \Phi_{,\psi} \\
                     + \left(\iota - \alpha G_{,\psi} - \alpha_{,\psi} G\right) \Phi_{,\theta}
                     + \left(1 + \alpha I'(\psi) + \alpha_{,\psi} I\right) \Phi_{,\zeta}\Big) \Big) \\
                     + \frac{q v_{||}}{B} \left((B_{,\theta} G - B_{,\zeta} I) \Phi_{,\psi}
                     + B_{,\psi} \left(I \Phi_{,\zeta} - G \Phi_{,\theta}\right)\right) \\
                     + v_{||} \big(m \mu \left(B_{,\theta} G'(\psi) - B_{,\zeta} I'(\psi)\right) \\
                     + q \left(\dot \alpha \left(G'(\psi) I - G I'(\psi)\right) \\
                     + G'(\psi) \Phi_{,\theta} - I'(\psi)\Phi_{,\zeta}\right)\big)\Bigg)\frac{1}{D}

   D = q (G + I(-\alpha G'(\psi) + \iota) + \alpha G I'(\psi))
                + \frac{m v_{\|}}{B} \left(-G'(\psi)I + G I'(\psi)\right)

Usage Examples
--------------

Unperturbed Tracing
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from firm3d.field.boozermagneticfield import (
       BoozerRadialInterpolant,
       InterpolatedBoozerField,
   )
   from firm3d.field.tracing import trace_particles_boozer, MaxToroidalFluxStoppingCriterion
   from firm3d.field.tracing_helpers import initialize_position_profile, initialize_velocity_uniform
   from firm3d.util.constants import (
       ALPHA_PARTICLE_MASS,
       ALPHA_PARTICLE_CHARGE,
       FUSION_ALPHA_PARTICLE_ENERGY,
   )
   import numpy as np

   # Setup magnetic field from VMEC output
   boozmn_filename = "boozmn_equilibrium.nc"
   bri = BoozerRadialInterpolant(boozmn_filename, order=3, no_K=True)

   field = InterpolatedBoozerField(
       bri,
       degree=3,
       ns_interp=48,
       ntheta_interp=48,
       nzeta_interp=48,
   )

   # Initialize particle positions and velocities
   nParticles = 1000
   points = initialize_position_profile(field, nParticles, lambda s: 1-s, comm=None)

   Ekin = FUSION_ALPHA_PARTICLE_ENERGY
   vpar0 = np.sqrt(2 * Ekin / ALPHA_PARTICLE_MASS)
   vpar_init = initialize_velocity_uniform(vpar0, nParticles, comm=None)

   # Trace particles
   res_tys, res_hits = trace_particles_boozer(
       field=field,
       stz_inits=points,
       parallel_speeds=vpar_init,
       tmax=1e-3,
       mass=ALPHA_PARTICLE_MASS,
       charge=ALPHA_PARTICLE_CHARGE,
       comm=None,
       Ekin=Ekin,
       stopping_criteria=[MaxToroidalFluxStoppingCriterion(1.0)],
       forget_exact_path=True,
       abstol=1e-8,
       reltol=1e-8
   )

Perturbed Tracing
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from firm3d.field.boozermagneticfield import (
       BoozerRadialInterpolant,
       InterpolatedBoozerField,
       ShearAlfvenHarmonic,
   )
   from firm3d.field.tracing import trace_particles_boozer_perturbed, MaxToroidalFluxStoppingCriterion
   from firm3d.field.tracing_helpers import initialize_position_profile, initialize_velocity_uniform
   from firm3d.util.constants import (
       ALPHA_PARTICLE_MASS,
       ALPHA_PARTICLE_CHARGE,
       FUSION_ALPHA_PARTICLE_ENERGY,
   )
   import numpy as np

   # Setup equilibrium field
   boozmn_filename = "boozmn_equilibrium.nc"
   bri = BoozerRadialInterpolant(boozmn_filename, order=3, no_K=True)

   field = InterpolatedBoozerField(
       bri,
       degree=3,
       ns_interp=48,
       ntheta_interp=48,
       nzeta_interp=48,
   )

   # Create shear Alfvén wave perturbation
   Phihat = -1.50119e3
   saw = ShearAlfvenHarmonic(
       Phihat, m=1, n=1, omega=136041, phase=0, B0=field
   )

   # Initialize particles
   nParticles = 1000
   points = initialize_position_profile(field, nParticles, lambda s: 1-s, comm=None)

   Ekin = FUSION_ALPHA_PARTICLE_ENERGY
   vpar0 = np.sqrt(2 * Ekin / ALPHA_PARTICLE_MASS)
   vpar_init = initialize_velocity_uniform(vpar0, nParticles, comm=None)

   # Calculate magnetic moment
   field.set_points(points)
   mu_init = (vpar0**2 - vpar_init**2)/(2*field.modB()[:,0])

   # Trace particles with perturbation
   res_tys, res_hits = trace_particles_boozer_perturbed(
       perturbed_field=saw,
       stz_inits=points,
       parallel_speeds=vpar_init,
       mus=mu_init,
       tmax=1e-3,
       mass=ALPHA_PARTICLE_MASS,
       charge=ALPHA_PARTICLE_CHARGE,
       comm=None,
       stopping_criteria=[MaxToroidalFluxStoppingCriterion(1.0)],
       forget_exact_path=True,
       abstol=1e-8,
       reltol=1e-8
   )
