Coulomb Collisions
==================

FIRM3D can advance guiding center orbits together with Monte Carlo Coulomb
collisions against one or more Maxwellian background species. This is the
model needed for slowing down calculations, thermalization, and collisional
transport of energetic particles, which the collisionless tracers described in
:doc:`guiding_center` cannot capture: without collisions there is no energy
exchange with the background plasma.

The scheme follows Hirvijoki et al., *Phys. Plasmas* **20**, 092505 (2013),
whose Eqs. (32)--(33) define the stochastic differential equations solved
here, and the ASCOT5 implementation of them.

Background Species
------------------

Each background species is described by a ``ThermalBackground``: density
(m\ :sup:`-3`) and temperature (eV) profiles as callables of the normalized
toroidal flux ``s``, plus a mass and a signed charge.

.. code-block:: python

   from firm3d.field.collisions import ThermalBackground
   from firm3d.util.constants import PROTON_MASS, ELEMENTARY_CHARGE

   n0 = 1e20   # m^-3
   T0 = 10e3   # eV (10 keV)

   deuterium = ThermalBackground(
       n_profile=lambda s: n0 * (1 - 0.9 * s**2),
       T_profile=lambda s: T0 * (1 - 0.9 * s),
       mass=2 * PROTON_MASS,
       charge=ELEMENTARY_CHARGE,
   )

The profiles are pre-evaluated on a uniform grid in ``s`` (``n_grid_points``,
default 512) and linearly interpolated by the C++ layer, so they are called
once at setup rather than once per step. At most ``firm3dpp.COLL_MAX_SPECIES``
(currently 8) species may be given; their collision coefficients are summed.

Coulomb Logarithm
~~~~~~~~~~~~~~~~~

The Coulomb logarithm is not a user input. It is computed locally from the
profiles as the ratio of the Debye length to the minimum impact parameter,

.. math::

   \ln\Lambda_{ab} = \ln\left(\frac{\lambda_D}{b_\mathrm{min}}\right),
   \qquad b_\mathrm{min} = \max(b_\mathrm{cl}, b_\mathrm{qm}),

.. math::

   b_\mathrm{cl} = \frac{|q_a q_b|}{4\pi\varepsilon_0 m_r v_\mathrm{eff}^2},
   \qquad
   b_\mathrm{qm} = \frac{\hbar}{2 m_r v_\mathrm{eff}},
   \qquad
   v_\mathrm{eff}^2 = v^2 + v_{\mathrm{th},b}^2

where :math:`\lambda_D` is the total Debye length over all species and
:math:`m_r = m_a m_b/(m_a + m_b)` is the reduced mass. Taking the larger of
the classical 90-degree deflection radius and the de Broglie wavelength covers
the quantum regime, which fast energetic particles reach against electrons.

Because :math:`\ln\Lambda_{ab}` follows from the profiles, badly chosen
profiles can drive it non-positive, where the binary collision model is
undefined. The entry points scan the profile grid before tracing and

- raise ``ValueError`` if :math:`\ln\Lambda \le 0` anywhere, and
- warn once if :math:`0 < \ln\Lambda < 2`, where the approximation is marginal.

Since :math:`\lambda_D \propto \sqrt{T/n}` and :math:`b_\mathrm{cl} \propto 1/T`
at low speed, :math:`\ln\Lambda` falls like :math:`\ln(T^{3/2}/\sqrt{n})`: these
thresholds are reached for cold, dense profiles, the strongly coupled regime
where binary collisions stop being the right picture. A profile that merely
tapers to zero at the edge is fine, because species with :math:`n \le 0` or
:math:`T \le 0` are treated as inactive and skipped.

Collision Coefficients
----------------------

With :math:`x = v/v_{\mathrm{th},b}`, the Chandrasekhar function

.. math::

   G(x) = \frac{\mathrm{erf}(x) - \frac{2x}{\sqrt{\pi}} e^{-x^2}}{2x^2}

and the Rosenbluth prefactor for species :math:`b`,

.. math::

   \Gamma_b = \frac{n_b q_a^2 q_b^2 \ln\Lambda_{ab}}{4\pi\varepsilon_0^2 m_a^2},

the pitch angle scattering frequency, parallel diffusion coefficient and
deterministic drag are

.. math::

   \nu_D^{(b)} = \frac{\Gamma_b [\mathrm{erf}(x) - G(x)]}{v^3},
   \qquad
   D_\parallel^{(b)} = \frac{\Gamma_b G(x)}{v},
   \qquad
   Q^{(b)} = -\frac{m_a v}{T_b} D_\parallel^{(b)},

with :math:`Q^{(b)}` fixed by the Einstein relation. The total drift in speed
adds the spurious drift terms that convert the Itô equation to the physical
one,

.. math::

   K^{(b)} = Q^{(b)}
             + \frac{\partial D_\parallel^{(b)}}{\partial v}
             + \frac{2 D_\parallel^{(b)}}{v}.

All three coefficients are additive over species; the Debye length in
:math:`\ln\Lambda_{ab}` is computed from all species at once, before the
per-species sum.

Stochastic Differential Equations
---------------------------------

In velocity space the state is the pair :math:`(v, \xi)`, where
:math:`v = |\mathbf{v}|` and :math:`\xi = v_\parallel/v` is the pitch:

.. math::

   dv = K(v, s)\, dt + \sqrt{2 D_\parallel}\, dW_v

   d\xi = \dot\xi_\mathrm{mirror}\, dt - \xi \nu_D\, dt
          + \sqrt{\nu_D (1 - \xi^2)}\, dW_\xi

where :math:`dW_v` and :math:`dW_\xi` are independent Wiener increments with
:math:`\mathbb{E}[dW^2] = dt`. The mirror force drift
:math:`\dot\xi_\mathrm{mirror}` is orbit physics, handled by the orbit stepper;
the collision operator contributes the remaining terms. The noise amplitude in
:math:`\xi` is what makes the pitch distribution relax to a uniform
distribution on :math:`[-1, 1]` in the isotropization limit
:math:`\nu_D t \gg 1`.

Numerical Scheme
----------------

Operator Splitting
~~~~~~~~~~~~~~~~~~

The system is advanced by Lie--Trotter splitting:

1. **Orbit half.** Advance :math:`(s, \theta, \zeta, v_\parallel)` by one
   adaptive Dormand--Prince step of physical time :math:`h` at fixed
   :math:`\mu`. No collision term appears in this right-hand side.
2. **Collision half.** At the accepted state, convert
   :math:`(v_\parallel, \mu) \to (v, \xi)` using :math:`|B|` at the current
   position, apply the drift and Milstein noise over the window :math:`h`, and
   convert back, which yields the updated :math:`\mu`.

Putting the whole collision operator in the second half is what makes
:math:`\mu` exactly conserved by the first. :math:`\mu` is then a parameter of
the orbit equations rather than a state variable, so any static-field guiding
center right-hand side can be used unchanged -- which is why ``gc_vac``,
``gc_noK`` and full ``gc`` are all supported for unperturbed tracing.

Sub-Cycling
~~~~~~~~~~~

The orbit stepper sizes :math:`h` from orbit dynamics alone, so the collision
rates have no influence on it. Applying them as a single explicit Euler kick
over the whole step is inaccurate whenever the rates are fast compared with
:math:`h`, which is the thermal regime, where :math:`\nu_D` and :math:`K`
diverge as :math:`v \to 0`. The kick is therefore subdivided into :math:`n`
sub-steps,

.. math::

   n = \max\left(\frac{\nu_D h}{\varepsilon},\;
                 \frac{|K| h}{\varepsilon v},\;
                 \left[\frac{\sqrt{2 D_\parallel h}}{\varepsilon v}\right]^2
           \right),
   \qquad \varepsilon = 0.05,

capped at :math:`10^4`. The drift terms scale as :math:`h` and so fall
linearly in :math:`n`, while the diffusive excursion scales as :math:`\sqrt{h}`
and so falls only as :math:`\sqrt{n}`, which is why it enters squared.
Sub-cycling is cheap because the position is frozen during a kick, so a sub-step
costs a coefficient evaluation and no field evaluation. For 3.52 MeV alphas
:math:`n = 1` throughout; only thermal speed particles sub-cycle.

Boundary Conditions
~~~~~~~~~~~~~~~~~~~

Following ASCOT5, the speed reflects off a thermal cutoff,

.. math::

   v < v_\mathrm{cut} \;\Rightarrow\; v \leftarrow 2 v_\mathrm{cut} - v,
   \qquad v_\mathrm{cut} = 0.1 \sqrt{T_\mathrm{min}/m_a},

where :math:`T_\mathrm{min}` is the coldest active background species, so the
coefficients are never evaluated deep below the thermal speed. The pitch
mirrors at its boundary, :math:`|\xi| > 1 \Rightarrow \xi \leftarrow
\mathrm{sign}(\xi)(2 - |\xi|)`; a hard clamp would pile up probability at
exactly :math:`\xi = \pm 1`.

Entry Points
------------

.. list-table::
   :header-rows: 1
   :widths: 34 22 44

   * - Function
     - Module
     - Notes
   * - ``trace_particles_boozer_with_collisions``
     - ``firm3d.field.collisions``
     - Unperturbed. ``gc``, ``gc_vac``, ``gc_noK``. Takes ``Ekin``.
   * - ``trace_particles_boozer_perturbed_with_collisions``
     - ``firm3d.field.collisions``
     - Shear Alfvén wave. ``gc_vac`` and ``gc_noK`` only. Takes
       ``(parallel_speeds, mus)``.
   * - ``trace_particles_boozer_with_collisions_gpu``
     - ``firm3d.catapult.tracing``
     - Final state only, no stopping criteria, scalar ``vtotal``.
   * - ``trace_particles_cartesian_with_collisions_gpu``
     - ``firm3d.catapult.tracing``
     - As above, in Cartesian coordinates; takes a ``flux_label`` callable.

All four accept a single ``ThermalBackground`` or a list of them.

On the CPU entry points, ``ode_solver`` may be ``"dormand_prince"``
(recommended) or ``"boost"``; ``"symplectic"`` is collisionless-only and is
rejected. The GPU tracer exposes a single ``tol``. Unlike the collisionless
tracers, the collisional entry points do not take the coordinate-plane
(``phases``/``n_zetas``/``m_thetas``/``omegas``) or parallel-speed (``vpars``)
crossing detectors, so ``res_hits`` records stopping criterion hits only.

``DP_hmin`` (default ``0.0``) sets a floor on the orbit step size in seconds;
:math:`10^{-10}`--:math:`10^{-9}` s keeps the solver from stalling at bounce
points. It bounds the *orbit* stepper only -- the stiffness the collision
terms introduce as :math:`v \to 0` is handled by sub-cycling instead.

Each particle uses RNG seed ``rng_seed + i``, so MPI runs are reproducible
regardless of how particles are distributed over ranks. The GPU tracer instead
uses a counter-based Philox stream keyed on the global particle index: also
reproducible, but a different stream, so the CPU and GPU tracers agree
statistically rather than element by element.

Output Format
-------------

Each entry of ``res_tys`` is an array of shape ``(2, 6)`` holding the initial
and final state, with columns

.. code-block:: text

   [t, s, theta, zeta, v_par, v]

The final column ``v`` is the extra one relative to the collisionless tracers.

The GPU entry point returns a single ``(nparticles, 7)`` array of final states,
``[t, s, theta, zeta, v_par, v, dt]``, with the final step size in the last
column as in ``trace_particles_boozer_gpu``.

Usage Examples
--------------

Unperturbed Collisional Tracing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np

   from firm3d.field.boozermagneticfield import (
       BoozerRadialInterpolant,
       InterpolatedBoozerField,
   )
   from firm3d.field.collisions import (
       ThermalBackground,
       trace_particles_boozer_with_collisions,
   )
   from firm3d.field.tracing import MaxToroidalFluxStoppingCriterion
   from firm3d.field.tracing_helpers import (
       initialize_position_profile,
       initialize_velocity_uniform,
   )
   from firm3d.util.constants import (
       ALPHA_PARTICLE_CHARGE,
       ALPHA_PARTICLE_MASS,
       ELECTRON_MASS,
       ELEMENTARY_CHARGE,
       FUSION_ALPHA_PARTICLE_ENERGY,
       PROTON_MASS,
   )

   bri = BoozerRadialInterpolant("boozmn_equilibrium.nc", order=3, no_K=True)
   field = InterpolatedBoozerField(
       bri, degree=3, ns_interp=48, ntheta_interp=48, nzeta_interp=48
   )

   # A 50/50 DT plasma. Densities in m^-3, temperatures in eV; both stay
   # finite at s = 1 so that ln(Lambda) remains well defined there.
   n0 = 1e20
   T0 = 10e3

   def n_ion(s):
       return 0.5 * n0 * (1 - 0.9 * s**2)

   def temperature(s):
       return T0 * (1 - 0.9 * s)

   deuterium = ThermalBackground(
       n_profile=n_ion,
       T_profile=temperature,
       mass=2 * PROTON_MASS,
       charge=ELEMENTARY_CHARGE,
   )
   tritium = ThermalBackground(
       n_profile=n_ion,
       T_profile=temperature,
       mass=3 * PROTON_MASS,
       charge=ELEMENTARY_CHARGE,
   )
   electrons = ThermalBackground(
       n_profile=lambda s: 2 * n_ion(s),  # quasineutrality
       T_profile=temperature,
       mass=ELECTRON_MASS,
       charge=-ELEMENTARY_CHARGE,
   )

   nParticles = 1000
   points = initialize_position_profile(field, nParticles, lambda s: 1 - s, comm=None)

   Ekin = FUSION_ALPHA_PARTICLE_ENERGY
   v0 = np.sqrt(2 * Ekin / ALPHA_PARTICLE_MASS)
   vpar_init = initialize_velocity_uniform(v0, nParticles, comm=None)

   res_tys, res_hits = trace_particles_boozer_with_collisions(
       field=field,
       stz_inits=points,
       parallel_speeds=vpar_init,
       backgrounds=[deuterium, tritium, electrons],
       tmax=1e-1,
       mass=ALPHA_PARTICLE_MASS,
       charge=ALPHA_PARTICLE_CHARGE,
       Ekin=Ekin,
       comm=None,
       stopping_criteria=[MaxToroidalFluxStoppingCriterion(1.0)],
       DP_hmin=1e-10,
       rng_seed=42,
   )

   # Slowing down: recover the energy history from the sixth column.
   energies = [0.5 * ALPHA_PARTICLE_MASS * ty[:, 5] ** 2 for ty in res_tys]

Perturbed Collisional Tracing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Collisions can be combined with a shear Alfvén wave. The wave does work on the
particle, so the speed is not a constant of the motion and the initial velocity
is given as :math:`(v_\parallel, \mu)` rather than an energy, exactly as for
``trace_particles_boozer_perturbed``.

.. code-block:: python

   from firm3d.field.boozermagneticfield import ShearAlfvenHarmonic
   from firm3d.field.collisions import (
       trace_particles_boozer_perturbed_with_collisions,
   )

   saw = ShearAlfvenHarmonic(-1.50119e3, m=1, n=1, omega=136041, phase=0, B0=field)

   field.set_points(points)
   mu_init = (v0**2 - vpar_init**2) / (2 * field.modB()[:, 0])

   res_tys, res_hits = trace_particles_boozer_perturbed_with_collisions(
       perturbed_field=saw,
       stz_inits=points,
       parallel_speeds=vpar_init,
       mus=mu_init,
       backgrounds=[deuterium, tritium, electrons],
       tmax=1e-3,
       mass=ALPHA_PARTICLE_MASS,
       charge=ALPHA_PARTICLE_CHARGE,
       comm=None,
       stopping_criteria=[MaxToroidalFluxStoppingCriterion(1.0)],
       rng_seed=42,
   )

GPU Collisional Tracing
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from firm3d.catapult.tracing import trace_particles_boozer_with_collisions_gpu

   final_states = trace_particles_boozer_with_collisions_gpu(
       field=field,
       stz_inits=points,
       parallel_speeds=vpar_init,
       backgrounds=[deuterium, tritium, electrons],
       tmax=1e-2,
       mass=ALPHA_PARTICLE_MASS,
       charge=ALPHA_PARTICLE_CHARGE,
       vtotal=v0,
       tol=1e-8,
       ns=48,
       ntheta=48,
       nzeta=48,
       rng_seed=42,
   )

   # (nparticles, 7): [t, s, theta, zeta, v_par, v, dt]
   s_final = final_states[:, 1]
   v_final = final_states[:, 5]

Note that ``vtotal`` is a single scalar that sets the initial speed of every
particle, and :math:`\mu` is derived from it as
:math:`(v_\mathrm{total}^2 - v_\parallel^2)/(2|B|)`. Passing
:math:`|v_\parallel| > v_\mathrm{total}` would give a negative :math:`\mu` and
is rejected. The GPU tracer requires ``field.field_type`` to be ``"vac"`` or
``""``.

GPU Collisional Tracing in Cartesian Coordinates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The thermal profiles are functions of the flux label :math:`s`, which the
Cartesian state does not carry. ``trace_particles_cartesian_with_collisions_gpu``
therefore takes a ``flux_label`` callable mapping cylindrical points
``(r, phi, z)`` to :math:`s`; its values are interpolated on the same grid as
the magnetic field and evaluated at the particle position after each accepted
orbit step, so the same 1D profiles serve both coordinate systems. With an
equilibrium available, the label can be built from
:class:`~firm3d.field.coordinates.BoozerCoordinateTransformer`; it must
return finite values on the whole grid box, and values above 1 outside the
last closed flux surface are clamped by the profile lookup.

.. code-block:: python

   from firm3d.catapult.tracing import trace_particles_cartesian_with_collisions_gpu

   final_states = trace_particles_cartesian_with_collisions_gpu(
       field=bsh,                        # simsopt InterpolatedField
       surface_classifier=sc_particle,   # simsopt SurfaceClassifier
       flux_label=s_of_rphiz,            # (N, 3) cylindrical points -> s
       xyz_inits=xyz,
       parallel_speeds=vpar_init,
       backgrounds=[deuterium, tritium, electrons],
       tmax=1e-2,
       mass=ALPHA_PARTICLE_MASS,
       charge=ALPHA_PARTICLE_CHARGE,
       vtotal=v0,
       tol=1e-8,
       rng_seed=42,
   )

   # (nparticles, 7): [t, x, y, z, v_par, v, dt]
   v_final = final_states[:, 5]

Cross-Code Benchmark: ARIES-CS
------------------------------

The collision operator is benchmarked against ASCOT5 on a shared birth
ensemble, so results compare marker by marker and not only
statistically.

The case is the ARIES-CS reactor equilibrium (``n3are_R7.75B5.7``, NFP = 3,
:math:`R_0 \approx 7.75` m, :math:`|B| \approx 5.7` T), with 10\ :sup:`4`
fusion-born 3.52 MeV alphas traced for 0.15 s against the D--T--electron
background

.. math::

   n_D(s) = n_T(s) = 10^{20}\,(1 - s^5)\ \mathrm{m^{-3}},
   \qquad
   T(s) = 11.5\,(1 - s) + 0.1\ \mathrm{keV}.

FIRM3D uses a Boozer spline resolution ``ns = ntheta = nzeta = 96`` and orbit
tolerance ``tol = 1e-10``, with the loss boundary at :math:`s = 1`. Both were
converged by scanning resolution 32--96 and tolerance
10\ :sup:`-8`--10\ :sup:`-10`: the loss fraction varies by a few markers
across that range, less than its own binomial uncertainty.

.. image:: figures/acs_collisional_timetraces.png
   :width: 100%
   :alt: Cumulative particle and energy loss against time, FIRM3D CPU and GPU against ASCOT5

.. list-table:: ARIES-CS collisional, N = 10\ :sup:`4`
   :header-rows: 1
   :widths: 40 30 30

   * -
     - Particle loss
     - Energy loss
   * - FIRM3D CPU
     - 29.00 %
     - 20.25 %
   * - FIRM3D GPU
     - 29.20 %
     - 20.30 %
   * - ASCOT5
     - 29.47 %
     - 20.46 %

The CPU and GPU entry points agree to 0.31 :math:`\sigma`. They are comparable
only statistically, never element by element, because the two draw from
different generators (Philox against mt19937_64) and take different step
sequences.
