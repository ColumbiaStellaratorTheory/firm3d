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

With ``forget_exact_path=False``, the GPU evaluates the same Dormand–Prince
continuous extension used by the CPU's default Boost solver. The returned
path contains the initial state, every multiple of ``dt_save`` reached, and
the final state at ``tmax`` even when it falls between save times. Multiple
samples can lie inside one accepted step; changing ``dt_save`` does not
change the adaptive step sequence. Final-state-only tracing also evaluates
the state at ``tmax`` rather than returning a later step endpoint.

This is one uninterrupted GPU launch, including for
``trace_particles_boozer_perturbed_gpu(..., forget_exact_path=False,
dt_save=...)``. The waves retain their absolute phase. Each particle may have
its own ``tmax``. Lost particles still stop at the first accepted endpoint
beyond the boundary; GPU boundary detection does not locate crossings with
the CPU's root finder.

The lower-level ``save_trajectories_boozer_gpu`` and
``save_trajectories_cartesian_gpu`` return seven-column rows containing the
time, four state components, the enclosing accepted step size, and magnetic
moment. An interpolated row is not an integrator checkpoint. These functions
omit the initial state; the ``trace_particles_*_gpu`` functions add it and
return the usual five-column CPU format.

The native ``firm3dpp.*_gpu_tracing`` bindings return flat NumPy arrays in
the field's precision. This avoids constructing a Python scalar for every
saved value before converting the result back to an array.

Trajectory storage requires a buffer of approximately
``7 * nparticles * (ceil(max(tmax) / dt_save) + 1)`` values on the GPU and
host. Unused rows for lost particles are removed from the returned arrays.
For large ensembles, use ``forget_exact_path=True`` when only endpoints are
needed, or trace smaller batches.

Saving also adds interpolation, transfer, and host assembly work. For
10,000 particles on one Perlmutter A100 80 GB GPU, the following
double-precision timings use ``tmax=1e-4``, ``tol=1e-8``, the bundled ATEN
equilibrium, and a ``15 x 15 x 15`` field table. They are medians of three
warm runs of the public tracing API, including allocation and output
assembly, with field tabulation excluded. Sample counts exclude the
initial state and include ``tmax``; lost particles return fewer rows.

.. list-table:: Dense saving performance measured on 2026-10-08
   :header-rows: 1

   * - Samples per surviving particle
     - Wall time (s)
     - Relative to endpoints only
     - GPU output buffer (MB)
   * - Endpoints only
     - 0.209
     - 1.00
     - 0.56
   * - 10
     - 0.279
     - 1.34
     - 5.6
   * - 100
     - 0.367
     - 1.76
     - 56
   * - 1000
     - 1.183
     - 5.67
     - 560

The 1000-sample case returns about 398 MB of five-column trajectories.
Its native binding takes 0.299 s, with most of the remaining time in
host conversion and assembly. Returning NumPy arrays directly reduced
the full call from 5.565 s to 1.183 s compared with boxing every saved
value in a Python list. These timings describe this workload; the
relative cost depends on integration length, tolerance, and save cadence.
Run ``python examples/benchmark_gpu_saving.py --output gpu-saving.json``
from the repository root to measure another configuration.

A separate Nsight Systems profile measured the integration kernel at
approximately 0.140 s for endpoints only and 0.154 s for 1000 samples,
about 10% more GPU execution time. Most of the full-call overhead in
this case comes from output transfer and host trajectory assembly.

Single precision
----------------

Each field object takes ``precision="single"``, which stores the table in
``float32`` and runs the kernels in single precision; the initial conditions
are cast to match. This halves the field-table and native output-buffer
memory. The public five-column CPU-format trajectories remain ``float64``.

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

At matched output times, a well-resolved double-precision orbit can also
have a tolerance error below the rounding floor of the single-precision
field and state. Tightening the solver tolerance does not remove that
rounding error; it should be included when comparing individual saved
states at a common physical time.
