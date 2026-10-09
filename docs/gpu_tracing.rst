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

This is one uninterrupted integration launch, including for
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
When there are no intermediate save times (including endpoint-only tracing),
the tracer captures the last accepted step and evaluates its continuous
extension in a short second GPU kernel. This keeps the interpolation's
register footprint out of the integration loop. The temporary step buffer
uses another ``29 * nparticles`` values in the field's precision on the GPU:
23.2 MB for 100,000 particles in double precision, or 11.6 MB in single
precision. It is freed before returning; no step history is retained.
For large ensembles, use ``forget_exact_path=True`` when only endpoints are
needed, or trace smaller batches.

Saving also adds interpolation, transfer, and host assembly work. For
10,000 particles on one Perlmutter A100 80 GB GPU, the following timings
describe the initial dense-output implementation at ``6e0fcb4a``. These
double-precision runs use ``tmax=1e-4``, ``tol=1e-8``, the bundled ATEN
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

A separate Nsight Systems profile of that initial implementation measured
the integration kernel at
approximately 0.140 s for endpoints only and 0.154 s for 1000 samples,
about 10% more GPU execution time. Most of the full-call overhead in
this case comes from output transfer and host trajectory assembly.

Comparison with the previous saving method
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The previous implementation restarted tracing at every save interval,
passing the last step size and magnetic moment to the next launch. It
saved the first accepted endpoint at or after each requested time and
skipped save intervals already passed by an earlier step.

A separate double-precision comparison on one Perlmutter A100 40 GB GPU used
the same 10,000 initial particles, field table, ``tmax=1e-4``, and ``tol=1e-8`` for
both implementations. Timings are medians of three warm calls and include
the full public API. Dense-output timings in this comparison describe the
initial implementation at ``6e0fcb4a``. The baseline GPU kernels and Python
tracing code are from the parent revision ``76089f7a``. Its byte-identical
CPU tracer object was reused when scratch-storage access stalled its
compilation.

.. list-table:: Previous saving versus dense output, measured on 2026-10-08
   :header-rows: 1

   * - Requested samples per survivor
     - Previous method (s)
     - Dense output (s)
     - Speedup with dense output
   * - Endpoints only
     - 0.181
     - 0.212
     - 0.85
   * - 10
     - 0.344
     - 0.294
     - 1.17
   * - 100
     - 1.788
     - 0.369
     - 4.84
   * - 1000
     - 15.064
     - 1.227
     - 12.27

At 1000 requested samples, the previous method returned a median of
839 samples per survivor (10th--90th percentiles: 623--976); every
survivor missed at least one requested sample. Dense output returned
all 1000. The previous method used 1000 native tracing calls; dense
output used one. Previous-method sample times were off the requested
grid, and final times overshot ``tmax`` by a median of 51 ns, with a
90th percentile of 128 ns. Double-precision dense output ended exactly
at ``tmax`` and changing the save interval left terminal states identical.

The tradeoff is GPU output storage: the previous method used at most
0.56 MB per launch, while dense output used 5.6, 56, and 560 MB for
10, 100, and 1000 samples respectively. Both methods also require field
and integration storage and retain the returned trajectories on the host.
For single precision, the 1000-sample case took 14.423 s previously and
0.952 s with dense output, a 15.16-fold speedup, with a 280 MB dense buffer.

Endpoint-only calls in that initial implementation were approximately 17%
slower.
Their native binding times were nearly unchanged (0.144 s versus
0.146 s in double precision); most of the additional time was in host
result assembly. The saving speedups therefore should not be interpreted
as an endpoint-only speedup. The endpoint-only host assembly was subsequently
changed to build the two-row trajectories together, avoiding per-particle
filtering and stacking. See the endpoint-only comparison below for the
performance of this updated path.

Final-state comparisons were also repeated with dense output evaluated at
each particle's previous-method terminal time, using the 9839 particles that
survived in both runs. For 1000 saves, the median, 90th percentile, and maximum
absolute circular differences in ``theta`` were ``4.5e-14``, ``6.0e-4``, and
``0.98`` radians in double precision, and ``2.3e-5``, ``6.6e-4``, and ``0.45``
in single precision. The previous method reported three additional survivors
at this save cadence. Differences were not solely due to endpoint overshoot;
restarting also changed the computed trajectory.
These are differences between the two methods, not errors against an
independent CPU reference.

To repeat the comparison, run the benchmark once with the previous
checkout's ``src`` directory in ``PYTHONPATH`` and ``--method existing``,
then with the new checkout and ``--method dense``. The option verifies
the loaded implementation; it does not select a different algorithm within
one build. The JSON records launch counts, timing, grid offsets, and sample
counts, and the accompanying NPZ stores terminal states for comparisons.

Endpoint-only comparison with master
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With ``forget_exact_path=True``, trajectories are assembled together on the
host. A separate endpoint specialization captures the enclosing step and
runs its dense interpolation after integration, keeping the polynomial's
register footprint out of the integration loop. State construction now
assigns each component to one thread, avoiding repeated calculations in
other lanes. Inactive particle slots do not read uninitialized stages.

A comparison on 2026-10-09 used current upstream master ``76089f7a`` and the
updated branch on one Perlmutter A100 80 GB GPU. Both builds ran in persistent
processes with identical initial particles, the ATEN equilibrium, a
``15 x 15 x 15`` table, and ``tol=1e-8``. Each case had one warm-up followed by
three measured calls per build, alternating their order. Field tabulation
and garbage collection were excluded; GPU allocation, transfer, and host
result assembly were included. The baseline used independently compiled
GPU kernels and bindings, with the byte-identical CPU tracer object reused
as described above.

All 22 combinations of ensemble size, integration time, and precision had
unchanged or lower median full-call times. The 1 ms and 10 ms cases improved
by 4.9--10.8%. Selected timings follow; these are wall-clock seconds, and
``tmax`` is the physical integration duration in seconds.

.. list-table:: Endpoint-only full public call
   :header-rows: 1

   * - Precision
     - Particles
     - tmax (s)
     - Master (s)
     - Updated branch (s)
   * - Double
     - 1,000
     - 1e-06
     - 0.00746
     - 0.00428
   * - Double
     - 10,000
     - 0.0001
     - 0.18352
     - 0.13940
   * - Double
     - 100,000
     - 0.0001
     - 1.20816
     - 0.83961
   * - Double
     - 100,000
     - 0.001
     - 8.34569
     - 7.55746
   * - Double
     - 10,000
     - 0.01
     - 14.14503
     - 13.25480
   * - Single
     - 1,000
     - 1e-06
     - 0.00678
     - 0.00301
   * - Single
     - 10,000
     - 0.0001
     - 0.14665
     - 0.11082
   * - Single
     - 100,000
     - 0.0001
     - 1.05046
     - 0.63827
   * - Single
     - 100,000
     - 0.001
     - 6.31722
     - 5.63245
   * - Single
     - 10,000
     - 0.01
     - 10.38244
     - 9.67437

The small single-precision case with 1,000 particles and ``tmax=1e-4`` was
approximately unchanged: 0.0774 s on master and 0.0772 s on the branch.
Its native binding was slightly slower (0.0731 s versus 0.0766 s); faster
host assembly offset that cost. Endpoint interpolation still adds work, so
the full-call gains should not be interpreted as zero native overhead for
every workload.

A separate nine-repeat check of that 1,000-particle case on an A100 40 GB
GPU also found lower full-call medians: 0.1307 s versus 0.1206 s in double
precision, and 0.0827 s versus 0.0770 s in single precision. This check used
three warm-ups and alternating call order on the same GPU for both builds.

The 33 GPU tests passed, including comparisons between endpoint-only and
history terminal states for Cartesian, equilibrium, and wave fields.
Compute Sanitizer memory, initialization, and race checks passed on the
loss/zero-duration/work-stealing case and the multiple-sample case, with
zero errors or hazards. The timing comparison covers the ATEN vacuum field;
correctness checks also cover the other supported kernels.

To repeat the paired comparison with a separately built master checkout:

.. code-block:: console

    python examples/benchmark_gpu_endpoints.py --baseline /path/to/master \
        --repeats 3 --warmups 1 --output gpu-endpoints.json

The default cases cover 1,000--100,000 particles and durations from 1 microsecond
to 10 milliseconds, in both precisions. JSON retains individual measurements,
native-call timings, loaded source paths, and source/build hashes.

Controlled cost of dense endpoints
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The comparison with master combines interpolation with changes that remove
redundant GPU calculations and reduce host assembly cost. To isolate endpoint
interpolation, a second comparison on 2026-10-09 used two builds of the updated
code at ``82af7f82``. Both retained those optimizations, the double-precision
integration clock, and identical Python tracing code. The control removed only
terminal-step capture, its temporary allocations, and the interpolation kernel,
and returned the first accepted endpoint at or beyond ``tmax``. The normal build
returned the dense endpoint at exactly ``tmax``. Both calls used
``forget_exact_path=True``.

This comparison used one Perlmutter A100 40 GB GPU, the same ATEN equilibrium,
table, particles, and tolerance as above, two warm-ups and five measured calls
per build, and alternating call order. It covered 16 combinations of precision,
1,000--100,000 particles, and durations of 1 microsecond, 0.1 millisecond, and
1 millisecond. The ATEN endpoint integration kernel's register count was
identical in the two builds: 88 in double precision and 90 in single precision.

For the 1 millisecond traces, the full-call median differences were between
-0.51% and +0.58%. At 100,000 particles, interpolation added approximately
16 milliseconds to calls taking 5.6--7.6 seconds, or 0.21--0.28%.
These small differences are comparable to the variation between repeated
calls and should be treated as workload-specific estimates.

.. list-table:: Controlled endpoint-only full public call
   :header-rows: 1

   * - Precision
     - Particles
     - tmax (s)
     - Raw endpoint (s)
     - Dense endpoint (s)
     - Difference
   * - Double
     - 1,000
     - 0.001
     - 1.07022
     - 1.07445
     - +0.39%
   * - Double
     - 10,000
     - 0.001
     - 1.32415
     - 1.32908
     - +0.37%
   * - Double
     - 100,000
     - 0.001
     - 7.55068
     - 7.56631
     - +0.21%
   * - Single
     - 1,000
     - 0.001
     - 0.68300
     - 0.68697
     - +0.58%
   * - Single
     - 10,000
     - 0.001
     - 0.97066
     - 0.96568
     - -0.51%
   * - Single
     - 100,000
     - 0.001
     - 5.62857
     - 5.64445
     - +0.28%

Shorter cases had substantial timing variation. For example, the initial
five-repeat medians suggested +30% for 100,000 particles at 1 microsecond in
double precision and +21% for 1,000 particles at 0.1 millisecond in single
precision. A follow-up on an A100 80 GB GPU used five warm-ups and fifteen
measured calls per build; the same cases then measured +2.32% and +0.55%.
Both builds shared the same GPU within each comparison.

.. list-table:: Short cases, fifteen-repeat full-call medians
   :header-rows: 1

   * - Precision
     - Particles
     - tmax (s)
     - Raw endpoint (s)
     - Dense endpoint (s)
     - Difference
   * - Double
     - 10,000
     - 1e-06
     - 0.00991
     - 0.01014
     - +2.24%
   * - Double
     - 100,000
     - 1e-06
     - 0.07393
     - 0.07564
     - +2.32%
   * - Double
     - 1,000
     - 0.0001
     - 0.10680
     - 0.11205
     - +4.92%
   * - Double
     - 100,000
     - 0.0001
     - 0.83376
     - 0.83555
     - +0.21%
   * - Single
     - 1,000
     - 0.0001
     - 0.06957
     - 0.06996
     - +0.55%
   * - Single
     - 100,000
     - 0.0001
     - 0.63476
     - 0.63896
     - +0.66%

Other short-case medians in the follow-up ranged from -3.31% to -1.11%; some
changed sign between runs. Individual short calls also showed intermittent
spikes in both builds. These data do not establish a precise universal
interpolation overhead for short traces. Allocation and terminal processing
costs can make a larger relative contribution when integration is short.

The comparison includes allocations, transfers, the endpoint interpolation
kernel, and public result assembly. Native-call timing includes allocations
and transfers too; it is not a measurement of GPU kernel execution alone.
With histories disabled, the additional GPU storage holds one terminal step
per particle (``29 * Nparticles * sizeof(T)``), independent of integration
duration. No additional right-hand-side evaluations are used for interpolation.

The control passed analytic Cartesian checks for 50,000 particles including
losses and zero-duration traces, in both precisions, and Compute Sanitizer
reported zero memory errors. Dense-history CPU agreement and save-cadence
independence tests also passed with the control build. Its overshooting
endpoint is intentional and should not be used for exact-time results.

To reproduce, apply ``examples/benchmark_gpu_endpoints_raw_control.patch`` to
an isolated copy of the feature checkout and build it with the same CUDA
configuration. Run from the normal feature checkout with:

.. code-block:: console

    python examples/benchmark_gpu_endpoints.py --baseline /path/to/raw-control \
        --baseline-kind raw_endpoint --repeats 5 --warmups 2 \
        --cases 10000:1e-6 100000:1e-6 1000:1e-4 10000:1e-4 100000:1e-4 \
            1000:1e-3 10000:1e-3 100000:1e-3 --output gpu-endpoint-ablation.json

``--baseline-kind`` labels and checks the baseline's API; it does not change
the algorithm in the loaded extension. The JSON records both extension hashes
and the matching Python source hashes along with individual measurements.

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
