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
        charge=charge, stopping_criteria=[MaxToroidalFluxStoppingCriterion(1.0)],
        forget_exact_path=True,
    )
    times, loss_fraction = compute_loss_fraction(res_tys)

The kernels stop a particle at :math:`s = 1` (or at the surface of the
classifier) and nowhere else, so ``stopping_criteria`` must be ``None`` or
``[MaxToroidalFluxStoppingCriterion(1.0)]``. ``Ekin`` is one value for all
particles. There is no ``comm``: an ensemble is traced on one GPU per call.
Options the kernels do not have (``abstol`` and ``reltol`` apart from ``tol``,
the Poincaré-section arguments, the choice of solver) are not accepted.

Differences from the CPU tracers that remain: the last row of a lost
particle is the state just past the crossing rather than the last state
inside; a survivor's final time can exceed ``tmax`` by up to one step;
:math:`\theta` is returned in :math:`(-\pi, \pi]` and :math:`\zeta` wrapped
to :math:`[0, 2\pi)`. Trajectories (``forget_exact_path=False``) are saved
by tracing in chunks of ``dt_save``: each row is the state at the first step
boundary at or after its save time, so its time is up to one step late, and a
save time that falls inside one step gets no row. Choose ``dt_save`` above the
step size. Trajectories cannot yet be saved this way in a perturbed field,
whose phase restarts with each chunk.

``advance_particles_*_gpu`` are the kernel-level calls the tracers are built
on: one launch, returning the state ``(t, s, theta, zeta, vpar, dt, mu)`` of
each particle, which can be fed back in to continue it.

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

The GPU test suite holds these statements: the derivative comparison for
every field type, and single-against-double tracing for the equilibrium,
finite-beta, perturbed and Cartesian kernels against the tolerance envelope
described above.
