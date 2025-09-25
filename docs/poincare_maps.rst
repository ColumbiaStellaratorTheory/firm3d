Poincaré Maps
=============

Poincaré maps are powerful tools for analyzing particle trajectories in magnetic confinement devices. They provide a reduced-dimensional representation of particle motion by plotting the intersection points of trajectories with a specific surface or condition. FIRM3D provides three main classes for computing different types of Poincaré maps.

Overview
--------

The three main classes are:

1. **TrappedPoincare**: For trapped particles (bouncing between mirror points)
2. **PassingPoincare**: For passing particles in unperturbed fields
3. **PassingPerturbedPoincare**: For passing particles in fields with shear Alfvén wave perturbations

Trapped Poincaré Maps
---------------------

The ``TrappedPoincare`` class computes Poincaré maps for trapped particles that bounce between mirror points where :math:`v_{||} = 0`.

**Key Features:**

- Maps trajectories from one mirror point to the next full bounce period.
- Uses helical angle :math:`\eta` as the mapping coordinate, which is automatically determined based on field helicity. For example, if :math:`M = 0` (e.g., QP or OP), then :math:`\eta = \theta`. If :math:`N = 0` (e.g., QA/OA, QH/OA), then :math:`\eta = n_{fp} \zeta`.
- Initial conditions sampled for same value of :math:`\mu` (magnetic moment)

**Usage Example:**

.. code-block:: python

    from firm3d.field.trajectory_helpers import TrappedPoincare
    from firm3d.util.constants import (
        ALPHA_PARTICLE_MASS,
        ALPHA_PARTICLE_CHARGE,
        FUSION_ALPHA_PARTICLE_ENERGY,
    )

    # Define mirror point for trapped particles
    s_mirror = 0.5      # flux surface for mirroring
    theta_mirror = np.pi / 2  # poloidal angle for mirroring
    zeta_mirror = 0     # toroidal angle for mirroring

    # Field helicity parameters
    helicity_M = 1      # poloidal helicity
    helicity_N = 0      # toroidal helicity

    # Create trapped Poincaré map
    poinc = TrappedPoincare(
        field,
        helicity_M,
        helicity_N,
        s_mirror,
        theta_mirror,
        zeta_mirror,
        mass=ALPHA_PARTICLE_MASS,
        charge=ALPHA_PARTICLE_CHARGE,
        Ekin=FUSION_ALPHA_PARTICLE_ENERGY,
        ns_poinc=120,      # number of s initial conditions
        neta_poinc=5,      # number of eta initial conditions
        Nmaps=1000,        # number of return maps
        reltol=1e-8,       # Relative tolerance
        abstol=1e-8,       # Absolute tolerance
        tmax=1e-3,
    )

    # Plot the Poincaré map
    poinc.plot_poincare(filename="trapped_poincare.pdf")

**Key Parameters:**

- **helicity_M, helicity_N**: Define the helicity of field strength contours
- **s_mirror, theta_mirror, zeta_mirror**: Single mirror point to compute magnetic moment
- **ns_poinc, neta_poinc**: Grid resolution for initial conditions
- **Nmaps**: Number of return maps to compute

Passing Poincaré Maps
---------------------

The ``PassingPoincare`` class computes Poincaré maps for passing particles in unperturbed magnetic fields.

**Key Features:**

- Maps trajectories from one :math:`\zeta = 0` plane to the next
- Assumes particles are passing (parallel velocity doesn't change sign)
- Uses pitch-angle variable :math:`\lambda = v_\perp^2/(v^2 B)` as constant of motion

**Usage Example:**

.. code-block:: python

    from firm3d.field.trajectory_helpers import PassingPoincare
    from firm3d.util.constants import (
        ALPHA_PARTICLE_MASS,
        ALPHA_PARTICLE_CHARGE,
        FUSION_ALPHA_PARTICLE_ENERGY,
    )

    # Create passing Poincaré map
    poinc = PassingPoincare(
        field,
        lam=0.0,           # pitch-angle variable
        sign_vpar=1.0,     # sign of parallel velocity (+1 or -1)
        mass=ALPHA_PARTICLE_MASS,
        charge=ALPHA_PARTICLE_CHARGE,
        Ekin=FUSION_ALPHA_PARTICLE_ENERGY,
        ns_poinc=120,      # number of s initial conditions
        ntheta_poinc=1,    # number of theta initial conditions
        Nmaps=1000,        # number of return maps
        reltol=1e-8,       # Relative tolerance
        abstol=1e-8,       # Absolute tolerance
    )

    # Plot the Poincaré map
    poinc.plot_poincare(filename="passing_poincare.pdf")

**Key Parameters:**

- **lam**: Pitch-angle variable :math:`\lambda = v_\perp^2/(v^2 B)`
- **sign_vpar**: Sign of parallel velocity (+1 or -1)
- **ns_poinc, ntheta_poinc**: Grid resolution for initial conditions

Perturbed Passing Poincaré Maps
-------------------------------

The ``PassingPerturbedPoincare`` class computes Poincaré maps for passing particles in magnetic fields with shear Alfvén wave perturbations.

**Key Features:**

- Handles time-dependent perturbations from shear Alfvén waves
- Uses helical angle :math:`\chi = M\theta - N\zeta` as mapping coordinate
- Computes shifted energy :math:`E' = n' E - \omega p_\eta` as constant of motion for quasisymmetric fields with helicity (M,N) with single-harmonic shear Alfvén waves
- Initial conditions sampled for same value of :math:`\mu` (magnetic moment) and :math:`E'=E'-n'\omega p_\eta` (shifted energy) where :math:`n'` is computed from the wave parameters and field helicity.

**Usage Example:**

.. code-block:: python

    from firm3d.field.trajectory_helpers import PassingPerturbedPoincare
    from firm3d.util.constants import (
        ALPHA_PARTICLE_MASS,
        ALPHA_PARTICLE_CHARGE,
        FUSION_ALPHA_PARTICLE_ENERGY,
    )

    # Create shear Alfvén wave perturbation
    Phihat = -1.50119e3
    saw = ShearAlfvenHarmonic(
        Phihat, m=1, n=1, omega=136041, phase=0, B0=field
    )

    # Point for evaluation of Eprime
    p0 = np.array([[0.5, 0.0, 0.0]])  # [s, theta, zeta]

    # Create perturbed passing Poincaré map
    poinc = PassingPerturbedPoincare(
        saw,
        sign_vpar=1.0,     # sign of parallel velocity
        mass=ALPHA_PARTICLE_MASS,
        charge=ALPHA_PARTICLE_CHARGE,
        helicity_M=1,      # field strength helicity
        helicity_N=0,
        Ekin=FUSION_ALPHA_PARTICLE_ENERGY,
        p0=p0,             # point for Eprime evaluation
        lam=0.1,           # pitch-angle variable
        ns_poinc=120,      # number of s initial conditions
        nchi_poinc=1,      # number of chi initial conditions
        Nmaps=1000,        # number of return maps
        reltol=1e-8,       # Relative tolerance
        abstol=1e-8,       # Absolute tolerance
    )

    # Plot the Poincaré map
    poinc.plot_poincare(filename="perturbed_poincare.pdf")

**Key Parameters:**

- **saw**: ShearAlfvenHarmonic instance representing the perturbation
- **helicity_M, helicity_N**: Field strength helicity parameters
- **Ekin, p0, lam**: Used to compute constants of motion (Eprime and mu)
- **ns_poinc, nchi_poinc**: Grid resolution for initial conditions

Data Access and Visualization
-----------------------------

All Poincaré map classes provide methods to access the computed data and create visualizations.

**Data Access:**

.. code-block:: python

    # Get Poincaré map data for passing particles (unperturbed)
    s_all, thetas_all, vpars_all, t_all = poinc.get_poincare_data()

    # Get Poincaré map data for trapped particles
    s_all, chis_all, etas_all, t_all = poinc.get_poincare_data()

    # Get Poincaré map data for passing particles (perturbed)
    s_all, chis_all, etas_all, vpars_all, t_all = poinc.get_poincare_data()

**Return Values:**

- **s_all**: List of lists containing radial coordinate s at each Poincaré return
- **thetas_all**: List of lists containing poloidal angle θ at each return (passing unperturbed)
- **chis_all**: List of lists containing helical angle χ at each return (trapped/perturbed)
- **etas_all**: List of lists containing mapping angle η at each return (trapped/perturbed)
- **vpars_all**: List of lists containing parallel velocity v_|| at each return (passing)
- **t_all**: List of lists containing time at each return

**Visualization:**

.. code-block:: python

    # Create and save plot
    ax = poinc.plot_poincare(filename="poincare_map.pdf")

    # Custom plotting
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.scatter(thetas_all[0], s_all[0], s=0.5, alpha=0.6)
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"$s$")
    ax.set_xlim([0, 2*np.pi])
    ax.set_ylim([0, 1])
    plt.savefig("custom_poincare.pdf")

Performance Considerations
--------------------------

**Parallel Computing:**
All classes support MPI parallelization for large-scale computations:

.. code-block:: python

    from mpi4py import MPI
    comm = MPI.COMM_WORLD

    poinc = PassingPoincare(
        field, lam, sign_vpar, mass, charge, Ekin,
        comm=comm,  # Enable parallel processing
        # ... other parameters
    )

**Solver Options:**
Tune ODE solver parameters for accuracy vs. speed:

.. code-block:: python

    # For adaptive solver (default)
    poinc = PassingPoincare(
        field, lam, sign_vpar, mass, charge, Ekin,
        reltol=1e-8,    # relative tolerance
        abstol=1e-8,    # absolute tolerance
        # ... other parameters
    )

    # For symplectic solver
    poinc = PassingPoincare(
        field, lam, sign_vpar, mass, charge, Ekin,
        solveSympl=True,  # Enable symplectic solver
        dt=1e-8,         # Fixed step size
        roottol=1e-10,   # Root finding tolerance
        # ... other parameters
    )
