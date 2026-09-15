from os.path import exists

import numpy as np
from scipy import integrate

from ..field.boozermagneticfield import (
    InterpolatedShearAlfvenWave,
    ShearAlfvenHarmonic,
    ShearAlfvenWave,
    ShearAlfvenWavesSuperposition,
)
from ..util.functions import proc0_print


def compute_loss_fraction(res_tys, tmin=1e-7, tmax=1e-2, ntime=1000):
    r"""
    Compute the fraction of particles lost as a function of time.

    Args:
        res_tys : List of particle trajectories, where each trajectory is a 2D
                  array with shape (nsteps, 5) containing time and coordinates
                  (t, s, theta, zeta, vpar).
        tmin : Minimum time to consider for loss fraction (default: 1e-7)
        tmax : Maximum time to consider for loss fraction (default: 1e-2)
        ntime : Number of time points to evaluate the loss fraction (default: 1000)
    Returns:
        times : A numpy array of shape (ntime,) containing the time points at
                which the loss fraction is evaluated.
        loss_frac : A numpy array of shape (ntime,) containing the fraction of
                    particles lost at each time point.
    """
    nparticles = len(res_tys)

    timelost = np.zeros((nparticles,))
    for ip in range(nparticles):
        timelost[ip] = res_tys[ip][-1, 0]

    times = np.logspace(np.log10(tmin), np.log10(tmax), ntime)

    loss_frac = np.zeros_like(times)
    for it in range(ntime):
        loss_frac[it] = np.count_nonzero(timelost < times[it] - 1e-15) / nparticles

    return times, loss_frac


def compute_trajectory_cylindrical(res_ty, field):
    r"""
    Compute the cylindrical coordinates (R, Z, phi) in a given
    BoozerMagneticField for each particle trajectory.

    Args:
        res_ty : A 2D numpy array of shape (nsteps, 5) containing the
                 trajectory of a single particle in Boozer coordinates
                 (s, theta, zeta, vpar). Tracing should be performed with
                 forget_exact_path=False to save the trajectory information.
        field : The :class:`BoozerMagneticField` instance used to set the
                points for the field.

    Returns:
        R_traj : A numpy array with shape (nsteps,) containing the radial
                 coordinate R for the particle trajectory.
        Z_traj : A numpy array with shape (nsteps,) containing the vertical
                 coordinate Z for the particle trajectory.
        phi_traj : A numpy array with shape (nsteps,) containing the
                   azimuthal angle phi for the particle trajectory.
    """
    nsteps = len(res_ty[:, 0])
    points = np.zeros((nsteps, 3))
    points[:, 0] = res_ty[:, 1]
    points[:, 1] = res_ty[:, 2]
    points[:, 2] = res_ty[:, 3]
    field.set_points(points)

    R_traj = field.R()[:, 0]
    Z_traj = field.Z()[:, 0]
    nu = field.nu()[:, 0]
    phi_traj = res_ty[:, 3] - nu

    return R_traj, phi_traj, Z_traj


def min_volumemodB(B0, NFP=None):
    r"""
    Estimate minimum magnetic-field magnitude over sampled surfaces
    by evaluating |B| on a uniform grid of flux surfaces.

    Args:
        B0  : The :class:`BoozerMagneticField` instance to evaluate.
        NFP : Number of field periods. If None, taken from B0.nfp.

    Returns:
        min_modB : Approximate minimum value of |B| in the sampled volume.
    """
    if NFP is None:
        NFP = B0.nfp

    s_grid = np.linspace(0, 1, 100)
    theta_grid = np.linspace(0, 2 * np.pi, 100, endpoint=False)
    zeta_grid = np.linspace(0, 2 * np.pi / NFP, 100, endpoint=False)
    [zeta_grid, theta_grid, s_grid] = np.meshgrid(zeta_grid, theta_grid, s_grid)
    points = np.zeros((len(theta_grid.flatten()), 3))
    points[:, 0] = s_grid.flatten()
    points[:, 1] = theta_grid.flatten()
    points[:, 2] = zeta_grid.flatten()

    B0.set_points(points)
    modB = B0.modB()[:, 0]
    return np.min(modB)


def chi(theta, zeta, helicity_M, helicity_N):
    r"""
    Compute the helical angle chi = M*theta - N*zeta.

    Args:
        theta : Poloidal angle.
        zeta : Toroidal angle.
        helicity_M : Poloidal helicity number.
        helicity_N : Toroidal helicity number.
    Returns:
        chi : The helical angle.
    """
    return helicity_M * theta - helicity_N * zeta


def eta(theta, zeta, helicity_Mp, helicity_Np):
    r"""
    Compute the mapping angle eta = Mp*theta - Np*zeta.

    Args:
        theta : Poloidal angle.
        zeta : Toroidal angle.
        helicity_Mp : Poloidal helicity number.
        helicity_Np : Toroidal helicity number.
    Returns:
        eta : The mapping angle.
    """
    return helicity_Mp * theta - helicity_Np * zeta


def chi_eta_to_theta_zeta(chi, eta, helicity_M, helicity_N, helicity_Mp, helicity_Np):
    r"""
    Convert helical angles (chi, eta) to (theta, zeta).

    Args:
        chi : Helical angle chi.
        eta : Mapping angle eta.
        helicity_M : Poloidal helicity number defining chi = M*theta - N*zeta.
        helicity_N : Toroidal helicity number defining chi = M*theta - N*zeta.
        helicity_Mp : Poloidal helicity number defining
                      eta = Mp*theta - Np*zeta.
        helicity_Np : Toroidal helicity number defining
                      eta = Mp*theta - Np*zeta.
    Returns:
        theta : Poloidal angle.
        zeta : Toroidal angle.
    """
    denom = helicity_Np * helicity_M - helicity_N * helicity_Mp
    theta = (helicity_Np * chi - helicity_N * eta) / denom
    zeta = (helicity_Mp * chi - helicity_M * eta) / denom

    return theta, zeta


def compute_peta(
    field_or_saw,
    points,
    vpar,
    mass,
    charge,
    helicity_M,
    helicity_N,
    helicity_Mp=None,
    helicity_Np=None,
):
    r"""
    Given a ShearAlfvenWave or BoozerMagneticField instance, a point in
    Boozer coordinates, and particle properties, compute the value of the
    canonical momentum, :math:`p_{\eta}`. This quantity is conserved under
    the unperturbed guiding center equations if the field strength is exactly
    quasisymmetric with helicity (M,N) and :math:`\alpha = 0`.

    :math:`p_{\eta} = (M G + N I) \left(\frac{m v_{\|\|}}{B} + q \alpha \right) +
    q (M \psi - M \psi')`

    If field_or_saw is a BoozerMagneticField instance, then alpha = 0.

    Args:
        field_or_saw : The BoozerMagneticField or ShearAlfvenWave instance.
        points : A numpy array of shape (npoints,4) containing the coordinates
                 (s,theta,zeta,t).
            If field_or_saw is a ShearAlfvenWave, then t is the time coordinate.
            If field_or_saw is a BoozerMagneticField, then t is ignored, and
            points is allowed to have shape (npoints,3) for (s,theta,zeta).
        vpar : A numpy array of shape (npoints,) containing the parallel velocity.
        mass : Mass of the particle.
        charge : Charge of the particle.
        helicity_M : Poloidal helicity of the magnetic field.
        helicity_N : Toroidal helicity of the magnetic field.
        helicity_Mp : Poloidal helicity of the mapping coordinate eta.
            If None, then eta is chosen based on the helicity of the field strength.
        helicity_Np : Toroidal helicity of the mapping coordinate eta.
            If None, then eta is chosen based on the helicity of the field strength.

    Returns:
        peta : A numpy array of shape (npoints,) containing the value of the canonical
            momentum :math:`p_{\eta}` at each point.
    """
    if points.shape[1] not in [3, 4]:
        raise ValueError(
            "Points must have shape (npoints, 4) for (s, theta, zeta, t) or "
            "(npoints, 3) for (s, theta, zeta)"
        )
    if isinstance(vpar, float):
        vpar = np.array([vpar])
    if isinstance(vpar, list):
        vpar = np.array(vpar)
    assert vpar.shape[0] == points.shape[0], (
        "vpar must have the same number of points as points"
    )

    if isinstance(
        field_or_saw,
        (ShearAlfvenWave, ShearAlfvenWavesSuperposition, InterpolatedShearAlfvenWave),
    ):
        field = field_or_saw.B0
        field_or_saw.set_points(points)
        alpha = field_or_saw.alpha()[:, 0]
    else:
        field = field_or_saw
        alpha = 0.0
        if points.shape[1] == 4:
            points = points[:, :3]
        field.set_points(points)

    modB = field.modB()[:, 0]
    G = field.G()[:, 0]
    I = field.I()[:, 0]
    psi = field.psi0 * points[:, 0]
    psip = field.psip()[:, 0]

    if helicity_Mp is None and helicity_Np is None:
        # If modB contours close poloidally, then use theta as mapping coordinate
        if helicity_M == 0:
            helicity_Mp = 1
            helicity_Np = 0
        # Otherwise, use zeta as mapping coordinate
        else:
            helicity_Mp = 0
            helicity_Np = -1
    else:
        if (helicity_Mp * helicity_N) == (helicity_Np * helicity_M):
            raise ValueError(
                "Chosen helicities (N, M, N', M') do not create a well "
                "defined Jacobian."
            )
    denom = helicity_Np * helicity_M - helicity_N * helicity_Mp
    peta = (
        -(
            (helicity_M * G + helicity_N * I) * (mass * vpar / modB + charge * alpha)
            + charge * (helicity_N * psi - helicity_M * psip)
        )
        / denom
    )
    return peta


def compute_Eprime(saw, points, vpar, mu, mass, charge, helicity_M, helicity_N):
    r"""
    Compute the invariant Eprime for a ShearAlfvenHarmonic instance given
    points in Boozer coordinates.

    Args:
        saw : An instance of ShearAlfvenHarmonic.
        points : A numpy array of shape (npoints,4) containing the coordinates
                 (s,theta,zeta,t).
        vpar : A numpy array of shape (npoints,) containing the parallel velocity.
        mu : Magnetic moment of the particle, vperp^2/(2 B).
        mass : Mass of the particle.
        charge : Charge of the particle.
        helicity_M : Poloidal helicity of the magnetic field strength.
        helicity_N : Toroidal helicity of the magnetic field strength.

    Returns:
        Eprime : A numpy array of shape (npoints,) containing the shifted energy
            invariant :math:`E' = n' E - \omega p_\eta` at each point.
    """
    if points.shape[1] != 4:
        raise ValueError("Points must have shape (npoints, 4) for (s, theta, zeta, t)")
    if isinstance(vpar, float):
        vpar = np.array([vpar])
    if isinstance(vpar, list):
        vpar = np.array(vpar)
    assert vpar.shape[0] == points.shape[0], (
        "vpar must have the same number of points as points"
    )
    if vpar.shape[0] != points.shape[0]:
        raise ValueError("vpar must have the same number of points as points")
    if isinstance(saw, ShearAlfvenHarmonic) is False:
        raise TypeError("Expected saw to be an instance of ShearAlfvenHarmonic")

    # If modB contours close poloidally, then use theta as mapping coordinate
    if helicity_M == 0:
        helicity_Mp = 1
        helicity_Np = 0
    # Otherwise, use zeta as mapping coordinate
    else:
        helicity_Mp = 0
        helicity_Np = -1

    Phim = saw.Phim
    Phin = saw.Phin
    omega = saw.omega

    # Compute the canonical momentum p_eta
    p_eta = compute_peta(saw, points, vpar, mass, charge, helicity_M, helicity_N)

    # Compute the energy E
    modB = saw.B0.modB()[:, 0]
    E = 0.5 * mass * vpar**2 + mass * mu * modB + charge * saw.Phi()[:, 0]

    # Compute the invariant Eprime
    nprime = (Phim * helicity_N - Phin * helicity_M) / (
        helicity_Np * helicity_M - helicity_N * helicity_Mp
    )
    Eprime = nprime * E - omega * p_eta
    return Eprime


def compute_reference_Eprime(
    saw,
    p0,
    lam,
    sign_vpar,
    mass,
    charge,
    Ekin,
    helicity_M,
    helicity_N,
    helicity_Mp,
    helicity_Np,
    Phim,
    Phin,
    omega,
):
    r"""
    Compute a single reference value of the shifted-energy invariant Eprime
    at a point p0, for a particle with fixed pitch angle lam and total
    kinetic energy Ekin.

    Unlike compute_Eprime, the canonical momentum here is evaluated on the
    unperturbed field saw.B0 rather than the full perturbed saw, so the
    perturbation's alpha contribution is excluded. This matches the
    convention used to pick a single Eprime slice to sample, as opposed to
    evaluating Eprime along an already-perturbed trajectory.

    Args:
        saw : A ShearAlfvenHarmonic instance. Only saw.B0 (the unperturbed
              field) is used to evaluate modB and the canonical momentum.
        p0 : A numpy array of shape (1, 3) containing the reference point
             (s, theta, zeta).
        lam : Pitch angle variable, lambda = vperp^2 / (v^2 B), assumed
              constant along the trajectory.
        sign_vpar : Desired sign of the parallel velocity (+1 or -1).
        mass : Particle mass.
        charge : Particle charge.
        Ekin : Total kinetic energy.
        helicity_M : Poloidal helicity of the field strength.
        helicity_N : Toroidal helicity of the field strength.
        helicity_Mp : Poloidal helicity of the mapping coordinate eta.
        helicity_Np : Toroidal helicity of the mapping coordinate eta.
        Phim : Poloidal mode number of the perturbation.
        Phin : Toroidal mode number of the perturbation.
        omega : Frequency of the perturbation.

    Returns:
        Eprime : The reference value of the shifted-energy invariant at p0.
    """
    v0 = np.sqrt(2 * Ekin / mass)  # Total velocity from kinetic energy
    saw.B0.set_points(p0)
    modB = saw.B0.modB()[0, 0]
    if 1 - lam * modB < 0:
        raise ValueError(
            "Invalid parameter p0: 1 - lambda * modB must be non-negative."
        )
    vpar = sign_vpar * v0 * np.sqrt(1 - lam * modB)  # Parallel velocity
    Peta0 = compute_peta(
        saw.B0,
        p0,
        vpar,
        mass,
        charge,
        helicity_M,
        helicity_N,
    )
    nprime = (Phim * helicity_N - Phin * helicity_M) / (
        helicity_Np * helicity_M - helicity_N * helicity_Mp
    )
    Eprime = nprime * Ekin - omega * Peta0
    return Eprime[0]


def calculate_crossings(h, h_res, radial_position):
    r"""
    Find radial locations where a drift-helicity profile crosses a resonant
    value.

    Args:
        h : Array of drift-helicity values along a profile.
        h_res : Resonant drift-helicity value to find crossings of.
        radial_position : Array of radial-coordinate values corresponding to
                           h.

    Returns:
        crossings : List of radial positions at which h crosses h_res.
    """
    diff = h - h_res
    sign_changes = np.where(np.sign(diff[:-1]) != np.sign(diff[1:]))[0]
    crossings = []
    for i in sign_changes:
        s = radial_position[i]
        crossings.append(s)
    return crossings


def calculate_QS_resonance(Phim, Phin, M, N, omega, drift_omega_zeta, ell):
    r"""
    Compute the drift-helicity value at which a quasisymmetric resonance
    between a perturbation mode (Phim, Phin) and the drift motion occurs.

    Args:
        Phim : Poloidal mode number of the perturbation.
        Phin : Toroidal mode number of the perturbation.
        M : Poloidal helicity of the field strength.
        N : Toroidal helicity of the field strength.
        omega : Frequency of the perturbation.
        drift_omega_zeta : Toroidal drift frequency.
        ell : Resonance harmonic index.

    Returns:
        h_res : Resonant drift-helicity value.
    """
    return (Phin - N * Phim - omega / drift_omega_zeta) / (Phim + ell) + N


def g(t, T):
    """
    Smooth bump weight on (0, T) with g=0 at t<=0 or t>=T.
    """
    t = np.asarray(t, dtype=float)
    T = float(T)

    s = t / T

    # denom = s*(1-s); positive only for s in (0,1)
    w = np.zeros_like(s)

    interior = (s > 0.0) & (s < 1.0)
    denom = s[interior] * (1.0 - s[interior])
    w[interior] = np.exp(-1.0 / denom)

    return w


def return_DA(array):
    """
    Compute the DA metric for a given momentum 2D array.
    Computes the normalized digit difference between the
    WBA at the first half of the trajectory and the final
    given point.
    Args:
        array : A numpy array of shape (npoints, 2) containing the time and
                momentum values.
    Returns:
        T : Final computed time of the trajectory.
        da_c : The computed DA metric value.
    """
    time_m = array[:, 0]
    momentum = array[:, 1]

    if len(time_m) < 4:
        return 0, np.nan

    # full trajectory
    T_idx = len(time_m)
    T_time = time_m
    T_mom = momentum
    T = float(T_time[-1])

    t_idx = int(T_idx / 2)
    t_time = T_time[:t_idx]
    t_mom = T_mom[:t_idx]
    t = float(t_time[-1])

    # weight arrays
    g_t = g(t_time, t)
    g_T = g(T_time, T)

    g_t_int = integrate.trapezoid(g_t, t_time)
    g_T_int = integrate.trapezoid(g_T, T_time)

    if np.isclose(g_t_int, 0.0) and np.isclose(g_T_int, 0.0):
        return T, np.nan

    t_wavg = integrate.trapezoid(g_t * t_mom, t_time) / g_t_int
    T_wavg = integrate.trapezoid(g_T * T_mom, T_time) / g_T_int

    # DA metric
    diff = np.abs(t_wavg - T_wavg)
    denom = 0.5 * (np.abs(t_wavg) + np.abs(T_wavg))

    if diff == 0.0:
        return T, 16

    ratio = diff / denom
    da_absolute = -np.log10(ratio)
    da_relative = -np.log10(diff / (np.nanmax(np.abs(T_mom))))

    return T, max(da_relative, da_absolute)


def _solve_vpar_energy(B0, point, mass, Ekin, mu, sgn):
    r"""
    Solve for the parallel velocity at the given (s, theta, zeta) position(s)
    from energy conservation, Ekin = 0.5 * mass * vpar**2 + mu * modB. Works
    for a single point or an array of points.

    Args:
        B0 : Magnetic field instance used to evaluate modB at the given
             points.
        point : Position(s) at which to evaluate vpar, as (s, theta, zeta)
                coordinates. Array of shape (3,) for a single point, or
                (npoints, 3) for multiple points.
        mass : Particle mass.
        Ekin : Total kinetic energy.
        mu : Magnetic moment (scalar, or array-like matching point).
        sgn : Desired sign of the parallel velocity (+1 or -1).

    Returns:
        vpar : Parallel velocity consistent with Ekin, or NaN where the
               perpendicular energy exceeds the total energy. Scalar if
               point is 1D, otherwise an array of length npoints.
    """
    scalar_input = point.ndim == 1
    points = point[None, :] if scalar_input else point
    B0.set_points(points)
    modB = B0.modB()[:, 0]

    energy_par = Ekin - mu * modB
    vpar = sgn * np.sqrt(np.maximum(2 * energy_par / mass, 0))
    vpar = np.where(energy_par > 0, vpar, np.nan)
    return vpar[0] if scalar_input else vpar


def _solve_vpar_perturbed(
    B0,
    saw,
    point,
    helicity_M,
    helicity_N,
    helicity_Np,
    helicity_Mp,
    mass,
    nprime,
    omega,
    charge,
    Eprime,
    mu,
    sgn,
):
    r"""
    Solve the perturbed-orbit quadratic for vpar at the given (s, theta, zeta)
    position(s) such that the shifted energy invariant equals Eprime. Works
    for a single point or an array of points.

    Args:
        B0 : Unperturbed magnetic field instance used to evaluate modB, G, I,
             psi0, and psip at the given points.
        saw : Perturbed field instance (wrapping B0) used to evaluate Phi and
              alpha at the given points, and to set the evaluation points
              shared with B0.
        point : Position(s) at which to evaluate vpar, as (s, theta, zeta)
                coordinates. Array of shape (3,) for a single point, or
                (npoints, 3) for multiple points.
        helicity_M : Poloidal helicity number defining chi = M*theta - N*zeta.
        helicity_N : Toroidal helicity number defining chi = M*theta - N*zeta.
        helicity_Np : Toroidal helicity number defining
                      eta = Mp*theta - Np*zeta.
        helicity_Mp : Poloidal helicity number defining
                      eta = Mp*theta - Np*zeta.
        mass : Particle mass.
        nprime : Coefficient n' in the shifted-energy invariant
                 Eprime = n' * E - omega * p_eta.
        omega : Coefficient omega in the shifted-energy invariant
                Eprime = n' * E - omega * p_eta.
        charge : Particle charge.
        Eprime : Prescribed value of the shifted-energy invariant.
        mu : Magnetic moment divided by mass (scalar, or array-like matching
             point).
        sgn : Desired sign of the parallel velocity (+1 or -1).

    Returns:
        vpar : Solution(s) for vpar, or NaN where no real root exists.
               Scalar if point is 1D, otherwise an array of length npoints.
    """
    scalar_input = point.ndim == 1
    points = point[None, :] if scalar_input else point
    s = points[:, 0]
    points4 = np.zeros((points.shape[0], 4))  # 4th column initialized as t = 0
    points4[:, :3] = points
    saw.set_points(points4)
    modB = B0.modB()[:, 0]
    G = B0.G()[:, 0]
    I = B0.I()[:, 0]
    psi = B0.psi0 * s
    psip = B0.psip()[:, 0]
    Phi = saw.Phi()[:, 0]
    alpha = saw.alpha()[:, 0]

    denom = helicity_Np * helicity_M - helicity_N * helicity_Mp  # - 1 in QA
    d_peta_d_vpar = (
        -((helicity_M * G + helicity_N * I) * (mass / modB)) / denom
    )  # G m/ modB in QA
    d_E_d_vpar2 = 0.5 * mass
    a = nprime * d_E_d_vpar2  # Coefficient of vpar^2
    b = -omega * d_peta_d_vpar  # Coefficient of vpar
    # Constant term
    c = (
        nprime * (mass * mu * modB + charge * Phi)
        + omega
        * (
            (helicity_M * G + helicity_N * I) * charge * alpha
            + charge * (helicity_N * psi - helicity_M * psip)
        )
        / denom
        - Eprime
    )
    discriminant = b**2 - 4 * a * c
    valid = discriminant >= 0

    if a != 0:
        # mask negatives before sqrt so we don't get warnings
        safe_disc = np.where(valid, discriminant, 0.0)
        result = (-b + sgn * np.sqrt(safe_disc)) / (2 * a)
    else:
        # discriminant = b**2 >= 0 always
        result = (-c / b) * sgn

    vpar = np.where(valid, result, np.nan)
    return vpar[0] if scalar_input else vpar


def _check_filepaths(filepaths):
    r"""
    Check whether all provided output file paths exist.

    Args:
        filepaths : Dictionary of file labels to filesystem paths.
    Returns:
        exists_all : True if every path exists, otherwise False.
    """
    return all(exists(fp) for fp in filepaths.values())


def compute_resonances(res_tys, res_hits, delta=1e-2):
    r"""
    Computes resonant particle orbits given the output of
    :func:`trace_particles_boozer`, ``res_tys`` and
    ``res_hits``, with ``forget_exact_path=False``. Resonance indicates a
    trajectory which returns to the same position
    at the :math:`\zeta = 0` plane after ``mpol`` poloidal turns and
    ``ntor`` toroidal turns.

    Args:
        res_tys: trajectory solution computed from :func:`trace_particles` or
                :func:`trace_particles_boozer` with ``forget_exact_path=False``
        res_hits: output of :func:`trace_particles_boozer` with `zetas = [0]`
        delta: the distance tolerance in the poloidal plane used to compute
                a resonant orbit. (defaults to 1e-2)

    Returns:
        resonances: list of 7d arrays containing resonant particle orbits. The
                elements of each array is
                ``[s0, theta0, zeta0, vpar0, t, mpol, ntor]``.
                Here ``(s0, theta0, zeta0, vpar0)`` indicates the
                initial position and parallel velocity of the particle, ``t``
                indicates the time of the  resonance, ``mpol`` is the number of
                poloidal turns of the orbit, and ``ntor`` is the number of
                toroidal turns.
    """
    nparticles = len(res_tys)
    resonances = []
    # Iterate over particles
    for ip in range(nparticles):
        nhits = len(res_hits[ip])
        s0 = res_tys[ip][0, 1]
        theta0 = res_tys[ip][0, 2]
        zeta0 = res_tys[ip][0, 3]
        theta0_mod = theta0 % (2 * np.pi)
        x0 = s0 * np.cos(theta0)
        y0 = s0 * np.sin(theta0)
        vpar0 = res_tys[ip][0, 4]
        for it in range(1, nhits):
            # Check whether phi hit or stopping criteria achieved
            if int(res_hits[ip][it, 1]) >= 0:
                s = res_hits[ip][it, 2]
                theta = res_hits[ip][it, 3]
                zeta = res_hits[ip][it, 4]
                theta_mod = theta % (2 * np.pi)
                x = s * np.cos(theta)
                y = s * np.sin(theta)
                dist = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)
                t = res_hits[ip][it, 0]
                if dist < delta:
                    proc0_print("Resonance found.")
                    proc0_print(
                        f"theta = {theta_mod}, theta0 = {theta0_mod}, "
                        f"s = {s}, s0 = {s0}"
                    )
                    mpol = np.rint((theta - theta0) / (2 * np.pi))
                    ntor = np.rint((zeta - zeta0) / (2 * np.pi))
                    resonances.append(
                        np.asarray([s0, theta0, zeta0, vpar0, t, mpol, ntor])
                    )
    return resonances


def compute_toroidal_transits(res_tys):
    r"""
    Computes the number of toroidal transits of an orbit.

    Args:
        res_tys: trajectory solution computed from :func:`trace_particles_boozer`
            with ``forget_exact_path=False``.

    Returns:
        ntransits: array with length ``len(res_tys)``. Each element contains the
                number of toroidal transits of the orbit.
    """
    nparticles = len(res_tys)
    ntransits = np.zeros((nparticles,))
    for ip in range(nparticles):
        ntraj = len(res_tys[ip][:, 0])
        if ntraj > 1:
            phi_init = res_tys[ip][0, 3]
            phi = res_tys[ip][-1, 3]
            ntransits[ip] = np.round((phi - phi_init) / (2 * np.pi))
    return ntransits


def compute_poloidal_transits(res_tys, ma=None, flux=True):
    r"""
    Computes the number of poloidal transits of an orbit. For the case of
    particles traced in a :class:`MagneticField` (not a :class:`BoozerMagneticField`),
    the poloidal angle is computed using the arctangent angle in the poloidal plane with
    respect to the coordinate axis, ``ma``,

    .. math::
        \theta = \tan^{-1} \left( \frac{R(\phi)-R_{\mathrm{ma}}(\phi)}
        {Z(\phi)-Z_{\mathrm{ma}}(\phi)} \right),

    where :math:`(R,\phi,Z)` are the cylindrical coordinates of the trajectory
    and :math:`(R_{\mathrm{ma}}(\phi),Z_{\mathrm{ma}(\phi)})` is the position
    of the coordinate axis.

    Args:
        res_tys: trajectory solution computed from :func:`trace_particles` or
                :func:`trace_particles_boozer` with ``forget_exact_path=False``.
        ma: an instance of :class:`Curve` representing the coordinate axis with
                respect to which the poloidal angle is computed. If orbit is
                computed in Boozer coordinates, ``ma`` should be ``None``.
        flux: if ``True``, ``res_tys`` represents the position in flux coordinates
                (should be ``True`` if computed from :func:`trace_particles_boozer`).
                If ``True``, ``ma`` is not used.
    Returns:
        ntransits: array with length ``len(res_tys)``. Each element contains the
                number of poloidal transits of the orbit.
    """
    nparticles = len(res_tys)
    ntransits = np.zeros((nparticles,))
    for ip in range(nparticles):
        ntraj = len(res_tys[ip][:, 0])
        if ntraj > 1:
            theta_init = res_tys[ip][0, 2]
            theta = res_tys[ip][-1, 2]
            ntransits[ip] = np.round((theta - theta_init) / (2 * np.pi))
    return ntransits


def return_chaotic_boolean_array(DA_at_tfinal, cutoff=3):
    r"""
    Return a boolean array classifying particles as chaotic or regular based
    on their final WBA digit accuracy.

    Args:
        DA_at_tfinal : Array-like of final WBA digit accuracy values.
        cutoff : Digit accuracy threshold for classifying chaos (default: 3).

    Returns:
        chaotic_indices : Boolean array of shape (N,) where True indicates
            a chaotic particle.
    """
    return np.array(DA_at_tfinal) < cutoff


def return_chaotic_percentage(DA_at_tfinal, cutoff=3):
    r"""
    Return the percentage of particles classified as chaotic based on their
    final WBA digit accuracy.

    Args:
        DA_at_tfinal : Array-like of final WBA digit accuracy values.
        cutoff : Digit accuracy threshold for classifying chaos (default: 3).

    Returns:
        chaotic_percentage : Percentage of particles classified as chaotic.
    """
    chaotic_indices = return_chaotic_boolean_array(DA_at_tfinal, cutoff)
    return np.mean(chaotic_indices) * 100


def return_chaotic_initial_conditions(DA_at_tfinal, s0, theta0, zeta0, vpar0, mus):
    r"""
    Return the initial conditions of particles classified as chaotic based on
    their final WBA digit accuracy.

    Args:
        DA_at_tfinal : Array-like of final WBA digit accuracy values.
        s0 : Array of initial flux-surface labels.
        theta0 : Array of initial poloidal angles.
        zeta0 : Array of initial toroidal angles.
        vpar0 : Array of initial parallel velocities.
        mus : Array of initial magnetic moments divided by mass.

    Returns:
        chaotic_points : Array of shape (N_chaotic, 3) of (s, theta, zeta)
            initial conditions for chaotic particles.
        chaotic_vpars : Array of shape (N_chaotic,) of initial parallel
            velocities for chaotic particles.
        chaotic_mus : Array of shape (N_chaotic,) of initial magnetic moments
            divided by mass for chaotic particles.
    """
    chaotic_indices = return_chaotic_boolean_array(DA_at_tfinal)

    chaotic_s0s = s0[chaotic_indices]
    chaotic_theta0s = theta0[chaotic_indices]
    chaotic_zeta0s = zeta0[chaotic_indices]
    chaotic_points = np.stack((chaotic_s0s, chaotic_theta0s, chaotic_zeta0s), axis=-1)
    chaotic_vpars = vpar0[chaotic_indices]
    chaotic_mus = mus[chaotic_indices]
    return chaotic_points, chaotic_vpars, chaotic_mus


def trajectory_to_vtk(res_ty, field, filename="trajectory"):
    r"""
    Save a single particle trajectory in Cartesian coordinates to a VTK file.
    Requires the pyevtk package to be installed.

    Args:
        res_ty : A 2D numpy array of shape (nsteps, 5) containing the trajectory of a
                 single particle in Boozer coordinates.
        field : The :class:`BoozerMagneticField` instance used for field evaluation.
        filename : The name of the output VTK file.
    """
    from pyevtk.hl import polyLinesToVTK

    R_traj, phi_traj, Z_traj = compute_trajectory_cylindrical(res_ty, field)

    X_traj = R_traj * np.cos(phi_traj)
    Y_traj = R_traj * np.sin(phi_traj)

    ppl = np.asarray([len(R_traj)])  # Number of points along trajectory
    polyLinesToVTK(filename, X_traj, Y_traj, Z_traj, pointsPerLine=ppl)
