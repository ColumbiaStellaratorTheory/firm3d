from warnings import warn

import numpy as np
from scipy import integrate

from .._core.util import parallel_loop_bounds
from ..field.boozermagneticfield import (
    ShearAlfvenHarmonic,
    ShearAlfvenWave,
    ShearAlfvenWavesSuperposition,
)
from ..field.tracing import (
    MaxToroidalFluxStoppingCriterion,
    MinToroidalFluxStoppingCriterion,
    trace_particles_boozer,
    trace_particles_boozer_perturbed,
)

__all__ = [
    "compute_loss_fraction",
    "compute_trajectory_cylindrical",
    "PassingPoincare",
    "PassingPerturbedPoincare",
    "TrappedPoincare",
    "trajectory_to_vtk",
]


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


class PassingPoincare:
    """
    Class to compute and store passing Poincare maps and related quantities
    for a given BoozerMagneticField.
    """

    def __init__(
        self,
        field,
        lam,
        sign_vpar,
        mass,
        charge,
        Ekin,
        ns_poinc=None,
        ntheta_poinc=None,
        s_init=None,
        thetas_init=None,
        Nmaps=500,
        comm=None,
        tmax=1e-2,
        solver_options=None,
    ):
        r"""
        Initialize and compute the passing Poincare map, evaluated by
        integrating the guiding center equations until the trajectory returns
        to the zeta = 0 plane.
        We assume that the particle is passing, so the parallel velocity does
        not change sign.

        Args:
            field : The :class:`BoozerMagneticField` instance.
            lam : Pitch-angle variable :math:`\lambda = v_\perp^2/(v^2 B)`.
            sign_vpar : Sign of the parallel velocity.
            mass : Particle mass.
            charge : Particle charge.
            Ekin : Particle total energy.
            s_init : List of initial s coordinates for the Poincare map.
                     (default: None, ns_poinc is used instead)
            thetas_init : List of initial theta coordinates for the Poincare
                          map. (default: None, ntheta_poinc is used instead)
            ns_poinc : Number of initial conditions in s for Poincare plot
                       (default: 120).
            ntheta_poinc : Number of initial conditions in theta for Poincare
                           plot (default: 2).
            Nmaps : Number of Poincare return maps to compute for each initial
                    condition (default: 500).
            comm : MPI communicator for parallel execution (default: None).
            tmax : Maximum integration time for each segment of the Poincare
                   map (default: 1e-2 s).
            solver_options : Dictionary of options to pass to the ODE solver
                             (default: {}).
        """
        if solver_options is None:
            solver_options = {}
        if sign_vpar not in [-1, 1]:
            raise ValueError("sign_vpar should be either -1 or +1")

        self.field = field
        self.lam = lam
        self.sign_vpar = sign_vpar
        self.mass = mass
        self.charge = charge
        self.Ekin = Ekin
        if s_init is not None and thetas_init is not None:
            self.s_init = s_init
            self.thetas_init = thetas_init
        else:
            if ns_poinc is None:
                ns_poinc = 120
            if ntheta_poinc is None:
                ntheta_poinc = 2
            s = np.linspace(0, 1, ns_poinc + 1, endpoint=False)[1::]
            thetas = np.linspace(0, 2 * np.pi, ntheta_poinc)
            s, thetas = np.meshgrid(s, thetas)
            self.s_init = s.flatten()
            self.thetas_init = thetas.flatten()
        self.Nmaps = Nmaps
        self.comm = comm
        self.tmax = tmax
        self.solver_options = solver_options
        self.vpars_init = self.initialize_passing_map()

        (
            self.s_all,
            self.thetas_all,
            self.vpars_all,
            self.t_all,
        ) = self.compute_passing_map()

    def initialize_passing_map(self):
        r"""
        Given a :class:`BoozerMagneticField` instance, this function generates
        initial positions for the passing Poincare return map. Particles are
        initialized on the zeta = 0 plane such that the parallel velocity is
        consistent with the prescribed total energy and pitch-angle
        variable, :math:`\lambda = v_\perp^2/(v^2 B)`.

        Returns:
            s_init : List of initial s coordinates for the Poincare map.
            thetas_init : List of initial theta coordinates for the Poincare map.
            vpars_init : List of initial parallel velocities for the Poincare map.
        """
        vtotal = np.sqrt(
            2 * self.Ekin / self.mass
        )  # Total velocity from kinetic energy

        def vpar_func(s, theta):
            point = np.zeros((1, 3))
            point[0, 0] = s
            point[0, 1] = theta
            self.field.set_points(point)
            modB = self.field.modB()[0, 0]
            # Skip any trapped particles
            if 1 - self.lam * modB < 0:
                return None
            else:
                return self.sign_vpar * vtotal * np.sqrt(1 - self.lam * modB)

        first, last = parallel_loop_bounds(self.comm, len(self.s_init))
        # For each point, find value of vpar such that lambda = vperp^2/(v^2 B)
        vpars_init = []
        for i in range(first, last):
            vpar = vpar_func(self.s_init[i], self.thetas_init[i])
            if vpar is not None:
                vpars_init.append(vpar)

        if self.comm is not None:
            vpars_init = [i for o in self.comm.allgather(vpars_init) for i in o]

        return vpars_init

    def passing_map(self, point):
        r"""
        Given the coordinates (s,theta,vpar) at zeta = 0, integrates the guiding
        center equations until the trajectory returns to the zeta = 0 plane.
        We assume that the particle is passing, so if vpar crosses through 0
        along the trajectory, a RuntimeError is raised. A RuntimeError is also
        raised if the particle leaves the s = 1 surface. The coordinates (s,theta,vpar)
        for the return map are returned.

        Args:
            point : A numpy array of shape (3,) containing the initial coordinates
                (s,theta,vpar).

        Returns:
            point : A numpy array of shape (3,) containing the coordinates
                (s,theta,vpar) when the trajectory returns to the zeta = 0 plane.
            time : The time taken to return to the zeta = 0 plane.
        """

        points = np.zeros((1, 3))
        points[:, 0] = point[0]
        points[:, 1] = point[1]
        points[:, 2] = 0
        # Set solver options needed for passing map
        res_tys, res_hits = trace_particles_boozer(
            self.field,
            points,
            [point[2]],
            tmax=self.tmax,
            mass=self.mass,
            charge=self.charge,
            Ekin=self.Ekin,
            vpars=[0.0],
            phases=[0.0],
            n_zetas=[1.0],
            m_thetas=[0.0],
            omegas=[0.0],
            stopping_criteria=[
                MaxToroidalFluxStoppingCriterion(1.0),
            ],
            forget_exact_path=False,
            vpars_stop=True,
            phases_stop=True,
            **self.solver_options,
        )
        if len(res_hits[0]) == 0:
            raise RuntimeError("No stopping criterion reached in passing_map.")

        res_hit = res_hits[0][0, :]  # Only check the first hit or stopping criterion

        if res_hit[1] == 0:  # Check that the zetas=[0] plane was hit
            point[0] = res_hit[2]
            point[1] = res_hit[3]
            point[2] = res_hit[5]
            time = res_hit[0]
            return point, time
        else:
            raise RuntimeError("Alternative stopping criterion reached in passing_map.")

    def compute_passing_map(self):
        r"""
        Evaluates the passing Poincare return map for the initialized particle
        positions.
        """
        Ntrj = len(self.s_init)

        s_all = []
        thetas_all = []
        vpars_all = []
        t_all = []
        first, last = parallel_loop_bounds(self.comm, Ntrj)
        for itrj in range(first, last):
            tr = [self.s_init[itrj], self.thetas_init[itrj], self.vpars_init[itrj]]
            s_traj = [tr[0]]
            thetas_traj = [tr[1]]
            vpars_traj = [tr[2]]
            t_traj = [0]
            for _jj in range(self.Nmaps):
                try:
                    tr, time = self.passing_map(tr)
                    s_traj.append(tr[0])
                    thetas_traj.append(tr[1])
                    vpars_traj.append(tr[2])
                    t_traj.append(time)
                except RuntimeError:
                    break
            s_all.append(s_traj)
            thetas_all.append(thetas_traj)
            vpars_all.append(vpars_traj)
            t_all.append(t_traj)

        if self.comm is not None:
            s_all = [i for o in self.comm.allgather(s_all) for i in o]
            thetas_all = [i for o in self.comm.allgather(thetas_all) for i in o]
            vpars_all = [i for o in self.comm.allgather(vpars_all) for i in o]
            t_all = [i for o in self.comm.allgather(t_all) for i in o]

        return s_all, thetas_all, vpars_all, t_all

    def compute_frequencies(self):
        """
        Compute the passing particle poloidal and toroidal transit frequencies
        and mean radial position.
        The frequency is only computed if the trajectory has completed at least
        one full Poincare return map before the trajectory is terminated due to
        a time limit or stopping criterion. Since the frequency will depend on
        field-line label and trapping well in a general 3D field, an average is
        taken over all trajectories initialized on the same flux surface and
        all Poincare return maps along each trajectory.
        Since the frequency profile will flatten in a phase-space island, it is
        sometimes easier to interpret the frequency profile in a field with
        enforced quasisymmetry (i.e., initialize BoozerRadialInterpolant with N
        prescribed).

        Returns:
            omega_theta : List of poloidal transit frequencies.
            omega_zeta : List of toroidal transit frequencies.
            init_s : List of initial s values for each trajectory.
        """
        if "axis" in self.solver_options and self.solver_options["axis"] != 0:
            raise ValueError(
                'ODE solver must integrate with solver_options["axis"]=0 to '
                "compute passing frequencies."
            )

        self.field.set_points(np.array([[1], [0], [0]]).T)
        sign_G = np.sign(self.field.G()[0])

        omega_theta = []
        omega_zeta = []
        init_s = []
        for s_traj, theta_traj, _vpar_traj, t_traj in zip(
            self.s_all, self.thetas_all, self.vpars_all, self.t_all
        ):
            if (
                len(s_traj) < 2
            ):  # Need at least one full Poincare return maps to compute frequency
                continue
            delta_theta = np.array(theta_traj[1::]) - np.array(theta_traj[0:-1])
            delta_t = t_traj[1::]
            delta_zeta = 2 * np.pi * self.sign_vpar * sign_G

            # Average over wells along one field line
            freq_theta = np.mean(delta_theta) / np.mean(delta_t)
            freq_zeta = delta_zeta / np.mean(delta_t)

            omega_theta.append(freq_theta)
            omega_zeta.append(freq_zeta)
            init_s.append(np.mean(s_traj))

        omega_theta = np.array(omega_theta)
        omega_zeta = np.array(omega_zeta)
        init_s = np.round(np.array(init_s), 9) # remove floating-point error in init_s

        s_prof = np.unique(init_s)
        omega_theta_prof = np.zeros((len(s_prof),))
        omega_zeta_prof = np.zeros((len(s_prof),))

        # Average over field-line label
        for i, s in enumerate(s_prof):
            omega_theta_prof[i] = np.mean(omega_theta[np.where(init_s == s)])
            omega_zeta_prof[i] = np.mean(omega_zeta[np.where(init_s == s)])

        return omega_theta_prof, omega_zeta_prof, s_prof

    def get_poincare_data(self):
        """
        Return the Poincare map data.

        Returns:
            s_all, thetas_all, vpars_all, t_all : Lists of trajectory data.
        """
        return self.s_all, self.thetas_all, self.vpars_all, self.t_all

    def plot_poincare(self, ax=None, filename="passing_poincare.pdf"):
        r"""
        Plot the passing Poincare map and save to a file. It is recommended to only
        call this function on MPI rank 0.

        Args:
            ax : Matplotlib axis to plot on. If None, a new figure and axis are
                 created. A figure created here is closed before returning, so
                 that repeated calls do not accumulate figures in pyplot's
                 global registry. A figure supplied by the caller is left open.
            filename : Name of the file to save the plot
                       (default: 'passing_poincare.pdf').
        Returns:
            ax : The Matplotlib axis containing the plot.
        """
        import matplotlib

        matplotlib.use("Agg")  # Don't use interactive backend
        import matplotlib.pyplot as plt

        created_fig = None
        if ax is None:
            created_fig, ax = plt.subplots()
        fig = ax.get_figure()

        ax.set_xlabel(r"$\theta$")
        ax.set_ylabel(r"$s$")
        ax.set_xlim([0, 2 * np.pi])
        ax.set_ylim([0, 1])
        for i in range(len(self.thetas_all)):
            ax.scatter(
                np.mod(self.thetas_all[i], 2 * np.pi),
                self.s_all[i],
                marker="o",
                s=0.5,
                edgecolors="none",
            )
        fig.savefig(filename)

        if created_fig is not None:
            plt.close(created_fig)

        return ax


class TrappedPoincare:
    """
    Class to compute and store trapped Poincare maps and related quantities
    for a given BoozerMagneticField.
    """

    def __init__(
        self,
        field,
        helicity_M,
        helicity_N,
        mass,
        charge,
        Ekin,
        s_mirror=None,
        theta_mirror=None,
        zeta_mirror=None,
        lam=None,
        ns_poinc=None,
        neta_poinc=None,
        s_init=None,
        etas_init=None,
        Nmaps=500,
        comm=None,
        tmax=1e-2,
        solver_options=None,
    ):
        r"""
        Initialize and compute the trapped Poincare map, evaluated by
        integrating the guiding center equations from the v_{\|} = 0 plane
        until the trajectory returns to the v_{\|} = 0 plane.
        The field strength contours are assumed to have helicity (M,N) in
        Boozer coordinates. The mapping coordinate, eta, is chosen based on
        the helicity of the field strength contours: if M = 0 (e.g., QP or
        OP), then eta=theta is used, if N = 0 (e.g., QA/OA, QH/OA), then
        eta = nfp*zeta is used.

        The particle mu is selected by providing a single mirror point,
        (s_mirror, theta_mirror, zeta_mirror), where the parallel velocity is
        zero. The pitch-angle variable, :math:`\lambda = v_\perp^2/(v^2 B)`,
        is computed and held fixed for all trajectories.

        Args:
            field : The :class:`BoozerMagneticField` instance.
            helicity_M : Approximate poloidal helicity of the field strength for
                         classifying ripple and barely-trapped particles.
            helicity_N : Approximate toroidal helicity of the field strength for
                         classifying ripple and barely-trapped particles.
            mass : Particle mass.
            charge : Particle charge.
            Ekin : Particle total energy.
            s_mirror : Initial s coordinate for the mirror point. If None, the
                       pitch-angle variable lam is used to find the mirror point.
            theta_mirror : Initial theta coordinate for the mirror point. If None
                           when lam is provided, the default value of chi = pi/2 is
                           used as an initial guess for the mirror point.
            zeta_mirror : Initial zeta coordinate for the mirror point. If None
                           when lam is provided, the default value of chi = pi/2 is
                           used as an initial guess for the mirror point.
            lam : Pitch-angle variable :math:`\lambda = v_\perp^2/(v^2 B)`.
                  If s_mirror, theta_mirror, zeta_mirror and lam are all provided,
                  then lam is used to find the mirror point, but s_mirror,
                  theta_mirror, zeta_mirror are still used to specify an initial
                  guess for the root solve to find the mirror point.
            ns_poinc : Number of initial conditions in s for Poincare plot
                       (default: 120).
            neta_poinc : Number of initial conditions in eta for Poincare plot
                         (default: 2).
            s_init : List of initial s coordinates for the Poincare map.
                     (default: None, ns_poinc is used instead)
            etas_init : List of initial eta coordinates for the Poincare map.
                        (default: None, neta_poinc is used instead)
            Nmaps : Number of Poincare return maps to compute for each initial
                    condition (default: 500).
            comm : MPI communicator for parallel execution (default: None).
            tmax : Maximum integration time for each segment of the Poincare
                   map (default: 1e-2 s).
            solver_options : Dictionary of options to pass to the ODE solver
                             (default: {}).
        """
        if solver_options is None:
            solver_options = {}
        self.field = field

        self.helicity_M = helicity_M
        self.helicity_N = helicity_N
        # If modB contours close poloidally, then use theta as mapping coordinate
        if self.helicity_M == 0:
            self.helicity_Mp = 1
            self.helicity_Np = 0
        # Otherwise, use zeta as mapping coordinate
        else:
            self.helicity_Mp = 0
            self.helicity_Np = self.field.nfp

        if lam is not None:
            if lam <= 0:
                raise ValueError("lam must be positive.")
            self.lam = lam
            self.modBcrit = 1 / self.lam
            if theta_mirror is None or zeta_mirror is None:
                # Default value for mirror point initial guess
                self.chi_mirror = np.pi / 2
            else:
                self.chi_mirror = self.chi(theta_mirror, zeta_mirror)
        elif (
            s_mirror is not None
            and theta_mirror is not None
            and zeta_mirror is not None
        ):
            field.set_points(np.array([[s_mirror], [theta_mirror], [zeta_mirror]]).T)
            self.modBcrit = field.modB()[0, 0]  # Magnetic field at mirror point
            self.lam = 1 / self.modBcrit  # lambda = v_perp^2/(v^2 B) = 1/modBcrit
            self.chi_mirror = self.chi(theta_mirror, zeta_mirror)
        else:
            raise ValueError(
                "Either lam or s_mirror, theta_mirror, zeta_mirror must be provided."
            )

        self.mass = mass
        self.charge = charge
        self.Ekin = Ekin
        if s_init is not None and etas_init is not None:
            self.s_init = s_init
            self.etas_init = etas_init
        else:
            if ns_poinc is None:
                ns_poinc = 120
            if neta_poinc is None:
                neta_poinc = 2
            self.ns_poinc = ns_poinc
            self.neta_poinc = neta_poinc
        self.Nmaps = Nmaps
        self.comm = comm
        self.tmax = tmax
        self.solver_options = solver_options

        denom = self.helicity_Np * self.helicity_M - self.helicity_N * self.helicity_Mp
        self.dtheta_dchi = self.helicity_Np / denom
        self.dzeta_dchi = self.helicity_Mp / denom

        (
            self.s_all,
            self.chis_all,
            self.etas_all,
            self.t_all,
        ) = self.compute_trapped_map()

    def chi(self, theta, zeta):
        r"""
        Compute the helical angle chi = M*theta - N*zeta.

        Args:
            theta : Poloidal angle.
            zeta : Toroidal angle.
        Returns:
            chi : The helical angle.
        """
        return self.helicity_M * theta - self.helicity_N * zeta

    def eta(self, theta, zeta):
        r"""
        Compute the mapping angle eta = Mp*theta - Np*zeta.

        Args:
            theta : Poloidal angle.
            zeta : Toroidal angle.
        Returns:
            eta : The mapping angle.
        """
        return self.helicity_Mp * theta - self.helicity_Np * zeta

    def chi_eta_to_theta_zeta(self, chi, eta):
        r"""
        Convert helical angles (chi, eta) to (theta, zeta).

        Args:
            chi : Helical angle chi.
            eta : Mapping angle eta.
        Returns:
            theta : Poloidal angle.
            zeta : Toroidal angle.
        """
        denom = self.helicity_Np * self.helicity_M - self.helicity_N * self.helicity_Mp
        theta = (self.helicity_Np * chi - self.helicity_N * eta) / denom
        zeta = (self.helicity_Mp * chi - self.helicity_M * eta) / denom

        return theta, zeta

    def trapped_map(self, point):
        r"""
        Integrates the gc equations from one mirror point to the next mirror point.
        point contains the [s, theta, zeta] coordinates and returns the same coordinates
        after mapping.

        Args:
            point : A numpy array of shape (3,) containing the initial coordinates
                (s,theta,zeta).
        Returns:
            point : A numpy array of shape (3,) containing the coordinates
                (s,theta,zeta) when the trajectory returns to the vpar = 0 plane.
            time : The time taken to return to the vpar = 0 plane.
        """
        theta, zeta = self.chi_eta_to_theta_zeta(point[1], point[2])
        points = np.zeros((1, 3))
        points[:, 0] = point[0]
        points[:, 1] = theta
        points[:, 2] = zeta

        # Set solver options needed for passing map
        res_tys, res_hits = trace_particles_boozer(
            self.field,
            points,
            [0],
            tmax=self.tmax,
            mass=self.mass,
            charge=self.charge,
            Ekin=self.Ekin,
            vpars=[0],
            stopping_criteria=[
                MinToroidalFluxStoppingCriterion(0.01),
                MaxToroidalFluxStoppingCriterion(1.0),
            ],
            forget_exact_path=False,
            vpars_stop=True,
            **self.solver_options,
        )

        if len(res_hits[0]) == 0:
            raise RuntimeError("No stopping criterion reached in trapped_map.")

        res_hit = res_hits[0][0, :]  # Only check the first hit or stopping criterion

        if res_hit[1] == 0:  # Check that the vpars=[0] plane was hit
            point[0] = res_hit[2]
            point[1] = self.chi(res_hit[3], res_hit[4])
            point[2] = self.eta(res_hit[3], res_hit[4])
            time = res_hit[0]
            return point, time
        else:
            raise RuntimeError("Alternative stopping criterion reached in passing_map.")

    def initialize_trapped_map(self):
        r"""
        For the given :class:`BoozerMagneticField` instance, this function
        generates initial positions for the trapped Poincare return map.
        Particles are initialized on the vpar = 0 plane such that the parallel
        velocity is consistent with the prescribed total energy and pitch-angle
        variable, :math:`\lambda = v_\perp^2/(v^2 B)` (i.e., all points are
        mirror points).

        Returns:
            s_init : List of initial s coordinates for the Poincare map.
            thetas_init : List of initial theta coordinates for the Poincare map.
            zetas_init : List of initial zeta coordinates for the Poincare map.
        """

        # We now compute all of the chi mirror points for each s and eta
        def chi_mirror_func(s, eta):
            from scipy.optimize import root_scalar

            point = np.zeros((1, 3))
            point[:, 0] = s

            # Peform root solve to find bounce point
            def diffmodB(chi):
                return modB_func(chi) - self.modBcrit

            def graddiffmodB(chi):
                theta, zeta = self.chi_eta_to_theta_zeta(chi, eta)
                point[:, 1] = theta
                point[:, 2] = zeta
                self.field.set_points(point)
                return (
                    self.field.dmodBdtheta()[0, 0] * self.dtheta_dchi
                    + self.field.dmodBdzeta()[0, 0] * self.dzeta_dchi
                )

            def modB_func(chi):
                theta, zeta = self.chi_eta_to_theta_zeta(chi, eta)
                point[:, 1] = theta
                point[:, 2] = zeta
                self.field.set_points(point)
                return self.field.modB()[0, 0]

            try:
                sol = root_scalar(
                    diffmodB,
                    fprime=graddiffmodB,
                    x0=self.chi_mirror,
                    method="toms748",
                    bracket=[0, np.pi],
                )
            except Exception as err:
                raise RuntimeError(
                    f"Root solve for chi_mirror failed! s = {s}, "
                    f"eta/(2*pi) = {eta / (2 * np.pi)}"
                ) from err
            if not sol.converged:
                raise RuntimeError(
                    f"Root solve for chi_mirror did not converge! s = {s}, "
                    f"eta/(2*pi) = {eta / (2 * np.pi)}"
                )
            return sol.root

        # Create mesh grid if not provided directly
        if not hasattr(self, "s_init") or not hasattr(self, "etas_init"):
            etas = np.linspace(0, 2 * np.pi, self.neta_poinc, endpoint=False)
            s = np.linspace(0, 1.0, self.ns_poinc + 1, endpoint=False)[1::]
            etas2d, s2d = np.meshgrid(etas, s)
            etas2d = etas2d.flatten()
            s2d = s2d.flatten()
        else:
            s2d = self.s_init
            etas2d = self.etas_init

        s_init = []
        chis_init = []
        etas_init = []
        first, last = parallel_loop_bounds(self.comm, len(etas2d))
        # For each point, find the mirror point in chi
        for i in range(first, last):
            try:
                chi = chi_mirror_func(s2d[i], etas2d[i])
                s_init.append(s2d[i])
                chis_init.append(chi)
                etas_init.append(etas2d[i])
            except RuntimeError:
                warn(
                    f"Root solve for chi_mirror failed! s = {s2d[i]}, "
                    f"eta/(2*pi) = {etas2d[i] / (2 * np.pi)}",
                    stacklevel=2,
                )

        if self.comm is not None:
            s_init = [i for o in self.comm.allgather(s_init) for i in o]
            chis_init = [i for o in self.comm.allgather(chis_init) for i in o]
            etas_init = [i for o in self.comm.allgather(etas_init) for i in o]

        return s_init, chis_init, etas_init

    def compute_trapped_map(self):
        r"""
        Evaluates the trapped Poincare return map for the initialized particle
        positions.
        """
        self.s_init, self.chis_init, self.etas_init = self.initialize_trapped_map()
        Ntrj = len(self.s_init)

        s_all = []
        chis_all = []
        etas_all = []
        t_all = []
        first, last = parallel_loop_bounds(self.comm, Ntrj)
        for itrj in range(first, last):
            tr = [self.s_init[itrj], self.chis_init[itrj], self.etas_init[itrj]]
            s_traj = [tr[0]]
            chis_traj = [tr[1]]
            etas_traj = [tr[2]]
            t_traj = [0]
            broken = False
            for _jj in range(self.Nmaps):
                try:
                    # Apply trapped map twice to return to same vpar = 0 plane
                    tr, time1 = self.trapped_map(tr)
                    tr, time2 = self.trapped_map(tr)
                    if np.abs(tr[1] - chis_traj[-1]) > 2 * np.pi:
                        warn(
                            "Barely trapped particle detected in trapped_map.",
                            stacklevel=2,
                        )
                        broken = True
                        break
                    s_traj.append(tr[0])
                    chis_traj.append(tr[1])
                    etas_traj.append(tr[2])
                    t_traj.append(time1 + time2)
                except RuntimeError:
                    broken = True
                    break
            if not broken:
                s_all.append(s_traj)
                chis_all.append(chis_traj)
                etas_all.append(etas_traj)
                t_all.append(t_traj)

        if self.comm is not None:
            s_all = [i for o in self.comm.allgather(s_all) for i in o]
            chis_all = [i for o in self.comm.allgather(chis_all) for i in o]
            etas_all = [i for o in self.comm.allgather(etas_all) for i in o]
            t_all = [i for o in self.comm.allgather(t_all) for i in o]

        return s_all, chis_all, etas_all, t_all

    def plot_poincare(self, ax=None, filename="trapped_poincare.pdf"):
        r"""
        Plot the trapped Poincare map and save to a file. It is recommended to only
        call this function on MPI rank 0.

        Args:
            ax : Matplotlib axis to plot on. If None, a new figure and axis are
                 created. A figure created here is closed before returning, so
                 that repeated calls do not accumulate figures in pyplot's
                 global registry. A figure supplied by the caller is left open.
            filename : Name of the file to save the plot
                       (default: 'trapped_poincare.pdf').
        Returns:
            ax : The Matplotlib axis containing the plot.
        """
        import matplotlib

        matplotlib.use("Agg")  # Don't use interactive backend
        import matplotlib.pyplot as plt

        created_fig = None
        if ax is None:
            created_fig, ax = plt.subplots()
        fig = ax.get_figure()

        ax.set_xlabel(r"$\eta$")
        ax.set_ylabel(r"$s$")
        ax.set_xlim([0, 2 * np.pi])
        ax.set_ylim([0, 1])
        for i in range(len(self.etas_all)):
            ax.scatter(
                np.mod(self.etas_all[i], 2 * np.pi),
                self.s_all[i],
                marker="o",
                s=0.5,
                edgecolors="none",
            )
        fig.savefig(filename)

        if created_fig is not None:
            plt.close(created_fig)

        return ax

    def get_poincare_data(self):
        """
        Return the Poincare map data.

        Returns:
            s_all, chis_all, etas_all, t_all : Lists of trajectory data.
        """
        return self.s_all, self.chis_all, self.etas_all, self.t_all

    def compute_frequencies(self):
        """
        Compute the trapped particle eta and bounce frequencies and mean
        radial position.
        The frequency is only computed if the trajectory has completed at least
        one full Poincare return map before the trajectory is terminated due to
        a time limit or stopping criterion. Since the frequency will depend on
        field-line label and trapping well in a general 3D field, an average is
        taken over all trajectories initialized on the same flux surface and
        all Poincare return maps along each trajectory.
        Since the frequency profile will flatten in a phase-space island, it is
        sometimes easier to interpret the frequency profile in a field with
        enforced quasisymmetry (i.e., initialize BoozerRadialInterpolant with N
        prescribed).

        Returns:
            omega_eta : List of mapping angle eta frequencies.
            omega_b : List of bounce frequencies.
            init_s : List of initial s values for each trajectory.
        """
        if self.solver_options["axis"] != 0:
            raise ValueError(
                'ODE solver must integrate with solver_options["axis"]=0 to '
                "compute trapped frequencies."
            )

        omega_eta = []
        omega_b = []
        init_s = []
        for s_traj, _chi_traj, eta_traj, t_traj in zip(
            self.s_all, self.chis_all, self.etas_all, self.t_all
        ):
            if (
                len(s_traj) < 2
            ):  # Need at least one full Poincare return maps to compute frequency
                continue
            delta_eta = np.array(eta_traj[1::]) - np.array(eta_traj[0:-1])
            delta_t = t_traj[1::]

            # Average over wells along one field line
            omega_eta.append(np.mean(delta_eta) / np.mean(delta_t))
            omega_b.append(2 * np.pi / np.mean(delta_t))  # bounce frequency
            init_s.append(np.mean(s_traj))

        omega_eta = np.array(omega_eta)
        omega_b = np.array(omega_b)
        init_s = np.round(np.array(init_s), 9) # remove floating-point error in init_s

        s_prof = np.unique(init_s)
        omega_eta_prof = np.zeros((len(s_prof),))
        omega_b_prof = np.zeros((len(s_prof),))

        # Average over field-line label
        for i, s in enumerate(s_prof):
            omega_eta_prof[i] = np.mean(omega_eta[np.where(init_s == s)])
            omega_b_prof[i] = np.mean(omega_b[np.where(init_s == s)])

        return omega_eta_prof, omega_b_prof, s_prof


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

    if isinstance(field_or_saw, ShearAlfvenWave):
        field = field_or_saw.B0
        field_or_saw.set_points(points)
        alpha = field_or_saw.alpha()[:, 0]
    else:
        field = field_or_saw
        alpha = 0.0
        if points.shape == 4:
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


def g(t, T):
    """
    Smooth bump weight on (0, T) with g=0 at t<=0 or t>=T.
    """
    t = np.asarray(t, dtype=float)
    T = float(T)

    s = t / T

    # denom = s*(1-s); positive only for s in (0,1)
    denom = s * (1.0 - s)
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
    da_c = -np.log10(ratio)

    return T, da_c


class PassingPerturbedPoincare:
    def __init__(
        self,
        saw,
        sign_vpar,
        mass,
        charge,
        helicity_M,
        helicity_N,
        helicity_Mp=None,
        helicity_Np=None,
        Eprime=None,
        mu=None,
        Ekin=None,
        p0=None,
        lam=None,
        ns_poinc=None,
        nchi_poinc=None,
        DA_poinc=False,
        nconvergence_points=None,
        s_init=None,
        chis_init=None,
        Nmaps=500,
        comm=None,
        tmax=1e-2,
        solver_options=None,
    ):
        """
        Initialize the PassingPerturbedPoincare class, which computes the
        Poincare return map for passing particles in a ShearAlfvenHarmonic
        magnetic field.

        The field strength contours are assumed to have helicity (M,N) in
        Boozer coordinates such that the field strength can be expressed as
        B(s,chi), where chi = M*theta - N*zeta is the helical angle.
        The mapping coordinate, eta, is chosen based on the helicity of the
        field strength contours: if M = 0 (e.g., QP or OP), then eta=theta is
        used, if N = 0 (e.g., QA/OA, QH/OA), then eta = zeta is used.

        The ShearAlfvenHarmonic can then be expressed in terms of the mapping
        coordinates with phase, m'*chi - n'*eta + omega * t.

        The map is evaluated by integrating the guiding center equations from
        the eta - omega/n' * t = 0 plane until the trajectory returns to the
        same plane.

        The map is well-defined (i.e., trajectories don't cross in the
        (s,chi=M*theta - N*zeta) plane) if the unperturbed field is
        quasisymmetric with helicity (M,N). However, the map can still be
        computed for a non-quasisymmetric field, but the trajectories may
        cross in the (s,chi) plane.

        The constants of motion are the magnetic moment, mu = vperp^2/(2 B),
        and the shifted energy, Eprime = n' * E - omega * p_eta, E is the
        total energy, and p_eta is the canonical momentum.

        These constants of motion can be prescribed directly with the Eprime
        and mu parameters. Alternatively, they can be computed from the
        prescribed pitch-angle variable, lam = vperp^2/(v^2 B), total
        unperturbed kinetic energy, Ekin, and a given point p0 in Boozer
        coordinates.

        Args:
            saw : An instance of ShearAlfvenHarmonic or ShearAlfvenWavesSuperposition.
            sign_vpar : Sign of the parallel velocity, either -1 or +1.
            mass : Mass of the particle.
            charge : Charge of the particle.
            helicity_M : Poloidal helicity of the magnetic field.
            helicity_N : Toroidal helicity of the magnetic field.
            helicity_Mp : Poloidal helicity of the phase variable eta.
            Defaults to None. If no value is given, Mp and Np are determined
            by field helicity.
            helicity_Np : Toroidal helicity of the phase variable eta.
            Defaults to None. If no value is given, Mp and Np are determined by field
            helicity.
            Eprime: Shifted energy, Eprime = n' * E - omega * p_eta.
            mu: Magnetic moment, mu = vperp^2/(2 B).
            Ekin: Total unperturbed kinetic energy of the particle, used to
                compute Eprime if not provided.
            p0: Initial point in Boozer coordinates for evaluation of Eprime.
            lam: Pitch angle variable, lambda = vperp^2/(v^2 B), used to
                compute mu if not provided.
            s_init : List of initial s coordinates for the Poincare map.
                (default: None, ns_poinc is used instead)
            chis_init : List of initial chi coordinates for the Poincare map.
                (default: None, nchi_poinc is used instead)
            ns_poinc : Number of initial conditions in s for Poincare plot
                (default: 120).
            nchi_poinc : Number of initial conditions in chi for Poincare plot
                (default: 2).
            Nmaps : Number of Poincare return maps to compute for each initial
                condition (default: 500).
            DA_poinc : Boolean value indicating whether chaos detection is desired
                (default: False)
            nconvergence_points : Integer value indicating the number of Weighted
                Birkhoff Average evaluations to assess convergence.
            comm : MPI communicator for parallel execution (default: None).
            tmax : Maximum integration time for each segment of the Poincare
                map (default: 1e-2 s).
            solver_options : Dictionary of options to pass to the ODE solver
                (default: {}).
        """
        if solver_options is None:
            solver_options = {}
        if not isinstance(saw, ShearAlfvenHarmonic) and not isinstance(
            saw, ShearAlfvenWavesSuperposition
        ):
            raise TypeError(
                "Expected saw to be an instance of ShearAlfvenHarmonic "
                "or ShearAlfvenWavesSuperposition"
            )
        if not isinstance(saw, ShearAlfvenHarmonic):
            dominant_saw = saw[0]
            raise Warning(
                "Expected saw to be an instance of ShearAlfvenHarmonic - "
                "Perturbed Energy Invariant may not be valid."
            )
        else:
            dominant_saw = None
        if sign_vpar not in [-1, 1]:
            raise ValueError("sign_vpar should be either -1 or +1")

        self.saw = saw
        self.B0 = saw.B0
        self.helicity_M = helicity_M
        self.helicity_N = helicity_N
        self._set_helicity_Np_Mp(helicity_Np, helicity_Mp)
        self.mass = mass
        self.charge = charge
        self.sign_vpar = sign_vpar

        self.DA_poinc = DA_poinc
        if DA_poinc:
            if nconvergence_points is None:
                self.nconvergence_points = 1
                self.WBA_transit_steps = [Nmaps - 1]
            else:
                self.nconvergence_points = nconvergence_points
                # set list of transits for each WBA evaluation
                transits_per_average = int(Nmaps / (nconvergence_points))
                self.WBA_transit_steps = np.linspace(
                    transits_per_average, Nmaps - 1, num=nconvergence_points, dtype=int
                ).tolist()
        else:
            self.nconvergence_points = 1
            self.WBA_transit_steps = 0

        if s_init is not None and chis_init is not None:
            self.s_init = s_init
            self.chis_init = chis_init
        else:
            if ns_poinc is None:
                ns_poinc = 120
            if nchi_poinc is None:
                nchi_poinc = 2
            self.ns_poinc = ns_poinc
            self.nchi_poinc = nchi_poinc
        self.Nmaps = Nmaps
        self.comm = comm
        self.tmax = tmax
        self.solver_options = solver_options

        # if using a ShearAlfvenWavesSuperposition, use the test_saw for
        # computing Eprime, Phin, Phim, and omega as the largest mode
        # this does not produce a poincare plot in the strict sense,
        # but you will be unable to visualize surfaces
        if dominant_saw is None:
            self.Phin = saw.Phin
            self.Phim = saw.Phim
            self.omega = saw.omega
        else:
            self.Phin = dominant_saw.Phin
            self.Phim = dominant_saw.Phim
            self.omega = dominant_saw.omega

        self.nprime = (self.Phim * self.helicity_N - self.Phin * self.helicity_M) / (
            self.helicity_Np * self.helicity_M - self.helicity_N * self.helicity_Mp
        )

        if Eprime is not None and mu is not None:
            self.Eprime = Eprime
            self.mu = mu
            self.Ekin = None
        elif Ekin is not None and lam is not None and p0 is not None:
            """
            Compute unperturbed values of mu, p_eta, and Eprime from the given
            parameters.
            """
            v0 = np.sqrt(2 * Ekin / mass)  # Total velocity from kinetic energy
            self.mu = 0.5 * lam * v0**2  # mu = vperp^2/(2 B)
            self.Ekin = Ekin  # Total kinetic energy
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
            Peta0_arr = np.asarray(Peta0)
            if Peta0_arr.size != 1:
                raise ValueError(
                    f"Peta0 must be scalar-like (size 1), got shape "
                    f"{Peta0_arr.shape} and size {Peta0_arr.size}"
                )
            self.Eprime = float(self.nprime * Ekin - self.omega * Peta0_arr.item())
        else:
            raise ValueError(
                "Either Eprime and mu must be provided, or Ekin, lam, and p0 "
                "must be provided."
            )

        # Initialize the passing map
        self.s_init, self.chis_init, self.vpars_init = self.initialize_passing_map()
        # If Ekin is not provided, compute it from the initial parallel velocity
        # this is only used for computing maximum time step in the ODE solver
        if self.Ekin is None:
            self.Ekin = 0.5 * self.mass * self.vpars_init[0] ** 2

        (
            self.s_all,
            self.chis_all,
            self.etas_all,
            self.vpars_all,
            self.t_all,
            self.DA_all,
            self.DA_times,
        ) = self.compute_passing_map()

    def _set_helicity_Np_Mp(self, helicity_Np, helicity_Mp):
        """
        Sets helicity of the phase variable eta based on user inputs.
        """
        if not ((helicity_Np is None) or (helicity_Mp is None)):
            # User specified both helicities
            self.helicity_Mp = helicity_Mp
            self.helicity_Np = helicity_Np
            return

        if (helicity_Np is None) and (helicity_Mp is None):
            # User did not specify helicity, choose default
            if self.helicity_M == 0:
                # modB contours close poloidally,
                # so can use theta as mapping coordinate
                self.helicity_Mp = 1
                self.helicity_Np = 0
            else:
                # use zeta as mapping coordinate
                self.helicity_Mp = 0
                self.helicity_Np = -1
            return

        raise ValueError(
            f"User must either specify both helicity_Np and helicity_Mp or leave both "
            f"of them None. Currently {helicity_Np=} while {helicity_Mp=}."
        )

    def initialize_passing_map(self):
        """
        Compute vpar given (s,chi) such that Eprime = Eprime0
        """

        def vpar_func_perturbed(s, chi):
            # Choose initial conditions on the eta = 0 plane
            theta, zeta = self.chi_eta_to_theta_zeta(chi, 0)
            point = np.zeros((1, 4))  # initialize with t = 0
            point[0, 0] = s
            point[0, 1] = theta
            point[0, 2] = zeta
            self.saw.set_points(point)
            modB = self.B0.modB()[0, 0]
            G = self.B0.G()[0, 0]
            I = self.B0.I()[0, 0]
            psi = self.B0.psi0 * s
            psip = self.B0.psip()[0, 0]
            Phi = self.saw.Phi()[0, 0]
            alpha = self.saw.alpha()[0, 0]
            denom = (
                self.helicity_Np * self.helicity_M - self.helicity_N * self.helicity_Mp
            )  # - 1 in QA
            d_peta_d_vpar = (
                -((self.helicity_M * G + self.helicity_N * I) * (self.mass / modB))
                / denom
            )  # G m/ modB in QA
            d_E_d_vpar2 = 0.5 * self.mass
            a = self.nprime * d_E_d_vpar2  # Coefficient of vpar^2
            b = -self.omega * d_peta_d_vpar  # Coefficient of vpar
            # Constant term
            c = (
                self.nprime * (self.mass * self.mu * modB + self.charge * Phi)
                + self.omega
                * (
                    (self.helicity_M * G + self.helicity_N * I) * self.charge * alpha
                    + self.charge * (self.helicity_N * psi - self.helicity_M * psip)
                )
                / denom
                - self.Eprime
            )
            if (b**2 - 4 * a * c) < 0:
                raise RuntimeError(
                    "No solution for vpar found! Check the parameters and "
                    "initial conditions."
                )
            elif a != 0:
                return (-b + self.sign_vpar * np.sqrt(b**2 - 4 * a * c)) / (2 * a)
            else:
                return (-c / b) * self.sign_vpar

        # Create mesh grid if not provided directly
        if not hasattr(self, "s_init") or not hasattr(self, "chis_init"):
            s = np.linspace(0, 1, self.ns_poinc + 1, endpoint=False)[1::]
            chis = np.linspace(0, 2 * np.pi, self.nchi_poinc)
            s, chis = np.meshgrid(s, chis)
            s = s.flatten()
            chis = chis.flatten()
        else:
            s = self.s_init
            chis = self.chis_init

        first, last = parallel_loop_bounds(self.comm, len(s))
        # For each point, find value of vpar such that lambda = vperp^2/(v^2 B)
        s_init = []
        chis_init = []
        vpars_init = []
        for i in range(first, last):
            try:
                vpar = vpar_func_perturbed(s[i], chis[i])
                s_init.append(s[i])
                chis_init.append(chis[i])
                vpars_init.append(vpar)
            except RuntimeError:
                continue

        if self.comm is not None:
            s_init = [i for o in self.comm.allgather(s_init) for i in o]
            chis_init = [i for o in self.comm.allgather(chis_init) for i in o]
            vpars_init = [i for o in self.comm.allgather(vpars_init) for i in o]

        return s_init, chis_init, vpars_init

    def chi(self, theta, zeta):
        r"""
        Compute the helical angle chi = M*theta - N*zeta.

        Args:
            theta : Poloidal angle.
            zeta : Toroidal angle.
        Returns:
            chi : The helical angle.
        """
        return self.helicity_M * theta - self.helicity_N * zeta

    def eta(self, theta, zeta):
        r"""
        Compute the mapping angle eta = Mp*theta - Np*zeta.

        Args:
            theta : Poloidal angle.
            zeta : Toroidal angle.
        Returns:
            eta : The mapping angle.
        """
        return self.helicity_Mp * theta - self.helicity_Np * zeta

    def chi_eta_to_theta_zeta(self, chi, eta):
        r"""
        Convert helical angles (chi, eta) to (theta, zeta).

        Args:
            chi : Helical angle chi.
            eta : Mapping angle eta.
        Returns:
            theta : Poloidal angle.
            zeta : Toroidal angle.
        """
        denom = self.helicity_Np * self.helicity_M - self.helicity_N * self.helicity_Mp
        theta = (self.helicity_Np * chi - self.helicity_N * eta) / denom
        zeta = (self.helicity_Mp * chi - self.helicity_M * eta) / denom

        return theta, zeta

    def passing_map(self, point, t, eta):
        r"""
        Integrates the GC equations from the provided point on the eta -
        omega/n' * t plane to the next intersection with this plane. An
        assumption is made that the particle is passing, so a RuntimeError
        is raised if the particle mirrors.

        Since the phase of the ShearAlfvenWave depends on time, the initial
        time, t, is passed as an argument.

        Args:
            point : A numpy array of shape (3,) containing the initial
                coordinates (s,chi,eta).
            t : Initial time at which the map is evaluated
        Returns:
            point : A numpy array of shape (3,) containing the coordinates
                (s,chi,eta).
            time : The time at which the trajectory returns to the eta -
                omega/n' * t plane.
            peta : Timeseries of the canonical momentum p_eta along the trajectory.
        """
        phase = self.omega * t
        self.saw.phase = phase
        theta, zeta = self.chi_eta_to_theta_zeta(point[1], eta)
        points = np.zeros((1, 3))
        points[:, 0] = point[0]
        points[:, 1] = theta
        points[:, 2] = zeta

        if self.helicity_M != 0:
            phases = [zeta * self.nprime]
            n_zetas = [self.nprime]
            m_thetas = [0]
            omegas = [self.omega]
        else:
            phases = [theta * self.nprime]
            n_zetas = [0]
            m_thetas = [self.nprime]
            omegas = [self.omega]

        res_tys, res_hits = trace_particles_boozer_perturbed(
            perturbed_field=self.saw,
            stz_inits=points,
            parallel_speeds=[point[2]],
            mus=[self.mu],
            tmax=self.tmax,
            mass=self.mass,
            charge=self.charge,
            Ekin=self.Ekin,
            phases=phases,
            n_zetas=n_zetas,
            m_thetas=m_thetas,
            omegas=omegas,
            vpars=[0],
            axis=0,
            stopping_criteria=[
                MinToroidalFluxStoppingCriterion(0.01),
                MaxToroidalFluxStoppingCriterion(1.0),
            ],
            forget_exact_path=not self.DA_poinc,
            vpars_stop=True,
            phases_stop=True,
            **self.solver_options,
        )
        if len(res_hits[0]) == 0:
            raise RuntimeError("No stopping criterion reached in passing_map.")

        res_hit = res_hits[0][0, :]  # Only check the first hit or stopping criterion

        # Check that the phases plane was hit (index 0 for first phase)
        if res_hit[1] == 0:
            point[0] = res_hit[2]
            point[1] = self.chi(res_hit[3], res_hit[4])
            point[2] = res_hit[5]
        else:
            raise RuntimeError("Alternative stopping criterion reached in passing_map.")

        if not self.DA_poinc:
            return point, res_hit[0] + t, self.eta(res_hit[3], res_hit[4])
        else:
            # define trajectories
            time_momentum = res_tys[0][:, 0]
            s_path = res_tys[0][:, 1]
            theta_path = res_tys[0][:, 2]
            zeta_path = res_tys[0][:, 3]
            vpar_path = res_tys[0][:, 4]

            # set points for trajectories:
            points_traj = np.zeros((len(time_momentum), 4))
            points_traj[:, 0] = s_path
            points_traj[:, 1] = theta_path
            points_traj[:, 2] = zeta_path
            points_traj[:, 3] = time_momentum

            Peta = compute_peta(
                self.saw,
                points_traj,
                vpar_path,
                self.mass,
                self.charge,
                self.helicity_M,
                self.helicity_N,
                helicity_Mp=self.helicity_Mp,
                helicity_Np=self.helicity_Np,
            )
            Peta = np.column_stack((time_momentum, Peta))
            return point, res_hit[0] + t, self.eta(res_hit[3], res_hit[4]), Peta

    def compute_passing_map(self):
        r"""
        Evaluates the passing Poincare return map for the initialized particle
        positions.
        """
        Ntrj = len(self.s_init)

        s_all = []
        chis_all = []
        etas_all = []
        vpars_all = []
        t_all = []
        DA_all = []
        DA_times = []
        first, last = parallel_loop_bounds(self.comm, Ntrj)
        for itrj in range(first, last):
            tr = [self.s_init[itrj], self.chis_init[itrj], self.vpars_init[itrj]]
            s_traj = [tr[0]]
            chis_traj = [tr[1]]
            eta_traj = [0]
            vpars_traj = [tr[2]]
            t_traj = [0]
            particle_DAs = []
            particle_DA_times = []
            for jj in range(self.Nmaps):
                try:
                    if self.DA_poinc:
                        if jj == 0:
                            tr, time, eta, Peta = self.passing_map(
                                tr, t_traj[-1], eta_traj[-1]
                            )
                        else:
                            tr, time, eta, Peta_iter = self.passing_map(
                                tr, t_traj[-1], eta_traj[-1]
                            )
                            Peta_iter[:, 0] += Peta[-1, 0]
                            Peta = np.vstack((Peta, Peta_iter[1:, :]))
                    else:
                        tr, time, eta = self.passing_map(tr, t_traj[-1], eta_traj[-1])
                    s_traj.append(tr[0])
                    chis_traj.append(tr[1])
                    vpars_traj.append(tr[2])
                    t_traj.append(time)
                    eta_traj.append(eta)
                    if self.DA_poinc and jj in self.WBA_transit_steps:
                        time_at_evaluation, DA_at_evaluation = return_DA(Peta)
                        particle_DAs.append(DA_at_evaluation)
                        particle_DA_times.append(jj)
                except RuntimeError:
                    if self.DA_poinc:
                        particle_DAs.append(np.nan)
                        particle_DA_times.append(np.nan)
                    break
            DA_all.append(particle_DAs)
            DA_times.append(particle_DA_times)
            s_all.append(s_traj)
            chis_all.append(chis_traj)
            etas_all.append(eta_traj)
            vpars_all.append(vpars_traj)
            t_all.append(t_traj)

        if self.comm is not None:
            s_all = [i for o in self.comm.allgather(s_all) for i in o]
            chis_all = [i for o in self.comm.allgather(chis_all) for i in o]
            etas_all = [i for o in self.comm.allgather(etas_all) for i in o]
            vpars_all = [i for o in self.comm.allgather(vpars_all) for i in o]
            t_all = [i for o in self.comm.allgather(t_all) for i in o]
            DA_all = [i for o in self.comm.allgather(DA_all) for i in o]
            DA_times = [i for o in self.comm.allgather(DA_times) for i in o]

        return s_all, chis_all, etas_all, vpars_all, t_all, DA_all, DA_times

    def plot_poincare(
        self,
        ax=None,
        filename="passing_poincare.pdf",
        convergence_test_indicies=None,
        DA_max=7,
        ylims=(0, 1),
    ):
        r"""
        Plot the passing Poincare map and save to a file. It is recommended to only
        call this function on MPI rank 0.
        Args:
            ax : Matplotlib axis to plot on. If None, a new figure and axis are
                 created. A figure created here is closed before returning, so
                 that repeated calls do not accumulate figures in pyplot's
                 global registry. A figure supplied by the caller is left open.
            filename : Name of the file to save the plot
                       (default: 'passing_poincare.pdf').
            convergence_test_indicies : Indices of initial conditions to show
            in convergence plot.
            DA_max : Maximum value of Digit Accuracy to show on colorbar
            ylims : Tuple specifying y-axis limits for the Poincare plot.
        Returns:
            ax : The Matplotlib axis containing the plot.
        """

        import matplotlib as mpl
        import matplotlib.pyplot as plt
        from matplotlib.cm import ScalarMappable

        mpl.use("Agg")  # Don't use interactive backend

        try:
            import cmcrameri.cm as cmc  # noqa: F401

            cmap = "cmc.managua"
        except ImportError:
            cmap = "viridis"

        star_ICs = False

        if convergence_test_indicies is None:
            convergence_test_indicies = list(range(len(self.s_all)))
        else:
            star_ICs = True

        if self.DA_poinc and self.nconvergence_points > 1:
            s_itrj_map = {}
            for itrj in convergence_test_indicies:
                s_itrj_map[itrj] = self.s_all[itrj][0]

            min_s = min(list(s_itrj_map.values()))
            max_s = max(list(s_itrj_map.values()))
            s_lst_true = list(s_itrj_map.values())
            cmap_s = mpl.colormaps["copper"].resampled(len(s_lst_true) ** 2)

        created_fig = None
        if ax is None:
            created_fig, ax = plt.subplots()
        fig = ax.get_figure()

        def normalize(numbers):
            if not numbers:
                return []
            min_val, max_val = 0, DA_max
            normalized_numbers = [(x - min_val) / (max_val - min_val) for x in numbers]
            return normalized_numbers

        if self.DA_poinc:
            final_DAs = []
            # retrieve final DA for each trajectory if the particle is not lost
            # put it into a list
            for elem in self.DA_all:
                if len(elem) == self.nconvergence_points:
                    final_DAs.append(elem[self.nconvergence_points - 1])
                else:
                    final_DAs.append(np.nan)
            # normalized DA values for colormap
            DA_norm_all = normalize(final_DAs)
            cmap_object = mpl.colormaps[cmap].resampled(len(self.DA_all) ** 2)

        ax.set_xlabel(r"$\chi$")
        ax.set_ylabel(r"$s$")
        ax.set_xlim([0, 2 * np.pi])
        ax.set_ylim([ylims[0], ylims[1]])

        for i in range(len(self.chis_all)):
            if self.DA_poinc:
                ax.scatter(
                    np.mod(self.chis_all[i], 2 * np.pi),
                    self.s_all[i],
                    marker="o",
                    s=0.75,
                    c=cmap_object(DA_norm_all[i]),
                    edgecolors="none",
                )
            else:
                ax.scatter(
                    np.mod(self.chis_all[i], 2 * np.pi),
                    self.s_all[i],
                    marker="o",
                    s=0.75,
                    edgecolors="none",
                )

        if star_ICs:
            # scatter initial conditions observed in convergence plot onto poincare plot
            for i in s_itrj_map:
                s_norm = (s_itrj_map[i] - min_s) / (max_s - min_s)
                ax.scatter(
                    np.mod(self.chis_all[i][0], 2 * np.pi),
                    self.s_all[i][0],
                    marker="*",
                    s=25,
                    color=cmap_s(s_norm),
                    edgecolors="magenta",
                )

        if self.DA_poinc:
            # make colorbar for DA values
            max_val = DA_max
            norm = plt.Normalize(0, max_val)
            fig.colorbar(
                ScalarMappable(norm=norm, cmap=mpl.colormaps[cmap]),
                ax=ax,
                orientation="vertical",
                label="Digit Accuracy",
            )
        fig.savefig(filename)

        # convergence plot - change in DA with number of transit evaluations
        # histogram of final DA values
        if self.DA_poinc and self.nconvergence_points > 1:
            fig_conv, ax2 = plt.subplots(1, 1)
            ax2.set_ylabel(r"Digit Accuracy")
            ax2.set_xlabel(r"Toroidal Periods")

            for itrj in s_itrj_map:
                ax2.plot(
                    self.DA_times[itrj],
                    self.DA_all[itrj],
                    color=cmap_s((s_itrj_map[itrj] - min_s) / (max_s - min_s)),
                    alpha=0.75,
                    label=f"{s_itrj_map[itrj]}",
                )
            norm = plt.Normalize(min(s_lst_true), max(s_lst_true))

            fig_conv.colorbar(
                ScalarMappable(norm=norm, cmap=cmap_s),
                ax=ax2,
                orientation="vertical",
                label="$s$",
            )

            fig_conv.tight_layout()
            fig_conv.savefig("convergence_" + filename)

            plt.clf()
            plt.hist(final_DAs)
            plt.tight_layout()
            plt.xlabel("Digit Accuracy")
            plt.title("Distribution of Digit Accuracy")
            plt.savefig("DA_histogram_" + filename)
            plt.close(fig_conv)

        if created_fig is not None:
            plt.close(created_fig)

        return ax

    def get_poincare_data(self):
        """
        Return the Poincare map data.

        Returns:
            s_all, chis_all, etas_all, vpars_all, t_all, DA_all, DA_times :
                Lists of trajectory data.
        """
        return (
            self.s_all,
            self.chis_all,
            self.etas_all,
            self.vpars_all,
            self.t_all,
            self.DA_all,
            self.DA_times,
        )


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
