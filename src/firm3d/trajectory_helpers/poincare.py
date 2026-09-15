from warnings import warn

import numpy as np

from .._core.util import parallel_loop_bounds
from ..field.boozermagneticfield import (
    ShearAlfvenHarmonic,
    ShearAlfvenWavesSuperposition,
)
from ..field.tracing import (
    MaxToroidalFluxStoppingCriterion,
    MinToroidalFluxStoppingCriterion,
    trace_particles_boozer,
    trace_particles_boozer_perturbed,
)

from ._utils import (
    calculate_crossings,
    calculate_QS_resonance,
    chi,
    chi_eta_to_theta_zeta,
    compute_peta,
    eta,
    return_DA,
    _solve_vpar_perturbed,
)


class PassingPoincare:
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
        helicity_N=None,
        helicity_M=None,
        helicity_Np=None,
        helicity_Mp=None,
        chaos_detection=False,
        nconvergence_points=None,
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
            sign_vpar : Sign of the parallel velocity (+1 or -1).
            mass : Particle mass.
            charge : Particle charge.
            Ekin : Particle total kinetic energy.
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
            helicity_M : Poloidal helicity of the field-strength contours.
                         Required when computing the canonical momentum
                         :math:`p_{\eta}` or performing chaos detection.
            helicity_N : Toroidal helicity of the field-strength contours.
                         Required when computing the canonical momentum
                         :math:`p_{\eta}` or performing chaos detection.
            helicity_Mp : Poloidal helicity of the mapping coordinate eta.
                          If None, determined automatically from helicity_M.
            helicity_Np : Toroidal helicity of the mapping coordinate eta.
                          If None, determined automatically from helicity_N.
            chaos_detection : If True, compute the Weighted Birkhoff Average
                              (WBA) digit accuracy along each trajectory
                              (default: False).
            nconvergence_points : Number of WBA evaluations per trajectory
                                  used to assess convergence of the chaos
                                  detection metric. If None and
                                  chaos_detection=True, a single evaluation
                                  at the end of the trajectory is used.
        """
        if solver_options is None:
            solver_options = {}
        if sign_vpar not in [-1, 1]:
            raise ValueError("sign_vpar should be either -1 or +1")

        self.helicity_N = helicity_N
        self.helicity_M = helicity_M

        if helicity_N is None or helicity_M is None:
            self.peta_profile = False
        else:
            self.peta_profile = True
            if helicity_Mp is None or helicity_Np is None:
                # If modB contours close poloidally,
                # use theta as mapping coordinate
                if self.helicity_M == 0:
                    helicity_Mp = 1
                    helicity_Np = 0
                # Otherwise, use zeta as mapping coordinate
                else:
                    helicity_Mp = 0
                    helicity_Np = field.nfp
            self.helicity_Mp = helicity_Mp
            self.helicity_Np = helicity_Np

        if (self.helicity_M is None or self.helicity_N is None) and chaos_detection:
            raise ValueError(
                "helicity_M and helicity_N must be provided for chaos detection."
            )

        self.field = field
        self.lam = lam
        self.sign_vpar = sign_vpar
        self.mass = mass
        self.charge = charge
        self.Ekin = Ekin
        if s_init is not None and thetas_init is not None:
            s_flat = s_init
            thetas_flat = thetas_init
        else:
            if ns_poinc is None:
                ns_poinc = 120
            if ntheta_poinc is None:
                ntheta_poinc = 2
            s = np.linspace(0, 1, ns_poinc + 1, endpoint=False)[1::]
            thetas = np.linspace(0, 2 * np.pi, ntheta_poinc)
            s, thetas = np.meshgrid(s, thetas)
            s_flat = s.flatten()
            thetas_flat = thetas.flatten()
        self.Nmaps = Nmaps
        self.comm = comm
        self.tmax = tmax
        self.DA_poinc = chaos_detection
        self.nconvergence_points = nconvergence_points
        self.solver_options = solver_options
        if self.DA_poinc:
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
            self.WBA_transit_steps = [Nmaps - 1]

        self.vpars_init, self.s_init, self.thetas_init = self.initialize_passing_map(
            s_flat, thetas_flat
        )
        (
            self.s_all,
            self.thetas_all,
            self.vpars_all,
            self.t_all,
            self.peta_all,
            self.DA_all,
            self.DA_times,
        ) = self.compute_passing_map()

    def initialize_passing_map(self, s_flat, thetas_flat):
        r"""
        Given a :class:`BoozerMagneticField` instance, this function generates
        initial positions for the passing Poincare return map. Particles are
        initialized on the zeta = 0 plane such that the parallel velocity is
        consistent with the prescribed total energy and pitch-angle
        variable, :math:`\lambda = v_\perp^2/(v^2 B)`.

        Returns:
            vpars_init : List of initial parallel velocities for the Poincare map.
            s_init : List of initial s coordinates for the Poincare map.
            thetas_init : List of initial theta coordinates for the Poincare map.
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

        first, last = parallel_loop_bounds(self.comm, len(s_flat))
        # For each point, find value of vpar such that lambda = vperp^2/(v^2 B)
        vpars_init = []
        s_init = []
        thetas_init = []
        for i in range(first, last):
            vpar = vpar_func(s_flat[i], thetas_flat[i])
            if vpar is not None:
                vpars_init.append(vpar)
                s_init.append(s_flat[i])
                thetas_init.append(thetas_flat[i])

        if self.comm is not None:
            vpars_init = [i for o in self.comm.allgather(vpars_init) for i in o]
            s_init = [i for o in self.comm.allgather(s_init) for i in o]
            thetas_init = [i for o in self.comm.allgather(thetas_init) for i in o]

        return vpars_init, s_init, thetas_init

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
            peta : A numpy array of shape (N, 2) containing trajectory time and
                the canonical momentum p_eta if the map was initialized with
                helicity_M and helicity_N; otherwise an empty list.
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
                MaxToroidalFluxStoppingCriterion(0.99),
            ],
            forget_exact_path=False,
            vpars_stop=True,
            phases_stop=True,
            **self.solver_options,
        )
        if len(res_hits[0]) == 0:
            raise RuntimeError("No stopping criterion reached in passing_map.")

        res_hit = res_hits[0][0, :]  # Only check the first hit or stopping criterion
        time_momentum = res_tys[0][:, 0]

        points_traj = np.zeros((res_tys[0].shape[0], 3))
        points_traj[:, 0] = res_tys[0][:, 1]
        points_traj[:, 1] = res_tys[0][:, 2]
        points_traj[:, 2] = res_tys[0][:, 3]
        vpar_path = res_tys[0][:, 4]
        if self.peta_profile:
            peta = compute_peta(
                self.field,
                points_traj,
                vpar_path,
                self.mass,
                self.charge,
                self.helicity_M,
                self.helicity_N,
                helicity_Mp=self.helicity_Mp,
                helicity_Np=self.helicity_Np,
            )
            peta = np.column_stack((time_momentum, peta))

        if res_hit[1] == 0:  # Check that the zetas=[0] plane was hit
            point[0] = res_hit[2]
            point[1] = res_hit[3]
            point[2] = res_hit[5]
            time = res_hit[0]
            if self.peta_profile:
                return point, time, peta
            else:
                return point, time, []
        else:
            raise RuntimeError("Alternative stopping criterion reached in passing_map.")

    def compute_passing_map(self):
        r"""
        Evaluates the passing Poincare return map for the initialized particle
        positions.

        Returns:
            s_all : List of s coordinate lists, one per trajectory.
            thetas_all : List of theta coordinate lists, one per trajectory.
            vpars_all : List of parallel velocity lists, one per trajectory.
            t_all : List of cumulative transit time lists, one per trajectory.
            peta_all : List of canonical momentum lists (empty if peta_profile
                is False).
            DA_all : List of WBA digit-accuracy lists, one per trajectory.
            DA_times : List of transit indices at which DA was evaluated.
        """
        Ntrj = len(self.s_init)

        s_all = []
        peta_all = []
        thetas_all = []
        vpars_all = []
        DA_all = []
        DA_times = []
        t_all = []
        first, last = parallel_loop_bounds(self.comm, Ntrj)
        for itrj in range(first, last):
            tr = [self.s_init[itrj], self.thetas_init[itrj], self.vpars_init[itrj]]
            s_traj = [tr[0]]
            points_traj = np.zeros((1, 3))
            points_traj[:, 0] = self.s_init[itrj]
            points_traj[:, 1] = self.thetas_init[itrj]
            points_traj[:, 2] = 0

            if self.peta_profile:
                peta = compute_peta(
                    self.field,
                    points_traj,
                    self.vpars_init[itrj],
                    self.mass,
                    self.charge,
                    self.helicity_M,
                    self.helicity_N,
                    helicity_Mp=self.helicity_Mp,
                    helicity_Np=self.helicity_Np,
                )
                peta_traj = [peta[0]]
                Peta = np.array([[0, peta[0]]])
            else:
                peta_traj = []

            thetas_traj = [tr[1]]
            vpars_traj = [tr[2]]
            particle_DAs = []
            particle_DA_times = []
            t_traj = [0]
            for _jj in range(self.Nmaps):
                try:
                    tr, time, Peta_iter = self.passing_map(tr)

                    if self.peta_profile:
                        peta_traj.append(Peta_iter[-1, 1])
                        # shift time column by orbit time
                        Peta_iter[:, 0] += Peta[-1, 0]
                        Peta = np.vstack((Peta, Peta_iter[1:, :]))
                    else:
                        peta_traj.append(np.nan)

                    t_traj.append(time)
                    s_traj.append(tr[0])

                    thetas_traj.append(tr[1])
                    vpars_traj.append(tr[2])
                    if self.DA_poinc and _jj in self.WBA_transit_steps:
                        time_at_evaluation, DA_at_evaluation = return_DA(Peta)
                        particle_DAs.append(DA_at_evaluation)
                        particle_DA_times.append(_jj)
                except RuntimeError:
                    break
            if self.peta_profile:
                peta_all.append(peta_traj)
            s_all.append(s_traj)
            thetas_all.append(thetas_traj)
            vpars_all.append(vpars_traj)
            t_all.append(t_traj)
            DA_all.append(particle_DAs)
            DA_times.append(particle_DA_times)

        if self.comm is not None:
            peta_all = [i for o in self.comm.allgather(peta_all) for i in o]
            s_all = [i for o in self.comm.allgather(s_all) for i in o]
            thetas_all = [i for o in self.comm.allgather(thetas_all) for i in o]
            vpars_all = [i for o in self.comm.allgather(vpars_all) for i in o]
            t_all = [i for o in self.comm.allgather(t_all) for i in o]
            DA_all = [i for o in self.comm.allgather(DA_all) for i in o]
            DA_times = [i for o in self.comm.allgather(DA_times) for i in o]

        return s_all, thetas_all, vpars_all, t_all, peta_all, DA_all, DA_times

    def compute_frequencies(self, s_profile=True):
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

        Args:
            s_profile : If True, return frequencies as a function of field-line
                label s (averaging over trajectories with the same initial s).
                If False, return frequencies as a function of canonical momentum
                p_eta.

        Returns:
            When s_profile is True:
                omega_theta_prof : Array of mean poloidal transit frequencies
                    per surface.
                omega_zeta_prof : Array of mean toroidal transit frequencies
                    per surface.
                s_prof : Array of unique flux-surface labels.
            When s_profile is False:
                omega_theta_prof : Array of mean poloidal transit frequencies per p_eta.
                omega_zeta_prof : Array of mean toroidal transit frequencies per p_eta.
                peta_prof : Array of unique canonical-momentum values.
                s_prof : Array of mean s values corresponding to each p_eta.
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
        init_peta = []

        if not s_profile:
            for s_traj, theta_traj, _vpar_traj, t_traj, peta_traj in zip(
                self.s_all, self.thetas_all, self.vpars_all, self.t_all, self.peta_all
            ):
                if (
                    len(s_traj) < 2
                ):  # Need at least one full Poincare return maps to compute frequency
                    continue
                delta_theta = np.array(theta_traj[1:]) - np.array(theta_traj[0:-1])

                delta_t = t_traj[1::]
                delta_zeta = 2 * np.pi * self.sign_vpar * sign_G

                # Average over wells along one field line
                freq_theta = np.mean(delta_theta) / np.mean(delta_t)
                freq_zeta = delta_zeta / np.mean(delta_t)

                omega_theta.append(freq_theta)
                omega_zeta.append(freq_zeta)
                init_s.append(np.mean(s_traj))
                init_peta.append(np.mean(peta_traj))
        else:
            for s_traj, theta_traj, _vpar_traj, t_traj in zip(
                self.s_all, self.thetas_all, self.vpars_all, self.t_all
            ):
                if (
                    len(s_traj) < 2
                ):  # Need at least one full Poincare return maps to compute frequency
                    continue
                delta_theta = np.array(theta_traj[1:]) - np.array(theta_traj[0:-1])

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
        init_s = np.array(init_s)
        init_peta = np.array(init_peta)

        s_prof = np.unique(init_s)
        peta_prof = np.unique(init_peta)

        if s_profile:
            # Average over field-line label
            omega_theta_prof = np.zeros((len(s_prof),))
            omega_zeta_prof = np.zeros((len(s_prof),))
            for i, s in enumerate(s_prof):
                omega_theta_prof[i] = np.mean(omega_theta[np.where(init_s == s)])
                omega_zeta_prof[i] = np.mean(omega_zeta[np.where(init_s == s)])
            return omega_theta_prof, omega_zeta_prof, s_prof
        else:
            # else, average over p_eta
            omega_theta_prof = np.zeros((len(peta_prof),))
            omega_zeta_prof = np.zeros((len(peta_prof),))
            s_prof = np.zeros((len(peta_prof),))
            for i, s in enumerate(peta_prof):
                omega_theta_prof[i] = np.mean(omega_theta[np.where(init_peta == s)])
                omega_zeta_prof[i] = np.mean(omega_zeta[np.where(init_peta == s)])
                s_prof[i] = np.mean(init_s[np.where(init_peta == s)])

            sort_idx = np.argsort(peta_prof)
            peta_prof = peta_prof[sort_idx]
            omega_theta_prof = omega_theta_prof[sort_idx]
            omega_zeta_prof = omega_zeta_prof[sort_idx]
            s_prof = s_prof[sort_idx]
            return omega_theta_prof, omega_zeta_prof, peta_prof, s_prof

    def get_poincare_data(self):
        """
        Return the Poincare map data.

        Returns:
            s_all, thetas_all, vpars_all, t_all : Lists of trajectory data.
        """
        return self.s_all, self.thetas_all, self.vpars_all, self.t_all

    def plot_poincare(
        self,
        ax=None,
        plot_fluxsurface=True,
        filename="passing_poincare.pdf",
        colorbar=True,
        DA_max=7,
        title="",
    ):
        r"""
        Plot the passing Poincare map and save to a file. It is recommended to only
        call this function on MPI rank 0.

        Args:
            ax : Matplotlib axis to plot on. If None, a new figure and axis are
                 created.
            plot_fluxsurface : If True (default), plot s on the y axis. If False, plot
                          p_eta on the y axis (requires helicity_M and helicity_N
                          to have been provided at construction).
            filename : Name of the file to save the plot
                       (default: 'passing_poincare.pdf').
            colorbar : If True, include a colorbar indicating the digit accuracy of the
                          Weighted Birkhoff Average if chaos_detection=True
                          (default: True). Will not error if WBA is not computed.
            DA_max : Maximum digit accuracy to display on the colorbar if colorbar=True
             (default: 7).
            title : Title for the plot (default: "").
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

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        if not plot_fluxsurface and not self.peta_profile:
            raise ValueError(
                "To plot with p_eta as y axis, the Poincare map "
                "must be initialized with helicity_M and helicity_N "
                "to compute p_eta along the trajectory."
            )

        def normalize(numbers):
            if not numbers:
                return []
            min_val, max_val = 0, DA_max
            normalized_numbers = [(x - min_val) / (max_val - min_val) for x in numbers]
            return normalized_numbers

        y_coordinate = self.s_all if plot_fluxsurface else self.peta_all

        convergence_test_indicies = list(range(len(y_coordinate)))
        if self.DA_poinc and self.nconvergence_points > 1:
            radial_itrj_map = {}
            for itrj in convergence_test_indicies:
                radial_itrj_map[itrj] = y_coordinate[itrj][0]

            min_radial = min(list(radial_itrj_map.values()))
            max_radial = max(list(radial_itrj_map.values()))
            radial_lst_true = list(radial_itrj_map.values())
            color_space = len(radial_lst_true) ** 2
            cmap_radial = mpl.colormaps["copper"].resampled(color_space)

        ax.set_xlabel(r"$\theta$")
        if plot_fluxsurface:
            ax.set_ylabel(r"$s$")
        else:
            ax.set_ylabel(r"$p_\eta$")
        ax.set_xlim([0, 2 * np.pi])
        if plot_fluxsurface:
            ax.set_ylim([0, 1])

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

        if self.DA_poinc:
            for i in range(len(self.thetas_all)):
                ax.scatter(
                    np.mod(self.thetas_all[i], 2 * np.pi),
                    self.s_all[i] if plot_fluxsurface else self.peta_all[i],
                    marker="o",
                    s=0.5,
                    c=cmap_object(DA_norm_all[i]),
                    edgecolors="none",
                )
            if colorbar:
                fig.colorbar(
                    ScalarMappable(
                        norm=plt.Normalize(0, DA_max), cmap=mpl.colormaps[cmap]
                    ),
                    ax=ax,
                    orientation="vertical",
                    label="Digit Accuracy",
                )
        else:
            for i in range(len(self.thetas_all)):
                ax.scatter(
                    np.mod(self.thetas_all[i], 2 * np.pi),
                    self.s_all[i] if plot_fluxsurface else self.peta_all[i],
                    marker="o",
                    s=0.5,
                    edgecolors="none",
                )
        if title != "":
            ax.set_title(title)
        fig.tight_layout()
        plt.savefig(filename, dpi=300)

        if self.DA_poinc and self.nconvergence_points > 1:
            fig_convergence, ax2 = plt.subplots(1, 1)
            ax2.set_ylabel(r"Digit Accuracy")
            ax2.set_xlabel(r"Toroidal Periods")

            for itrj in radial_itrj_map:
                ax2.plot(
                    self.DA_times[itrj],
                    self.DA_all[itrj],
                    color=cmap_radial(
                        (radial_itrj_map[itrj] - min_radial) / (max_radial - min_radial)
                    ),
                    alpha=0.75,
                    label=f"{radial_itrj_map[itrj]}",
                )
            norm = plt.Normalize(min(radial_lst_true), max(radial_lst_true))
            fig_convergence.colorbar(
                ScalarMappable(norm=norm, cmap=cmap_radial),
                ax=ax2,
                orientation="vertical",
                label=r"$s$" if plot_fluxsurface else r"$P_\eta$",
            )

            fig_convergence.tight_layout()
            plt.savefig(filename[:-4] + "_convergence.pdf")
            plt.clf()

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
        helicity_Mp=None,
        helicity_Np=None,
        chaos_detection=False,
        nconvergence_points=None,
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
            helicity_Mp : Poloidal helicity of the mapping coordinate eta.
                          If None, determined automatically from helicity_M.
            helicity_Np : Toroidal helicity of the mapping coordinate eta.
                          If None, determined automatically from helicity_N.
            solver_options : Dictionary of options to pass to the ODE solver
                             (default: {}).
            chaos_detection : If True, compute the Weighted Birkhoff Average
                              (WBA) digit accuracy along each trajectory
                              (default: False).
            nconvergence_points : Number of WBA evaluations per trajectory used
                                  to assess convergence of the chaos detection
                                  metric. If None and chaos_detection=True, a
                                  single evaluation at the end of the trajectory
                                  is used.
        """
        if solver_options is None:
            solver_options = {}
        self.field = field

        self.helicity_M = helicity_M
        self.helicity_N = helicity_N
        if (self.helicity_M is None or self.helicity_N is None) and chaos_detection:
            raise ValueError(
                "helicity_M and helicity_N must be provided for chaos detection."
            )

        if helicity_Mp is None or helicity_Np is None:
            # If modB contours close poloidally, then use theta as mapping coordinate
            if self.helicity_M == 0:
                helicity_Mp = 1
                helicity_Np = 0
            # Otherwise, use zeta as mapping coordinate
            else:
                helicity_Mp = 0
                helicity_Np = self.field.nfp
        self.helicity_Mp = helicity_Mp
        self.helicity_Np = helicity_Np

        if lam is not None:
            if lam <= 0:
                raise ValueError("lam must be positive.")
            self.lam = lam
            self.modBcrit = 1 / self.lam
            if theta_mirror is None or zeta_mirror is None:
                # Default value for mirror point initial guess
                self.chi_mirror = np.pi / 2
            else:
                self.chi_mirror = chi(
                    theta_mirror, zeta_mirror, self.helicity_M, self.helicity_N
                )
        elif (
            s_mirror is not None
            and theta_mirror is not None
            and zeta_mirror is not None
        ):
            field.set_points(np.array([[s_mirror], [theta_mirror], [zeta_mirror]]).T)
            self.modBcrit = field.modB()[0, 0]  # Magnetic field at mirror point
            self.lam = 1 / self.modBcrit  # lambda = v_perp^2/(v^2 B) = 1/modBcrit
            self.chi_mirror = chi(
                theta_mirror, zeta_mirror, self.helicity_M, self.helicity_N
            )
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

        self.DA_poinc = chaos_detection
        if self.DA_poinc:
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
            self.WBA_transit_steps = [Nmaps - 1]

        (
            self.s_all,
            self.chis_all,
            self.etas_all,
            self.t_all,
            self.DA_all,
            self.DA_times,
        ) = self.compute_trapped_map()

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
            peta : A numpy array of shape (N, 2) containing trajectory time and peta.
        """
        theta, zeta = chi_eta_to_theta_zeta(
            point[1],
            point[2],
            self.helicity_M,
            self.helicity_N,
            self.helicity_Mp,
            self.helicity_Np,
        )
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
                MaxToroidalFluxStoppingCriterion(0.99),
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
            point[1] = chi(res_hit[3], res_hit[4], self.helicity_M, self.helicity_N)
            point[2] = eta(res_hit[3], res_hit[4], self.helicity_Mp, self.helicity_Np)
            time = res_hit[0]
        else:
            raise RuntimeError("Alternative stopping criterion reached in passing_map.")

        if not self.DA_poinc:
            return point, time

        # define trajectories
        time_momentum = res_tys[0][:, 0]
        s_path = res_tys[0][:, 1]
        theta_path = res_tys[0][:, 2]
        zeta_path = res_tys[0][:, 3]
        vpar_path = res_tys[0][:, 4]

        # set points for trajectories:
        points_traj = np.zeros((len(time_momentum), 3))
        points_traj[:, 0] = s_path
        points_traj[:, 1] = theta_path
        points_traj[:, 2] = zeta_path

        peta = compute_peta(
            self.field,
            points_traj,
            vpar_path,
            self.mass,
            self.charge,
            self.helicity_M,
            self.helicity_N,
            self.helicity_Mp,
            self.helicity_Np,
        )
        peta = np.column_stack((time_momentum, peta))
        return point, time, peta

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
                theta, zeta = chi_eta_to_theta_zeta(
                    chi,
                    eta,
                    self.helicity_M,
                    self.helicity_N,
                    self.helicity_Mp,
                    self.helicity_Np,
                )
                point[:, 1] = theta
                point[:, 2] = zeta
                self.field.set_points(point)
                return (
                    self.field.dmodBdtheta()[0, 0] * self.dtheta_dchi
                    + self.field.dmodBdzeta()[0, 0] * self.dzeta_dchi
                )

            def modB_func(chi):
                theta, zeta = chi_eta_to_theta_zeta(
                    chi,
                    eta,
                    self.helicity_M,
                    self.helicity_N,
                    self.helicity_Mp,
                    self.helicity_Np,
                )
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

        Returns:
            s_all : List of s coordinate lists, one per trajectory.
            chis_all : List of helical angle chi lists, one per trajectory.
            etas_all : List of mapping angle eta lists, one per trajectory.
            t_all : List of cumulative bounce-period time lists, one per trajectory.
            DA_all : List of WBA digit-accuracy lists, one per trajectory.
            DA_times : List of bounce indices at which DA was evaluated.
        """
        self.s_init, self.chis_init, self.etas_init = self.initialize_trapped_map()
        Ntrj = len(self.s_init)

        s_all = []
        chis_all = []
        etas_all = []
        DA_all = []
        DA_times = []
        t_all = []
        first, last = parallel_loop_bounds(self.comm, Ntrj)
        for itrj in range(first, last):
            tr = [self.s_init[itrj], self.chis_init[itrj], self.etas_init[itrj]]
            s_traj = [tr[0]]
            chis_traj = [tr[1]]
            etas_traj = [tr[2]]
            t_traj = [0]
            broken = False
            particle_DAs = []
            particle_DA_times = []
            for jj in range(self.Nmaps):
                try:
                    if self.DA_poinc:
                        if jj == 0:
                            tr, time1, Peta = self.trapped_map(tr)
                        else:
                            tr, time1, Peta_iter = self.trapped_map(tr)
                            Peta_iter[:, 0] += Peta[-1, 0]
                            Peta = np.vstack((Peta, Peta_iter[1:, :]))

                        tr, time2, Peta_iter = self.trapped_map(tr)
                        Peta_iter[:, 0] += Peta[-1, 0]
                        Peta = np.vstack((Peta, Peta_iter[1:, :]))
                    else:
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
                    if self.DA_poinc and jj in self.WBA_transit_steps:
                        time_at_evaluation, DA_at_evaluation = return_DA(Peta)
                        particle_DAs.append(DA_at_evaluation)
                        particle_DA_times.append(jj)
                except RuntimeError:
                    broken = True
                    # @NOTE: not returning DA
                    break
            if not broken:
                s_all.append(s_traj)
                chis_all.append(chis_traj)
                etas_all.append(etas_traj)
                t_all.append(t_traj)
                DA_all.append(particle_DAs)
                DA_times.append(particle_DA_times)

        if self.comm is not None:
            s_all = [i for o in self.comm.allgather(s_all) for i in o]
            chis_all = [i for o in self.comm.allgather(chis_all) for i in o]
            etas_all = [i for o in self.comm.allgather(etas_all) for i in o]
            t_all = [i for o in self.comm.allgather(t_all) for i in o]
            DA_all = [i for o in self.comm.allgather(DA_all) for i in o]
            DA_times = [i for o in self.comm.allgather(DA_times) for i in o]

        return s_all, chis_all, etas_all, t_all, DA_all, DA_times

    def plot_poincare(
        self,
        ax=None,
        filename="trapped_poincare.pdf",
        convergence_test_indicies=None,
        DA_max=None,
    ):
        r"""
        Plot the trapped Poincare map and save to a file. It is recommended to only
        call this function on MPI rank 0.

        Args:
            ax : Matplotlib axis to plot on. If None, a new figure and axis are
                 created.
            filename : Name of the file to save the plot
                       (default: 'trapped_poincare.pdf').
            convergence_test_indicies : List of trajectory indices to include in
                the plot. If None, all trajectories are plotted.
            DA_max : Maximum digit accuracy to display on the colorbar. If None
                     and chaos_detection=True, defaults to 7.
        Returns:
            ax : The Matplotlib axis containing the plot.
        """
        import matplotlib as mpl

        mpl.use("Agg")  # Don't use interactive backend
        import matplotlib.pyplot as plt
        from matplotlib.cm import ScalarMappable

        try:
            import cmcrameri.cm as cmc  # noqa: F401

            cmap = "cmc.managua"
        except ImportError:
            cmap = "viridis"

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        if convergence_test_indicies is None:
            convergence_test_indicies = list(range(len(self.s_all)))

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
            if DA_max is None:
                DA_max = 7  # np.nanmax(final_DAs)

        def normalize(numbers):
            if not numbers:
                return []
            min_val, max_val = 0, DA_max
            normalized_numbers = [(x - min_val) / (max_val - min_val) for x in numbers]
            return normalized_numbers

        if self.DA_poinc:
            DA_norm_all = normalize(final_DAs)
            cmap_object = mpl.colormaps[cmap].resampled(len(self.DA_all) ** 2)

        ax.set_xlabel(r"$\eta$")
        ax.set_ylabel(r"$s$")
        ax.set_xlim([0, 2 * np.pi])
        ax.set_ylim([0, 1])
        for i in range(len(self.etas_all)):
            if self.DA_poinc:
                ax.scatter(
                    np.mod(self.etas_all[i], 2 * np.pi),
                    self.s_all[i],
                    marker="o",
                    s=0.5,
                    c=cmap_object(DA_norm_all[i]),
                    edgecolors="none",
                )
            else:
                ax.scatter(
                    np.mod(self.etas_all[i], 2 * np.pi),
                    self.s_all[i],
                    marker="o",
                    s=0.5,
                    edgecolors="none",
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
        plt.savefig(filename)

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
        init_s = np.array(init_s)

        s_prof = np.unique(init_s)
        omega_eta_prof = np.zeros((len(s_prof),))
        omega_b_prof = np.zeros((len(s_prof),))

        # Average over field-line label
        for i, s in enumerate(s_prof):
            omega_eta_prof[i] = np.mean(omega_eta[np.where(init_s == s)])
            omega_b_prof[i] = np.mean(omega_b[np.where(init_s == s)])

        return omega_eta_prof, omega_b_prof, s_prof


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
        chaos_detection=False,
        nconvergence_points=None,
        s_init=None,
        chis_init=None,
        Nmaps=500,
        comm=None,
        tmax=1e-2,
        dt_save=1e-6,
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
            chaos_detection : If True, compute the Weighted Birkhoff Average
                (WBA) digit accuracy along each trajectory (default: False).
            nconvergence_points : Integer value indicating the number of Weighted
                Birkhoff Average evaluations to assess convergence.
            comm : MPI communicator for parallel execution (default: None).
            tmax : Maximum integration time for each segment of the Poincare
                map (default: 1e-2 s).
            dt_save : Save interval for the ODE solver along each trajectory
                (default: 1e-6 s).
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
            warn(
                "Expected saw to be an instance of ShearAlfvenHarmonic - "
                "Perturbed Energy Invariant may not be valid.",
                stacklevel=2,
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

        self.chaos_detection = chaos_detection
        if chaos_detection:
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
            self.WBA_transit_steps = [0]

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
        self.dt_save = dt_save
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
            self.v0 = np.sqrt(2 * Ekin / mass)  # Total velocity from kinetic energy
            self.mu = 0.5 * lam * self.v0**2  # mu = vperp^2/(2 B)
            self.Ekin = Ekin  # Total kinetic energy
            saw.B0.set_points(p0)
            modB = saw.B0.modB()[0, 0]
            if 1 - lam * modB < 0:
                raise ValueError(
                    "Invalid parameter p0: 1 - lambda * modB must be non-negative."
                )
            vpar = sign_vpar * self.v0 * np.sqrt(1 - lam * modB)  # Parallel velocity
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
        Generate initial conditions (s, chi, vpar) on the eta = 0 plane such
        that the shifted energy invariant equals self.Eprime.

        Returns:
            s_init : List of initial s coordinates.
            chis_init : List of initial chi = M*theta - N*zeta coordinates.
            vpars_init : List of initial parallel velocities.
        """

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
            theta, zeta = chi_eta_to_theta_zeta(
                chis[i],
                0,
                self.helicity_M,
                self.helicity_N,
                self.helicity_Mp,
                self.helicity_Np,
            )
            point = np.array([s[i], theta, zeta])
            vpar = _solve_vpar_perturbed(
                self.B0,
                self.saw,
                point,
                self.helicity_M,
                self.helicity_N,
                self.helicity_Np,
                self.helicity_Mp,
                self.mass,
                self.nprime,
                self.omega,
                self.charge,
                self.Eprime,
                self.mu,
                self.sign_vpar,
            )
            if np.isnan(vpar):
                continue
            s_init.append(s[i])
            chis_init.append(chis[i])
            vpars_init.append(float(vpar))

        if self.comm is not None:
            s_init = [i for o in self.comm.allgather(s_init) for i in o]
            chis_init = [i for o in self.comm.allgather(chis_init) for i in o]
            vpars_init = [i for o in self.comm.allgather(vpars_init) for i in o]

        return s_init, chis_init, vpars_init

    def passing_map(self, point, t, eta0):
        r"""
        Integrates the GC equations from the provided point on the eta -
        omega/n' * t plane to the next intersection with this plane. An
        assumption is made that the particle is passing, so a RuntimeError
        is raised if the particle mirrors.

        Since the phase of the ShearAlfvenWave depends on time, the initial
        time, t, is passed as an argument.

        Args:
            point : A numpy array of shape (3,) containing the initial
                coordinates (s,chi,vpar).
            t : Initial time at which the map is evaluated
            eta0 : Initial eta coordinate at which the map is evaluated
        Returns:
            point : A numpy array of shape (3,) containing the coordinates
                (s,chi,vpar).
            time : The time at which the trajectory returns to the eta -
                omega/n' * t plane.
            eta : The eta coordinate when the trajectory returns to the plane.
            Peta : Only returned when self.chaos_detection is True. A numpy array
                of shape (N, 2) containing trajectory time and the canonical
                momentum p_eta along the trajectory.
        """
        phase = self.omega * t
        self.saw.phase = phase
        theta, zeta = chi_eta_to_theta_zeta(
            point[1],
            eta0,
            self.helicity_M,
            self.helicity_N,
            self.helicity_Mp,
            self.helicity_Np,
        )
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
            dt_save=self.dt_save,
            phases=phases,
            n_zetas=n_zetas,
            m_thetas=m_thetas,
            omegas=omegas,
            vpars=[0],
            axis=0,
            stopping_criteria=[
                MinToroidalFluxStoppingCriterion(0.01),
                MaxToroidalFluxStoppingCriterion(0.99),
            ],
            forget_exact_path=not self.chaos_detection,
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
            point[1] = chi(res_hit[3], res_hit[4], self.helicity_M, self.helicity_N)
            point[2] = res_hit[5]
        else:
            raise RuntimeError("Alternative stopping criterion reached in passing_map.")

        if not self.chaos_detection:
            return (
                point,
                res_hit[0] + t,
                eta(res_hit[3], res_hit[4], self.helicity_Mp, self.helicity_Np),
            )
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
            return (
                point,
                res_hit[0] + t,
                eta(res_hit[3], res_hit[4], self.helicity_Mp, self.helicity_Np),
                Peta,
            )

    def compute_passing_map(self):
        r"""
        Evaluates the passing Poincare return map for the initialized particle
        positions.

        Returns:
            s_all : List of s coordinate lists, one per trajectory.
            chis_all : List of chi coordinate lists, one per trajectory.
            etas_all : List of eta coordinate lists, one per trajectory.
            vpars_all : List of parallel velocity lists, one per trajectory.
            t_all : List of cumulative transit time lists, one per trajectory.
            DA_all : List of WBA digit-accuracy lists, one per trajectory.
            DA_times : List of transit indices at which DA was evaluated.
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
                    if self.chaos_detection:
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
                    if self.chaos_detection and jj in self.WBA_transit_steps:
                        time_at_evaluation, DA_at_evaluation = return_DA(Peta)
                        particle_DAs.append(DA_at_evaluation)
                        particle_DA_times.append(jj)
                except RuntimeError:
                    if self.chaos_detection:
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

    def convergence_plot(
        self,
        ax=None,
        convergence_test_indicies=None,
        DA_max=7,
        filename="DA_convergence.pdf",
    ):
        r"""
        Plot the convergence of the Weighted Birkhoff Average for the trajectories
        specified by `convergence_test_indicies` and save to a file. It is recommended
        to only call this function on MPI rank 0.
        Args:
            convergence_test_indicies : Indices of initial conditions to show in DA
                convergence plot.
            DA_max : Maximum value of Digit Accuracy to show on colorbar
            filename : Name of the file to save the plot (default: 'DA_convergence.pdf
        Returns:
            fig, ax : The Matplotlib figure and axis containing the plot.
        """
        if not self.chaos_detection or self.nconvergence_points <= 1:
            raise ValueError(
                "Convergence plot is only meaningful if chaos_detection is True and "
                "nconvergence_points is greater than 1."
            )
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        from matplotlib.cm import ScalarMappable

        mpl.use("Agg")  # Don't use interactive backend

        if convergence_test_indicies is None:
            convergence_test_indicies = list(range(len(self.s_all)))

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        if self.chaos_detection and self.nconvergence_points > 1:
            s_itrj_map = {}
            for itrj in convergence_test_indicies:
                s_itrj_map[itrj] = self.s_all[itrj][0]

            min_s = min(list(s_itrj_map.values()))
            max_s = max(list(s_itrj_map.values()))
            s_lst_true = list(s_itrj_map.values())
            cmap_s = mpl.colormaps["copper"].resampled(len(s_lst_true) ** 2)

        ax.set_ylabel(r"Digit Accuracy")
        ax.set_xlabel(r"Toroidal Periods")

        for itrj in s_itrj_map:
            ax.plot(
                self.DA_times[itrj],
                self.DA_all[itrj],
                color=cmap_s((s_itrj_map[itrj] - min_s) / (max_s - min_s)),
                alpha=0.75,
                label=f"{s_itrj_map[itrj]}",
            )
        norm = plt.Normalize(min(s_lst_true), max(s_lst_true))
        fig.colorbar(
            ScalarMappable(norm=norm, cmap=cmap_s),
            ax=ax,
            orientation="vertical",
            label="$s$",
        )

        fig.tight_layout()
        plt.savefig(filename[:-4] + "_convergence.pdf")

        return ax

    def plot_poincare(
        self,
        ax=None,
        filename="passing_poincare.pdf",
        convergence_test_indicies=None,
        DA_max=7,
        resonance_lines=None,
        line_plotting_kwargs=None,
        ylims=(0, 1),
        DA_colorbar=True,
        plot_legend=True,
        bg_field=None,
        s_axis_label=True,
    ):
        r"""
        Plot the passing Poincare map and save to a file. It is recommended to only
        call this function on MPI rank 0.
        Args:
            ax : Matplotlib axis to plot on. If None, a new figure and axis are
                 created.
            filename : Name of the file to save the plot
                (default: 'passing_poincare.pdf').
            convergence_test_indicies : Indices of initial conditions to show
                in DA convergence plot.
            DA_max : Maximum value of Digit Accuracy to show on colorbar
            resonance_lines : List of resonance lines to plot on Poincare map.
                Each element should be a s value.
            line_plotting_kwargs : List of dictionaries of keyword arguments for
                plotting resonance lines. Should be the same length as
                `resonance_lines`.
            ylims : Tuple specifying y-axis limits for the Poincare plot.
            DA_colorbar : Boolean indicating for DA colorbar inclusion.
            plot_legend : Boolean indicating whether to include a legend for the
                resonance lines.
            bg_field : Magnetic field to use for plotting resonance lines. Should
                be the background field of the pertubation with perfect QS enforced.
                If None, the unperturbed field, B0, is used.
            s_axis_label : Boolean for s-axis label. True by default.
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

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        if bg_field is None:
            bg_field = self.B0

        if line_plotting_kwargs is None and resonance_lines is not None:
            line_plotting_kwargs = [
                {} for _ in resonance_lines if resonance_lines is not None
            ]

        if self.chaos_detection and self.nconvergence_points > 1:
            s_itrj_map = {}
            for itrj in convergence_test_indicies:
                s_itrj_map[itrj] = self.s_all[itrj][0]

            min_s = min(list(s_itrj_map.values()))
            max_s = max(list(s_itrj_map.values()))
            s_lst_true = list(s_itrj_map.values())
            cmap_s = mpl.colormaps["copper"].resampled(len(s_lst_true) ** 2)

        def normalize(numbers):
            if not numbers:
                return []
            min_val, max_val = 0, DA_max
            normalized_numbers = [(x - min_val) / (max_val - min_val) for x in numbers]
            return normalized_numbers

        if self.chaos_detection:
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
        if s_axis_label:
            ax.set_ylabel(r"$s$")
        ax.set_xlim([0, 2 * np.pi])
        ax.set_ylim([ylims[0], ylims[1]])

        for i in range(len(self.chis_all)):
            # if particle completeled less than 10%
            # than their perscribed transits, skip
            if len(self.chis_all[i]) < int(self.Nmaps * 0.1):
                continue
            if self.chaos_detection:
                ax.scatter(
                    np.mod(self.chis_all[i], 2 * np.pi),
                    self.s_all[i],
                    marker="o",
                    s=2,
                    c=cmap_object(DA_norm_all[i]),
                    edgecolors="none",
                )
            else:
                ax.scatter(
                    np.mod(self.chis_all[i], 2 * np.pi),
                    self.s_all[i],
                    marker="o",
                    s=2,
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

        if self.chaos_detection:
            # make colorbar for DA values
            max_val = DA_max
            norm = plt.Normalize(0, max_val)
            if DA_colorbar:
                fig.colorbar(
                    ScalarMappable(norm=norm, cmap=mpl.colormaps[cmap]),
                    ax=ax,
                    orientation="vertical",
                    label="Digit Accuracy",
                )

        lines_2 = []
        if resonance_lines is not None:
            cmap = plt.get_cmap("Wistia")
            n_lines = len(resonance_lines)
            for i, arr in enumerate(resonance_lines):
                if "color" not in line_plotting_kwargs[i]:
                    line_plotting_kwargs[i]["color"] = cmap(i / max(n_lines - 1, 1))
                lines_2.append((arr, line_plotting_kwargs[i]["color"]))
                theta = np.pi / 2

                chi_val = chi(theta, 0, self.helicity_M, self.helicity_N)
                theta_vp, zeta_vp = chi_eta_to_theta_zeta(
                    chi_val,
                    0,
                    self.helicity_M,
                    self.helicity_N,
                    self.helicity_Mp,
                    self.helicity_Np,
                )
                point = np.array([arr, theta_vp, zeta_vp])
                vp = _solve_vpar_perturbed(
                    self.B0,
                    self.saw,
                    point,
                    self.helicity_M,
                    self.helicity_N,
                    self.helicity_Np,
                    self.helicity_Mp,
                    self.mass,
                    self.nprime,
                    self.omega,
                    self.charge,
                    self.Eprime,
                    self.mu,
                    self.sign_vpar,
                )
                if np.isnan(vp):
                    raise RuntimeError(
                        "No solution for vpar found! Check the parameters and "
                        "initial conditions."
                    )
                vp = float(vp)
                point = np.zeros((1, 3))
                point[:, 0] = arr
                point[:, 1] = theta
                point[:, 2] = 0
                bg_field.set_points(point)
                lam = (self.v0**2 - vp**2) / (self.v0**2 * bg_field.modB()[0, 0])
                unperturbed_path_map = PassingPoincare(
                    field=bg_field,
                    lam=lam,
                    sign_vpar=self.sign_vpar,
                    mass=self.mass,
                    charge=self.charge,
                    Ekin=self.Ekin,
                    s_init=[arr],
                    comm=None,
                    Nmaps=100,
                    helicity_N=self.helicity_N,
                    helicity_M=self.helicity_M,
                    helicity_Mp=self.helicity_Mp,
                    helicity_Np=self.helicity_Np,
                    thetas_init=[theta],
                    solver_options={"axis": 0},
                )

                s_upt, theta_upt, vpar_upt, t_upt = (
                    unperturbed_path_map.get_poincare_data()
                )
                chis = chi(
                    np.array(theta_upt[0]),
                    np.array([2 * np.pi * i for i in range(len(theta_upt[0]))]),
                    self.helicity_M,
                    self.helicity_N,
                )
                s_upt = np.array(s_upt[0])
                pa_data = np.column_stack((chis, s_upt))
                pa_data[:, 0] = np.mod(pa_data[:, 0], (2 * np.pi))
                pa_data = pa_data[pa_data[:, 0].argsort()]
                ax.plot(
                    pa_data[:, 0],
                    pa_data[:, 1],
                    **line_plotting_kwargs[i],
                )
            if plot_legend:
                ax.legend()
        fig.tight_layout()
        fig.savefig(filename[:-4] + ".pdf")

        # convergence plot - change in DA with number of transit evaluations
        # histogram of final DA values
        if star_ICs:
            self.convergence_plot(
                convergence_test_indicies=convergence_test_indicies, DA_max=DA_max
            )
        return ax, lines_2

    def get_poincare_data(self):
        """
        Return the Poincare map data.

        Returns:
            s_all, chis_all, etas_all, vpars_all, t_all : Lists of trajectory data.
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


def compute_rotational_profile(
    field,
    pitch,
    sgn,
    mass,
    charge,
    Ekin,
    helicity_M,
    helicity_N,
    helicity_Mp,
    helicity_Np,
    comm,
    ns_poinc=100,
    Nmaps=75,
    s_profile=False,
    tmax=1e-2,
    solver_options=None,
):
    r"""
    Compute the rotational-transform and orbit-helicity profile from a
    passing-particle Poincare map at a fixed pitch angle.

    Args:
        field : The (unperturbed) BoozerMagneticField instance to trace in.
        pitch : Pitch angle variable, lambda = vperp^2 / (v^2 B).
        sgn : Desired sign of the parallel velocity (+1 or -1).
        mass : Particle mass.
        charge : Particle charge.
        Ekin : Total kinetic energy.
        helicity_M : Poloidal helicity of the field strength.
        helicity_N : Toroidal helicity of the field strength.
        helicity_Mp : Poloidal helicity of the mapping coordinate eta.
        helicity_Np : Toroidal helicity of the mapping coordinate eta.
        comm : MPI communicator.
        ns_poinc : Number of initial flux-surface labels for the Poincare map
                   (default: 100).
        Nmaps : Number of Poincare return maps to compute (default: 75).
        s_profile : If True, average over flux-surface label instead of
                    initial condition (default: False).
        tmax : Maximum integration time (default: 1e-2).
        solver_options : Dict of solver options passed to PassingPoincare. If
                          None, defaults to {"axis": 0}.

    Returns:
        profiles : Array of shape (npoints, 4) containing, for each initial
                   condition and sorted by radial coordinate, the columns
                   (radial_position, omega_theta, omega_zeta, orbit_helicity).
    """
    if solver_options is None:
        solver_options = {"axis": 0}
    poinc = PassingPoincare(
        field,
        np.abs(pitch),
        sgn,
        mass,
        charge,
        Ekin,
        ns_poinc=ns_poinc,
        ntheta_poinc=1,
        Nmaps=Nmaps,
        comm=comm,
        tmax=tmax,
        solver_options=solver_options,
        helicity_M=helicity_M,
        helicity_N=helicity_N,
        helicity_Mp=helicity_Mp,
        helicity_Np=helicity_Np,
    )
    data = poinc.compute_frequencies(s_profile=s_profile)
    # returns omega_theta_prof, omega_zeta_prof, peta_prof, s_prof
    # or omega_theta_prof, omega_zeta_prof, s_prof if s_profile is True
    data = np.column_stack(
        [
            data[2],
            data[0],
            data[1],
            [data[0][i] / data[1][i] for i in range(len(data[0]))],
        ]
    )
    profiles = data[data[:, 0].argsort()]
    # sort by radial coordinate
    return profiles


def accumulate_resonance_crossings(
    harmonics,
    profile,
    pitch_angle,
    mode_numbers,
    helicity_M,
    helicity_N,
    omega,
    max_ell,
):
    r"""
    Update harmonics in place with the resonance crossings found in profile
    at the given pitch angle, for each mode and ell in range(-max_ell,
    max_ell + 1).

    Args:
        harmonics : Dict of the form {h: {}} to populate (h indexes into
                    mode_numbers). On return, harmonics[h][ell] is a list of
                    the form [[pitch_angles], [radii]], one entry per
                    distinct crossing line found across calls.
        profile : Rotational-transform profile as returned by
                  compute_rotational_profile, with columns
                  (radial_position, omega_theta, omega_zeta, orbit_helicity).
        pitch_angle : Pitch angle at which profile was computed.
        mode_numbers : Dict {h: (Phim_h, Phin_h)} of poloidal/toroidal mode
                       numbers to search for resonances.
        helicity_M : Poloidal helicity of the field strength.
        helicity_N : Toroidal helicity of the field strength.
        omega : Frequency of the perturbation.
        max_ell : Resonances are searched for ell in
                  range(-max_ell, max_ell + 1).
    """
    drift_helicity = profile[:, 3]
    radial_position = profile[:, 0]
    for h, (Phim_h, Phin_h) in mode_numbers.items():
        for ell in range(-max_ell, max_ell + 1):
            h_res = calculate_QS_resonance(
                Phim_h,
                Phin_h,
                helicity_M,
                helicity_N,
                omega,
                np.mean(profile[:, 2]),
                ell=ell,
            )
            crossings = calculate_crossings(drift_helicity, h_res, radial_position)

            for crossing_index, radius in enumerate(crossings):
                if ell in harmonics[h]:
                    # if the resonance location intercepts the rotational
                    # profile multiple times, we want to store all of the
                    # crossing locations. if this is the first entry, start
                    # empty lists
                    if crossing_index > (len(harmonics[h][ell]) - 1):
                        harmonics[h][ell].append([[], []])
                    harmonics[h][ell][crossing_index][0].append(pitch_angle)
                    harmonics[h][ell][crossing_index][1].append(radius)
                else:
                    harmonics[h][ell] = [[[pitch_angle], [radius]]]
