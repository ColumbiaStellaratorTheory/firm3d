from warnings import warn

import numpy as np

from .._core.util import parallel_loop_bounds
from ..field.boozermagneticfield import (
    ShearAlfvenHarmonic,
    ShearAlfvenWavesSuperposition,
)
from ..field.tracing import (
    MaxToroidalFluxStoppingCriterion,
    trace_particles_boozer,
    trace_particles_boozer_perturbed,
)

from ._utils import compute_peta, return_DA


class WBAPerturbedParticles:
    def __init__(
        self,
        saw,
        mass,
        charge,
        Ekin,
        Phin,
        Phim,
        omega,
        helicity_N,
        helicity_M,
        helicity_Mp=None,
        helicity_Np=None,
        points=None,
        v_pars=None,
        mu_per_mass=None,
        tmax=1e-2,
        min_timestep=1e-7,
        tol=1e-9,
        gc_tys=None,
        savedata=False,
        save_gc_trajectories=False,
        savepath="",
        comm=None,
        solver_options=None,
        nconvergence_points=1,
    ):
        r"""
        Initialize the WBAPerturbedParticles instance for computing the
        Weighted Birkhoff Average (WBA) digit accuracy of guiding-center
        trajectories in a perturbed ShearAlfvenWave field.

        Either pre-traced trajectories (gc_tys) or initial conditions (points,
        v_pars, mu_per_mass) must be provided. If initial conditions are given,
        particles are traced using trace_particles_boozer_perturbed.

        Args:
            saw                  : A :class:`ShearAlfvenHarmonic` or compatible
                                perturbed-field instance.
            mass                 : Particle mass.
            charge               : Particle charge.
            Ekin                 : Total kinetic energy.
            Phin                 : Toroidal mode number of the wave harmonic,
                                used to compute nprime = (Phim*N - Phin*M)/denom.
            Phim                 : Poloidal mode number of the wave harmonic,
                                used to compute nprime.
            omega                : Wave frequency.
            helicity_N           : Toroidal helicity of the field-strength contours.
            helicity_M           : Poloidal helicity of the field-strength contours.
            helicity_Mp          : Poloidal helicity of the mapping coordinate eta.
                                If None, determined automatically from helicity_M.
            helicity_Np          : Toroidal helicity of the mapping coordinate eta.
                                If None, determined automatically from helicity_N.
            points               : Array of shape (N, 3) or (N, 4) containing
                                initial coordinates (s, theta, zeta[, t]).
                                Required if gc_tys is None.
            v_pars               : Array of initial parallel velocities. Required
                                if gc_tys is None.
            mu_per_mass          : Array of initial magnetic moments divided by
                                mass. Required if gc_tys is None.
            tmax                 : Maximum integration time per particle
                                (default: 1e-2 s).
            min_timestep         : Minimum time-step size used as the save
                                interval (default: 1e-7 s).
            tol                  : ODE solver tolerance (default: 1e-9).
            gc_tys               : List of pre-traced trajectory arrays. If
                                provided, tracing is skipped (default: None).
            savedata             : If True, save diagnostics to disk
                                (default: False).
            save_gc_trajectories : If True, save raw trajectory arrays to disk
                                (default: False).
            savepath             : Prefix for output file names (default: '').
            comm                 : MPI communicator for parallel execution
                                (default: None).
            solver_options       : Dictionary of additional options passed to the
                                ODE solver (default: {}).
            nconvergence_points  : Number of intermediate WBA evaluations per
                                trajectory (default: 1).
        """
        if gc_tys is None and points is None:
            raise ValueError("Need to provide traced trajectories or points to trace.")

        if points is not None and (v_pars is None or mu_per_mass is None):
            raise ValueError(
                "If providing points to trace, need v_pars and mu_per_mass."
            )

        if points is not None:
            if points.shape[1] not in [3, 4]:
                raise ValueError(
                    "Points must have shape (npoints, 4) for (s, theta, zeta, t) or "
                    "(npoints, 3) for (s, theta, zeta)"
                )
            if points.shape[1] == 4:
                points = points[:, :3]

        if not isinstance(saw, ShearAlfvenHarmonic) and not isinstance(
            saw, ShearAlfvenWavesSuperposition
        ):
            raise TypeError(
                "Expected saw to be an instance of ShearAlfvenHarmonic "
                "or ShearAlfvenWavesSuperposition"
            )

        if solver_options is None:
            solver_options = {}

        self.saw = saw
        self.B0 = saw.B0
        self.helicity_M = helicity_M
        self.helicity_N = helicity_N

        if helicity_Mp is None or helicity_Np is None:
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
        self.helicity_Mp = helicity_Mp
        self.helicity_Np = helicity_Np

        self.Phin = Phin
        self.Phim = Phim
        self.omega = omega
        self.mass = mass
        self.charge = charge
        self.Ekin = Ekin
        self.nprime = (self.Phim * self.helicity_N - self.Phin * self.helicity_M) / (
            self.helicity_Np * self.helicity_M - self.helicity_N * self.helicity_Mp
        )
        self.vtotal = np.sqrt(2 * self.Ekin / mass)

        self.comm = comm
        self.verbose = False
        if self.comm is None or self.comm.rank == 0:
            self.verbose = True

        self.solver_options = solver_options

        self.tmax = tmax
        self.min_timestep = min_timestep
        self.convergence_points = nconvergence_points

        self.savedata = savedata
        self.save_gc_trajectories = save_gc_trajectories

        # set parameters for convergence plot
        expected_length = int(self.tmax / self.min_timestep)
        expected_step = int(expected_length / self.convergence_points)
        self.WBA_transit_indicies = np.linspace(
            expected_step, expected_length - 1, num=nconvergence_points, dtype=int
        ).tolist()
        self.convergence_plot = nconvergence_points > 1

        self.tol = tol

        if self.savedata:
            self.IC_filepaths = "initial_conditions.txt"
            self.final_filepaths = {
                "DA": savepath + "DA_walllosttimes.txt",
                "TRAJS": savepath + "gc_tys.txt",
                "DATA": savepath + "DATA.txt",
            }

        if gc_tys is None:
            self.trace = True
        else:
            self.trace = False
            self.gc_tys = gc_tys

            if points is None:
                s_ic = []
                theta_ic = []
                zeta_ic = []
                v_pars = []

                for elem in gc_tys:
                    s_ic.append(elem[-1, 1])
                    theta_ic.append(elem[-1, 2])
                    zeta_ic.append(elem[-1, 3])
                    v_pars.append(elem[-1, 4])

                points = np.zeros((len(gc_tys), 3))
                points[:, 0] = s_ic
                points[:, 1] = theta_ic
                points[:, 2] = zeta_ic

            if mu_per_mass is None:
                warn(
                    "Expected mu_per_mass to be provided with gc_tys. "
                    "Computing mu_per_mass from gc_tys with reference energy, this"
                    " may be inaccurate if not provided directly.",
                    stacklevel=2,
                )
                mu_per_mass = []
                for i in range(len(gc_tys)):
                    self.B0.set_points(points[i, :])
                    modB = self.B0.modB()[:, 0]

                    eperp_per_mass = (self.Ekin / self.mass) - 0.5 * v_pars[i] ** 2
                    mu_per_mass.append(eperp_per_mass / modB[0])

        self.points0 = points
        self.v_pars0 = v_pars
        self.mu_per_mass0 = mu_per_mass

        (self.gc_tys, self.DAs, self.wall_lost, self.dense_output) = (
            self.trace_particles()
        )
        return

    def trace_particles(self):
        r"""
        Trace perturbed particle trajectories and compute DA outputs.

        Returns:
            res_tys : List of raw trajectory arrays (only populated when
                self.save_gc_trajectories is True; otherwise an empty list, or
                self.gc_tys when self.trace is False).
            DA_data : List of final digit-accuracy values, one per particle.
            wall_lost : List of final integration times per particle (used to
                flag wall-loss events).
            dense_output : List of per-particle summaries of the form
                [start_state, end_state, mean_state, convergence_data].
        """
        first, last = parallel_loop_bounds(self.comm, self.points0.shape[0])

        DA_data = []
        dense_output = []
        wall_lost = []
        res_tys = []

        for itrj in range(first, last):
            if self.trace:
                pts = np.zeros((1, 3))
                pts[:, 0] = self.points0[itrj, 0]
                pts[:, 1] = self.points0[itrj, 1]
                pts[:, 2] = self.points0[itrj, 2]
                gc_tys, gc_zeta_hits = trace_particles_boozer_perturbed(
                    perturbed_field=self.saw,
                    stz_inits=pts,
                    parallel_speeds=[self.v_pars0[itrj]],
                    mus=[self.mu_per_mass0[itrj]],
                    tmax=self.tmax,
                    mass=self.mass,
                    charge=self.charge,
                    Ekin=self.Ekin,
                    abstol=self.tol,
                    reltol=self.tol,
                    dt_save=self.min_timestep,
                    stopping_criteria=[MaxToroidalFluxStoppingCriterion(1.0)],
                    mode="gc_noK",
                    ODE_solver="dormand_prince",
                    **self.solver_options,
                )
                points_trajectory = gc_tys[0]
            else:
                points_trajectory = self.gc_tys[itrj]

            time_momentum, s_path, theta_path, zeta_path, vpar_path = (
                points_trajectory[:, 0],
                points_trajectory[:, 1],
                points_trajectory[:, 2],
                points_trajectory[:, 3],
                points_trajectory[:, 4],
            )
            points_trajectory = np.column_stack(
                (s_path, theta_path, zeta_path, time_momentum)
            )
            idx_wall = np.argmax(s_path >= 1) if np.any(s_path >= 1) else None
            if idx_wall is not None and s_path[idx_wall] >= 1:
                idx_wall -= 1
                points_trajectory = points_trajectory[:idx_wall, :]
                vpar_path = vpar_path[:idx_wall]

            self.saw.set_points(points_trajectory)

            modB = self.saw.B0.modB()[:, 0]
            Phi = self.saw.Phi()[:, 0]

            weighted_mu = self.mu_per_mass0[itrj] * self.mass

            E = 0.5 * self.mass * vpar_path**2 + weighted_mu * modB + self.charge * Phi

            Peta_values = compute_peta(
                self.saw,
                points_trajectory,
                vpar_path,
                self.mass,
                self.charge,
                self.helicity_M,
                self.helicity_N,
                self.helicity_Mp,
                self.helicity_Np,
            )
            Eprime = self.nprime * E - self.omega * Peta_values

            if points_trajectory.shape[0] > 10:
                stack_data = np.column_stack((points_trajectory[:, -1], Peta_values))
                time_eval, DA_eval = return_DA(stack_data)
                final_DA = DA_eval
            else:
                final_DA = np.nan

            convergence_times = []
            convergence_petas = []
            convergence_energies = []
            convergence_DAs = []

            for _conv_index, timing_index in enumerate(self.WBA_transit_indicies):
                if timing_index > len(time_momentum):
                    break
                convergence_times.append(time_momentum[timing_index])
                convergence_petas.append(Peta_values[timing_index])
                convergence_energies.append(E[timing_index])

                stack_data = np.column_stack(
                    (time_momentum[:timing_index], Peta_values[:timing_index])
                )
                time_eval, DA_eval = return_DA(stack_data)
                convergence_DAs.append(DA_eval)

            convergence_data = [
                convergence_times,
                convergence_petas,
                convergence_DAs,
                convergence_energies,
            ]
            # start state:
            # [s, theta, zeta, vpar, mu, peta, E, Eprime]
            # end state:
            # [t, s, theta, zeta, vpar, peta, E, Eprime, DA]
            # mean state:
            # [s_mean, peta_mean, E_mean, Eprime_mean]
            start_state = [
                points_trajectory[0, 0],
                points_trajectory[0, 1],
                points_trajectory[0, 2],
                vpar_path[0],
                weighted_mu,
                Peta_values[0],
                E[0],
                Eprime[0],
            ]

            end_state = [
                points_trajectory[-1, 3],
                points_trajectory[-1, 0],
                points_trajectory[-1, 1],
                points_trajectory[-1, 2],
                vpar_path[-1],
                Peta_values[-1],
                E[-1],
                Eprime[-1],
                final_DA,
            ]

            mean_state = [
                np.mean(points_trajectory[:, 0]),
                np.mean(Peta_values),
                np.mean(E),
                np.mean(Eprime),
            ]

            particle_out = [start_state, end_state, mean_state, convergence_data]
            DA_data.append(final_DA)
            wall_lost.append(points_trajectory[-1, 3])
            dense_output.append(particle_out)
            if self.save_gc_trajectories:
                res_tys.append(points_trajectory)

        if self.comm is not None:
            res_tys = [i for o in self.comm.allgather(res_tys) for i in o]
            DA_data = [i for o in self.comm.allgather(DA_data) for i in o]
            wall_lost = [i for o in self.comm.allgather(wall_lost) for i in o]
            dense_output = [i for o in self.comm.allgather(dense_output) for i in o]

        if self.verbose:
            import pickle

            if self.save_gc_trajectories:
                self.gc_tys = res_tys
                with open(self.final_filepaths["TRAJS"], "wb") as f:
                    pickle.dump(res_tys, f)
            if self.savedata:
                np.savetxt(
                    self.final_filepaths["DA"], np.column_stack((DA_data, wall_lost))
                )
                with open(self.final_filepaths["DATA"], "wb") as f:
                    pickle.dump(dense_output, f)
        if not self.trace:
            res_tys = self.gc_tys
        self.build_lists(dense_output)
        return res_tys, DA_data, wall_lost, dense_output

    def build_lists(self, dense_output):
        r"""
        Process trajectory summary data into lists that can be compared
        across many particles.

        Populates self.DAs_at_loss, self.DA_at_tfinal, self.lost_total,
        self.final_times, self.Peta_init/mean/final, self.E_init/mean/final,
        self.Eprime_init/mean/final, and the convergence_* arrays from the
        list of per-particle state tuples produced by trace_particles.

        Args:
            dense_output : List of per-particle trajectory summaries, each a
                list of the form
                [start_state, end_state, mean_state, convergence_data].
        """

        DAs_at_loss = []
        DA_tfinal = []

        lost_total = []
        final_times = []

        mus = []
        Peta_init = []
        Peta_mean = []
        Peta_final = []
        E_init = []
        E_mean = []
        E_final = []
        Eprime_init = []
        Eprime_mean = []
        Eprime_final = []

        s0 = []
        theta0 = []
        zeta0 = []
        vpar0 = []

        convergence_times = []
        convergence_petas = []
        convergence_energies = []
        convergence_DAs = []

        for elem in dense_output:
            # start state vector:
            #   [s, theta, zeta, vpar, mu, peta, E, Eprime]
            # end state vector:
            #   [t, s, theta, zeta, vpar, peta, E, Eprime, DA]
            # mean state vector:
            #   [s_mean, peta_mean, E_mean, Eprime_mean]
            # convergence state vector:
            #   [times, petas, DAs, energies]

            start_state = elem[0]
            end_state = elem[1]
            means = elem[2]
            convergence = elem[3]

            final_time = end_state[0]

            final_times.append(final_time)

            if final_time < (self.tmax - (5 * self.min_timestep)):
                lost_total.append(1)
                DA_tfinal.append(np.nan)
            else:
                lost_total.append(0)
                DA_tfinal.append(end_state[8])

            DAs_at_loss.append(end_state[8])

            s0.append(start_state[0])
            theta0.append(start_state[1])
            zeta0.append(start_state[2])
            vpar0.append(start_state[3])

            mus.append(start_state[4])

            Peta_init.append(start_state[5])
            Peta_mean.append(means[1])
            Peta_final.append(end_state[5])

            E_init.append(start_state[6])
            E_mean.append(means[2])
            E_final.append(end_state[6])

            Eprime_init.append(start_state[7])
            Eprime_mean.append(means[3])
            Eprime_final.append(end_state[7])

            convergence_times.append(convergence[0])
            convergence_petas.append(convergence[1])
            convergence_DAs.append(convergence[2])
            convergence_energies.append(convergence[3])

        self.mus = mus
        self.DAs_at_loss = DAs_at_loss
        self.DA_at_tfinal = DA_tfinal
        self.lost_total = lost_total
        self.final_times = final_times

        self.Peta_init = Peta_init
        self.Peta_mean = Peta_mean
        self.Peta_final = Peta_final
        self.E_init = E_init
        self.E_mean = E_mean
        self.E_final = E_final
        self.Eprime_init = Eprime_init
        self.Eprime_mean = Eprime_mean
        self.Eprime_final = Eprime_final

        self.s0 = s0
        self.theta0 = theta0
        self.zeta0 = zeta0
        self.vpar0 = vpar0

        self.convergence_times = convergence_times
        self.convergence_petas = convergence_petas
        self.convergence_DAs = convergence_DAs
        self.convergence_energies = convergence_energies
        return


class WBAParticles:
    def __init__(
        self,
        B0,
        mass,
        charge,
        Ekin,
        helicity_N,
        helicity_M,
        helicity_Mp=None,
        helicity_Np=None,
        points=None,
        v_pars=None,
        gc_tys=None,
        tmax=1e-2,
        min_timestep=1e-7,
        savedata=False,
        save_gc_trajectories=False,
        savepath="",
        comm=None,
        solver_options=None,
        tol=1e-9,
        convergence_points=1,
    ):
        r"""
        Initialize a Weighted Birkhoff Average (WBA) computation for guiding-
        center trajectories in an unperturbed BoozerMagneticField.

        Either pre-traced trajectories (gc_tys) or initial conditions
        (points, v_pars) must be provided. If initial conditions are given,
        particles are traced with trace_particles_boozer.

        Args:
            B0 : The :class:`BoozerMagneticField` instance.
            mass : Particle mass.
            charge : Particle charge.
            Ekin : Total kinetic energy.
            helicity_N : Toroidal helicity of the field-strength contours.
            helicity_M : Poloidal helicity of the field-strength contours.
            helicity_Mp : Poloidal helicity of the mapping coordinate eta. If
                None, determined automatically from helicity_M.
            helicity_Np : Toroidal helicity of the mapping coordinate eta. If
                None, determined automatically from helicity_N.
            points : Array of shape (N, 3) of initial (s, theta, zeta). Required
                if gc_tys is None.
            v_pars : Array of initial parallel velocities. Required if gc_tys is
                None.
            gc_tys : Pre-traced trajectory arrays; if provided, tracing is
                skipped.
            tmax : Maximum integration time per particle (s).
            min_timestep : Save interval for the ODE solver (s).
            savedata : If True, save DA arrays and trajectories to disk.
            save_gc_trajectories : If True, persist raw trajectories.
            savepath : Prefix for output filenames.
            comm : MPI communicator for parallel execution.
            solver_options : Extra options passed to the ODE solver.
            tol : Absolute and relative ODE tolerance.
            convergence_points : Number of intermediate WBA evaluations per
                trajectory.
        """
        self.B0 = B0
        self.helicity_M = helicity_M
        self.helicity_N = helicity_N

        if gc_tys is None and points is None:
            raise ValueError("Need to provide traced trajectories or points to trace.")

        if points is not None and (v_pars is None):
            raise ValueError(
                "If providing points to trace, need v_pars and mu_per_mass."
            )

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
        self.helicity_Mp = helicity_Mp
        self.helicity_Np = helicity_Np

        self.mass = mass
        self.charge = charge
        self.Ekin = Ekin
        self.convergence_points = convergence_points
        self.vtotal = np.sqrt(2 * self.Ekin / mass)

        self.comm = comm
        self.verbose = False
        if self.comm is None or self.comm.rank == 0:
            self.verbose = True

        if solver_options is None:
            solver_options = {}
        self.solver_options = solver_options
        self.tmax = tmax

        self.min_timestep = min_timestep
        self.savedata = savedata
        self.savepath = savepath
        self.save_gc_trajectories = save_gc_trajectories

        # set parameters for convergence plot
        expected_length = int(self.tmax / self.min_timestep)
        expected_step = int(expected_length / self.convergence_points)
        self.WBA_transit_indicies = np.linspace(
            expected_step, expected_length - 1, num=self.convergence_points, dtype=int
        ).tolist()
        self.convergence_plot = self.convergence_points > 1

        self.points0 = points
        self.v_pars0 = v_pars

        self.tol = tol

        if gc_tys is None:
            self.trace = True
        else:
            self.trace = False
            self.gc_tys = gc_tys

        if self.savedata:
            self.IC_filepaths = "initial_conditions.txt"
            self.final_filepaths = {
                "DA": self.savepath + "DA_walllosttimes.txt",
                "TRAJS": self.savepath + "gc_tys.txt",
                "DATA": self.savepath + "DATA.txt",
            }
        self.gc_tys, self.DAs, self.wall_lost, self.dense_output = (
            self.trace_particles()
        )
        self.build_lists(self.dense_output)

    def trace_particles(self):
        r"""
        Trace unperturbed particle trajectories and compute per-particle WBA
        digit-accuracy diagnostics.

        If self.trace is True, particles are integrated with
        trace_particles_boozer from the initial conditions (self.points0,
        self.v_pars0). Otherwise, pre-traced trajectories in self.gc_tys are
        reused. For each trajectory, the canonical momentum p_eta is evaluated,
        and the WBA digit accuracy is computed at every index in
        self.WBA_transit_indicies as well as at the final time.

        Returns:
            res_tys : List of raw trajectory arrays (only populated when
                self.save_gc_trajectories is True; otherwise an empty list, or
                self.gc_tys when self.trace is False).
            DA_data : List of final digit-accuracy values, one per particle.
            wall_lost : List of final integration times per particle (used to
                flag wall-loss events).
            dense_output : List of per-particle summaries of the form
                [start_state, end_state, convergence_data].
        """

        shape = self.points0.shape[0] if self.trace else len(self.gc_tys)

        first, last = parallel_loop_bounds(self.comm, shape)

        DA_data = []
        dense_output = []
        wall_lost = []
        res_tys = []

        for itrj in range(first, last):
            if self.trace:
                pt = np.zeros((1, 3))
                pt[0, 0] = self.points0[itrj, 0]
                pt[0, 1] = self.points0[itrj, 1]
                pt[0, 2] = self.points0[itrj, 2]
                gc_tys, gc_zeta_hits = trace_particles_boozer(
                    self.B0,
                    stz_inits=pt,
                    parallel_speeds=[self.v_pars0[itrj]],
                    tmax=self.tmax,
                    mass=self.mass,
                    charge=self.charge,
                    Ekin=self.Ekin,
                    abstol=self.tol,
                    reltol=self.tol,
                    dt_save=self.min_timestep,
                    stopping_criteria=[MaxToroidalFluxStoppingCriterion(1.0)],
                    mode="gc_noK",
                    ODE_solver="dormand_prince",
                    **self.solver_options,
                )
                points_trajectory = gc_tys[0]
            else:
                points_trajectory = self.gc_tys[itrj]

            time_momentum = points_trajectory[:, 0]
            s_path = points_trajectory[:, 1]
            theta_path = points_trajectory[:, 2]
            zeta_path = points_trajectory[:, 3]
            vpar_path = points_trajectory[:, 4]

            points_trajectory = np.column_stack((s_path, theta_path, zeta_path))
            idx_wall = np.argmax(s_path >= 1) if np.any(s_path >= 1) else None
            if idx_wall is not None and s_path[idx_wall] >= 1:
                idx_wall -= 1
                points_trajectory = points_trajectory[:idx_wall, :]
                time_momentum = time_momentum[:idx_wall]
                vpar_path = vpar_path[:idx_wall]

            Peta_values = compute_peta(
                self.B0,
                points=points_trajectory,
                vpar=vpar_path,
                mass=self.mass,
                charge=self.charge,
                helicity_M=self.helicity_M,
                helicity_N=self.helicity_N,
            )

            if points_trajectory.shape[0] > 8:
                stack_data = np.column_stack((points_trajectory[:, -1], Peta_values))
                time_eval, DA_eval = return_DA(stack_data)
                final_DA = DA_eval
            else:
                final_DA = np.nan

            convergence_times = []
            convergence_petas = []
            convergence_DAs = []

            for _conv_index, timing_index in enumerate(self.WBA_transit_indicies):
                if timing_index > len(time_momentum):
                    break
                convergence_times.append(time_momentum[timing_index])
                convergence_petas.append(Peta_values[timing_index])

                stack_data = np.column_stack(
                    (time_momentum[:timing_index], Peta_values[:timing_index])
                )
                time_eval, DA_eval = return_DA(stack_data)
                convergence_DAs.append(DA_eval)

            convergence_data = [
                convergence_times,
                convergence_petas,
                convergence_DAs,
            ]
            self.B0.set_points(points_trajectory[0:1, :])
            B = self.B0.modB()[0, 0]
            mu = (1 / 2) * self.mass * (self.vtotal**2 - vpar_path[0] ** 2) / B
            # start state vector:  [s, theta, zeta, vpar, mu]
            # end state vector:   [t, s, theta, zeta, vpar,  DA]
            start_state = [
                points_trajectory[0, 0],
                points_trajectory[0, 1],
                points_trajectory[0, 2],
                vpar_path[0],
                mu,
            ]

            end_state = [
                time_momentum[-1],
                points_trajectory[-1, 0],
                points_trajectory[-1, 1],
                points_trajectory[-1, 2],
                vpar_path[-1],
                final_DA,
            ]

            particle_out = [start_state, end_state, convergence_data]
            DA_data.append(final_DA)
            wall_lost.append(time_momentum[-1])
            dense_output.append(particle_out)

        if self.comm is not None:
            res_tys = [i for o in self.comm.allgather(res_tys) for i in o]
            DA_data = [i for o in self.comm.allgather(DA_data) for i in o]
            wall_lost = [i for o in self.comm.allgather(wall_lost) for i in o]
            dense_output = [i for o in self.comm.allgather(dense_output) for i in o]

        if self.verbose:
            import pickle

            if self.save_gc_trajectories and self.trace:
                with open(self.final_filepaths["TRAJS"], "wb") as f:
                    pickle.dump(res_tys, f)
            if self.savedata:
                np.savetxt(
                    self.final_filepaths["DA"], np.column_stack((DA_data, wall_lost))
                )
                with open(self.final_filepaths["DATA"], "wb") as f:
                    pickle.dump(dense_output, f)
        if not self.trace:
            res_tys = self.gc_tys
        return res_tys, DA_data, wall_lost, dense_output

    def build_lists(self, dense_output):
        r"""
        Process trajectory summary data into lists that can be compared
        across many particles.

        Populates self.DAs_at_loss, self.DA_at_tfinal, self.lost_total,
        self.final_times, self.s0, self.theta0, self.zeta0, self.vpar0,
        self.mus, and the convergence_* arrays from the list of per-particle
        state tuples produced by trace_particles.

        Args:
            dense_output : List of per-particle trajectory summaries, each a
                list of the form [start_state, end_state, convergence_data].
        """

        DAs_at_loss = []
        DA_tfinal = []

        lost_total = []
        final_times = []

        s0 = []
        theta0 = []
        zeta0 = []
        vpar0 = []
        mu0 = []

        convergence_times = []
        convergence_petas = []
        convergence_DAs = []

        for elem in dense_output:
            # start state vector:
            #   [s, theta, zeta, vpar, mu]
            # end state vector:
            #   [t, s, theta, zeta, vpar, DA]
            # convergence state vector:
            #   [times, petas, DAs]

            start_state = elem[0]
            end_state = elem[1]
            convergence = elem[2]

            final_time = end_state[0]

            final_times.append(final_time)

            if final_time < (self.tmax - (5 * self.min_timestep)):
                lost_total.append(1)
                DA_tfinal.append(np.nan)
            else:
                lost_total.append(0)
                DA_tfinal.append(end_state[5])

            DAs_at_loss.append(end_state[5])

            s0.append(start_state[0])
            theta0.append(start_state[1])
            zeta0.append(start_state[2])
            vpar0.append(start_state[3])
            mu0.append(start_state[4])

            convergence_times.append(convergence[0])
            convergence_petas.append(convergence[1])
            convergence_DAs.append(convergence[2])

        self.DAs_at_loss = DAs_at_loss
        self.DA_at_tfinal = DA_tfinal
        self.lost_total = lost_total
        self.final_times = final_times

        self.s0 = s0
        self.theta0 = theta0
        self.zeta0 = zeta0
        self.vpar0 = vpar0
        self.mus = mu0
        self.convergence_times = convergence_times
        self.convergence_petas = convergence_petas
        self.convergence_DAs = convergence_DAs

        return
