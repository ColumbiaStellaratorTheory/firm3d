from os.path import exists

import numpy as np
from scipy.stats import binned_statistic_2d

from .._core.util import parallel_loop_bounds
from ..util.functions import proc0_print
from ..field.tracing import (
    MaxToroidalFluxStoppingCriterion,
    trace_particles_boozer,
    trace_particles_boozer_perturbed,
)
from ..field.tracing_helpers import initialize_position_uniform_surf

from ._utils import (
    _check_filepaths,
    _solve_vpar_energy,
    _solve_vpar_perturbed,
    compute_peta,
    min_volumemodB,
    return_DA,
)


def _resolve_heatmap_bins(nx, ny, nlambda_points, ns_points, npoints, randomize):
    r"""
    Determine the number of (pitch, radial) bins used by plot_heatmap.

    Values supplied by the user are always used as given. A value of None is
    replaced by a default taken from the sampling grid if the initial
    conditions were generated on a structured (mu, s) grid: nlambda_points bins
    along the pitch axis and ns_points bins along the radial axis. Since the
    bins span the range of the data, one bin per grid point places each grid
    value in its own bin. If the initial conditions were randomized, or the
    grid resolution is unknown, the defaults are instead set from the cube root
    of the number of particles.

    Args:
        nx : Requested number of pitch bins, or None to use the default.
        ny : Requested number of radial bins, or None to use the default.
        nlambda_points : Number of magnetic-moment values in the sampling grid,
            or None if unknown.
        ns_points : Number of flux surfaces in the sampling grid, or None if
            unknown.
        npoints : Number of traced particles.
        randomize : If True, the initial conditions were sampled randomly
            rather than on a structured grid.

    Returns:
        nx : Number of pitch bins.
        ny : Number of radial bins.
    """
    npoints_default = max(int(np.cbrt(npoints)), 1)
    if nx is None:
        if randomize or nlambda_points is None:
            nx = npoints_default
        else:
            nx = max(int(nlambda_points), 1)
    if ny is None:
        if randomize or ns_points is None:
            ny = npoints_default
        else:
            ny = max(int(ns_points), 1)
    return nx, ny


class MapEquilibrium:
    def __init__(
        self,
        B0,
        mass,
        charge,
        Ekin,
        helicity_N,
        helicity_M,
        plot_s=True,
        helicity_Mp=None,
        helicity_Np=None,
        mu_lims=None,
        sign=1,
        ns_points=25,
        particles_per_surface=25,
        nlambda_points=25,
        savedata=True,
        savepath="",
        randomize_particles=False,
        number_of_particles=10000,
        min_timestep=1e-7,
        nconvergence_points=100,
        s_lims=None,
        min_DA_time=None,
        comm=None,
        tmax=1e-2,
        tol=1e-10,
        solver_options=None,
    ):
        """
        Initialize phase-space sampling, particle tracing, and diagnostic evaluation
        for guiding-center orbits in an equilibrium Boozer magnetic field.

        This class generates or accepts a collection of initial particle conditions
        and traces the corresponding guiding-center orbits in the equilibrium field.
        Per-particle diagnostics such as digit accuracy, wall-loss
        status, and phase-space coordinates used for plotting are computed and stored.

        The sampled phase space is parameterized by Boozer coordinates together with
        either:
            - a fixed total kinetic energy Ekin, or
            - a fixed shifted-energy slice Eprime.

        If Eprime is provided, the initialization solves for vpar from the invariant
        constraint rather than sampling vpar directly from fixed Ekin. This uses a
        reference kinetic energy Ekin for computing mu from vperp.

        Initial conditions may be supplied directly or generated internally in one of
        two ways:
            - randomly throughout the plasma volume, or
            - on a structured grid in surface label and pitch coordinate, with multiple
              particles sampled on each surface.

        Args:
            B0 : The :class:`BoozerMagneticField` instance.
            mass : Particle mass.
            charge : Particle charge.
            Ekin : Total kinetic energy.
            helicity_N : Toroidal helicity of the field-strength contours.
            helicity_M : Poloidal helicity of the field-strength contours.
            plot_s : If True, use flux-surface label s as the radial plot
                coordinate; otherwise use p_eta (default: True).
            helicity_Mp : Poloidal helicity of the mapping coordinate eta.
                If None, determined automatically from helicity_M.
            helicity_Np : Toroidal helicity of the mapping coordinate eta.
                If None, determined automatically from helicity_N.
            mu_lims : Two-element list [mu_min, mu_max] bounding the sampled
                magnetic moments. If None, limits are set from Ekin and the
                minimum |B| in the volume.
            sign : Sign of the parallel velocity (+1 or -1, default: 1).
            ns_points : Number of flux surfaces in the structured grid
                (default: 25).
            particles_per_surface : Number of particles per flux surface
                (default: 25).
            nlambda_points : Number of magnetic-moment values in the structured
                grid (default: 25).
            savedata : If True, save initial conditions and tracing results to
                disk (default: True).
            savepath : Prefix for output file names (default: '').
            randomize_particles : If True, sample initial conditions uniformly
                at random rather than on a structured grid (default: False).
            number_of_particles : Total number of particles when
                randomize_particles is True (default: 10000).
            min_timestep : ODE save interval and minimum time step for computing
                the expected trajectory length (default: 1e-7 s).
            nconvergence_points : Number of intermediate WBA evaluations per
                trajectory (default: 100).
            s_lims : Two-element list [s_min, s_max] bounding the sampled flux
                surfaces (default: [0.05, 0.95]).
            min_DA_time : Minimum trajectory time required before a DA value is
                recorded; trajectories shorter than this receive NaN
                (default: 0).
            comm : MPI communicator for parallel execution (default: None).
            tmax : Maximum integration time per particle (default: 1e-2 s).
            tol : Absolute and relative ODE solver tolerance (default: 1e-10).
            solver_options : Dictionary of additional options passed to the ODE
                solver (default: {}).
        """
        if solver_options is None:
            solver_options = {}
        self.solver_options = solver_options
        if s_lims is None:
            s_lims = [0.05, 0.95]

        self.B0 = B0
        self.mass = mass
        self.charge = charge
        self.Ekin = Ekin
        self.helicity_N = helicity_N
        self.helicity_M = helicity_M
        self.min_volmodB = min_volumemodB(self.B0)

        if mu_lims is None:
            mu_max = (Ekin) / self.min_volmodB
            mu_min = 0
        else:
            mu_max = mu_lims[1]
            mu_min = mu_lims[0]
        self.mu_min = mu_min
        self.mu_max = mu_max

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
        self.tol = tol
        self.plot_s = plot_s

        # set timing parameters
        self.tmax = tmax
        self.min_timestep = min_timestep

        # set communicator parameters
        self.comm = comm
        self.verbose = False
        if self.comm is None or self.comm.rank == 0:
            self.verbose = True

        self.sign = sign

        self.s_min = s_lims[0]
        self.s_max = s_lims[1]

        if min_DA_time is None:
            min_DA_time = 0
        if min_DA_time > self.tmax:
            raise ValueError("min_DA_time must be less than or equal to tmax.")
        self.min_DA_time = min_DA_time

        # plotting settings
        self.savedata = savedata
        if savepath != "":
            savepath += "_"

        self.savepath = savepath
        load_ics = False
        load_files = False
        if savedata:
            self.res_filepaths = {
                "tys": savepath + "DA_data.txt",
                "ICs": savepath + "initial_conditions.txt",
            }
            if exists(self.res_filepaths["ICs"]):
                load_ics = True
            if exists(self.res_filepaths["tys"]):
                load_files = True
        self.convergence_points = nconvergence_points

        self.randomize = randomize_particles
        if not randomize_particles:
            self.ns_points = ns_points
            self.nlambda_points = nlambda_points
        else:
            xy_pts = int(np.sqrt(number_of_particles / particles_per_surface))
            self.ns_points = xy_pts
            self.nlambda_points = xy_pts

        if load_ics:
            initial_conditions = np.loadtxt(self.res_filepaths["ICs"])
            s, thetas, zetas, vpar, mu = (
                initial_conditions[:, 0],
                initial_conditions[:, 1],
                initial_conditions[:, 2],
                initial_conditions[:, 3],
                initial_conditions[:, 4],
            )
        else:
            self.particles_per_surface = particles_per_surface
            s, thetas, zetas, vpar, mu = self.initialize_particles()

        self.s, self.thetas, self.zetas, self.vpar, self.mu = s, thetas, zetas, vpar, mu

        self.expected_length = int(self.tmax / self.min_timestep)
        self.expected_step = int(self.expected_length / self.convergence_points)
        self.WBA_transit_indicies = np.linspace(
            self.expected_step,
            self.expected_length - 1,
            num=self.convergence_points,
            dtype=int,
        ).tolist()
        self.convergence_plot = self.convergence_points > 1

        self.trace_particles(load_files)
        return

    def initialize_particles(self):
        r"""
        Generate initial particle positions, parallel velocities, and magnetic
        moments for tracing.

        Initial flux surfaces and magnetic moments are sampled either on a
        structured (s, mu) grid or uniformly at random, depending on
        self.randomize. On each surface, particles are scattered uniformly using
        initialize_position_uniform_surf. Points with vpar^2 < 0 from energy
        conservation are dropped.

        Returns:
            s : List of initial s coordinates.
            thetas : List of initial theta coordinates.
            zetas : List of initial zeta coordinates.
            vpar : List of initial parallel velocities (signed by self.sign).
            mu : List of initial magnetic moments.
        """
        if self.randomize:
            mus = np.random.uniform(self.mu_min, self.mu_max, self.nlambda_points)

            s_linspace = np.random.uniform(self.s_min, self.s_max, self.ns_points)
        else:
            mus = np.linspace(self.mu_min, self.mu_max, self.nlambda_points)

            s_linspace = np.linspace(self.s_min, self.s_max, self.ns_points)

        surfaces, mus = np.meshgrid(s_linspace, mus)

        surfaces_flat = surfaces.flatten()
        mus_flat = mus.flatten()

        s = []
        thetas = []
        zetas = []
        vpars = []
        mus_tot = []
        for particle_index in range(len(surfaces_flat)):
            points_temp = initialize_position_uniform_surf(
                self.B0,
                self.particles_per_surface,
                surfaces_flat[particle_index],
                comm=self.comm,
            )

            mu = np.full(points_temp.shape[0], mus_flat[particle_index])

            vpar = _solve_vpar_energy(
                self.B0,
                points_temp,
                self.mass,
                self.Ekin,
                mu,
                self.sign,
            )

            # remove unphysical points
            neg_idx = np.where(np.isnan(vpar))[0]
            vpar = np.delete(vpar, neg_idx)
            points_temp = np.delete(points_temp, neg_idx, axis=0)
            mu = np.delete(mu, neg_idx, axis=0)

            vpars += vpar.tolist()
            s += list(points_temp[:, 0])
            thetas += list(points_temp[:, 1])
            zetas += list(points_temp[:, 2])
            mus_tot += mu.tolist()

        return s, thetas, zetas, vpars, mus_tot

    def trace_particles(self, load_files=False):
        """
        Trace particles in the equilibrium field and compute per-particle
        diagnostics. Results are passed to build_lists for storage.

        Args:
            load_files : If True and the data file exists at
                self.res_filepaths["tys"], load previously traced results from
                disk instead of re-tracing (default: False).
        """
        import pickle

        if load_files:  # noqa: SIM102
            if self.verbose:
                proc0_print("Reading File")
            with open(self.res_filepaths["tys"], "rb") as f:
                res_tys = pickle.load(f)
            self.build_lists(res_tys)
            return

        first, last = parallel_loop_bounds(self.comm, len(self.s))
        gc_tys = []

        for itrj in range(first, last):
            point = np.zeros((1, 3))  # initialize with t = 0
            point[:, 0] = self.s[itrj]
            point[:, 1] = self.thetas[itrj]
            point[:, 2] = self.zetas[itrj]

            vpar = [self.vpar[itrj]]
            mu = self.mu[itrj]
            res_tys, res_zeta_hits = trace_particles_boozer(
                self.B0,
                point,
                vpar,
                tmax=self.tmax,
                mass=self.mass,
                charge=self.charge,
                Ekin=self.Ekin,
                stopping_criteria=[MaxToroidalFluxStoppingCriterion(1.0)],
                forget_exact_path=False,
                dt_save=self.min_timestep,
                abstol=self.tol,
                reltol=self.tol,
                **self.solver_options,
            )
            points_trajectory = res_tys[0]

            if points_trajectory.ndim != 2:
                continue

            time_momentum, s_path, theta_path, zeta_path, vpar_path = (
                points_trajectory[:, 0],
                points_trajectory[:, 1],
                points_trajectory[:, 2],
                points_trajectory[:, 3],
                points_trajectory[:, 4],
            )
            points_trajectory = np.column_stack((s_path, theta_path, zeta_path))

            # avoid tracing bug that sometimes causes last pt to be outside wall
            idx_wall = np.argmax(s_path >= 1) if np.any(s_path >= 1) else None
            if idx_wall is not None and s_path[idx_wall] >= 1:
                idx_wall -= 1
                points_trajectory = points_trajectory[:idx_wall, :]
                vpar_path = vpar_path[:idx_wall]
                time_momentum = time_momentum[:idx_wall]

            Peta_values = compute_peta(
                self.B0,
                points_trajectory,
                vpar_path,
                self.mass,
                self.charge,
                self.helicity_M,
                self.helicity_N,
                self.helicity_Mp,
                self.helicity_Np,
            )

            # start_state = [s, theta, zeta, vpar, p_eta_0, mu]
            start_state = [
                points_trajectory[0, 0],
                points_trajectory[0, 1],
                points_trajectory[0, 2],
                vpar[0],
                Peta_values[0],
                mu,
            ]

            if len(Peta_values) > 10:
                stack_data = np.column_stack((time_momentum, Peta_values))
                time_eval, DA_eval = return_DA(stack_data)
                final_DA = DA_eval
            else:
                final_DA = np.nan

            # end_state = [time, s, theta, zeta, vpar, p_eta_f, DA]
            end_state = [
                time_momentum[-1],
                points_trajectory[-1, 0],
                points_trajectory[-1, 1],
                points_trajectory[-1, 2],
                vpar_path[-1],
                Peta_values[-1],
                final_DA,
            ]

            convergence_times = []
            convergence_petas = []
            convergence_DAs = []

            for _conv_index, timing_index in enumerate(self.WBA_transit_indicies):
                if timing_index >= len(time_momentum):
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

            gc_tys.append([start_state, end_state, convergence_data])

        if self.comm is not None:
            proc0_print(f"{self.comm.rank=} done tracing particles")
            gc_tys = [i for o in self.comm.allgather(gc_tys) for i in o]

        if self.verbose and self.savedata:
            with open(self.res_filepaths["tys"], "wb") as f:
                pickle.dump(gc_tys, f)

        self.build_lists(gc_tys)
        return

    def build_lists(self, res_tys):
        r"""
        Unpack per-particle trajectory summaries into flat instance attributes
        suitable for plotting and aggregation.

        Populates self.DAs_at_loss, self.DA_at_tfinal, self.pitch, self.lost_total,
        self.final_times, self.radial_coordinate_start, self.s0, self.mu0, and the
        convergence_* arrays from the list produced by trace_particles.
        If self.verbose, initial conditions are written to disk.

        Args:
            res_tys : List of per-particle summaries, each of the form
                    [start_state, end_state, convergence_data].
        """
        DAs_at_loss = []

        DA_tfinal = []

        lost_total = []
        final_times = []
        radial_coordinate_start = []
        pitch = []

        s0 = []
        theta0 = []
        zeta0 = []
        vpar0 = []
        mu0 = []

        convergence_times = []
        convergence_petas = []
        convergence_DAs = []

        tolerance = 5 * self.min_timestep

        for i in range(len(res_tys)):
            # start_state = [s, theta, zeta, vpar, p_eta_0, mu]
            # end_state = [time, s, theta, zeta, vpar, p_eta_f, DA]

            start_state = res_tys[i][0]
            end_state = res_tys[i][1]
            convergence_data = res_tys[i][2]

            pitch.append(start_state[5] / self.Ekin)
            if self.plot_s:
                radial_coordinate_start.append(start_state[0])
            else:
                radial_coordinate_start.append(start_state[4])
            final_time = end_state[0]
            final_times.append(final_time)

            if final_time < self.min_DA_time:
                DAs_at_loss.append(np.nan)
            else:
                DAs_at_loss.append(end_state[6])

            s0.append(start_state[0])
            theta0.append(start_state[1])
            zeta0.append(start_state[2])
            vpar0.append(start_state[3])
            mu0.append(start_state[5])

            # params that depend on loss
            if final_time < (self.tmax - tolerance):
                lost_total.append(1)
                DA_tfinal.append(np.nan)
            else:
                lost_total.append(0)
                DA_tfinal.append(end_state[6])
            convergence_times.append(convergence_data[0])
            convergence_petas.append(convergence_data[1])
            convergence_DAs.append(convergence_data[2])

        self.DAs_at_loss = DAs_at_loss
        self.DA_at_tfinal = DA_tfinal
        self.pitch = pitch

        self.lost_total = lost_total
        self.final_times = final_times
        self.radial_coordinate_start = radial_coordinate_start

        self.s0 = s0
        self.mu0 = mu0

        self.convergence_times = convergence_times
        self.convergence_petas = convergence_petas
        self.convergence_DAs = convergence_DAs

        if self.savedata and self.verbose:
            np.savetxt(
                self.savepath + "initial_conditions.txt",
                np.column_stack((s0, theta0, zeta0, vpar0, mu0)),
            )

        return

    def plot_heatmap(
        self,
        nx=None,
        ny=None,
        savepath="heatmap_digit_accuracy.pdf",
        DA_at_loss=True,
        ax=None,
        DA_max=7,
        peta_exp=None,
        statistic="mean",
        plot_losses=False,
    ):
        r"""
        Create and save a 2D heatmap of digit accuracy in the
        (pitch angle, flux-surface label) plane, with the trapped-passing
        boundary overlaid as a fitted curve.

        Args:
            nx          : Number of bins along the pitch-angle axis. If None,
                        defaults to nlambda_points for a structured grid of
                        initial conditions, or the cube root of the number of
                        particles if they were randomized (default: None).
            ny          : Number of bins along the radial axis. If None,
                        defaults to ns_points for a structured grid of
                        initial conditions, or the cube root of the number of
                        particles if they were randomized (default: None).
            savepath    : File path for the output heatmap image
                        (default: 'heatmap_digit_accuracy.pdf').
            DA_at_loss : If True, use the digit accuracy value at the time of
                        loss; otherwise use the value at the end of the full
                        integration (default: True).
            ax          : Matplotlib axis to plot on. If None, a new figure and
                        axis are created.
            DA_max      : Maximum digit accuracy value shown on the colorbar. If
                        None, the maximum DA in the data is used instead
                        (default: 7).
            statistic   : Aggregation statistic passed to binned_statistic_2d
                        (default: 'mean').
            plot_losses : If True, overlay loss-fraction markers per bin
                        (default: False).
            peta_exp : Exponent to apply to the p_eta values axis for plotting.
                Should be the integer of magnitude (eg. -19)

        Returns:
            None
        """
        import matplotlib as mpl
        import matplotlib.pyplot as plt

        fDA = np.array(self.DAs_at_loss) if DA_at_loss else np.array(self.DA_at_tfinal)

        if DA_max is None:
            DA_max = np.nanmax(fDA)

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

        nx, ny = _resolve_heatmap_bins(
            nx,
            ny,
            self.nlambda_points,
            self.ns_points,
            len(self.pitch),
            self.randomize,
        )

        def trapped_passing_function(s, pitch):
            # pitch not weighted by modB
            resolution = 500
            points_temp = initialize_position_uniform_surf(
                self.B0,
                resolution,
                s,
                comm=None,
            )
            self.B0.set_points(points_temp)
            modB = self.B0.modB()[:, 0]
            max_modB = np.max(modB)
            mu = pitch * self.Ekin

            vp_temp = _solve_vpar_energy(
                self.B0,
                points_temp,
                self.mass,
                self.Ekin,
                mu,
                self.sign,
            )

            mask = ~np.isnan(vp_temp)
            points_temp = points_temp[mask]
            modB = modB[mask]
            vp_temp = vp_temp[mask]

            if points_temp.shape[0] == 0:
                return 0, [None]

            peta = compute_peta(
                self.B0,
                points_temp,
                vp_temp,
                self.mass,
                self.charge,
                self.helicity_M,
                self.helicity_N,
                self.helicity_Mp,
                self.helicity_Np,
            )
            # mask = ~np.isnan(peta)
            # peta = peta[mask]

            if np.any(1 - (pitch * max_modB) < 0):
                return 1, peta
            return 0, peta

        def make_boundary(petas, pitches, trapped):

            trapped_vals, pitch_edges, radlike_edges, binnumber = binned_statistic_2d(
                pitches,
                petas,
                trapped,
                statistic="mean",
                bins=[70, 70],
            )
            pitch_c = 0.5 * (pitch_edges[:-1] + pitch_edges[1:])
            radlike_c = 0.5 * (radlike_edges[:-1] + radlike_edges[1:])

            T = np.nan_to_num(trapped_vals, nan=0.0)
            # x, y -> (pitch, peta) dimensions

            boundary_pitch, boundary_radlike = [], []

            for peta_i in range(0, T.shape[1]):
                pitch_data = T[:, peta_i]
                if not pitch_data.any() or pitch_data.all():
                    continue
                if self.sign == 1:
                    pitch_i = int(np.argmax(pitch_data > 0.5))
                else:
                    pitch_i = int(np.argmin(pitch_data > 0.5))
                pitch_value = pitch_c[pitch_i]
                boundary_pitch.append(pitch_value)
                boundary_radlike.append(radlike_c[peta_i])

            boundary_pitch = np.array(boundary_pitch)
            boundary_radlike = np.array(boundary_radlike)

            order = np.argsort(boundary_pitch)
            boundary_pitch = boundary_pitch[order]
            boundary_radlike = boundary_radlike[order]
            boundary_pitch *= self.min_volmodB
            return boundary_pitch, boundary_radlike

        normalized_pitch = np.array(self.pitch) * self.min_volmodB * self.sign
        radial_coordinate_start = np.array(self.radial_coordinate_start)

        stat, x_edges, y_edges, binnumber = binned_statistic_2d(
            normalized_pitch,
            radial_coordinate_start,
            fDA,
            statistic=statistic,
            bins=[nx, ny],
        )
        norm = mpl.colors.Normalize(vmin=0, vmax=DA_max)
        X2, Y2 = np.meshgrid(x_edges, y_edges)
        im2 = ax.pcolormesh(X2, Y2, stat.T, shading="auto", cmap=cmap, norm=norm)
        fig.colorbar(im2, ax=ax, label="Digit Accuracy")

        ax.set_xlabel(r"$\lambda = \frac{\mu}{E} \text{sign}(v_{||})$")

        s_scope = np.linspace(self.s_min, self.s_max, 75)[::-1]
        pa_min, pa_max = (self.mu_min / self.Ekin), (self.mu_max / self.Ekin)
        pa_scope = np.linspace(pa_min, pa_max, 75)

        rad_like_tp = []
        pa_tp = []
        trapped_vals = []

        for s in s_scope:
            for pa in pa_scope:
                normalized_pitch_i = pa * self.sign
                trapped, rad_like = trapped_passing_function(s, pa)
                if trapped == 1 and self.plot_s:
                    rad_like_tp.append(s)
                    pa_tp.append(normalized_pitch_i * self.min_volmodB)
                    break
                if not self.plot_s:
                    if rad_like[0] is None:
                        continue
                    if not isinstance(rad_like, list):
                        rad_like = rad_like.tolist()
                    rad_like_tp += rad_like
                    pa_tp += [normalized_pitch_i] * len(rad_like)
                    trapped_vals += [trapped] * len(rad_like)

        if self.plot_s:
            ax.set_ylabel(r"$s$")
        else:
            ax.set_ylabel(r"$P_\eta$")
            pa_tp, rad_like_tp = make_boundary(rad_like_tp, pa_tp, trapped_vals)

        if len(pa_tp) > 3:
            rad_like_tp = np.array(rad_like_tp)
            pa_tp = np.array(pa_tp)
            trapped_vals = np.array(trapped_vals)
            coeffs = np.polyfit(pa_tp, rad_like_tp, 2)
            poly = np.poly1d(coeffs)
            pa_fit = np.linspace(min(pa_tp), max(pa_tp), 100)
            s_fit = poly(pa_fit)
            min_idx = np.argmin(s_fit)
            if self.sign == 1:
                s_fit = s_fit[: min_idx + 1]
                pa_fit = pa_fit[: min_idx + 1]
            else:
                s_fit = s_fit[min_idx:]
                pa_fit = pa_fit[min_idx:]

            ax.plot(
                pa_fit,
                s_fit,
                color="grey",
                linewidth=5,
                label="Trapped-passing boundary",
                zorder=20,
            )

        if peta_exp is not None:
            import matplotlib.ticker as mticker

            class FixedOrderFormatter(mticker.ScalarFormatter):
                def __init__(self, order, fformat="%1.1f", mathText=True):
                    self.oom = order
                    self.fformat = fformat
                    super().__init__(useMathText=mathText)

                def _set_order_of_magnitude(self):
                    self.orderOfMagnitude = self.oom

                def _set_format(self):
                    self.format = self.fformat
                    if self._useMathText:
                        self.format = f"$\\mathdefault{{{self.format}}}$"

            ax.yaxis.set_major_formatter(FixedOrderFormatter(peta_exp))
            ax.ticklabel_format(
                axis="y",
                scilimits=(0, 0),  # ensure sci notation is used
            )

        if plot_losses:
            lost_frac, x_edges, y_edges, _ = binned_statistic_2d(
                normalized_pitch,
                radial_coordinate_start,
                np.array(self.lost_total),
                statistic="max",
                bins=[nx, ny],
            )
            x_centers = 0.5 * (x_edges[1:] + x_edges[:-1])
            y_centers = 0.5 * (y_edges[1:] + y_edges[:-1])
            Xc, Yc = np.meshgrid(x_centers, y_centers)
            xf = Xc.ravel()
            yf = Yc.ravel()
            lost_frac = np.nan_to_num(lost_frac, nan=0.0)
            af = lost_frac.T.ravel()
            ax.scatter(
                xf,
                yf,
                marker="s",
                s=15,
                c="darkred",
                alpha=af,
                zorder=10,
            )

        plt.tight_layout()
        plt.savefig(savepath)
        plt.clf()
        for i in range(len(self.convergence_times)):
            plt.plot(self.convergence_times[i], self.convergence_DAs[i], alpha=0.5)
        plt.savefig(savepath[:-4] + "_convergence.png", dpi=300)


class MapPhaseSpace:
    r"""
    Phase-space mapping and digit-accuracy diagnostics for guiding-center
    orbits in a perturbed (ShearAlfvenWave) magnetic field.

    This class traces particles in a ShearAlfvenHarmonic or
    ShearAlfvenWavesSuperposition field and computes per-particle diagnostics
    including the Weighted Birkhoff Average (WBA) digit accuracy, wall-loss
    status, and the perturbed energy invariant
    E' = n' * E - omega * p_eta.

    Initial conditions are generated either on a structured grid in
    (s, \mu) space or drawn uniformly from the
    plasma volume. Particles lost in the equilibrium field before tracing in
    the perturbed field are removed.
    """

    def __init__(
        self,
        saw,
        Phin_max,
        Phim_max,
        omega,
        mass,
        charge,
        Ekin,
        helicity_N,
        helicity_M,
        helicity_Mp=None,
        helicity_Np=None,
        Eprime=None,
        sign_vpar=1,
        tmax=1e-2,
        plot_s=False,
        min_timestep=1e-6,
        ns_points=35,
        particles_per_surface=20,
        nlambda_points=35,
        randomize_particles=False,
        number_of_particles=10000,
        s_lims=None,
        mu_lims=None,
        comm=None,
        tol=1e-9,
        min_DA_time=None,
        solver_options=None,
        savedata=True,
        file_name="",
        convergence_points=1,
    ):
        r"""
        Initialize the MapPhaseSpace instance, generate initial conditions, and
        trace particles in the perturbed field.

        Args:
            saw                  : A :class:`ShearAlfvenHarmonic` or
                                :class:`ShearAlfvenWavesSuperposition` instance.
            Phin_max             : Toroidal mode number of the dominant wave harmonic,
                                used to compute nprime.
            Phim_max             : Poloidal mode number of the dominant wave harmonic,
                                used to compute nprime.
            omega                : Wave frequency.
            mass                 : Particle mass.
            charge               : Particle charge.
            Ekin                 : Total unperturbed kinetic energy.
            helicity_N           : Toroidal helicity of the field-strength contours.
            helicity_M           : Poloidal helicity of the field-strength contours.
            helicity_Mp          : Poloidal helicity of the mapping coordinate eta.
                                If None, determined automatically from helicity_M.
            helicity_Np          : Toroidal helicity of the mapping coordinate eta.
                                If None, determined automatically from helicity_N.
            Eprime               : Fixed value of the shifted energy invariant
                                n' * E - omega * p_eta. If provided, initial
                                parallel velocities are solved from this constraint
                                rather than from Ekin directly (default: None).
            sign_vpar            : Sign of the parallel velocity, either +1 or -1
                                (default: 1).
            tmax                 : Maximum integration time per particle
                                (default: 1e-2 s).
            plot_s               : If True, use the flux-surface label s as the radial
                                plot coordinate; otherwise use p_eta (default: False).
            min_timestep         : Minimum time-step size used as the save interval and
                                for computing the expected trajectory length
                                (default: 1e-6 s).
            ns_points            : Number of flux surfaces in the structured grid
                                (default: 35).
            particles_per_surface : Number of particles sampled on each flux surface
                                (default: 20).
            nlambda_points       : Number of magnetic-moment values in the structured
                                grid (default: 35).
            randomize_particles  : If True, sample initial conditions uniformly at
                                random rather than on a structured grid
                                (default: False).
            number_of_particles  : Number of particles to sample when
                                randomize_particles is True (default: 10000).
            s_lims               : Two-element list [s_min, s_max] bounding the
                                sampled flux surfaces (default: [0.01, 0.975]).
            mu_lims              : Two-element list [mu_min, mu_max] bounding the
                                sampled magnetic moments. If [None, None], limits
                                are set from Ekin and the minimum |B| in the volume
                                (default: [None, None]).
            comm                 : MPI communicator for parallel execution
                                (default: None).
            tol                  : Absolute and relative ODE solver tolerance
                                (default: 1e-10).
            solver_options       : Dictionary of additional options passed to the ODE
                                solver (default: {}).
            savedata             : If True, save initial conditions and final
                                diagnostics to disk (default: True).
            file_name            : Prefix for output file names (default: '').
            convergence_points   : Number of intermediate WBA evaluations per
                                trajectory used to assess convergence
                                (default: 1).
        """

        # set field parameters
        self.saw = saw
        self.B0 = saw.B0
        self.helicity_M = helicity_M
        self.helicity_N = helicity_N

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

        self.Phim = Phim_max
        self.Phin = Phin_max
        self.nprime = (self.Phim * helicity_N - self.Phin * helicity_M) / (
            helicity_Np * helicity_M - helicity_N * helicity_Mp
        )
        self.omega = omega
        self.omegan = self.omega / self.nprime

        self.tol = tol

        # set timing parameters
        self.tmax = tmax
        self.min_timestep = min_timestep

        self.Ekin = Ekin
        self.sign = sign_vpar
        self.mass = mass
        self.charge = charge

        if Eprime is None:
            self.Eprime_slice = False
        else:
            self.Eprime_slice = True
        self.Eprime = Eprime

        if s_lims is None:
            self.s_min = 0.01
            self.s_max = 0.975
        else:
            self.s_min = s_lims[0]
            self.s_max = s_lims[1]

        # set communicator parameters
        self.comm = comm
        self.verbose = False
        if self.comm is None or self.comm.rank == 0:
            self.verbose = True

        self.solver_options = solver_options if solver_options is not None else {}

        self.min_volmodB = min_volumemodB(self.B0)
        self.plot_s = plot_s

        if min_DA_time is None:
            min_DA_time = 0
        if min_DA_time > self.tmax:
            raise ValueError("min_DA_time must be less than or equal to tmax.")
        self.min_DA_time = min_DA_time

        # plotting settings
        self.savedata = savedata
        if file_name != "":
            file_name += "_"
        self.savepath = file_name
        self.convergence_points = convergence_points

        if mu_lims is None:
            self.mu_min = 0
            self.mu_max = Ekin / self.min_volmodB
        else:
            self.mu_min = mu_lims[0]
            self.mu_max = mu_lims[1]

        if self.savedata:
            self.final_filepaths = {
                "DA": self.savepath + "DA_losttimes.txt",
                "ICs": self.savepath + "initial_conditions.txt",
            }
            self.res_filepaths = {
                "tys": self.savepath + "DA_data.txt",
            }

        self.randomize = randomize_particles
        if randomize_particles:
            xy_pts = int(np.sqrt(number_of_particles / particles_per_surface))
            self.ns_points = xy_pts
            self.nlambda_points = xy_pts
        else:
            self.ns_points = ns_points
            self.nlambda_points = nlambda_points

        load_files = False
        if savedata and exists(self.final_filepaths["ICs"]):
            load_files = True

        if load_files:
            initial_conditions = np.loadtxt(self.final_filepaths["ICs"])
            self.s, self.thetas, self.zetas, self.vpar, self.mus_per_mass = (
                initial_conditions[:, 0],
                initial_conditions[:, 1],
                initial_conditions[:, 2],
                initial_conditions[:, 3],
                initial_conditions[:, 4],
            )
        else:
            self.particles_per_surface = particles_per_surface

            s, thetas, zetas, vpar, mu_per_mass = self.initialize_particles()

            initial_points = np.zeros((len(s), 3))  # initialize with t = 0
            initial_points[:, 0] = s
            initial_points[:, 1] = thetas
            initial_points[:, 2] = zetas
            self.s, self.thetas, self.zetas, self.vpar, self.mus_per_mass = (
                self.remove_equilibrium_lost_particles(
                    initial_points, vpar, mu_per_mass
                )
            )

        # set parameters for convergence plot
        expected_length = int(self.tmax / self.min_timestep)
        expected_step = int(expected_length / self.convergence_points)
        self.WBA_transit_indicies = np.linspace(
            expected_step, expected_length - 1, num=self.convergence_points, dtype=int
        ).tolist()
        self.convergence_plot = self.convergence_points > 1

        self.trace_particles()
        return

    def initialize_particles(self):
        r"""
        Generate initial particle positions, parallel velocities, and magnetic
        moments for tracing in the perturbed field.

        Initial flux surfaces and magnetic moments are sampled either on a
        structured (s, mu) grid or uniformly at random, depending on
        self.randomize. On each surface, particles are distributed uniformly
        using initialize_position_uniform_surf. Points with no valid vpar
        solution are dropped.

        Returns:
            s : List of initial s coordinates.
            thetas : List of initial theta coordinates.
            zetas : List of initial zeta coordinates.
            vpars : List of initial parallel velocities (signed by self.sign).
            mus_per_mass : List of initial magnetic moments divided by mass.
        """
        if self.randomize:
            mus = np.random.uniform(self.mu_min, self.mu_max, self.nlambda_points)

            s_linspace = np.random.uniform(self.s_min, self.s_max, self.ns_points)
        else:
            mus = np.linspace(self.mu_min, self.mu_max, self.nlambda_points)

            s_linspace = np.linspace(self.s_min, self.s_max, self.ns_points)

        surfaces, mus = np.meshgrid(s_linspace, mus)

        surfaces_flat = surfaces.flatten()
        mus_flat = mus.flatten()

        s = []
        thetas = []
        zetas = []
        vpars = []
        mus_per_mass = []
        for particle_index in range(len(surfaces_flat)):
            points_temp = initialize_position_uniform_surf(
                self.B0,
                self.particles_per_surface,
                surfaces_flat[particle_index],
                comm=self.comm,
            )

            mu = np.full(points_temp.shape[0], mus_flat[particle_index])
            mu_per_mass = mu / self.mass

            if self.Eprime_slice:
                vpar_temp = _solve_vpar_perturbed(
                    self.B0,
                    self.saw,
                    points_temp,
                    self.helicity_M,
                    self.helicity_N,
                    self.helicity_Np,
                    self.helicity_Mp,
                    self.mass,
                    self.nprime,
                    self.omega,
                    self.charge,
                    self.Eprime,
                    mu_per_mass,
                    self.sign,
                )
            else:
                vpar_temp = _solve_vpar_energy(
                    self.B0,
                    points_temp,
                    self.mass,
                    self.Ekin,
                    mu,
                    self.sign,
                )
            # remove unphysical points
            neg_idx = np.where(np.isnan(vpar_temp))[0]
            vpar_temp = np.delete(vpar_temp, neg_idx)
            points_temp = np.delete(points_temp, neg_idx, axis=0)
            mu = np.delete(mu, neg_idx, axis=0)
            mu_per_mass = np.delete(mu_per_mass, neg_idx, axis=0)

            s += list(points_temp[:, 0])
            thetas += list(points_temp[:, 1])
            zetas += list(points_temp[:, 2])
            vpars += vpar_temp.tolist()
            mus_per_mass += mu_per_mass.tolist()
        return (
            s,
            thetas,
            zetas,
            vpars,
            mus_per_mass,
        )

    def remove_equilibrium_lost_particles(self, points, vpars_init, mus_per_mass):
        r"""
        Trace particles briefly in the unperturbed equilibrium field and drop any
        that hit the s = 1 wall before tracing in the perturbed field.

        Args:
            points : Array of shape (N, 3) of initial (s, theta, zeta).
            vpars_init : Array of initial parallel velocities.
            mus_per_mass : Array of initial magnetic moments per mass.

        Returns:
            s, thetas, zetas, vpars, mus_per_mass : Filtered arrays with
                equilibrium-lost particles removed.
        """
        # trace particles in equilibrium field to see if any are lost
        gc_tys, gc_zeta_hits = trace_particles_boozer(
            field=self.B0,
            stz_inits=points,
            parallel_speeds=vpars_init,
            tmax=2e-3,
            mass=self.mass,
            charge=self.charge,
            Ekin=self.Ekin,
            comm=self.comm,
            forget_exact_path=True,
            dt_save=self.min_timestep,
            tol=self.tol,
            stopping_criteria=[MaxToroidalFluxStoppingCriterion(1.0)],
            mode="gc_noK",
            **self.solver_options,
        )

        lost_tolerance = 5 * self.min_timestep
        # assert len(gc_tys) == len(points)
        assert len(gc_tys) == len(points)

        # check if any particles were lost to the wall
        lost_total = []
        for i in range(len(gc_tys)):
            if gc_tys[i][-1, 0] < (2e-3 - lost_tolerance):  # noqa: SIM102
                lost_total.append(i)

        # remove wall lost particles from the list of evaluated particles
        points = np.delete(points, lost_total, axis=0)
        vpars_init = np.delete(vpars_init, lost_total, axis=0)
        mus_per_mass = np.delete(mus_per_mass, lost_total, axis=0)

        return points[:, 0], points[:, 1], points[:, 2], vpars_init, mus_per_mass

    def trace_particles(self):
        r"""
        Trace all initialized particles in the perturbed field and compute
        per-particle diagnostics.

        For each particle, integrates the guiding-center equations in the
        ShearAlfvenWave field, computes the canonical momentum p_eta, the total
        energy E, the shifted energy Eprime, and the WBA digit accuracy.
        Results are collected across MPI ranks, saved to
        disk if self.savedata is True, and passed to build_lists.
        """
        import pickle

        if self.savedata:  # noqa: SIM102
            if _check_filepaths(self.res_filepaths):  # noqa: SIM102
                if self.verbose:
                    proc0_print("Reading File")
                with open(self.res_filepaths["tys"], "rb") as f:
                    res_tys = pickle.load(f)
                self.build_lists(res_tys)
                return

        if self.verbose:
            proc0_print("Tracing particles in perturbed field...")

        first, last = parallel_loop_bounds(self.comm, len(self.s))

        res_tys = []

        for itrj in range(first, last):
            point = np.zeros((1, 3))  # initialize with t = 0
            point[:, 0] = self.s[itrj]
            point[:, 1] = self.thetas[itrj]
            point[:, 2] = self.zetas[itrj]

            vpar = [self.vpar[itrj]]
            mu_pm = [self.mus_per_mass[itrj]]

            gc_tys, gc_zeta_hits = trace_particles_boozer_perturbed(
                perturbed_field=self.saw,
                stz_inits=point,
                parallel_speeds=vpar,
                mus=mu_pm,
                tmax=self.tmax,
                mass=self.mass,
                charge=self.charge,
                Ekin=self.Ekin,
                abstol=1e-9,
                reltol=1e-9,
                stopping_criteria=[MaxToroidalFluxStoppingCriterion(1.0)],
                dt_save=self.min_timestep,
                mode="gc_noK",
                ODE_solver="dormand_prince",
                **self.solver_options,
            )

            points_trajectory = gc_tys[0]

            if points_trajectory.ndim != 2:
                continue

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

            # avoid tracing bug that sometimes causes last pt to be outside wall
            idx_wall = np.argmax(s_path >= 1) if np.any(s_path >= 1) else None
            if idx_wall is not None and s_path[idx_wall] >= 1:
                idx_wall -= 1
                points_trajectory = points_trajectory[:idx_wall, :]
                vpar_path = vpar_path[:idx_wall]
                time_momentum = time_momentum[:idx_wall]

            self.saw.set_points(points_trajectory)

            modB = self.saw.B0.modB()[:, 0]
            Phi = self.saw.Phi()[:, 0]

            weighted_mu = mu_pm[0] * self.mass

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

            # start state vector:  [s, theta, zeta, vpar, peta, E, mu, Eprime]
            # end state vector:
            # [t, s, theta, zeta, vpar, peta, E, mu, Eprime, DA]
            # mean state vector:  [s_mean, peta_mean, E_mean, Eprime_mean]
            start_state = [
                points_trajectory[0, 0],
                points_trajectory[0, 1],
                points_trajectory[0, 2],
                vpar[0],
                Peta_values[0],
                E[0],
                weighted_mu,
                Eprime[0],
            ]

            end_state = [
                points_trajectory[-1, -1],
                points_trajectory[-1, 0],
                points_trajectory[-1, 1],
                points_trajectory[-1, 2],
                vpar_path[-1],
                Peta_values[-1],
                E[-1],
                weighted_mu,
                Eprime[-1],
                final_DA,
            ]

            mean_state = [
                np.mean(points_trajectory[:, 0]),
                np.mean(Peta_values),
                np.mean(E),
                np.mean(Eprime),
            ]

            convergence_times = []
            convergence_petas = []
            convergence_energies = []
            convergence_DAs = []

            for _conv_index, timing_index in enumerate(self.WBA_transit_indicies):
                if timing_index >= len(time_momentum):
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

            particle_out = [start_state, end_state, mean_state, convergence_data]
            res_tys.append(particle_out)

        if self.comm is not None:
            proc0_print(f"{self.comm.rank=} done tracing particles")
            res_tys = [i for o in self.comm.allgather(res_tys) for i in o]

        if self.verbose and self.savedata:
            with open(self.res_filepaths["tys"], "wb") as f:
                pickle.dump(res_tys, f)

        self.build_lists(res_tys)
        return

    def build_lists(self, res_tys):
        r"""
        Process trajectory summaries into organized diagnostic
        lists stored as instance attributes.

        Populates self.DAs_at_loss, self.DA_at_tfinal, self.lost_total,
        self.final_times, self.pitch, self.Plot_Radial,
        self.Peta_init/mean/final, self.E_init/mean/final,
        and the convergence_* arrays.

        Args:
            res_tys : List of per-particle summaries, each of the form
                    [start_state, end_state, mean_state, convergence_data].
        """

        DAs_at_loss = []
        DA_tfinal = []

        lost_total = []
        final_times = []
        pitch = []

        Plot_Radial = []

        Peta_init = []
        Peta_mean = []
        Peta_final = []
        E_mean = []
        E_final = []
        E_init = []

        s0 = []
        theta0 = []
        zeta0 = []
        vpar0 = []
        mus0 = []

        convergence_times = []
        convergence_petas = []
        convergence_energies = []
        convergence_DAs = []

        tolerance = 5 * self.min_timestep

        for elem in res_tys:
            # start state vector:
            #   [s, theta, zeta, vpar, peta, E, mu, Eprime]
            # end state vector:
            #   [t, s, theta, zeta, vpar, peta, E, mu, Eprime, DA]
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

            pitch_val = float(start_state[6]) / self.Ekin
            pitch.append(pitch_val)

            if self.plot_s:
                Plot_Radial.append(start_state[0])
            else:
                Plot_Radial.append(start_state[4])

            Peta_init.append(start_state[4])
            Peta_mean.append(means[1])
            Peta_final.append(end_state[5])
            E_mean.append(means[2])
            E_final.append(end_state[6])
            E_init.append(start_state[5])

            if final_time < (self.tmax - tolerance):
                lost_total.append(1)
                DA_tfinal.append(np.nan)
            else:
                lost_total.append(0)
                DA_tfinal.append(end_state[9])

            if final_time < self.min_DA_time:
                DAs_at_loss.append(np.nan)
            else:
                DAs_at_loss.append(end_state[9])

            s0.append(start_state[0])
            theta0.append(start_state[1])
            zeta0.append(start_state[2])
            vpar0.append(start_state[3])
            mus0.append(start_state[6])

            convergence_times.append(convergence[0])
            convergence_petas.append(convergence[1])
            convergence_energies.append(convergence[3])
            convergence_DAs.append(convergence[2])

        self.DAs_at_loss = DAs_at_loss
        self.DA_at_tfinal = DA_tfinal
        self.lost_total = lost_total
        self.final_times = final_times

        self.pitch = pitch
        self.Plot_Radial = Plot_Radial

        self.Peta_init = Peta_init
        self.Peta_mean = Peta_mean
        self.Peta_final = Peta_final
        self.E_mean = E_mean
        self.E_final = E_final
        self.E_init = E_init

        self.convergence_times = convergence_times
        self.convergence_petas = convergence_petas
        self.convergence_DAs = convergence_DAs
        self.convergence_energies = convergence_energies

        if self.verbose:
            mu_per_mass0 = np.array(mus0) / self.mass
            if self.savedata:
                if not exists(self.final_filepaths["ICs"]):
                    np.savetxt(
                        self.final_filepaths["ICs"],
                        np.column_stack((s0, theta0, zeta0, vpar0, mu_per_mass0)),
                    )
                if not exists(self.final_filepaths["DA"]):
                    np.savetxt(
                        self.final_filepaths["DA"],
                        np.column_stack((DAs_at_loss, final_times)),
                    )
        return

    def surface_trapped_func_Eprime(self, mu, surface):
        r"""
        `    Determine whether a particle with the given pitch angle is trapped on a
            specified flux surface.

            Samples modB over the surface and checks whether the parallel velocity
            calculated via the Eprime rootsolve would become imaginary at the maximum
            field strength on that surface, indicating trapping.
            The prescribed reference Ekin is used.

            Args:
                mu : Magnetic moment.
                surface     : Flux-surface label s at which to evaluate trapping.

            Returns:
                trapped : List of integers (0 or 1) indicating whether each sampled
                        point is trapped.
                peta   : List of radial like coordinates corresponding to each sampled
                        point.
        """
        resolution = 500
        points = initialize_position_uniform_surf(self.B0, resolution, surface)
        points_phase = np.column_stack(
            (points, np.zeros(points.shape[0]))
        )  # add time column
        self.saw.set_points(points_phase)
        modB = self.saw.B0.modB()[:, 0]
        Phi = self.saw.Phi()[:, 0]
        max_modB = np.max(modB)

        sign_arrs = np.ones_like(modB) * self.sign

        mu_pm = mu / self.mass

        vp_temp = _solve_vpar_perturbed(
            self.B0,
            self.saw,
            points,
            self.helicity_M,
            self.helicity_N,
            self.helicity_Np,
            self.helicity_Mp,
            self.mass,
            self.nprime,
            self.omega,
            self.charge,
            self.Eprime,
            mu_pm,
            sign_arrs,
        )

        mask = ~np.isnan(vp_temp)
        points = points[mask]
        points_phase = points_phase[mask]
        modB = modB[mask]
        Phi = Phi[mask]
        vp_temp = vp_temp[mask]

        E = 0.5 * self.mass * vp_temp**2 + mu * modB + self.charge * Phi

        E_potential = mu * max_modB + self.charge * Phi

        output = ((E - E_potential) < 0).astype(int)
        output = output.tolist()
        if self.plot_s:
            return np.sum(output), surface
        else:
            if points.shape[0] == 0:
                return [], []

            peta = compute_peta(
                self.B0,
                points,
                vp_temp,
                self.mass,
                self.charge,
                self.helicity_M,
                self.helicity_N,
                self.helicity_Mp,
                self.helicity_Np,
            )

            return output, peta.tolist()

    def return_peta_trapped_contoured_boundary(self, negate_peta=False):
        r"""
        Estimate the trapped-passing boundary in the (pitch, p_eta) plane by
        sampling many points on flux surfaces, binning the trapped/passing
        indicator, and tracing the column-wise transition. The boundary is then
        fit with a quadratic polynomial.

        Args:
            negate_peta : If True, flip the sign of the y axis.

        Returns:
            poly : numpy.poly1d quadratic fit of the boundary.
            pitch_fit : Pitch coordinates of the fitted curve.
            radlike_fit : Radial-like (s or p_eta) values of the fitted curve.
        """
        volume_boundary_radlike = []
        volume_boundary_pitch = []
        volume_trapped = []
        radial_space = 100
        pa_space = 100

        s_vals = np.linspace(self.s_min, self.s_max, radial_space)
        mu_vals = np.linspace(self.mu_min, self.mu_max, pa_space)

        for s_val in s_vals:
            for mu_val in mu_vals:
                trapped, radial_like = self.surface_trapped_func_Eprime(mu_val, s_val)

                pitch_val = (mu_val / self.Ekin) * self.min_volmodB
                pitch_val *= self.sign

                if self.plot_s:
                    volume_boundary_radlike.append(s_val)
                    volume_trapped.append(trapped)
                    volume_boundary_pitch.append(pitch_val)
                    continue

                volume_boundary_radlike += radial_like
                volume_trapped += trapped
                pitch_lst = [pitch_val] * len(radial_like)
                volume_boundary_pitch += pitch_lst

        volume_boundary_pitch = np.array(volume_boundary_pitch)
        volume_boundary_radlike = np.array(volume_boundary_radlike)

        if negate_peta:
            volume_boundary_radlike = -volume_boundary_radlike
        volume_trapped = np.array(volume_trapped)

        trapped_vals, pitch_edges, radlike_edges, binnumber = binned_statistic_2d(
            volume_boundary_pitch,
            volume_boundary_radlike,
            volume_trapped,
            statistic="max",
            bins=[int(pa_space * 0.80), int(radial_space * 0.80)],
        )

        T = np.nan_to_num(trapped_vals, nan=0.0).T
        # x, y -> (pitch, peta) dimensions

        boundary_pitch, boundary_radlike = [], []

        for peta_i in range(0, T.shape[0]):
            peta_data = T[peta_i, :]
            if not peta_data.any():
                continue
            pitch_i = int(np.argmax(peta_data == 1))
            boundary_pitch.append(pitch_edges[pitch_i])
            boundary_radlike.append(radlike_edges[peta_i])
            continue

        if len(boundary_pitch) == 0:
            return None, None, None
        boundary_pitch = np.array(boundary_pitch)
        boundary_radlike = np.array(boundary_radlike)

        order = np.argsort(boundary_pitch)
        boundary_pitch = boundary_pitch[order]
        boundary_radlike = boundary_radlike[order]

        # prevents polyfit from recieving a line in constant pitch
        # and making an ill conditioned linear fit
        if (boundary_pitch.max() - boundary_pitch.min()) < 0.001:
            real_data = np.array(self.Plot_Radial)
            if negate_peta:
                real_data = -real_data
            # enforce that the trapped fit is larger than the smallest
            # sampled particle in peta and smaller than the
            # largest simulated particle in peta
            condition = (boundary_radlike > real_data.min()) & (
                boundary_radlike < real_data.max()
            )
            boundary_radlike = boundary_radlike[condition]
            boundary_pitch = boundary_pitch[condition]
            return None, boundary_pitch, boundary_radlike
        # make a linear fit
        coeffs = np.polyfit(boundary_pitch, boundary_radlike, 1)
        poly = np.poly1d(coeffs)

        pitch_fit = np.linspace(boundary_pitch.min(), boundary_pitch.max(), 300)
        radlike_fit = poly(pitch_fit)

        return poly, pitch_fit, radlike_fit

    def plot_heatmap(
        self,
        nx=None,
        ny=None,
        savepath="heatmap_digit_accuracy.pdf",
        ax=None,
        DA_max=7,
        statistic="mean",
        DA_at_loss=True,
        plot_losses=False,
        negate_peta=False,
        lost_fraction=False,
    ):
        r"""
        Plot a 2D heatmap of digit accuracy in the (pitch, radial-like) plane and
        overlay the fitted trapped-passing boundary. Optionally overlay loss
        fractions as triangle markers per bin.

        Args:
            nx : Number of bins along the pitch axis. If None, defaults to
                nlambda_points for a structured grid of initial conditions, or
                the cube root of the number of particles if they were
                randomized (default: None).
            ny : Number of bins along the radial axis. If None, defaults to
                ns_points for a structured grid of initial conditions, or the
                cube root of the number of particles if they were randomized
                (default: None).
            savepath : Output file path for the heatmap.
            ax : Matplotlib axis. If None, a new figure and axis are created.
            DA_max : Maximum DA value shown on the colorbar. If None, the
                maximum DA in the data is used instead (default: 7).
            statistic : Aggregation statistic passed to binned_statistic_2d.
            DA_at_loss : If True, use the DA value at loss; otherwise the final
                integration DA.
            plot_losses : If True, overlay loss-fraction markers per bin.
            negate_peta : If True, flip the sign of the y axis (useful when
                plotting against -p_eta).

        Returns:
            ax : The Matplotlib axis containing the plot.
        """
        import matplotlib as mpl
        import matplotlib.pyplot as plt

        if self.verbose:
            proc0_print("Plotting...")

        if ax is None:
            fig, ax = plt.subplots(figsize=(16, 12))
        else:
            fig = ax.get_figure()

        try:
            import cmcrameri.cm as cmc  # noqa: F401

            cmap = "cmc.managua"

        except ImportError:
            cmap = "viridis"

        nx, ny = _resolve_heatmap_bins(
            nx,
            ny,
            self.nlambda_points,
            self.ns_points,
            len(self.pitch),
            self.randomize,
        )

        DA_values = self.DAs_at_loss if DA_at_loss else self.DA_at_tfinal

        if DA_max is None:
            DA_max = np.nanmax(np.array(DA_values))

        norm = mpl.colors.Normalize(vmin=0, vmax=DA_max)

        plotting_pitch_normalized = np.array(self.pitch) * self.min_volmodB
        plotting_pitch_normalized *= self.sign

        DA_stats, x_edges, y_edges, binnumber = binned_statistic_2d(
            plotting_pitch_normalized,
            np.array(self.Plot_Radial),
            np.array(DA_values),
            statistic=statistic,
            bins=[nx, ny],
        )

        X, Y = np.meshgrid(x_edges, y_edges)
        if negate_peta:
            Y *= -1
        im2 = ax.pcolormesh(X, Y, DA_stats.T, shading="auto", cmap=cmap, norm=norm)

        poly, pa_fit, rad_fit = self.return_peta_trapped_contoured_boundary(
            negate_peta=negate_peta
        )
        self.trapped_boundary_fit = poly
        self.trapped_boundary_fit_pitch = pa_fit
        self.trapped_boundary_fit_radial = rad_fit

        if pa_fit is not None and rad_fit is not None:
            ax.plot(pa_fit, rad_fit, color="gray", linewidth=10)
        else:
            proc0_print("Fitting Trapped Passing Boundary Failed")

        colorlabel = "Digit Accuracy"

        if plot_losses:
            from matplotlib.cm import ScalarMappable
            from matplotlib.colors import Normalize

            lost_stat = "mean" if lost_fraction else "max"
            lost_frac, x_edges, y_edges, _ = binned_statistic_2d(
                plotting_pitch_normalized,
                np.array(self.Plot_Radial),
                np.array(self.lost_total),
                statistic=lost_stat,
                bins=[nx, ny],
            )
            x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
            y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
            Xc, Yc = np.meshgrid(x_centers, y_centers)
            xf = Xc.ravel()
            yf = Yc.ravel()
            lost_frac = np.nan_to_num(lost_frac, nan=0.0)
            af = lost_frac.T.ravel()
            if negate_peta:
                yf = yf * -1
            sm = ScalarMappable(cmap="Reds", norm=Normalize(vmin=0, vmax=1))
            sm.set_array([])  # avoids warnings on older matplotlib
            ax.scatter(
                xf,
                yf,
                marker="s",
                s=100,
                c="red",
                alpha=af,
                zorder=10,
            )
            if lost_fraction:
                fig.colorbar(sm, ax=ax, label="Particle Loss Fraction")

        ax.set_xlabel(r"$\lambda = \frac{\mu}{E} \text{sign}(v_{\|})$")

        if self.plot_s:
            ax.set_ylabel(r"$s$")
        else:
            if negate_peta:
                ax.set_ylabel(r"$-P_\eta$")
            else:
                ax.set_ylabel(r"$P_\eta$")

        fig.tight_layout()
        fig.colorbar(im2, ax=ax, label=colorlabel)
        plt.savefig(savepath, dpi=400)
        return ax
