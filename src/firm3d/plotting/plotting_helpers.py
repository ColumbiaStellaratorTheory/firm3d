import numpy as np

from ..util.mpi import verbose

__all__ = [
    "plot_trajectory_poloidal",
    "plot_trajectory_overhead_cyl",
    "plot_resonance_lines",
]


def plot_trajectory_poloidal(res_ty, helicity_M=1, helicity_N=0, ax=None):
    r"""
    Given the trajectory of a single particle in Boozer coordinates, plot
    in the pseudo-poloidal plane (x,y) where x = sqrt(s) cos(chi)
    and y = sqrt(s) sin(chi), where chi = helicity_M * theta - helicity_N * zeta.
    If helicity_M and helicity_N correspond with the helicity of the field-strength
    contours, then this plot will visualize the "banana" trajectories of trapped
    particles. Circles in this plane correspond with flux surfaces. The dashed circle
    is the initial flux surface, and the solid circle is the plasma boundary.

    Args:
        res_ty : A 2D numpy array of shape (nsteps, 5) containing the
                 trajectory of a single particle in Boozer coordinates
                 (s, theta, zeta, vpar). Tracing should be performed with
                 forget_exact_path=False to save the trajectory information.
        helicity_M : Helicity M value for the transformation (default: 1)
        helicity_N : Helicity N value for the transformation (default: 0)
        ax : Optional matplotlib Axes object to plot on. If None, a new
             figure and axes will be created.

    Returns:
        ax : The matplotlib Axes object containing the plot.
    """
    s = res_ty[:, 1]
    theta = res_ty[:, 2]
    zeta = res_ty[:, 3]
    chi = helicity_M * theta - helicity_N * zeta
    x = np.sqrt(s) * np.cos(chi)
    y = np.sqrt(s) * np.sin(chi)

    chi_grid = np.linspace(0, 2 * np.pi, 100)
    s0 = s[0]
    x0 = np.sqrt(s0) * np.cos(chi_grid)
    y0 = np.sqrt(s0) * np.sin(chi_grid)

    x1 = np.cos(chi_grid)
    y1 = np.sin(chi_grid)

    if verbose:
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots()

        ax.plot(x, y)
        ax.plot(x0, y0, color="black", linestyle="--")
        ax.plot(x1, y1, color="black", linestyle="-")
        ax.set_xlabel(r"$\sqrt{s} \cos(\chi)$")
        ax.set_ylabel(r"$\sqrt{s} \sin(\chi)$")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
    else:
        ax = None

    return ax


def plot_resonance_lines(
    ax,
    harmonics,
    mode_numbers,
    trapped_boundary_fit,
    trapped_boundary_fit_pitch,
    harmonic_cmap,
    norm,
    possible_linestyles,
    max_ell,
    min_crossing_points=2,
    poly_degree=2,
    n_fit_points=100,
):
    r"""
    Fit and plot resonance lines accumulated by
    :func:`firm3d.trajectory_helpers.accumulate_resonance_crossings` onto a
    phase-space heatmap axis, as a function of pitch angle.

    For each (harmonic, ell) pair, resonance crossing points are fit with a
    polynomial in pitch angle and drawn up to the first point where the fit
    crosses the trapped-passing boundary (beyond which the resonance line
    would lie in the trapped region).

    Args:
        ax : Matplotlib axis to plot the resonance lines on (e.g. the
             heatmap axis).
        harmonics : Dict of the form {h: {ell: [[pitch_angles, radii], ...]}}
                    as populated by accumulate_resonance_crossings.
        mode_numbers : Dict {h: (Phim_h, Phin_h)} of poloidal/toroidal mode
                       numbers, used for legend labels.
        trapped_boundary_fit : Callable mapping an array of pitch angles to
                                the fitted trapped-passing boundary radial
                                location at each pitch angle (e.g.
                                heat_map.trapped_boundary_fit).
        trapped_boundary_fit_pitch : Array of pitch-angle values over which
                                     trapped_boundary_fit was originally fit
                                     (e.g. heat_map.trapped_boundary_fit_pitch).
                                     Used to exclude resonance lines starting
                                     outside the fit's valid domain.
        harmonic_cmap : Colormap used to color lines by harmonic index.
        norm : Normalization instance mapping harmonic index h to [0, 1] for
               harmonic_cmap.
        possible_linestyles : List of line styles indexed by ell + max_ell.
        max_ell : Maximum |ell| present in harmonics.
        min_crossing_points : Minimum number of (pitch, radius) points needed
                              to fit a resonance line (default: 2).
        poly_degree : Degree of the polynomial fit (default: 2).
        n_fit_points : Number of points to evaluate the fit at (default: 100).

    Returns:
        labels_lines : List of Line2D handles, one per (h, ell) pair, for the
                       first crossing line of each.
        labels_text : List of label strings corresponding to labels_lines.
    """
    labels_lines = []
    labels_text = []
    for h in harmonics:
        Phim_h, Phin_h = mode_numbers[h]

        for ell in harmonics[h]:
            for crossing_line_index, crossing_line in enumerate(harmonics[h][ell]):
                resonance_peta = np.asarray(crossing_line[1])
                resonance_pitch = np.asarray(crossing_line[0])

                if len(crossing_line[1]) < min_crossing_points:
                    continue  # skip if not enough points to fit a curve

                # smooth resonance lines with a polynomial fit
                coeffs = np.polyfit(resonance_pitch, resonance_peta, poly_degree)
                poly = np.poly1d(coeffs)
                pa_fit = np.linspace(
                    min(resonance_pitch), max(resonance_pitch), n_fit_points
                )
                s_fit = poly(pa_fit)

                color = harmonic_cmap(norm(h))

                # fit a curve to the resonance points to plot on the heatmap
                trapped_passing_fit = trapped_boundary_fit(pa_fit)

                # ignore resonance lines which start near the trapped-passing
                # boundary, in the region where the fit is inaccurate due to
                # numerical noise
                if np.abs(trapped_boundary_fit_pitch[0]) < np.abs(resonance_pitch[0]):
                    continue

                diff = trapped_passing_fit - s_fit
                sign_changes = np.where(np.sign(diff[:-1]) != np.sign(diff[1:]))[0]

                # don't repeat label if resonance line crosses multiple times
                if crossing_line_index == 0:
                    label = rf"m,n={Phim_h},{Phin_h} $\ell$={ell}"
                else:
                    label = None

                stop_index = len(pa_fit) if len(sign_changes) == 0 else sign_changes[0]

                if stop_index == 0:
                    continue  # skip if resonance line is entirely in trapped region
                (line,) = ax.plot(
                    pa_fit[:stop_index],
                    s_fit[:stop_index],
                    linewidth=5,
                    linestyle=possible_linestyles[ell + max_ell],
                    label=label,
                    color=color,
                )

                if crossing_line_index == 0:
                    labels_lines.append(line)
                    labels_text.append(label)

    return labels_lines, labels_text


def plot_trajectory_overhead_cyl(res_ty, field, ax=None):
    r"""
    Given the trajectory of a single particle in Boozer coordinates, plot
    in the overhead cylindrical plane (X,Y) where X = R cos(phi) and Y = R sin(phi).
    This plot visualizes an overhead view of the particle trajectory in
    cylindrical coordinates.
    The dashed circle corresponds to the magnetic axis.

    Args:
        res_ty : A 2D numpy array of shape (nsteps, 5) containing the
                 trajectory of a single particle in Boozer coordinates
                 (s, theta, zeta, vpar).
        field : The :class:`BoozerMagneticField` instance used to set
                the points for the field.
        ax : Optional matplotlib Axes object to plot on. If None, a new
             figure and axes will be created.

    Returns:
        ax : The matplotlib Axes object containing the plot.
    """
    from ..trajectory_helpers import compute_trajectory_cylindrical

    R, phi, Z = compute_trajectory_cylindrical(res_ty, field)

    X = R * np.cos(phi)
    Y = R * np.sin(phi)

    zetas_grid = np.linspace(0, 2 * np.pi, 100)
    points = np.zeros((len(zetas_grid), 3))
    points[:, 2] = zetas_grid
    field.set_points(points)
    R_axis = field.R()[:, 0]
    nu = field.nu()[:, 0]
    phi_axis = zetas_grid - nu

    X_axis = R_axis * np.cos(phi_axis)
    Y_axis = R_axis * np.sin(phi_axis)

    if verbose:
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(X, Y)
        ax.plot(X_axis, Y_axis, linestyle="--", color="black")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
    else:
        ax = None

    return ax
