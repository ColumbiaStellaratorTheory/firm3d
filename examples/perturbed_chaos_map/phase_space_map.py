import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt

from firm3d.field.boozermagneticfield import (
    BoozerRadialInterpolant,
    InterpolatedBoozerField,
    ShearAlfvenHarmonic,
)
from firm3d.plotting.plotting_helpers import plot_resonance_lines
from firm3d.trajectory_helpers import (
    MapPhaseSpace,
    PassingPerturbedPoincare,
    accumulate_resonance_crossings,
    calculate_crossings,
    calculate_QS_resonance,
    compute_reference_Eprime,
    compute_rotational_profile,
    min_volumemodB,
)
from firm3d.saw.ae3d import AE3DEigenvector
from firm3d.util.constants import (
    ALPHA_PARTICLE_CHARGE,
    ALPHA_PARTICLE_MASS,
    FUSION_ALPHA_PARTICLE_ENERGY,
)
from firm3d.util.functions import proc0_print
from firm3d.util.mpi import comm_world, verbose

# harmonic to isolate for this case
harmonic = 1
sign_vpar = 1
harmonics = {harmonic: {}}
max_ell = 2
# list of possible line styles for the resonance lines, indexed by ell value
# (shifted by max_ell to avoid negative indices)
#  ell =                 -2           -1      0           1           2
possible_linestyles = [(0, (1, 5)), "dotted", "solid", "dashed", (0, (5, 5))]

helicity_M = 1
helicity_N = -4
helicity_Mp = 0
helicity_Np = -1

boozmn_filename = "../inputs/boozmn_beta2.5_QH.nc"
perfect = True  # enforce perfect QS?
AE_filename = "QH_10harmonics_scale0_00464159.npy"
plot_losses = False

# poincare parameters
nchi_poinc = 5
ns_poinc = 500
Nmaps = 1500

# Eprime parameters
lam = 0.0
p0_int = 0.1

order = 3
degree = 3
resolution = 45
# resolution for perfect QS enforced if needed
res_p = 40

mpl.rcParams["font.size"] = 25  # base font size
mpl.rcParams["axes.labelsize"] = 25  # x/y labels
mpl.rcParams["axes.titlesize"] = 25
mpl.rcParams["xtick.labelsize"] = 25
mpl.rcParams["ytick.labelsize"] = 25
mpl.rcParams["legend.fontsize"] = 25
mpl.rcParams["figure.titlesize"] = 25

ns_interp = resolution  # number of radial grid points for interpolation
ntheta_interp = resolution  # number of poloidal grid points for interpolation
nzeta_interp = resolution  # number of toroidal grid points for interpolation


dir = "coprop" if sign_vpar else "counter"

if perfect:
    bri = BoozerRadialInterpolant(
        boozmn_filename,
        order,
        no_K=True,
        helicity_M=helicity_M,
        helicity_N=helicity_N,
        comm=comm_world,
    )
else:
    bri = BoozerRadialInterpolant(boozmn_filename, order, no_K=True, comm=comm_world)
field = InterpolatedBoozerField(
    bri,
    degree,
    ns_interp=ns_interp,
    ntheta_interp=ntheta_interp,
    nzeta_interp=nzeta_interp,
)

if perfect:
    bri_p, field_p = bri, field
else:
    bri_p = BoozerRadialInterpolant(
        boozmn_filename,
        order,
        no_K=True,
        helicity_M=helicity_M,
        helicity_N=helicity_N,
        comm=comm_world,
    )
    field_p = InterpolatedBoozerField(
        bri_p,
        degree,
        ns_interp=res_p,
        ntheta_interp=res_p,
        nzeta_interp=res_p,
    )


AE_temp = AE3DEigenvector.load_from_numpy(AE_filename)
omega = np.sqrt(AE_temp.eigenvalue) * 1000
Phihat = (AE_temp.s_coords, AE_temp.harmonics[harmonic].amplitudes)
Phin = AE_temp.harmonics[harmonic].n
Phim = AE_temp.harmonics[harmonic].m
saw = ShearAlfvenHarmonic(Phihat, Phim=Phim, Phin=Phin, omega=omega, B0=field, phase=0)

filepath = f"qh_{Phim}_{Phin}_{dir}" if perfect else f"bqh_{Phim}_{Phin}_{dir}"

mass = ALPHA_PARTICLE_MASS
charge = ALPHA_PARTICLE_CHARGE
Ekin = FUSION_ALPHA_PARTICLE_ENERGY

# Calculate Eprime for the given parameters
p0 = np.zeros((1, 3))
p0[0, 0] = p0_int  # s
Eprime = compute_reference_Eprime(
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
)

min_volmodB = min_volumemodB(field)

heat_map = MapPhaseSpace(
    saw,
    Phin,
    Phim,
    omega,
    mass,
    charge,
    Ekin,
    helicity_N,
    helicity_M,
    helicity_Mp,
    helicity_Np,
    Eprime=Eprime,
    sign_vpar=sign_vpar,
    tmax=1e-2,
    comm=comm_world,
    savedata=True,
    file_name=filepath,
    convergence_points=5,
)
proc0_print("Finished computing heatmap ")

map = PassingPerturbedPoincare(
    saw,
    sign_vpar,
    mass,
    charge,
    helicity_M,
    helicity_N,
    helicity_Mp=helicity_Mp,
    helicity_Np=helicity_Np,
    Ekin=Ekin,
    p0=p0,
    lam=lam,
    ns_poinc=ns_poinc,
    nchi_poinc=nchi_poinc,
    Nmaps=Nmaps,
    chaos_detection=True,
    comm=comm_world,
    nconvergence_points=5,
)
proc0_print("Finished computing poincare ")

# mode numbers of each harmonic we're searching for resonances with
mode_numbers = {h: (AE_temp.harmonics[h].m, AE_temp.harmonics[h].n) for h in harmonics}

max_mu = Ekin / min_volmodB
# iterate through pitch angles
mu_harmonics = np.linspace(0, max_mu, 50)

plt.clf()
plt.xlabel("s", fontsize=12)
plt.ylabel("h", fontsize=12)
mpl.rcParams["xtick.labelsize"] = 12
mpl.rcParams["ytick.labelsize"] = 12

for plot_counter, mu_h in enumerate(mu_harmonics):
    # compute rotational profile for given pitch angle
    profile = compute_rotational_profile(
        field_p,
        mu_h / Ekin,
        sign_vpar,
        mass,
        charge,
        Ekin,
        helicity_M,
        helicity_N,
        helicity_Mp,
        helicity_Np,
        comm_world,
        ns_poinc=100,
        Nmaps=75,
        tmax=1e-2,
    )
    if profile.shape[0] < 2:
        continue  # skip if not enough points to compute resonance:
    pitch_angle_h = (sign_vpar * np.abs(mu_h) / Ekin) * min_volmodB
    drift_helicity = profile[:, 3]
    radial_position = profile[:, 0]

    if verbose and (
        plot_counter % 5 == 0
    ):  # only plot some pitch angles to avoid overcrowding
        plt.plot(
            radial_position,
            drift_helicity,
            label=f"$\\lambda$={pitch_angle_h:.2f}",
        )

    # creates a dictionary of the form
    #  {harmonic: {ell: [[resonance_peta], [resonance_radius]]}}
    # to store the resonance locations for each harmonic and ell value
    accumulate_resonance_crossings(
        harmonics,
        profile,
        pitch_angle_h,
        mode_numbers,
        helicity_M,
        helicity_N,
        omega,
        max_ell,
    )


plt.clf()

mpl.rcParams["xtick.labelsize"] = 25
mpl.rcParams["ytick.labelsize"] = 25

# compute rotational profile for single pitch angle for poincare plot
ell_list = []
rad_list = []
lines_modes = []
profile = compute_rotational_profile(
    field_p,
    lam,
    sign_vpar,
    mass,
    charge,
    Ekin,
    helicity_M,
    helicity_N,
    helicity_Mp,
    helicity_Np,
    comm_world,
    ns_poinc=100,
    Nmaps=75,
    s_profile=True,
    tmax=1e-2,
)
for ell in range(-max_ell, max_ell + 1):
    h_res = calculate_QS_resonance(
        Phim, Phin, helicity_M, helicity_N, omega, np.mean(profile[:, 2]), ell=ell
    )
    crossings = calculate_crossings(profile[:, 3], h_res, profile[:, 0])
    for radius in crossings:
        ell_list.append(ell)
        rad_list.append(radius)
        lines_modes.append(harmonic)

# list of harmonics that we calculated resonances for
hlist = list(harmonics.keys())

# create a colormap for the lines based on the harmonic number
harmonic_cmap = mpl.cm.cividis
harmonic_cmap = harmonic_cmap.reversed()
if len(hlist) > 1:
    norm = mpl.colors.Normalize(vmin=hlist[0], vmax=hlist[-1])
else:
    norm = mpl.colors.Normalize(vmin=hlist[0], vmax=hlist[0] + 1)

if comm_world is not None:
    comm_world.Barrier()

if verbose:
    # make a list of line colors for each harmonic
    linecolors = [harmonic_cmap(norm(h)) for h in lines_modes]

    line_kwargs_nolabel = []
    line_kwargs_labeled = []
    for i in range(len(ell_list)):
        # find the line styles for each resonance line based on the ell value
        ls = possible_linestyles[ell_list[i] + max_ell]

        line_kwargs_nolabel.append(
            {
                "color": linecolors[i],
                "linestyle": ls,
                "linewidth": 5,
            }
        )
        # avoid repeating labels for multiple resonance lines
        # with the same ell value
        if i > 0:  # noqa: SIM102
            if ell_list[i] == ell_list[i - 1]:  # noqa: SIM102
                line_kwargs_labeled.append(
                    {"color": linecolors[i], "linestyle": ls, "linewidth": 5}
                )
                continue
        line_kwargs_labeled.append(
            {
                "color": linecolors[i],
                "linestyle": ls,
                "linewidth": 5,
                "label": rf"$\ell$={ell_list[i]}",
            }
        )

    # create subplots
    fig, (ax_left, ax_center, ax_right, ax_dummy) = plt.subplots(
        1, 4, gridspec_kw={"width_ratios": [1, 4, 4, 0.61]}, figsize=(34, 12)
    )

    # plot the perturbation amplitude as a function of radius on the left
    ax_left.plot(Phihat[1][:-1], Phihat[0])
    filename = filepath + "poincare.png"

    # plot the resonance lines on the pertubation magnitude plot
    for i, radius in enumerate(rad_list):
        ell = ell_list[i]
        proc0_print(f"{i}: ell={ell}, arr={radius}")
        ax_left.plot(
            [min(Phihat[1]), max(Phihat[1])],
            [radius, radius],
            **line_kwargs_nolabel[i],
        )

    ax_left.set_xlabel(r"$\delta \phi$ [V]")
    ax_left.set_ylabel(r"$s_0$")

    # plot the poincare plot in the center, with resonance lines
    ax_center, lines_colors = map.plot_poincare(
        ax=ax_center,
        filename=filename,
        resonance_lines=rad_list,
        plot_legend=False,
        line_plotting_kwargs=line_kwargs_labeled,
        bg_field=field_p,
        DA_colorbar=False,
        s_axis_label=False,
    )
    ax_center.set_ylim(0, 1)
    ax_left.set_ylim(0, 1)
    legend_handles, legend_labels = ax_center.get_legend_handles_labels()
    if ax_center.get_legend() is not None:
        ax_center.get_legend().remove()
    ax_center.set_title(rf"$\lambda$ = {lam * sign_vpar}")
    ax_center.sharey(ax_left)

    proc0_print("Finished plotting poincare ")

    # plot heatmap
    ax_right = heat_map.plot_heatmap(
        ax=ax_right, savepath=filepath + "heatmap.png", plot_losses=plot_losses
    )

    fig = ax_right.get_figure()

    # fit and plot resonance lines on top of the heatmap
    labels_lines, labels_text = plot_resonance_lines(
        ax_right,
        harmonics,
        mode_numbers,
        heat_map.trapped_boundary_fit,
        heat_map.trapped_boundary_fit_pitch,
        harmonic_cmap,
        norm,
        possible_linestyles,
        max_ell,
        min_crossing_points=2,
        poly_degree=2,
        n_fit_points=100,
    )
    ax_dummy.remove()
    ax_right.legend(
        labels_lines,
        labels_text,
        loc="upper left",
        bbox_to_anchor=(1.2, 1.0),
        borderaxespad=0.0,
    )
    fig.savefig(filepath + f"{filepath[:-1]}.png", dpi=400)

    plt.clf()

if comm_world is not None:
    comm_world.Barrier()
