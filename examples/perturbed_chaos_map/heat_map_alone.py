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
    accumulate_resonance_crossings,
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

from firm3d.util.functions import in_github_actions
from firm3d.util.mpi import comm_world, verbose

# harmonic to isolate for this case
harmonic = 1
sign_vpar = 1
harmonics = {harmonic: {}}
max_ell = 1
# list of possible line styles for the resonance lines, indexed by ell value
# (shifted by max_ell to avoid negative indices)
#  ell =                 -1      0           1
possible_linestyles = ["dotted", "solid", "dashed"]

helicity_M = 1
helicity_N = -4
helicity_Mp = 0
helicity_Np = -1

boozmn_filename = "../inputs/boozmn_beta2.5_QH.nc"
perfect = True  # enforce perfect QS?
AE_filename = "QH_10harmonics_scale0_00464159.npy"
plot_losses = False

ns_points = 5 if in_github_actions else 30  # number of radial grid points for heatmap
particles_per_surface = (
    2 if in_github_actions else 20
)  # number of particles per radial grid point for heatmap
nlambda_points = 5 if in_github_actions else 30  # number of lambda points for heatmap
tmax = 1e-4 if in_github_actions else 1e-2  # maximum time for trajectory integration

# Eprime parameters
lam = 0.0
p0_int = 0.1

order = 3
degree = 3
resolution = 10 if in_github_actions else 48
# resolution for perfect QS enforced if needed
res_p = 10 if in_github_actions else 48

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
    ns_points=ns_points,
    particles_per_surface=particles_per_surface,
    nlambda_points=nlambda_points,
    sign_vpar=sign_vpar,
    tmax=tmax,
    comm=comm_world,
    savedata=not in_github_actions,
    file_name=filepath,
)

# mode numbers of each harmonic we're searching for resonances with
mode_numbers = {h: (AE_temp.harmonics[h].m, AE_temp.harmonics[h].n) for h in harmonics}

max_mu = Ekin / min_volmodB
# iterate through pitch angles
mu_harmonics = np.linspace(0, max_mu, resolution)

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
        ns_poinc=resolution * 2,
        Nmaps=resolution,
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
    if comm_world is not None:
        comm_world.Barrier()


plt.clf()

mpl.rcParams["xtick.labelsize"] = 25
mpl.rcParams["ytick.labelsize"] = 25

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

if verbose and not in_github_actions:
    fig, (ax, ax_dummy) = plt.subplots(
        1, 2, gridspec_kw={"width_ratios": [4, 0.61]}, figsize=(17, 12)
    )

    # plot heatmap
    ax = heat_map.plot_heatmap(
        ax=ax,
        savepath=filepath + "heatmap.png",
        plot_losses=plot_losses,
        DA_at_loss=plot_losses,
    )

    fig = ax.get_figure()

    # fit and plot resonance lines on top of the heatmap
    labels_lines, labels_text = plot_resonance_lines(
        ax,
        harmonics,
        mode_numbers,
        heat_map.trapped_boundary_fit,
        heat_map.trapped_boundary_fit_pitch,
        harmonic_cmap,
        norm,
        possible_linestyles,
        max_ell,
        min_crossing_points=3,
        poly_degree=1,
        n_fit_points=resolution * 2,
    )
    ax_dummy.remove()
    ax.legend(
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
