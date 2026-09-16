import matplotlib as mpl

from firm3d.field.boozermagneticfield import (
    BoozerRadialInterpolant,
    InterpolatedBoozerField,
)
from firm3d.trajectory_helpers import (
    MapEquilibrium,
)
from firm3d.util.constants import (
    ALPHA_PARTICLE_CHARGE,
    ALPHA_PARTICLE_MASS,
    FUSION_ALPHA_PARTICLE_ENERGY,
)
from firm3d.util.functions import in_github_actions
from firm3d.util.mpi import comm_world, verbose

folder = "DATA/"
fname = ""
boozmn_filename = "../inputs/boozmn_beta2.5_QA.nc"

mpl.rcParams["font.size"] = 14  # base font size
mpl.rcParams["axes.labelsize"] = 14  # x/y labels
mpl.rcParams["axes.titlesize"] = 14
mpl.rcParams["xtick.labelsize"] = 14
mpl.rcParams["ytick.labelsize"] = 14
mpl.rcParams["legend.fontsize"] = 14
mpl.rcParams["figure.titlesize"] = 14

order = 3
degree = 3
resolution = 10 if in_github_actions else 48  # Resolution for field interpolation
ns_interp = resolution  # number of radial grid points for interpolation
ntheta_interp = resolution  # number of poloidal grid points for interpolation
nzeta_interp = resolution  # number of toroidal grid points for interpolation

tmax = 1e-4 if in_github_actions else 1e-2  # maximum time for trajectory integration
ns_points = 5 if in_github_actions else 30  # number of radial grid points for heatmap
particles_per_surface = (
    2 if in_github_actions else 20
)  # number of particles per radial grid point for heatmap
nlambda_points = 5 if in_github_actions else 30  # number of lambda points for heatmap

helicity_M = 1  # field strength helicity (QA)
helicity_N = 0  # field strength helicity (QA)
helicity_Mp = 0
helicity_Np = -1

bri = BoozerRadialInterpolant(
    boozmn_filename,
    order,
    no_K=True,
    comm=comm_world,
)
field = InterpolatedBoozerField(
    bri,
    degree,
    ns_interp=ns_interp,
    ntheta_interp=ntheta_interp,
    nzeta_interp=nzeta_interp,
)

sign_vpar = 1  # 1 for co-passing, -1 for counter-passing
mass = ALPHA_PARTICLE_MASS
charge = ALPHA_PARTICLE_CHARGE
Ekin = FUSION_ALPHA_PARTICLE_ENERGY


heat_map = MapEquilibrium(
    field,
    mass=mass,
    charge=charge,
    Ekin=Ekin,
    sign=sign_vpar,
    helicity_N=helicity_N,
    helicity_M=helicity_M,
    helicity_Mp=helicity_Mp,
    helicity_Np=helicity_Np,
    ns_points=ns_points,
    particles_per_surface=particles_per_surface,
    nlambda_points=nlambda_points,
    tmax=tmax,
    comm=comm_world,
    savedata=not in_github_actions,
    savepath="",
)

if verbose and not in_github_actions:
    heat_map.plot_heatmap(DA_at_loss=False)
