The example presented here is for a FAR3D initial value run with a single fast species. In the FAR3D initial value run, the temperature and density profiles are inputted into `Data.txt`. This specific example is for the Landreman beta 2.5% QH.

Running FAR3D produces `phi_0000` and `temp_grwth_omega` files. `phi_0000` contains the amplitude data ($\varphi$) for each Fourier mode across flux surfaces, and `temp_grwth_omega` contains the growth rate and frequency for each mode. In `far3d_numpy_file_maker.ipynb`, the eigenfrequency from `temp_grwth_omega` and the file `phi_0000` are inputted. The output is an object of type `FAR3DEigenvector`, which contains the amplitude for each Foruier mode from `phi_0000` and the eigenfrequency. The `FAR3DEigenvector` is exported as a numpy file.

`tracing_with_FAR3D_AE.py` reads the numpy file and provided VMEC equilibrium file in Boozer coordinates (boozmn.nc) and performs the perturbed particle tracing for an fusion-born alpha particle population.

On perlmutter (08.17.26), the wallclock time is about 8 minutes using the included slurm script.

FAR3D: J. Varela, *et al.* Stability optimization of energetic particle driven modes in nuclear fusion devices: the FAR3d gyrofluid code. *Front. Phys.* 12:1422411. 2024. doi: 10.3389/fphy.2024.1422411.
