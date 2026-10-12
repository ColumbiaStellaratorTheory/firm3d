This example computes the kinetic passing Poincare map in the bundled ATEN
configuration, scaled to the size and field strength of ARIES-CS. It uses
the original CPU example parameters: 120 co-passing birth-energy alpha
particles (mu = 0), 1000 returns, 48 interpolation cells per coordinate,
and tolerance 1e-8. Smaller settings are used in GitHub Actions.

From this directory, run either backend:

python passing_map.py --backend cpu
python passing_map.py --backend catapult

CATAPULT requires a GPU. Both use PassingPoincare with the same parameters.
Outside GitHub Actions, rank zero saves poincare_cpu.png or
poincare_catapult.png in the current directory. Adjust the parameters
near the top of the script to change the launch grid or solver settings.
