This example computes the passing particle Poincare map in the ATEN configuration scaled
to the size and field strength of ARIES-CS. Co-passing alpha particles (mu = 0, sign(v_{\|}) = +1) with
the alpha particle birth energy are assumed.

The passing map is computed with the chaos_detection parameter enabled. Weighted Birkhoff Averaging is
added as a setting to the passing map and applied to a particle canonical momentum
to create a numerical metric (digit accuracy) of chaos, implemented as described in:

N. Duignan and J. D. Meiss. "Distinguishing between regular and chaotic orbits of flows by the weighted birkhoff average." Physical Nonlinear Phenomena. (2023): 449:133749.

Set use_catapult = True near the top of passing_map.py to use GPU tracing.
The default False uses CPU tracing with the same parameters. Run from this
directory with "python passing_map.py". Both backends use dense-output roots.
