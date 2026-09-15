---
title: 'FIRM3D: Fast ion reduced models in 3D'
tags:
  - Python
  - C++
  - CUDA
  - plasma physics
  - stellarators
  - fusion
  - energetic particles
authors:
  - name: Elizabeth Paul
    orcid: 0000-0002-9355-5595
    corresponding: true
    affiliation: 1
  - name: Alexey Knyazev
    orcid: 0000-0001-8333-859X
    affiliation: 1
  - name: Michael Czekanski
    orcid: 0009-0005-2520-3415
    affiliation: 2
  - name: Alexandra Lachmann
    orcid: 0000-0002-8341-107X
    affiliation: 1
  - name: Abdullah Hyder
    orcid: 0000-0003-4410-3661
    affiliation: 1
  - name: Christopher Albert
    orcid: 0000-0003-4773-416X
    affiliation: 3
  - name: Matt Landreman
    orcid: 0000-0002-7233-577X
    affiliation: 4
affiliations:
 - name: Department of Applied Physics and Applied Mathematics, Columbia University, USA
   index: 1
 - name: Department of Statistics and Data Science, Cornell University, USA
   index: 2
 - name: Graz University of Technology, Austria
   index: 3
 - name: University of Maryland, College Park, USA
   index: 4
date: 11 May 2026
bibliography: paper.bib
---

# Summary

The dynamics of energetic particle (EP) species, born from fusion reactions or plasma heating schemes, are critical for predicting the behavior of magnetic confinement fusion experiments and future fusion reactors. Because energetic particles are largely collisionless, the orbits of Monte Carlo samples drawn from a given distribution function can be efficiently integrated in prescribed electromagnetic fields. In addition to the static magneto-hydrodynamic (MHD) equilibrium fields produced by the electromagnetic coils of a fusion device, MHD waves can be excited by—and resonantly transport—energetic particle populations.

FIRM3D is an open-source Python/C++/CUDA software suite for modeling energetic particle dynamics in 3D magnetic fields, available at [https://github.com/ColumbiaStellaratorTheory/firm3d](https://github.com/ColumbiaStellaratorTheory/firm3d). Its guiding-center integration routines grew out of SIMSOPT [@2021Landreman], extended with additional physics and diagnostics not typically required in the stellarator optimization context, as a standalone framework with minimal dependencies for the broader stellarator and plasma physics community.

Components of FIRM3D include:

- Interfaces with MHD equilibrium and wave stability software (BOOZ\_XFORM, AE3D, FAR3D).
- CPU and GPU parallelized integration of the guiding center orbit equation, with symplectic and Runge-Kutta integrator options.
- Orbit visualization and transport diagnostics, including Poincaré maps, orbit classification, and weighted Birkhoff averaging.

# Statement of need

Recent advances in stellarator optimization [@2022LandremanOpt] have produced equilibria that satisfy many physics and engineering constraints for a fusion reactor. One critical feature is the ability to confine guiding center trajectories of energetic particle species. Through quasisymmetry—a hidden symmetry of the field strength that provides integrability of guiding center motion—stellarator magnetic fields can now be designed with excellent energetic particle confinement.

However, perturbing electromagnetic fields can still transport energetic particles. The class of MHD waves of primary concern are Alfvén eigenmodes (AEs), which are driven unstable by free energy in the EP distribution function and can resonantly transport EPs. Alfvénic activity is considered the major limitation to alpha confinement in burning tokamak plasmas [@2014Gorelenkov], and has been observed in numerous stellarator experiments [@2011Toi; @2020Rahbarnia]. Given the recent growth of the private fusion industry, companies pursuing the stellarator path to fusion require tools for assessing EP-driven wave stability and the resulting EP transport. FIRM3D addresses this need by providing a modular, GPU-accelerated energetic particle simulation framework that bridges MHD equilibrium solvers and wave stability codes, intended for plasma physicists and fusion engineers who require EP simulation capabilities beyond those available in optimization-focused codes.

# State of the field

Existing tools such as ASCOT5 [@varje2019ascot5] provide high-fidelity Monte Carlo EP tracking, primarily for tokamak geometry, while SIMPLE [@2020Albert] offers symplectic guiding-center integration without native interfaces to stellarator MHD-stability codes or GPU acceleration. Extending SIMSOPT's optimization-oriented guiding-center routines directly would have tied FIRM3D's dependencies and release cycle to a much larger codebase not organized around EP-specific diagnostics, motivating a standalone package instead. FIRM3D complements the above tools by combining tight integration with the stellarator optimization ecosystem (SIMSOPT, BOOZ\_XFORM) and MHD wave-stability codes (AE3D, FAR3D), a choice of symplectic and Runge-Kutta integrators for cross-validation, and native GPU acceleration via CUDA, all within a lightweight open-source Python package.

# Software design

Integration of guiding center trajectories is performed given the magnetic field, particle initial conditions, and integrator specification. The equilibrium magnetic field is typically provided through an interface with BOOZ\_XFORM [@booz_xform] via the `BoozerMagneticField` class. Since BOOZ\_XFORM computes the Fourier harmonics of the magnetic field on a uniform radial grid, Lagrange interpolation is used to evaluate the field throughout the volume. The magnetic field perturbation corresponding to an MHD mode from AE3D [@2010Spong] or FAR3D [@2024Varela] can then be superimposed on the interpolated equilibrium field; such modes are stored on a radial grid of Fourier harmonics in Boozer coordinates. Helper functions are provided to generate particle initial conditions from a known distribution function or by preserving a conserved quantity such as energy or canonical momentum.

Three integrators are offered. An interface to the Boost Runge-Kutta Dormand-Prince 5 method [@BoostOdeint] provides adaptive step size control and dense output. A custom Dormand-Prince 5 implementation adds minimum step size control based on [@2007Press]. A symplectic integrator for non-canonical guiding-center orbits, using the explicit-implicit Euler scheme of [@2020Albert], trades pointwise accuracy for long-time stability of conserved quantities.

Since the performance bottlenecks are field interpolation and trajectory integration, the Lagrange interpolating polynomials and integrators are implemented in C++, with Python interfaces via pybind11 [@pybind11]. MPI parallelization over Fourier harmonics and OpenMP parallelization over interpolant nodes accelerate field setup. Because guiding center trajectories are independent, Monte Carlo samples are trivially parallelized over CPUs or GPUs; CUDA kernels implement field interpolation and trajectory integration on GPU. \autoref{fig:cpu_scaling} shows that tracing a fusion birth distribution in the Wistell-A equilibrium for $10^{-2}$ seconds on one NERSC Perlmutter CPU node (128 threads) versus one NVIDIA A100 favors the CPU below $10^3$ Monte Carlo samples, due to GPU launch latency, but is roughly an order of magnitude faster on GPU above that threshold; see [@2026Czekanski] for GPU implementation details.

![Wall-clock scaling with number of Monte Carlo samples on 1 Perlmutter CPU node (128 threads) vs. 1 NVIDIA A100. \label{fig:cpu_scaling}](figures/cpu_scaling_particles.png){height="5cm"}

These design choices are verified against known conservation laws and an independent code. For time-independent fields, the guiding-center Lagrangian conserves total energy $E$; Runge-Kutta methods suffer from net energy drift over time, while the symplectic integrator exhibits long-time stability with a conserved time-averaged energy (\autoref{fig:momentum_error}, left). For a perfectly quasisymmetric field, the toroidal canonical momentum $P_{\eta}$ is also conserved [@1995Boozer]; at a Dormand-Prince tolerance of $10^{-9}$ and 64 Lagrange interpolation nodes, its relative error converges to approximately $10^{-8}$ (\autoref{fig:momentum_error}, right). \autoref{fig:simple_orbit} benchmarks FIRM3D against SIMPLE [@2020Albert]: a trapped 3.5 MeV alpha particle trajectory in the precise QH equilibrium [@2022LandremanPrecise] agrees to a relative error of $7.8\times 10^{-3}$ in the $s$ coordinate at $10^{-3}$ seconds, and loss fractions for 5000 3.5 MeV alpha particles sampled from a fusion birth distribution in the precise QA equilibrium [@2022LandremanPrecise] and traced for $10^{-2}$ seconds are identical between codes.

![Left: Energy vs. time for an alpha particle in the $\beta = 2.5\%$ Landreman QH configuration, comparing Dormand-Prince and symplectic integrators. Right: Relative error in canonical momentum $P_{\eta}$ vs. Lagrange interpolation resolution and Dormand-Prince tolerance, for a perfectly quasisymmetric field. \label{fig:momentum_error}](figures/conservation.png)

![Comparison with SIMPLE. Left: trapped 3.5 MeV alpha particle orbit in the precise QH equilibrium. Right: loss fraction for 5000 3.5 MeV alpha particles in the precise QA equilibrium. \label{fig:simple_orbit}](figures/combined_figure_2.png)

Installation instructions and API documentation are available at [https://firm3d.readthedocs.io/](https://firm3d.readthedocs.io/). Examples are available in the repository at [https://github.com/ColumbiaStellaratorTheory/firm3d](https://github.com/ColumbiaStellaratorTheory/firm3d). FIRM3D is released under the MIT License. A suite of unit and regression tests is run automatically on CPUs and GPUs via continuous integration on GitHub Actions.

# Research impact statement

The FIRM3D routines and their SIMSOPT precursors have already enabled published research: a survey of EP loss mechanisms [@2022Paul], AE-induced transport analysis [@2023Paul; @knyazev2026shear], trapped EP resonance theory [@2025Chambliss], and analysis of the Helios reactor concept's alpha confinement [@von2026alpha; @swanson2025overview]. \autoref{fig:classification} and \autoref{fig:wba} illustrate the orbit classification, transport diagnostics, and AE-induced transport capabilities used in this work. For \autoref{fig:classification}, $5\times 10^{5}$ alpha particles are sampled from a fusion birth distribution in the $\beta = 2.5\%$ QA configuration [@2022LandremanPrecise] and traced for $10^{-2}$ seconds (0.62% lost). Trapping class (banana, ripple, barely trapped) is identified along each trajectory; of the lost particles, 29% are banana-class, 0.12% ripple-trapped, 0.03% barely trapped, and 4.8% transition between classes, with 66% exiting promptly before classification is possible. The normalized variation $\sqrt{\langle \Delta J_{\|}^2 \rangle}/\langle J_{\|} \rangle$ of the parallel adiabatic invariant $J_{\|} = \oint dl\, v_{\|}$ measures integrability [@2023Albert], and $\gamma_c$ measures convective transport [@2022Paul]; most lost banana particles exceed the 1% non-integrability threshold while $\gamma_c < 0.2$, indicating orbital chaos drives the losses. For \autoref{fig:wba}, passing particles with $\mu/E = 0.1$ are traced in the $\beta = 2.5\%$ QA equilibrium [@2022LandremanOpt] with a single-harmonic AE ($m = 30$, $n = 14$, $\omega = 136$ kHz; Table 1 of [@2023Paul]). The weighted Birkhoff average of $P_{\zeta}$ [@duignan2023distinguishing] serves as an integrability diagnostic, with digit accuracy below 3 indicating chaotic motion [@knyazev2026shear].

![Left: Trapped particle Poincaré map showing chaotic layers responsible for banana-drift diffusion. Right: Convective and diffusive transport measures for banana-trapped orbits. \label{fig:classification}](figures/classification.png)

![Kinetic Poincaré plot colored by weighted Birkhoff average digit accuracy, an indicator of chaos. \label{fig:wba}](figures/passing_poincare.png){width="60%"}

# AI usage disclosure

Claude Code (Anthropic) was used to assist with portions of the FIRM3D codebase and its readthedocs documentation. All AI-assisted code and documentation changes were reviewed by the maintainers and validated against the project's CPU and GPU continuous-integration test suite before merging.

# Acknowledgements

We acknowledge the SIMSOPT development team for providing the foundational guiding center integration routines. We acknowledge funding through the U.S. Department of Energy under contracts DE-SC0024630, DE-SC0024548, and DE-AC02-09CH11466, and through the Simons Foundation collaboration 'Hidden Symmetries and Fusion Energy,' Grant No. 601958. This research used resources of the National Energy Research Scientific Computing Center (NERSC), a DOE Office of Science User Facility, under NERSC award ERCAP0031926.

# References
