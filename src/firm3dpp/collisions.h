#pragma once

#include <vector>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <algorithm>
#include <utility>

using std::vector;

#ifdef __CUDACC__
  #define FIRM3D_HD __host__ __device__
#else
  #define FIRM3D_HD
#endif

// Physical constants (SI).  
static constexpr double COLL_EPSILON0    = 8.8541878188e-12;   // F/m
static constexpr double COLL_SQRT_PI     = 1.7724538509055159; // sqrt(pi)
static constexpr double COLL_HBAR        = 1.054571817e-34;    // J·s (reduced Planck)
static constexpr double COLL_EV_TO_J     = 1.602176634e-19;   // J/eV

// Maximum number of background species.
static constexpr int COLL_MAX_SPECIES = 8;

// Reflecting speed boundary as a fraction of sqrt(T_b / m_a), following
// ASCOT5 (MCCC_CUTOFF in mccc.h): "if the guiding center energy goes below
// this, it is mirrored to prevent collision coefficients from diverging."
static constexpr double COLL_CUTOFF      = 0.1;

// --------------------------------------------------------------------------
// Chandrasekhar G function: G(x) = [erf(x) - (2x/sqrt(pi)) exp(-x^2)] / (2x^2)
// Satisfies G'(x) = (2/sqrt(pi)) exp(-x^2) - 2 G(x)/x
// --------------------------------------------------------------------------
FIRM3D_HD inline double chandrasekhar_G(double x) {
    if (x == 0.0) return 0.0;
    return (std::erf(x) - (2.0 * x / COLL_SQRT_PI) * std::exp(-x * x)) / (2.0 * x * x);
}

// dG/dx = (2/sqrt(pi)) exp(-x^2) - 2 G(x)/x
FIRM3D_HD inline double chandrasekhar_G_deriv(double x) {
    if (x == 0.0) return 2.0 / (3.0 * COLL_SQRT_PI);
    return (2.0 / COLL_SQRT_PI) * std::exp(-x * x) - 2.0 * chandrasekhar_G(x) / x;
}

// --------------------------------------------------------------------------
// Background species specification.
//
// Density n_grid (m^-3) and temperature T_grid (eV) are stored on a uniform
// grid in s in [0, 1]. Simple linear interpolation is used.
// --------------------------------------------------------------------------
struct ThermalBackground {
    vector<double> s_grid;   // uniform s in [0,1], length >= 2
    vector<double> n_grid;   // number density m^-3 at each s
    vector<double> T_grid;   // temperature eV at each s
    double mass;             // kg
    double charge;           // C (signed)
};


// --------------------------------------------------------------------------
// One background species in the plain-pointer form the collision routines
// take.  Same content as ThermalBackground, but the profile samples are
// referenced by raw pointer instead of held in std::vector, so the struct
// works inside CUDA kernels as well as on the CPU.
//
// --------------------------------------------------------------------------
struct ThermalBackgroundView {
    const double* n_grid;    // number density m^-3, n_points samples
    const double* T_grid;    // temperature eV, n_points samples
    int    n_points;         // >= 2
    double s_min, s_max;
    double mass;             // kg
    double charge;           // C (signed)

    FIRM3D_HD double interp(const double* vals, double s) const {
        double lo = s_min, hi = s_max;
        if (s < lo) {
            s = lo;
        } else if (s > hi) {
            s = hi;
        }
        double ds = (hi - lo) / (n_points - 1);
        int i = (int)((s - lo) / ds);
        // Keep i + 1 in bounds.
        if (i < 0) {
            i = 0;
        } else if (i > n_points - 2) {
            i = n_points - 2;
        }
        double t = (s - (lo + i * ds)) / ds;
        return vals[i] * (1.0 - t) + vals[i + 1] * t;
    }

    FIRM3D_HD double n(double s) const { return interp(n_grid, s); }
    FIRM3D_HD double T_eV(double s) const { return interp(T_grid, s); }
};

// --------------------------------------------------------------------------
// Total Debye length at s: 1/lambda_D^2 = sum_b n_b q_b^2 / (eps0 T_b),
// active (n > 0, T > 0) species only.  0 when none is active.
// --------------------------------------------------------------------------
FIRM3D_HD inline double debye_length(
    const ThermalBackgroundView* backgrounds, int n_backgrounds, double s)
{
    double inv_lD_sq = 0.0;
    for (int ib = 0; ib < n_backgrounds; ++ib) {
        const ThermalBackgroundView& bg = backgrounds[ib];
        double n_b = bg.n(s), T_b = bg.T_eV(s) * COLL_EV_TO_J;
        if (n_b > 0.0 && T_b > 0.0) {
            inv_lD_sq += n_b * bg.charge * bg.charge / (COLL_EPSILON0 * T_b);
        }
    }
    double lambda_D = 0.0;
    if (inv_lD_sq > 0.0) {
        lambda_D = 1.0 / std::sqrt(inv_lD_sq);
    }
    return lambda_D;
}

// --------------------------------------------------------------------------
// Coulomb logarithm for EP (species a) against one background species:
// ln(lambda_D / b_min) where b_min = max(b_cl, b_qm).
//
// b_cl  = |qa qb| / (4 pi eps0 mr vbar)   classical 90-deg deflection
// b_qm  = hbar / (2 mr sqrt(vbar))         de Broglie wavelength
// vbar  = v^2 + v_th_b^2   (v_th_b = sqrt(2 T_b / m_b))
//
// vbar handles slow EP (v << v_th,e) and fast EP (v >> v_th,b); taking
// max(b_cl, b_qm) handles the quantum regime (fast EP against electrons)
// where the de Broglie wavelength exceeds the classical distance of closest
// approach.  This matches the ASCOT5 convention (Hirvijoki et al., Comput.
// Phys. Commun. 185, 1310, 2014).
//
// See also: NRL Plasma Formulary (Huba), "Collision Parameters" section;
// Spitzer, Physics of Fully Ionized Gases (1962), Ch. 5.
//
// T_b in J. 
// --------------------------------------------------------------------------
FIRM3D_HD inline double coulomb_log(
    double v, double m_a, double q_a,
    double m_b, double q_b, double T_b, double lambda_D)
{
    double v_th     = std::sqrt(2.0 * T_b / m_b);
    double m_r      = m_a * m_b / (m_a + m_b);
    double v_eff_sq = v * v + v_th * v_th;
    double bcl      = std::abs(q_a * q_b)
                      / (4.0 * M_PI * COLL_EPSILON0 * m_r * v_eff_sq);
    double bqm      = COLL_HBAR / (2.0 * m_r * std::sqrt(v_eff_sq));
    double b_min    = bqm;
    if (bcl > bqm) {
        b_min = bcl;
    }
    return std::log(lambda_D / b_min);
}

// --------------------------------------------------------------------------
// Collision coefficients for EP (species a) against one Maxwellian background.
//
// Returns the summed coefficients for the collision kick in (v, xi):
//   D_par      : parallel velocity diffusion  [m^2/s^3]
//   dD_par_dv  : d(D_par)/dv               [m/s^2]  (for Milstein)
//   nu_D       : pitch-angle scattering freq [s^-1]; also the deterministic
//                pitch drift, dxi/dt|_coll = -xi * nu_D
//   K          : deterministic drift in v   [m/s^2]
//                K = Q + d(D_par)/dv + 2*D_par/v
//   v_cutoff   : reflecting speed boundary  [m/s]
// --------------------------------------------------------------------------
struct CollisionCoefficients {
    double D_par;       // m^2/s^3
    double dD_par_dv;   // m/s^2
    double nu_D;        // s^-1
    double K;           // m/s^2  (total deterministic drift in v)
    double v_cutoff;    // m/s   (reflecting speed boundary, ASCOT5 style)
};

// Status returned by the device-safe core.
enum CollisionStatus { COLL_OK = 0, COLL_LNLAMBDA_NONPOSITIVE = 1 };

FIRM3D_HD inline int compute_collision_coefficients_core(
    double v,                     // EP speed [m/s]
    double s,                     // flux surface label
    double m_a,                   // EP mass [kg]
    double q_a,                   // EP charge [C]
    const ThermalBackgroundView* backgrounds,
    int n_backgrounds,
    CollisionCoefficients* out)
{
    CollisionCoefficients c = {0.0, 0.0, 0.0, 0.0, 0.0};

    double lambda_D = debye_length(backgrounds, n_backgrounds, s);

    // Coldest active species sets the reflecting boundary v_cutoff.
    double T_min = HUGE_VAL;
    for (int ib = 0; ib < n_backgrounds; ++ib) {
        const ThermalBackgroundView& bg = backgrounds[ib];
        double n_b = bg.n(s), T_b = bg.T_eV(s) * COLL_EV_TO_J;
        if (n_b > 0.0 && T_b > 0.0 && T_b < T_min) {
            T_min = T_b;
        }
    }
    if (T_min < HUGE_VAL)
        c.v_cutoff = COLL_CUTOFF * std::sqrt(T_min / m_a);

    for (int ib = 0; ib < n_backgrounds; ++ib) {
        const ThermalBackgroundView& bg = backgrounds[ib];
        double n_b = bg.n(s);
        double T_b = bg.T_eV(s) * COLL_EV_TO_J;
        if (n_b <= 0.0 || T_b <= 0.0) continue;

        double m_b    = bg.mass;
        double q_b    = bg.charge;

        // Background thermal speed and normalized EP speed
        double v_th   = std::sqrt(2.0 * T_b / m_b);
        double x      = v / v_th;

        double lnL = coulomb_log(v, m_a, q_a, m_b, q_b, T_b, lambda_D);
        if (lnL <= 0.0) {
            return COLL_LNLAMBDA_NONPOSITIVE;
        }

        double G      = chandrasekhar_G(x);
        double Gp     = chandrasekhar_G_deriv(x);   // dG/dx
        double erf_x  = std::erf(x);

        // Prefactor: Gamma = n_b q_a^2 q_b^2 lnL / (4 pi eps0^2 m_a^2)  [m^3/s^4]
        double Gamma  = n_b * q_a * q_a * q_b * q_b * lnL
                        / (4.0 * M_PI * COLL_EPSILON0 * COLL_EPSILON0 * m_a * m_a);

        // Parallel diffusion and its v-derivative
        double D_par_b   = Gamma * G / v;
        // d(G/v)/dv = G'/(v_th * v) - G/v^2  =>  d(D_par)/dv = Gamma*(G'/v_th - G/v)/v
        double dD_par_dv_b = Gamma * (Gp / v_th - G / v) / v;

        // Pitch-angle scattering frequency: nu_D = Gamma*(erf(x) - G(x)) / v^3
        double nu_D_b = Gamma * (erf_x - G) / (v * v * v);

        // Drag: Q = -(m_a v / T_b) * D_par
        //                             = -Gamma * G * m_a / T_b.
        // Matches ASCOT5 mccc_coefs_Q and guarantees relaxation to the
        // background Maxwellian.
        double Q_b = -Gamma * G * m_a / T_b;

        // Total deterministic drift in v: K = Q + d(D_par)/dv + 2*D_par/v
        double K_b = Q_b + dD_par_dv_b + 2.0 * D_par_b / v;

        c.D_par     += D_par_b;
        c.dD_par_dv += dD_par_dv_b;
        c.nu_D      += nu_D_b;
        c.K         += K_b;
    }
    *out = c;
    return COLL_OK;
}

// --------------------------------------------------------------------------
// Convert ThermalBackground structs into the plain-pointer form
// (ThermalBackgroundView) that the collision routines take.  Fills
// views[0..n) with pointers into each background's grids and returns n.
//
// The pointer form exists because the collision routines also run in CUDA
// kernels, where std::vector cannot be used. 
// --------------------------------------------------------------------------
inline int make_thermal_views(
    const vector<ThermalBackground>& backgrounds,
    ThermalBackgroundView views[COLL_MAX_SPECIES])
{
    const int n_bg = (int)backgrounds.size();
    for (int ib = 0; ib < n_bg; ++ib) {
        const ThermalBackground& bg = backgrounds[ib];
        views[ib].n_grid   = bg.n_grid.data();
        views[ib].T_grid   = bg.T_grid.data();
        views[ib].n_points = (int)bg.s_grid.size();
        views[ib].s_min    = bg.s_grid.front();
        views[ib].s_max    = bg.s_grid.back();
        views[ib].mass     = bg.mass;
        views[ib].charge   = bg.charge;
    }
    return n_bg;
}

// --------------------------------------------------------------------------
// Smallest Coulomb logarithm over the background species at one (v, s), and
// the index of the species attaining it; {HUGE_VAL, -1} when none is active.
// --------------------------------------------------------------------------
inline std::pair<double, int> min_coulomb_log(
    double v,
    double s,
    double m_a,
    double q_a,
    const vector<ThermalBackground>& backgrounds)
{
    ThermalBackgroundView views[COLL_MAX_SPECIES];
    const int n_bg = make_thermal_views(backgrounds, views);
    double lambda_D = debye_length(views, n_bg, s);
    double lnL_min = HUGE_VAL;
    int ib_min = -1;
    if (lambda_D > 0.0) {
        for (int ib = 0; ib < n_bg; ++ib) {
            double n_b = views[ib].n(s), T_b = views[ib].T_eV(s) * COLL_EV_TO_J;
            if (n_b <= 0.0 || T_b <= 0.0) continue;
            double lnL = coulomb_log(
                v, m_a, q_a, views[ib].mass, views[ib].charge, T_b, lambda_D);
            if (lnL < lnL_min) {
                lnL_min = lnL;
                ib_min = ib;
            }
        }
    }
    return {lnL_min, ib_min};
}

// --------------------------------------------------------------------------
// Host wrapper: keeps the vector-of-ThermalBackground interface and restores
// the exception, so every existing caller and the whole test suite are
// unaffected by the split.  The physics is entirely in the core above, which
// is what a device build compiles.
// --------------------------------------------------------------------------
inline CollisionCoefficients compute_collision_coefficients(
    double v,
    double s,
    double m_a,
    double q_a,
    const vector<ThermalBackground>& backgrounds)
{
    ThermalBackgroundView views[COLL_MAX_SPECIES];
    const int n_bg = make_thermal_views(backgrounds, views);

    CollisionCoefficients c = {0.0, 0.0, 0.0, 0.0, 0.0};
    int status = compute_collision_coefficients_core(
        v, s, m_a, q_a, views, n_bg, &c);

    if (status == COLL_LNLAMBDA_NONPOSITIVE) {
        throw std::runtime_error(
            "collisions: the background profiles give an unphysical Coulomb "
            "logarithm (ln_Lambda <= 0); keep T finite where n > 0");
    }
    return c;
}

// --------------------------------------------------------------------------
// Apply one collision step to (v, xi) after an accepted orbit step of size h:
// deterministic drift by explicit Euler, plus the Milstein noise term.
// Modifies v and xi in-place.
//
// This matches the (v, xi) half of ASCOT5's mccc_gc_milstein.c. 
//
// --------------------------------------------------------------------------
FIRM3D_HD inline void milstein_collision_step(
    double& v,
    double& xi,
    const CollisionCoefficients& coef,
    double h,                    // accepted orbit step size [s] (physical time)
    double dW_v,                 // N(0,h) increment for v
    double dW_xi)                // N(0,h) increment for xi
{
    // Noise amplitudes. 
    double g_v  = std::sqrt(2.0 * coef.D_par);
    double g_xi = std::sqrt(coef.nu_D * (1.0 - xi) * (1.0 + xi));

    // Deterministic drift, evaluated at the pre-step (v, xi) as in ASCOT5
    double drift_v  = coef.K * h;
    double drift_xi = -xi * coef.nu_D * h;

    // Milstein corrections
    double milstein_v  = 0.5 * coef.dD_par_dv * (dW_v * dW_v - h);
    double milstein_xi = -0.5 * xi * coef.nu_D  * (dW_xi * dW_xi - h);

    v  += drift_v  + g_v  * dW_v  + milstein_v;
    xi += drift_xi + g_xi * dW_xi + milstein_xi;

    // Boundary conditions, following ASCOT5 (mccc_gc_milstein.c):
    //  - the speed reflects off the thermal cutoff v_cutoff, so the
    //    collision coefficients, which diverge as v -> 0, are never
    //    evaluated deep below the background thermal speed;
    //  - the pitch mirrors at |xi| = 1 (a hard clamp would pile up
    //    probability at exactly +-1).
    // The final clamp on xi only acts on pathologically large steps.
    if (v < coef.v_cutoff)
        v = 2.0 * coef.v_cutoff - v;
    if (xi > 1.0) {
        xi = 2.0 - xi;      // reflect off xi = +1
    } else if (xi < -1.0) {
        xi = -2.0 - xi;     // reflect off xi = -1
    }
    if (xi < -1.0) {
        xi = -1.0;
    } else if (xi > 1.0) {
        xi = 1.0;
    }
}

// --------------------------------------------------------------------------
// Sub-stepping control for the collision kick.
//
// The orbit integrator chooses its step from orbit dynamics alone.  
// Applying them as one explicit-Euler kick over that whole step is inaccurate 
// whenever the collision rates are fast, e.g. in the thermal regime.
//
// Sub-cycling fixes this. The particle position is frozen
// during a kick, so a sub-step needs no field evaluation, only a coefficient
// re-evaluation at the updated speed.  
//
// --------------------------------------------------------------------------

// Fraction of a collision timescale a single sub-step may cover.
static constexpr double COLL_SUBSTEP_SAFETY = 0.05;
// Runaway guard. 
static constexpr int COLL_MAX_SUBSTEPS = 10000;

FIRM3D_HD inline int collision_substeps(
    double v,
    const CollisionCoefficients& coef,
    double h)
{
    if (!(h > 0.0) || !(v > 0.0)) return 1;   // also rejects NaN

    // Fractional change each process would impose over the whole step h; a
    // sub-step of h/n must keep every one below COLL_SUBSTEP_SAFETY.
    //
    // n = drift / safety.
    double pitch_drift = coef.nu_D * h;              // |d xi|
    double speed_drift = std::fabs(coef.K) * h / v;  // |d v| / v
    double max_drift = std::fmax(pitch_drift, speed_drift);
    double n_for_drift = max_drift / COLL_SUBSTEP_SAFETY;

    double speed_variance = 2.0 * coef.D_par * h;    // <dv^2> over the step
    double speed_spread = 0.0;                       // rms |d v| / v
    if (speed_variance > 0.0) {
        speed_spread = std::sqrt(speed_variance) / v;
    }
    double spread_ratio = speed_spread / COLL_SUBSTEP_SAFETY;
    double n_for_diffusion = spread_ratio * spread_ratio;

    double n_needed = n_for_diffusion;
    if (n_for_drift > n_for_diffusion) {
        n_needed = n_for_drift;
    }

    n_needed = std::fmin(std::fmax(n_needed, 1.0), (double)COLL_MAX_SUBSTEPS);
    return (int)std::ceil(n_needed);
}
