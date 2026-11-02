#include <gsl/gsl_vector.h>
#include <gsl/gsl_multiroots.h>
#include <gsl/gsl_errno.h>
#include "boozermagneticfield.h"
#include "symplectic.h"
#include <cassert>
#include <cmath>
#include <stdexcept>
#include <string>
#include "tracing_helpers.h"


using std::shared_ptr;
using std::vector;
using std::tuple;
using std::array;
using Array2 = BoozerMagneticField::Array2;

//
// Evaluates magnetic field in Boozer canonical coordinates (r, theta, zeta)
// and stores results in the SymplField object
//
void SymplField::eval_field(double s, double theta, double zeta)
{
    double Btheta, Bzeta, dBtheta, dBzeta, modB2;

    // The implicit step may probe s < 0 while an orbit passes through the
    // magnetic axis.
    double s_eval = std::abs(s);

    stz[0, 0] = s_eval; stz[0, 1] = theta; stz[0, 2] = zeta;
    field->set_points(stz);
    // A = psi \nabla \theta - psip \nabla \zeta
    Atheta = s*field->psi0;
    Azeta =  -field->psip()(0);
    dAtheta[0] = field->psi0; // dAthetads
    dAzeta[0] = -field->iota()(0)*field->psi0; // dAzetads
    for (int i=1; i<3; i++)
    {
        dAtheta[i] = 0.0;
        dAzeta[i] = 0.0;
    }

    modB = field->modB()(0);
    dmodB[0] = field->modB_derivs_ref()(0);
    dmodB[1] = field->modB_derivs_ref()(1);
    dmodB[2] = field->modB_derivs_ref()(2);

    Btheta = field->I()(0);
    Bzeta = field->G()(0);
    dBtheta = field->dIds()(0);
    dBzeta = field->dGds()(0);

    modB2 = pow(modB, 2);

    htheta = Btheta/modB;
    hzeta = Bzeta/modB;
    dhtheta[0] = dBtheta/modB - Btheta*dmodB[0]/modB2;
    dhzeta[0] = dBzeta/modB - Bzeta*dmodB[0]/modB2;

    for (int i=1; i<3; i++)
    {
        dhtheta[i] = -Btheta*dmodB[i]/modB2;
        dhzeta[i] = -Bzeta*dmodB[i]/modB2;
    }

}

// compute pzeta normalized by m * vnorm**2 * tnorm 
double SymplField::get_pzeta(double vpar) {
    double pzeta = (m*hzeta*vpar + q*Azeta/vnorm) / (m * vnorm * tnorm);
    return pzeta;
}

// computes values of H, ptheta and vpar at z=(s, theta, zeta, pzeta)
void SymplField::get_val(double pzeta) {
    vpar = (pzeta * m * vnorm * tnorm - q*Azeta/vnorm)/(hzeta*m);
    // H is normalized by m * vnorm**2 
    H = pow(vpar,2)/2.0 + mu*modB/pow(vnorm,2);
    // ptheta is normalized by vnorm
    ptheta = (m*htheta*vpar + q*Atheta/vnorm) / (m * vnorm * tnorm);
}

// computes H, ptheta and vpar at z=(s, theta, zeta, pzeta) and their derivatives
void SymplField::get_derivatives(double pzeta) {
    get_val(pzeta);

    for (int i=0; i<3; i++)
        dvpar[i] = -q*dAzeta[i]/(vnorm*hzeta*m) - (vpar/hzeta)*dhzeta[i];

    dvpar[3]   = vnorm * tnorm / hzeta; // dvpardpzeta

    for (int i=0; i<3; i++)
        dH[i] = vpar*dvpar[i] + mu*dmodB[i]/pow(vnorm,2);
    dH[3]   = vpar*dvpar[3]; // dHdpzeta

    // ptheta = (m*htheta*vpar + q*Atheta/vnorm) / (m * vnorm * tnorm);
    for (int i=0; i<3; i++)
        dptheta[i] = (m*dvpar[i]*htheta + m*vpar*dhtheta[i] + q*dAtheta[i]/vnorm) / (m * vnorm * tnorm);
    dptheta[3] = (m*htheta*dvpar[3]) / (m * vnorm * tnorm); // dpthetadpzeta
}

double SymplField::get_dsdtau() {
    // H is normalized by m*vnorm**2 
    // ptheta is normalized by m*vnorm**2 * tnorm
    // dsdt is normalized by tnorm
    return (-dH[1] + dptheta[3]*dH[2] - dptheta[2]*dH[3])/dptheta[0];
}

double SymplField::get_dthdtau() {
    // H is normalized by m*vnorm**2 
    // ptheta is normalized by m*vnorm**2 * tnorm
    // dthdt is normalized by tnorm
    return dH[0]/dptheta[0];
}

double SymplField::get_dzedtau() {
    // vpar is normalized by vnorm
    // H is normalized by m*vnorm**2 
    // ptheta is normalized by m*vnorm**2 * tnorm
    // htheta and hzeta have units of length -> normalized by tnorm/vnorm 
    return (vpar * tnorm * vnorm - dH[0]/dptheta[0]*htheta) / (hzeta);
}

double SymplField::get_dvpardtau() {
    double dsdtau = get_dsdtau();
    double dthdtau = get_dthdtau();
    double dzedtau = get_dzedtau();
    double dpzdtau = (-dH[2] + dH[0]*dptheta[2]/dptheta[0]);

    return dvpar[0] * dsdtau + dvpar[1] * dthdtau + dvpar[2] * dzedtau + dvpar[3] * dpzdtau;
}

double cubic_hermite_interp(double t_last, double t_current, double y_last, double y_current, double dy_last, double dy_current, double t)
{
    double dt = t_current - t_last;
    return (3*dt*pow(t-t_last,2) - 2*pow(t-t_last,3))/pow(dt,3) * y_current
            + (pow(dt,3)-3*dt*pow(t-t_last,2)+2*pow(t-t_last,3))/pow(dt,3) * y_last
            + pow(t-t_last,2)*(t-t_current)/pow(dt,2) * dy_current
            + (t-t_last)*pow(t-t_current,2)/pow(dt,2) * dy_last;
}

// Vector-based version of sympl_dense for the new solve function
class sympl_dense_vector {
public:
    // for interpolation
    array<double, 2> bracket_s = {};
    array<double, 2> bracket_dsdtau = {};
    array<double, 2> bracket_theta = {};
    array<double, 2> bracket_dthdtau = {};
    array<double, 2> bracket_zeta = {};
    array<double, 2> bracket_dzedtau = {};
    array<double, 2> bracket_vpar = {};
    array<double, 2> bracket_dvpardtau = {};

    // bounds of interval for interpolation between time steps
    double tau_last = 0.0;
    double tau_current = 0.0;

    void update(double tau, double dtau, vector<double> y, SymplField f) {
        tau_last = tau;
        tau_current = tau + dtau;
        
        // Store the state and derivatives at the endpoints
        bracket_s[0] = y[0];
        bracket_theta[0] = y[1];
        bracket_zeta[0] = y[2];
        bracket_vpar[0] = y[3];
        
        // Calculate derivatives at tau
        // Convert to physical coordinates only for field evaluation
        f.eval_field(y[0], y[1], y[2]);
        f.get_derivatives(f.get_pzeta(y[3]));
        vector<double> ydot(4);
        bracket_dsdtau[0] = f.get_dsdtau();
        bracket_dthdtau[0] = f.get_dthdtau();
        bracket_dzedtau[0] = f.get_dzedtau();
        bracket_dvpardtau[0] = f.get_dvpardtau();
        
        // Calculate state at tau+dtau (this is approximate)
        double dtau_small = dtau * 0.1;
        vector<double> y_next = y;
        y_next[0] += dtau_small * bracket_dsdtau[0];
        y_next[1] += dtau_small * bracket_dthdtau[0];
        y_next[2] += dtau_small * bracket_dzedtau[0];
        y_next[3] += dtau_small * bracket_dvpardtau[0];
        
        bracket_s[1] = y_next[0];
        bracket_theta[1] = y_next[1];
        bracket_zeta[1] = y_next[2];
        bracket_vpar[1] = y_next[3];
        
        // Calculate derivatives at tau+dtau (approximate)
        // Convert to physical coordinates only for field evaluation
        f.eval_field(y_next[0], y_next[1], y_next[2]);
        f.get_derivatives(f.get_pzeta(y_next[3]));
        vector<double> ydot_next(4);
        bracket_dsdtau[1] = f.get_dsdtau();
        bracket_dthdtau[1] = f.get_dthdtau();
        bracket_dzedtau[1] = f.get_dzedtau();
        bracket_dvpardtau[1] = f.get_dvpardtau();
    }
    
    void calc_state(double eval_tau, vector<double> &temp) {
        assert (tau_last <= eval_tau && eval_tau <= tau_current);
        temp.resize(4);
        temp[0] = cubic_hermite_interp(tau_last, tau_current, bracket_s[0], bracket_s[1], bracket_dsdtau[0], bracket_dsdtau[1], eval_tau);
        // Across a step in which the orbit crossed the axis, s turns around at
        // zero while the cubic through the two endpoints can undershoot past
        // it. Negative s is not a coordinate any consumer of the dense output
        // can use, and would spuriously trip an inner stopping criterion.
        if (temp[0] < 0.0) {
            temp[0] = 0.0;
        }
        temp[1] = cubic_hermite_interp(tau_last, tau_current, bracket_theta[0], bracket_theta[1], bracket_dthdtau[0], bracket_dthdtau[1], eval_tau);
        temp[2] = cubic_hermite_interp(tau_last, tau_current, bracket_zeta[0], bracket_zeta[1], bracket_dzedtau[0], bracket_dzedtau[1], eval_tau);
        temp[3] = cubic_hermite_interp(tau_last, tau_current, bracket_vpar[0], bracket_vpar[1], bracket_dvpardtau[0], bracket_dvpardtau[1], eval_tau);
    }
};

class f_quasi_params_vector{
public:
    double ptheta_old;
    double dtau;
    vector<double> z;
    SymplField f;
};

int f_euler_quasi_func_vector(const gsl_vector* x, void* p, gsl_vector* f)
{
    struct f_quasi_params_vector * params = (struct f_quasi_params_vector *)p;
    const double ptheta_old = (params->ptheta_old);
    const double dtau = (params->dtau);
    auto z = (params->z);
    SymplField field = (params->f);

    const double x0 = gsl_vector_get(x,0);
    const double x1 = gsl_vector_get(x,1);

    // s < 0 is handled by eval_field as a reflection through the axis, and a
    // trial point just outside s = 1 is how the solver walks a particle up to
    // the loss boundary, so neither may be rejected here. Only a non-finite
    // trial point is unusable.
    if (!std::isfinite(x0) || !std::isfinite(x1)) {
        return GSL_EDOM;
    }

    field.eval_field(x0, z[1], z[2]);
    field.get_derivatives(x1);

    // Apply normalization factor to make order unity. This uses dAtheta/ds
    // (= psi0) rather than Atheta itself: Atheta = s*psi0 vanishes on the
    // magnetic axis, so dividing by its square would make the fixed absolute
    // tolerance of gsl_multiroot_test_residual scale like 1/s^2. At small s
    // that demands a residual below the double precision roundoff floor and
    // the root solve can never converge, regardless of how well conditioned
    // the step actually is.
    double norm = field.q * field.dAtheta[0] / (field.m * field.vnorm * field.vnorm * field.tnorm);
    const double f0 = (field.dptheta[0]*(field.ptheta - ptheta_old)
        + dtau*(field.dH[1]*field.dptheta[0] - field.dH[0]*field.dptheta[1])) / (norm * norm); // corresponds with (2.6) in JPP 2020
    const double f1  = (field.dptheta[0]*(x1 - z[3])
        + dtau*(field.dH[2]*field.dptheta[0] - field.dH[0]*field.dptheta[2])) / (norm * norm); // corresponds with (2.7) in JPP 2020

    gsl_vector_set(f, 0, f0);
    gsl_vector_set(f, 1, f1);

    return GSL_SUCCESS;
}

// Perform hermite interpolation between timesteps for computing stopping criteria
void sympl_dense::update(double t, double dt, array<double, 4>  y, SymplField f) {
    tlast = t;
    tcurrent = t+dt;

    bracket_s[0] = bracket_s[1];
    bracket_theta[0] = bracket_theta[1];
    bracket_zeta[0] = bracket_zeta[1];
    bracket_vpar[0] = bracket_vpar[1];

    bracket_dsdt[0] =  bracket_dsdt[1];
    bracket_dthdt[0] = bracket_dthdt[1];
    bracket_dzedt[0] = bracket_dzedt[1];
    bracket_dvpardt[0] = bracket_dvpardt[1];

    bracket_s[1] = y[0];
    bracket_theta[1] = y[1];
    bracket_zeta[1] = y[2];
    bracket_vpar[1] = y[3];

    bracket_dsdt[1] = f.get_dsdtau();
    bracket_dthdt[1] = f.get_dthdtau();
    bracket_dzedt[1] = f.get_dzedtau();
    bracket_dvpardt[1] = f.get_dvpardtau();
}

// Perform hermite interpolation between timesteps for computing stopping criteria
void sympl_dense::calc_state(double eval_t, State &temp) {
    assert (tlast <= eval_t && eval_t <= tcurrent);
    temp[0] = cubic_hermite_interp(tlast, tcurrent, bracket_s[0], bracket_s[1], bracket_dsdt[0], bracket_dsdt[1], eval_t);
    temp[1] = cubic_hermite_interp(tlast, tcurrent, bracket_theta[0], bracket_theta[1], bracket_dthdt[0], bracket_dthdt[1], eval_t);
    temp[2] = cubic_hermite_interp(tlast, tcurrent, bracket_zeta[0], bracket_zeta[1], bracket_dzedt[0], bracket_dzedt[1], eval_t);
    temp[3] = cubic_hermite_interp(tlast, tcurrent, bracket_vpar[0], bracket_vpar[1], bracket_dvpardt[0], bracket_dvpardt[1], eval_t);
}

// see https://github.com/itpplasma/SIMPLE/blob/master/SRC/
//         orbit_symplectic_quasi.f90:timestep_euler1_quasi
tuple<vector<vector<double>>, vector<vector<double>>> solve_sympl_vector(
    SymplField f,
    vector<double> y, 
    double tau_max,
    double dtau,
    double roottol,
    vector<double> phases,
    vector<double> n_zetas,
    vector<double> m_thetas,
    vector<double> omegas,
    vector<shared_ptr<StoppingCriterion>> stopping_criteria,
    vector<double> vpars,
    bool phases_stop,
    bool vpars_stop,
    bool forget_exact_path,
    bool predictor_step,
    double dtau_save
)
{
    double abstol = 0;
    if (phases.size() != n_zetas.size() || phases.size() != m_thetas.size() || phases.size() != omegas.size()) {
        throw std::invalid_argument("phases, n_zetas, m_thetas, and omegas need to have matching length.");
    }

    vector<vector<double>> res = {};
    vector<vector<double>> res_hits = {};
    double tau = 0.0;
    bool stop = false;

    vector<double> z(4); // s, theta, zeta, pzeta
    vector<double> temp(4);

    y[3] /= f.vnorm;
    // y = [y1, y2, y3, vpar]
    // Translate y to z
    // y = [s, theta, zeta, vpar]
    // z = [s, theta, zeta, pzeta]
    // pzeta = m*vpar*hzeta + q*Azeta
    z[0] = y[0];
    z[1] = y[1];
    z[2] = y[2];
    f.eval_field(z[0], z[1], z[2]);
    z[3] = f.get_pzeta(y[3]);
    f.get_derivatives(z[3]);
    double ptheta_old = f.ptheta;
    double tau_last = tau;

    // for interpolation
    sympl_dense_vector dense;
    dense.update(tau, dtau, y, f);

    // set up root solvers
    const gsl_multiroot_fsolver_type * Newt = gsl_multiroot_fsolver_hybrids;
    gsl_multiroot_fsolver *s_euler;
    s_euler = gsl_multiroot_fsolver_alloc(Newt, 2);

    struct f_quasi_params_vector params = {ptheta_old, dtau, z, f};
    gsl_multiroot_function F_euler_quasi = {&f_euler_quasi_func_vector, 2, &params};
    gsl_vector* xvec_quasi = gsl_vector_alloc(2);

    int status;
    int iter = 0;
    double s_guess = z[0];
    double pzeta_guess = z[3];

    do {
        // Save initial point
        if (tau==0){
            double t_save = tau * f.tnorm;
            vector<double> stzvt(4);
            y_to_stzvt(y, stzvt, 0, f.vnorm, f.tnorm);
            vector<double> save_point = {t_save};
            save_point.insert(save_point.end(), stzvt.begin(), stzvt.end());
            res.push_back(save_point);
        }

        params.ptheta_old = ptheta_old;
        params.z = z;
        params.dtau = dtau;
        gsl_vector_set(xvec_quasi, 0, s_guess);
        gsl_vector_set(xvec_quasi, 1, pzeta_guess);
        gsl_multiroot_fsolver_set(s_euler, &F_euler_quasi, xvec_quasi);

        int root_iter = 0;
        // Solve implicit part of time-step with some quasi-Newton
        // applied to f_euler1_quasi. This corresponds with (2.6)-(2.7) in JPP 2020,
        // which are solved for x = [s, pzeta].
        do
          {
            root_iter++;

            status = gsl_multiroot_fsolver_iterate(s_euler);


            if (status) {  /* check if solver is stuck */
                printf("iter = %3u x = % .10e % .10e "
                        "f(x) = % .10e % .10e\n",
                        root_iter,
                        gsl_vector_get (s_euler->x, 0),
                        gsl_vector_get (s_euler->x, 1),
                        gsl_vector_get (s_euler->f, 0),
                        gsl_vector_get (s_euler->f, 1));
              printf ("status = %s\n", gsl_strerror (status));
              break;
            }
            status = gsl_multiroot_test_residual(s_euler->f, roottol); //tolerance --> roottol ~ 1e-15
          }
        while (status == GSL_CONTINUE && root_iter < 50);
        iter++;

        double s_trial = gsl_vector_get(s_euler->x, 0);
        double pzeta_trial = gsl_vector_get(s_euler->x, 1);

        // An unconverged root must not be used as though it were a completed
        // step. Raise rather than silently recording a stopping-criterion hit,
        // which would report a confined orbit as a spurious loss.
        bool solver_failed = (status != GSL_SUCCESS && status != GSL_CONTINUE);
        bool state_invalid = !std::isfinite(s_trial) || !std::isfinite(pzeta_trial);
        if (solver_failed || state_invalid) {
            gsl_multiroot_fsolver_free(s_euler);
            gsl_vector_free(xvec_quasi);
            std::string reason = state_invalid
                ? "the step left the domain of the field (s = " + std::to_string(s_trial) + ")"
                : "the root solve failed to converge (" + std::string(gsl_strerror(status))
                    + ") at s = " + std::to_string(s_trial);
            throw std::runtime_error(
                "Symplectic solver: " + reason + " at t = "
                + std::to_string(tau * f.tnorm) + " s. If s is small, roottol may be "
                "too tight to be achievable in double precision there; try loosening it.");
        }

        // Axis chart switch. A converged s < 0 means the orbit passed through
        // the magnetic axis during this step: (s, theta) and (-s, theta + pi)
        // are the same physical point, so commit the crossing by reflecting.
        // pzeta and hence vpar carry across unchanged, so the physical state
        // is continuous; ptheta is re-derived from the reflected state below.
        if (s_trial < 0.0) {
            s_trial = -s_trial;
            z[1] += M_PI;
        }

        z[0] = s_trial;      // s
        z[3] = pzeta_trial;  // pzeta

        // We now evaluate the explicit part of the time-step at [s, pzeta]
        // given by the Euler step.
        f.eval_field(z[0], z[1], z[2]);
        f.get_derivatives(z[3]);

        // z[1] = theta
        // z[2] = zeta
        // dH[0] = dH/dr
        // dptheta[0] = dptheta/dr
        // htheta = G/B
        // hzeta = I/B
        z[1] = z[1] + dtau*f.dH[0]/f.dptheta[0]; // (2.9) in JPP 2020
        z[2] = z[2] + dtau*(f.vpar * f.tnorm * f.vnorm - f.dH[0]/f.dptheta[0]*f.htheta)/f.hzeta; // (2.10) in JPP 2020

        // Translate z back to normalized y coordinates
        f.eval_field(z[0], z[1], z[2]);
        f.get_derivatives(z[3]);
        y[0] = z[0];
        y[1] = z[1];
        y[2] = z[2];
        y[3] = f.vpar;
        ptheta_old = f.ptheta;

        dense.update(tau, dtau, y, f); // tau_last = tau; tau_current = tau+dtau;

        if (predictor_step) {
            s_guess = z[0] + dtau*(-f.dH[1] + f.dptheta[3]*f.dH[2] - f.dptheta[2]*f.dH[3])/f.dptheta[0];
            pzeta_guess = z[3] + dtau*(- f.dH[2] + f.dH[0]*f.dptheta[2]/f.dptheta[0]); // corresponds with (2.7s) in JPP 2020
            // A guess with s < 0 is legitimate for an orbit approaching the
            // axis; only a non-finite guess is unusable.
            if (!std::isfinite(s_guess)) {
                s_guess = z[0];
            }
            if (!std::isfinite(pzeta_guess)) {
                pzeta_guess = z[3];
            }
        } else {
            s_guess = z[0];
            pzeta_guess = z[3];
        }

        tau += dtau;

        double tau_current = tau;

        stop = check_stopping_criteria(
            4,
            iter,
            res_hits,
            dense,
            tau_last,
            tau_current,
            dtau,
            abstol,
            phases,
            n_zetas,
            m_thetas,
            omegas,
            stopping_criteria,
            vpars,
            phases_stop,
            vpars_stop,
            0,
            f.vnorm,
            f.tnorm
        );

        // Save path if forget_exact_path = False
        if (forget_exact_path == 0 && !stop) {
            // This will give the first save point after tau_last
            double tau_save_last =  (std::floor(tau_last/dtau_save) + 1) * dtau_save;
            for (double tau_save = tau_save_last; tau_save <= std::min(tau_current, tau_max); tau_save += dtau_save) {
                if (tau_save != 0) { // tau = 0 is already saved.
                    dense.calc_state(tau_save, temp);
                    double t_save = tau_save * f.tnorm;
                    vector<double> stzvt(4);
                    y_to_stzvt(temp, stzvt, 0, f.vnorm, f.tnorm);
                    vector<double> save_point = {t_save};
                    save_point.insert(save_point.end(), stzvt.begin(), stzvt.end());
                    res.push_back(save_point);
                }
            }
        }

        tau_last = tau_current;
    } while(tau < tau_max && !stop);

    // Save tau = tau_max if we have not already saved it 
    if (stop) {
        tau_max = tau_last;
    }
    double t_max = tau_max * f.tnorm;
    if (t_max - res.back()[0] > 1e-15) {
        dense.calc_state(tau_max, y);
        vector<double> stzvt(4);
        y_to_stzvt(y, stzvt, 0, f.vnorm, f.tnorm);
        vector<double> final_state = {t_max};
        final_state.insert(final_state.end(), stzvt.begin(), stzvt.end());
        res.push_back(final_state);
    }

    gsl_multiroot_fsolver_free(s_euler);
    gsl_vector_free(xvec_quasi);

    return std::make_tuple(res, res_hits);
}
