#pragma once

#include <boost/math/tools/roots.hpp>
#include "tracing_helpers.h"
#include <array>
#include <vector>
#include <tuple>
#include <memory>
#include <functional>
#include <iostream>

using std::array;
using std::shared_ptr;
using std::vector;
using std::tuple;

class StoppingCriterion {
    public:
        // Should return true if the Criterion is satisfied.
        virtual bool operator()(int iter, double dt, double t, double x, double y, double z, double vpar=0) = 0;
        virtual ~StoppingCriterion() {}
};

class ZetaStoppingCriterion : public StoppingCriterion {
    private:
        int nfp;
    public:
        ZetaStoppingCriterion(int nfp) : nfp(nfp) {
        };
        bool operator()(int iter, double dt, double t, double s, double theta, double zeta, double vpar=0) override {
            return std::abs(zeta)>=2*M_PI/nfp;
        };
};

class VparStoppingCriterion : public StoppingCriterion {
    private:
        double vpar_crit;
    public:
        VparStoppingCriterion(double vpar_crit) : vpar_crit(vpar_crit) {
        };
        bool operator()(int iter, double dt, double t, double x, double y, double z, double vpar) override {
            return std::abs(vpar)<=vpar_crit;
        };
};

class ToroidalTransitStoppingCriterion : public StoppingCriterion {
    private:
        int max_transits;
        double zeta_last;
        double zeta_init;
    public:
        ToroidalTransitStoppingCriterion(int max_transits) : max_transits(max_transits) {
        };
        bool operator()(int iter, double dt, double t, double s, double theta, double zeta, double vpar=0) override {
            if (iter == 1) {
              zeta_last = M_PI;
            }
            double this_zeta = zeta;
            if (iter == 1) {
              zeta_init = this_zeta;
            }
            zeta_last = this_zeta;
            int ntransits = std::abs(std::floor((this_zeta-zeta_init)/(2*M_PI)));
            return ntransits>=max_transits;
        };
};

class MaxToroidalFluxStoppingCriterion : public StoppingCriterion {
    private:
        double max_s;
    public:
        MaxToroidalFluxStoppingCriterion(double max_s) : max_s(max_s) {};
        bool operator()(int iter, double dt, double t, double s, double theta, double zeta, double vpar=0) override {
            return s>=max_s;
        };
};

class MinToroidalFluxStoppingCriterion : public StoppingCriterion {
    private:
        double min_s;
    public:
        MinToroidalFluxStoppingCriterion(double min_s) : min_s(min_s) {};
        bool operator()(int iter, double dt, double t, double s, double theta, double zeta, double vpar=0) override {
            return s<=min_s;
        };
};

class IterationStoppingCriterion : public StoppingCriterion {
    private:
        int max_iter;
    public:
        IterationStoppingCriterion(int max_iter) : max_iter(max_iter) {};
        bool operator()(int iter, double dt, double t, double s, double theta, double zeta, double vpar=0) override {
            return iter>max_iter;
        };
};

class StepSizeStoppingCriterion : public StoppingCriterion {
    private:
        int min_dt;
    public:
        StepSizeStoppingCriterion(int min_dt) : min_dt(min_dt) {};
        bool operator()(int iter, double dt, double t, double s, double theta, double zeta, double vpar=0) override {
            return dt<min_dt;
        };
};

template<std::size_t m, std::size_t n>
array<double, m+n> join(const array<double, m>& a, const array<double, n>& b)
{
     array<double, m+n> res;
     for (int i = 0; i < m; ++i) {
         res[i] = a[i];
     }
     for (int i = 0; i < n; ++i) {
         res[i+m] = b[i];
     }
     return res;
}

inline void stzvt_to_y(const vector<double>& stzvt, vector<double>& y, int axis, double vnorm, double tnorm)
{
    if (y.size() != 4 && y.size() != 5) {
        throw std::invalid_argument("y must have size 4 or 5.");
    }
    if (stzvt.size() != y.size()) {
        throw std::invalid_argument("stzvt must have the same size as y.");
    }
    double s, theta;
    if (axis == 1) {
        y[0] = sqrt(stzvt[0]) * cos(stzvt[1]);
        y[1] = sqrt(stzvt[0]) * sin(stzvt[1]);
    } else if (axis == 2) {
        y[0] = stzvt[0] * cos(stzvt[1]);
        y[1] = stzvt[0] * sin(stzvt[1]);
    } else if (axis == 0) {
        y[0] = stzvt[0];
        y[1] = stzvt[1];
    } else {
        throw std::invalid_argument("axis must be 0, 1, or 2.");
    }
    y[2] = stzvt[2];
    y[3] = stzvt[3] / vnorm; // velocity normalization
    if (y.size() == 5) {
        y[4] = stzvt[4] / tnorm; // time normalization
    }
}

inline void y_to_stzvt(const vector<double>& y, vector<double>& stzvt, int axis, double vnorm, double tnorm)
{
    if (y.size() != 4 && y.size() != 5) {
        throw std::invalid_argument("y must have size 4 or 5.");
    }
    if (stzvt.size() != y.size()) {
        throw std::invalid_argument("stzvt must have the same size as y.");
    }
    double s, theta;
    if (axis == 1) {
        s = pow(y[0], 2) + pow(y[1], 2);
        theta = atan2(y[1], y[0]);
    } else if (axis == 2) {
        s = sqrt(pow(y[0], 2) + pow(y[1], 2));
        theta = atan2(y[1], y[0]);
    } else if (axis == 0) {
        s = y[0];
        theta = y[1];
    } else {
        throw std::invalid_argument("axis must be 0, 1, or 2.");
    }
    stzvt[0] = s;
    stzvt[1] = theta;
    stzvt[2] = y[2];
    stzvt[3] = y[3] * vnorm; // velocity normalization
    if (y.size() == 5) {
        stzvt[4] = y[4] * tnorm; // time normalization
    }
}

inline void stzvtdot_to_ydot(const vector<double>& stzvtdot, const vector<double>& stzvt, vector<double>& ydot, int axis, double vnorm, double tnorm)
{
    if (stzvtdot.size() != 4 && stzvtdot.size() != 5) {
        throw std::invalid_argument("stzvtdot must have size 4 or 5.");
    }
    if (stzvt.size() != ydot.size() && stzvtdot.size() != ydot.size()) {
        throw std::invalid_argument("stzvtdot, stzvt, and ydot must have the same size.");
    }
    double sdot = stzvtdot[0];
    double tdot = stzvtdot[1];
    double s = stzvt[0];
    double theta = stzvt[1];
    if (axis==1) {
        ydot[0] = sdot*cos(theta)/(2*sqrt(s)) - sqrt(s) * sin(theta) * tdot;
        ydot[1] = sdot*sin(theta)/(2*sqrt(s)) + sqrt(s) * cos(theta) * tdot;
    } else if (axis==2) {
        ydot[0] = sdot*cos(theta) - s * sin(theta) * tdot;
        ydot[1] = sdot*sin(theta) + s * cos(theta) * tdot;
    } else if (axis==0) {
        ydot[0] = sdot;
        ydot[1] = tdot;
    } else {
        throw std::invalid_argument("axis must be 0, 1, or 2.");
    }
    ydot[0] = ydot[0] * tnorm;
    ydot[1] = ydot[1] * tnorm;
    ydot[2] = stzvtdot[2] * tnorm;
    ydot[3] = stzvtdot[3] * tnorm / vnorm;

    if (stzvtdot.size() == 5) {
        ydot[4] = 1;
    }
}

// Vector-based version of check_stopping_criteria
template<class DENSE>
bool check_stopping_criteria(
    int state_size,
    int iter,
    vector<vector<double>> &res_hits,
    DENSE& dense,
    double tau_last,
    double tau_current,
    double dtau,
    double abstol,
    vector<double> phases,
    vector<double> n_zetas,
    vector<double> m_thetas,
    vector<double> omegas,
    vector<shared_ptr<StoppingCriterion>> stopping_criteria,
    vector<double> vpars,
    bool phases_stop,
    bool vpars_stop,
    int axis,
    double vnorm,
    double tnorm
)
{
    boost::math::tools::eps_tolerance<double> roottol(-int(std::log2(abstol)));
    uintmax_t rootmaxit = 200;
    vector<double> y(state_size), stzvt(state_size), stzvt_current(state_size);

    bool stop = false;
    vector<double> ykeep(state_size, 0.0);

    double dt = dtau * tnorm;

    dense.calc_state(tau_last, y);
    y_to_stzvt(y, stzvt, axis, vnorm, tnorm);
    double t_last = tau_last * tnorm;
    double theta_last = stzvt[1];
    double zeta_last = stzvt[2];
    double vpar_last = stzvt[3];

    dense.calc_state(tau_current, y);
    y_to_stzvt(y, stzvt_current, axis, vnorm, tnorm);
    double t_current = tau_current * tnorm;
    double s_current = stzvt_current[0];
    double theta_current = stzvt_current[1];
    double zeta_current = stzvt_current[2];
    double vpar_current = stzvt_current[3];

    // Now check whether we have hit any of the vpar planes
    for (int i = 0; i < vpars.size(); ++i) {
        double vpar = vpars[i];
        if((vpar_last-vpar != 0) && (vpar_current-vpar != 0) && (((vpar_last-vpar > 0) ? 1 : ((vpar_last-vpar < 0) ? -1 : 0)) != ((vpar_current-vpar > 0) ? 1 : ((vpar_current-vpar < 0) ? -1 : 0)))){ // check whether vpar = vpars[i] was crossed
            std::function<double(double)> rootfun = [&dense, &y, &vpar_last, &vpar, &stzvt, &axis, &vnorm, &tnorm](double tau){
                dense.calc_state(tau, y);
                y_to_stzvt(y, stzvt, axis, vnorm, tnorm);
                if (vpar == 0) {
                    return (stzvt[3]-vpar);
                } else {
                    // Normalize by vpar since it can be large in magnitude
                    return (stzvt[3]-vpar)/vpar;
                }
            };
            auto root = toms748_solve(rootfun, tau_last, tau_current, vpar_last-vpar, vpar_current-vpar, roottol, rootmaxit);
            double f0 = rootfun(root.first);
            double f1 = rootfun(root.second);
            double tau_root = std::abs(f0) < std::abs(f1) ? root.first : root.second;
            double t_root = tau_root * tnorm;
            dense.calc_state(tau_root, y);
            y_to_stzvt(y, stzvt, axis, vnorm, tnorm);
            vector<double> hit_state = {t_root, double(i) + n_zetas.size()};
            hit_state.insert(hit_state.end(), stzvt.begin(), stzvt.end());
            res_hits.push_back(hit_state);
            if (vpars_stop) {
                stop = true;
                break;
            }
        }
    }
    
    assert(n_zetas.size() == m_thetas.size());
    assert(n_zetas.size() == omegas.size());
    assert(n_zetas.size() == phases.size());
    
    /// Now check whether we have hit any of the phase planes:
    //  n*zeta + m*theta - omega*t = consts
    for (int i = 0; i < n_zetas.size(); ++i) {
        double phase = phases[i];
        double nz = n_zetas[i];
        double mt = m_thetas[i];
        double omega = omegas[i];
        double phase_last = nz * zeta_last + mt * theta_last - omega*t_last;
        double phase_current = nz * zeta_current + mt * theta_current - omega*t_current;
        
        if((std::floor((phase_last-phase)/(2*M_PI)) != std::floor((phase_current-phase)/(2*M_PI))) && (phase_current != phase) && (phase_last != phase)) { // check whether phase+k*2pi for some k was crossed
            int fak = std::round(((phase_last+phase_current)/2-phase)/(2*M_PI));
            double phase_shift = fak*2*M_PI + phase;
            assert((phase_last <= phase_shift && phase_shift <= phase_current) || (phase_current <= phase_shift && phase_shift <= phase_last));
            
            std::function<double(double)> rootfun = [&phase_shift, &nz, &mt, &omega, &dense, &y, &stzvt, &axis, &vnorm, &tnorm](double tau){
                dense.calc_state(tau, y);
                double t = tau * tnorm;
                y_to_stzvt(y, stzvt, axis, vnorm, tnorm);
                return nz * stzvt[2] + mt * stzvt[1] - omega*t - phase_shift;
            };
            auto root = toms748_solve(rootfun, tau_last, tau_current, phase_last - phase_shift, phase_current - phase_shift, roottol, rootmaxit);
            double f0 = rootfun(root.first);
            double f1 = rootfun(root.second);
            double tau_root = std::abs(f0) < std::abs(f1) ? root.first : root.second;
            double t_root = tau_root * tnorm;
            dense.calc_state(tau_root, y);
            y_to_stzvt(y, stzvt, axis, vnorm, tnorm);
            vector<double> hit_state = {t_root, double(i)};
            hit_state.insert(hit_state.end(), stzvt.begin(), stzvt.end());
            res_hits.push_back(hit_state);
            if (phases_stop && !stop) {
                stop = true;
                break;
            }
        }
    }
    
    // check whether we have satisfied any of the extra stopping criteria (e.g. left a surface)
    for (int i = 0; i < stopping_criteria.size(); ++i) {
        if(stopping_criteria[i] && (*stopping_criteria[i])(iter, dt, t_current, s_current, theta_current, zeta_current, vpar_current)){
            stop = true;
            vector<double> hit_state = {t_current, -1-double(i)};
            hit_state.insert(hit_state.end(), stzvt_current.begin(), stzvt_current.end());
            res_hits.push_back(hit_state);
            break;
        }
    }

    return stop;
}
