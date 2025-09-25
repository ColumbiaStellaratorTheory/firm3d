#pragma once
#include <memory>
#include <vector>
#include "boozermagneticfield.h"
#include "shearalfvenwave.h"
#include "regular_grid_interpolant_3d.h"
#include "tracing_helpers.h"

using std::array;
using std::shared_ptr;
using std::vector;
using std::tuple;
using std::function;

// Base class for RHS functions
class BaseRHS {
public:
    virtual ~BaseRHS() = default;
    virtual void operator()(const vector<double>& y, vector<double>& dydt, double t) = 0;
    virtual int get_state_size() const = 0;
};

// Overloaded solve() function that accepts a BaseRHS object
tuple<vector<vector<double>>, vector<vector<double>>>
solve(
    BaseRHS& rhs,
    vector<double> stzvt,
    double tau_max,
    double dtau,
    double dtau_max,
    double abstol,
    double reltol, 
    vector<double> phases,
    vector<double> n_zetas,
    vector<double> m_thetas,
    vector<double> omegas,
    vector<shared_ptr<StoppingCriterion>> stopping_criteria,
    double dtau_save,
    vector<double> vpars,
    bool phases_stop=false,
    bool vpars_stop=false,
    bool forget_exact_path=false,
    int axis=0,
    double vnorm=1,
    double tnorm=1,
    string ode_solver="boost",
    double DP_hmin=0.0
);

tuple<vector<vector<double>>, vector<vector<double>>>
particle_guiding_center_boozer_perturbed_tracing(
        shared_ptr<ShearAlfvenWave> perturbed_field,
        vector<double> stz_init,
        double m,
        double q,
        double vtotal,
        double vtang,
        double mu,
        double tmax,
        double abstol,
        double reltol,
        bool vacuum,
        bool noK,
        vector<double> phases,
        vector<double> n_zetas,
        vector<double> m_thetas,
        vector<double> omegas,
        vector<shared_ptr<StoppingCriterion>> stopping_criteria,
        double dt_save=1e-6,
        bool phases_stop=false,
        bool vpars_stop=false,
        bool forget_exact_path=false,
        int axis=0,
        vector<double> vpars={},
        string ode_solver="boost",
        double DP_hmin=0.0
);


tuple<vector<vector<double>>, vector<vector<double>>>
particle_guiding_center_boozer_tracing(
        shared_ptr<BoozerMagneticField> field,
        vector<double> stz_init,
        double m,
        double q,
        double vtotal,
        double vtang,
        double tmax,
        bool vacuum,
        bool noK,
        vector<double> phases={},
        vector<double> n_zetas={},
        vector<double> m_thetas={},
        vector<double> omegas={},
        vector<double> vpars={},
        vector<shared_ptr<StoppingCriterion>> stopping_criteria={},
        double dt_save=1e-6,
        bool forget_exact_path=false,
        bool phases_stop=false,
        bool vpars_stop=false,
        int axis=0,
        double abstol=1e-9,
        double reltol=1e-9,
        string ode_solver="boost",
        bool predictor_step=true,
        double roottol=1e-9,
        double dt=1e-7,
        double DP_hmin=0.0
);
