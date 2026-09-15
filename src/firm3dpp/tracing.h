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

namespace py = pybind11;


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

vector<double> simsopt_derivs_boozer(shared_ptr<BoozerMagneticField> field, vector<double> loc, double m, double q, double vtotal, double vtang, bool vacuum);
vector<double> simsopt_derivs_saw(shared_ptr<ShearAlfvenWave> perturbed_field, vector<double> loc, double m, double q, double vtotal, double vtang, double time, std::string rhs);

#ifdef USE_CUDA
template<typename T>
vector<T> cartesian_gpu_tracing(py::array_t<T> quad_pts, py::array_t<double> rrange,
        py::array_t<double> phirange, py::array_t<double> zrange, py::array_t<T> xyz_init, double m, double q, double vtotal, py::array_t<T> vtang, 
        py::array_t<double> tmax, double tol, py::array_t<T> dt_in, py::array_t<T> mu_in, int nparticles);

template<typename T>
vector<T> boozer_gpu_tracing(py::array_t<T> quad_pts, py::array_t<double> srange,
        py::array_t<double> trange, py::array_t<double> zrange, py::array_t<T> stz_init, double m, double q, double vtotal, py::array_t<T> vtang, 
        py::array_t<double> tmax, double tol, py::array_t<T> dt_in, py::array_t<T> mu_in, double psi0, int nparticles, bool vacuum);
        
template<typename T>
vector<T> boozer_saw_gpu_tracing(py::array_t<T> quad_pts, py::array_t<double> srange, py::array_t<double> trange, py::array_t<double> zrange, 
        double saw_omega, py::array_t<double> saw_srange, py::array_t<int> saw_m, py::array_t<int> saw_n, py::array_t<T> saw_phihats, int saw_nharmonics,
        py::array_t<T> stz_init, double m, double q, double vtotal, py::array_t<T> vtang, py::array_t<double> tmax, double tol, py::array_t<T> dt_in, py::array_t<T> mu_in, double psi0, int nparticles);

template<typename T>
vector<T> boozer_saw_nok_gpu_tracing(py::array_t<T> quad_pts, py::array_t<double> srange, py::array_t<double> trange, py::array_t<double> zrange, 
        double saw_omega, py::array_t<double> saw_srange, py::array_t<int> saw_m, py::array_t<int> saw_n, py::array_t<T> saw_phihats, int saw_nharmonics,
        py::array_t<T> stz_init, double m, double q, double vtotal, py::array_t<T> vtang, py::array_t<double> tmax, double tol, py::array_t<T> dt_in, py::array_t<T> mu_in, double psi0, int nparticles);

template<typename T>
py::array_t<T> test_gpu_interpolation(py::array_t<T> quad_pts, py::array_t<double> srange, py::array_t<double> trange, py::array_t<double> zrange, py::array_t<T> loc, std::string coordinates, int n_points);

template<typename T>
py::array_t<T> test_derivatives_cartesian(py::array_t<T> quad_pts, py::array_t<double> x1_range, py::array_t<double> x2_range, py::array_t<double> x3_range, py::array_t<T> loc, py::array_t<T> vpar, double v_total, double m, double q, int n_points);

py::array_t<double> test_derivatives_boozer(py::array_t<double> quad_pts, py::array_t<double> srange, py::array_t<double> trange, py::array_t<double> zrange, py::array_t<double> loc, py::array_t<double> vpar, double v_total, double m, double q,  double psi0, int n_points, bool vacuum = false);
py::array_t<double> test_derivatives_saw(py::array_t<double> quad_pts, py::array_t<double> x1_range, py::array_t<double> x2_range, py::array_t<double> x3_range, 
        double saw_omega, py::array_t<double> saw_srange, py::array_t<int> saw_m, py::array_t<int> saw_n, py::array_t<double> saw_phihats, int saw_nharmonics,
        py::array_t<double> loc, py::array_t<double> vpar, py::array_t<double> time, double v_total, double m, double q,  double psi0, int n_points);
py::array_t<double> test_derivatives_saw_nok(py::array_t<double> quad_pts, py::array_t<double> x1_range, py::array_t<double> x2_range, py::array_t<double> x3_range, 
        double saw_omega, py::array_t<double> saw_srange, py::array_t<int> saw_m, py::array_t<int> saw_n, py::array_t<double> saw_phihats, int saw_nharmonics,
        py::array_t<double> loc, py::array_t<double> vpar, py::array_t<double> time, double v_total, double m, double q,  double psi0, int n_points);

vector<double> test_timestep_cartesian(py::array_t<double> quad_pts, py::array_t<double> rrange,
        py::array_t<double> phirange, py::array_t<double> zrange, py::array_t<double> loc_init, double m, double q, double vtotal, py::array_t<double> vtang,
        double tol, int nparticles);

vector<double> test_timestep_boozer(py::array_t<double> quad_pts, py::array_t<double> srange,
        py::array_t<double> trange, py::array_t<double> zrange, py::array_t<double> stz_init, double m, double q, double vtotal, py::array_t<double> vtang, 
        double tol, double psi0, int nparticles, bool vacuum);
        
vector<double> test_timestep_saw(py::array_t<double> quad_pts, py::array_t<double> srange, py::array_t<double> trange, py::array_t<double> zrange, 
        double saw_omega, py::array_t<double> saw_srange, py::array_t<int> saw_m, py::array_t<int> saw_n, py::array_t<double> saw_phihats, int saw_nharmonics,
        py::array_t<double> stz_init, double m, double q, double vtotal, py::array_t<double> vtang, py::array_t<double> time,
        double tol, double psi0, int nparticles);


vector<double> test_timestep_saw_nok(py::array_t<double> quad_pts, py::array_t<double> x1_range, py::array_t<double> x2_range, py::array_t<double> x3_range, 
        double saw_omega, py::array_t<double> saw_srange, py::array_t<int> saw_m, py::array_t<int> saw_n, py::array_t<double> saw_phihats, int saw_nharmonics,
        py::array_t<double> loc_init, double m, double q, double v_total, py::array_t<double> vtang, py::array_t<double> time,
        double tol, double psi0, int nparticles);
#endif