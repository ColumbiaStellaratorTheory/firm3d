#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/functional.h"
namespace py = pybind11;
using std::shared_ptr;
using std::vector;
#include "tracing.h"
#include "tracing_helpers.h"
#ifdef USE_GSL
    #include "symplectic.h"
#endif

#ifdef USE_CUDA
extern "C" vector<double> cartesian_gpu_tracing(py::array_t<double> quad_pts, py::array_t<double> rrange,
        py::array_t<double> phirange, py::array_t<double> zrange, py::array_t<double> xyz_init, double m, double q, double vtotal, py::array_t<double> vtang, 
        double tmax, double tol, py::array_t<double> dt_in, int nparticles);

extern "C" vector<double> boozer_gpu_tracing(py::array_t<double> quad_pts, py::array_t<double> srange,
    py::array_t<double> trange, py::array_t<double> zrange, py::array_t<double> stz_init, double m, double q, double vtotal, py::array_t<double> vtang, 
    double tmax, double tol, py::array_t<double> dt_in, double psi0, int nparticles, bool vacuum=false);

extern "C" vector<double> boozer_saw_gpu_tracing(py::array_t<double> quad_pts, py::array_t<double> srange, py::array_t<double> trange, py::array_t<double> zrange, 
        double saw_omega, py::array_t<double> saw_srange, py::array_t<int> saw_m, py::array_t<int> saw_n, py::array_t<double> saw_phihats, int saw_nharmonics,
        py::array_t<double> stz_init, double m, double q, double vtotal, py::array_t<double> vtang, double tmax, double tol, py::array_t<double> dt_in, double psi0, int nparticles);

extern "C" vector<double> boozer_saw_nok_gpu_tracing(py::array_t<double> quad_pts, py::array_t<double> srange, py::array_t<double> trange, py::array_t<double> zrange, 
        double saw_omega, py::array_t<double> saw_srange, py::array_t<int> saw_m, py::array_t<int> saw_n, py::array_t<double> saw_phihats, int saw_nharmonics,
        py::array_t<double> stz_init, double m, double q, double vtotal, py::array_t<double> vtang, double tmax, double tol, py::array_t<double> dt_in, double psi0, int nparticles);

extern "C" py::array_t<double> test_gpu_interpolation(py::array_t<double> quad_pts, py::array_t<double> srange, py::array_t<double> trange, py::array_t<double> zrange, py::array_t<double> loc, std::string coordinates, int n_points);
extern "C" py::array_t<double> test_derivatives_cartesian(py::array_t<double> quad_pts, py::array_t<double> srange, py::array_t<double> trange, py::array_t<double> zrange, py::array_t<double> loc, py::array_t<double> vpar, double v_total, double m, double q,  int n_points);
extern "C" py::array_t<double> test_derivatives_boozer(py::array_t<double> quad_pts, py::array_t<double> srange, py::array_t<double> trange, py::array_t<double> zrange, py::array_t<double> loc, py::array_t<double> vpar, double v_total, double m, double q,  double psi0, int n_points, bool vacuum = false);
extern "C" py::array_t<double> test_derivatives_saw(py::array_t<double> quad_pts, py::array_t<double> x1_range, py::array_t<double> x2_range, py::array_t<double> x3_range, 
        double saw_omega, py::array_t<double> saw_srange, py::array_t<int> saw_m, py::array_t<int> saw_n, py::array_t<double> saw_phihats, int saw_nharmonics,
        py::array_t<double> loc, py::array_t<double> vpar, py::array_t<double> time, double v_total, double m, double q,  double psi0, int n_points);
extern "C" py::array_t<double> test_derivatives_saw_nok(py::array_t<double> quad_pts, py::array_t<double> x1_range, py::array_t<double> x2_range, py::array_t<double> x3_range, 
        double saw_omega, py::array_t<double> saw_srange, py::array_t<int> saw_m, py::array_t<int> saw_n, py::array_t<double> saw_phihats, int saw_nharmonics,
        py::array_t<double> loc, py::array_t<double> vpar, py::array_t<double> time, double v_total, double m, double q,  double psi0, int n_points);

extern "C" vector<double> test_timestep_cartesian(py::array_t<double> quad_pts, py::array_t<double> rrange,
        py::array_t<double> phirange, py::array_t<double> zrange, py::array_t<double> loc_init, double m, double q, double vtotal, py::array_t<double> vtang,
        double tol, int nparticles);

extern "C" vector<double> test_timestep_boozer(py::array_t<double> quad_pts, py::array_t<double> srange,
        py::array_t<double> trange, py::array_t<double> zrange, py::array_t<double> stz_init, double m, double q, double vtotal, py::array_t<double> vtang, 
        double tol, double psi0, int nparticles, bool vacuum);
        
extern "C" vector<double> test_timestep_saw(py::array_t<double> quad_pts, py::array_t<double> srange, py::array_t<double> trange, py::array_t<double> zrange, 
        double saw_omega, py::array_t<double> saw_srange, py::array_t<int> saw_m, py::array_t<int> saw_n, py::array_t<double> saw_phihats, int saw_nharmonics,
        py::array_t<double> stz_init, double m, double q, double vtotal, py::array_t<double> vtang, py::array_t<double> time,
        double tol, double psi0, int nparticles);


extern "C" vector<double> test_timestep_saw_nok(py::array_t<double> quad_pts, py::array_t<double> x1_range, py::array_t<double> x2_range, py::array_t<double> x3_range, 
        double saw_omega, py::array_t<double> saw_srange, py::array_t<int> saw_m, py::array_t<int> saw_n, py::array_t<double> saw_phihats, int saw_nharmonics,
        py::array_t<double> loc_init, double m, double q, double v_total, py::array_t<double> vtang, py::array_t<double> time,
        double tol, double psi0, int nparticles);
#endif

void init_tracing(py::module_ &m){
    py::class_<StoppingCriterion, shared_ptr<StoppingCriterion>>(m, "StoppingCriterion", py::module_local());
    py::class_<IterationStoppingCriterion, shared_ptr<IterationStoppingCriterion>, StoppingCriterion>(m, "IterationStoppingCriterion", py::module_local())
        .def(py::init<int>());
    py::class_<MaxToroidalFluxStoppingCriterion, shared_ptr<MaxToroidalFluxStoppingCriterion>, StoppingCriterion>(m, "MaxToroidalFluxStoppingCriterion", py::module_local())
        .def(py::init<double>());
    py::class_<MinToroidalFluxStoppingCriterion, shared_ptr<MinToroidalFluxStoppingCriterion>, StoppingCriterion>(m, "MinToroidalFluxStoppingCriterion", py::module_local())
        .def(py::init<double>());
    py::class_<ToroidalTransitStoppingCriterion, shared_ptr<ToroidalTransitStoppingCriterion>, StoppingCriterion>(m, "ToroidalTransitStoppingCriterion", py::module_local())
        .def(py::init<int>());
    py::class_<StepSizeStoppingCriterion, shared_ptr<StepSizeStoppingCriterion>, StoppingCriterion>(m, "StepSizeStoppingCriterion", py::module_local())
        .def(py::init<double>());

    m.def("particle_guiding_center_boozer_tracing", &particle_guiding_center_boozer_tracing,
        py::arg("field"),
        py::arg("stz_init"),
        py::arg("m"),
        py::arg("q"),
        py::arg("vtotal"),
        py::arg("vtang"),
        py::arg("tmax"),
        py::arg("vacuum"),
        py::arg("noK"),
        py::arg("phases")=vector<double>{},
        py::arg("n_zetas")=vector<double>{},
        py::arg("m_thetas")=vector<double>{},
        py::arg("omegas")=vector<double>{},
        py::arg("vpars")=vector<double>{},
        py::arg("stopping_criteria")=vector<shared_ptr<StoppingCriterion>>{},
        py::arg("dt_save")=1e-6,
        py::arg("forget_exact_path")=false,
        py::arg("phases_stop")=false,
        py::arg("vpars_stop")=false,
        py::arg("axis")=0,
        py::arg("abstol")=1e-9,
        py::arg("reltol")=1e-9,
        py::arg("ODE_solver")="boost",
        py::arg("predictor_step")=true,
        py::arg("roottol")=1e-9,
        py::arg("dt")=1e-7,
        py::arg("DP_hmin")=0.0
        );

    m.def("particle_guiding_center_boozer_perturbed_tracing", &particle_guiding_center_boozer_perturbed_tracing,
        py::arg("pertrurbed_field"),
        py::arg("stz_init"),
        py::arg("m"),
        py::arg("q"),
        py::arg("vtotal"),
        py::arg("vtang"),
        py::arg("mu"),
        py::arg("tmax"),
        py::arg("abstol"),
        py::arg("reltol"),
        py::arg("vacuum"),
        py::arg("noK"),
        py::arg("phases")=vector<double>{},
        py::arg("n_zetas")=vector<double>{},
        py::arg("m_thetas")=vector<double>{},
        py::arg("omegas")=vector<double>{},
        py::arg("stopping_criteria")=vector<shared_ptr<StoppingCriterion>>{},
        py::arg("dt_save")=1e-6,
        py::arg("phases_stop")=false,
        py::arg("vpars_stop")=false,
        py::arg("forget_exact_path")=false,
        py::arg("axis")=0,
        py::arg("vpars")=vector<double>{},
        py::arg("ODE_solver")="boost",
        py::arg("DP_hmin")=0.0
    );

#ifdef USE_CUDA
    m.def("cartesian_gpu_tracing", &cartesian_gpu_tracing,
        py::arg("quad_pts"),
        py::arg("rrange"),
        py::arg("phirange"),
        py::arg("zrange"),
        py::arg("xyz_init"),
        py::arg("m"),
        py::arg("q"),
        py::arg("vtotal"),
        py::arg("vtang"),
        py::arg("tmax"),
        py::arg("tol"),
        py::arg("dt_in"),
        py::arg("nparticles")
        );

    
    m.def("boozer_gpu_tracing", &boozer_gpu_tracing,
        py::arg("quad_pts"),
        py::arg("srange"),
        py::arg("trange"),
        py::arg("zrange"),
        py::arg("stz_init"),
        py::arg("m"),
        py::arg("q"),
        py::arg("vtotal"),
        py::arg("vtang"),
        py::arg("tmax"),
        py::arg("tol"),
        py::arg("dt_in"),
        py::arg("psi0"),
        py::arg("nparticles"),
        py::arg("vacuum") = false
        );


    m.def("boozer_saw_gpu_tracing", &boozer_saw_gpu_tracing,
        py::arg("quad_pts"),
        py::arg("srange"),
        py::arg("trange"),
        py::arg("zrange"),
        py::arg("saw_omega"),
        py::arg("saw_srange"),
        py::arg("saw_m"),
        py::arg("saw_n"),
        py::arg("saw_phihats"),
        py::arg("saw_nharmonics"),
        py::arg("stz_init"),
        py::arg("m"),
        py::arg("q"),
        py::arg("vtotal"),
        py::arg("vtang"),
        py::arg("tmax"),
        py::arg("tol"),
        py::arg("dt_in"),
        py::arg("psi0"),
        py::arg("nparticles")
        );

        m.def("boozer_saw_nok_gpu_tracing", &boozer_saw_nok_gpu_tracing,
        py::arg("quad_pts"),
        py::arg("srange"),
        py::arg("trange"),
        py::arg("zrange"),
        py::arg("saw_omega"),
        py::arg("saw_srange"),
        py::arg("saw_m"),
        py::arg("saw_n"),
        py::arg("saw_phihats"),
        py::arg("saw_nharmonics"),
        py::arg("stz_init"),
        py::arg("m"),
        py::arg("q"),
        py::arg("vtotal"),
        py::arg("vtang"),
        py::arg("tmax"),
        py::arg("tol"),
        py::arg("dt_in"),
        py::arg("psi0"),
        py::arg("nparticles")
        );

    m.def("test_gpu_interpolation", &test_gpu_interpolation,
        py::arg("quad_pts"),
        py::arg("srange"),
        py::arg("trange"),
        py::arg("zrange"),
        py::arg("loc"),
        py::arg("coordinates"),
        py::arg("n_points")
        );


    m.def("test_derivatives_cartesian", &test_derivatives_cartesian,
        py::arg("quad_pts"),
        py::arg("srange"),
        py::arg("trange"),
        py::arg("zrange"),
        py::arg("loc"),
        py::arg("vpar"),
        py::arg("v_total"),
        py::arg("m"),
        py::arg("q"),
        py::arg("n_points")
        );

    m.def("test_derivatives_boozer", &test_derivatives_boozer,
        py::arg("quad_pts"),
        py::arg("srange"),
        py::arg("trange"),
        py::arg("zrange"),
        py::arg("loc"),
        py::arg("vpar"),
        py::arg("v_total"),
        py::arg("m"),
        py::arg("q"),
        py::arg("psi0"),
        py::arg("n_points"),
        py::arg("vacuum") = false
    );

    m.def("test_derivatives_saw", &test_derivatives_saw,
        py::arg("quad_pts"),
        py::arg("srange"),
        py::arg("trange"),
        py::arg("zrange"),
        py::arg("saw_omega"),
        py::arg("saw_srange"),
        py::arg("saw_m"),
        py::arg("saw_n"),
        py::arg("saw_phihats"),
        py::arg("saw_nharmonics"),
        py::arg("loc"),
        py::arg("vpar"),
        py::arg("time"),
        py::arg("v_total"),
        py::arg("m"),
        py::arg("q"),
        py::arg("psi0"),
        py::arg("n_points")
    );

        m.def("test_derivatives_saw_nok", &test_derivatives_saw_nok,
        py::arg("quad_pts"),
        py::arg("srange"),
        py::arg("trange"),
        py::arg("zrange"),
        py::arg("saw_omega"),
        py::arg("saw_srange"),
        py::arg("saw_m"),
        py::arg("saw_n"),
        py::arg("saw_phihats"),
        py::arg("saw_nharmonics"),
        py::arg("loc"),
        py::arg("vpar"),
        py::arg("time"),
        py::arg("v_total"),
        py::arg("m"),
        py::arg("q"),
        py::arg("psi0"),
        py::arg("n_points")
    );

    m.def("simsopt_derivs_boozer", &simsopt_derivs_boozer,
        py::arg("field"),
        py::arg("loc"),
        py::arg("m"),
        py::arg("q"),
        py::arg("vtotal"),
        py::arg("vtang"),
        py::arg("vacuum")
        );
    
    m.def("simsopt_derivs_saw", &simsopt_derivs_saw,
        py::arg("perturbed_field"),
        py::arg("loc"),
        py::arg("m"),
        py::arg("q"),
        py::arg("vtotal"),
        py::arg("vtang"),
        py::arg("time"),
        py::arg("rhs")    
        );


    m.def("test_timestep_cartesian", &test_timestep_cartesian,
        py::arg("quad_pts"),
        py::arg("rrange"),
        py::arg("phirange"),
        py::arg("zrange"),
        py::arg("loc_init"),
        py::arg("m"),
        py::arg("q"),
        py::arg("vtotal"),
        py::arg("vtang"),
        py::arg("tol"),
        py::arg("nparticles")
        );

    m.def("test_timestep_boozer", &test_timestep_boozer,
        py::arg("quad_pts"),
        py::arg("srange"),
        py::arg("trange"),
        py::arg("zrange"),
        py::arg("stz_init"),
        py::arg("m"),
        py::arg("q"),
        py::arg("vtotal"),
        py::arg("vtang"),
        py::arg("tol"),
        py::arg("psi0"),
        py::arg("nparticles"),
        py::arg("vacuum")=false
        );

    m.def("test_timestep_saw", &test_timestep_saw,
        py::arg("quad_pts"),
        py::arg("srange"),
        py::arg("trange"),
        py::arg("zrange"),
        py::arg("saw_omega"),
        py::arg("saw_srange"),
        py::arg("saw_m"),
        py::arg("saw_n"),
        py::arg("saw_phihats"),
        py::arg("saw_nharmonics"),
        py::arg("stz_init"),
        py::arg("m"),
        py::arg("q"),
        py::arg("vtotal"),
        py::arg("vtang"),
        py::arg("time"),
        py::arg("tol"),
        py::arg("psi0"),
        py::arg("nparticles")
        );

        m.def("test_timestep_saw_nok", &test_timestep_saw_nok,
        py::arg("quad_pts"),
        py::arg("srange"),
        py::arg("trange"),
        py::arg("zrange"),
        py::arg("saw_omega"),
        py::arg("saw_srange"),
        py::arg("saw_m"),
        py::arg("saw_n"),
        py::arg("saw_phihats"),
        py::arg("saw_nharmonics"),
        py::arg("stz_init"),
        py::arg("m"),
        py::arg("q"),
        py::arg("vtotal"),
        py::arg("vtang"),
        py::arg("time"),
        py::arg("tol"),
        py::arg("psi0"),
        py::arg("nparticles")
        );
#endif


}
