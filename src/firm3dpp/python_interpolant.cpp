#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/functional.h"
#include "xtensor-python/pytensor.hpp"     // Numpy bindings

typedef xt::pytensor<double, 2, xt::layout_type::row_major> Array2;
using std::shared_ptr;
using std::vector;

namespace py = pybind11;
#include "regular_grid_interpolant_3d.h"
#include "mpi_utils.h"

#ifdef USE_MPI
using firm3dpp::mpi::get_mpi_comm_from_fortran;

// Wrapper function for MPI version of interpolate_batch
void interpolate_batch_mpi(RegularGridInterpolant3D<Array2>& self, 
                           std::function<Vec(Vec, Vec, Vec)>& f, 
                           long long fortran_handle) {
    MPI_Comm comm = get_mpi_comm_from_fortran(fortran_handle);
    self.interpolate_batch(f, comm);
}
#endif

void init_interpolant(py::module_ &m){

    py::class_<InterpolationRule, shared_ptr<InterpolationRule>>(m, "InterpolationRule", py::module_local(), "Abstract class for interpolation rules on an interval.")
        .def_readonly("degree", &InterpolationRule::degree, "The degree of the polynomial. The number of interpolation points is `degree+1`.")
        // nodes and scalings exposed for InterpolatedBoozerField serialization
        .def_readonly("nodes", &InterpolationRule::nodes, "The interpolation nodes within each cell.")
        .def_readonly("scalings", &InterpolationRule::scalings, "The scaling factors for interpolation weights.");

    py::class_<UniformInterpolationRule, shared_ptr<UniformInterpolationRule>, InterpolationRule>(m, "UniformInterpolationRule", py::module_local(), "Polynomial interpolation using equispaced points.")
        .def(py::init<int>())
        .def_readonly("degree", &UniformInterpolationRule::degree, "The degree of the polynomial. The number of interpolation points is `degree+1`.");
    py::class_<ChebyshevInterpolationRule, shared_ptr<ChebyshevInterpolationRule>, InterpolationRule>(m, "ChebyshevInterpolationRule", py::module_local(), "Polynomial interpolation using Chebyshev points.")
        .def(py::init<int>())
        .def_readonly("degree", &ChebyshevInterpolationRule::degree, "The degree of the polynomial. The number of interpolation points is `degree+1`.");

    py::class_<RegularGridInterpolant3D<Array2>, shared_ptr<RegularGridInterpolant3D<Array2>>>(m, "RegularGridInterpolant3D", py::module_local(),
            R"pbdoc(
            Interpolates a (vector valued) function on a uniform grid.
            This interpolant is optimized for fast function evaluation (at the cost of memory usage). The main purpose of this class is to be used to interpolate magnetic fields and then use the interpolant for tasks such as fieldline or particle tracing for which the field needs to be evaluated many many times.
            )pbdoc")
        .def(py::init<InterpolationRule, RangeTriplet, RangeTriplet, RangeTriplet, int, bool, std::function<std::vector<bool>(Vec, Vec, Vec)>>())
        .def(py::init<InterpolationRule, RangeTriplet, RangeTriplet, RangeTriplet, int, bool>())
        .def("interpolate_batch", 
             py::overload_cast<std::function<Vec(Vec, Vec, Vec)>&>(&RegularGridInterpolant3D<Array2>::interpolate_batch),
             "Interpolate a function by evaluating the function on all interpolation nodes simultanuously.")
#ifdef USE_MPI
        .def("interpolate_batch",
             [](RegularGridInterpolant3D<Array2>& self, std::function<Vec(Vec, Vec, Vec)>& f, long long fortran_handle) {
                 interpolate_batch_mpi(self, f, fortran_handle);
             },
             py::arg("f"), py::arg("comm_fortran"),
             "Interpolate a function with MPI parallelization. 'comm_fortran' should be a Fortran MPI communicator handle (obtained from comm.py2f() in Python).")
#endif
        .def("evaluate", &RegularGridInterpolant3D<Array2>::evaluate, "Evaluate the interpolant at a point.")
        .def("evaluate_batch", &RegularGridInterpolant3D<Array2>::evaluate_batch, "Evaluate the interpolant at multiple points (faster than `evaluate` as it uses prefetching).")
        // Serialization for InterpolatedBoozerField save/load
        .def("get_interpolant_data", &RegularGridInterpolant3D<Array2>::get_interpolant_data)
        .def("set_interpolant_data", &RegularGridInterpolant3D<Array2>::set_interpolant_data);
}
