#include <cmath>
#include <iostream>
#include <stdexcept>

#if __has_include(<boost/numeric/odeint.hpp>)
#include "ode_solvers.h"

class TestRHS : public BaseRHS {
public:
    double rate;
    explicit TestRHS(double rate = 1) : rate(rate) {}
    void operator()(const vector<double>& y, vector<double>& dy, double t) override {
        dy[0] = rate * y[0] + std::sin(t);
        dy[1] = -0.3 * y[1] + std::cos(2*t);
    }
    int get_state_size() const override { return 2; }
};

void check(bool condition, const char* message) {
    if(!condition) throw std::runtime_error(message);
}
void close(double actual, double expected, double tol = 3e-14) {
    check(std::isfinite(actual) && std::isfinite(expected), "Nonfinite dense output");
    if(std::abs(actual - expected) > tol * std::max(1.0, std::abs(expected))) {
        std::cerr << "actual=" << actual << " expected=" << expected << '\n';
        throw std::runtime_error("CPU dense output differs from Boost");
    }
}

void compare_step(ODESolver& solver, TestRHS& rhs, const vector<double>& initial,
                  double start, double h) {
    using Stepper = boost::numeric::odeint::runge_kutta_dopri5<vector<double>>;
    Stepper reference;
    vector<double> k1(2), k7(2), end(2), sampled(2), actual(2);
    rhs(initial, k1, start);
    reference.do_step(rhs, initial, k1, start, end, k7, h);
    for(int i = 0; i <= 100; ++i) {
        double time = start + h * (i/100.0);
        reference.calc_state(time, sampled, initial, k1, start, end, k7, start+h);
        solver.calc_state(time, actual);
        check(actual.size() == initial.size(), "Wrong dense output size");
        for(size_t j = 0; j < actual.size(); ++j) close(actual[j], sampled[j]);
    }
}

int main() {
    TestRHS rhs;
    vector<double> initial{1.2, -0.7};
    for(double h : {0.5, 0.1, 0.01}) {
        const double start = 0.37;
        DormandPrinceSolver custom(1, 1, h);
        custom.initialize(initial, start, h, rhs);
        vector<double> launch;
        custom.calc_state(start, launch);
        check(launch == initial, "Dense output before the first step changed");
        auto step = custom.do_step(rhs);
        close(step.second - step.first, h);
        compare_step(custom, rhs, initial, start, h);
        bool rejected = false;
        try { custom.calc_state(start - 1e-3, launch); }
        catch(const std::runtime_error&) { rejected = true; }
        check(rejected, "Out-of-step sample was accepted");

        DopriBoostSolver wrapped(1, 1, h);
        wrapped.initialize(initial, start, h, rhs);
        step = wrapped.do_step(rhs);
        close(step.second - step.first, h);
        compare_step(wrapped, rhs, initial, start, h);
    }

    // The custom controller retains h_min even when the requested tolerance
    // would require a smaller step. Compare its forced steps against Boost
    // using the same initial state and step size, including the second FSAL step.
    TestRHS fast(40);
    const double hmin = 0.02;
    auto limited = create_dormand_prince_solver(1e-15, 1e-15, 0.05, hmin);
    limited->initialize(initial, 0.1, 1e-4, fast);
    for(int i = 0; i < 2; ++i) {
        vector<double> start_state = limited->current_state();
        double start_time = limited->current_time();
        auto step = limited->do_step(fast);
        close(step.second - step.first, hmin);
        compare_step(*limited, fast, start_state, start_time, hmin);
        check(limited->get_nsteps() == i+1, "Step diagnostic changed");
        // Forced acceptance does not count as a tolerance-satisfying good step.
        check(limited->get_ngood() == 0, "Forced-step diagnostic changed");
        check(limited->get_nbad() >= i+1, "Forced step did not exercise rejection");
        close(limited->get_hmin(), hmin);
        close(limited->get_hmax(), hmin);
        close(limited->get_hnext(), hmin);
    }
    std::cout << "CPU custom and Boost interpolation agree; minimum-step controls preserved.\n";
}
#else
int main() {
    std::cout << "Boost headers unavailable.\n";
    return 77;
}
#endif
