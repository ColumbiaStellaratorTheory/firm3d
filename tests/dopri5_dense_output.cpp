#include "dopri5_dense_output.h"
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>

#if __has_include(<boost/numeric/odeint/stepper/runge_kutta_dopri5.hpp>)
#include <boost/numeric/odeint/stepper/runge_kutta_dopri5.hpp>
#define HAVE_BOOST_DENSE_REFERENCE
#endif

void check(double actual, double expected, double tol) {
    if(std::abs(actual - expected) > tol){
        throw std::runtime_error("Dormand-Prince dense output disagrees with reference");
    }
}

template<typename T>
void polynomial_tests() {
    const double nodes[] = {0, 0.2, 0.3, 0.8, 8.0/9.0, 1, 1};
    const T h = T(0.4);
    const T y0 = T(1.7);
    const double tol = sizeof(T) == sizeof(float) ? 3e-6 : 3e-14;
    // y' = t^degree has an exact polynomial integral. Many samples share
    // the same step, testing both endpoints and the interior extension.
    for(int degree = 0; degree <= 3; ++degree){
        T k[7];
        for(int j = 0; j < 7; ++j){
            k[j] = T(std::pow(double(h) * nodes[j], degree));
        }
        for(int i = 0; i <= 100; ++i){
            const double theta = i / 100.0;
            const double exact = double(y0) + std::pow(double(h) * theta, degree + 1) / (degree + 1);
            check(dopri5_dense_state(y0, h, theta, k[0], k[2], k[3], k[4], k[5], k[6]), exact, tol);
        }
    }
}

int main() {
    polynomial_tests<double>();
    polynomial_tests<float>();
#ifdef HAVE_BOOST_DENSE_REFERENCE
    using State = std::array<double, 1>;
    boost::numeric::odeint::runge_kutta_dopri5<State> stepper;
    const auto rhs = [](const State& y, State& dy, double t){dy[0] = y[0] + std::sin(t);};
    const double weights[7][7] = {
        {}, {1.0/5.0}, {3.0/40.0, 9.0/40.0},
        {44.0/45.0, -56.0/15.0, 32.0/9.0},
        {19372.0/6561.0, -25360.0/2187.0, 64448.0/6561.0, -212.0/729.0},
        {9017.0/3168.0, -355.0/33.0, 46732.0/5247.0, 49.0/176.0, -5103.0/18656.0},
        {35.0/384.0, 0, 500.0/1113.0, 125.0/192.0, -2187.0/6784.0, 11.0/84.0}
    };
    const double nodes[] = {0, 0.2, 0.3, 0.8, 8.0/9.0, 1, 1};
    for(double h : {0.5, 0.1, 0.01}){
        State y0{1.2}, end{}, k1{}, k7{};
        const double t0 = 0.37;
        rhs(y0, k1, t0);
        stepper.do_step(rhs, y0, k1, t0, end, k7, h);
        double k[7];
        for(int j = 0; j < 7; ++j){
            State stage = y0, deriv{};
            for(int l = 0; l < j; ++l){stage[0] += h * weights[j][l] * k[l];}
            rhs(stage, deriv, t0 + h * nodes[j]);
            k[j] = deriv[0];
        }
        for(int i = 0; i <= 100; ++i){
            double theta = i / 100.0;
            State reference{};
            stepper.calc_state(t0 + theta * h, reference, y0, k1, t0, end, k7, t0 + h);
            check(dopri5_dense_state(y0[0], h, theta, k[0], k[2], k[3], k[4], k[5], k[6]), reference[0], 3e-14);
        }
    }
    std::cout << "Dense output matches Boost's CPU continuous extension.\n";
#endif
    std::cout << "Float and double polynomial checks passed.\n";
}
