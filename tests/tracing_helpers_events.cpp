#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>

#if __has_include(<boost/math/tools/toms748_solve.hpp>)
#include "tracing_helpers.h"
#include "dopri5_events.h"

struct LinearStep {
    vector<double> left, right;
    void calc_state(double tau, vector<double>& y) {
        if (tau == 0 || tau == 1) {
            y = tau == 0 ? left : right;
        } else {
            for (size_t i = 0; i < y.size(); ++i) {
                y[i] = left[i] + tau * (right[i] - left[i]);
            }
        }
    }
};

void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void check_event(bool velocity, double target, double left, double right,
                 double expected_time, bool stopping) {
    LinearStep dense{{0.3, 0.2, 0, 1}, {0.3, 0.2, 0, 1}};
    dense.left[velocity ? 3 : 2] = left;
    dense.right[velocity ? 3 : 2] = right;
    vector<vector<double>> hits;
    bool stop = check_stopping_criteria(
        4, 1, hits, dense, 0, 1, 1, 1e-12,
        velocity ? vector<double>{} : vector<double>{target},
        velocity ? vector<double>{} : vector<double>{1},
        velocity ? vector<double>{} : vector<double>{0},
        velocity ? vector<double>{} : vector<double>{0}, {},
        velocity ? vector<double>{target} : vector<double>{},
        stopping, stopping, 0, 1, 1);
    bool expected_hit = expected_time >= 0;
    check(hits.size() == size_t(expected_hit), "Missing or duplicate CPU crossing");
    check(stop == (expected_hit && stopping), "CPU crossing stop flag differs");
    if (expected_hit) {
        check(hits[0].size() == 6 && hits[0][1] == 0, "CPU hit format changed");
        check(std::abs(hits[0][0] - expected_time) < 1e-12, "Wrong CPU root time");
        double value = hits[0][velocity ? 5 : 4];
        double expected = left + expected_time * (right - left);
        check(std::abs(value - expected) < 1e-12, "Wrong CPU root state");
    }
}

struct PolynomialStep {
    firm3d_events::Step step{};
    void calc_state(double tau, vector<double>& y) {
        using firm3d_events::value;
        y = {std::hypot(value(step.state[0], tau), value(step.state[1], tau)),
             step.theta(tau), value(step.state[2], tau) + step.zeta_offset,
             value(step.state[3], tau)};
    }
};

// Compare candidate selection against the actual CPU helper on identical
// interpolants, without relying on CPU and GPU controllers taking equal steps.
void check_candidates(PolynomialStep dense, double limit) {
    const vector<double> phases{0, 0.25}, ns{1, 1}, ms{0, 0}, ws{0, 0};
    const vector<double> velocities{0, 0.25};
    vector<vector<double>> hits;
    check_stopping_criteria(4, 1, hits, dense, 0, limit, 1, 1e-12,
        phases, ns, ms, ws, {}, velocities, false, false, 0, 1, 1);
    for (int i = 0; i < 4; ++i) {
        double plane[]{phases[i % 2], 1, 0, 0};
        double root = i < 2 ? firm3d_events::phase_crossing(dense.step, plane, limit)
            : firm3d_events::velocity_crossing(dense.step.state[3], velocities[i - 2], limit);
        auto hit = std::find_if(hits.begin(), hits.end(),
            [i](const vector<double>& row) { return row[1] == i; });
        check((root <= limit) == (hit != hits.end()), "CPU/GPU crossing selection differs");
        if (hit != hits.end())
            check(std::abs(root - (*hit)[0]) < 1e-10, "CPU/GPU bracketed roots differ");
    }
}

int main() {
    for (bool stopping : {false, true}) {
        for (double sign : {1.0, -1.0}) {
            for (double phase : {0.0, 0.25}) {
                for (int turn : {-2, 0, 2}) {
                    double plane = phase + turn * 2 * M_PI;
                    check_event(false, phase, plane - sign * 0.5, plane, 1, stopping);
                    // The adjoining step must not rediscover the same hit.
                    check_event(false, phase, plane, plane + sign * 0.5, -1, stopping);
                    check_event(false, phase, plane - sign * 0.5,
                                plane + sign * 0.5, 0.5, stopping);
                    check_event(false, phase, plane - sign * 0.5,
                                plane - sign * 0.25, -1, stopping);
                    check_event(false, phase, plane, plane, -1, stopping);
                }
            }
            check_event(true, 0, sign, 0, 1, stopping);
            check_event(true, 0, 0, -sign, -1, stopping);
            check_event(true, 0, sign, -sign, 0.5, stopping);
            check_event(true, 0, sign, sign * 0.5, -1, stopping);
            check_event(true, 0, 0, 0, -1, stopping);
            for (double target : {-2.0, 2.0}) {
                check_event(true, target, target + sign, target, 1, stopping);
                check_event(true, target, target, target - sign, -1, stopping);
            }
        }
    }
    // Refining an exact endpoint must not exhaust the next plane's root budget.
    LinearStep dense{{0.3, 0.2, 0, 1}, {0.3, 0.2, 1, 1}};
    vector<vector<double>> hits;
    check_stopping_criteria(4, 1, hits, dense, 0, 1, 1, 1e-12,
        {1, 0.25}, {1, 1}, {0, 0}, {0, 0}, {}, {}, false, false, 0, 1, 1);
    check(hits.size() == 2, "Missing CPU phase plane");
    check(hits[0][0] == 1 && std::abs(hits[1][0] - 0.25) < 1e-12,
          "CPU phase planes share a consumed root budget");
    PolynomialStep polynomial;
    polynomial.step.h = 1;
    polynomial.step.state[0].c[0] = 0.3;
    for (double sign : {1.0, -1.0}) {
        polynomial.step.state[2].degree = polynomial.step.state[3].degree = 1;
        polynomial.step.state[2].c[0] = polynomial.step.state[3].c[0] = 0;
        polynomial.step.state[2].c[1] = sign * 20 * firm3d_events::period;
        polynomial.step.state[3].c[1] = sign;
        check_candidates(polynomial, 1);
        // A target at the launch is excluded, even if crossed again later.
        polynomial.step.state[3].degree = 2;
        polynomial.step.state[3].c[2] = -sign * 2;
        check_candidates(polynomial, 1);
    }
    polynomial.step.state[2].degree = polynomial.step.state[3].degree = 2;
    polynomial.step.state[2].c[0] = -0.25;
    polynomial.step.state[2].c[1] = 2;
    polynomial.step.state[2].c[2] = -2;
    polynomial.step.state[3].c[0] = 0.1875;
    polynomial.step.state[3].c[1] = -1;
    polynomial.step.state[3].c[2] = 1;
    check_candidates(polynomial, 1); // Interior reversals without endpoint brackets.
    check_candidates(polynomial, 0.4); // Horizon clipping creates a bracket.
    polynomial.step.state[2].c[0] = polynomial.step.state[3].c[0] = 1;
    polynomial.step.state[2].c[1] = polynomial.step.state[3].c[1] = -2;
    polynomial.step.state[2].c[2] = polynomial.step.state[3].c[2] = 1;
    check_candidates(polynomial, 1); // Right-endpoint touches are retained.
    std::cout << "CPU endpoint crossing checks passed.\n";
}
#else
int main() {
    std::cout << "Boost headers unavailable.\n";
    return 77;
}
#endif
