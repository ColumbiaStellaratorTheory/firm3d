#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>

#if __has_include(<boost/math/tools/toms748_solve.hpp>)
#include "tracing_helpers.h"

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
    std::cout << "CPU endpoint crossing checks passed.\n";
}
#else
int main() {
    std::cout << "Boost headers unavailable.\n";
    return 77;
}
#endif
