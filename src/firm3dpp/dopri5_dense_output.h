#ifndef FIRM3D_DOPRI5_DENSE_OUTPUT_H
#define FIRM3D_DOPRI5_DENSE_OUTPUT_H

#ifdef __CUDACC__
#define FIRM3D_DENSE_HD __host__ __device__
#else
#define FIRM3D_DENSE_HD
#endif

// Boost's Dormand-Prince continuous-extension weights. k2 has zero weight.
// Sampling and event-polynomial construction share this coefficient table.
struct Dopri5DenseWeight {
    double endpoint, scale, offset, slope, denominator;
};

template<int Stage>
FIRM3D_DENSE_HD constexpr Dopri5DenseWeight dopri5_dense_weight() {
    if constexpr(Stage == 1) return {35.0/384, 5, 2558722523LL, 31403016, 11282082432LL};
    else if constexpr(Stage == 3) return {500.0/1113, 100, 882725551, 15701508, 32700410799LL};
    else if constexpr(Stage == 4) return {125.0/192, 25, 443332067, 31403016, 1880347072};
    else if constexpr(Stage == 5) return {-2187.0/6784, 32805, 23143187, 3489224, 199316789632LL};
    else if constexpr(Stage == 6) return {11.0/84, 55, 29972135, 7076736, 822651844};
    else {
        static_assert(Stage == 7, "Dense output uses stages 1, 3, 4, 5, 6, and 7");
        return {0, 10, 7414447, 829305, 29380423};
    }
}

template<int Stage, typename T>
FIRM3D_DENSE_HD T dopri5_dense_correction(T theta) {
    constexpr auto w = dopri5_dense_weight<Stage>();
    return T(w.scale) * (T(w.offset) - T(w.slope) * theta) / T(w.denominator);
}

// The continuous extension used by Boost odeint's runge_kutta_dopri5,
// which is also the CPU tracer's default dense output. k2 has zero weight.
// Call before replacing k1 with k7 for the next FSAL step.
template<typename T>
FIRM3D_DENSE_HD T dopri5_dense_state(T y0, T h, double fraction,
                                    T k1, T k3, T k4, T k5, T k6, T k7) {
    // Cancellation in the continuous-extension weights can amplify float
    // roundoff. Evaluate the sampling polynomial in double and round once;
    // the integrator's state and stages still use the requested precision.
    if constexpr(sizeof(T) < sizeof(double)){
        return T(dopri5_dense_state<double>(double(y0), double(h), fraction,
            double(k1), double(k3), double(k4), double(k5), double(k6), double(k7)));
    }
    const T theta = T(fraction);
    const T xm1 = theta - T(1);
    const T a = theta * theta * (T(3) - T(2) * theta);
    const T b = theta * theta * xm1;
    const T c = theta * theta * xm1 * xm1;
    const T d = theta * xm1 * xm1;
    const T x1 = dopri5_dense_correction<1>(theta);
    const T x3 = dopri5_dense_correction<3>(theta);
    const T x4 = dopri5_dense_correction<4>(theta);
    const T x5 = dopri5_dense_correction<5>(theta);
    const T x6 = dopri5_dense_correction<6>(theta);
    const T x7 = dopri5_dense_correction<7>(theta);
    return y0 + h * ((a * T(dopri5_dense_weight<1>().endpoint) - c * x1 + d) * k1
                  + (a * T(dopri5_dense_weight<3>().endpoint) + c * x3) * k3
                  + (a * T(dopri5_dense_weight<4>().endpoint) - c * x4) * k4
                  + (a * T(dopri5_dense_weight<5>().endpoint) + c * x5) * k5
                  + (a * T(dopri5_dense_weight<6>().endpoint) - c * x6) * k6
                  + (b + c * x7) * k7);
}

// Expand the same continuous extension for root isolation. k contains
// {k1, k3, k4, k5, k6, k7}; coefficients are ordered by increasing power.
// Stage differences keep constant derivatives exactly linear.
template<typename T>
FIRM3D_DENSE_HD void dopri5_dense_coefficients(T y0, T h, const T* k, double* c) {
    const Dopri5DenseWeight weights[] = {dopri5_dense_weight<3>(),
        dopri5_dense_weight<4>(), dopri5_dense_weight<5>(),
        dopri5_dense_weight<6>(), dopri5_dense_weight<7>()};
    c[0] = double(y0); c[1] = double(h) * double(k[0]);
    for(int i = 2; i <= 5; ++i) c[i] = 0;
#ifdef __CUDACC__
#pragma unroll
#endif
    for(int j = 0; j < 5; ++j){
        const auto w = weights[j];
        double a = w.scale * w.offset / w.denominator;
        double slope = w.scale * w.slope / w.denominator;
        double signed_a = j % 2 ? -a : a;
        double signed_slope = j % 2 ? -slope : slope;
        double delta = double(h) * (double(k[j + 1]) - double(k[0]));
        c[2] += delta * (j < 4 ? 3*w.endpoint + signed_a : -1 + signed_a);
        c[3] += delta * (j < 4 ? -2*w.endpoint - 2*signed_a - signed_slope : 1 - 2*signed_a - signed_slope);
        c[4] += delta * (signed_a + 2*signed_slope);
        c[5] -= delta * signed_slope;
    }
}

#undef FIRM3D_DENSE_HD
#endif
