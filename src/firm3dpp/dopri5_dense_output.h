#ifndef FIRM3D_DOPRI5_DENSE_OUTPUT_H
#define FIRM3D_DOPRI5_DENSE_OUTPUT_H

#ifdef __CUDACC__
#define FIRM3D_DENSE_HD __host__ __device__
#else
#define FIRM3D_DENSE_HD
#endif

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
    const T x1 = T(5) * (T(2558722523LL) - T(31403016) * theta) / T(11282082432LL);
    const T x3 = T(100) * (T(882725551) - T(15701508) * theta) / T(32700410799LL);
    const T x4 = T(25) * (T(443332067) - T(31403016) * theta) / T(1880347072LL);
    const T x5 = T(32805) * (T(23143187) - T(3489224) * theta) / T(199316789632LL);
    const T x6 = T(55) * (T(29972135) - T(7076736) * theta) / T(822651844);
    const T x7 = T(10) * (T(7414447) - T(829305) * theta) / T(29380423);
    return y0 + h * ((a * T(35.0 / 384.0) - c * x1 + d) * k1
                  + (a * T(500.0 / 1113.0) + c * x3) * k3
                  + (a * T(125.0 / 192.0) - c * x4) * k4
                  + (a * T(-2187.0 / 6784.0) + c * x5) * k5
                  + (a * T(11.0 / 84.0) - c * x6) * k6
                  + (b + c * x7) * k7);
}

#undef FIRM3D_DENSE_HD
#endif
