#ifndef FIRM3D_DOPRI5_EVENTS_H
#define FIRM3D_DOPRI5_EVENTS_H

#include <cmath>
#include <cfloat>

#ifdef __CUDACC__
#define FIRM3D_EVENT_HD __host__ __device__
#else
#define FIRM3D_EVENT_HD
#endif

namespace firm3d_events {

// A phase derivative can have degree 14 after clearing the atan2 denominator.
constexpr int max_degree = 14;
constexpr double period = 6.283185307179586476925286766559;
struct Polynomial {
    double c[max_degree + 1]{};
    int degree = 0;
};

FIRM3D_EVENT_HD inline double value(const Polynomial& p, double x) {
    double y = p.c[p.degree];
    for(int i = p.degree - 1; i >= 0; --i) y = y * x + p.c[i];
    return y;
}

FIRM3D_EVENT_HD inline Polynomial derivative(const Polynomial& p) {
    Polynomial d;
    d.degree = p.degree > 0 ? p.degree - 1 : 0;
    for(int i = 1; i <= p.degree; ++i) d.c[i - 1] = i * p.c[i];
    return d;
}

FIRM3D_EVENT_HD inline double roundoff(const Polynomial& p, double x) {
    double scale = fabs(p.c[p.degree]);
    for(int i = p.degree - 1; i >= 0; --i) scale = scale * fabs(x) + fabs(p.c[i]);
    return 32 * DBL_EPSILON * scale;
}

FIRM3D_EVENT_HD inline double bisect(const Polynomial& p, double a, double b) {
    double fa = value(p, a), fb = value(p, b);
    if(fa == 0) return a;
    if(fb == 0) return b;
    for(int i = 0; i < 80; ++i){
        double m = a + (b - a) * 0.5;
        if(m == a || m == b) break;
        double fm = value(p, m);
        if(fm == 0) return m;
        if((fa > 0) == (fm > 0)){ a = m; fa = fm; }
        else { b = m; fb = fm; }
    }
    return fabs(fa) < fabs(fb) ? a : b;
}

// Derivative roots partition a polynomial into monotone intervals. Build the
// derivative hierarchy iteratively, so device root isolation needs no recursion.
FIRM3D_EVENT_HD inline int roots(Polynomial p, double lo, double hi, double* out) {
    while(p.degree && p.c[p.degree] == 0) --p.degree;
    double previous[max_degree + 1]{}, current[max_degree + 1]{};
    int nprevious = 0;
    for(int degree = 1; degree <= p.degree; ++degree){
        Polynomial d;
        d.degree = degree;
        const int order = p.degree - degree;
        for(int j = 0; j <= degree; ++j){
            double coefficient = p.c[j + order];
            for(int k = j + 1; k <= j + order; ++k) coefficient *= k;
            d.c[j] = coefficient;
        }
        int count = 0;
        double a = lo, fa = value(d, a);
        if(fabs(fa) <= roundoff(d, a)){ current[count++] = a; fa = 0; }
        for(int interval = 0; interval <= nprevious; ++interval){
            double b = interval < nprevious ? previous[interval] : hi;
            double fb = value(d, b);
            bool zero_b = fabs(fb) <= roundoff(d, b);
            if(zero_b) fb = 0;
            if(b > a && fa != 0 && fb != 0 && (fa > 0) != (fb > 0)){
                double root = bisect(d, a, b);
                if(count < max_degree+1 && (count == 0 || root > current[count - 1])) current[count++] = root;
            }
            if(zero_b && count < max_degree+1 && (count == 0 || b > current[count - 1])){
                current[count++] = b;
            }
            a = b; fa = fb;
        }
        // Roundoff can identify both a stationary point and its adjacent root.
        nprevious = 0;
        for(int j = 0; j < count; ++j){
            if(nprevious == 0 || current[j] - previous[nprevious - 1] >
                    64 * DBL_EPSILON * fmax(1.0, fabs(current[j]))){
                previous[nprevious++] = current[j];
            }
        }
    }
    for(int j = 0; j < nprevious; ++j) out[j] = previous[j];
    return nprevious;
}

// Zero derivative at a tangency is not a crossing; an odd-order zero is.
FIRM3D_EVENT_HD inline int crossing_direction(Polynomial p, double x) {
    for(int order = 1; order <= max_degree && p.degree; ++order){
        p = derivative(p);
        double d = value(p, x);
        if(fabs(d) > roundoff(p, x)) return order % 2 ? (d > 0 ? 1 : -1) : 0;
    }
    return 0;
}

// Expand the same Boost continuous extension as dopri5_dense_state. Express
// higher coefficients in stage differences to make constant RHS exactly linear.
template<typename T>
FIRM3D_EVENT_HD Polynomial dense_polynomial(T y0, T h, const T* k) {
    const double b[] = {500.0/1113, 125.0/192, -2187.0/6784, 11.0/84};
    const double a[] = {100.0*882725551/32700410799.0,
        25.0*443332067/1880347072.0, 32805.0*23143187/199316789632.0,
        55.0*29972135/822651844.0, 10.0*7414447/29380423.0};
    const double slope[] = {100.0*15701508/32700410799.0,
        25.0*31403016/1880347072.0, 32805.0*3489224/199316789632.0,
        55.0*7076736/822651844.0, 10.0*829305/29380423.0};
    Polynomial p;
    p.degree = 5; p.c[0] = double(y0); p.c[1] = double(h) * double(k[0]);
    for(int j = 0; j < 5; ++j){
        double delta = double(h) * (double(k[j + 1]) - double(k[0]));
        double signed_a = j % 2 ? -a[j] : a[j];
        double signed_slope = j % 2 ? -slope[j] : slope[j];
        p.c[2] += delta * (j < 4 ? 3*b[j] + signed_a : -1 + signed_a);
        p.c[3] += delta * (j < 4 ? -2*b[j] - 2*signed_a - signed_slope : 1 - 2*signed_a - signed_slope);
        p.c[4] += delta * (signed_a + 2*signed_slope);
        p.c[5] -= delta * signed_slope;
    }
    return p;
}

FIRM3D_EVENT_HD inline Polynomial multiply(const Polynomial& a, const Polynomial& b) {
    Polynomial result;
    result.degree = a.degree + b.degree;
    for(int i = 0; i <= a.degree; ++i)
        for(int j = 0; j <= b.degree; ++j) result.c[i+j] += a.c[i] * b.c[j];
    return result;
}

struct Step {
    Polynomial state[4];
    double t, h, theta_offset, zeta_offset;
    double branch_roots[max_degree+1]{};
    int nbranches = 0;

    FIRM3D_EVENT_HD double theta(double q) const {
        double x = value(state[0], q), y = value(state[1], q);
        double angle = atan2(y, x) + theta_offset;
        for(int i = 0; i < nbranches; ++i){
            double root = branch_roots[i];
            if(root > q) break;
            if(value(state[0], root) >= 0) continue;
            int sign = crossing_direction(state[1], root);
            if(!sign) continue;
            if(root == q){
                angle = (sign < 0 ? period/2 : -period/2) + theta_offset;
                // Earlier roots have already contributed to the winding below.
                for(int j = 0; j < i; ++j)
                    if(value(state[0], branch_roots[j]) < 0)
                        angle -= period * crossing_direction(state[1], branch_roots[j]);
                return angle;
            }
            if(root > 0 || (sign < 0 && atan2(value(state[1], 0), value(state[0], 0)) > 0)
                         || (sign > 0 && atan2(value(state[1], 0), value(state[0], 0)) < 0)){
                angle -= sign * period;
            }
        }
        return angle;
    }
    FIRM3D_EVENT_HD double phase(const double* plane, double q) const {
        return plane[1] * (value(state[2], q) + zeta_offset)
             + (plane[2] == 0 ? 0 : plane[2] * theta(q))
             - plane[3] * (t + h*q);
    }
};

FIRM3D_EVENT_HD inline Polynomial phase_derivative(const Step& step, const double* plane) {
    Polynomial rate = derivative(step.state[2]);
    for(int i = 0; i <= rate.degree; ++i) rate.c[i] *= plane[1];
    rate.c[0] -= plane[3] * step.h;
    if(plane[2] == 0) return rate;
    Polynomial r2 = multiply(step.state[0], step.state[0]);
    Polynomial y2 = multiply(step.state[1], step.state[1]);
    for(int i = 0; i <= y2.degree; ++i) r2.c[i] += y2.c[i];
    Polynomial result = multiply(rate, r2);
    Polynomial xyprime = multiply(step.state[0], derivative(step.state[1]));
    Polynomial yxprime = multiply(step.state[1], derivative(step.state[0]));
    for(int i = 0; i <= xyprime.degree; ++i)
        result.c[i] += plane[2] * (xyprime.c[i] - yxprime.c[i]);
    return result;
}

FIRM3D_EVENT_HD inline double next_phase(const Step& step, const double* plane,
                                        double after, double limit) {
    Polynomial rate = phase_derivative(step, plane);
    double turning[max_degree + 1];
    int count = roots(rate, 0, limit, turning);
    double a = after;
    for(int interval = 0; interval <= count; ++interval){
        double b = interval < count ? turning[interval] : limit;
        if(b <= a) continue;
        double pa = step.phase(plane, a), pb = step.phase(plane, b);
        const int sign = pb > pa ? 1 : (pb < pa ? -1 : 0);
        if(sign){
            double turns = (pa - plane[0]) / period;
            double shift = sign > 0 ? floor(turns) + 1 : ceil(turns) - 1;
            double target = plane[0] + period * shift;
            if(sign > 0 ? target <= pa : target >= pa){
                shift += sign;
                target = plane[0] + period * shift;
            }
            if(sign > 0 ? pb >= target : pb <= target){
                double left = a, right = b;
                for(int j = 0; j < 80; ++j){
                    double mid = left + (right - left) * 0.5;
                    if(mid == left || mid == right) break;
                    double pm = step.phase(plane, mid);
                    if(pm == target){ left = right = mid; break; }
                    if(sign > 0 ? pm < target : pm > target) left = mid;
                    else right = mid;
                }
                // Choose the crossed side of the final bracket so the next
                // search cannot rediscover this plane through roundoff.
                double root = right;
                if(root > after){
                    // A stationary touch is excluded; stationary inflections cross.
                    if(fabs(value(rate, root)) > roundoff(rate, root) ||
                       crossing_direction(rate, root) == 0) return root;
                }
            }
        }
        a = b;
    }
    return limit + 1;
}

} // namespace firm3d_events
#undef FIRM3D_EVENT_HD
#endif
