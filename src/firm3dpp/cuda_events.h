#ifndef FIRM3D_CUDA_EVENTS_H
#define FIRM3D_CUDA_EVENTS_H

#include "dopri5_events.h"
#include <algorithm>
#include <pybind11/stl.h>

// Only event-enabled kernel specializations use these buffers. The ordinary
// endpoint and history kernels compile out event evaluation entirely.
struct DeviceEvents {
    double *planes = nullptr, *vpars = nullptr, *criteria = nullptr;
    double *hits = nullptr, *end_times = nullptr, *theta_offsets = nullptr;
    double *cursors = nullptr, *transit_start = nullptr, *last_phase_time = nullptr;
    int *counts = nullptr, *phase_counts = nullptr, *iterations = nullptr;
    int* overflow = nullptr;
    int nplanes = 0, nvpars = 0, ncriteria = 0, capacity = 0, max_phase_hits = 0;
    bool phases_stop = false, vpars_stop = false;
    double max_phase_interval = 0;
};

struct EventRequest {
    vector<double> planes, vpars, criteria, theta_offsets;
    int capacity = 0, max_phase_hits = 0;
    bool phases_stop = false, vpars_stop = false;
    double max_phase_interval = 0;
    py::array_t<double> hits, end_times;
    bool enabled() const { return !planes.empty() || !vpars.empty() || !criteria.empty(); }

    explicit EventRequest(py::dict options = {}) {
        if(options.size() == 0) return;
        planes = options["planes"].cast<vector<double>>();
        vpars = options["vpars"].cast<vector<double>>();
        capacity = options["max_hits"].cast<int>();
        max_phase_hits = options["max_phase_hits"].cast<int>();
        if(options.contains("max_phase_interval")) max_phase_interval = options["max_phase_interval"].cast<double>();
        phases_stop = options["phases_stop"].cast<bool>();
        vpars_stop = options["vpars_stop"].cast<bool>();
        theta_offsets = options["theta_offsets"].cast<vector<double>>();
        if(planes.size() % 4 || capacity <= 0 || max_phase_hits < 0 ||
           !std::isfinite(max_phase_interval) || max_phase_interval < 0 ||
           (planes.empty() && (phases_stop || max_phase_hits || max_phase_interval)) ||
           (vpars.empty() && vpars_stop)){
            throw std::invalid_argument("Invalid GPU event configuration");
        }
        for(double x : planes) if(!std::isfinite(x)) throw std::invalid_argument("Nonfinite phase plane");
        for(double x : vpars) if(!std::isfinite(x)) throw std::invalid_argument("Nonfinite vpar plane");
        vector<int> levelsets = options.contains("levelset_indices")
            ? options["levelset_indices"].cast<vector<int>>() : vector<int>{};
        int criterion_index = 0;
        for(auto criterion : options["stopping_criteria"].cast<vector<shared_ptr<StoppingCriterion>>>()){
            double kind, limit;
            if(std::find(levelsets.begin(), levelsets.end(), criterion_index++) != levelsets.end()){ kind = 6; limit = 0; }
            else if(!criterion){ kind = 5; limit = 0; }
            else if(auto c = std::dynamic_pointer_cast<MaxToroidalFluxStoppingCriterion>(criterion)){ kind = 0; limit = c->get_max_s(); }
            else if(auto c = std::dynamic_pointer_cast<MinToroidalFluxStoppingCriterion>(criterion)){ kind = 1; limit = c->get_min_s(); }
            else if(auto c = std::dynamic_pointer_cast<IterationStoppingCriterion>(criterion)){ kind = 2; limit = c->get_max_iter(); }
            else if(auto c = std::dynamic_pointer_cast<ToroidalTransitStoppingCriterion>(criterion)){ kind = 3; limit = c->get_max_transits(); }
            else if(auto c = std::dynamic_pointer_cast<StepSizeStoppingCriterion>(criterion)){ kind = 4; limit = c->get_min_dt(); }
            else throw std::invalid_argument("This stopping criterion cannot run on the GPU");
            if(!std::isfinite(limit)) throw std::invalid_argument("Stopping limits must be finite");
            criteria.insert(criteria.end(), {kind, limit});
        }
    }
};

// Wave input allocations must also be released when event-buffer overflow or
// validation raises after the trace.
struct WaveBufferGuard {
    void *m, *n, *phihats;
    ~WaveBufferGuard() { cudaFree(m); cudaFree(n); cudaFree(phihats); }
};

__device__ inline bool write_event(DeviceEvents events, int particle, double time,
                                   int index, const double* state) {
    int count = events.counts[particle];
    if(count >= events.capacity){ atomicExch(events.overflow, 1); return false; }
    double* row = events.hits + (size_t(particle) * events.capacity + count) * 6;
    row[0] = time; row[1] = index;
    for(int i = 0; i < 4; ++i) row[i + 2] = state[i];
    events.counts[particle] = count + 1;
    return true;
}

template<typename T>
__device__ T classifier_at(const T* xyz, const T* quadpts) {
    T phi = atan2(xyz[1], xyz[0]);
    T period = T(grid_ranges_d[5]);
    phi = fmod(phi, period);
    phi += period * (phi < 0);
    if(xyz[2] < 0){
        phi = fmod(T(2*M_PI) - phi, period);
        phi += period * (phi < 0);
    }
    T point[3] = {hypot(xyz[0], xyz[1]), phi, fabs(xyz[2])};
    T shape[3][4];
    int cell[3];
    for(int d = 0; d < 3; ++d){
        T offset = (point[d] - T(grid_ranges_d[4*d])) * T(grid_ranges_d[4*d+3]);
        int index = max(0, min(3*(int(offset)/3), int(grid_ranges_d[4*d+2])-4));
        T x = offset - T(index);
        shape[d][0] = -T(1.0/6.0)*(x-1)*(x-2)*(x-3);
        shape[d][1] = T(0.5)*x*(x-2)*(x-3);
        shape[d][2] = -T(0.5)*x*(x-1)*(x-3);
        shape[d][3] = T(1.0/6.0)*x*(x-1)*(x-2);
        cell[d] = index/3;
    }
    const int base = 64*(7*(cell[0]*n_x2_d*n_x3_d + cell[1]*n_x3_d + cell[2]) + 6);
    T value = 0;
    for(int i = 0; i < 4; ++i)
        for(int j = 0; j < 4; ++j)
            for(int k = 0; k < 4; ++k)
                value += shape[0][i]*shape[1][j]*shape[2][k]*quadpts[base+16*i+4*j+k];
    return value;
}

template<typename T, RHS id>
__device__ void record_step_events(DeviceEvents events, int particle, int p,
                                   T* state, const T* derivs, const T* x_temp,
                                   double time, T h, double tmax, double zeta_offset,
                                   const T* quadpts) {
    using namespace firm3d_events;
    Step step;
    step.t = time; step.h = double(h);
    step.theta_offset = events.theta_offsets[particle];
    step.zeta_offset = zeta_offset;
    constexpr int nd = map_rhs_to_n_deriv_outputs<id>();
    constexpr int stages[] = {0, 2, 3, 4, 5, 6};
    for(int component = 0; component < 4; ++component){
        T k[6];
        for(int j = 0; j < 6; ++j) k[j] = derivs[(nd*stages[j] + component)*PARTICLES_PER_BLOCK + p];
        step.state[component] = dense_polynomial(state[component*PARTICLES_PER_BLOCK + p], h, k);
    }
    if constexpr(map_rhs_to_coord<id>() == CoordSys::Boozer){
        step.nbranches = roots(step.state[1], 0, 1, step.branch_roots);
    }
    double limit = fmin(1.0, (tmax - time) / step.h);
    const double step_limit = limit;
    const double endpoint_time = fmin(tmax, time + step.h);
    int stop_index = 0;
    bool endpoint_stop = false;
    bool root_stop = false;
    bool interval_stop = false;
    const int iteration = ++events.iterations[particle];
    const double x_end = limit == 1 ? double(x_temp[PARTICLES_PER_BLOCK+p]) : value(step.state[0], limit);
    const double y_end = limit == 1 ? double(x_temp[2*PARTICLES_PER_BLOCK+p]) : value(step.state[1], limit);
    const double zeta_end = (limit == 1 ? double(x_temp[3*PARTICLES_PER_BLOCK+p])
                                      : value(step.state[2], limit)) + step.zeta_offset;
    if(iteration == 1) events.transit_start[particle] = zeta_end;
    for(int j = 0; j < events.ncriteria; ++j){
        int kind = int(events.criteria[2*j]);
        double bound = events.criteria[2*j+1];
        double s = hypot(x_end, y_end);
        bool stop = kind == 0 ? s >= bound : kind == 1 ? s <= bound :
                    kind == 2 ? iteration > bound : kind == 3 ?
                    fabs(floor((zeta_end - events.transit_start[particle])/period)) >= bound :
                    kind == 4 ? step.h < bound : false;
        if constexpr(map_rhs_to_coord<id>() == CoordSys::Cartesian){
            // The Cartesian classifier is a requested endpoint criterion.
            // Evaluate the same table at a clipped final point when necessary.
            if(kind == 6){
                if(limit == 1){ stop = derivs[(nd*6 + 4)*PARTICLES_PER_BLOCK+p] < 0; }
                else {
                    T point[3] = {T(x_end), T(y_end), T(zeta_end)};
                    stop = classifier_at(point, quadpts) < 0;
                }
            }
        }
        if(stop){ stop_index = -1-j; endpoint_stop = true; break; }
    }
    // A cursor per event label preserves simultaneous hits: advancing a global
    // cursor would discard a second plane or velocity level at the same time.
    double* cursors = events.cursors;
    if(events.nplanes+events.nvpars) cursors += size_t(particle)*(events.nplanes+events.nvpars);
    for(int j = 0; j < events.nplanes+events.nvpars; ++j) cursors[j] = 0;
    while(true){
        // A section resets its deadline within the same accepted step. Search
        // through the deadline before timing out, so an exact-deadline return
        // is retained, as on the CPU.
        limit = step_limit;
        if(events.max_phase_interval > 0){
            double deadline = events.last_phase_time[particle] + events.max_phase_interval;
            limit = fmin(limit, fmax(0.0, (deadline - time) / step.h));
        }
        double next = limit + 1;
        int index = -1;
        for(int j = 0; j < events.nplanes; ++j){
            double root = next_phase(step, events.planes + 4*j, cursors[j], limit);
            if(root < next){ next = root; index = j; }
        }
        for(int j = 0; j < events.nvpars; ++j){
            Polynomial velocity = step.state[3];
            velocity.c[0] -= events.vpars[j];
            double zeros[max_degree+1];
            double after = cursors[events.nplanes+j];
            int count = roots(velocity, after, limit, zeros);
            for(int k = 0; k < count; ++k){
                if(zeros[k] > after &&
                   (zeros[k] < next || (zeros[k] == next && index < events.nplanes)) &&
                   crossing_direction(velocity, zeros[k])){
                    next = zeros[k]; index = events.nplanes + j; break;
                }
            }
        }
        if(index < 0 || next > limit){
            interval_stop = events.max_phase_interval > 0 &&
                events.last_phase_time[particle] + events.max_phase_interval <= endpoint_time;
            if(interval_stop) endpoint_stop = false;
            break;
        }
        double hit[4];
        for(int j = 0; j < 4; ++j) hit[j] = double(T(value(step.state[j], next)));
        if constexpr(map_rhs_to_coord<id>() == CoordSys::Boozer){
            hit[0] = hypot(hit[0], hit[1]);
            hit[1] = step.theta(next);
            hit[2] = value(step.state[2], next)+step.zeta_offset;
        }
        if(index >= events.nplanes) hit[3] = events.vpars[index - events.nplanes];
        bool stored = write_event(events, particle, time + next*step.h, index, hit);
        bool stop;
        if(index < events.nplanes){
            if(events.max_phase_interval > 0) events.last_phase_time[particle] = time + next*step.h;
            int count = ++events.phase_counts[particle];
            stop = events.phases_stop || (events.max_phase_hits && count >= events.max_phase_hits);
        } else stop = events.vpars_stop;
        if(stop || !stored){ limit = next; endpoint_stop = false; root_stop = true; break; }
        cursors[index] = next;
    }
    if(endpoint_stop){
        double hit[4];
        for(int j = 0; j < 4; ++j) hit[j] = step_limit == 1
            ? double(x_temp[(j+1)*PARTICLES_PER_BLOCK+p]) : double(T(value(step.state[j], step_limit)));
        if constexpr(map_rhs_to_coord<id>() == CoordSys::Boozer){
            hit[0] = hypot(hit[0], hit[1]); hit[1] = step.theta(limit);
            hit[2] += step.zeta_offset;
        }
        write_event(events, particle, endpoint_time, stop_index, hit);
    }
    if(endpoint_stop || root_stop || interval_stop || limit < fmin(1.0, (tmax-time)/step.h) ||
       (events.phases_stop && events.phase_counts[particle]) ||
       (events.max_phase_hits && events.phase_counts[particle] >= events.max_phase_hits)){
        events.end_times[particle] = interval_stop
            ? events.last_phase_time[particle] + events.max_phase_interval
            : (endpoint_stop ? endpoint_time : time + limit*step.h);
    }
    // Preserve poloidal winding across atan2's branch cut. Toroidal winding
    // is maintained by the tracer for both event and ordinary output paths.
    if constexpr(map_rhs_to_coord<id>() == CoordSys::Boozer){
        double wrapped_theta = atan2(double(x_temp[2*PARTICLES_PER_BLOCK+p]),
                                     double(x_temp[PARTICLES_PER_BLOCK+p]));
        events.theta_offsets[particle] = period * round((step.theta(1)-wrapped_theta)/period);
    }
}

#endif
