#ifndef FIRM3D_CUDA_EVENTS_H
#define FIRM3D_CUDA_EVENTS_H

#include "dopri5_events.h"
#include <pybind11/stl.h>

// Only event-enabled kernel specializations use these buffers. The ordinary
// endpoint and history kernels compile out event evaluation entirely.
struct DeviceEvents {
    double *planes = nullptr, *vpars = nullptr, *criteria = nullptr;
    double *hits = nullptr, *end_times = nullptr, *offsets = nullptr;
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
        for(auto criterion : options["stopping_criteria"].cast<vector<shared_ptr<StoppingCriterion>>>()){
            double kind, limit;
            if(!criterion){ kind = 5; limit = 0; }
            else if(auto c = std::dynamic_pointer_cast<MaxToroidalFluxStoppingCriterion>(criterion)){ kind = 0; limit = c->gpu_limit(); }
            else if(auto c = std::dynamic_pointer_cast<MinToroidalFluxStoppingCriterion>(criterion)){ kind = 1; limit = c->gpu_limit(); }
            else if(auto c = std::dynamic_pointer_cast<IterationStoppingCriterion>(criterion)){ kind = 2; limit = c->gpu_limit(); }
            else if(auto c = std::dynamic_pointer_cast<ToroidalTransitStoppingCriterion>(criterion)){ kind = 3; limit = c->gpu_limit(); }
            else if(auto c = std::dynamic_pointer_cast<StepSizeStoppingCriterion>(criterion)){ kind = 4; limit = c->gpu_limit(); }
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

template<typename T>
py::object event_result(py::array_t<T> output, const EventRequest& request) {
    if(request.enabled()) return py::make_tuple(output, request.hits, request.end_times);
    return output;
}

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

template<typename T, RHS id>
__device__ void record_step_events(DeviceEvents events, int particle, int p,
                                   T* state, const T* derivs, const T* x_temp,
                                   double time, T h, double tmax) {
    using namespace firm3d_events;
    Step step;
    step.t = time; step.h = double(h);
    step.theta_offset = events.offsets[2 * size_t(particle)];
    step.zeta_offset = events.offsets[2 * size_t(particle) + 1];
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
    int stop_index = 0;
    bool endpoint_stop = false;
    bool root_stop = false;
    bool interval_stop = false;
    const int iteration = ++events.iterations[particle];
    const double zeta_end = double(x_temp[3*PARTICLES_PER_BLOCK+p]) + step.zeta_offset;
    if(iteration == 1) events.transit_start[particle] = zeta_end;
    for(int j = 0; limit == 1 && j < events.ncriteria; ++j){
        int kind = int(events.criteria[2*j]);
        double bound = events.criteria[2*j+1];
        double s = hypot(double(x_temp[PARTICLES_PER_BLOCK+p]),
                         double(x_temp[2*PARTICLES_PER_BLOCK+p]));
        bool stop = kind == 0 ? s >= bound : kind == 1 ? s <= bound :
                    kind == 2 ? iteration > bound : kind == 3 ?
                    fabs(floor((zeta_end - events.transit_start[particle])/period)) >= bound :
                    kind == 4 ? step.h < bound : false;
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
                if(zeros[k] > after && zeros[k] <= next && crossing_direction(velocity, zeros[k])){
                    next = zeros[k]; index = events.nplanes + j; break;
                }
            }
        }
        if(index < 0 || next > limit){
            interval_stop = events.max_phase_interval > 0 &&
                events.last_phase_time[particle] + events.max_phase_interval <= time + step_limit*step.h;
            if(interval_stop) endpoint_stop = false;
            break;
        }
        double hit[4];
        for(int j = 0; j < 4; ++j) hit[j] = double(T(value(step.state[j], next)));
        if constexpr(map_rhs_to_coord<id>() == CoordSys::Boozer){
            hit[0] = hypot(hit[0], hit[1]);
            hit[1] = step.theta(next);
            hit[2] = fmod(value(step.state[2], next)+step.zeta_offset, period);
            if(hit[2] < 0) hit[2] += period;
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
        for(int j = 0; j < 4; ++j) hit[j] = double(x_temp[(j+1)*PARTICLES_PER_BLOCK+p]);
        if constexpr(map_rhs_to_coord<id>() == CoordSys::Boozer){
            hit[0] = hypot(hit[0], hit[1]); hit[1] = step.theta(limit);
            hit[2] = fmod(hit[2]+step.zeta_offset, period);
            if(hit[2] < 0) hit[2] += period;
        }
        write_event(events, particle, time + limit*step.h, stop_index, hit);
    }
    if(endpoint_stop || root_stop || interval_stop || limit < fmin(1.0, (tmax-time)/step.h) ||
       (events.phases_stop && events.phase_counts[particle]) ||
       (events.max_phase_hits && events.phase_counts[particle] >= events.max_phase_hits)){
        events.end_times[particle] = interval_stop
            ? events.last_phase_time[particle] + events.max_phase_interval
            : time + limit*step.h;
    }
    // The next accepted step starts with wrapped coordinates. Preserve angle
    // winding internally, without depending on the trajectory saving cadence.
    if constexpr(map_rhs_to_coord<id>() == CoordSys::Boozer){
        double wrapped_theta = atan2(double(x_temp[2*PARTICLES_PER_BLOCK+p]),
                                     double(x_temp[PARTICLES_PER_BLOCK+p]));
        events.offsets[2*size_t(particle)] = period * round((step.theta(1)-wrapped_theta)/period);
        T z = x_temp[3*PARTICLES_PER_BLOCK+p];
        T wrapped = fmod(z, T(period)); if(wrapped < 0) wrapped += T(period);
        events.offsets[2*size_t(particle)+1] += double(z)-double(wrapped);
    }
}

#endif
