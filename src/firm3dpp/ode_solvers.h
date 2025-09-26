#ifndef ODE_SOLVERS_H
#define ODE_SOLVERS_H

#include <vector>
#include <array>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <limits>
#include <cmath>
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <boost/numeric/odeint.hpp>
# include "tracing.h" // For BaseRHS

/**
 * ODESolver Base Class
 * 
 * This is an abstract base class for all ODE solvers.
 Works with vector<double> state and BaseRHS& interface
 */
class ODESolver {
public:
    virtual ~ODESolver() = default;

    /**
     * Initialize the solver with initial conditions
     * 
     * @param y0: Initial state vector
     * @param t0: Initial time
     * @param h0: Initial timestep suggestion
     * @param rhs: The RHS function object
     * Note: the RHS may be needed for the solver to precompute things, 
     * but now RHS is passed both here and at do_step.
     */
    virtual void initialize(
        const vector<double>& y0,
        double t0,
        double h0,
        BaseRHS& rhs
    ) = 0;
    
    /**
     * Perform one integration step
     * 
     * @param rhs: The right-hand side function object
     * @return: A pair of (t_start, t_end) for the step taken
     */
    virtual std::pair<double, double> do_step(BaseRHS& rhs) = 0;
    
    /**
     * Calculate state at any time within the last step (dense output)
     * 
     * @param t: Time at which to evaluate the state
     * @param y_out: Output state vector
     * 
     */
    virtual void calc_state(double t, vector<double>& y_out) = 0;
    
    virtual double current_time() const = 0;
    
    virtual const vector<double>& current_state() const = 0;
    
    /**
     * Diagnostic functions
     * 
     */
    virtual int get_nsteps() const {
        throw std::logic_error("get_nsteps() not implemented for this solver");
    }
    
    virtual int get_ngood() const {
        throw std::logic_error("get_ngood() not implemented for this solver");
    }
    
    virtual int get_nbad() const {
        throw std::logic_error("get_nbad() not implemented for this solver");
    }
    
    virtual double get_hmin() const {
        throw std::logic_error("get_hmin() not implemented for this solver");
    }
    
    virtual double get_hmax() const {
        throw std::logic_error("get_hmax() not implemented for this solver");
    }
    
    virtual double get_hnext() const {
        throw std::logic_error("get_hnext() not implemented for this solver");
    }

protected:
    vector<double> current_state_;
    double current_time_;
    double h_;  // Current timestep
    
    int nsteps_ = 0;
    int ngood_ = 0;
    int nbad_ = 0;
    double hmin_ = std::numeric_limits<double>::max();
    double hmax_ = 0.0;
    double hnext_ = 0.0;
};

/**
 * Boost Dormand-Prince Wrapper
 * 
 * This wraps the existing Boost implementation to match new interface
 */
class DopriBoostSolver : public ODESolver {
public:
    using Stepper = boost::numeric::odeint::runge_kutta_dopri5<vector<double>>;
    using DenseStepper = typename boost::numeric::odeint::result_of::make_dense_output<Stepper>::type;
    
    // Wrapper for boost::odeint since it can't work with abstract BaseRHS directly
    struct RHSWrapper {
        BaseRHS& rhs;
        RHSWrapper(BaseRHS& r) : rhs(r) {}
        void operator()(const vector<double>& x, vector<double>& dxdt, double t) {
            rhs(x, dxdt, t);
        }
    };
    
    /**
     * Constructor
     * 
     * @param abstol: Absolute tolerance for error control
     * @param reltol: Relative tolerance for error control
     * @param h_max: Maximum allowed timestep
     */
    DopriBoostSolver(double abstol, double reltol, double h_max)
        : abstol_(abstol), reltol_(reltol), h_max_(h_max) {
        dense_stepper_ = boost::numeric::odeint::make_dense_output(
            abstol_, reltol_, h_max_, Stepper()
        );
    }
    
    void initialize(
        const vector<double>& y0,
        double t0,
        double h0,
        BaseRHS& rhs) override {
        this->current_state_ = y0;
        this->current_time_ = t0;
        this->h_ = h0;
        
        // Create wrapper and initialize dense stepper
        RHSWrapper rhs_wrapper(rhs);
        dense_stepper_.initialize(y0, t0, h0);
        
        // Reset diagnostics
        this->nsteps_ = 0;
        this->hmin_ = std::numeric_limits<double>::max();
        this->hmax_ = 0.0;
        this->hnext_ = h0;
    }
    
    /**
     * Perform one integration step
     */
    std::pair<double, double> do_step(BaseRHS& rhs) override {
        double t_start = dense_stepper_.current_time();
        
        // Create wrapper and perform the step
        RHSWrapper rhs_wrapper(rhs);
        auto step_result = dense_stepper_.do_step(rhs_wrapper);
        
        this->current_time_ = dense_stepper_.current_time();
        this->current_state_ = dense_stepper_.current_state();
        
        double t_end = std::get<1>(step_result);
        double h_taken = t_end - t_start;
        
        // Update diagnostics
        this->nsteps_++;
        if (h_taken > 0) {
            this->hmin_ = std::min(this->hmin_, h_taken);
            this->hmax_ = std::max(this->hmax_, h_taken);
            this->h_ = h_taken;
        }
        
        // Note: this is approximation since Boost
        // manages timestep internally 
        this->hnext_ = this->h_;
        
        return std::make_pair(t_start, t_end);
    }
    
    /**
     * Calculate state at arbitrary time (i.e. "dense output")
     */
    void calc_state(double t, vector<double>& y_out) override {
        dense_stepper_.calc_state(t, y_out);
    }
    
    double current_time() const override {
        return this->current_time_;
    }
    
    const vector<double>& current_state() const override {
        return this->current_state_;
    }
    
    // Boost doesn't track good/bad steps internally
    int get_nsteps() const override { return this->nsteps_; }
    
    int get_ngood() const override { 
        throw std::logic_error("get_ngood() not tracked by Boost solver - use get_nsteps() instead");
    }
    
    int get_nbad() const override { 
        throw std::logic_error("get_nbad() not tracked by Boost solver");
    }
    
    double get_hmin() const override { return this->hmin_; }
    double get_hmax() const override { return this->hmax_; }
    
    double get_hnext() const override { 
        // Boost manages this internally, so this is an approximation
        return this->hnext_;
    }

private:
    double abstol_;
    double reltol_;
    double h_max_;
    
    DenseStepper dense_stepper_;
};

/**
 * Dormand-Prince Implementation
 * 
 * Dormand-Prince method 
 * with minimum timestep support (h_min)
 */
class DormandPrinceSolver : public ODESolver {
public:
    /**
     * 
     * @param abstol: Absolute tolerance for error control
     * @param reltol: Relative tolerance for error control  
     * @param h_max: Maximum allowed timestep
     * @param h_min: Minimum allowed timestep
     */
    DormandPrinceSolver(double abstol, double reltol, double h_max, double h_min = 0.0)
        : abstol_(abstol), reltol_(reltol), h_max_(h_max), h_min_(h_min) {
        
        // Initialize Dormand-Prince coefficients (Butcher tableau)
        // These are the standard DP5(4) coefficients from Hairer & Wanner
        
        c2_ = 1.0/5.0;
        c3_ = 3.0/10.0;
        c4_ = 4.0/5.0;
        c5_ = 8.0/9.0;
        c6_ = 1.0;
        c7_ = 1.0;
        
        // Runge-Kutta matrix coefficients
        a21_ = 1.0/5.0;
        
        a31_ = 3.0/40.0;
        a32_ = 9.0/40.0;
        
        a41_ = 44.0/45.0;
        a42_ = -56.0/15.0;
        a43_ = 32.0/9.0;
        
        a51_ = 19372.0/6561.0;
        a52_ = -25360.0/2187.0;
        a53_ = 64448.0/6561.0;
        a54_ = -212.0/729.0;
        
        a61_ = 9017.0/3168.0;
        a62_ = -355.0/33.0;
        a63_ = 46732.0/5247.0;
        a64_ = 49.0/176.0;
        a65_ = -5103.0/18656.0;
        
        a71_ = 35.0/384.0;
        a72_ = 0.0;
        a73_ = 500.0/1113.0;
        a74_ = 125.0/192.0;
        a75_ = -2187.0/6784.0;
        a76_ = 11.0/84.0;
        
        // Error coefficients; difference between 5th and 4th order.
        e1_ = 35.0/384.0 - 5179.0/57600.0;
        e2_ = 0.0;
        e3_ = 500.0/1113.0 - 7571.0/16695.0;
        e4_ = 125.0/192.0 - 393.0/640.0;
        e5_ = -2187.0/6784.0 - (-92097.0/339200.0);
        e6_ = 11.0/84.0 - 187.0/2100.0;
        e7_ = 0.0 - 1.0/40.0;
        
        // Dense output coefficients
        d1_ = -12715105075.0/11282082432.0;
        d3_ = 87487479700.0/32700410799.0;
        d4_ = -10690763975.0/1880347072.0;
        d5_ = 701980252875.0/199316789632.0;
        d6_ = -1453857185.0/822651844.0;
        d7_ = 69997945.0/29380423.0;
    }
    
    void initialize(const vector<double>& y0, double t0, double h0, BaseRHS& rhs) override {
        this->current_state_ = y0;
        this->current_time_ = t0;
        this->h_ = h0;
        
        // Ensure initial timestep respects bounds
        this->h_ = std::min(this->h_, h_max_);
        if (h_min_ > 0) {
            this->h_ = std::max(this->h_, h_min_);
        }
        
        // Reset diagnostics
        this->nsteps_ = 0;
        this->ngood_ = 0;
        this->nbad_ = 0;
        this->hmin_ = std::numeric_limits<double>::max();
        this->hmax_ = 0.0;
        this->hnext_ = this->h_;
        
        // Pre-allocate work arrays for RK stages
        k1_ = y0;
        k2_ = y0;
        k3_ = y0;
        k4_ = y0;
        k5_ = y0;
        k6_ = y0;
        k7_ = y0;
        y_tmp_ = y0;
        y_err_ = y0;
        
        // Store for dense output
        y_old_ = y0;
        t_old_ = t0;
        h_old_ = 0.0;
        k1_for_dense_ = y0;
        k3_stored_ = y0;
        k4_stored_ = y0;
        k5_stored_ = y0;
        k6_stored_ = y0;
        k7_stored_ = y0;
        
        // Compute initial derivative
        rhs(this->current_state_, k1_, this->current_time_);
        k1_stored_ = k1_;
        k1_for_dense_ = k1_;  // For potential dense output at first step
    }
    
    std::pair<double, double> do_step(BaseRHS& rhs) override {
        const double SAFETY = 0.9;  // Safety factor for step size control
        const double ERRCON = 1.89e-4;  // Error constant
        const int MAX_TRIES = 500;  // Maximum attempts before giving up
        
        double t_start = this->current_time_;
        double h = this->hnext_;
        bool step_accepted = false;
        int tries = 0;
        
        double err_max = 0.0;
        while (!step_accepted && tries < MAX_TRIES) {
            tries++;
            
            // Ensure h respects bounds
            h = std::min(h, h_max_);
            if (h_min_ > 0) {
                h = std::max(h, h_min_);
            }
            
            // Stage 1 is already computed; i.e. the "FSAL" approach
            k1_ = k1_stored_;
            
            // Stage 2
            for (size_t i = 0; i < this->current_state_.size(); ++i) {
                y_tmp_[i] = this->current_state_[i] + h * a21_ * k1_[i];
            }
            rhs(y_tmp_, k2_, t_start + c2_ * h);
            
            // Stage 3
            for (size_t i = 0; i < this->current_state_.size(); ++i) {
                y_tmp_[i] = this->current_state_[i] + h * (a31_ * k1_[i] + a32_ * k2_[i]);
            }
            rhs(y_tmp_, k3_, t_start + c3_ * h);
            
            // Stage 4
            for (size_t i = 0; i < this->current_state_.size(); ++i) {
                y_tmp_[i] = this->current_state_[i] + h * (a41_ * k1_[i] + a42_ * k2_[i] + a43_ * k3_[i]);
            }
            rhs(y_tmp_, k4_, t_start + c4_ * h);
            
            // Stage 5
            for (size_t i = 0; i < this->current_state_.size(); ++i) {
                y_tmp_[i] = this->current_state_[i] + h * (a51_ * k1_[i] + a52_ * k2_[i] + 
                                                           a53_ * k3_[i] + a54_ * k4_[i]);
            }
            rhs(y_tmp_, k5_, t_start + c5_ * h);
            
            // Stage 6
            for (size_t i = 0; i < this->current_state_.size(); ++i) {
                y_tmp_[i] = this->current_state_[i] + h * (a61_ * k1_[i] + a62_ * k2_[i] + 
                                                           a63_ * k3_[i] + a64_ * k4_[i] + a65_ * k5_[i]);
            }
            rhs(y_tmp_, k6_, t_start + c6_ * h);
            
            // Stage 7 (5th order solution)
            vector<double> y_new = this->current_state_;
            for (size_t i = 0; i < y_new.size(); ++i) {
                y_new[i] = this->current_state_[i] + h * (a71_ * k1_[i] + a72_ * k2_[i] + 
                                                          a73_ * k3_[i] + a74_ * k4_[i] + 
                                                          a75_ * k5_[i] + a76_ * k6_[i]);
            }
            
            // Compute k7 for next step (FSAL)
            rhs(y_new, k7_, t_start + h);
            
            // Error estimate
            err_max = 0.0;
            for (size_t i = 0; i < this->current_state_.size(); ++i) {
                y_err_[i] = h * (e1_ * k1_[i] + e3_ * k3_[i] + e4_ * k4_[i] + 
                                e5_ * k5_[i] + e6_ * k6_[i] + e7_ * k7_[i]);
                
                double scale = abstol_ + reltol_ * std::max(std::abs(this->current_state_[i]), 
                                                            std::abs(y_new[i]));
                err_max = std::max(err_max, std::abs(y_err_[i]) / scale);
            }
            
            // Step size control
            if (err_max <= 1.0) {
                // Accept step
                step_accepted = true;
                
                // Store for dense output
                y_old_ = this->current_state_;
                t_old_ = this->current_time_;
                h_old_ = h;
                
                // Store k values for dense output
                k1_for_dense_ = k1_;  // Store k1 from beginning of this step
                k3_stored_ = k3_;
                k4_stored_ = k4_;
                k5_stored_ = k5_;
                k6_stored_ = k6_;
                k7_stored_ = k7_;
                
                // Update state
                this->current_state_ = y_new;
                this->current_time_ = t_start + h;
                
                // Store k7 for FSAL property (will be k1 of next step)
                k1_stored_ = k7_;
                
                // Update diagnostics
                this->nsteps_++;
                this->ngood_++;
                this->hmin_ = std::min(this->hmin_, h);
                this->hmax_ = std::max(this->hmax_, h);
                this->h_ = h;
                
                // Compute next timestep
                if (err_max > ERRCON) {
                    this->hnext_ = SAFETY * h * std::pow(err_max, -0.2);
                } else {
                    this->hnext_ = 5.0 * h;  // Maximum growth factor
                }
                this->hnext_ = std::min(this->hnext_, h_max_);
                
            } else {
                // Reject step
                this->nbad_++;
                
                // Reduce timestep
                double h_new = SAFETY * h * std::pow(err_max, -0.25);
                h_new = std::max(h_new, 0.1 * h);  // Maximum reduction factor
                
                if (h_min_ > 0 && h_new < h_min_) {
                    // Can't reduce further, accept anyway if h_min is set
                    step_accepted = true;
                    
                    // Store for dense output
                    y_old_ = this->current_state_;
                    t_old_ = this->current_time_;
                    h_old_ = h;
                    
                    // Store k values for dense output
                    // k1_ contains the k1 from the beginning of this step (set from k1_stored_ via FSAL)
                    k1_for_dense_ = k1_;  // Save k1 from beginning of this step for dense output
                    k3_stored_ = k3_;
                    k4_stored_ = k4_;
                    k5_stored_ = k5_;
                    k6_stored_ = k6_;
                    k7_stored_ = k7_;
                    
                    // Update state
                    this->current_state_ = y_new;
                    this->current_time_ = t_start + h;
                    
                    // Store k7 for FSAL property (will be k1 of next step)
                    k1_stored_ = k7_;
                    
                    // Update diagnostics
                    this->nsteps_++;
                    this->hmin_ = std::min(this->hmin_, h);
                    this->hmax_ = std::max(this->hmax_, h);
                    this->h_ = h;
                    this->hnext_ = h_min_;
                } else {
                    h = h_new;
                }
            }
        }
        
        if (!step_accepted) {
            throw std::runtime_error("DormandPrinceSolver: Failed to converge after " + 
                                   std::to_string(MAX_TRIES) + " attempts");
        }
        
        return std::make_pair(t_start, this->current_time_);
    }
    
    void calc_state(double t, vector<double>& y_out) override {
        // Dense output interpolation using 5th order polynomial
        // This allows evaluation at any point within the last step
        
        if (t < t_old_ - 1e-14 || t > this->current_time_ + 1e-14) {
            std::ostringstream oss;
            oss << std::scientific << "DormandPrinceSolver::calc_state: t outside last step interval. t - t_old_ = " << (t - t_old_) << ", current_time_ - t = " << (this->current_time_ - t);
            throw std::runtime_error(oss.str());
        }
        
        if (h_old_ == 0.0) {
            // No step taken yet
            y_out = this->current_state_;
            return;
        }
        
        double theta = (t - t_old_) / h_old_;
        double theta1 = 1.0 - theta;
        
        // Compute dense output coefficients
        vector<double> rcont1 = y_out, rcont2 = y_out, rcont3 = y_out, rcont4 = y_out, rcont5 = y_out;
        
        for (size_t i = 0; i < y_out.size(); ++i) {
            rcont1[i] = y_old_[i];
            double ydiff = this->current_state_[i] - y_old_[i];
            rcont2[i] = ydiff;
            double bspl = h_old_ * k1_for_dense_[i] - ydiff;
            rcont3[i] = bspl;
            rcont4[i] = ydiff - h_old_ * k7_stored_[i] - bspl;
            rcont5[i] = h_old_ * (d1_ * k1_for_dense_[i] + d3_ * k3_stored_[i] + d4_ * k4_stored_[i] +
                                  d5_ * k5_stored_[i] + d6_ * k6_stored_[i] + d7_ * k7_stored_[i]);
        }
        
        // Evaluate interpolation polynomial
        for (size_t i = 0; i < y_out.size(); ++i) {
            y_out[i] = rcont1[i] + theta * (rcont2[i] + theta1 * (rcont3[i] + 
                      theta * (rcont4[i] + theta1 * rcont5[i])));
        }
    }
    
    double current_time() const override {
        return this->current_time_;
    }
    
    const vector<double>& current_state() const override {
        return this->current_state_;
    }
    
    int get_nsteps() const override { return this->nsteps_; }
    int get_ngood() const override { return this->ngood_; }
    int get_nbad() const override { return this->nbad_; }
    double get_hmin() const override { return this->hmin_; }
    double get_hmax() const override { return this->hmax_; }
    double get_hnext() const override { return this->hnext_; }

private:
    double abstol_;
    double reltol_;
    double h_max_;
    double h_min_;
    
    // Butcher tableau coefficients
    double c2_, c3_, c4_, c5_, c6_, c7_;
    double a21_, a31_, a32_, a41_, a42_, a43_;
    double a51_, a52_, a53_, a54_;
    double a61_, a62_, a63_, a64_, a65_;
    double a71_, a72_, a73_, a74_, a75_, a76_;
    double e1_, e2_, e3_, e4_, e5_, e6_, e7_;
    double d1_, d3_, d4_, d5_, d6_, d7_;
    
    // Work arrays for Runge-Kutta stages
    vector<double> k1_, k2_, k3_, k4_, k5_, k6_, k7_;
    vector<double> y_tmp_, y_err_;
    vector<double> k1_stored_;  // For FSAL property
    
    // Dense output storage
    vector<double> y_old_;
    double t_old_, h_old_;
    vector<double> k1_for_dense_, k3_stored_, k4_stored_, k5_stored_, k6_stored_, k7_stored_;
};

/**
 * Factory function to create Boost-based Dormand-Prince solver
 */
std::unique_ptr<ODESolver> create_dopri_boost_solver(
    double abstol, double reltol, double h_max) {
    return std::make_unique<DopriBoostSolver>(abstol, reltol, h_max);
}

/**
 * Factory function to create Dormand-Prince solver
 */
std::unique_ptr<ODESolver> create_dormand_prince_solver(
    double abstol, double reltol, double h_max, double h_min = 0.0) {
    return std::make_unique<DormandPrinceSolver>(abstol, reltol, h_max, h_min);
}

#endif // ODE_SOLVERS_H
