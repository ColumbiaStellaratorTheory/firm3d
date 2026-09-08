#include <cuda_runtime.h>
#include <iostream>
#include "tracing.h"
#include <math.h>
#include "xtensor-python/pyarray.hpp"     // Numpy bindings
typedef xt::pyarray<double> PyArray;
#include "xtensor-python/pytensor.hpp"     // Numpy bindings
typedef xt::pytensor<double, 2, xt::layout_type::row_major> PyTensor;
using std::shared_ptr;
using std::vector;
namespace py = pybind11;

#define THREADS_PER_BLOCK 32
#define PARTICLES_PER_BLOCK 8
#define FULL_MASK 0xffffffff

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// enum used for templating
// https://stackoverflow.com/questions/9116267/how-can-i-use-an-enumeration-as-a-template-parameter
enum class RHS {GC_CartesianVacuum, GC_BoozerVacuum, GC_Boozer, GC_BoozerVacuumSAW, GC_BoozerNoKSAW};

enum class CoordSys {Cartesian, Boozer};

template<RHS id>
__host__ __device__ constexpr CoordSys map_rhs_to_coord(){
    if constexpr(id == RHS::GC_BoozerVacuum || id == RHS::GC_Boozer || id == RHS::GC_BoozerVacuumSAW || id == RHS::GC_BoozerNoKSAW){
        return CoordSys::Boozer;
    } else if constexpr (id == RHS::GC_CartesianVacuum) {
        return CoordSys::Cartesian;
    }
}

template<RHS id>
__host__ __device__ constexpr int map_rhs_to_n_interpolants(){
    if constexpr(id == RHS::GC_CartesianVacuum){
        return 7;
    } else if constexpr(id == RHS::GC_BoozerVacuum){
        return 6;
    } else if constexpr(id == RHS::GC_Boozer){
        return 12;
    } else if constexpr(id == RHS::GC_BoozerVacuumSAW || id == RHS::GC_BoozerNoKSAW){
        return 10; 
    }
}

// each rhs needs a different number of outputs. 
// GC rhs need 4 derivative components, Cartesian tracing needs to track the signed distance fn to boundary
template<RHS id>
__host__ __device__ constexpr int map_rhs_to_n_deriv_outputs(){
    if constexpr(id == RHS::GC_BoozerVacuum || id == RHS::GC_Boozer || id == RHS::GC_BoozerVacuumSAW || id == RHS::GC_BoozerNoKSAW){
        return 4;
    } else if constexpr(id == RHS::GC_CartesianVacuum){
        return 5;
    }
}

// this is a helper function to convert python arrays to C++ arrays
template <typename T>
__host__ T* create_array(py::array_t<T> x){
    py::buffer_info buf = x.request();
    T* arr = static_cast<T*>(buf.ptr);
    return arr;
} 

/* below are declarations of data that are stored in the constant memory cache on the GPU
 * they are referenced like global variables and need to be set here, or copied to before the kernel is launched
 * accesses to constant memory are serialized and broadcasted across a warp
 * used when data doesn't change for the lifetime of the program and all threads use the same value
 */

// position weights for Dormand-Prince 5 timesteps
__constant__ double dp5_wgts[7][7] = {
    {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
    {1.0/5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
    {3.0 / 40.0, 9.0 / 40.0, 0.0, 0.0, 0.0, 0.0},
    {44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0, 0.0, 0.0, 0.0, 0.0},
    {19372.0 / 6561.0, -25360.0 / 2187.0, 64448.0 / 6561.0, -212.0 / 729.0, 0.0, 0.0, 0.0},
    {9017.0 / 3168.0, -355.0 / 33.0, 46732.0 / 5247.0, 49.0 / 176.0,-5103.0 / 18656.0, 0.0, 0.0},
    {35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0, 11.0 / 84.0, 0.0}
};
// position weights for Dormand-Prince 5 timesteps
__constant__ double dp5_t_wgts[7] = {
    0.0, 1.0/5.0, 3.0/10.0, 4.0/5.0, 8.0/9.0, 1.0, 1.0
};

// error estimation weights for Dormand-Prince 5 timesteps
__constant__ double bhat_wgts[7] = {
    71.0/57600.0, 0.0, -71.0/16695.0, 71.0/1920.0,
    -17253.0/339200.0, 22.0/525.0, -1.0/40.0
};

// each RHS has an associated 3 dimensional coordinates system (x1, x2, x3)
// grid information for all 3 dimensions of the interpolant start, end, n_pts, 1/grid_size
__constant__ double grid_ranges_d[12];

// store the particle's mass, charge, absolute and relative tolerances for timestepping
__constant__ double mass_d, charge_d, atol_d, rtol_d; 
__constant__ int n_x2_d, n_x3_d, n_x23_d; // stores the number of interpolant cells in x2 and x3 direction, along with their product
__constant__ int nparticles_d; // number of particles being traced
__constant__ double v_total_d; // initial velocity

__constant__ double psi0_d; // used for Boozer RHS only
__constant__ double inv_psi0_charge_d; // used for Boozer RHS only, precompute 1 / charge_d * psi0_d
__constant__ double saw_srange_d[4]; // used for SAW RHS only

__constant__ bool rescale_abstol_var_d = true;
__constant__ bool is_test_d = false;

// global counter for workstealing
__device__ int next_particle_d;

// interpolate performs tricubic interpolation in the r, phi, z coordinates
// which we assume is on a regular grid
// the name of these coordinates only reflects the original cylindircal coordinates
// this function works in general
// 
// the n interpolant elements are written to out in order
// interpolant data is stored in data in 64 interpolation pt windows 
// with n contiguous entries at each point
// 
// shape values are precomputed in build_state and here we are computing the needed inner product
// cell_index_start stores the grid index for interpolation in the r, phi, z coordinates
// r_shape, phi_shape, z_shape store shape function elements
// nphi and nz indicate how many grid pts there are in phi and z directions
//
template <typename T, int n> __device__ void interpolate(T*  out, const T* __restrict__ data, const int* __restrict__ cell_index_start,
    const T* __restrict__ shape_fun_vals, const bool* __restrict__ is_valid){

    // 8 threads per particles, iterate over particles in the block
    for(int p=threadIdx.x / 8; p<PARTICLES_PER_BLOCK; p+= THREADS_PER_BLOCK/8){
        // cell_ids from build_state
        const int i = cell_index_start[3*p];
        const int j = cell_index_start[3*p + 1];
        const int k = cell_index_start[3*p + 2];

        // constant third dimension index for the thread
        const int kk = threadIdx.x % 4;

        // second dimension index for the thread, 0 -> 2 and 1 -> 3
        const int jj = (threadIdx.x / 4) % 2; 

        // compute the base offset for the interpolant data for this thread
        const int base_offset = 64*n*(i*n_x2_d * n_x3_d + j*n_x3_d + k) + 4*jj + kk;

        // each thread handles two j,k pairs, compute shape contributions
        const T shape_jk1 = shape_fun_vals[(8 + kk)*PARTICLES_PER_BLOCK + p] * shape_fun_vals[(4 + jj)*PARTICLES_PER_BLOCK + p];
        const T shape_jk2 = shape_fun_vals[(8 + kk)*PARTICLES_PER_BLOCK + p] * shape_fun_vals[(4 + jj + 2)*PARTICLES_PER_BLOCK + p];

        // only issue loads if the particle is still alive
        const bool p_valid = is_valid[p];

        // iterate over interpolant elements
        // in gc boozer vacuum, this is modB, dmodBds, dmodBdtheta, dmodBdzeta, G, iota
        for(int zz=0; zz<n; ++zz){
            T local_val = 0.0;
            if(p_valid){ // check particle is still alive
                for(int ii=0; ii<4; ++ii){
                    int base = base_offset + 16*ii ;
                    const T shape_i = shape_fun_vals[ii*PARTICLES_PER_BLOCK + p]; //shape_fun_vals[(8 + kk)*PARTICLES_PER_BLOCK + p];

                    local_val += shape_i * (shape_jk1 * data[base + 64*zz] + 
                                            shape_jk2 * data[base + 8 + 64*zz]);
                }
            }

            // warp reduction down to 4 elements (one per particle)
            for (int offset = 4; offset > 0; offset /= 2) {
                local_val += __shfl_down_sync(FULL_MASK, local_val, offset);
            }
            // one thread per particle writes the result, used by calc_derivs
            if(threadIdx.x % 8 == 0 && p_valid){
                out[PARTICLES_PER_BLOCK*zz + p] = local_val;
            }
        }
    }
}


// calc_derivs implementation for guiding center cartesian vacuum tracing
template <typename T, int deriv_id>
__device__ void rhs_GC_CartesianVacuum(T* derivs, const T* __restrict__ x_temp, const T* __restrict__ block_interpolants, const bool* __restrict__ symmetry_exploited, const T* __restrict__ mu){

    T x = x_temp[1*PARTICLES_PER_BLOCK];
    T y = x_temp[2*PARTICLES_PER_BLOCK];
    T z = x_temp[3*PARTICLES_PER_BLOCK];
    T v_par = x_temp[4*PARTICLES_PER_BLOCK];

    T B_r = block_interpolants[0*PARTICLES_PER_BLOCK];
    T B_phi = block_interpolants[1*PARTICLES_PER_BLOCK];
    T B_z = block_interpolants[2*PARTICLES_PER_BLOCK];
    T GradAbsB_r = block_interpolants[3*PARTICLES_PER_BLOCK];
    T GradAbsB_phi = block_interpolants[4*PARTICLES_PER_BLOCK];
    T GradAbsB_z = block_interpolants[5*PARTICLES_PER_BLOCK];

    if(symmetry_exploited[0]){
        B_r *= T(-1.0);
        GradAbsB_phi *= T(-1.0);
        GradAbsB_z *= T(-1.0);
    }

    T phi = atan2(y, x);
    T B_x = cos(phi) * B_r - sin(phi) * B_phi;
    T B_y = sin(phi) * B_r + cos(phi) * B_phi;
    T GradAbsB_x = cos(phi) * GradAbsB_r - sin(phi) * GradAbsB_phi;
    T GradAbsB_y = sin(phi) * GradAbsB_r + cos(phi) * GradAbsB_phi;

    T AbsB = sqrt(B_x*B_x + B_y*B_y + B_z*B_z);
    T v_perp2 = 2*mu[0]*AbsB;
    T fak1 = (v_par/AbsB);
    T fak2 = (T(mass_d)/(T(charge_d)*pow(AbsB, 3)))*(0.5*v_perp2 + v_par*v_par);

    T BcrossGradAbsB_elt = B_y*GradAbsB_z - B_z*GradAbsB_y;
    derivs[(5*deriv_id + 0)*PARTICLES_PER_BLOCK] = fak1*B_x + fak2*BcrossGradAbsB_elt;
    BcrossGradAbsB_elt = B_z*GradAbsB_x - B_x*GradAbsB_z;
    derivs[(5*deriv_id + 1)*PARTICLES_PER_BLOCK] = fak1*B_y + fak2*BcrossGradAbsB_elt;
    BcrossGradAbsB_elt = B_x*GradAbsB_y - B_y*GradAbsB_x;
    derivs[(5*deriv_id + 2)*PARTICLES_PER_BLOCK] = fak1*B_z + fak2*BcrossGradAbsB_elt;
    derivs[(5*deriv_id + 3)*PARTICLES_PER_BLOCK] = -mu[0]*(B_x*GradAbsB_x + B_y*GradAbsB_y + B_z*GradAbsB_z)/AbsB;
    // derivs[(6*deriv_id + 4)*PARTICLES_PER_BLOCK] = AbsB; // AbsB
    derivs[(5*deriv_id + 4)*PARTICLES_PER_BLOCK] = block_interpolants[6*PARTICLES_PER_BLOCK]; // boundary dist fn

}


// calc_derivs implementation for guiding center boozer vacuum tracing
template <typename T, int deriv_id> 
__device__ void rhs_GC_BoozerVacuum(T* derivs, const T* __restrict__ x_temp, const T* __restrict__ block_interpolants, const bool* __restrict__ symmetry_exploited, const T* __restrict__ mu){

    T x1 = x_temp[1*PARTICLES_PER_BLOCK];
    T x2 = x_temp[2*PARTICLES_PER_BLOCK];

    T inv_s = rhypot(x1, x2);
    T v_par = x_temp[4*PARTICLES_PER_BLOCK];

    T modB = block_interpolants[0*PARTICLES_PER_BLOCK];
    T dmodBds = block_interpolants[1*PARTICLES_PER_BLOCK];
    T dmodBdtheta = block_interpolants[2*PARTICLES_PER_BLOCK];
    T dmodBdzeta = block_interpolants[3*PARTICLES_PER_BLOCK];
    T G = block_interpolants[4*PARTICLES_PER_BLOCK];
    T iota = block_interpolants[5*PARTICLES_PER_BLOCK];
    T mu_val = mu[0];

    T modB_inv_G = modB / G;

    T sign = symmetry_exploited[0] ? (T)-1.0 : (T)1.0;
    dmodBdtheta *= sign;
    dmodBdzeta *= sign;


    T fak1 = T(mass_d)*v_par*v_par/modB + T(mass_d)*mu_val;
    T sdot = -dmodBdtheta*(fak1 * T(inv_psi0_charge_d));
    T tdot = dmodBds*(fak1 * T(inv_psi0_charge_d)) + iota*(v_par*modB_inv_G);

    derivs[(4*deriv_id + 0)*PARTICLES_PER_BLOCK] = sdot*x1*inv_s - x2*tdot;
    derivs[(4*deriv_id + 1)*PARTICLES_PER_BLOCK] = sdot*x2*inv_s + x1*tdot;
    derivs[(4*deriv_id + 2)*PARTICLES_PER_BLOCK] = (v_par*modB_inv_G);
    derivs[(4*deriv_id + 3)*PARTICLES_PER_BLOCK] = -(iota*dmodBdtheta + dmodBdzeta)*mu_val*modB_inv_G;
 
}


// calc_derivs implementation for general guiding center Boozer tracing (with K != 0)
// The equations in this function match those for the CPU tracing at
// tracing.cpp::GuidingCenterBoozerRHS
template<typename T, int deriv_id>
__device__ void rhs_GC_Boozer(T* derivs, const T* __restrict__ x_temp, const T* __restrict__ block_interpolants, const bool* __restrict__ symmetry_exploited, const T* __restrict__ mu){

    T x1 = x_temp[1*PARTICLES_PER_BLOCK];
    T x2 = x_temp[2*PARTICLES_PER_BLOCK];

    T s = sqrt(x1*x1 + x2*x2);
    T theta = atan2(x2, x1);
    T zeta = x_temp[3*PARTICLES_PER_BLOCK];
    T v_par = x_temp[4*PARTICLES_PER_BLOCK];

    T modB = block_interpolants[0*PARTICLES_PER_BLOCK];
    T dmodBdpsi = block_interpolants[1*PARTICLES_PER_BLOCK] / T(psi0_d);
    T dmodBdtheta = block_interpolants[2*PARTICLES_PER_BLOCK];
    T dmodBdzeta = block_interpolants[3*PARTICLES_PER_BLOCK];
    T G = block_interpolants[4*PARTICLES_PER_BLOCK];
    T dGdpsi = block_interpolants[5*PARTICLES_PER_BLOCK] / T(psi0_d);
    T I = block_interpolants[6*PARTICLES_PER_BLOCK];
    T dIdpsi = block_interpolants[7*PARTICLES_PER_BLOCK] / T(psi0_d);
    T iota = block_interpolants[8*PARTICLES_PER_BLOCK];
    T K = block_interpolants[9*PARTICLES_PER_BLOCK];
    T dKdtheta = block_interpolants[10*PARTICLES_PER_BLOCK];
    T dKdzeta = block_interpolants[11*PARTICLES_PER_BLOCK];

    T mu_val = mu[0];

    if(symmetry_exploited[0]){
        dmodBdtheta *= T(-1.0);
        dmodBdzeta *= T(-1.0);
        K *= T(-1.0);
    }

    // General guiding center equations (mode='gc')
    // C = - m v|| K,zeta /|B| - q iota + m v|| G' / |B|
    // F = - m v|| K,theta /|B| + q + m v|| I' / |B|
    // D = (F G - C I) / iota
    
    T C = -T(mass_d) * v_par * dKdzeta / modB - T(charge_d) * iota + T(mass_d) * v_par * dGdpsi / modB;
    T F = -T(mass_d) * v_par * dKdtheta / modB + T(charge_d) + T(mass_d) * v_par * dIdpsi / modB;
    T D = (F * G - C * I) / iota;

    T fak1 = T(mass_d) * v_par * v_par / modB + T(mass_d) * mu_val;
    
    // sdot = (I |B|,zeta - G |B|,theta) m (v||^2/|B| + mu) / (iota D psi0)
    T sdot = (I * dmodBdzeta - G * dmodBdtheta) * fak1 / (iota * D * T(psi0_d));
    
    // tdot = ((G |B|,psi - K |B|,zeta) m (v||^2/|B| + mu) - C v|| |B|) / (iota D)
    T tdot = ((G * dmodBdpsi - K * dmodBdzeta) * fak1 - C * v_par * modB) / (iota * D);
    
    // zetadot = (F v|| |B| - (|B|,psi I - |B|,theta K) m (v||^2/|B| + mu)) / (iota D)
    T zetadot = (F * v_par * modB - (dmodBdpsi * I - dmodBdtheta * K) * fak1) / (iota * D);
    
    // v||dot = (C |B|,theta - F |B|,zeta) mu |B| / (iota D)
    T vpardot = (C * dmodBdtheta - F * dmodBdzeta) * mu_val * modB / (iota * D);

    derivs[(4*deriv_id + 0)*PARTICLES_PER_BLOCK] = sdot*cos(theta) - s*sin(theta)*tdot;
    derivs[(4*deriv_id + 1)*PARTICLES_PER_BLOCK] = sdot*sin(theta) + s*cos(theta)*tdot;
    derivs[(4*deriv_id + 2)*PARTICLES_PER_BLOCK] = zetadot;
    derivs[(4*deriv_id + 3)*PARTICLES_PER_BLOCK] = vpardot;
};

// calc_derivs implementation for guiding center boozer vacuum tracing with Shear Alfven Waves
template <typename T, int deriv_id>
__device__ void rhs_GC_BoozerVacuumSAW(T* derivs, const T* __restrict__ x_temp, const T* __restrict__ block_interpolants, const bool* __restrict__ symmetry_exploited, const T* __restrict__ mu,
                                         T saw_omega, int* saw_m, int* saw_n, T* saw_phihats, int saw_nharmonics){

    T time = x_temp[0];
    T x1 = x_temp[1*PARTICLES_PER_BLOCK];
    T x2 = x_temp[2*PARTICLES_PER_BLOCK];

    T s = sqrt(x1*x1 + x2*x2);
    T theta = atan2(x2, x1);
    T zeta = x_temp[3*PARTICLES_PER_BLOCK];
    T v_par = x_temp[4*PARTICLES_PER_BLOCK];

    T modB = block_interpolants[0*PARTICLES_PER_BLOCK];
    T dmodBdpsi = block_interpolants[1*PARTICLES_PER_BLOCK] / T(psi0_d);
    T dmodBdtheta = block_interpolants[2*PARTICLES_PER_BLOCK];
    T dmodBdzeta = block_interpolants[3*PARTICLES_PER_BLOCK];
    T G = block_interpolants[4*PARTICLES_PER_BLOCK];
    T dGdpsi = block_interpolants[5*PARTICLES_PER_BLOCK] / T(psi0_d);
    T I = block_interpolants[6*PARTICLES_PER_BLOCK];
    T dIdpsi = block_interpolants[7*PARTICLES_PER_BLOCK] / T(psi0_d);
    T iota = block_interpolants[8*PARTICLES_PER_BLOCK];
    T diotadpsi = block_interpolants[9*PARTICLES_PER_BLOCK] / T(psi0_d);

    T mu_val = mu[0];

    if(symmetry_exploited[0]){
        dmodBdtheta *= T(-1.0);
        dmodBdzeta *= T(-1.0);
    }

    // accumulate over harmonics
    int s_index = (s - saw_srange_d[0]) / (saw_srange_d[3]);
    s_index = min(s_index, (int)saw_srange_d[2]-1);
    T s_diff = s - s_index*saw_srange_d[3];

    // rhs values from SAW 
    T dphidpsi = 0.0;
    T dphidtheta = 0.0;
    T dphidzeta = 0.0;

    T dalphadpsi = 0.0;
    T dalphadtheta = 0.0;
    T alphadot = 0.0;

    for(int i=0; i<saw_nharmonics; ++i){
        T left_phihat = saw_phihats[s_index*saw_nharmonics + i];
        T right_phihat = saw_phihats[min(s_index+1, (int)saw_srange_d[2]-1)*saw_nharmonics + i];
        T s_slope = (right_phihat - left_phihat) / saw_srange_d[3];

        int m = saw_m[i];
        int n = saw_n[i];
        T alpha_fac = (iota *m - n) / (saw_omega * G);
        T dalpha_fac_dpsi = diotadpsi * m / (saw_omega * G);

        T pt_cos = cos(m*theta - n*zeta + saw_omega*time);
        T pt_sin = sin(m*theta - n*zeta + saw_omega*time);

        T phihat_i = left_phihat + s_slope*(s_diff);
        T dphihatdpsi = s_slope / T(psi0_d);

        T phi_i = phihat_i * pt_sin;
        T dphidpsi_i = dphihatdpsi * pt_sin;
        T phidot_i = phihat_i * pt_cos * saw_omega;
        T dphidtheta_i = phidot_i * (m / saw_omega);
        T dphidzeta_i = -phidot_i * (n / saw_omega);

        T alphadot_i = -phidot_i * alpha_fac;
        T dalphadpsi_i = -dphidpsi_i * alpha_fac - phi_i*dalpha_fac_dpsi;
        T dalphadtheta_i = -dphidtheta_i * alpha_fac;

        dphidpsi += dphidpsi_i;
        dphidtheta += dphidtheta_i;
        dphidzeta += dphidzeta_i;

        alphadot += alphadot_i;
        dalphadpsi += dalphadpsi_i;
        dalphadtheta += dalphadtheta_i;
        
    }

    T fak1 = T(mass_d)*v_par*v_par/modB + T(mass_d)*mu_val;

    T sdot = (-dmodBdtheta*fak1/T(charge_d) + dalphadtheta*modB*v_par - dphidtheta) / T(psi0_d);
    T tdot = (dmodBdpsi*fak1 / T(charge_d)) + (iota - dalphadpsi*G)*v_par*modB / G + dphidpsi;

    derivs[(4*deriv_id + 0)*PARTICLES_PER_BLOCK] = sdot*cos(theta) - s * sin(theta) * tdot;
    derivs[(4*deriv_id + 1)*PARTICLES_PER_BLOCK] = sdot*sin(theta) + s*cos(theta)*tdot;
    derivs[(4*deriv_id + 2)*PARTICLES_PER_BLOCK] = v_par*modB/G;
    derivs[(4*deriv_id + 3)*PARTICLES_PER_BLOCK] = -modB/(G*T(mass_d)) * (T(mass_d)*mu_val*(dmodBdzeta + dalphadtheta*dmodBdpsi*G \
                + dmodBdtheta*(iota - dalphadpsi*G)) + T(charge_d)*(alphadot*G \
                + dalphadtheta*G*dphidpsi + (iota - dalphadpsi*G)*dphidtheta + dphidzeta)) \
                + v_par/modB * (dmodBdtheta*dphidpsi - dmodBdpsi*dphidtheta);

};


// calc_derivs implementation for guiding center boozer NoK tracing with Shear Alfven Waves
template <typename T, int deriv_id> 
__device__ void rhs_GC_BoozerNoKSAW(T* derivs, const T* __restrict__ x_temp, const T* __restrict__ block_interpolants, const bool* __restrict__ symmetry_exploited, const T* __restrict__ mu,
                                     T saw_omega, int* saw_m, int* saw_n, T* saw_phihats, int saw_nharmonics){

    T time = x_temp[0];
    T x1 = x_temp[1*PARTICLES_PER_BLOCK];
    T x2 = x_temp[2*PARTICLES_PER_BLOCK];

    T s = sqrt(x1*x1 + x2*x2);
    T theta = atan2(x2, x1);
    T zeta = x_temp[3*PARTICLES_PER_BLOCK];
    T v_par = x_temp[4*PARTICLES_PER_BLOCK];

    T modB = block_interpolants[0*PARTICLES_PER_BLOCK];
    T dmodBdpsi = block_interpolants[1*PARTICLES_PER_BLOCK] / T(psi0_d);
    T dmodBdtheta = block_interpolants[2*PARTICLES_PER_BLOCK];
    T dmodBdzeta = block_interpolants[3*PARTICLES_PER_BLOCK];
    T G = block_interpolants[4*PARTICLES_PER_BLOCK];
    T dGdpsi = block_interpolants[5*PARTICLES_PER_BLOCK] / T(psi0_d);
    T I = block_interpolants[6*PARTICLES_PER_BLOCK];
    T dIdpsi = block_interpolants[7*PARTICLES_PER_BLOCK] / T(psi0_d);
    T iota = block_interpolants[8*PARTICLES_PER_BLOCK];
    T diotadpsi = block_interpolants[9*PARTICLES_PER_BLOCK] / T(psi0_d);

    T mu_val = mu[0];

    if(symmetry_exploited[0]){
        dmodBdtheta *= T(-1.0);
        dmodBdzeta *= T(-1.0);
    }

    // accumulate over harmonics
    int s_index = (s - saw_srange_d[0]) / (saw_srange_d[3]);
    s_index = min(s_index, (int)saw_srange_d[2]-1);
    T s_diff = s - s_index*saw_srange_d[3];

    // rhs values from SAW 
    T dphidpsi = 0.0;
    T dphidtheta = 0.0;
    T dphidzeta = 0.0;
    T dalphadzeta = 0.0;

    T alpha = 0.0;
    T dalphadpsi = 0.0;
    T dalphadtheta = 0.0;
    T alphadot = 0.0;

    for(int i=0; i<saw_nharmonics; ++i){
        T left_phihat = saw_phihats[s_index*saw_nharmonics + i];
        T right_phihat = saw_phihats[min(s_index+1, (int)saw_srange_d[2]-1)*saw_nharmonics + i];
        T s_slope = (right_phihat - left_phihat) / saw_srange_d[3];

        int m = saw_m[i];
        int n = saw_n[i];
        T alpha_fac = (iota *m - n) / (saw_omega * (G + iota*I));
        T dalpha_fac_dpsi = diotadpsi * m / (saw_omega * (G + iota*I)) - alpha_fac / (G+iota*I) * (dGdpsi + diotadpsi*I + iota*dIdpsi);

        T pt_cos = cos(m*theta - n*zeta + saw_omega*time);
        T pt_sin = sin(m*theta - n*zeta + saw_omega*time);

        T phihat_i = left_phihat + s_slope*(s_diff);
        T dphihatdpsi = s_slope / T(psi0_d);

        T phi_i = phihat_i * pt_sin;
        T dphidpsi_i = dphihatdpsi * pt_sin;
        T phidot_i = phihat_i * pt_cos * saw_omega;
        T dphidtheta_i = phidot_i * (m / saw_omega);
        T dphidzeta_i = -phidot_i * (n / saw_omega);

        T alpha_i = -phi_i*alpha_fac;
        T alphadot_i = -phidot_i * alpha_fac;
        T dalphadpsi_i = -dphidpsi_i * alpha_fac - phi_i*dalpha_fac_dpsi;
        T dalphadtheta_i = -dphidtheta_i * alpha_fac;
        T dalphadzeta_i = -dphidzeta_i*alpha_fac;

        dphidpsi += dphidpsi_i;
        dphidtheta += dphidtheta_i;
        dphidzeta += dphidzeta_i;

        alpha += alpha_i;
        alphadot += alphadot_i;
        dalphadpsi += dalphadpsi_i;
        dalphadtheta += dalphadtheta_i;
        dalphadzeta += dalphadzeta_i;
    }
    T fak1 = T(mass_d)*v_par*v_par/modB + T(mass_d)*mu_val;
    T denom = (T(charge_d)*(G + I*(-alpha*dGdpsi + iota) + alpha*G*dIdpsi)
            + T(mass_d)*v_par/modB * (-dGdpsi*I + G*dIdpsi)); 
    T sdot = (-G*dphidtheta*T(charge_d) + I*dphidzeta*T(charge_d) + modB*T(charge_d)*v_par*(dalphadtheta*G-dalphadzeta*I) + (-dmodBdtheta*G + dmodBdzeta*I)*fak1)/(denom*T(psi0_d));
    T tdot = (G*T(charge_d)*dphidpsi + modB*T(charge_d)*v_par*(-dalphadpsi*G - alpha*dGdpsi + iota) - dGdpsi*T(mass_d)*v_par*v_par \
                    + dmodBdpsi*G*fak1)/denom;
    derivs[(4*deriv_id + 0)*PARTICLES_PER_BLOCK] = sdot*cos(theta) - s * sin(theta) * tdot;
    derivs[(4*deriv_id + 1)*PARTICLES_PER_BLOCK] = sdot*sin(theta) + s*cos(theta)*tdot;
    derivs[(4*deriv_id + 2)*PARTICLES_PER_BLOCK] = v_par*modB/G;
    derivs[(4*deriv_id + 3)*PARTICLES_PER_BLOCK] = (modB*T(charge_d)/T(mass_d) * ( -T(mass_d)*mu_val * (dmodBdzeta*(1 + dalphadpsi*I + alpha*dIdpsi) \
                    + dmodBdpsi*(dalphadtheta*G - dalphadzeta*I) + dmodBdtheta*(iota - alpha*dGdpsi - dalphadpsi*G)) \
                    - T(charge_d)*(alphadot*(G + I*(iota - alpha*dGdpsi) + alpha*G*dIdpsi) \
                    + (dalphadtheta*G - dalphadzeta*I)*dphidpsi \
                    + (iota - alpha*dGdpsi - dalphadpsi*G)*dphidtheta \
                    + (1 + alpha*dIdpsi + dalphadpsi*I)*dphidzeta)) \
                    + T(charge_d)*v_par/modB * ((dmodBdtheta*G - dmodBdzeta*I)*dphidpsi \
                    + dmodBdpsi*(I*dphidzeta - G*dphidtheta)) \
                    + v_par*(T(mass_d)*mu_val*(dmodBdtheta*dGdpsi - dmodBdzeta*dIdpsi) \
                    + T(charge_d)*(alphadot*(dGdpsi*I-G*dIdpsi) + dGdpsi*dphidtheta - dIdpsi*dphidzeta)))/denom;

};

// calc_derivs computes the derivatives at points for which the corresponding
// i,j,k indices and shape functions have been precomputed
// the results are stored in the appropriate region of derivs
//
// this function is templated across rhs options
template<typename T, RHS id, int deriv_id, typename... Args>  
__device__ void calc_derivs(T* derivs, const T* __restrict__ quadpts_arr, const T* __restrict__ x_temp, const bool* __restrict__ symmetry_exploited, 
                                const int* __restrict__ cell_index_start, const T* __restrict__ shape_fun_vals, const T* __restrict__ mu, const bool* __restrict__ is_valid, 
                                // optional parameters for SAW cases
                                T saw_omega = 0, int* saw_m = nullptr, int* saw_n = nullptr, 
                                T* saw_phihats = nullptr, int saw_nharmonics = 0){
    
    
    constexpr int n = map_rhs_to_n_interpolants<id>();
    __shared__ T block_interpolants[n*PARTICLES_PER_BLOCK];
    interpolate<T, n>(block_interpolants, quadpts_arr, cell_index_start, shape_fun_vals, is_valid);
    __syncthreads();

    if(threadIdx.x < PARTICLES_PER_BLOCK && is_valid[threadIdx.x]){
        if constexpr (id == RHS::GC_CartesianVacuum){
            rhs_GC_CartesianVacuum<T, deriv_id>(derivs, x_temp, block_interpolants + threadIdx.x, symmetry_exploited, mu);
        } else if constexpr(id == RHS::GC_BoozerVacuum){
            rhs_GC_BoozerVacuum<T, deriv_id>(derivs, x_temp, block_interpolants + threadIdx.x, symmetry_exploited, mu);
        } else if constexpr(id == RHS::GC_Boozer){
            rhs_GC_Boozer<T, deriv_id>(derivs, x_temp, block_interpolants + threadIdx.x, symmetry_exploited, mu);
        } else if constexpr(id == RHS::GC_BoozerVacuumSAW){
            rhs_GC_BoozerVacuumSAW<T, deriv_id>(derivs, x_temp, block_interpolants + threadIdx.x, symmetry_exploited, mu,
                saw_omega, saw_m, saw_n, saw_phihats, saw_nharmonics);
        } else if constexpr(id == RHS::GC_BoozerNoKSAW){
            rhs_GC_BoozerNoKSAW<T, deriv_id>(derivs, x_temp, block_interpolants + threadIdx.x, symmetry_exploited, mu,
                saw_omega, saw_m, saw_n, saw_phihats,saw_nharmonics);
        }
    }
};




// map to grid takes a particle location and maps it to the interpolation grid
// this is where stellarator symmetry is exploited
// and points outside of the magnetic field are mapped to the nearest cell
//
// this function is templated across rhs implementations but it's possible
// that this should be considered to be the implementation for a coordinate
// system independent of rhs
//
// interp_pt stores where the interpolant should be evaluated (in the grid)
// xyz stores the particle's current location via the x_temp array in shared
//  memory. 
// symmetry_exploited is a bool indicating whether stellarator symmetry was exploited
// there's an option for optional parameters

// map_to_grid implementation for Cartesian tracing
template <typename T>
__device__ void map_to_grid_cartesian(T* interp_pt, T* x_temp, bool* symmetry_exploited){
    T x = x_temp[1*PARTICLES_PER_BLOCK + threadIdx.x];
    T y = x_temp[2*PARTICLES_PER_BLOCK + threadIdx.x];
    T z = x_temp[3*PARTICLES_PER_BLOCK + threadIdx.x];

    // convert to cylindrical coordinates for interpolation
    T r = sqrt(x*x + y*y);
    T phi = atan2(y, x);

    // restrict phi to [0, 2pi / nfp]
    T period = grid_ranges_d[5];
    phi = fmod(phi, period);
    phi += period*(phi < 0);

    // exploit stellarator symmetry
    symmetry_exploited[threadIdx.x] = z < 0;
    if(symmetry_exploited[threadIdx.x]){
        z = -z;
        phi = 2*M_PI - phi;
        phi = fmod(phi, period);
        phi += period*(phi < 0);
    }

    interp_pt[threadIdx.x] = r;
    interp_pt[PARTICLES_PER_BLOCK + threadIdx.x] = phi;
    interp_pt[2*PARTICLES_PER_BLOCK + threadIdx.x] = z;
} 

// map_to_grid implementation for Boozer tracing
template <typename T>
__device__ void map_to_grid_boozer(T* interp_pt, T* x_temp, bool* symmetry_exploited){

    T x1 = x_temp[1*PARTICLES_PER_BLOCK + threadIdx.x];
    T x2 = x_temp[2*PARTICLES_PER_BLOCK + threadIdx.x];
    T s = hypot(x1, x2);
    interp_pt[threadIdx.x] = s;
    symmetry_exploited[threadIdx.x] = x2 < 0;

    // we want to exploit periodicity in the B-field, but leave sin(theta) unchanged
    // compute the following without fmod atan2 in [-pi, pi]
    // T t = fmod(theta, T(2*M_PI));
    // t += 2*M_PI*(t < 0);
    T t = atan2(fabs(x2), x1);
    interp_pt[PARTICLES_PER_BLOCK + threadIdx.x] = t;


    T z = x_temp[3*PARTICLES_PER_BLOCK + threadIdx.x]; // zeta
    // we can modify z because it's only used to access the B-field location
    T period = grid_ranges_d[9];
    z = fmod(z, period);
    z += period*(z < 0);

    // exploit stellarator symmetry
    if(symmetry_exploited[threadIdx.x]){
        z = period - z;

    }
    interp_pt[2*PARTICLES_PER_BLOCK + threadIdx.x] = z;
}


template<typename T, CoordSys coord>
__device__ void map_to_grid(T* interp_pt, T* xyz, bool* symmetry_exploited){
    if constexpr (coord == CoordSys::Cartesian){
        map_to_grid_cartesian(interp_pt, xyz, symmetry_exploited);
    } else if constexpr (coord == CoordSys::Boozer){
        map_to_grid_boozer(interp_pt, xyz, symmetry_exploited);
    }
};                                    



// build_state is part of the DP5 implementation
template <typename T, RHS id, int deriv_id>
__device__ void build_state(T* x_temp, bool* symmetry_exploited, int* cell_index_start,
                            T* shape_fun_vals, const T* __restrict__ state, const T* __restrict__ derivs, 
                            const T* __restrict__ t, const T* __restrict__ dt, const bool* __restrict__ is_valid){

    // store time
    if(threadIdx.x < PARTICLES_PER_BLOCK && is_valid[threadIdx.x]){
        x_temp[threadIdx.x] = t[threadIdx.x] + T(dp5_t_wgts[deriv_id])*dt[threadIdx.x];
    }

    // store current location in x_temp
    for (int idx=threadIdx.x; idx<4*PARTICLES_PER_BLOCK; idx+=PARTICLES_PER_BLOCK) {
        int particle_id = idx % PARTICLES_PER_BLOCK;
        int state_var = idx / PARTICLES_PER_BLOCK;

        x_temp[(state_var+1)*PARTICLES_PER_BLOCK + particle_id] = state[state_var*PARTICLES_PER_BLOCK + particle_id];
        T dt_particle = dt[particle_id];
        for(int j=0; j<deriv_id; ++j){
            constexpr int n_deriv_entries = map_rhs_to_n_deriv_outputs<id>();
            x_temp[(state_var+1)*PARTICLES_PER_BLOCK + particle_id] += dt_particle * T(dp5_wgts[deriv_id][j]) * derivs[(n_deriv_entries*j+state_var)*PARTICLES_PER_BLOCK + particle_id];
        }
    }
    __syncthreads();
    

    // one thread per particle maps the position to the interpolant grid.
    __shared__ T interp_pt[3*PARTICLES_PER_BLOCK];

    if(threadIdx.x < PARTICLES_PER_BLOCK && is_valid[threadIdx.x]){
        constexpr CoordSys coord = map_rhs_to_coord<id>();
        map_to_grid<T, coord>(interp_pt, x_temp, symmetry_exploited);
    }
    __syncthreads();

    // threads map each coordinate for each particle and compute its shape functions.
    int particle_id = threadIdx.x % PARTICLES_PER_BLOCK;
    if(threadIdx.x < 3*PARTICLES_PER_BLOCK && is_valid[particle_id]){ // assumes 32 threads per block, 8 particles per block (3*8 = 24 < 32)
        int coord_id = threadIdx.x / PARTICLES_PER_BLOCK;

        T value = interp_pt[threadIdx.x];
        T inv_grid_size = grid_ranges_d[coord_id*4 + 3];
        T min_bound = grid_ranges_d[coord_id*4 + 0];

        T raw_offset = (value - min_bound) * inv_grid_size;
        int index = 3*((int) raw_offset / 3);
        index = min(index, (int)grid_ranges_d[coord_id*4 + 2]-4);
        index = max(index, 0);

        T value_rel = raw_offset - (T) index;

        // compute 4 shape values for this coordinate and particle
        T x_minus_1 = value_rel - (T)1.0;
        T x_minus_2 = value_rel - (T)2.0;
        T x_minus_3 = value_rel - (T)3.0;

        constexpr T one_sixth = (T) (1.0/6.0);
        constexpr T one_half = (T) 0.5;

        // compute shared terms for shape functions
        T prod23 = x_minus_2 * x_minus_3;
        T prodx1 = value_rel * x_minus_1;
        int base_idx = (coord_id*4)*PARTICLES_PER_BLOCK + particle_id;

        shape_fun_vals[base_idx + 0] = - one_sixth * x_minus_1 * prod23;
        shape_fun_vals[base_idx + 1*PARTICLES_PER_BLOCK] = one_half * value_rel * prod23;
        shape_fun_vals[base_idx + 2*PARTICLES_PER_BLOCK] = -one_half * prodx1 * x_minus_3;
        shape_fun_vals[base_idx + 3*PARTICLES_PER_BLOCK] = one_sixth * prodx1 * x_minus_2;

        cell_index_start[3*particle_id + coord_id] = index/3;
    }
    __syncthreads();

};


template<typename T>
__device__ void max_stepsize_cartesian(T* dtmax, T* loc){
    T x = loc[1*PARTICLES_PER_BLOCK + threadIdx.x];
    T y = loc[2*PARTICLES_PER_BLOCK + threadIdx.x];
    T z = loc[3*PARTICLES_PER_BLOCK + threadIdx.x];
    T v_par = loc[4*PARTICLES_PER_BLOCK + threadIdx.x];

    T r = sqrt(x*x + y*y);
    dtmax[threadIdx.x] = r*0.5*T(M_PI) / T(v_total_d);
}


template<typename T>
__device__ void max_stepsize_boozer(T* dtmax, T* interpolants){
    T modB = interpolants[PARTICLES_PER_BLOCK*0 + threadIdx.x];
    T G = interpolants[PARTICLES_PER_BLOCK*4 + threadIdx.x];
    dtmax[threadIdx.x] = (G / modB)*0.5*T(M_PI) / T(v_total_d);
}

// calculate maximum allowable timestep to allow at most a quarter of a revolution per step
template<typename T, CoordSys coord>
__device__ void calc_max_timestep_size(T* dtmax, T* loc, T* interpolants){
    if constexpr (coord == CoordSys::Cartesian){
        max_stepsize_cartesian(dtmax, loc);
    } else if constexpr (coord == CoordSys::Boozer){
        max_stepsize_boozer(dtmax, interpolants);
    } else{
        printf("max timestep size calculation not implemented for this coordinate system\n");
    }
};



// set up particles for tracing
// use the derivatives function to calculate mu, max step size
// store these values for the remainder of tracing
template<typename T, RHS id, typename... Args>
__device__ void setup_particle(T* mu, T* t, T* dt, T* dtmax, T* x_temp, bool* symmetry_exploited, int* cell_index_start,
                            const T* __restrict__ quad_pts, T* shape_fun_vals, T* state, T* derivs,
                            bool* is_valid, Args... args){

    if(threadIdx.x < PARTICLES_PER_BLOCK && is_valid[threadIdx.x]){
        // t[threadIdx.x] = 0.0;
        symmetry_exploited[threadIdx.x] = false;
    }
    build_state<T, id, 0>(x_temp, symmetry_exploited, cell_index_start,
                        shape_fun_vals, state, derivs, t, dt, is_valid);
    __syncthreads();
    constexpr int n = map_rhs_to_n_interpolants<id>();
    __shared__ T block_interpolants[n*PARTICLES_PER_BLOCK];
    interpolate<T, n>(block_interpolants, quad_pts, cell_index_start, shape_fun_vals, is_valid);
    __syncthreads();

    if(threadIdx.x < PARTICLES_PER_BLOCK && is_valid[threadIdx.x]){
        T v_par = state[3*PARTICLES_PER_BLOCK + threadIdx.x];
        T v_perp2 = T(v_total_d)*T(v_total_d) - v_par*v_par;
        
        T modB;
        constexpr CoordSys coord = map_rhs_to_coord<id>();
        if constexpr(coord == CoordSys::Boozer){
           modB = block_interpolants[0*PARTICLES_PER_BLOCK + threadIdx.x];
        } else if constexpr(coord == CoordSys::Cartesian){

            T x = x_temp[1*PARTICLES_PER_BLOCK+threadIdx.x];
            T y = x_temp[2*PARTICLES_PER_BLOCK+threadIdx.x];
            T phi = atan2(y, x);

            T B_r = block_interpolants[0*PARTICLES_PER_BLOCK + threadIdx.x];
            T B_phi = block_interpolants[1*PARTICLES_PER_BLOCK + threadIdx.x];
            T B_z = block_interpolants[2*PARTICLES_PER_BLOCK + threadIdx.x];

            T B_x = cos(phi) * B_r - sin(phi) * B_phi;
            T B_y = sin(phi) * B_r + cos(phi) * B_phi;

            modB = sqrt(B_x*B_x + B_y*B_y + B_z*B_z);
            
        }
        if(mu[threadIdx.x] == -1.0){ // dummy value from python when mu needs to be computed
            mu[threadIdx.x] = v_perp2 / (2*modB);
        }

        calc_max_timestep_size<T, coord>(dtmax, x_temp, block_interpolants);

        if(dt[threadIdx.x] == -1.0){ // dummy value from python when dt needs to be computed
            dt[threadIdx.x] = 1e-3*dtmax[threadIdx.x];
        }
    }
}

// a kernel to calculate dt, dtmax, t, mu in global memory
template<typename T, RHS id, typename... Args>
__global__ void setup_kernel(T* init_pos, const T* __restrict__ quadpts_arr, T* mu, T* dt, T* dtmax,
                                T* t, T* derivs, int nparticles, Args... args){

    int idx = threadIdx.x + blockIdx.x*PARTICLES_PER_BLOCK;
    __shared__ T x_temp[5*PARTICLES_PER_BLOCK];
    T* block_derivs = derivs + 7*map_rhs_to_n_deriv_outputs<id>()*blockIdx.x*PARTICLES_PER_BLOCK;
    __shared__ bool symmetry_exploited[PARTICLES_PER_BLOCK];
    __shared__ int cell_index_start[3*PARTICLES_PER_BLOCK];
    __shared__ T shape_fun_vals[12*PARTICLES_PER_BLOCK];
    __shared__ T state[4*PARTICLES_PER_BLOCK];

    T* block_dt = dt + blockIdx.x*PARTICLES_PER_BLOCK;
    T* block_t = t + blockIdx.x*PARTICLES_PER_BLOCK;
    T* block_mu = mu + blockIdx.x*PARTICLES_PER_BLOCK;
    T* block_dtmax = dtmax + blockIdx.x*PARTICLES_PER_BLOCK;

    bool is_valid = idx < nparticles && threadIdx.x < PARTICLES_PER_BLOCK;
    __shared__ bool is_valid_arr[PARTICLES_PER_BLOCK];
    if(threadIdx.x < PARTICLES_PER_BLOCK){
        is_valid_arr[threadIdx.x] = is_valid;
    }
    if(is_valid){
        for(int i=0; i<4; ++i){
            state[i*PARTICLES_PER_BLOCK + threadIdx.x] = init_pos[4*idx + i];
        }
    }
    __syncthreads();

    setup_particle<T, id>(block_mu, block_t, block_dt, block_dtmax, x_temp, symmetry_exploited, cell_index_start,
                            quadpts_arr, shape_fun_vals, state, block_derivs, is_valid_arr, args...);

    
}

template<typename T>
__device__ void check_has_left_cartesian(bool* has_left, const T* __restrict__ state, const T* __restrict__ derivs){
    constexpr int n_deriv_outputs = map_rhs_to_n_deriv_outputs<RHS::GC_CartesianVacuum>();
    has_left[threadIdx.x] = derivs[(6*n_deriv_outputs + 4)*PARTICLES_PER_BLOCK + threadIdx.x] < 0; // boundary dist fn at new location
}

template<typename T>
__device__ void check_has_left_boozer(bool* has_left, const T* __restrict__ state, const T* __restrict__ derivs){
    T x1 = state[0*PARTICLES_PER_BLOCK + threadIdx.x];
    T x2 = state[1*PARTICLES_PER_BLOCK + threadIdx.x];
    T s = hypot(x1, x2);

    has_left[threadIdx.x] = s >= 1; 
}


// determine whether a particle has been lost or not
// in cartesian coordinates, we check the signed distance function
// in boozer coordinates we check for s >= 1
template<typename T, CoordSys coord>
__device__ void check_has_left(bool* has_left, const T* __restrict__ state, const T* __restrict__ derivs){
    if constexpr (coord == CoordSys::Cartesian){
        check_has_left_cartesian(has_left, state, derivs);
    } else if constexpr (coord == CoordSys::Boozer){
        check_has_left_boozer(has_left, state, derivs);
    } else{
        printf("default check_has_left not implemented\n");
    }
};


// this function estimates error, accepts/rejects the proposed step
// and adjust the step size
template<typename T, RHS id>
__device__ void adjust_time(T* t, T* dt, double* tmax, T* state, T* __restrict__ derivs, const T* __restrict__ x_temp, 
                            bool* has_left, const T* __restrict__ dtmax, const bool* __restrict__ is_valid){
    // identify a particle and state index
    const int p = threadIdx.x % PARTICLES_PER_BLOCK;   // particle
    const int state_id = threadIdx.x / PARTICLES_PER_BLOCK; // state variable

    const bool active = is_valid[p] && !(has_left[p] || t[p] >= tmax[p]);
    const T dt_p = dt[p];
    
    // Compute  error
    // https://live.boost.org/doc/libs/1_82_0/libs/numeric/odeint/doc/html/boost_numeric_odeint/odeint_in_detail/steppers.html
    // resolve typo in boost docs: https://numerical.recipes/book.html
    T error_elt = 0.0;
    if(active){
        const T state_i = state[state_id*PARTICLES_PER_BLOCK + p];
        constexpr int n_deriv_entries = map_rhs_to_n_deriv_outputs<id>();
        const T deriv_i = derivs[(n_deriv_entries*0 + state_id)*PARTICLES_PER_BLOCK + p];
        error_elt = T(bhat_wgts[0])*deriv_i;
        for(int j=2; j<7; ++j){
            error_elt += T(bhat_wgts[j])*derivs[(n_deriv_entries*j + state_id)*PARTICLES_PER_BLOCK + p];
        }
        error_elt *= dt_p;
        const T atol_i = (rescale_abstol_var_d) && (state_id == 3) ?  atol_d * T(v_total_d) : atol_d;
        error_elt = fabs(error_elt) / (atol_i + rtol_d*(fabs(state_i) + dt_p*fabs(deriv_i)));
    }

    // reduction to find the maximum error across all state variables for this particle
    error_elt = max(error_elt, __shfl_down_sync(FULL_MASK, error_elt, 16));
    error_elt = max(error_elt, __shfl_down_sync(FULL_MASK, error_elt, 8));

    // thread i holds max value for particle i
    const T max_err = __shfl_sync(FULL_MASK, error_elt, p); // each thread reads from thread p
    const bool accept = active && (max_err <= 1.0);
    if(accept){
        state[state_id*PARTICLES_PER_BLOCK + p] = x_temp[(state_id+1)*PARTICLES_PER_BLOCK + p];

        // if accepting a state in Boozer coordinates, wrap zeta to [0, 2pi/nfp] to avoid numerical issues with large values of zeta 
        if constexpr (map_rhs_to_coord<id>() == CoordSys::Boozer){
            if (state_id == 2) { // zeta
                T period = T(grid_ranges_d[9]);
                state[state_id*PARTICLES_PER_BLOCK + p] = fmod(state[state_id*PARTICLES_PER_BLOCK + p], period);
                state[state_id*PARTICLES_PER_BLOCK + p] += grid_ranges_d[9]*(state[state_id*PARTICLES_PER_BLOCK + p] < 0);
            }
        }

        constexpr int n_deriv_outputs = map_rhs_to_n_deriv_outputs<id>();
        // copy derivatives to the first slot for the next step
        derivs[(n_deriv_outputs*0 + state_id)*PARTICLES_PER_BLOCK + p] = derivs[(n_deriv_outputs*6 + state_id)*PARTICLES_PER_BLOCK + p]; 
    }
    if(active && state_id == 0){ // now one thread per particle
        T dt_new = dt_p*0.9;
        T exponent = 0.0;
        if(max_err > 1.0){
            exponent = -1.0/3.0;
        }
        if(max_err < 0.5) {
            exponent = -1.0/5.0;
        }
        dt_new *= pow(max_err, exponent);
        dt_new = max(dt_new, T(0.2) * dt_p);
        dt_new = min(dt_new, T(5.0) * dt_p);

        if(accept){
            if(0.5 < max_err){
                dt_new = dt_p;
            }
            t[p] += dt_p;
        }
        dt_new = min(dt_new, dtmax[p]);
        dt[p] = dt_new;
    }
    __syncthreads();
    if(accept && state_id == 0){
        check_has_left<T, map_rhs_to_coord<id>()>(has_left, state, derivs);
    }
}

// helper function for a single DP5 evaluation
template<typename T, RHS id, int deriv_id, typename... Args>
__device__ void dp5_one_step(T* x_temp, T* derivs, const T* __restrict__ quadpts_arr, int* cell_index_start,
                            T* shape_fun_vals, const T* __restrict__ t, const T* __restrict__ dt,
                            bool* symmetry_exploited, const T* __restrict__ state, const T* __restrict__ mu, const bool* __restrict__ is_valid, Args... args){
    // if the thread is responsible for a particle, compute the point at which the derivative will be computed
    build_state<T, id, deriv_id>(x_temp, symmetry_exploited, cell_index_start, shape_fun_vals, state, derivs, t, dt, is_valid);
    // ensure that all threads have updated x_temp before calculating derivatives, where a data race would occur
    __syncthreads();
    calc_derivs<T, id, deriv_id>(derivs + threadIdx.x, quadpts_arr, x_temp + threadIdx.x, symmetry_exploited + threadIdx.x, cell_index_start,
         shape_fun_vals, mu + threadIdx.x, is_valid, args...);

    // ensure all particles have derivative calculations before accepting/rejecting timestep
    __syncthreads();
}


/*
 * This function puts it all together. The while loop keeps track of the work the block has remaining
 * The inner loop computes the 7 Dormand Prince derivative estimates.
 * Everything lives in shared memory except the data for the interpolant
 */
template<typename T, RHS id, typename... Args>
__global__ void  particle_trace_kernel(T* out, T* init_pos, const T* __restrict__ quadpts_arr, T* derivs, T* mu, 
                                            double* tmax,T* t, T* dt, T* dtmax, Args... args){
    int idx = threadIdx.x + blockIdx.x*PARTICLES_PER_BLOCK;

    __shared__ T x_temp[5 * PARTICLES_PER_BLOCK];
    T* block_derivs = derivs + blockIdx.x*PARTICLES_PER_BLOCK*7*map_rhs_to_n_deriv_outputs<id>(); 
    __shared__ double block_tmax[PARTICLES_PER_BLOCK];
    __shared__ T block_dt[PARTICLES_PER_BLOCK];
    __shared__ bool symmetry_exploited[PARTICLES_PER_BLOCK];
    __shared__ int cell_index_start[3*PARTICLES_PER_BLOCK];
    __shared__ T shape_fun_vals[12*PARTICLES_PER_BLOCK]; // 4 shape function values for each of the 3 coordinates
    __shared__ T block_mu[PARTICLES_PER_BLOCK];
    __shared__ T block_t[PARTICLES_PER_BLOCK];
    __shared__ T block_dtmax[PARTICLES_PER_BLOCK];
    __shared__ T state[4 * PARTICLES_PER_BLOCK];
    __shared__ bool has_left[PARTICLES_PER_BLOCK];


    bool is_valid = idx < nparticles_d && threadIdx.x < PARTICLES_PER_BLOCK;
    __shared__ bool is_valid_arr[PARTICLES_PER_BLOCK];
    if(threadIdx.x < PARTICLES_PER_BLOCK){
        is_valid_arr[threadIdx.x] = is_valid;
    }
    // if thread is responsible for a valid particle id, load that particle's data
    if(is_valid){
        // block_t[threadIdx.x] = 0.0;
        has_left[threadIdx.x] = false;
        for(int i=0; i<4; ++i){
            state[i*PARTICLES_PER_BLOCK + threadIdx.x] = init_pos[4*idx + i];
        }
        block_dt[threadIdx.x] = dt[idx]; // copy input dt
        block_mu[threadIdx.x] = mu[idx]; // copy input mu
        block_t[threadIdx.x] = t[idx]; // copy input t
        block_tmax[threadIdx.x] = tmax[idx]; // copy input tmax
        block_dtmax[threadIdx.x] = dtmax[idx]; // copy input dtmax
    }
    __syncthreads();

    // if there exists a particle which is real and hasn't not reached tmax or left, keep tracing
    while(__syncthreads_count(is_valid_arr[threadIdx.x % PARTICLES_PER_BLOCK] && 
                            !(block_t[threadIdx.x % PARTICLES_PER_BLOCK] >= block_tmax[threadIdx.x % PARTICLES_PER_BLOCK] || has_left[threadIdx.x % PARTICLES_PER_BLOCK])) > 0){
        if(__syncthreads_count(is_valid_arr[threadIdx.x % PARTICLES_PER_BLOCK] && block_t[threadIdx.x % PARTICLES_PER_BLOCK] == 0.0) > 0){
            dp5_one_step<T, id, 0>(x_temp, block_derivs, quadpts_arr, cell_index_start, shape_fun_vals, block_t, block_dt,
                                symmetry_exploited, state, block_mu, is_valid_arr, args...);
        }
        dp5_one_step<T, id, 1>(x_temp, block_derivs, quadpts_arr, cell_index_start, shape_fun_vals, block_t, block_dt,
                            symmetry_exploited, state, block_mu, is_valid_arr, args...);
        dp5_one_step<T, id, 2>(x_temp, block_derivs, quadpts_arr, cell_index_start, shape_fun_vals, block_t, block_dt,
                            symmetry_exploited, state, block_mu, is_valid_arr, args...);
        dp5_one_step<T, id, 3>(x_temp, block_derivs, quadpts_arr, cell_index_start, shape_fun_vals, block_t, block_dt,
                            symmetry_exploited, state, block_mu, is_valid_arr, args...);
        dp5_one_step<T, id, 4>(x_temp, block_derivs, quadpts_arr, cell_index_start, shape_fun_vals, block_t, block_dt,
                            symmetry_exploited, state, block_mu, is_valid_arr, args...);
        dp5_one_step<T, id,  5>(x_temp, block_derivs, quadpts_arr, cell_index_start, shape_fun_vals, block_t, block_dt,
                            symmetry_exploited, state, block_mu, is_valid_arr, args...);
        dp5_one_step<T, id, 6>(x_temp, block_derivs, quadpts_arr, cell_index_start, shape_fun_vals, block_t, block_dt,
                            symmetry_exploited, state, block_mu, is_valid_arr, args...);
        adjust_time<T, id>(block_t, block_dt, block_tmax, state, block_derivs, x_temp, has_left, block_dtmax, is_valid_arr);
        

        // if the particle has left, go get another one
        if(threadIdx.x < PARTICLES_PER_BLOCK && is_valid_arr[threadIdx.x] && \
            (block_t[threadIdx.x] >= block_tmax[threadIdx.x] || has_left[threadIdx.x])){
            // write output for current particle
            out[7*idx] = block_t[threadIdx.x];
            for(int i=0; i<4; ++i){
                out[7*idx + i + 1] = state[i*PARTICLES_PER_BLOCK + threadIdx.x];
            }
            out[7*idx + 5] = block_dt[threadIdx.x];
            out[7*idx + 6] = block_mu[threadIdx.x];

            // load the next particle
            idx = atomicAdd(&next_particle_d, 1);
            if(idx < nparticles_d){
                T* loc_arr = init_pos + 4*idx;
                for(int i=0; i<4; ++i){
                    state[i*PARTICLES_PER_BLOCK + threadIdx.x] = init_pos[4*idx + i];
                }
                block_dt[threadIdx.x] = dt[idx];
                block_mu[threadIdx.x] = mu[idx];
                block_t[threadIdx.x] = t[idx];
                block_dtmax[threadIdx.x] = dtmax[idx];
                block_tmax[threadIdx.x] = tmax[idx];
                has_left[threadIdx.x] = false;
                symmetry_exploited[threadIdx.x] = false;
            } else {
                is_valid_arr[threadIdx.x] = false;
            }
        }
        __syncthreads();
    }
    return;
}


template<typename T, RHS id, typename... Args>
vector<T> gpu_tracing(py::array_t<T> quad_pts, py::array_t<double> x1_range, py::array_t<double> x2_range, py::array_t<double> x3_range, 
    py::array_t<T> loc_init, double m, double q, double vtotal, py::array_t<T> vtang, py::array_t<double> tmax, double tol, py::array_t<T> dt_in, py::array_t<T> mu_in,
    int nparticles, Args... args){

    //  read data in from python
    T* loc_init_arr = create_array(loc_init);
    T* vtang_arr = create_array(vtang);
    T* quadpts_arr = create_array(quad_pts);
    double* x1_range_arr = create_array(x1_range);
    double* x2_range_arr = create_array(x2_range);
    double* x3_range_arr = create_array(x3_range);
    T* dt_in_arr = create_array(dt_in);
    T* mu_in_arr = create_array(mu_in);
    double* tmax_arr = create_array(tmax);

    // allocate and copy to device memory
    double x1_range_ext[4];
    double x2_range_ext[4];
    double x3_range_ext[4];

    for(int i=0; i<3; ++i){
        x1_range_ext[i] = x1_range_arr[i];
        x2_range_ext[i] = x2_range_arr[i];
        x3_range_ext[i] = x3_range_arr[i];
    }
    // precompute inverse grid sizes
    x1_range_ext[3] = (x1_range_ext[2] - 1) / (x1_range_ext[1] - x1_range_ext[0]) ;
    x2_range_ext[3] = (x2_range_ext[2] - 1) /(x2_range_ext[1] - x2_range_ext[0]) ;
    x3_range_ext[3] = (x3_range_ext[2] - 1)/ (x3_range_ext[1] - x3_range_ext[0]) ;

    int n_x1 = (x1_range_ext[2]-1)/3;
    int n_x2 = (x2_range_ext[2]-1)/3;
    int n_x3 = (x3_range_ext[2]-1)/3;
    int n_x23 = 64*n_x1*n_x2*n_x3;

    // combine ranges into one array
    double grid_ranges[12];
    for(int i=0; i<4; ++i){
        grid_ranges[i] = x1_range_ext[i];
        grid_ranges[4 + i] = x2_range_ext[i];
        grid_ranges[8 + i] = x3_range_ext[i];
    }

    gpuErrchk(cudaMemcpyToSymbol(grid_ranges_d, grid_ranges, 12*sizeof(double)) );
    gpuErrchk(cudaMemcpyToSymbol(mass_d, &m, sizeof(double)));
    gpuErrchk(cudaMemcpyToSymbol(charge_d, &q, sizeof(double)));
    gpuErrchk(cudaMemcpyToSymbol(atol_d, &tol, sizeof(double)));
    gpuErrchk(cudaMemcpyToSymbol(rtol_d, &tol, sizeof(double)));
    gpuErrchk(cudaMemcpyToSymbol(v_total_d, &vtotal, sizeof(double)));


    gpuErrchk(cudaMemcpyToSymbol(n_x2_d, &n_x2, sizeof(int)) );
    gpuErrchk(cudaMemcpyToSymbol(n_x3_d, &n_x3, sizeof(int)) );
    gpuErrchk(cudaMemcpyToSymbol(n_x23_d, &n_x23, sizeof(int)) );
    gpuErrchk(cudaMemcpyToSymbol(nparticles_d, &nparticles, sizeof(int)));

    T init_pos[4*nparticles];
    // load initial conditions
    for(int i=0; i<nparticles; ++i){
        int start = 3*i;

        T s = loc_init_arr[start];
        T theta = loc_init_arr[start+1];
        
        for(int j=0; j<3; j++){
            init_pos[4*i + j] = loc_init_arr[start + j];
        }
        init_pos[4*i + 3] = vtang_arr[i];
    }
   
    T* init_pos_d;
    gpuErrchk(cudaMalloc((void**)&init_pos_d, 4 * nparticles * sizeof(T)) );
    gpuErrchk(cudaMemcpy(init_pos_d, init_pos, 4 * nparticles * sizeof(T), cudaMemcpyHostToDevice) );

    T* quadpts_d;
    gpuErrchk(cudaMalloc((void**)&quadpts_d, quad_pts.size() * sizeof(T)) ); 
    gpuErrchk(cudaMemcpy(quadpts_d, quadpts_arr, quad_pts.size() * sizeof(T), cudaMemcpyHostToDevice) );

    T* dt_d;
    gpuErrchk(cudaMalloc((void**)&dt_d, dt_in.size() * sizeof(T)) ); 
    gpuErrchk(cudaMemcpy(dt_d, dt_in_arr, dt_in.size() * sizeof(T), cudaMemcpyHostToDevice) );

    // scratch spaces
    T* mu_d;
    gpuErrchk(cudaMalloc((void**)&mu_d, nparticles * sizeof(T)) ); 
    gpuErrchk(cudaMemcpy(mu_d, mu_in_arr, nparticles * sizeof(T), cudaMemcpyHostToDevice) );

    T* t_d;
    cudaMalloc((void**)&t_d, nparticles*sizeof(T));
    cudaMemset(t_d, 0, nparticles*sizeof(T));  

    double* tmax_d;
    gpuErrchk(cudaMalloc((void**)&tmax_d, nparticles * sizeof(double)) ); 
    gpuErrchk(cudaMemcpy(tmax_d, tmax_arr, nparticles * sizeof(double), cudaMemcpyHostToDevice) );

    T* dtmax_d;
    gpuErrchk(cudaMalloc((void**)&dtmax_d, nparticles * sizeof(T)) ); 

    T* out_d;
    gpuErrchk(cudaMalloc((void**)&out_d, 7 * nparticles * sizeof(T)) ); 

    // launch params
    int nthreads = THREADS_PER_BLOCK;
    int setup_nblks = nparticles / PARTICLES_PER_BLOCK + 1;

    // compute maximum number of blocks for a single wave for persistent threads
    int numSMs;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
    int blocks_per_sm;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm,
        particle_trace_kernel<T, id>, THREADS_PER_BLOCK, 0);
    int nblks = blocks_per_sm * numSMs;

    int scratch_nblks = max(setup_nblks, nblks);
    T* derivs_d;
    cudaMalloc((void**)&derivs_d, 7*map_rhs_to_n_deriv_outputs<id>()*scratch_nblks*PARTICLES_PER_BLOCK*sizeof(T));


    setup_kernel<T, id><<<setup_nblks, nthreads>>>(init_pos_d, quadpts_d, mu_d, dt_d, dtmax_d,
                                            t_d, derivs_d, nparticles, args...);


    // initialize global counter
    int n_total_threads = nblks*PARTICLES_PER_BLOCK;
    gpuErrchk(cudaMemcpyToSymbol(next_particle_d, &n_total_threads, sizeof(int)) );
    particle_trace_kernel<T, id><<<nblks, nthreads>>>(out_d, init_pos_d, quadpts_d, derivs_d, mu_d, tmax_d, t_d, dt_d, dtmax_d, args...);

    T out[7*nparticles];
    gpuErrchk(cudaMemcpy(out, out_d, 7 * nparticles * sizeof(T), cudaMemcpyDeviceToHost) );

    gpuErrchk( cudaFree(quadpts_d) );
    gpuErrchk( cudaFree(init_pos_d) );
    gpuErrchk( cudaFree(out_d) );
    gpuErrchk(cudaFree(derivs_d));
    gpuErrchk(cudaFree(dt_d));
    gpuErrchk(cudaFree(t_d));
    gpuErrchk(cudaFree(tmax_d));
    gpuErrchk(cudaFree(dtmax_d));
    gpuErrchk(cudaFree(mu_d));
    vector<T> particle_output(7*nparticles);
    for(int i=0; i<7*nparticles; ++i){
        particle_output[i] = out[i];
    }

    return particle_output;
}

template<typename T>
vector<T> cartesian_gpu_tracing(py::array_t<T> quad_pts, py::array_t<double> rrange,
        py::array_t<double> phirange, py::array_t<double> zrange, py::array_t<T> xyz_init, double m, double q, double vtotal, py::array_t<T> vtang, 
        py::array_t<double> tmax, double tol, py::array_t<T> dt_in, py::array_t<T> mu_in, int nparticles){
            return gpu_tracing<T, RHS::GC_CartesianVacuum>(quad_pts, rrange, phirange, zrange, xyz_init, m, q, vtotal, vtang, tmax, tol, dt_in, mu_in, nparticles);
        }

template vector<double> cartesian_gpu_tracing<double>(py::array_t<double> quad_pts, py::array_t<double> rrange,
        py::array_t<double> phirange, py::array_t<double> zrange, py::array_t<double> xyz_init, double m, double q, double vtotal, py::array_t<double> vtang, 
        py::array_t<double> tmax, double tol, py::array_t<double> dt_in, py::array_t<double> mu_in, int nparticles);

template vector<float> cartesian_gpu_tracing<float>(py::array_t<float> quad_pts, py::array_t<double> rrange,
        py::array_t<double> phirange, py::array_t<double> zrange, py::array_t<float> xyz_init, double m, double q, double vtotal, py::array_t<float> vtang, 
        py::array_t<double> tmax, double tol, py::array_t<float> dt_in, py::array_t<float> mu_in, int nparticles);

template<typename T>
vector<T> boozer_gpu_tracing(py::array_t<T> quad_pts, py::array_t<double> srange,
        py::array_t<double> trange, py::array_t<double> zrange, py::array_t<T> stz_init, double m, double q, double vtotal, py::array_t<T> vtang, 
        py::array_t<double> tmax, double tol, py::array_t<T> dt_in, py::array_t<T> mu_in, double psi0, int nparticles, bool vacuum){

    // read data in from python
    // T* stz_init_arr = create_array(stz_init);
    
    // for(int i=0; i<nparticles; ++i){
    //     T s = stz_init_arr[3*i];
    //     T theta = stz_init_arr[3*i+1];

    //     stz_init_arr[3*i] = s*cos(theta);
    //     stz_init_arr[3*i+1] = s*sin(theta);
    // }
    double inv_psi0_charge = 1.0 / (psi0*q);
    gpuErrchk(cudaMemcpyToSymbol(psi0_d, &psi0, sizeof(double)));
    gpuErrchk(cudaMemcpyToSymbol(inv_psi0_charge_d, &inv_psi0_charge, sizeof(double)));

    std::vector<T> results;
    if (vacuum) {
        results = gpu_tracing<T, RHS::GC_BoozerVacuum>(quad_pts, srange, trange, zrange, stz_init, m, q, vtotal, vtang, tmax, tol, dt_in, mu_in, nparticles);
    } else {
        results = gpu_tracing<T, RHS::GC_Boozer>(quad_pts, srange, trange, zrange, stz_init, m, q, vtotal, vtang, tmax, tol, dt_in, mu_in, nparticles);
    }

    // for(int i=0; i<nparticles; ++i){
    //     T x1 = results[7*i+1];
    //     T x2 = results[7*i+2];

    //     results[7*i+1] = sqrt(x1*x1 + x2*x2);
    //     results[7*i+2] = atan2(x2, x1);
    // }

    return results;
}

template vector<double> boozer_gpu_tracing<double>(py::array_t<double> quad_pts, py::array_t<double> srange,
        py::array_t<double> trange, py::array_t<double> zrange, py::array_t<double> stz_init, double m, double q, double vtotal, py::array_t<double> vtang, 
        py::array_t<double> tmax, double tol, py::array_t<double> dt_in, py::array_t<double> mu_in, double psi0, int nparticles, bool vacuum);

template vector<float> boozer_gpu_tracing<float>(py::array_t<float> quad_pts, py::array_t<double> srange,
        py::array_t<double> trange, py::array_t<double> zrange, py::array_t<float> stz_init, double m, double q, double vtotal, py::array_t<float> vtang, 
        py::array_t<double> tmax, double tol, py::array_t<float> dt_in, py::array_t<float> mu_in, double psi0, int nparticles, bool vacuum);

template<typename T>
vector<T> boozer_saw_gpu_tracing(py::array_t<T> quad_pts, py::array_t<double> srange, py::array_t<double> trange, py::array_t<double> zrange, 
        double saw_omega, py::array_t<double> saw_srange, py::array_t<int> saw_m, py::array_t<int> saw_n, py::array_t<T> saw_phihats, int saw_nharmonics,
        py::array_t<T> stz_init, double m, double q, double vtotal, py::array_t<T> vtang, py::array_t<double> tmax, double tol, py::array_t<T> dt_in, py::array_t<T> mu_in, double psi0, int nparticles){

    //  read data in from python
    T* stz_init_arr = create_array(stz_init);
    double* saw_srange_arr = create_array(saw_srange);
    int* saw_m_arr = create_array(saw_m);
    int* saw_n_arr = create_array(saw_n);
    T* saw_phihats_arr = create_array(saw_phihats);

    int* saw_m_d;
    gpuErrchk( cudaMalloc((void**)&saw_m_d, saw_m.size() * sizeof(int)) );
    gpuErrchk( cudaMemcpy(saw_m_d, saw_m_arr, saw_m.size() * sizeof(int), cudaMemcpyHostToDevice) );

    int* saw_n_d;
    gpuErrchk( cudaMalloc((void**)&saw_n_d, saw_n.size() * sizeof(int)) );
    gpuErrchk( cudaMemcpy(saw_n_d, saw_n_arr, saw_n.size() * sizeof(int), cudaMemcpyHostToDevice) );

    T* saw_phihats_d;
    gpuErrchk( cudaMalloc((void**)&saw_phihats_d, saw_phihats.size() * sizeof(T)) );
    gpuErrchk( cudaMemcpy(saw_phihats_d, saw_phihats_arr, saw_phihats.size() * sizeof(T), cudaMemcpyHostToDevice) );
    
    // for(int i=0; i<nparticles; ++i){
    //     T s = stz_init_arr[3*i];
    //     T theta = stz_init_arr[3*i+1];

    //     stz_init_arr[3*i] = s*cos(theta);
    //     stz_init_arr[3*i+1] = s*sin(theta);
    // }
    // copy saw s_range to constant memory
    double saw_srange_ext[4];
    for(int i=0; i<3; ++i){
        saw_srange_ext[i] = saw_srange_arr[i];   
    }
    saw_srange_ext[3] = (saw_srange_ext[1] - saw_srange_ext[0]) / (saw_srange_ext[2] - 1);
    gpuErrchk(cudaMemcpyToSymbol(saw_srange_d, saw_srange_ext, 4*sizeof(double)) );
    gpuErrchk(cudaMemcpyToSymbol(psi0_d, &psi0, sizeof(double)));

    std::vector<T> results =  gpu_tracing<T, RHS::GC_BoozerVacuumSAW>(quad_pts, srange, trange, zrange, stz_init, m, q, vtotal, vtang, tmax, tol, dt_in, mu_in, nparticles,
                                                                        saw_omega, saw_m_d, saw_n_d, saw_phihats_d, saw_nharmonics);

    gpuErrchk( cudaFree(saw_m_d) );
    gpuErrchk( cudaFree(saw_n_d) );
    gpuErrchk( cudaFree(saw_phihats_d) );

    // for(int i=0; i<nparticles; ++i){
    //     T x1 = results[7*i+1];
    //     T x2 = results[7*i+2];

    //     results[7*i+1] = sqrt(x1*x1 + x2*x2);
    //     results[7*i+2] = atan2(x2, x1);
    // }

    return results;
}

template vector<double> boozer_saw_gpu_tracing<double>(py::array_t<double> quad_pts, py::array_t<double> srange, py::array_t<double> trange, py::array_t<double> zrange, 
        double saw_omega, py::array_t<double> saw_srange, py::array_t<int> saw_m, py::array_t<int> saw_n, py::array_t<double> saw_phihats, int saw_nharmonics,
        py::array_t<double> stz_init, double m, double q, double vtotal, py::array_t<double> vtang, py::array_t<double> tmax, double tol, py::array_t<double> dt_in, py::array_t<double> mu_in, double psi0, int nparticles);

template vector<float> boozer_saw_gpu_tracing<float>(py::array_t<float> quad_pts, py::array_t<double> srange, py::array_t<double> trange, py::array_t<double> zrange, 
        double saw_omega, py::array_t<double> saw_srange, py::array_t<int> saw_m, py::array_t<int> saw_n, py::array_t<float> saw_phihats, int saw_nharmonics,
        py::array_t<float> stz_init, double m, double q, double vtotal, py::array_t<float> vtang, py::array_t<double> tmax, double tol, py::array_t<float> dt_in, py::array_t<float> mu_in, double psi0, int nparticles);

template<typename T>
vector<T> boozer_saw_nok_gpu_tracing(py::array_t<T> quad_pts, py::array_t<double> srange, py::array_t<double> trange, py::array_t<double> zrange, 
        double saw_omega, py::array_t<double> saw_srange, py::array_t<int> saw_m, py::array_t<int> saw_n, py::array_t<T> saw_phihats, int saw_nharmonics,
        py::array_t<T> stz_init, double m, double q, double vtotal, py::array_t<T> vtang, py::array_t<double> tmax, double tol, py::array_t<T> dt_in, py::array_t<T> mu_in, double psi0, int nparticles){

    //  read data in from python
    T* stz_init_arr = create_array(stz_init);
    double* saw_srange_arr = create_array(saw_srange);
    int* saw_m_arr = create_array(saw_m);
    int* saw_n_arr = create_array(saw_n);
    T* saw_phihats_arr = create_array(saw_phihats);

    int* saw_m_d;
    cudaMalloc((void**)&saw_m_d, saw_m.size() * sizeof(int));
    cudaMemcpy(saw_m_d, saw_m_arr, saw_m.size() * sizeof(int), cudaMemcpyHostToDevice);

    int* saw_n_d;
    cudaMalloc((void**)&saw_n_d, saw_n.size() * sizeof(int));
    cudaMemcpy(saw_n_d, saw_n_arr, saw_n.size() * sizeof(int), cudaMemcpyHostToDevice);

    T* saw_phihats_d;
    cudaMalloc((void**)&saw_phihats_d, saw_phihats.size() * sizeof(T));
    cudaMemcpy(saw_phihats_d, saw_phihats_arr, saw_phihats.size() * sizeof(T), cudaMemcpyHostToDevice);
    
    // for(int i=0; i<nparticles; ++i){
    //     T s = stz_init_arr[3*i];
    //     T theta = stz_init_arr[3*i+1];

    //     stz_init_arr[3*i] = s*cos(theta);
    //     stz_init_arr[3*i+1] = s*sin(theta);
    // }
    // copy saw s_range to constant memory
    double saw_srange_ext[4];
    for(int i=0; i<3; ++i){
        saw_srange_ext[i] = saw_srange_arr[i];   
    }
    saw_srange_ext[3] = (saw_srange_ext[1] - saw_srange_ext[0]) / (saw_srange_ext[2] - 1);
    gpuErrchk(cudaMemcpyToSymbol(saw_srange_d, saw_srange_ext, 4*sizeof(double)) );
    gpuErrchk(cudaMemcpyToSymbol(psi0_d, &psi0, sizeof(double)));

    std::vector<T> results =  gpu_tracing<T, RHS::GC_BoozerNoKSAW>(quad_pts, srange, trange, zrange, stz_init, m, q, vtotal, vtang, tmax, tol, dt_in, mu_in, nparticles,
                                                                        saw_omega, saw_m_d, saw_n_d, saw_phihats_d, saw_nharmonics);

    gpuErrchk( cudaFree(saw_m_d) );
    gpuErrchk( cudaFree(saw_n_d) );
    gpuErrchk( cudaFree(saw_phihats_d) );

    // for(int i=0; i<nparticles; ++i){
    //     T x1 = results[7*i+1];
    //     T x2 = results[7*i+2];

    //     results[7*i+1] = sqrt(x1*x1 + x2*x2);
    //     results[7*i+2] = atan2(x2, x1);
    // }
    return results;
}

template vector<double> boozer_saw_nok_gpu_tracing<double>(py::array_t<double> quad_pts, py::array_t<double> srange, py::array_t<double> trange, py::array_t<double> zrange, 
        double saw_omega, py::array_t<double> saw_srange, py::array_t<int> saw_m, py::array_t<int> saw_n, py::array_t<double> saw_phihats, int saw_nharmonics,
        py::array_t<double> stz_init, double m, double q, double vtotal, py::array_t<double> vtang, py::array_t<double> tmax, double tol, py::array_t<double> dt_in, py::array_t<double> mu_in, double psi0, int nparticles);

template vector<float> boozer_saw_nok_gpu_tracing<float>(py::array_t<float> quad_pts, py::array_t<double> srange, py::array_t<double> trange, py::array_t<double> zrange, 
        double saw_omega, py::array_t<double> saw_srange, py::array_t<int> saw_m, py::array_t<int> saw_n, py::array_t<float> saw_phihats, int saw_nharmonics,
        py::array_t<float> stz_init, double m, double q, double vtotal, py::array_t<float> vtang, py::array_t<double> tmax, double tol, py::array_t<float> dt_in, py::array_t<float> mu_in, double psi0, int nparticles);

/*
 * This function accounts for exploiting stellarator symmetry
 * It is only used in the interpolant test.
 */ 
template<typename T>
__device__ void account_for_symmetry_cartesian(T* interpolants, bool* symmetry_exploited){
    if(symmetry_exploited[threadIdx.x]){
        interpolants[0] *= T(-1.0);
        interpolants[4] *= T(-1.0);
        interpolants[5] *= T(-1.0);
    }
}

template<typename T, RHS id>
__device__ void account_for_symmetry_boozer(T* interpolants, bool* symmetry_exploited){
    // modB, dmodBds, dmodBdtheta, dmodBdzeta, G, iota
    if(symmetry_exploited[threadIdx.x]){
        interpolants[2] *= T(-1.0);
        interpolants[3] *= T(-1.0);
    }
}
template<typename T, CoordSys coord>
__device__ void account_for_symmetry(T* interpolants, bool* symmetry_exploited){
    if constexpr(coord == CoordSys::Cartesian ){
        account_for_symmetry_cartesian(interpolants, symmetry_exploited);
    } else if constexpr(coord == CoordSys::Boozer){
        account_for_symmetry_boozer(interpolants, symmetry_exploited);
    } else{
        printf("default account_for_symmetry not implemented\n");
    }
};



// RHS-aware symmetry correction used by the interpolation test helper
template<typename T, RHS id, int n>
__device__ void account_for_symmetry_rhs(T* interpolants, bool* symmetry_exploited){
    if(!symmetry_exploited[threadIdx.x]) return;
    if constexpr (id == RHS::GC_CartesianVacuum){
        interpolants[0] *= T(-1.0);
        interpolants[4] *= T(-1.0);
        interpolants[5] *= T(-1.0);
    } else if constexpr (id == RHS::GC_BoozerVacuum || id == RHS::GC_BoozerVacuumSAW){
        // Only theta/zeta derivatives flip sign
        interpolants[2] *= T(-1.0);
        interpolants[3] *= T(-1.0);
    } else if constexpr (id == RHS::GC_Boozer){
        // 12-field ordering: flip dB/dtheta, dB/dzeta, and K
        interpolants[2] *= T(-1.0);  // d|B|/dtheta
        interpolants[3] *= T(-1.0);  // d|B|/dzeta
        if constexpr (n >= 12) {
            interpolants[9] *= T(-1.0); // K
        }
    }
}


template <typename T, RHS id, int n>
__global__ void test_gpu_interpolation_kernel(T* quad_pts, T* loc, T* out, T* derivs, T* dt, T* t, int n_points){
    int idx = threadIdx.x + blockIdx.x*PARTICLES_PER_BLOCK;

    __shared__ T x_temp[5 * PARTICLES_PER_BLOCK];
    __shared__ bool symmetry_exploited[PARTICLES_PER_BLOCK];
    __shared__ int cell_index_start[3*PARTICLES_PER_BLOCK];
    __shared__ T shape_fun_vals[12*PARTICLES_PER_BLOCK];
    __shared__ T state[4 * PARTICLES_PER_BLOCK];
    T* block_derivs = derivs + 7*map_rhs_to_n_deriv_outputs<id>()*blockIdx.x*PARTICLES_PER_BLOCK;
    T* block_dt = dt + blockIdx.x*PARTICLES_PER_BLOCK;
    T* block_t = t + blockIdx.x*PARTICLES_PER_BLOCK;

    __shared__ T block_interpolants[n*PARTICLES_PER_BLOCK];

    T* loc_arr = loc + 3*idx;
    T* out_arr  =  out + idx*n;

    bool is_valid = idx < n_points && threadIdx.x < PARTICLES_PER_BLOCK;
    
    __shared__ bool is_valid_arr[PARTICLES_PER_BLOCK];
    if(threadIdx.x < PARTICLES_PER_BLOCK){
        is_valid_arr[threadIdx.x] = is_valid;
    }
    if(is_valid){
        block_dt[threadIdx.x] = 1e-3; // needed for build_state
        symmetry_exploited[threadIdx.x] = false;
        for(int i=0; i<3; ++i){
            state[i*PARTICLES_PER_BLOCK + threadIdx.x] = loc_arr[i];
        }
        state[3*PARTICLES_PER_BLOCK + threadIdx.x] = 0.0; // dummy vpar value
        block_t[threadIdx.x] = 0.0; // dummy time value


        for(int i=0; i<n; ++i){
            block_interpolants[i*PARTICLES_PER_BLOCK + threadIdx.x] = 0.0;
        }
    } 

    build_state<T, id, 0>(x_temp, symmetry_exploited, cell_index_start, shape_fun_vals, state, block_derivs, block_t, block_dt, is_valid_arr);

    __syncthreads();
    interpolate<T, n>(block_interpolants, quad_pts, cell_index_start, shape_fun_vals, is_valid_arr);
    __syncthreads();
    
    if(is_valid){
        for(int i=0; i<n; ++i){
            out_arr[i] = block_interpolants[i*PARTICLES_PER_BLOCK + threadIdx.x];

        }
        // Apply symmetry fixes with RHS/layout awareness
        account_for_symmetry_rhs<T, id, n>(out_arr, symmetry_exploited);
    }
}


template<typename T>
py::array_t<T> test_gpu_interpolation(py::array_t<T> quad_pts, py::array_t<double> x1_range, py::array_t<double> x2_range, py::array_t<double> x3_range, py::array_t<T> loc, std::string rhs, int n_points){
    // read data in from python
    T* quadpts_arr = create_array<T>(quad_pts);
    double* x1_range_arr = create_array(x1_range);
    double* x2_range_arr = create_array(x2_range);
    double* x3_range_arr = create_array(x3_range);
    T* loc_arr = create_array<T>(loc);
    
    // map input data
    // Cartesian Coordinates
    if(rhs == "cartesian_vacuum"){
        for(int i=0; i<n_points; ++i){
            T x = loc_arr[3*i] * cos(loc_arr[3*i + 1]);
            T y = loc_arr[3*i] * sin(loc_arr[3*i + 1]);
            
            loc_arr[3*i] = x;
            loc_arr[3*i+1] = y;
        }
    }

    // Boozer Coordinates
    if((rhs == "boozer_vacuum") || (rhs == "boozer_saw_vacuum") || (rhs == "boozer")) {
        for(int i=0; i<n_points; ++i){
            T x1 = loc_arr[3*i] * cos(loc_arr[3*i + 1]);
            T x2 = loc_arr[3*i] * sin(loc_arr[3*i + 1]);
            
            loc_arr[3*i] = x1;
            loc_arr[3*i+1] = x2;
        }
    }

    int n;
    if(rhs == "cartesian_vacuum"){
        n = 7;
    } else if(rhs == "boozer_vacuum"){
        n = 6;
    } else if(rhs == "boozer_saw_vacuum"){
        n = 10;
    } else if(rhs == "boozer"){
        n = 12;
    }

    // allocate and copy to device memory
    double x1_range_ext[4];
    double x2_range_ext[4];
    double x3_range_ext[4];

    for(int i=0; i<3; ++i){
        x1_range_ext[i] = x1_range_arr[i];
        x2_range_ext[i] = x2_range_arr[i];
        x3_range_ext[i] = x3_range_arr[i];
    }
    // precompute inverse grid sizes
    x1_range_ext[3] = (x1_range_ext[2] - 1) / (x1_range_ext[1] - x1_range_ext[0]) ;
    x2_range_ext[3] = (x2_range_ext[2] - 1) /(x2_range_ext[1] - x2_range_ext[0]) ;
    x3_range_ext[3] = (x3_range_ext[2] - 1)/ (x3_range_ext[1] - x3_range_ext[0]) ;

    int n_x1 = (x1_range_ext[2]-1)/3;
    int n_x2 = (x2_range_ext[2]-1)/3;
    int n_x3 = (x3_range_ext[2]-1)/3;
    int n_x23 = 64*n_x1*n_x2*n_x3;

    double grid_ranges[12];
    for(int i=0; i<4; ++i){
        grid_ranges[i] = x1_range_ext[i];
        grid_ranges[4 + i] = x2_range_ext[i];
        grid_ranges[8 + i] = x3_range_ext[i];
    }

    gpuErrchk(cudaMemcpyToSymbol(grid_ranges_d, grid_ranges, 12*sizeof(double)) );

    gpuErrchk(cudaMemcpyToSymbol(n_x2_d, &n_x2, sizeof(int)) );
    gpuErrchk(cudaMemcpyToSymbol(n_x3_d, &n_x3, sizeof(int)) );
    gpuErrchk(cudaMemcpyToSymbol(n_x23_d, &n_x23, sizeof(int)) );

    T* quadpts_d;
    cudaMalloc((void**)&quadpts_d, quad_pts.size() * sizeof(T));
    cudaMemcpy(quadpts_d, quadpts_arr, quad_pts.size() * sizeof(T), cudaMemcpyHostToDevice);

    T* loc_d;
    cudaMalloc((void**)&loc_d, loc.size() * sizeof(T));
    cudaMemcpy(loc_d, loc_arr, loc.size() * sizeof(T), cudaMemcpyHostToDevice);

    T* derivs_d;
    cudaMalloc((void**)&derivs_d, 7*5*n_points * sizeof(T));
    T* dt_d;
    cudaMalloc((void**)&dt_d, n_points * sizeof(T));

    T* t_d;
    cudaMalloc((void**)&t_d, n_points * sizeof(T));

    T* out_d;
    cudaMalloc((void**)&out_d, n*n_points * sizeof(T));

    int nthreads = THREADS_PER_BLOCK;
    int nblks = n_points / PARTICLES_PER_BLOCK + 1;

    if(rhs == "cartesian_vacuum"){
        test_gpu_interpolation_kernel<T, RHS::GC_CartesianVacuum, 7><<<nblks, nthreads>>>(quadpts_d, loc_d, out_d, derivs_d, dt_d, t_d, n_points);
    } else if(rhs == "boozer_vacuum") {
        test_gpu_interpolation_kernel<T, RHS::GC_BoozerVacuum, 6><<<nblks, nthreads>>>(quadpts_d, loc_d, out_d, derivs_d, dt_d, t_d, n_points);
    } else if(rhs == "boozer_saw_vacuum") {
        test_gpu_interpolation_kernel<T, RHS::GC_BoozerVacuumSAW, 10><<<nblks, nthreads>>>(quadpts_d, loc_d, out_d, derivs_d, dt_d, t_d, n_points);
    } else if(rhs == "boozer") {
        test_gpu_interpolation_kernel<T, RHS::GC_Boozer, 12><<<nblks, nthreads>>>(quadpts_d, loc_d, out_d, derivs_d, dt_d, t_d, n_points);
    }
    T out[n*n_points];
    gpuErrchk( cudaMemcpy(&out, out_d, n*n_points * sizeof(T), cudaMemcpyDeviceToHost) );

    auto result = py::array_t<T>(n*n_points, out);

    gpuErrchk( cudaFree(quadpts_d) );
    gpuErrchk( cudaFree(loc_d) );
    gpuErrchk( cudaFree(out_d) );

    return result;

}

// compile for desired types
template py::array_t<double> test_gpu_interpolation<double>(py::array_t<double> quad_pts, py::array_t<double> x1_range, py::array_t<double> x2_range, py::array_t<double> x3_range, py::array_t<double> loc, std::string rhs, int n_points);
template py::array_t<float> test_gpu_interpolation<float>(py::array_t<float> quad_pts, py::array_t<double> x1_range, py::array_t<double> x2_range, py::array_t<double> x3_range, py::array_t<float> loc, std::string rhs, int n_points);

template<typename T, RHS id, typename... Args>
__global__ void test_gpu_derivs_kernel(T* quad_pts, T* init_pos, T* time, T* out, T* derivs, T* mu, T* dt, T* dtmax, int n_points, Args... args){
    int idx = threadIdx.x + blockIdx.x*PARTICLES_PER_BLOCK;    
    T* loc_arr = init_pos + 4*idx;
    T* out_arr  =  out + 4*idx;

    __shared__ T x_temp[5 * PARTICLES_PER_BLOCK];
    T* block_derivs = derivs + 7*map_rhs_to_n_deriv_outputs<id>()*blockIdx.x*PARTICLES_PER_BLOCK;
    __shared__ bool symmetry_exploited[PARTICLES_PER_BLOCK];
    __shared__ int cell_index_start[3*PARTICLES_PER_BLOCK];
    __shared__ T shape_fun_vals[12*PARTICLES_PER_BLOCK];
    T* block_mu = mu + blockIdx.x*PARTICLES_PER_BLOCK;
    T* block_dt = dt + blockIdx.x*PARTICLES_PER_BLOCK;
    T* block_t = time + blockIdx.x*PARTICLES_PER_BLOCK;
    T* block_dtmax = dtmax + blockIdx.x*PARTICLES_PER_BLOCK;
    __shared__ T state[4 * PARTICLES_PER_BLOCK];

    bool is_valid = idx < n_points && threadIdx.x < PARTICLES_PER_BLOCK;
    __shared__ bool is_valid_arr[PARTICLES_PER_BLOCK];
    if(threadIdx.x < PARTICLES_PER_BLOCK){
        is_valid_arr[threadIdx.x] = is_valid;
    }
    if(is_valid){
        for(int i=0; i<4; ++i){
            state[i*PARTICLES_PER_BLOCK + threadIdx.x] = loc_arr[i];
        }
    }
    __syncthreads();

    build_state<T, id, 0>(x_temp, symmetry_exploited, cell_index_start,
            shape_fun_vals, state, block_derivs, block_t, block_dt, is_valid_arr);
    __syncthreads();
    calc_derivs<T, id, 0>(block_derivs + threadIdx.x, quad_pts, x_temp + threadIdx.x, symmetry_exploited + threadIdx.x,
         cell_index_start, shape_fun_vals, block_mu + threadIdx.x, is_valid_arr, args...);
    __syncthreads();

    if(is_valid){
        // copy back
        for(int i=0; i<4; ++i){
            out_arr[i] = block_derivs[i*PARTICLES_PER_BLOCK + threadIdx.x];
        }

    }
}

template<typename T, RHS id, typename... Args>
py::array_t<T> test_gpu_derivatives(py::array_t<T> quad_pts, py::array_t<double> x1_range, py::array_t<double> x2_range, py::array_t<double> x3_range,
                                 py::array_t<T> loc, py::array_t<T> vpar, py::array_t<T> time, double v_total, double m, double q, int n_points, Args... args){

    T* quadpts_arr = create_array<T>(quad_pts);
    double* x1_range_arr = create_array<double>(x1_range);
    double* x2_range_arr = create_array<double>(x2_range);
    double* x3_range_arr = create_array<double>(x3_range);
    T* loc_arr = create_array<T>(loc);
    T* vpar_arr = create_array<T>(vpar);
    T* time_arr = create_array<T>(time);

    T init_pos[4*n_points];
    // load initial conditions
    for(int i=0; i<n_points; ++i){
        T x1 = loc_arr[3*i + 0] * cos(loc_arr[3*i + 1]);
        T x2 = loc_arr[3*i + 0] * sin(loc_arr[3*i + 1]);
        init_pos[4*i + 0] = x1;
        init_pos[4*i + 1] = x2;
        init_pos[4*i + 2] = loc_arr[3*i + 2];
        init_pos[4*i + 3] = vpar_arr[i];
    }
    
    T* quadpts_d;
    cudaMalloc((void**)&quadpts_d, quad_pts.size() * sizeof(T));
    cudaMemcpy(quadpts_d, quadpts_arr, quad_pts.size() * sizeof(T), cudaMemcpyHostToDevice);

    T* init_pos_d;
    cudaMalloc((void**)&init_pos_d, 4*n_points * sizeof(T));
    cudaMemcpy(init_pos_d, init_pos, 4*n_points * sizeof(T), cudaMemcpyHostToDevice);

    T* time_d;
    cudaMalloc((void**)&time_d, n_points*sizeof(T));
    cudaMemcpy(time_d, time_arr, n_points * sizeof(T), cudaMemcpyHostToDevice);

    // scratch spaces
    T* derivs_d;
    cudaMalloc((void**)&derivs_d, 7*map_rhs_to_n_deriv_outputs<id>()*n_points*sizeof(T));
    
    std::vector<T> dt_init(n_points, T(-1.0));
    T* dt_d;
    cudaMalloc((void**)&dt_d, n_points*sizeof(T));
    cudaMemcpy(dt_d, dt_init.data(), n_points*sizeof(T), cudaMemcpyHostToDevice);

    T* dtmax_d;
    cudaMalloc((void**)&dtmax_d, n_points*sizeof(T));

    std::vector<T> mu_init(n_points, T(-1.0));
    T* mu_d;
    cudaMalloc((void**)&mu_d, n_points*sizeof(T));
    cudaMemcpy(mu_d, mu_init.data(), n_points*sizeof(T), cudaMemcpyHostToDevice);

    T* out_d;
    cudaMalloc((void**)&out_d, 4*n_points * sizeof(T));

    // allocate and copy to device memory
    double x1_range_ext[4];
    double x2_range_ext[4];
    double x3_range_ext[4];

    for(int i=0; i<3; ++i){
        x1_range_ext[i] = x1_range_arr[i];
        x2_range_ext[i] = x2_range_arr[i];
        x3_range_ext[i] = x3_range_arr[i];
    }
    // precompute inverse grid sizes
    x1_range_ext[3] = (x1_range_ext[2] - 1) / (x1_range_ext[1] - x1_range_ext[0]) ;
    x2_range_ext[3] = (x2_range_ext[2] - 1) /(x2_range_ext[1] - x2_range_ext[0]) ;
    x3_range_ext[3] = (x3_range_ext[2] - 1)/ (x3_range_ext[1] - x3_range_ext[0]) ;

    int n_x1 = (x1_range_ext[2]-1)/3;
    int n_x2 = (x2_range_ext[2]-1)/3;
    int n_x3 = (x3_range_ext[2]-1)/3;
    int n_x23 = 64*n_x1*n_x2*n_x3;
    T tmax = 1e-2; // needed for setup_particle

    double grid_ranges[12];
    for(int i=0; i<4; ++i){
        grid_ranges[i] = x1_range_ext[i];
        grid_ranges[4 + i] = x2_range_ext[i];
        grid_ranges[8 + i] = x3_range_ext[i];
    }

    gpuErrchk(cudaMemcpyToSymbol(grid_ranges_d, grid_ranges, 12*sizeof(double)) );
    gpuErrchk(cudaMemcpyToSymbol(mass_d, &m, sizeof(double)));
    gpuErrchk(cudaMemcpyToSymbol(charge_d, &q, sizeof(double)));
    gpuErrchk(cudaMemcpyToSymbol(v_total_d, &v_total, sizeof(double)));

    gpuErrchk(cudaMemcpyToSymbol(n_x2_d, &n_x2, sizeof(int)) );
    gpuErrchk(cudaMemcpyToSymbol(n_x3_d, &n_x3, sizeof(int)) );
    gpuErrchk(cudaMemcpyToSymbol(n_x23_d, &n_x23, sizeof(int)) );

    bool is_test = true;
    gpuErrchk(cudaMemcpyToSymbol(is_test_d, &is_test, sizeof(bool)) ); 

    int nthreads = THREADS_PER_BLOCK;
    int nblks = n_points / PARTICLES_PER_BLOCK + 1;

    setup_kernel<T, id><<<nblks, nthreads>>>(init_pos_d, quadpts_d, mu_d, dt_d, dtmax_d,
                                                time_d, derivs_d, n_points, args...);
    test_gpu_derivs_kernel<T, id><<<nblks, nthreads>>>(quadpts_d, init_pos_d, time_d, out_d, derivs_d, mu_d,
                                                             dt_d, dtmax_d, n_points, args...);
    
    T out[4*n_points];
    gpuErrchk( cudaMemcpy(&out, out_d, 4*n_points * sizeof(T), cudaMemcpyDeviceToHost) );
    auto result = py::array_t<T>(4*n_points, out);
    
    gpuErrchk( cudaFree(quadpts_d) );
    gpuErrchk( cudaFree(init_pos_d) );
    gpuErrchk( cudaFree(time_d) );
    gpuErrchk( cudaFree(derivs_d) );
    gpuErrchk( cudaFree(mu_d) );
    gpuErrchk( cudaFree(dt_d) );
    gpuErrchk( cudaFree(dtmax_d) );
    gpuErrchk( cudaFree(out_d) );

    return result;
}


template<typename T>
py::array_t<T> test_derivatives_cartesian(py::array_t<T> quad_pts, py::array_t<double> x1_range, py::array_t<double> x2_range, py::array_t<double> x3_range, py::array_t<T> loc, py::array_t<T> vpar, double v_total, double m, double q, int n_points){        
    py::array_t<T> time = py::array_t<T>(n_points); // dummy time
    std::fill(time.mutable_data(), time.mutable_data() + n_points, 0.0);
    return test_gpu_derivatives<T, RHS::GC_CartesianVacuum>(quad_pts, x1_range, x2_range, x3_range, loc, vpar, time, v_total, m, q, n_points);
}

template py::array_t<double> test_derivatives_cartesian(py::array_t<double> quad_pts, py::array_t<double> x1_range, py::array_t<double> x2_range, py::array_t<double> x3_range, py::array_t<double> loc, py::array_t<double> vpar, double v_total, double m, double q, int n_points);  
template py::array_t<float> test_derivatives_cartesian(py::array_t<float> quad_pts, py::array_t<double> x1_range, py::array_t<double> x2_range, py::array_t<double> x3_range, py::array_t<float> loc, py::array_t<float> vpar, double v_total, double m, double q, int n_points);    


py::array_t<double> test_derivatives_boozer(py::array_t<double> quad_pts, py::array_t<double> x1_range, py::array_t<double> x2_range, py::array_t<double> x3_range, py::array_t<double> loc, py::array_t<double> vpar, double v_total, double m, double q, double psi0, int n_points, bool vacuum){
    gpuErrchk(cudaMemcpyToSymbol(psi0_d, &psi0, sizeof(double)));
    double inv_psi0_charge = 1.0 / (psi0*q);
    gpuErrchk(cudaMemcpyToSymbol(inv_psi0_charge_d, &inv_psi0_charge, sizeof(double)));

    py::array_t<double> time = py::array_t<double>(n_points); // dummy time
    std::fill(time.mutable_data(), time.mutable_data() + n_points, 0.0);

    if (vacuum) {
        return test_gpu_derivatives<double, RHS::GC_BoozerVacuum>(quad_pts, x1_range, x2_range, x3_range, loc, vpar, time, v_total, m, q, n_points);
    } else {
        return test_gpu_derivatives<double, RHS::GC_Boozer>(quad_pts, x1_range, x2_range, x3_range, loc, vpar, time, v_total, m, q, n_points);
    }
}

py::array_t<double> test_derivatives_saw(py::array_t<double> quad_pts, py::array_t<double> x1_range, py::array_t<double> x2_range, py::array_t<double> x3_range, 
        double saw_omega, py::array_t<double> saw_srange, py::array_t<int> saw_m, py::array_t<int> saw_n, py::array_t<double> saw_phihats, int saw_nharmonics,
        py::array_t<double> loc, py::array_t<double> vpar, py::array_t<double> time, double v_total, double m, double q,  double psi0, int n_points){

    double* saw_srange_arr = create_array(saw_srange);
    int* saw_m_arr = create_array(saw_m);
    int* saw_n_arr = create_array(saw_n);
    double* saw_phihats_arr = create_array(saw_phihats);
    
    int* saw_m_d;
    cudaMalloc((void**)&saw_m_d, saw_m.size() * sizeof(int));
    cudaMemcpy(saw_m_d, saw_m_arr, saw_m.size() * sizeof(int), cudaMemcpyHostToDevice);

    int* saw_n_d;
    cudaMalloc((void**)&saw_n_d, saw_n.size() * sizeof(int));
    cudaMemcpy(saw_n_d, saw_n_arr, saw_n.size() * sizeof(int), cudaMemcpyHostToDevice);

    double* saw_phihats_d;
    cudaMalloc((void**)&saw_phihats_d, saw_phihats.size() * sizeof(double));
    cudaMemcpy(saw_phihats_d, saw_phihats_arr, saw_phihats.size() * sizeof(double), cudaMemcpyHostToDevice);

    // allocate and copy to device memory
    double saw_srange_ext[4];
    for(int i=0; i<3; ++i){
        saw_srange_ext[i] = saw_srange_arr[i];
        
    }
    saw_srange_ext[3] = (saw_srange_ext[1] - saw_srange_ext[0]) / (saw_srange_ext[2] - 1);

    gpuErrchk(cudaMemcpyToSymbol(psi0_d, &psi0, sizeof(double)));
    gpuErrchk(cudaMemcpyToSymbol(saw_srange_d, saw_srange_ext, 4*sizeof(double)) );
    
    py::array_t<double> out = test_gpu_derivatives<double, RHS::GC_BoozerVacuumSAW>(quad_pts, x1_range, x2_range, x3_range, loc, vpar, time, v_total, m, q, n_points,
                                                                        saw_omega, saw_m_d, saw_n_d, saw_phihats_d, saw_nharmonics);

    gpuErrchk( cudaFree(saw_m_d) );
    gpuErrchk( cudaFree(saw_n_d) );
    gpuErrchk( cudaFree(saw_phihats_d) );
    return out;
}

py::array_t<double> test_derivatives_saw_nok(py::array_t<double> quad_pts, py::array_t<double> x1_range, py::array_t<double> x2_range, py::array_t<double> x3_range, 
        double saw_omega, py::array_t<double> saw_srange, py::array_t<int> saw_m, py::array_t<int> saw_n, py::array_t<double> saw_phihats, int saw_nharmonics,
        py::array_t<double> loc, py::array_t<double> vpar, py::array_t<double> time, double v_total, double m, double q,  double psi0, int n_points){

    double* saw_srange_arr = create_array(saw_srange);
    int* saw_m_arr = create_array(saw_m);
    int* saw_n_arr = create_array(saw_n);
    double* saw_phihats_arr = create_array(saw_phihats);

    int* saw_m_d;
    cudaMalloc((void**)&saw_m_d, saw_m.size() * sizeof(int));
    cudaMemcpy(saw_m_d, saw_m_arr, saw_m.size() * sizeof(int), cudaMemcpyHostToDevice);

    int* saw_n_d;
    cudaMalloc((void**)&saw_n_d, saw_n.size() * sizeof(int));
    cudaMemcpy(saw_n_d, saw_n_arr, saw_n.size() * sizeof(int), cudaMemcpyHostToDevice);

    double* saw_phihats_d;
    cudaMalloc((void**)&saw_phihats_d, saw_phihats.size() * sizeof(double));
    cudaMemcpy(saw_phihats_d, saw_phihats_arr, saw_phihats.size() * sizeof(double), cudaMemcpyHostToDevice);

    // allocate and copy to device memory
    double saw_srange_ext[4];
    for(int i=0; i<3; ++i){
        saw_srange_ext[i] = saw_srange_arr[i];
    }
    saw_srange_ext[3] = (saw_srange_ext[1] - saw_srange_ext[0]) / (saw_srange_ext[2] - 1);

    gpuErrchk(cudaMemcpyToSymbol(psi0_d, &psi0, sizeof(double)));
    gpuErrchk(cudaMemcpyToSymbol(saw_srange_d, saw_srange_ext, 4*sizeof(double)) );
        
    py::array_t<double> out = test_gpu_derivatives<double, RHS::GC_BoozerNoKSAW>(quad_pts, x1_range, x2_range, x3_range, loc, vpar, time, v_total, m, q, n_points,
                                                                        saw_omega, saw_m_d, saw_n_d, saw_phihats_d, saw_nharmonics);
    gpuErrchk( cudaFree(saw_m_d) );
    gpuErrchk( cudaFree(saw_n_d) );
    gpuErrchk( cudaFree(saw_phihats_d) );

    return out;
}

template<RHS id, typename... Args>
__global__ void test_gpu_timestep_kernel(double* out, double* init_pos, double* quadpts_arr, double* derivs, double* mu, 
                                            double* tmax, double* t, double* dt, double* dtmax, int nparticles, Args... args){
    int idx = threadIdx.x + blockIdx.x*PARTICLES_PER_BLOCK;
    double* loc_arr = init_pos + 4*idx;

    __shared__ double x_temp[5 * PARTICLES_PER_BLOCK];
    __shared__ double block_dt[PARTICLES_PER_BLOCK];
    __shared__ bool symmetry_exploited[PARTICLES_PER_BLOCK];
    double* block_derivs = derivs + 7*map_rhs_to_n_deriv_outputs<id>()*blockIdx.x*PARTICLES_PER_BLOCK;
    double* block_tmax = tmax + blockIdx.x*PARTICLES_PER_BLOCK;
    __shared__ int cell_index_start[3 * PARTICLES_PER_BLOCK];
    __shared__ double shape_fun_vals[12 * PARTICLES_PER_BLOCK];
    __shared__ double block_mu[PARTICLES_PER_BLOCK];
    __shared__ double block_t[PARTICLES_PER_BLOCK];
    __shared__ double block_dtmax[PARTICLES_PER_BLOCK];
    __shared__ double state[4 * PARTICLES_PER_BLOCK];
    __shared__ bool has_left[PARTICLES_PER_BLOCK];

    bool is_valid = idx < nparticles && threadIdx.x < PARTICLES_PER_BLOCK;
    __shared__ bool is_valid_arr[PARTICLES_PER_BLOCK];
    if(threadIdx.x < PARTICLES_PER_BLOCK){
        is_valid_arr[threadIdx.x] = is_valid;
    }
    // if thread is responsible for a valid particle id, load that particle's data
    if(is_valid){
        has_left[threadIdx.x] = false;
        for(int i=0; i<4; ++i){
            state[i*PARTICLES_PER_BLOCK + threadIdx.x] = loc_arr[i];
        }

        // load shared dt, t, mu, dtmax
        block_dt[threadIdx.x] = dt[idx];
        block_mu[threadIdx.x] = mu[idx];
        block_t[threadIdx.x] = t[idx];
        block_dtmax[threadIdx.x] = dtmax[idx];

    }
    __syncthreads();

    // if there exists a particle at t=0, which is a real particle, then keep tracing
    while(__syncthreads_count(block_t[threadIdx.x % PARTICLES_PER_BLOCK] == 0.0  && is_valid_arr[threadIdx.x % PARTICLES_PER_BLOCK]) > 0){
        // calculate the 7 Dormand-Prince 5 derivatives
        dp5_one_step<double, id, 0>(x_temp, block_derivs, quadpts_arr, cell_index_start, shape_fun_vals, block_t, block_dt,
                            symmetry_exploited, state, block_mu, is_valid_arr, args...);
        dp5_one_step<double, id, 1>(x_temp, block_derivs, quadpts_arr, cell_index_start, shape_fun_vals, block_t, block_dt,
                            symmetry_exploited, state, block_mu, is_valid_arr, args...);
        dp5_one_step<double, id, 2>(x_temp, block_derivs, quadpts_arr, cell_index_start, shape_fun_vals, block_t, block_dt,
                            symmetry_exploited, state, block_mu, is_valid_arr, args...);
        dp5_one_step<double, id, 3>(x_temp, block_derivs, quadpts_arr, cell_index_start, shape_fun_vals, block_t, block_dt,
                            symmetry_exploited, state, block_mu, is_valid_arr, args...);
        dp5_one_step<double, id, 4>(x_temp, block_derivs, quadpts_arr, cell_index_start, shape_fun_vals, block_t, block_dt,
                            symmetry_exploited, state, block_mu, is_valid_arr, args...);
        dp5_one_step<double, id, 5>(x_temp, block_derivs, quadpts_arr, cell_index_start, shape_fun_vals, block_t, block_dt,
                            symmetry_exploited, state, block_mu, is_valid_arr, args...);
        dp5_one_step<double, id, 6>(x_temp, block_derivs, quadpts_arr, cell_index_start, shape_fun_vals, block_t, block_dt,
                            symmetry_exploited, state, block_mu, is_valid_arr, args...);
        adjust_time<double, id>(block_t, block_dt, block_tmax, state, block_derivs, x_temp, has_left, block_dtmax, is_valid_arr);
        // if the particle moved, write output and load the next particle that is needed
        if(threadIdx.x < PARTICLES_PER_BLOCK && is_valid_arr[threadIdx.x] && block_t[threadIdx.x] != 0.0){
            // write output for current particle
            out[5*idx] = block_t[threadIdx.x];
            for(int i=0; i<4; ++i){
                out[5*idx + i + 1] = state[i*PARTICLES_PER_BLOCK + threadIdx.x];
            }

            // load the next particle
            idx = atomicAdd(&next_particle_d, 1);
            if(idx < nparticles){
                loc_arr = init_pos + 4*idx;
                for(int i=0; i<4; ++i){
                    state[i*PARTICLES_PER_BLOCK + threadIdx.x] = loc_arr[i];
                }
                block_dt[threadIdx.x] = dt[idx];
                block_mu[threadIdx.x] = mu[idx];
                block_t[threadIdx.x] = t[idx];
                block_dtmax[threadIdx.x] = dtmax[idx];
                has_left[threadIdx.x] = false;
                symmetry_exploited[threadIdx.x] = false;
            } else {
                is_valid_arr[threadIdx.x] = false;
            }
        }

        __syncthreads();
    }
    return;
}

template<RHS id, typename... Args>
vector<double> test_gpu_timestep(py::array_t<double> quad_pts, py::array_t<double> x1_range,
        py::array_t<double> x2_range, py::array_t<double> x3_range, py::array_t<double> loc_init, double m, double q, double vtotal, py::array_t<double> vtang, 
        double tol, int nparticles, Args... args){

    //  read data in from python
    double* loc_init_arr = create_array(loc_init);
    double* vtang_arr = create_array(vtang);
    double init_pos[4*nparticles];
    // load initial conditions
    for(int i=0; i<nparticles; ++i){
        double s     = loc_init_arr[3*i];
        double theta = loc_init_arr[3*i + 1];
        init_pos[4*i + 0] = s * cos(theta);
        init_pos[4*i + 1] = s * sin(theta);
        init_pos[4*i + 2] = loc_init_arr[3*i + 2];
        init_pos[4*i + 3] = vtang_arr[i];
    }
    
    double* quadpts_arr = create_array(quad_pts);
    double* x1_range_arr = create_array(x1_range);
    double* x2_range_arr = create_array(x2_range);
    double* x3_range_arr = create_array(x3_range);

    // allocate and copy to device memory
    double x1_range_ext[4];
    double x2_range_ext[4];
    double x3_range_ext[4];

    for(int i=0; i<3; ++i){
        x1_range_ext[i] = x1_range_arr[i];
        x2_range_ext[i] = x2_range_arr[i];
        x3_range_ext[i] = x3_range_arr[i];
    }
    // precompute inverse grid sizes
    x1_range_ext[3] = (x1_range_ext[2] - 1) / (x1_range_ext[1] - x1_range_ext[0]) ;
    x2_range_ext[3] = (x2_range_ext[2] - 1) /(x2_range_ext[1] - x2_range_ext[0]) ;
    x3_range_ext[3] = (x3_range_ext[2] - 1)/ (x3_range_ext[1] - x3_range_ext[0]) ;

    int n_x1 = (x1_range_ext[2]-1)/3;
    int n_x2 = (x2_range_ext[2]-1)/3;
    int n_x3 = (x3_range_ext[2]-1)/3;
    int n_x23 = 64*n_x1*n_x2*n_x3;
    double tmax = 1e-2; // needed for setup_particle

    double grid_ranges[12];
    for(int i=0; i<4; ++i){
        grid_ranges[i] = x1_range_ext[i];
        grid_ranges[4 + i] = x2_range_ext[i];
        grid_ranges[8 + i] = x3_range_ext[i];
    }

    gpuErrchk(cudaMemcpyToSymbol(grid_ranges_d, grid_ranges, 12*sizeof(double)) );
    gpuErrchk(cudaMemcpyToSymbol(mass_d, &m, sizeof(double)));
    gpuErrchk(cudaMemcpyToSymbol(charge_d, &q, sizeof(double)));
    gpuErrchk(cudaMemcpyToSymbol(atol_d, &tol, sizeof(double)));
    gpuErrchk(cudaMemcpyToSymbol(rtol_d, &tol, sizeof(double)));
    gpuErrchk(cudaMemcpyToSymbol(v_total_d, &vtotal, sizeof(double)));

    gpuErrchk(cudaMemcpyToSymbol(n_x2_d, &n_x2, sizeof(int)) );
    gpuErrchk(cudaMemcpyToSymbol(n_x3_d, &n_x3, sizeof(int)) );
    gpuErrchk(cudaMemcpyToSymbol(n_x23_d, &n_x23, sizeof(int)) );

    double* init_pos_d;
    gpuErrchk(cudaMalloc((void**)&init_pos_d, 4 * nparticles * sizeof(double)) );
    gpuErrchk(cudaMemcpy(init_pos_d, init_pos, 4 * nparticles * sizeof(double), cudaMemcpyHostToDevice) );

    double* quadpts_d;
    gpuErrchk( cudaMalloc((void**)&quadpts_d, quad_pts.size() * sizeof(double)) );
    gpuErrchk( cudaMemcpy(quadpts_d, quadpts_arr, quad_pts.size() * sizeof(double), cudaMemcpyHostToDevice) );

    std::vector<double> mu_init(nparticles, -1.0);
    double* mu_d;
    cudaMalloc((void**)&mu_d, nparticles*sizeof(double));
    cudaMemcpy(mu_d, mu_init.data(), nparticles*sizeof(double), cudaMemcpyHostToDevice);

    double* t_d;
    cudaMalloc((void**)&t_d, nparticles*sizeof(double));
    cudaMemset(t_d, 0, nparticles*sizeof(double)); 

    // scratch workspaces
    std::vector<double> dt_init(nparticles, double(-1.0));
    double* dt_d;
    cudaMalloc((void**)&dt_d, nparticles*sizeof(double));
    cudaMemcpy(dt_d, dt_init.data(), nparticles*sizeof(double), cudaMemcpyHostToDevice);

    std::vector<double> tmax_init(nparticles, double(1e-2));
    double* tmax_d;
    cudaMalloc((void**)&tmax_d, nparticles*sizeof(double));
    cudaMemcpy(tmax_d, tmax_init.data(), nparticles*sizeof(double), cudaMemcpyHostToDevice);

    double* dtmax_d;
    cudaMalloc((void**)&dtmax_d, nparticles*sizeof(double));

    double* out_d;
    gpuErrchk( cudaMalloc((void**)&out_d, 5 * nparticles * sizeof(double)) );

    // launch params
    int nthreads = THREADS_PER_BLOCK;
    int setup_nblks = nparticles / PARTICLES_PER_BLOCK + 1;

    // compute maximum number of blocks for a single wave for persistent threads
    int numSMs;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
    int blocks_per_sm;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm,
        test_gpu_timestep_kernel<id>, THREADS_PER_BLOCK, 0);
    int nblks = blocks_per_sm * numSMs;

    int scratch_nblks = max(setup_nblks, nblks);
    double* derivs_d;
    cudaMalloc((void**)&derivs_d, 7*map_rhs_to_n_deriv_outputs<id>()*scratch_nblks*PARTICLES_PER_BLOCK*sizeof(double));


    setup_kernel<double, id><<<setup_nblks, nthreads>>>(init_pos_d, quadpts_d, mu_d, dt_d, dtmax_d,
                                            t_d, derivs_d, nparticles, args...);

    // initialize global counter
    int n_total_threads = nblks*PARTICLES_PER_BLOCK;
    gpuErrchk(cudaMemcpyToSymbol(next_particle_d, &n_total_threads, sizeof(int)) );
    test_gpu_timestep_kernel<id><<<nblks, nthreads>>>(out_d, init_pos_d, quadpts_d, derivs_d, mu_d, tmax_d, t_d, dt_d, dtmax_d, nparticles, args...);

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    double out[5*nparticles];
    gpuErrchk( cudaMemcpy(out, out_d, 5 * nparticles * sizeof(double), cudaMemcpyDeviceToHost) );

    vector<double> particle_output(5*nparticles);
    for(int i=0; i<5*nparticles; ++i){
        particle_output[i] = out[i];
    }

    gpuErrchk( cudaFree(init_pos_d) );
    gpuErrchk( cudaFree(quadpts_d) );
    gpuErrchk( cudaFree(out_d) );
    gpuErrchk( cudaFree(derivs_d));
    gpuErrchk( cudaFree(mu_d));
    gpuErrchk( cudaFree(dt_d));
    gpuErrchk( cudaFree(t_d));
    gpuErrchk( cudaFree(dtmax_d));

    return particle_output;
}

vector<double> test_timestep_cartesian(py::array_t<double> quad_pts, py::array_t<double> x1_range,
        py::array_t<double> x2_range, py::array_t<double> x3_range, py::array_t<double> loc_init, double m, double q, double vtotal, py::array_t<double> vtang, 
        double tol, int nparticles){
    bool rescale_abstol_var = false;
    gpuErrchk(cudaMemcpyToSymbol(rescale_abstol_var_d, &rescale_abstol_var, sizeof(bool)) );
    return test_gpu_timestep<RHS::GC_CartesianVacuum>(quad_pts, x1_range, x2_range, x3_range, loc_init, m, q, vtotal, vtang, tol, nparticles);
}

vector<double> test_timestep_boozer(py::array_t<double> quad_pts, py::array_t<double> x1_range,
        py::array_t<double> x2_range, py::array_t<double> x3_range, py::array_t<double> loc_init, double m, double q, double vtotal, py::array_t<double> vtang, 
        double tol, double psi0, int nparticles, bool vacuum){

    gpuErrchk(cudaMemcpyToSymbol(psi0_d, &psi0, sizeof(double)));
    double inv_psi0_charge = 1.0 / (psi0*q);
    gpuErrchk(cudaMemcpyToSymbol(inv_psi0_charge_d, &inv_psi0_charge, sizeof(double)));
    vector<double> particle_output;
    if (vacuum) {
        particle_output = test_gpu_timestep<RHS::GC_BoozerVacuum>(quad_pts, x1_range, x2_range, x3_range, loc_init, m, q, vtotal, vtang, tol, nparticles);
    } else {
        particle_output = test_gpu_timestep<RHS::GC_Boozer>(quad_pts, x1_range, x2_range, x3_range, loc_init, m, q, vtotal, vtang, tol, nparticles);
    }

    for(int i=0; i<nparticles; ++i){
        double x1 = particle_output[5*i + 1];
        double x2 = particle_output[5*i + 2];
        double s = sqrt(x1*x1 + x2*x2);
        double theta = atan2(x2, x1);

        particle_output[5*i] = particle_output[5*i];
        particle_output[5*i+1] = s;
        particle_output[5*i+2] = theta;
        particle_output[5*i+3] = particle_output[5*i+3];
        particle_output[5*i+4] = particle_output[5*i+4];
    }

    return particle_output;
}

vector<double> test_timestep_saw(py::array_t<double> quad_pts, py::array_t<double> x1_range, py::array_t<double> x2_range, py::array_t<double> x3_range, 
        double saw_omega, py::array_t<double> saw_srange, py::array_t<int> saw_m, py::array_t<int> saw_n, py::array_t<double> saw_phihats, int saw_nharmonics,
        py::array_t<double> loc_init, double m, double q, double v_total, py::array_t<double> vtang, py::array_t<double> time,
        double tol, double psi0, int nparticles){
 
    double* saw_srange_arr = create_array(saw_srange);
    int* saw_m_arr = create_array(saw_m);
    int* saw_n_arr = create_array(saw_n);
    double* saw_phihats_arr = create_array(saw_phihats);

    int* saw_m_d;
    cudaMalloc((void**)&saw_m_d, saw_m.size() * sizeof(int));
    cudaMemcpy(saw_m_d, saw_m_arr, saw_m.size() * sizeof(int), cudaMemcpyHostToDevice);

    int* saw_n_d;
    cudaMalloc((void**)&saw_n_d, saw_n.size() * sizeof(int));
    cudaMemcpy(saw_n_d, saw_n_arr, saw_n.size() * sizeof(int), cudaMemcpyHostToDevice);

    double* saw_phihats_d;
    cudaMalloc((void**)&saw_phihats_d, saw_phihats.size() * sizeof(double));
    cudaMemcpy(saw_phihats_d, saw_phihats_arr, saw_phihats.size() * sizeof(double), cudaMemcpyHostToDevice);

    double* out_d;
    gpuErrchk( cudaMalloc((void**)&out_d, 5 * nparticles * sizeof(double)) );

    // allocate and copy to device memory
    double saw_srange_ext[4];
    for(int i=0; i<3; ++i){
        saw_srange_ext[i] = saw_srange_arr[i];
    }
    saw_srange_ext[3] = (saw_srange_ext[1] - saw_srange_ext[0]) / (saw_srange_ext[2] - 1);

    gpuErrchk(cudaMemcpyToSymbol(psi0_d, &psi0, sizeof(double)));
    gpuErrchk(cudaMemcpyToSymbol(saw_srange_d, saw_srange_ext, 4*sizeof(double)) );
    vector<double> particle_output = test_gpu_timestep<RHS::GC_BoozerVacuumSAW>(quad_pts, x1_range, x2_range, x3_range, loc_init, m, q, v_total, vtang, tol, nparticles,
                                                                        saw_omega, saw_m_d, saw_n_d, saw_phihats_d, saw_nharmonics);
    for(int i=0; i<nparticles; ++i){
        double x1 = particle_output[5*i + 1];
        double x2 = particle_output[5*i + 2];
        double s = sqrt(x1*x1 + x2*x2);
        double theta = atan2(x2, x1);

        particle_output[5*i] = particle_output[5*i];
        particle_output[5*i+1] = s;
        particle_output[5*i+2] = theta;
        particle_output[5*i+3] = particle_output[5*i+3];
        particle_output[5*i+4] = particle_output[5*i+4];
    }
    gpuErrchk( cudaFree(saw_m_d) );
    gpuErrchk( cudaFree(saw_n_d) );
    gpuErrchk( cudaFree(saw_phihats_d) );
    gpuErrchk( cudaFree(out_d) );
 
    return particle_output;

};

vector<double> test_timestep_saw_nok(py::array_t<double> quad_pts, py::array_t<double> x1_range, py::array_t<double> x2_range, py::array_t<double> x3_range, 
        double saw_omega, py::array_t<double> saw_srange, py::array_t<int> saw_m, py::array_t<int> saw_n, py::array_t<double> saw_phihats, int saw_nharmonics,
        py::array_t<double> loc_init, double m, double q, double v_total, py::array_t<double> vtang, py::array_t<double> time,
        double tol, double psi0, int nparticles){
 
    double* saw_srange_arr = create_array(saw_srange);
    int* saw_m_arr = create_array(saw_m);
    int* saw_n_arr = create_array(saw_n);
    double* saw_phihats_arr = create_array(saw_phihats);

    int* saw_m_d;
    cudaMalloc((void**)&saw_m_d, saw_m.size() * sizeof(int));
    cudaMemcpy(saw_m_d, saw_m_arr, saw_m.size() * sizeof(int), cudaMemcpyHostToDevice);

    int* saw_n_d;
    cudaMalloc((void**)&saw_n_d, saw_n.size() * sizeof(int));
    cudaMemcpy(saw_n_d, saw_n_arr, saw_n.size() * sizeof(int), cudaMemcpyHostToDevice);

    double* saw_phihats_d;
    cudaMalloc((void**)&saw_phihats_d, saw_phihats.size() * sizeof(double));
    cudaMemcpy(saw_phihats_d, saw_phihats_arr, saw_phihats.size() * sizeof(double), cudaMemcpyHostToDevice);

    double* out_d;
    gpuErrchk( cudaMalloc((void**)&out_d, 5 * nparticles * sizeof(double)) );

    // allocate and copy to device memory
    double saw_srange_ext[4];
    for(int i=0; i<3; ++i){
        saw_srange_ext[i] = saw_srange_arr[i];
    }
    saw_srange_ext[3] = (saw_srange_ext[1] - saw_srange_ext[0]) / (saw_srange_ext[2] - 1);

    gpuErrchk(cudaMemcpyToSymbol(psi0_d, &psi0, sizeof(double)));
    gpuErrchk(cudaMemcpyToSymbol(saw_srange_d, saw_srange_ext, 4*sizeof(double)) );
    vector<double> particle_output = test_gpu_timestep<RHS::GC_BoozerNoKSAW>(quad_pts, x1_range, x2_range, x3_range, loc_init, m, q, v_total, vtang, tol, nparticles,
                                                                        saw_omega, saw_m_d, saw_n_d, saw_phihats_d, saw_nharmonics);
    for(int i=0; i<nparticles; ++i){
        double x1 = particle_output[5*i + 1];
        double x2 = particle_output[5*i + 2];
        double s = sqrt(x1*x1 + x2*x2);
        double theta = atan2(x2, x1);

        particle_output[5*i] = particle_output[5*i];
        particle_output[5*i+1] = s;
        particle_output[5*i+2] = theta;
        particle_output[5*i+3] = particle_output[5*i+3];
        particle_output[5*i+4] = particle_output[5*i+4];
    }
    gpuErrchk( cudaFree(saw_m_d) );
    gpuErrchk( cudaFree(saw_n_d) );
    gpuErrchk( cudaFree(saw_phihats_d) );
    gpuErrchk( cudaFree(out_d) );

    return particle_output;

};
