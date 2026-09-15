#include "regular_grid_interpolant_3d.h"
#include <xtensor/xarray.hpp>
#include "xtensor/xlayout.hpp"
#define _USE_MATH_DEFINES
#include <math.h>
#include <boost/format.hpp>
#include <iostream>
#include <iomanip>
#ifdef USE_MPI
#include <mpi.h>
#endif

#define _EPS_ 1e-13

template<class Array>
const int RegularGridInterpolant3D<Array>::MAX_NODES;

// Copy the dof-ordered vals array into per-cell contiguous blocks
// (local_vals_flat + cell_offsets), which is the storage evaluate_local()
// reads. vals is freed afterwards; get_interpolant_data() reconstructs it
// on demand via reconstruct_vals().
template<class Array>
void RegularGridInterpolant3D<Array>::build_local_vals() {
    int degree = rule.degree;
    cell_offsets.assign((size_t)nx*ny*nz, -1);
    local_vals_flat.assign((size_t)cells_to_keep * local_vals_size, 0.);
    int64_t flat_offset = 0;
    for (int xidx = 0; xidx < nx; ++xidx) {
        for (int yidx = 0; yidx < ny; ++yidx) {
            for (int zidx = 0; zidx < nz; ++zidx) {
                int meshidx = idx_cell(xidx, yidx, zidx);
                if(skip_cell[meshidx])
                    continue;
                cell_offsets[meshidx] = flat_offset;
                double* local_vals = local_vals_flat.data() + flat_offset;
                for (int i = 0; i < degree+1; ++i) {
                    for (int j = 0; j < degree+1; ++j) {
                        for (int k = 0; k < degree+1; ++k) {
                            size_t offset = (size_t)value_size*full_to_reduced_map[idx_dof(xidx*degree+i, yidx*degree+j, zidx*degree+k)];
                            int offset_local = value_size * idx_dof_local(i, j, k);
                            for (int l = 0; l < value_size; ++l) {
                                local_vals[offset_local + l] = vals[offset + l];
                            }
                        }
                    }
                }
                flat_offset += local_vals_size;
            }
        }
    }
    Vec().swap(vals);
}

template<class Array>
Vec RegularGridInterpolant3D<Array>::reconstruct_vals() const {
    int degree = rule.degree;
    Vec v((size_t)dofs_to_keep * value_size, 0.);
    for (int xidx = 0; xidx < nx; ++xidx) {
        for (int yidx = 0; yidx < ny; ++yidx) {
            for (int zidx = 0; zidx < nz; ++zidx) {
                int64_t flat_offset = cell_offsets[idx_cell(xidx, yidx, zidx)];
                if(flat_offset < 0)
                    continue;
                const double* local_vals = local_vals_flat.data() + flat_offset;
                for (int i = 0; i < degree+1; ++i) {
                    for (int j = 0; j < degree+1; ++j) {
                        for (int k = 0; k < degree+1; ++k) {
                            size_t offset = (size_t)value_size*full_to_reduced_map[idx_dof(xidx*degree+i, yidx*degree+j, zidx*degree+k)];
                            int offset_local = value_size * idx_dof_local(i, j, k);
                            for (int l = 0; l < value_size; ++l) {
                                v[offset + l] = local_vals[offset_local + l];
                            }
                        }
                    }
                }
            }
        }
    }
    return v;
}

template<class Array>
void RegularGridInterpolant3D<Array>::interpolate_batch(std::function<Vec(Vec, Vec, Vec)> &f) {
    vals.assign((size_t)dofs_to_keep * value_size, 0.);
    int BATCH_SIZE = 16384;
    int NUM_BATCHES = dofs_to_keep/BATCH_SIZE + (dofs_to_keep % BATCH_SIZE != 0);

    for (int i = 0; i < NUM_BATCHES; ++i) {
        uint32_t first = i * BATCH_SIZE;
        uint32_t last = std::min((uint32_t)((i+1) * BATCH_SIZE), dofs_to_keep);
        Vec xsub(last-first, 0.), ysub(last-first, 0.), zsub(last-first, 0.);
        dof_coords(first, last, xsub, ysub, zsub);
        Vec fxyzsub  = f(xsub, ysub, zsub);
        for (int j = 0; j < last-first; ++j) {
            for (int l = 0; l < value_size; ++l) {
                vals[first * value_size + j * value_size + l] = fxyzsub[j * value_size + l];
            }
        }
    }
    build_local_vals();
}

#ifdef USE_MPI
template<class Array>
void RegularGridInterpolant3D<Array>::interpolate_batch(std::function<Vec(Vec, Vec, Vec)> &f, MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    int total_dofs = dofs_to_keep;
    int base = total_dofs / size;
    int rem = total_dofs % size;
    int local_dofs = base + (rank < rem ? 1 : 0);
    int local_first = rank * base + (rank < rem ? rank : rem);
    int local_last = local_first + local_dofs;

    int BATCH_SIZE = 16384;
    Vec local_vals(local_dofs * value_size, 0.);

    for (int first = local_first; first < local_last; first += BATCH_SIZE) {
        int last = std::min(first + BATCH_SIZE, local_last);
        Vec xsub(last-first, 0.), ysub(last-first, 0.), zsub(last-first, 0.);
        dof_coords(first, last, xsub, ysub, zsub);
        Vec fxyzsub = f(xsub, ysub, zsub);

        int local_offset = (first - local_first) * value_size;
        for (int j = 0; j < last-first; ++j) {
            for (int l = 0; l < value_size; ++l) {
                local_vals[local_offset + j * value_size + l] = fxyzsub[j * value_size + l];
            }
        }
    }
    std::vector<int> recv_counts(size, 0);
    std::vector<int> recv_displs(size, 0);
    for (int r = 0; r < size; ++r) {
        int r_dofs = base + (r < rem ? 1 : 0);
        recv_counts[r] = r_dofs * value_size;
        if (r > 0) {
            recv_displs[r] = recv_displs[r - 1] + recv_counts[r - 1];
        }
    }

    vals = Vec(total_dofs * value_size, 0.);
    MPI_Allgatherv(
        local_vals.data(),
        local_dofs * value_size,
        MPI_DOUBLE,
        vals.data(),
        recv_counts.data(),
        recv_displs.data(),
        MPI_DOUBLE,
        comm
    );
    // Build the per-cell evaluation storage (same on all ranks)
    build_local_vals();
}
#endif

template<class Array>
void RegularGridInterpolant3D<Array>::evaluate_batch(Array& xyz, Array& fxyz){
    if(fxyz.layout() != xt::layout_type::row_major)
          throw std::runtime_error("fxyz needs to be in row-major storage order");
    int npoints = xyz.shape(0);
    for (int i = 0; i < npoints; ++i) {
        evaluate_inplace(xyz(i, 0), xyz(i, 1), xyz(i, 2), fxyz.data() + value_size*i);
    }
}

template<class Array>
Vec RegularGridInterpolant3D<Array>::evaluate(double x, double y, double z){
    Vec fxyz(value_size, 0.);
    evaluate_inplace(x, y, z, fxyz.data());
    return fxyz;

}

template<class Array>
void RegularGridInterpolant3D<Array>::evaluate_inplace(double x, double y, double z, double* res){

    // to avoid funny business when the data is just a tiny bit out of bounds
    // due to machine precision, we perform this check and shift
    if(x >= xmax) x -= _EPS_;
    else if (x <= xmin) x += _EPS_;
    if(y >= ymax) y -= _EPS_;
    else if (y <= ymin) y += _EPS_;
    if(z >= zmax) z -= _EPS_;
    else if (z <= zmin) z += _EPS_;

    int xidx = int(nx*(x-xmin)/(xmax-xmin)); // find idx so that xmesh[xidx] <= x <= xs[xidx+1]
    int yidx = int(ny*(y-ymin)/(ymax-ymin));
    int zidx = int(nz*(z-zmin)/(zmax-zmin));
    if(!out_of_bounds_ok){
        if(xidx < 0 || xidx >= nx)
            throw std::runtime_error((boost::format("xidx=%d not within [0, %d]") % xidx % (nx-1)).str());
        if(yidx < 0 || yidx >= ny)
            throw std::runtime_error((boost::format("yidx=%d not within [0, %d]") % yidx % (ny-1)).str());
        if(zidx < 0 || zidx >= nz)
            throw std::runtime_error((boost::format("zidx=%d not within [0, %d]") % zidx % (nz-1)).str());
    } else {
        // Clamp cell indices to match CUDA kernel convention (cuda_kernel.cu:700-704).
        xidx = std::max(0, std::min(nx-1, xidx));
        yidx = std::max(0, std::min(ny-1, yidx));
        zidx = std::max(0, std::min(nz-1, zidx));
    }
    double xlocal = (x-xmesh[xidx])/hx;
    double ylocal = (y-ymesh[yidx])/hy;
    double zlocal = (z-zmesh[zidx])/hz;
    return evaluate_local(xlocal, ylocal, zlocal, idx_cell(xidx, yidx, zidx), res);
}

// Evaluate one cell: res[l] = sum_i p_i(x) [ sum_j p_j(y) [ sum_k p_k(z)
// v[i,j,k,l] ] ], over the (degree+1)^3 dofs of the cell, whose values are
// stored contiguously (VS per dof) at vp.
template<int VS>
static inline void interp_cell_accumulate(const double* __restrict vp, const double* pkxs, const double* pkys, const double* pkzs, int degree, double* __restrict res)
{
    double sumi[VS];
    for (int l = 0; l < VS; ++l)
        sumi[l] = 0.;
    for (int i = 0; i < degree+1; ++i) {
        double sumj[VS];
        for (int l = 0; l < VS; ++l)
            sumj[l] = 0.;
        for (int j = 0; j < degree+1; ++j) {
            double sumk[VS];
            for (int l = 0; l < VS; ++l)
                sumk[l] = 0.;
            for (int k = 0; k < degree+1; ++k) {
                double pkz = pkzs[k];
                for (int l = 0; l < VS; ++l)
                    sumk[l] += vp[l] * pkz;
                vp += VS;
            }
            double pjy = pkys[j];
            for (int l = 0; l < VS; ++l)
                sumj[l] += sumk[l] * pjy;
        }
        double pix = pkxs[i];
        for (int l = 0; l < VS; ++l)
            sumi[l] += sumj[l] * pix;
    }
    for (int l = 0; l < VS; ++l)
        res[l] = sumi[l];
}

template<class Array>
void RegularGridInterpolant3D<Array>::evaluate_local(double x, double y, double z, int cell_idx, double* res)
{
    int degree = rule.degree;
    int64_t flat_offset = cell_offsets[cell_idx];
    if (flat_offset < 0) {
        if(out_of_bounds_ok)
            return;
        else
            throw std::runtime_error((boost::format("cell_idx=%d is outside the interpolation domain") % cell_idx).str());
    }

    double pkxs[MAX_NODES], pkys[MAX_NODES], pkzs[MAX_NODES];
    if(xsimd::simd_type<double>::size >= 3){
        simd_t xyz;
        xyz[0] = x;
        xyz[1] = y;
        xyz[2] = z;
        for (int k = 0; k < degree+1; ++k) {
            simd_t temp = this->rule.basis_fun(k, xyz);
            pkxs[k] = temp[0];
            pkys[k] = temp[1];
            pkzs[k] = temp[2];
        }
    } else {
        for (int k = 0; k < degree+1; ++k) {
            pkxs[k] = this->rule.basis_fun(k, x);
            pkys[k] = this->rule.basis_fun(k, y);
            pkzs[k] = this->rule.basis_fun(k, z);
        }
    }

    const double* val_ptr = local_vals_flat.data() + flat_offset;
    switch (value_size) {
        // dispatch the common small sizes to compile-time kernels
        case 1: interp_cell_accumulate<1>(val_ptr, pkxs, pkys, pkzs, degree, res); return;
        case 2: interp_cell_accumulate<2>(val_ptr, pkxs, pkys, pkzs, degree, res); return;
        case 3: interp_cell_accumulate<3>(val_ptr, pkxs, pkys, pkzs, degree, res); return;
        case 4: interp_cell_accumulate<4>(val_ptr, pkxs, pkys, pkzs, degree, res); return;
    }
    // general case (value_size > 4): same nesting as above, with scratch
    // sized at runtime. The scratch vectors allocate on every call, which is
    // acceptable because no field interpolant takes this path: everything in
    // boozermagneticfield_interpolated.h has value_size <= 3.
    const double* __restrict vp = val_ptr;
    double* __restrict out = res;
    std::vector<double> sumj(value_size), sumk(value_size);
    for (int l = 0; l < value_size; ++l) {
        out[l] = 0.;
    }
    for (int i = 0; i < degree+1; ++i) {
        std::fill(sumj.begin(), sumj.end(), 0.);
        for (int j = 0; j < degree+1; ++j) {
            std::fill(sumk.begin(), sumk.end(), 0.);
            for (int k = 0; k < degree+1; ++k) {
                double pkz = pkzs[k];
                for (int l = 0; l < value_size; ++l) {
                    sumk[l] += vp[l] * pkz;
                }
                vp += value_size;
            }
            double pjy = pkys[j];
            for (int l = 0; l < value_size; ++l) {
                sumj[l] += sumk[l] * pjy;
            }
        }
        double pix = pkxs[i];
        for (int l = 0; l < value_size; ++l) {
            out[l] += sumj[l] * pix;
        }
    }
}

template<class Array>
std::pair<double, double> RegularGridInterpolant3D<Array>::estimate_error(std::function<Vec(Vec, Vec, Vec)> &f, int samples) {
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(0.0, +1.0);
    double err = 0;
    double errsq = 0;
    Vec xs(samples, 0.);
    Vec ys(samples, 0.);
    Vec zs(samples, 0.);
    Array xyz = xt::zeros<double>({samples, 3});
    Array fhxyz = xt::zeros<double>({samples, value_size});
    for (int i = 0; i < samples; ++i) {
        xs[i] = xmin + distribution(generator)*(xmax-xmin);
        ys[i] = ymin + distribution(generator)*(ymax-ymin);
        zs[i] = zmin + distribution(generator)*(zmax-zmin);
        xyz(i, 0) = xs[i];
        xyz(i, 1) = ys[i];
        xyz(i, 2) = zs[i];
    }
    Vec fx = f(xs, ys, zs);
    this->evaluate_batch(xyz, fhxyz);
    for (int i = 0; i < samples; ++i) {
        double diff = 0.;
        for (int l = 0; l < value_size; ++l) {
            diff += std::pow(fx[value_size*i+l]-fhxyz(i, l), 2);
        }
        diff = std::sqrt(diff);
        err += diff;
        errsq += diff*diff;
    }
    double mean = err/samples;
    double std = std::sqrt((errsq - err*err/samples)/(samples-1)/samples);
    return std::make_pair(mean-std, mean+std);
}



Vec linspace(double min, double max, int n, bool endpoint) {
    Vec res(n, 0.);
    if(endpoint) {
        double h = (max-min)/(n-1);
        for (int i = 0; i < n; ++i)
            res[i] = min + i*h;
    } else {
        double h = (max-min)/n;
        for (int i = 0; i < n; ++i)
            res[i] = min + i*h;
    }
    return res;
}

template<class Array>
std::map<std::string, std::vector<double>> RegularGridInterpolant3D<Array>::get_interpolant_data() const {
    std::map<std::string, std::vector<double>> data;
    data["vals"] = reconstruct_vals();
    data["nx"] = {static_cast<double>(nx)};
    data["ny"] = {static_cast<double>(ny)};
    data["nz"] = {static_cast<double>(nz)};
    data["hx"] = {hx};
    data["hy"] = {hy};
    data["hz"] = {hz};
    data["xmin"] = {xmin};
    data["ymin"] = {ymin};
    data["zmin"] = {zmin};
    data["xmax"] = {xmax};
    data["ymax"] = {ymax};
    data["zmax"] = {zmax};
    data["value_size"] = {static_cast<double>(value_size)};
    data["degree"] = {static_cast<double>(rule.degree)};
    return data;
}

template<class Array>
void RegularGridInterpolant3D<Array>::set_interpolant_data(const std::map<std::string, std::vector<double>>& data) {
    // Look up "vals" in the map. find() returns an iterator pointing to a
    // std::pair<key, value>. Access the value via .second (key is .first).
    auto it = data.find("vals");
    if (it == data.end())
        throw std::runtime_error("set_interpolant_data: no 'vals' entry in data");
    if (it->second.size() != (size_t)dofs_to_keep * value_size)
        throw std::runtime_error((boost::format("set_interpolant_data: 'vals' has size %d but this grid needs %d") % it->second.size() % ((size_t)dofs_to_keep * value_size)).str());
    vals = it->second;

    // Rebuild the cell-indexed evaluation storage from vals - same
    // construction as interpolate_batch() but without calling the function
    // to compute values.
    build_local_vals();
}
