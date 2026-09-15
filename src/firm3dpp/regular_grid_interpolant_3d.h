#pragma once

#include "simdhelpers.h"
#include <map>
#include <algorithm>
#include <functional>
#include <iostream>
#include <random>
#include <stdexcept>
#include <stdint.h>
#include <tuple>
#include <vector>

// Raw MPI support (works with any MPI implementation, including mpi4py)
#ifdef USE_MPI
#include <mpi.h>
#endif

using Vec = std::vector<double>;
using RangeTriplet = std::tuple<double, double, int>;

Vec linspace(double min, double max, int n, bool endpoint);

class InterpolationRule {
    /* An InterpolationRule consists of a list of interpolation nodes and then
     * uses standard Lagrange interpolation, that is for each node x_i we get a
     * basis function p_i that satisfies p_i(x_i) = 1 and p_i(x_j)=0. This
     * function is given by p_i(x) = [Π_{i≠j} (x-x_i)/(x_j-x_i)]. To compute
     * this efficiently, we store the denominator in `scalings`, i.e.
     * scalings[j] = [Π_{i≠j} 1/(x_j-x_i)].
    */

    protected:
        void build_scalings(){
            for(int idx = 0; idx < degree+1; ++idx) {
                for(int i = 0; i < degree+1; ++i) {
                    if(i == idx) continue;
                    scalings[idx] *= 1./(nodes[idx]-nodes[i]);
                }
            }
        }
    public:
        Vec nodes;
        Vec scalings;
        const int degree;
        InterpolationRule(int degree) : degree(degree), nodes(degree+1, 0.), scalings(degree+1, 1.0) { }

        double basis_fun(int idx, double x) const {
            // evaluate the basisfunction p_idx at location x
            double res = scalings[idx];
            for(int i = 0; i < degree+1; ++i) {
                if(i == idx) continue;
                res *= (x-nodes[i]);
            }
            return res;
        }

        simd_t basis_fun(int idx, simd_t x) const {
            // evaluate the basisfunction p_idx at multiple locations stored in x
            simd_t res(scalings[idx]);
            for(int i = 0; i < degree+1; ++i) {
                if(i == idx) continue;
                res *= (x-simd_t(nodes[i]));
            }
            return res;
        }
};

template<class Array>
class RegularGridInterpolant3D {
    /* This class implements a vector-valued piecewise polynomial interpolant
     * on a regular grid in three dimensions.  There are many ways to
     * implemented interpolants, the implementation here is done to favour
     * speed over memory efficiency, and is designed for vector valued
     * functions for which SIMD vectorization pays off.
     *
     * The interpolant assumes a regular mesh on a rectangular cuboid. In order
     * to work on other geometries with reasonable efficiency, parts of the
     * mesh can be "skipped". This is done as follows: if a function is meant
     * to be interpolated on some irregular domain Ω, one first picks a
     * rectangular domain D so that Ω⊂D, and then defines a function `skip`
     * that returns true outside of Ω and false inside of Ω. Internally, we
     * then skip all cells for which the function returns true on all eight
     * corners. For some pictures that explain the idea behind the skip
     * function and its caveats, please check the discussion under
     * https://github.com/hiddenSymmetries/simsopt/pull/227
     */
    private:
        const int nx, ny, nz;  // number of cells in x, y, and z direction
        double hx, hy, hz; // gridsize in x, y, and z direction
        const double xmin, ymin, zmin; // lower bounds of the x, y, and z coordinates
        const double xmax, ymax, zmax; // lower bounds of the x, y, and z coordinates
        const int value_size; // number of output dimensions of the interpolant, i.e. space that is mapped into
        const InterpolationRule rule; // the interpolation rule to use on each cell in the grid
        const bool out_of_bounds_ok; // whether to do nothing or throw an error when the interpolant is queried at an out-of-bounds point

        // location of the mesh nodes in [xmin, xmax], [ymin, ymax], and [zmin, zmax]
        // has size nx+1, ny+1, nz+1 respectively
        Vec xmesh, ymesh, zmesh;

        // location of the mesh nodes in [xmin, xmax], [ymin, ymax], and [zmin, zmax]. superset of xmesh, ymesh, zmesh
        // has size nx*degree + 1, ny*degree + 1, and nz*degree + 1 respectively
        Vec xdof, ydof, zdof;

        // dof-ordered function values, of size dofs_to_keep * value_size.
        // Transient: freed once the per-cell storage below is built from it,
        // rebuilt on demand by get_interpolant_data().
        Vec vals;
        // Evaluation storage: one contiguous block of (degree+1)^3 * value_size
        // values per kept cell (dofs in idx_dof_local order, values contiguous
        // per dof). cell_offsets[cell] is the block's offset, or -1 for skipped
        // cells; it is sized nx*ny*nz from construction and evaluate_inplace()
        // keeps cell indices in range, so lookups index it directly.
        Vec local_vals_flat;
        std::vector<int64_t> cell_offsets;
        std::vector<bool> skip_cell; // whether to skip each cell or not
        // since we are skipping some dofs, we need mappings into the list of
        // reduced dofs, e.g. if we skip dofs 3, then reduced to full would
        // look like [0, 1, 2, 4, 5, ...]
        std::vector<uint32_t> reduced_to_full_map, full_to_reduced_map;

        uint32_t cells_to_skip, cells_to_keep, dofs_to_skip, dofs_to_keep; // which cells and dofs we skip and keep
        int local_vals_size; // length of each cell's block in local_vals_flat

        static const int MAX_NODES = 16; // upper bound on degree+1, so basis function values fit in stack buffers

        inline int idx_dof(int i, int j, int k) const {
            int degree = rule.degree;
            return i*(ny*degree+1)*(nz*degree+1) + j*(nz*degree+1) + k;
        }

        inline int idx_cell(int i, int j, int k) const {
            return i*ny*nz + j*nz + k;
        }

        inline int idx_mesh(int i, int j, int k) const {
            return i*(ny+1)*(nz+1) + j*(nz+1) + k;
        }

        inline int idx_dof_local(int i, int j, int k) const {
            int degree = rule.degree;
            return i*(degree+1)*(degree+1) + j*(degree+1) + k;
        }

        void evaluate_inplace(double x, double y, double z, double* res);
        void evaluate_local(double x, double y, double z, int cell_idx, double *res);
        void build_local_vals(); // build local_vals_flat/cell_offsets from vals, then free vals
        Vec reconstruct_vals() const; // rebuild the dof-ordered vals array from local_vals_flat

        // Fill xsub/ysub/zsub (each of size last-first) with the coordinates
        // of the reduced dofs [first, last). Computed on the fly from the 1d
        // dof arrays rather than stored: the full tensor-product coordinate
        // arrays would triple the interpolant's persistent memory footprint.
        void dof_coords(uint32_t first, uint32_t last, Vec& xsub, Vec& ysub, Vec& zsub) const {
            int degree = rule.degree;
            int nyd = ny*degree+1, nzd = nz*degree+1;
            for (uint32_t idx = first; idx < last; ++idx) {
                uint32_t full = reduced_to_full_map[idx];
                uint32_t i = full / (nyd*nzd);
                uint32_t rem = full % (nyd*nzd);
                xsub[idx-first] = xdof[i];
                ysub[idx-first] = ydof[rem / nzd];
                zsub[idx-first] = zdof[rem % nzd];
            }
        }

    public:

        RegularGridInterpolant3D(InterpolationRule rule, RangeTriplet xrange, RangeTriplet yrange, RangeTriplet zrange, int value_size, bool out_of_bounds_ok, std::function<std::vector<bool>(Vec, Vec, Vec)> skip) :
            rule(rule),
            xmin(std::get<0>(xrange)), xmax(std::get<1>(xrange)), nx(std::get<2>(xrange)),
            ymin(std::get<0>(yrange)), ymax(std::get<1>(yrange)), ny(std::get<2>(yrange)),
            zmin(std::get<0>(zrange)), zmax(std::get<1>(zrange)), nz(std::get<2>(zrange)),
            value_size(value_size), out_of_bounds_ok(out_of_bounds_ok)
        {
            int degree = rule.degree;
            if(degree + 1 > MAX_NODES)
                throw std::invalid_argument("Interpolation degree too large: at most degree 15 is supported");
            hx = (xmax-xmin)/nx;
            hy = (ymax-ymin)/ny;
            hz = (zmax-zmin)/nz;

            // build a regular mesh on [xmin, xmax] x [ymin, ymax] x [zmin, zmax]
            xmesh = linspace(xmin, xmax, nx+1, true);
            ymesh = linspace(ymin, ymax, ny+1, true);
            zmesh = linspace(zmin, zmax, nz+1, true);

            int nmesh = (nx+1)*(ny+1)*(nz+1);
            Vec xmeshtensor(nmesh, 0.);
            Vec ymeshtensor(nmesh, 0.);
            Vec zmeshtensor(nmesh, 0.);

            for (int i = 0; i <= nx; ++i) {
                for (int j = 0; j <= ny; ++j) {
                    for (int k = 0; k <= nz; ++k) {
                        int offset = idx_mesh(i, j, k);
                        xmeshtensor[offset] = xmesh[i];
                        ymeshtensor[offset] = ymesh[j];
                        zmeshtensor[offset] = zmesh[k];
                    }
                }
            }
            // for each node in the mesh, check whether it is in the domain, or
            // outside of it (i.e. should be skipped)
            std::vector<bool> skip_mesh = skip(xmeshtensor, ymeshtensor, zmeshtensor);
            // cells are entirely ignored if *all* of its eight corners are
            // outside the domain
            skip_cell = std::vector<bool>(nx*ny*nz, false);
            cells_to_skip = 0;
            for (int i = 0; i < nx; ++i) {
                for (int j = 0; j < ny; ++j) {
                    for (int k = 0; k < nz; ++k) {
                        bool skip_this_one = (
                                skip_mesh[idx_mesh(i  , j  , k)] && skip_mesh[idx_mesh(i  , j  , k+1)] &&
                                skip_mesh[idx_mesh(i  , j+1, k)] && skip_mesh[idx_mesh(i  , j+1, k+1)] &&
                                skip_mesh[idx_mesh(i+1, j  , k)] && skip_mesh[idx_mesh(i+1, j  , k+1)] &&
                                skip_mesh[idx_mesh(i+1, j+1, k)] && skip_mesh[idx_mesh(i+1, j+1, k+1)]
                                );
                        if(skip_this_one) {
                            skip_cell[idx_cell(i, j, k)] = true;
                            cells_to_skip++;
                        }
                    }
                }
            }
            cells_to_keep = nx*ny*nz - cells_to_skip;

            // now build the interpolation points in 1d.
            xdof = Vec(nx*degree+1, 0.);
            ydof = Vec(ny*degree+1, 0.);
            zdof = Vec(nz*degree+1, 0.);
            int i, j;
            for (i = 0; i < nx; ++i) {
                for (j = 0; j < degree+1; ++j) {
                    xdof[i*degree+j] = xmesh[i] + rule.nodes[j]*hx;
                }
            }
            for (i = 0; i < ny; ++i) {
                for (j = 0; j < degree+1; ++j) {
                    ydof[i*degree+j] = ymesh[i] + rule.nodes[j]*hy;
                }
            }
            for (i = 0; i < nz; ++i) {
                for (j = 0; j < degree+1; ++j) {
                    zdof[i*degree+j] = zmesh[i] + rule.nodes[j]*hz;
                }
            }
            uint32_t n =  (nx*degree+1)*(ny*degree+1)*(nz*degree+1);
            // We need to figure out which of these dofs we keep, and which
            // to discard.  To do this, we loop over the cells, and for each
            // cell that shouldn't be skipped, we mark all dofs in that cell.

            std::vector<bool> skip_dof(n, true);
            for (int i = 0; i < nx; ++i) {
                for (int j = 0; j < ny; ++j) {
                    for (int k = 0; k < nz; ++k) {
                        if(!skip_cell[idx_cell(i, j, k)]){
                            for (int ii = 0; ii <= degree; ++ii) {
                                for (int jj = 0; jj <= degree; ++jj) {
                                    for (int kk = 0; kk <= degree; ++kk) {
                                        skip_dof[idx_dof(i*degree + ii, j*degree + jj, k*degree + kk)] = false;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            // Count how many dofs are skipped in total, and how many to keep
            dofs_to_skip = 0;
            for (uint32_t i = 0; i < n; ++i) {
                dofs_to_skip += skip_dof[i];
            }
            dofs_to_keep = n - dofs_to_skip;
            // Build a map that maps indices from the reduced set of interpolation
            // points to the full set, and its inverse
            reduced_to_full_map = std::vector<uint32_t>(dofs_to_keep, 0);
            full_to_reduced_map = std::vector<uint32_t>(n, 0);
            uint32_t ctr = 0;
            for (uint32_t i = 0; i < n; ++i) {
                full_to_reduced_map[i] = i - ctr;
                if(skip_dof[i])
                    ctr++;
                else {
                    reduced_to_full_map[i-ctr] = i;
                }
            }

            // each cell's block holds (degree+1)^3 dofs with value_size
            // values per dof; the scalar kernels need no padding or alignment
            local_vals_size = (degree+1)*(degree+1)*(degree+1)*value_size;
            cell_offsets.assign((size_t)nx*ny*nz, -1);
        }
        RegularGridInterpolant3D(InterpolationRule rule, RangeTriplet xrange, RangeTriplet yrange, RangeTriplet zrange, int value_size, bool out_of_bounds_ok) :
            RegularGridInterpolant3D(rule, xrange, yrange, zrange, value_size, out_of_bounds_ok, [](Vec x, Vec y, Vec z){ return std::vector<bool>(x.size(), false); })
            {}

        void interpolate_batch(std::function<Vec(Vec, Vec, Vec)> &f); // build the interpolant
#ifdef USE_MPI
        void interpolate_batch(std::function<Vec(Vec, Vec, Vec)> &f, MPI_Comm comm); // build the interpolant with MPI parallelization (raw MPI, compatible with mpi4py)
#endif

        Vec evaluate(double x, double y, double z); // evaluate the interpolant at one location
        void evaluate_batch(Array& xyz, Array& fxyz); // evluate the interpolant at multiple locations

        std::pair<double, double> estimate_error(std::function<Vec(Vec, Vec, Vec)> &f, int samples);

        // Serialization for InterpolatedBoozerField save/load.
        // get_interpolant_data(): exports vals array and grid params for JSON.
        // set_interpolant_data(): restores vals and rebuilds the per-cell
        //                         storage (local_vals_flat/cell_offsets) used
        //                         for evaluation.
        std::map<std::string, std::vector<double>> get_interpolant_data() const;
        void set_interpolant_data(const std::map<std::string, std::vector<double>>& data);
};

class UniformInterpolationRule : public InterpolationRule {
    protected:
        using InterpolationRule::build_scalings;
    public:
        using InterpolationRule::nodes;
        using InterpolationRule::scalings;
        using InterpolationRule::degree;
        UniformInterpolationRule(int degree) : InterpolationRule(degree) {
            double degreeinv = double(1.)/degree;
            for (int i = 0; i < degree+1; ++i) {
                nodes[i] = i*degreeinv;
            }
            build_scalings();
        }
};


class ChebyshevInterpolationRule : public InterpolationRule {
    protected:
        using InterpolationRule::build_scalings;
    public:
        using InterpolationRule::nodes;
        using InterpolationRule::scalings;
        using InterpolationRule::degree;
        ChebyshevInterpolationRule(int degree) : InterpolationRule(degree) {
            double degreeinv = double(1.)/degree;
            for (int i = 0; i < degree+1; ++i) {
                nodes[i] = (-0.5)*std::cos(i*M_PI*degreeinv) + 0.5;
            }
            build_scalings();
        }
};
