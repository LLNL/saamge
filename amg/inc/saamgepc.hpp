/*
    SAAMGE: smoothed aggregation element based algebraic multigrid hierarchies
            and solvers.

    Copyright (c) 2018, Lawrence Livermore National Security,
    LLC. Developed under the auspices of the U.S. Department of Energy by
    Lawrence Livermore National Laboratory under Contract
    No. DE-AC52-07NA27344. Written by Delyan Kalchev, Andrew T. Barker,
    and Panayot S. Vassilevski. Released under LLNL-CODE-667453.

    This file is part of SAAMGE. 

    Please also read the full notice of copyright and license in the file
    LICENSE.

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License (as
    published by the Free Software Foundation) version 2.1 dated February
    1999.

    This program is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the IMPLIED WARRANTY OF
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the terms and
    conditions of the GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with this program; if not, see
    <http://www.gnu.org/licenses/>.
*/

// Wrapper for the mesh based saamge
// Patrick V. Zulian
// zulian1@llnl.gov
// 10 August 2016


#ifndef SAAMGE_SAAMGE_PC_HPP
#define SAAMGE_SAAMGE_PC_HPP 

#include <fstream>
#include <iostream>
#include <memory>
#include <assert.h>

#include <mpi.h>
#include <mfem.hpp>
#include <saamge.hpp>

namespace saamge
{

/// @todo this and SpectralAMGSolver are basically the same thing
class SAAMGePC : public mfem::Solver
{
public:
    SAAMGePC(const std::shared_ptr<mfem::ParFiniteElementSpace> &fe, 
             mfem::Array<int> &ess_bdr);

    SAAMGePC();
    ~SAAMGePC();

    void InitDefaults();
    void Print(std::ostream &os = std::cout) const;

    /// shouldn't this be part of the constructor?
    bool Make(const std::shared_ptr<mfem::ParBilinearForm> &a,
              const std::shared_ptr<mfem::HypreParMatrix>  &A,
              mfem::SparseMatrix &Al);

    void Destroy();

    void Mult(const mfem::Vector &x, mfem::Vector &y) const override;
    void MultTranspose(const mfem::Vector &x, mfem::Vector &y) const override;

    inline void SetOperator(const Operator &op) override {
        // Can be implemented once Al and a are not needed anymore
        // HOW: Pass the fe space and ess_bdr to the constructor and the matrix
        // (not the bilinear form) to SetOperator and call Make(fe, op, ess_bdr);
        assert(false && "not be implemented yet");
    }

private:
    std::shared_ptr<mfem::ParFiniteElementSpace> fe;
    std::vector<int> nparts_arr;
    mfem::Array<int> ess_bdr;

    int num_levels;
    int elems_per_agg;
    int first_elems_per_agg;
    bool do_aggregates;
    int first_nu_pro;
    int nu_pro;
    int nu_relax;
    int spe10_scale;
    double theta;
    double first_theta;
    int polynomial_coarse;
    bool correct_nulspace;
    bool direct_eigensolver;
    bool has_stuff_to_destroy;

    ml_data_t *ml_data;
    ElementMatrixProvider * emp;
    std::shared_ptr<VCycleSolver> Bprec;
    std::shared_ptr<mfem::HypreParMatrix> A;
    agg_partitioning_relations_t *agg_part_rels;
};

}

#endif //SAAMGE_SAAMGE_PC_HPP
