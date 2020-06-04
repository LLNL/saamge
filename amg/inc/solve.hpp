/*! \file
    \brief Linear solvers that can be used as coarse level solvers.

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

#pragma once
#ifndef _SOLVE_HPP
#define _SOLVE_HPP

#include "common.hpp"
#include "smpr.hpp"
#include "tg_data.hpp"
#include "ml.hpp"
#include <mfem.hpp>

namespace saamge
{

/* Types */

/**
   This class is really just a V-cycle that we only use with
   a particularly chosen interp matrix, so we believe it is
   somehow a different class...

   some of the options are basicaly deprecated/inactive:
     use_hypre_smoother
     spectral_cycles (always == 1)
*/
class CorrectNullspace : public mfem::Solver
{
public:
    CorrectNullspace(mfem::HypreParMatrix& A,
                     mfem::HypreParMatrix * interp,
                     int smoother_steps,
                     bool smooth_phat,
                     bool v_cycle,
                     bool iterative_mode);
    ~CorrectNullspace();
    virtual void SetOperator(const mfem::Operator &op);
    virtual void Mult(const mfem::Vector &x, mfem::Vector &y) const;
private:
    int smoother_steps; // two purposes: if use_hypre_smoother, gives number of GS steps,
                        // otherwise, gives nu (which is more like degree than steps) for Delyan's Chebyshev smoother
    bool v_cycle; // this is for the nulspace (coarsest) level, whether to do CG-HYPRE or just HYPRE
    bool use_hypre_smoother;
    bool smooth_phat;

    smpr_ft smoother;
    void * smoother_data;

    mfem::HypreParMatrix * A;
    mfem::HypreParMatrix * Ac; // this is matrix at nulspace level
    mfem::HypreParMatrix * interp;
    mfem::HypreParMatrix * restr;

    mfem::Solver * correct_nullspace_coarse;

    int spectral_cycles; // only makes sense without --spectral-cg option
    int cumulative_spectral_cg; // only makes sense with --spectral-cg option
    double spectral_cg_tol; // for use with --spectral-cg option
};

/**
   @brief Solves with Hypre CG preconditioned with Hypre BoomerAMG

   That is, no spectral method at all, for use in comparisons etc.

   In many places I am moving away from HyprePCG, towards
   mfem::CGSolver because the latter allows more flexible 
   preconditioning. However, this version is still normally
   used as a solver on the coarsest level.

   This class intended to replace solve_spd_amg(),
   solve_spd_amg_init(), etc. as a solve_t
*/
class AMGSolver : public mfem::Solver
{
public:
    AMGSolver(mfem::HypreParMatrix& A, bool iterative_mode, double rel_tol=1e-12, int iters_coeff=10);
    ~AMGSolver();
    virtual void SetOperator(const mfem::Operator &op) {};
    virtual void Mult(const mfem::Vector &x, mfem::Vector &y) const;
    void SetRelTol(double rtol) { rel_tol = rtol; }
private:
    mfem::HypreSolver *amg; /*!< AMG preconditioner. */
    mfem::HyprePCG *pcg; /*!< PCG solver. */
    double rel_tol;
    mutable int cumulative_iterations; 
    int ref_cntr; /*!< Reference counter. @todo remove */
    int iters_coeff; /*!< multiply by matrix size for max iterations */
};

/**
   @brief Does a V-cycle with the given tg_data struct

   Basically a wrapper for the C-struct as we move to being a bit
   more C++-like.

   Replaces solve_spd_Vcycle()
*/
class VCycleSolver : public mfem::Solver
{
public:
    VCycleSolver(tg_data_t * tg_data, bool iterative_mode);
    ~VCycleSolver();
    virtual void SetOperator(const mfem::Operator &op);
    virtual void Mult(const mfem::Vector &x, mfem::Vector &y) const;
private:
    tg_data_t * tg_data;
    mfem::HypreParMatrix * A;
};

/**
   @brief Adds the effect of two solvers.
*/
class AdditiveSolver : public mfem::Solver
{
public:
    AdditiveSolver(mfem::Solver& solver1, mfem::Solver& solver2) :
        solver1(solver1), solver2(solver2) { }
    ~AdditiveSolver() { }
    virtual void SetOperator(const mfem::Operator &op) { }
    virtual void Mult(const mfem::Vector &b, mfem::Vector &x) const
    {
        SA_ASSERT(b.Size() == x.Size());
        mfem::Vector t(b.Size());
        solver1.Mult(b, x);
        solver2.Mult(b, t);
        x += t;
    }
private:
    mfem::Solver &solver1;
    mfem::Solver &solver2;
};

/**
   @brief Encapsulates and wraps the spectral smoothed aggregation
   spectral element AMG solver in a user-friendly way.

   @param polynomial_coarse is here to allow user to specify that we
          should add rigid body modes for elasticity, in which case you
          set it to 1, otherwise -1 is probably best.
*/
class SpectralAMGSolver : public mfem::Solver
{
public:
    /**
       ess_bdr is the first argument to fes.GetEssentialVDofs():
         fes.GetEssentialVDofs(*ess_bdr, ess_dofs);
       ie, ess_bdr has length (number of attributes) and is nonzero
       for boundary attributes that are essential
    */
    SpectralAMGSolver(mfem::HypreParMatrix& Ag,
                      mfem::ParBilinearForm& aform,
                      mfem::SparseMatrix& Alocal,
                      mfem::Array<int>& ess_bdr,
                      int elems_per_agg, int num_levels,
                      int nu_pro, int nu_relax, double theta,
                      int polynomial_coarse, bool coarse_direct);
    ~SpectralAMGSolver();

    /// Solver interface
    void SetOperator(const mfem::Operator &op);

    /// Operator interface
    void Mult(const mfem::Vector &x, mfem::Vector &y) const;

private:
    mfem::HypreParMatrix& Ag_;

    int* nparts_arr_;
    agg_partitioning_relations_t* agg_part_rels_;
    ml_data_t* ml_data_;

    VCycleSolver* v_cycle_;
};

/* Functions */
/*! \brief Empty coarse solver.

    It does nothing.

    \param A (IN) The sparse matrix of the linear system to be solved.
    \param b (IN) The right-hand side.
    \param x (OUT) The approximated solution.
    \param data (IN/OUT) Not used.
*/
void solve_empty(
mfem::HypreParMatrix& A, const mfem::Vector& b,
    mfem::Vector& x, void *data);

/*! \brief Inexact coarse solve by a TG(or V)-cycle.

    \param A (IN) The sparse matrix of the coarse linear system.
    \param b (IN) The right-hand side (some residual).
    \param x (OUT) The approximated solution.
    \param data (IN) Must be of type \b tg_data_t. This is the TG data for the
                     next (coarser) level where \a A lives.

    \warning \a A must be an s.p.d. matrix.
    \warning The coarse matrix (\b Ac) in \a data must be already computed.
             This function does not produce the coarse operator.
*/
void solve_spd_Vcycle(
    mfem::HypreParMatrix& A, const mfem::HypreParVector& b,
    mfem::HypreParVector& x, void *data);
void solve_spd_Wcycle(
    mfem::HypreParMatrix& A, const mfem::HypreParVector& b,
    mfem::HypreParVector& x, void *data);

} // namespace saamge

#endif // _SOLVE_HPP
