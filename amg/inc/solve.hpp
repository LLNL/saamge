/*! \file
    \brief Linear solvers that can be used as coarse level solvers.

    SAAMGE: smoothed aggregation element based algebraic multigrid hierarchies
            and solvers.

    Copyright (c) 2016, Lawrence Livermore National Security,
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
#include <mfem.hpp>

using namespace mfem;

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
    CorrectNullspace(HypreParMatrix& A,
                     HypreParMatrix * interp,
                     int smoother_steps,
                     bool smooth_phat,
                     bool v_cycle,
                     bool iterative_mode);
    ~CorrectNullspace();
    virtual void SetOperator(const Operator &op);
    virtual void Mult(const Vector &x, Vector &y) const;
private:
    int smoother_steps; // two purposes: if use_hypre_smoother, gives number of GS steps,
                        // otherwise, gives nu (which is more like degree than steps) for Delyan's Chebyshev smoother
    bool v_cycle; // this is for the nulspace (coarsest) level, whether to do CG-HYPRE or just HYPRE
    bool use_hypre_smoother;
    bool smooth_phat;

    smpr_ft smoother;
    void * smoother_data;

    HypreParMatrix * A;
    HypreParMatrix * Ac; // this is matrix at nulspace level
    HypreParMatrix * interp;
    HypreParMatrix * restr;

    Solver * correct_nullspace_coarse;

    int spectral_cycles; // only makes sense without --spectral-cg option
    int cumulative_spectral_cg; // only makes sense with --spectral-cg option
    double spectral_cg_tol; // for use with --spectral-cg option
};

/**
   This solves with Hypre CG preconditioned with Hypre BoomerAMG

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
    AMGSolver(HypreParMatrix& A, bool iterative_mode);
    ~AMGSolver();
    virtual void SetOperator(const Operator &op) {};
    virtual void Mult(const Vector &x, Vector &y) const;
private:
    HypreSolver *amg; /*!< AMG preconditioner. */
    HyprePCG *pcg; /*!< PCG solver. */
    mutable int cumulative_iterations; 
    int ref_cntr; /*!< Reference counter. */
};

/**
   this class intended to replace solve_spd_Vcycle() as a solve_t
*/
class VCycleSolver : public mfem::Solver
{
public:
    VCycleSolver(tg_data_t * tg_data, bool iterative_mode);
    ~VCycleSolver();
    virtual void SetOperator(const Operator &op);
    virtual void Mult(const Vector &x, Vector &y) const;
private:
    tg_data_t * tg_data;
    HypreParMatrix * A;
};


/* Options */

/*! \brief The configuration class of this module.
*/
CONFIG_BEGIN_CLASS_DECLARATION(SOLVE)

    /*! The absolute tolerance for the solve procedures. */
    CONFIG_DECLARE_OPTION(double, atol);

    /*! The relative tolerance for the solve procedures. */
    CONFIG_DECLARE_OPTION(double, rtol);

    /*! The coefficient for the maximal number of iterations. This coefficient
        is multiplied to the size of the matrix to determine the desired number
        of iterations. */
    CONFIG_DECLARE_OPTION(double, iters_coeff);

CONFIG_END_CLASS_DECLARATION(SOLVE)

CONFIG_BEGIN_INLINE_CLASS_DEFAULTS(SOLVE)
    CONFIG_DEFINE_OPTION_DEFAULT(atol, 0.),
    CONFIG_DEFINE_OPTION_DEFAULT(rtol, 1e-12),
    CONFIG_DEFINE_OPTION_DEFAULT(iters_coeff, 10.)
CONFIG_END_CLASS_DEFAULTS

/* Functions */
/*! \brief Empty coarse solver.

    It only make \a x equal to zero. That is, it disables coarsest correction.

    \param A (IN) The sparse matrix of the linear system to be solved.
    \param b (IN) The right-hand side.
    \param x (OUT) The approximated solution.
    \param data (IN/OUT) Not used.
*/
void solve_empty(HypreParMatrix& A, HypreParVector& b, HypreParVector& x,
                 void *data);

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
void solve_spd_Vcycle(HypreParMatrix& A, const HypreParVector& b, HypreParVector& x,
                      void *data);
void solve_spd_Wcycle(HypreParMatrix& A, const HypreParVector& b, HypreParVector& x,
                      void *data);


#endif // _SOLVE_HPP
