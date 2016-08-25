/*! \file
    \brief Linear solvers that can be used as coarse level solvers.

    SAAMGE: smoothed aggregation element based algebraic multigrid hierarchies
            and solvers.

    Copyright (c) 2015, Lawrence Livermore National Security,
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
#include <mfem.hpp>

/* Types */
/*! \brief A function type for a (coarse) solver execution.

    It is called whenever systems with the matrix on the coarsest level have to
    be solved.

    \param A (IN) The sparse matrix of the linear system to be solved.
    \param b (IN) The right-hand side.
    \param x (IN/OUT) The approximated solution. It might be used as an input,
                      to the solver, of the initial guess (depends on the
                      solver).
    \param data (IN/OUT) Solver specific data (see \b solve_t).
*/
typedef void (*solve_ft)(HypreParMatrix& A, HypreParVector& b,
                         HypreParVector& x, void *data);

/*! \brief A function type for a (coarse) solver initialization.

    It is called whenever the matrix on the coarsest level is computed
    (created).

    \param A (IN) The sparse matrix of the linear system (being solved with
                  this solver). That is, it is the operator on the coarsest
                  level.

    \returns Solver specific data (see \b solve_t).

    \warning The returned data would probably need to be deallocated.
*/
typedef void *(*solve_init_ft)(HypreParMatrix& A);

/*! \brief A function type for a (coarse) solver destruction.

    It is called whenever the matrix on the coarsest level is being freed
    (destroyed).

    \param A (IN) The sparse matrix of the linear system (being solved with
                  this solver). That is, it is the operator on the coarsest
                  level.
    \param data (IN/OUT) Solver specific data (see \b solve_t).

    \returns A pointer that will replace the current \b data (see \b solve_t).
             It should normally be NULL.
*/
typedef void *(*solve_free_ft)(HypreParMatrix& A, void *data);

/*! \brief A function type for a (coarse) solver data copying.

    It is called whenever the matrix on the coarsest level is being copied.

    \param A (IN) The sparse matrix of the linear system (being solved with
                  this solver). That is, it is the operator on the coarsest
                  level.
    \param data (IN/OUT) Solver specific data (see \b solve_t).

    \returns A copy of \a data.

    \warning The returned data would probably need to be deallocated.
*/
typedef void *(*solve_copy_ft)(HypreParMatrix& A, void *data);

/*! \brief A (coarse) solver structure combining functions and data together.

    A typical use of this structure is in \b tg_data_t in 'tg.*'.

    XXX: It may look a bit like a class (in OOP). However, while a class (its
         C++ implementation) "encapsulates" data in a structure that is passed
         implicitly to the class's methods, here the functions are also in the
         structure's data. So it looks more like defining interface. This is a
         spot where inheritance with its polymorphic capabilities seems to be
         an appropriate and nice alternative.
*/
typedef struct {
    solve_init_ft solve_init; /*!< A solver initialization routine. */
    solve_free_ft solve_free; /*!< A solver destructor routine. */
    solve_copy_ft solve_copy; /*!< A solver copy routine. */
    solve_ft solver; /*!< A solver. */
    void *data; /*!< Solver's data. Not owned by this structure and it is not
                     freed and/or allocated here. */
} solve_t;

/*! \brief Data for the AMG solver.
*/
typedef struct {
    HypreSolver *amg; /*!< AMG preconditioner. */
    HyprePCG *pcg; /*!< PCG solver. */
    int ref_cntr; /*!< Reference counter. */
} solve_amg_t;

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

/*! \brief AMG solver initialization.

    \param A (IN) The sparse matrix of the linear system (being solved with
                  this solver). That is, it is the operator on the coarsest
                  level.

    \returns Solver specific data of type \b solve_amg_t.

    \warning The returned data will be deallocated by \b solve_spd_AMG_free.
    \warning \a A must be an s.p.d. matrix.
    \warning The tolerances and maximal number of iterations are determined
             using options \b rtol and \b iters_coeff.
*/
void *solve_spd_AMG_init(HypreParMatrix& A);

/*! \brief AMG solver destruction.

    \param A (IN) The sparse matrix of the linear system (being solved with
                  this solver). That is, it is the operator on the coarsest
                  level.
    \param data (IN/OUT) Of type \b solve_amg_t.

    \returns NULL.
*/
void *solve_spd_AMG_free(HypreParMatrix& A, void *data);

/*! \brief AMG solver data copying.

    \param A (IN) The sparse matrix of the linear system (being solved with
                  this solver). That is, it is the operator on the coarsest
                  level.
    \param data (IN/OUT) Of type \b solve_amg_t.

    \returns A copy of \a data.

    \warning The returned data will be deallocated by \b solve_spd_AMG_free.
*/
void *solve_spd_AMG_copy(HypreParMatrix& A, void *data);

/*! \brief Solves a system using AMG.

    \param A (IN) The sparse matrix of the linear system to be solved.
    \param b (IN) The right-hand side.
    \param x (OUT) The approximated solution.
    \param data (IN) Of type \b solve_amg_t.
*/
void solve_spd_AMG(HypreParMatrix& A, HypreParVector& b, HypreParVector& x,
                   void *data);

#endif // _SOLVE_HPP
