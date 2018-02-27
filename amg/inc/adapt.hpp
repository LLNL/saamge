/*! \file
    \brief Adaptation related functionality.

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
#ifndef _ADAPT_HPP
#define _ADAPT_HPP

#include "common.hpp"
#include <mfem.hpp>
#include "tg.hpp"

/* Defines */
/*! \def ADAPT_XBAD_MAX_ITER_FLAG
    \brief A flag denoting that maximal number of iterations is reached.
*/
/*! \def ADAPT_XBAD_ERR_TOL_FLAG
    \brief A flag denoting that the desired error tolerance is reached.
*/
/*! \def ADAPT_XBAD_ERR_INC_FLAG
    \brief A flag indicating the presence of divergence (the error increased).
*/
/*! \def ADAPT_XBAD_ALL_FLAGS
    \brief All flags set.
*/
#define ADAPT_XBAD_MAX_ITER_FLAG    0x01
#define ADAPT_XBAD_ERR_TOL_FLAG     0x02
#define ADAPT_XBAD_ERR_INC_FLAG     0x80

#define ADAPT_XBAD_ALL_FLAGS        (ADAPT_XBAD_MAX_ITER_FLAG | \
                                     ADAPT_XBAD_ERR_TOL_FLAG | \
                                     ADAPT_XBAD_ERR_INC_FLAG)

namespace saamge
{

/* Functions */
/*! \brief Approximates xbad.

    Here the terms xbad and \b err (error) are used interchangeably.

    Using the current TG method (hierarchy) solves
    \f$ A\mathbf{x} = \mathbf{0} \f$.

    Uses the hierarchy in \a tg_data. It stops when the maximal number of
    iterations (\a maxiter) is reach during this execution of the function, or
    the A-norm of the error becomes smaller than \a atol or \a rtol
    times the A-norm of the initial error.

    A current call of the function might be a continuation of previous call(s).
    Then, \a *executed_iters must be the total number of TG iterations done for
    approximation xbad before the current call. Also, \a xbad should hold
    the currently (before this call) approximation of xbad, and \a *err0 -- the
    A-norm of the very first (initial) approximation of xbad from which
    everything started.

    Otherwise (a first call is being executed), \a *executed_iters must be 0,
    and the initial xbad will be initialized in this function. This
    initial xbad always gets normalized.

    \param A (IN) The matrix of the system being solved (usually the global
                  stiffness matrix).
    \param agg_part_rels (IN) The partitioning relations.
    \param maxiter (IN) The maximal number of iteration to be done during this
                        call of the function. Should be at least one;
    \param tg_data (IN) Data for the TG algorithm.
    \param xbad (IN/OUT) Outputs a xbad approximation. As input it is the
                         current xbad approximation (see the description).
    \param cf (OUT) Here the (asymptotic) convergence factor from the last
                    iteration is returned.
    \param acf (OUT) Here the average convergence factor is returned (including
                     all the iterations from previous calls when the current
                     call is a continuation).
    \param err (OUT) Here the A-norm of the last approximation of xbad is
                     returned.
    \param err0 (IN/OUT) Here the A-norm of the initial approximation of xbad
                         is returned (the very first, including the previous
                         call(s)). As input it is used when the current call is
                         a continuation (see the description).
    \param executed_iters (IN/OUT) Here the number of iterations executed
                                   during the current call is returned.
                                   See the description about how it is used as
                                   input.
    \param rtol (IN) Relative tolerance (see the description).
    \param atol (IN) Absolute tolerance (see the description).
    \param normalize (IN) Instructs if to normalize the xbad approximation
                          after each step. If this is
                          \em true, the average convergence factor (output)
                          will not make sense.
    \param output (IN) Whether to generate any console output. This way output
                       can be suppressed independent of the global output level.

    \returns The reason the function stopped (see \b ADAPT_XBAD_<*> flags).
*/
int adapt_approx_xbad(
    mfem::HypreParMatrix& A,
    const agg_partitioning_relations_t& agg_part_rels, int maxiter,
    tg_data_t *tg_data, mfem::HypreParVector& xbad, double *cf, double *acf,
    double *err, double *err0, int *executed_iters, double rtol/*=10e-12*/, double atol/*=10e-24*/,
    bool normalize/*=1*/, bool output=true);

}

#endif // _ADAPT_HPP
