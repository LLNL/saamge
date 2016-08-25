/*! \file
    \brief Routines that assemble the tentative prolongator from local ones.

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
#ifndef _CONTRIB_HPP
#define _CONTRIB_HPP

#include "common.hpp"
#include <mfem.hpp>
#include "aggregates.hpp"

/* Types */
/*! \brief The data used for building the tentative interpolant.
*/
typedef struct {
    int rows; /*!< The rows of the interpolant matrix. */
    int filled_cols; /*!< How many columns are currently introduced in the
                          interpolant matrix. */
    SparseMatrix *tent_interp; /*!< The partially built matrix of the tentative
                                    interpolant. */
} contrib_tent_struct_t;

/*! \brief Functions inputing aggregates contributions to tentative prolongator.

    See \em contrib.hpp.

    \param tent_int_struct (IN) The structure returned by a call to
                                \b contrib_tent_init.
    \param agg_part_rels (IN) The partitioning relations.
    \param cut_evects_arr (IN) The vectors from all local eigenvalue problems.
    \param eps (IN) Tolerance for the SVD and probably also for something else.
*/
typedef void (*contrib_agg_ft)(contrib_tent_struct_t *tent_int_struct,
                          const agg_partititoning_relations_t& agg_part_rels,
                          DenseMatrix * const *cut_evects_arr, double eps);

/* Options */

/*! \brief The configuration class of this module.
*/
CONFIG_BEGIN_CLASS_DECLARATION(CONTRIB)

    /*! If \em true will enforce the tentative coarse basis functions to be
        zero on the part of the boundary with essential boundary conditions. */
    CONFIG_DECLARE_OPTION(bool, avoid_ess_brd_dofs);

CONFIG_END_CLASS_DECLARATION(CONTRIB)

CONFIG_BEGIN_INLINE_CLASS_DEFAULTS(CONTRIB)
    CONFIG_DEFINE_OPTION_DEFAULT(avoid_ess_brd_dofs, true)
CONFIG_END_CLASS_DEFAULTS

/* Functions */
/*! \brief Initiates the process of building the tentative interpolant.

    \param ND (IN) The number of fine DoFs.

    \returns The structure needed in \b contrib_tent_insert_from_local and
             freed/finalized by \b contrib_tent_finalize

    \warning The returned structure must be finalized and freed by the caller
             using contrib_tent_finalize
*/
contrib_tent_struct_t *contrib_tent_init(int ND);

/*! \brief Produces the final tentative interpolant.

    This is tha last phase of the tentative interpolant construction. It's
    called after \b contrib_tent_init and all \b contrib_tent_insert_from_local.
    \a tent_int_struct gets freed.

    \param tent_int_struct (IN) The structure returned by a call to
                                \b contrib_tent_init and already with all local
                                tentative interpolants embedded by
                                \b contrib_tent_insert_from_local.

    \returns The local tentative interpolant as a finalized sparse matrix.

    \warning After calling this function no more calls of
             \b contrib_tent_insert_from_local can be made with
             \a tent_int_struct as a parameter.
*/
SparseMatrix *contrib_tent_finalize(contrib_tent_struct_t *tent_int_struct);

/*! \brief Inserts (embeds) the local tentative interpolant in the global one.

    \param tent_int_struct (IN) The structure returned by a call to
                                \b contrib_tent_init.
    \param agg_part_rels (IN) The partitioning relations.
    \param local (IN) The local tentative interpolant.
    \param restriction (IN) The array that for each DoF of the aggregate
                            maps its global number (as returned by
                            \b agg_restrict_to_agg).

    \warning \a tent_int_struct must be already initiated by a call to
             \b contrib_tent_init. Actually it must be the returned value from
             the call.
*/
void contrib_tent_insert_from_local(contrib_tent_struct_t *tent_int_struct,
                            const agg_partititoning_relations_t& agg_part_rels,
                            const DenseMatrix& local, const int *restriction);


/*! \brief Loops over all big aggregates and fills in the tentative interpolant.

    SVD is performed every time vectors are restricted to aggregates.

    \param tent_int_struct (IN) The structure returned by a call to
                                \b contrib_tent_init.
    \param agg_part_rels (IN) The partitioning relations.
    \param cut_evects_arr (IN) The vectors from all local eigenvalue problems.
    \param eps (IN) Tolerance for the SVD.
*/
void contrib_big_aggs_svd(contrib_tent_struct_t *tent_int_struct,
                          const agg_partititoning_relations_t& agg_part_rels,
                          DenseMatrix * const *cut_evects_arr, double eps);

/*! \brief Loops over all big aggregates and fills in the tentative interpolant.

    SVD is NOT performed.

    \param tent_int_struct (IN) The structure returned by a call to
                                \b contrib_tent_init.
    \param agg_part_rels (IN) The partitioning relations.
    \param cut_evects_arr (IN) The vectors from all local eigenvalue problems.
    \param eps (IN) Not used.
*/
void contrib_big_aggs_nosvd(contrib_tent_struct_t *tent_int_struct,
                            const agg_partititoning_relations_t& agg_part_rels,
                            DenseMatrix * const *cut_evects_arr, double eps);

#endif // _CONTRIB_HPP
