/*! \file
    \brief Routines that assemble the tentative prolongator from local ones.

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
#ifndef _CONTRIB_HPP
#define _CONTRIB_HPP

#include "common.hpp"
#include <mfem.hpp>
#include "aggregates.hpp"

#include "SharedEntityCommunication.hpp"

using namespace mfem;

/* Types */
/*! \brief The data used for building the tentative interpolant.
*/
typedef struct {
    int rows; /*!< The rows of the interpolant matrix. */
    int filled_cols; /*!< How many columns are currently introduced in the
                          interpolant matrix. */
    SparseMatrix *tent_interp; /*!< The partially built matrix of the tentative
                                    interpolant. */
    Array<double> * local_coarse_one_representation; /*! ATB building coarse_one_representation on the fly (we are going to just copy this pointer to interp_data) */
    // int coarse_ones_values_per_agg; /*! ATB number of values per agg to use in construction of coarse_one_representation (larger means larger coarse space on coarsest of three levels) */
    int coarse_truedof_offset;
    int * mis_numcoarsedof;

    DenseMatrix ** mis_tent_interps;
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
                          const agg_partitioning_relations_t& agg_part_rels,
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

void contrib_filter_boundary(contrib_tent_struct_t *tent_int_struct,
                             const agg_partitioning_relations_t& agg_part_rels,
                             DenseMatrix& local, 
                             const int *restriction);

void contrib_tent_insert_simple(contrib_tent_struct_t *tent_int_struct,
                                const agg_partitioning_relations_t& agg_part_rels,
                                DenseMatrix& local, 
                                const int *restriction);

/*! \brief Inserts (embeds) the local tentative interpolant in the global one.

    \param tent_int_struct (IN) The structure returned by a call to
                                \b contrib_tent_init.
    \param agg_part_rels (IN) The partitioning relations.
    \param local (IN/OUT) The local tentative interpolant, this may also be modified by boundary conditions (new ATB 11 May 2015)
    \param restriction (IN) The array that for each DoF of the aggregate
                            maps its global number (as returned by
                            \b agg_restrict_to_agg).

    \warning \a tent_int_struct must be already initiated by a call to
             \b contrib_tent_init. Actually it must be the returned value from
             the call.
*/
void contrib_tent_insert_from_local(contrib_tent_struct_t *tent_int_struct,
                            const agg_partitioning_relations_t& agg_part_rels,
                            DenseMatrix& local, const int *restriction);

/*!
  like contrib_mises, assume no eigenvalue problem, just use constant
  vector
*/
void contrib_ones(contrib_tent_struct_t *tent_int_struct,
                  const agg_partitioning_relations_t& agg_part_rels);

/*! \brief Visits all MISes and fills in the tentative interpolant.

  this was originally copied from serial SAAMGE's contrib_ref_aggs

  \param tent_int_struct (IN) The structure returned by a call to
         \b contrib_tent_init.
  \param agg_part_rels (IN) The partitioning relations.
  \param cut_evects_arr (IN) The vectors from all local eigenvalue problems.
  \param eps (IN) Tolerance for the SVD.
*/
void contrib_mises(contrib_tent_struct_t *tent_int_struct,
                   const agg_partitioning_relations_t& agg_part_rels,
                   DenseMatrix * const *cut_evects_arr,
                   double eps, bool scaling_P);

#endif // _CONTRIB_HPP
