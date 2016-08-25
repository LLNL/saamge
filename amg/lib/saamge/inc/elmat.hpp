/*! \file
    \brief Routines computing element matrices.

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

    XXX Notes:

    If OpenMP (or similar) would be used, functions of type
    \b agg_elmat_callback_ft MUST be thread-safe.
*/

#pragma once
#ifndef _ELMAT_HPP
#define _ELMAT_HPP

#include "common.hpp"
#include <mfem.hpp>
#include "aggregates.hpp"

/* Types */
/*! \brief The type of data an element matrix callback uses.
*/
typedef enum {
    ELMAT_DATA_TYPE_ARRAY, /*!< The data is an array of matrices. */
    ELMAT_DATA_TYPE_BILINEAR_FORM, /*!< The data is a \em BilinearForm. */
    ELMAT_DATA_TYPE_OTHER, /*!< Some other type. */

    ELMAT_DATA_TYPE_MAX /*!< Sentinel. */
} elmat_data_t;

/* Functions */
/*! \brief Crates an array of element matrices from a callback.

    \param elmat_callback (IN) A callback that returns element matrices.
    \param agg_part_rels (IN) The partitioning relations.
    \param data (IN/OUT) \a elmat_callback specific data.

    \returns An array of pointers to all element matrices.

    \warning The returned array of pointers to matrices must be freed by the
             caller using, e.g., \b mbox_free_matr_arr.
    \warning It must be compiled with Run-Time Type Information (RTTI) enabled
             for \em dynamic_cast to work.
*/
Matrix **elmat_create_array(agg_elmat_callback_ft elmat_callback,
                const agg_partititoning_relations_t *agg_part_rels, void *data);

/*! \brief A function returning element matrices from an array.

    Simply a wrapper that allows to pass an array of already built element
    matrices.

    \param elno (IN) The element number.
    \param agg_part_rels (IN) The partitioning relations (Not used, can be
                              NULL).
    \param data (IN) An array of \b Matrix pointers (which are either
                     \b SparseMatrix pointers or \b DenseMatrix pointers).
                     Type: \b ELMAT_DATA_TYPE_ARRAY.
    \param free_matr (OUT) Indicates whether the returned matrix must be freed
                           by the caller.

    \returns A (sparse or dense, depending on the array) element matrix.
*/
Matrix *elmat_from_array(int elno,
                         const agg_partititoning_relations_t *agg_part_rels,
                         void *data, bool& free_matr);

/*! \brief A function returning standard element matrices in a geometric level.

    \param elno (IN) The element number.
    \param agg_part_rels (IN) The partitioning relations. Can be NULL.
    \param data (IN) Simply a BilinearForm pointer.
                     Type: \b ELMAT_DATA_TYPE_BILINEAR_FORM.
    \param free_matr (OUT) Indicates whether the returned matrix must be freed
                           by the caller.

    \returns A sparse element matrix.

    \warning It must be compiled with Run-Time Type Information (RTTI) enabled
             for \em dynamic_cast to work.
*/
Matrix *elmat_standard_geometric_sparse(int elno,
                         const agg_partititoning_relations_t *agg_part_rels,
                         void *data, bool& free_matr);

/*! \brief A function returning standard element matrices in a geometric level.

    \param elno (IN) The element number.
    \param agg_part_rels (IN) The partitioning relations. Can be NULL.
    \param data (IN) Simply a BilinearForm pointer.
                     Type: \b ELMAT_DATA_TYPE_BILINEAR_FORM.
    \param free_matr (OUT) Indicates whether the returned matrix must be freed
                           by the caller.

    \returns A dense element matrix.

    \warning It must be compiled with Run-Time Type Information (RTTI) enabled
             for \em dynamic_cast to work.
*/
Matrix *elmat_standard_geometric_dense(int elno,
                         const agg_partititoning_relations_t *agg_part_rels,
                         void *data, bool& free_matr);

#endif // _ELMAT_HPP
