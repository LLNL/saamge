/*! \file
    \brief Routines computing element matrices.

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
#include "levels.hpp"

using namespace mfem;

/* Functions */
/*! \brief A function returning element matrices for a non-geometric level.

    It uses a prolongator to define coarse DoFs, and respectively
    the coarse element matrices.

    It is a 'general' implementation that can use 'any' intergrid operators. It
    simply restricts the prolongator (or, equivalently, the restriction
    operator) to the AE and performs RAP.

    We want to compute an element matrix on a 'current' level using 'fine'
    information.

    TODO: Almost surely a more efficient approach exists but currently we use
          this simple implementation.

    \param elno (IN) The element number.
    \param finer_agg_part_rels (IN) The partitioning relations for the 'fine'
                                    level.
    \param finer_AE_stiffm (IN) The local stiffness matrix of the corresponding
                                AE on the 'fine' level. It MUST be the one
                                corresponding to \a elno.
    \param restr (IN) The global restriction operator defining the transfer
                      from the 'fine' to the 'current' level.
    \param elem_to_dof (IN) The relation between elements and DoFs on the
                            'current' level.
    \param ND (IN) The number of DoFs on the 'current' level.

    \returns The element matrix.

    \warning The returned sparse matrix must be freed by the caller.

    ORPHAN CODE---never called

    DEPRECATED
*/
SparseMatrix *elmat_using_interp_restr(
    int elno,
    const agg_partitioning_relations_t& finer_agg_part_rels,
    const SparseMatrix& finer_AE_stiffm, const SparseMatrix& restr,
    const Table& elem_to_dof, int ND);

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

    ORPHAN CODE --- never called

    DEPRECATED
*/
Matrix *elmat_standard_geometric_sparse(
    int elno,
    const agg_partitioning_relations_t *agg_part_rels,
    void *data, bool& free_matr);

/**
   Provides element matrices for the construction of agglomerate matrices.
*/
class ElementMatrixProvider
{
public:
    ElementMatrixProvider(const agg_partitioning_relations_t& agg_part_rels) : agg_part_rels(agg_part_rels) {}
    virtual ~ElementMatrixProvider() {};
    
    virtual Matrix * GetMatrix(int elno, bool& free_matr) const = 0;
protected:
    const agg_partitioning_relations_t& agg_part_rels;
};

/**
   Standard elmat for fine level, basically uses an mfem::ParBilinearForm
   to assemble everything.
*/
class ElementMatrixStandardGeometric : public ElementMatrixProvider
{
public:
    ElementMatrixStandardGeometric(const agg_partitioning_relations_t& agg_part_rels, ParBilinearForm* form);
    virtual Matrix * GetMatrix(int elno, bool& free_matr) const;
private:
    ParBilinearForm* form;
};

/**
   Standard elmat for coarse level.

   At its heart this is a RAP, using a saved local MIS interp matrix 
   (unsmoothed) and the fine level agglomerate matrix (also must be saved...)
*/
class ElementMatrixParallelCoarse : public ElementMatrixProvider
{
public:
    ElementMatrixParallelCoarse(const agg_partitioning_relations_t& agg_part_rels, levels_level_t * level);
    virtual Matrix * GetMatrix(int elno, bool& free_matr) const;
private:
    levels_level_t * level;
};

/**
   Returns element matrices from an array of generated matrices...

   Not really used yet, but might make sense for future algebraic interface.
*/
class ElementMatrixArray : public ElementMatrixProvider
{
public:
    ElementMatrixArray(const agg_partitioning_relations_t& agg_part_rels, const Matrix * const *elem_matrs);
    virtual Matrix * GetMatrix(int elno, bool& free_matr) const;
private:
    const Matrix * const *elem_matrs;
};

#endif // _ELMAT_HPP
