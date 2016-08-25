/*! \file

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

#include "common.hpp"
#include "elmat.hpp"
#include <mfem.hpp>
#include "aggregates.hpp"
#include "mbox.hpp"

/* Functions */

Matrix **elmat_create_array(agg_elmat_callback_ft elmat_callback,
                const agg_partititoning_relations_t *agg_part_rels, void *data)
{
    SA_ASSERT(agg_part_rels);
    SA_ASSERT(agg_part_rels->elem_to_dof);
    SA_ASSERT(elmat_callback);
    bool free_matr;
    Matrix *elem_matr;
    SparseMatrix *spm;
    const int elems = agg_part_rels->elem_to_dof->Size();
    SA_ASSERT(0 < elems);
    Matrix **elem_arr = (Matrix **)new Matrix *[elems];
    SA_ASSERT(elem_arr);

    for (int i=0; i < elems; ++i)
    {
        elem_matr = elmat_callback(i, agg_part_rels, data, free_matr);
        SA_ASSERT(elem_matr);

        if (free_matr)
        {
            elem_arr[i] = elem_matr;
            continue;
        }

        spm = dynamic_cast<SparseMatrix *>(elem_matr);
        if (spm)
            elem_arr[i] = mbox_copy_sparse_matr(spm);
        else
        {
            SA_ASSERT(dynamic_cast<DenseMatrix *>(elem_matr));
            elem_arr[i] =
                new DenseMatrix(*dynamic_cast<DenseMatrix *>(elem_matr));
        }
        SA_ASSERT(elem_arr[i]);
    }

    return elem_arr;
}

Matrix *elmat_from_array(int elno,
                         const agg_partititoning_relations_t *agg_part_rels,
                         void *data, bool& free_matr)
{
    SA_ASSERT(!(agg_part_rels) || (agg_part_rels->elem_to_dof &&
                                    0 <= elno &&
                                    elno < agg_part_rels->elem_to_dof->Size()));
    SA_ASSERT(data);

    Matrix * const * const elem_matrs = (Matrix * const *)data;
    SA_ASSERT(elem_matrs[elno]);

    free_matr = false;
    return elem_matrs[elno];
}

Matrix *elmat_standard_geometric_sparse(int elno,
                         const agg_partititoning_relations_t *agg_part_rels,
                         void *data, bool& free_matr)
{
    SA_ASSERT(data);
    BilinearForm *bf = (BilinearForm *)data;
    SA_ASSERT(!agg_part_rels || !agg_part_rels->elem_to_dof ||
              agg_part_rels->elem_to_dof->Size() == bf->GetFES()->GetNE());
    SA_ASSERT(0 <= elno && elno < bf->GetFES()->GetNE());
    SA_ASSERT(!agg_part_rels || !agg_part_rels->elem_to_dof ||
              agg_part_rels->elem_to_dof->RowSize(elno) ==
              bf->GetFES()->GetFE(elno)->GetDof());
    DenseMatrix elmat(bf->GetFES()->GetFE(elno)->GetDof());

    bf->ComputeElementMatrix(elno, elmat);

    free_matr = true;
    return mbox_convert_dense_to_sparse(elmat);
}

Matrix *elmat_standard_geometric_dense(int elno,
                         const agg_partititoning_relations_t *agg_part_rels,
                         void *data, bool& free_matr)
{
    SA_ASSERT(data);
    BilinearForm *bf = (BilinearForm *)data;
    SA_ASSERT(!agg_part_rels || !agg_part_rels->elem_to_dof ||
              agg_part_rels->elem_to_dof->Size() == bf->GetFES()->GetNE());
    SA_ASSERT(0 <= elno && elno < bf->GetFES()->GetNE());
    SA_ASSERT(!agg_part_rels || !agg_part_rels->elem_to_dof ||
              agg_part_rels->elem_to_dof->RowSize(elno) ==
              bf->GetFES()->GetFE(elno)->GetDof());
    DenseMatrix *elmat =
        new DenseMatrix(bf->GetFES()->GetFE(elno)->GetDof());

    bf->ComputeElementMatrix(elno, *elmat);

    free_matr = true;
    return elmat;
}
