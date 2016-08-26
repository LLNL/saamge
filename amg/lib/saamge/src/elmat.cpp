/*! \file

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

#include "common.hpp"
#include "elmat.hpp"
#include <mfem.hpp>
#include "aggregates.hpp"
#include "levels.hpp"
#include "mbox.hpp"

/* Functions */

SparseMatrix *elmat_using_interp_restr(int elno,
    const agg_partitioning_relations_t& finer_agg_part_rels,
    const SparseMatrix& finer_AE_stiffm, const SparseMatrix& restr,
    const Table& elem_to_dof, int ND)
{
    SA_ASSERT(0 < finer_agg_part_rels.ND);
    SA_ASSERT(finer_agg_part_rels.AE_to_dof);

#if (SA_IS_DEBUG_LEVEL(8))
    SA_ASSERT(const_cast<Table *>(finer_agg_part_rels.AE_to_dof)->Width() ==
              finer_agg_part_rels.ND); //XXX: May affect efficiency!
#endif

    SA_ASSERT(const_cast<SparseMatrix&>(finer_AE_stiffm).Finalized());
    SA_ASSERT(finer_AE_stiffm.Size() == finer_AE_stiffm.Width());
    SA_ASSERT(finer_AE_stiffm.Size() ==
              finer_agg_part_rels.AE_to_dof->RowSize(elno));

    // Build the local restriction matrix:

    SA_ASSERT(const_cast<SparseMatrix&>(restr).Finalized());
    SA_ASSERT(restr.Size() == ND);
    const int * const grestr_I = restr.GetI();
    const int * const grestr_J = restr.GetJ();
    const double * const grestr_Data = restr.GetData();

    const int * const elem_dofs = elem_to_dof.GetRow(elno);
    const int num_elem_dofs = elem_to_dof.RowSize(elno);

    SparseMatrix local_restr(num_elem_dofs, finer_AE_stiffm.Size());

    int i, k;
    for (i=0; i < num_elem_dofs; ++i)
    {
#ifdef SA_ASSERTS
        int added = 0;
#endif
        const int row_num = elem_dofs[i];
        SA_ASSERT(0 <= row_num && row_num < ND);
        SA_ASSERT(0 < grestr_I[row_num + 1] - grestr_I[row_num]);

        for (k = grestr_I[row_num]; k < grestr_I[row_num + 1]; ++k)
        {
            SA_ASSERT(0 <= grestr_J[k] && grestr_J[k] < finer_agg_part_rels.ND);
            const int local_dof = agg_map_id_glob_to_AE(grestr_J[k], elno,
                                                        finer_agg_part_rels);
            const double data = grestr_Data[k];
            if (0 > local_dof || 0. == data)
                continue;
#ifdef SA_ASSERTS
            ++added;
#endif
            SA_ASSERT(0 <= local_dof && local_dof < finer_AE_stiffm.Size());
            local_restr.Set(i, local_dof, data);
            SA_ASSERT(local_restr.RowSize(i) == added);
        }
        SA_ASSERT(0 <= added && added <= finer_AE_stiffm.Size() &&
                  added <= grestr_I[row_num + 1] - grestr_I[row_num]);
    }
    local_restr.Finalize();

    // Compute and return the element matrix.
    // return RAP(const_cast<SparseMatrix&>(finer_AE_stiffm), local_restr);
    return RAP(finer_AE_stiffm, local_restr);
}


Matrix *elmat_standard_geometric_sparse(
    int elno,
    const agg_partitioning_relations_t *agg_part_rels,
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

ElementMatrixStandardGeometric::ElementMatrixStandardGeometric(
    const agg_partitioning_relations_t& agg_part_rels,
    ParBilinearForm * form) 
    : 
    ElementMatrixProvider(agg_part_rels),
    form(form)
{
}

/**
   This is ripped from a function called elmat_standard_geometric_dense().
   See elmat_standard_geometric_sparse for an alternative implementation.
*/
Matrix * ElementMatrixStandardGeometric::GetMatrix(
    int elno, bool& free_matr) const
{
    // return elmat_standard_geometric_dense(elno, agg_part_rels, (void*) form, free_matr);
    SA_ASSERT(form);
    BilinearForm *bf = form;
    SA_ASSERT(!agg_part_rels.elem_to_dof ||
              agg_part_rels.elem_to_dof->Size() == bf->GetFES()->GetNE());
    SA_ASSERT(0 <= elno && elno < bf->GetFES()->GetNE());
    SA_ASSERT(!agg_part_rels.elem_to_dof ||
              agg_part_rels.elem_to_dof->RowSize(elno) ==
              bf->GetFES()->GetFE(elno)->GetDof() * bf->GetFES()->GetVDim());
    // this is a bit unnecessary, because ComputeElementMatrix will set the size...
    DenseMatrix *elmat =
        new DenseMatrix(bf->GetFES()->GetFE(elno)->GetDof() * bf->GetFES()->GetVDim());

    bf->ComputeElementMatrix(elno, *elmat);
    SA_ASSERT(elmat->Size() == bf->GetFES()->GetFE(elno)->GetDof() * bf->GetFES()->GetVDim());

    free_matr = true;
    return elmat;
}

ElementMatrixParallelCoarse::ElementMatrixParallelCoarse(
    const agg_partitioning_relations_t& agg_part_rels,
    levels_level_t *level) 
    : 
    ElementMatrixProvider(agg_part_rels),
    level(level)
{
}

Matrix * ElementMatrixParallelCoarse::GetMatrix(
    int elno, bool& free_matr) const
{
    // return elmat_parallel(elno, agg_part_rels, (void*) level, free_matr);
    if (agg_part_rels.testmesh)
        SA_PRINTF("elmat_parallel(%d)\n", elno);
    // const levels_level_t * const finer_level = (levels_level_t *)data;
    const levels_level_t * const finer_level = level;
    Table * AE_to_mis = finer_level->agg_part_rels->AE_to_mis;
    Table * mis_to_dof = finer_level->agg_part_rels->mis_to_dof;
    int * mis_numcoarsedof = finer_level->tg_data->interp_data->mis_numcoarsedof;

    SparseMatrix * finer_AE_stiffm = finer_level->tg_data->interp_data->AEs_stiffm[elno];
    int ae_finedof = finer_AE_stiffm->Size();
    int ae_coarsedof = 0;
    Array<int> mis_in_AE;
    AE_to_mis->GetRow(elno,mis_in_AE);
    mis_in_AE.Sort(); // note you are actually modifying AE_to_mis Table here
    for (int j=0; j<mis_in_AE.Size(); ++j)
    {
        ae_coarsedof += mis_numcoarsedof[mis_in_AE[j]];
    }

    // int * elem_dofs = agg_part_rels->elem_to_dof->GetRow(elno);

    SparseMatrix local_interp(ae_finedof, ae_coarsedof);

    // possible issues:
    // (1) we're building interp below, can we build restr directly?
    // (2) column ordering has been a problem in the past, so worth checking again even though it looks okay now
    // (3) this may be an efficiency disaster

    int localcol = 0; // this variable not necessary, used to check column ordering
    for (int j=0; j<mis_in_AE.Size(); ++j)
    {
        Array<int> local_interp_rows;
        Array<int> local_interp_columns;

        int mis = mis_in_AE[j];
        int num_finedof_in_mis = mis_to_dof->RowSize(mis);
        int * finedof_in_mis = mis_to_dof->GetRow(mis);
        for (int i=0; i<num_finedof_in_mis; ++i)
        {
            int finedof = finedof_in_mis[i];
            int dof_in_AE = agg_map_id_glob_to_AE(finedof, elno, *finer_level->agg_part_rels);
            SA_ASSERT(dof_in_AE >= 0);
            SA_ASSERT(dof_in_AE < local_interp.Height());
            local_interp_rows.Append(dof_in_AE);
        }

        for (int i=0; i<mis_numcoarsedof[mis]; ++i) // assume here that we go through mis_in_AE in the same order that we go through mises in contrib_mises...
        {
            int coarse_dof_num = agg_part_rels.mis_coarsedofoffsets[mis] + i;
            // next line is a potentially somewhat expensive reverse lookup (is there a way to use dof_to_elem?)
            int column_to_put = agg_elem_in_col(elno, coarse_dof_num, *agg_part_rels.elem_to_dof);
            SA_ASSERT(column_to_put >= 0);
            SA_ASSERT(column_to_put < local_interp.Width());
            local_interp_columns.Append(column_to_put);
            localcol++;
        }

        local_interp.AddSubMatrix(local_interp_rows, local_interp_columns, 
                                  *finer_level->tg_data->interp_data->mis_tent_interps[mis]);
    }

    // Compute and return the element matrix.
    free_matr = true;
    local_interp.Finalize();

    SparseMatrix * local_restr = Transpose(local_interp);
    Matrix * out = RAP(*finer_AE_stiffm, *local_restr);

    if (agg_part_rels.testmesh && elno == 0 && PROC_RANK == 0)
    {
        std::ofstream out1("local_interp_0.0.mat");
        local_interp.Print(out1);
        std::ofstream out2("finer_AE_stiffm_0.0.mat");
        finer_AE_stiffm->Print(out2);
        std::ofstream out3("coarse_elmat_0.0.mat");
        out->Print(out3);
    }

    delete local_restr;
    return out;
}

ElementMatrixArray::ElementMatrixArray(const agg_partitioning_relations_t& agg_part_rels,
                                       const Matrix * const *elem_matrs) : 
    ElementMatrixProvider(agg_part_rels),
    elem_matrs(elem_matrs)
{
}

Matrix * ElementMatrixArray::GetMatrix(
    int elno, bool& free_matr) const
{
    // return elmat_from_array(elno, agg_part_rels, (void*) elem_matrs, free_matr);
    SA_ASSERT(agg_part_rels.elem_to_dof &&
              0 <= elno &&
              elno < agg_part_rels.elem_to_dof->Size());
    SA_ASSERT(elem_matrs);

    // Matrix * const * const elem_matrs = (Matrix * const *)data;
    SA_ASSERT(elem_matrs[elno]);

    free_matr = false;
    // don't like the const_cast, but previously it was a c-style cast...
    return const_cast<Matrix*>(elem_matrs[elno]); 
}
