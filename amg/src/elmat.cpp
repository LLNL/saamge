/*! \file

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

#include "common.hpp"
#include "elmat.hpp"
#include <mfem.hpp>
#include "aggregates.hpp"
#include "levels.hpp"
#include "mbox.hpp"

namespace saamge
{
using namespace mfem;

ElementMatrixStandardGeometric::ElementMatrixStandardGeometric(
    const agg_partitioning_relations_t& agg_part_rels,
    SparseMatrix & assembled_processor_matrix,
    ParBilinearForm * form) 
    : 
    ElementMatrixProvider(agg_part_rels),
    form(form),
    assembled_processor_matrix_(assembled_processor_matrix),
    bdr_cond_imposed_(true),
    assemble_ess_diag_(true)
{
    is_geometric = true;
}

Matrix * ElementMatrixStandardGeometric::BuildAEStiff(int elno) const
{
    return agg_build_AE_stiffm_with_global(
        assembled_processor_matrix_, elno, agg_part_rels, this,
        bdr_cond_imposed_, assemble_ess_diag_);
}

/**
   This is ripped from a function called elmat_standard_geometric_dense().
   See elmat_standard_geometric_sparse for an alternative implementation.
*/
Matrix * ElementMatrixStandardGeometric::GetMatrix(
    int elno, bool& free_matr) const
{
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

Matrix *ElementDomainLFVectorStandardGeometric::GetMatrix(int elno, bool& free_matr) const
{
    SA_ASSERT(dlfi);
    SA_ASSERT(fes);
    SA_ASSERT(!agg_part_rels.elem_to_dof ||
              agg_part_rels.elem_to_dof->Size() == fes->GetNE());
    SA_ASSERT(0 <= elno && elno < fes->GetNE());
    SA_ASSERT(!agg_part_rels.elem_to_dof ||
              agg_part_rels.elem_to_dof->RowSize(elno) ==
              fes->GetFE(elno)->GetDof() * fes->GetVDim());

    Vector elvect;
    ElementTransformation *eltrans;

    eltrans = fes->GetElementTransformation(elno);
    dlfi->AssembleRHSElementVect(*fes->GetFE(elno), *eltrans, elvect);
    SA_ASSERT(elvect.Size() == fes->GetFE(elno)->GetDof() * fes->GetVDim());

#ifdef SA_ASSERTS
    if (agg_part_rels.elem_to_dof)
    {
        Array<int> vdofs;
        fes->GetElementVDofs(elno, vdofs);
        SA_ASSERT(vdofs.Size() == agg_part_rels.elem_to_dof->RowSize(elno));
        for (int i=0; i < vdofs.Size(); ++i)
            SA_ASSERT(vdofs[i] == agg_part_rels.elem_to_dof->GetRow(elno)[i]);
    }
#endif

    free_matr = true;
    return mbox_create_diag_sparse_steal(elvect);
}

ElementMatrixParallelCoarse::ElementMatrixParallelCoarse(
    const agg_partitioning_relations_t& agg_part_rels,
    levels_level_t *level) 
    : 
    ElementMatrixProvider(agg_part_rels),
    level(level)
{
    is_geometric = false;
}

Matrix * ElementMatrixParallelCoarse::BuildAEStiff(int elno) const
{
    return agg_build_AE_stiffm(elno, agg_part_rels, this);
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

    Matrix * finer_AE_stiffm =
        finer_level->tg_data->interp_data->AEs_stiffm[elno];
    int ae_finedof = finer_AE_stiffm->Height();
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
    // (2) column ordering has been a problem in the past, so worth checking
    //     again even though it looks okay now
    // (3) this may be an efficiency disaster

    int localcol = 0; // debug variable used to check column ordering
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
            int dof_in_AE = agg_map_id_glob_to_AE(finedof, elno,
                                                  *finer_level->agg_part_rels);
            SA_ASSERT(dof_in_AE >= 0);
            SA_ASSERT(dof_in_AE < local_interp.Height());
            local_interp_rows.Append(dof_in_AE);
        }
        // assume in following loop that we go through mis_in_AE in the same
        // order that we go through mises in contrib_mises...
        for (int i=0; i<mis_numcoarsedof[mis]; ++i) 
        {
            int coarse_dof_num = agg_part_rels.mis_coarsedofoffsets[mis] + i;
            // next line is a potentially somewhat expensive reverse lookup (is there a way to use dof_to_elem?)
            int column_to_put = agg_elem_in_col(elno, coarse_dof_num,
                                                *agg_part_rels.elem_to_dof);
            SA_ASSERT(column_to_put >= 0);
            SA_ASSERT(column_to_put < local_interp.Width());
            local_interp_columns.Append(column_to_put);
            localcol++;
        }

        local_interp.AddSubMatrix(
            local_interp_rows, local_interp_columns, 
            *finer_level->tg_data->interp_data->mis_tent_interps[mis]);
    }

    // Compute and return the element matrix.
    free_matr = true;
    local_interp.Finalize();

    Matrix *out;
    SparseMatrix *sfiner_AE_stiffm = dynamic_cast<SparseMatrix *>(finer_AE_stiffm);
    SparseMatrix *local_restr = Transpose(local_interp);
    if (sfiner_AE_stiffm)
        out = RAP(*sfiner_AE_stiffm, *local_restr);
    else
    {
        DenseMatrix *dfiner_AE_stiffm = dynamic_cast<DenseMatrix *>(finer_AE_stiffm);
        SA_ASSERT(dfiner_AE_stiffm);
        DenseMatrix RA, dlocal_interp;
        mbox_mult_sparse_to_dense(*local_restr, *dfiner_AE_stiffm, RA);
        mbox_convert_sparse_to_dense(local_interp, dlocal_interp);
        DenseMatrix *RAP = new DenseMatrix;
        Mult(RA, dlocal_interp, *RAP);
        out = RAP;
    }

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

ElementMatrixArray::ElementMatrixArray(
    const agg_partitioning_relations_t& agg_part_rels,
    const Array<SparseMatrix *>& elem_matrs)
    : 
    ElementMatrixProvider(agg_part_rels),
    elem_matrs(elem_matrs)
{
    is_geometric = false;
}

Matrix * ElementMatrixArray::BuildAEStiff(int elno) const
{
    SA_ASSERT(agg_part_rels.elem_to_dof &&
              0 <= elno &&
              elno < agg_part_rels.elem_to_dof->Size());
    SA_ASSERT(elem_matrs);

    // Matrix * const * const elem_matrs = (Matrix * const *)data;
    SA_ASSERT(elem_matrs[elno]);

    return elem_matrs[elno];
}

Matrix * ElementMatrixArray::GetMatrix(
    int elno, bool& free_matr) const
{
    free_matr = false;
    return BuildAEStiff(elno);
}

ElementMatrixDenseArray::ElementMatrixDenseArray(
    const agg_partitioning_relations_t& agg_part_rels,
    const Array<DenseMatrix *>& elem_matrs)
    :
    ElementMatrixProvider(agg_part_rels),
    elem_matrs(elem_matrs)
{
    is_geometric = false;
}

Matrix * ElementMatrixDenseArray::BuildAEStiff(int elno) const
{
    return agg_build_AE_stiffm(elno, agg_part_rels, this);
}

Matrix * ElementMatrixDenseArray::GetMatrix(
    int elno, bool& free_matr) const
{
    SA_ASSERT(agg_part_rels.elem_to_dof &&
              0 <= elno &&
              elno < agg_part_rels.elem_to_dof->Size());
    SA_ASSERT(elem_matrs);
    SA_ASSERT(elem_matrs[elno]);

    free_matr = false;
    return elem_matrs[elno];
}

} // namespace saamge
