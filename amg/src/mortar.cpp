/*
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
#include "mortar.hpp"
#include <mfem.hpp>
#include "aggregates.hpp"
#include "interp.hpp"
#include "xpacks.hpp"
#include "nonconf.hpp"

namespace saamge
{
using namespace mfem;

/**
    Obtains the mortar condensed element matrices and maintains some additional data for later use.
    This is an over-constrained version with identity bases on the faces.
*/
static inline
void mortar_condensed_local_matrices_overconstrained(interp_data_t& interp_data, const agg_partitioning_relations_t& agg_part_rels)
{
    const int nparts = agg_part_rels.nparts;
    SA_ASSERT(interp_data.AEs_stiffm);
    SA_ASSERT(interp_data.celements_cdofs_offsets);
    SA_ASSERT(interp_data.cfaces_cdofs_offsets);
    SA_ASSERT(!interp_data.invAii);
    SA_ASSERT(!interp_data.invAiiAib);
    SA_ASSERT(!interp_data.schurs);

    interp_data.invAii = new DenseMatrix*[nparts];
    interp_data.invAiiAib = new DenseMatrix*[nparts];
    interp_data.schurs = new DenseMatrix*[nparts];

    DenseMatrix *Aii;
    DenseMatrix X0;
    for (int i=0; i < nparts; ++i)
    {
        SA_ASSERT(interp_data.AEs_stiffm[i]);
        SA_ASSERT(interp_data.AEs_stiffm[i]->Height() == interp_data.AEs_stiffm[i]->Width());
        SA_ASSERT(interp_data.AEs_stiffm[i]->Height() == interp_data.celements_cdofs_offsets[i+1] -
                                                         interp_data.celements_cdofs_offsets[i]);
        const int ndofs = agg_part_rels.AE_to_dof->RowSize(i);
        const int interior_dofs = interp_data.AEs_stiffm[i]->Height();
        SA_ASSERT(interior_dofs <= ndofs);
        const int num_cfaces = agg_part_rels.AE_to_cface->RowSize(i);
        const int * const cfaces = agg_part_rels.AE_to_cface->GetRow(i);
        int bdr_dofs=0;
        for (int j=0; j < num_cfaces; ++j)
        {
            const int cface = cfaces[j];
            SA_ASSERT(0 <= cface && cface < agg_part_rels.num_cfaces);
            bdr_dofs += interp_data.cfaces_cdofs_offsets[cface+1] -
                        interp_data.cfaces_cdofs_offsets[cface];
        }
        Aii = new DenseMatrix(interior_dofs + bdr_dofs);
        X0.SetSize(interior_dofs + bdr_dofs, bdr_dofs);
        X0 = 0.0;
        for (int j=0; j < interior_dofs; ++j)
        {
            Array<int> cols;
            Vector srow;
            interp_data.AEs_stiffm[i]->GetRow(j, cols, srow);
            for (int k=0; k < cols.Size(); ++k)
            {
                SA_ASSERT(0 <= cols[k] && cols[k] < interior_dofs);
                SA_ASSERT(0.0 == (*Aii)(j, cols[k]));
                (*Aii)(j, cols[k]) = srow(k);
            }
        }
        Array<int> map(ndofs);
        int intdofs_ctr = 0;
        for (int j=0; j < ndofs; ++j)
        {
            const int gj = agg_num_col_to_glob(*agg_part_rels.AE_to_dof, i, j);
            SA_ASSERT(0 <= gj && gj < agg_part_rels.ND);
            if (!SA_IS_SET_A_FLAG(agg_part_rels.agg_flags[gj], AGG_ON_ESS_DOMAIN_BORDER_FLAG))
            {
                SA_ASSERT(intdofs_ctr < interior_dofs);
                map[j] = intdofs_ctr;
                ++intdofs_ctr;
            } else
                map[j] = -1;
        }
        SA_ASSERT(intdofs_ctr == interior_dofs);
        SparseMatrix *D = mbox_snd_diagA_sparse_from_sparse(*interp_data.AEs_stiffm[i]);
        delete interp_data.AEs_stiffm[i];
        interp_data.AEs_stiffm[i] = NULL;
        SA_ASSERT(D->Height() == interior_dofs);
        SA_ASSERT(D->NumNonZeroElems() == interior_dofs);
        const Vector diag(D->GetData(), D->Height());
        int cface_offset = 0;
        for (int j=0; j < num_cfaces; ++j)
        {
            const int cface = cfaces[j];
            SA_ASSERT(0 <= cface && cface < agg_part_rels.num_cfaces);
            const int cface_dofs = interp_data.cfaces_cdofs_offsets[cface+1] -
                                   interp_data.cfaces_cdofs_offsets[cface];
            SA_ASSERT(cface_offset + cface_dofs <= bdr_dofs);
            SA_ASSERT(agg_part_rels.cface_to_dof->RowSize(cface) == agg_part_rels.cfaces_dof_size[cface]);
            const int num_dofs = agg_part_rels.cface_to_dof->RowSize(cface);
            SA_ASSERT(cface_dofs <= num_dofs);
            const int * const dofs = agg_part_rels.cface_to_dof->GetRow(cface);
            int ctr = 0;
            for (int k=0; k < num_dofs; ++k)
            {
                if (SA_IS_SET_A_FLAG(agg_part_rels.agg_flags[dofs[k]], AGG_ON_ESS_DOMAIN_BORDER_FLAG))
                    continue;
                const int loc = agg_map_id_glob_to_AE(dofs[k], i, agg_part_rels);
                SA_ASSERT(0 <= loc && loc < ndofs);
                const int ldof = map[loc];
                SA_ASSERT(0 <= ldof && ldof < interior_dofs);
                SA_ASSERT(ctr < cface_dofs && cface_offset + ctr < bdr_dofs);
                SA_ASSERT(0.0 == (*Aii)(ldof, interior_dofs + cface_offset + ctr));
                (*Aii)(ldof, interior_dofs + cface_offset + ctr) = diag(ldof);
                SA_ASSERT(0.0 == (*Aii)(interior_dofs + cface_offset + ctr, ldof));
                (*Aii)(interior_dofs + cface_offset + ctr, ldof) = diag(ldof);
                SA_ASSERT(0.0 == X0(interior_dofs + cface_offset + ctr, cface_offset + ctr));
                X0(interior_dofs + cface_offset + ctr, cface_offset + ctr) = -diag(ldof);
                ++ctr;
            }
            SA_ASSERT(cface_dofs == ctr);
            cface_offset += cface_dofs;
        }
        SA_ASSERT(cface_offset == bdr_dofs);
        delete D;

        Aii->Invert();
        interp_data.invAii[i] = Aii;
        Aii = NULL;
        interp_data.invAiiAib[i] = new DenseMatrix(interp_data.invAii[i]->Height(), X0.Width());
        Mult(*interp_data.invAii[i], X0, *interp_data.invAiiAib[i]);
        interp_data.schurs[i] = new DenseMatrix(X0.Width(), interp_data.invAiiAib[i]->Width());
        MultAtB(X0, *interp_data.invAiiAib[i], *interp_data.schurs[i]);
        interp_data.schurs[i]->Neg();
        SA_ASSERT(interp_data.schurs[i]->Height() == interp_data.schurs[i]->Width() &&
                  interp_data.schurs[i]->Height() == bdr_dofs);
        SA_ASSERT(interp_data.invAii[i]->Height() == interp_data.invAii[i]->Width() &&
                  interp_data.invAii[i]->Height() == interior_dofs + bdr_dofs);
        SA_ASSERT(interp_data.invAiiAib[i]->Height() == interior_dofs + bdr_dofs &&
                  interp_data.invAiiAib[i]->Width() == bdr_dofs);
    }
    delete [] interp_data.AEs_stiffm;
    interp_data.AEs_stiffm = NULL;
}

/**
    Obtains the mortar condensed element matrices and maintains some additional data for later use.
*/
static inline
void mortar_condensed_local_matrices(interp_data_t& interp_data, const agg_partitioning_relations_t& agg_part_rels)
{
    const int nparts = agg_part_rels.nparts;
    SA_ASSERT(interp_data.AEs_stiffm);
    SA_ASSERT(interp_data.celements_cdofs_offsets);
    SA_ASSERT(interp_data.cfaces_cdofs_offsets);
    SA_ASSERT(interp_data.cfaces_bases);
    SA_ASSERT(!interp_data.invAii);
    SA_ASSERT(!interp_data.invAiiAib);
    SA_ASSERT(!interp_data.schurs);

    interp_data.invAii = new DenseMatrix*[nparts];
    interp_data.invAiiAib = new DenseMatrix*[nparts];
    interp_data.schurs = new DenseMatrix*[nparts];

    DenseMatrix *Aii;
    DenseMatrix X0;
    for (int i=0; i < nparts; ++i)
    {
        SA_ASSERT(interp_data.AEs_stiffm[i]);
        SA_ASSERT(interp_data.AEs_stiffm[i]->Height() == interp_data.AEs_stiffm[i]->Width());
        SA_ASSERT(interp_data.AEs_stiffm[i]->Height() == interp_data.celements_cdofs_offsets[i+1] -
                                                         interp_data.celements_cdofs_offsets[i]);
        const int ndofs = agg_part_rels.AE_to_dof->RowSize(i);
        const int interior_dofs = interp_data.AEs_stiffm[i]->Height();
        SA_ASSERT(interior_dofs <= ndofs);
        const int num_cfaces = agg_part_rels.AE_to_cface->RowSize(i);
        const int * const cfaces = agg_part_rels.AE_to_cface->GetRow(i);
        int bdr_dofs=0;
        for (int j=0; j < num_cfaces; ++j)
        {
            const int cface = cfaces[j];
            SA_ASSERT(0 <= cface && cface < agg_part_rels.num_cfaces);
            SA_ASSERT(interp_data.cfaces_bases[cface]);
            SA_ASSERT(interp_data.cfaces_bases[cface]->Width() == interp_data.cfaces_cdofs_offsets[cface+1] -
                                                                  interp_data.cfaces_cdofs_offsets[cface]);
            bdr_dofs += interp_data.cfaces_cdofs_offsets[cface+1] -
                        interp_data.cfaces_cdofs_offsets[cface];
        }
        Aii = new DenseMatrix(interior_dofs + bdr_dofs);
        X0.SetSize(interior_dofs + bdr_dofs, bdr_dofs);
        X0 = 0.0;
        for (int j=0; j < interior_dofs; ++j)
        {
            Array<int> cols;
            Vector srow;
            interp_data.AEs_stiffm[i]->GetRow(j, cols, srow);
            for (int k=0; k < cols.Size(); ++k)
            {
                SA_ASSERT(0 <= cols[k] && cols[k] < interior_dofs);
                SA_ASSERT(0.0 == (*Aii)(j, cols[k]));
                (*Aii)(j, cols[k]) = srow(k);
            }
        }
        Array<int> map(ndofs);
        int intdofs_ctr = 0;
        for (int j=0; j < ndofs; ++j)
        {
            const int gj = agg_num_col_to_glob(*agg_part_rels.AE_to_dof, i, j);
            SA_ASSERT(0 <= gj && gj < agg_part_rels.ND);
            if (!SA_IS_SET_A_FLAG(agg_part_rels.agg_flags[gj], AGG_ON_ESS_DOMAIN_BORDER_FLAG))
            {
                SA_ASSERT(intdofs_ctr < interior_dofs);
                map[j] = intdofs_ctr;
                ++intdofs_ctr;
            } else
                map[j] = -1;
        }
        SA_ASSERT(intdofs_ctr == interior_dofs);
        SparseMatrix *D = mbox_snd_diagA_sparse_from_sparse(*interp_data.AEs_stiffm[i]);
        delete interp_data.AEs_stiffm[i];
        interp_data.AEs_stiffm[i] = NULL;
        SA_ASSERT(D->Height() == interior_dofs);
        SA_ASSERT(D->NumNonZeroElems() == interior_dofs);
        const Vector diag(D->GetData(), D->Height());
        int cface_offset = 0;
        for (int j=0; j < num_cfaces; ++j)
        {
            const int cface = cfaces[j];
            SA_ASSERT(0 <= cface && cface < agg_part_rels.num_cfaces);
            const int cface_dofs = interp_data.cfaces_cdofs_offsets[cface+1] -
                                   interp_data.cfaces_cdofs_offsets[cface];
            SA_ASSERT(cface_offset + cface_dofs <= bdr_dofs);
            SA_ASSERT(agg_part_rels.cface_to_dof->RowSize(cface) == agg_part_rels.cfaces_dof_size[cface]);
            SA_ASSERT(interp_data.cfaces_bases[cface]->Height() == agg_part_rels.cfaces_dof_size[cface]);
            const int num_dofs = agg_part_rels.cface_to_dof->RowSize(cface);
            SA_ASSERT(cface_dofs <= num_dofs);
            const int * const dofs = agg_part_rels.cface_to_dof->GetRow(cface);
            for (int l=0; l < cface_dofs; ++l)
            {
                const Vector cface_base(interp_data.cfaces_bases[cface]->GetData() + l * num_dofs, num_dofs);
                for (int k=0; k < num_dofs; ++k)
                {
                    if (SA_IS_SET_A_FLAG(agg_part_rels.agg_flags[dofs[k]], AGG_ON_ESS_DOMAIN_BORDER_FLAG))
                    {
//                        SA_ASSERT(0.0 == cface_base(k));
                        continue;
                    }
                    const int loc = agg_map_id_glob_to_AE(dofs[k], i, agg_part_rels);
                    SA_ASSERT(0 <= loc && loc < ndofs);
                    const int ldof = map[loc];
                    SA_ASSERT(0 <= ldof && ldof < interior_dofs);
                    SA_ASSERT(cface_offset + l < bdr_dofs);
                    const double Bval = diag(ldof) * cface_base(k);
                    SA_ASSERT(0.0 == (*Aii)(ldof, interior_dofs + cface_offset + l));
                    (*Aii)(ldof, interior_dofs + cface_offset + l) = Bval;
                    SA_ASSERT(0.0 == (*Aii)(interior_dofs + cface_offset + l, ldof));
                    (*Aii)(interior_dofs + cface_offset + l, ldof) = Bval;
                }
                for (int k=0; k < cface_dofs; ++k)
                {
                    const Vector cface_base1(interp_data.cfaces_bases[cface]->GetData() + k * num_dofs, num_dofs);
                    double Xval = 0.0;
                    for (int m=0; m < num_dofs; ++m)
                    {
                        if (SA_IS_SET_A_FLAG(agg_part_rels.agg_flags[dofs[m]], AGG_ON_ESS_DOMAIN_BORDER_FLAG))
                        {
//                            SA_ASSERT(0.0 == cface_base(m));
//                            SA_ASSERT(0.0 == cface_base1(m));
                            continue;
                        }
                        const int loc = agg_map_id_glob_to_AE(dofs[m], i, agg_part_rels);
                        SA_ASSERT(0 <= loc && loc < ndofs);
                        const int ldof = map[loc];
                        SA_ASSERT(0 <= ldof && ldof < interior_dofs);
                        Xval += diag(ldof) * cface_base(m) * cface_base1(m);
                    }
                    SA_ASSERT(0.0 == X0(interior_dofs + cface_offset + l, cface_offset + k));
                    X0(interior_dofs + cface_offset + l, cface_offset + k) = -Xval;
                }
            }
            cface_offset += cface_dofs;
        }
        SA_ASSERT(cface_offset == bdr_dofs);
        delete D;

        Aii->Invert();
        interp_data.invAii[i] = Aii;
        Aii = NULL;
        interp_data.invAiiAib[i] = new DenseMatrix(interp_data.invAii[i]->Height(), X0.Width());
        Mult(*interp_data.invAii[i], X0, *interp_data.invAiiAib[i]);
        interp_data.schurs[i] = new DenseMatrix(X0.Width(), interp_data.invAiiAib[i]->Width());
        MultAtB(X0, *interp_data.invAiiAib[i], *interp_data.schurs[i]);
        interp_data.schurs[i]->Neg();
        SA_ASSERT(interp_data.schurs[i]->Height() == interp_data.schurs[i]->Width() &&
                  interp_data.schurs[i]->Height() == bdr_dofs);
        SA_ASSERT(interp_data.invAii[i]->Height() == interp_data.invAii[i]->Width() &&
                  interp_data.invAii[i]->Height() == interior_dofs + bdr_dofs);
        SA_ASSERT(interp_data.invAiiAib[i]->Height() == interior_dofs + bdr_dofs &&
                  interp_data.invAiiAib[i]->Width() == bdr_dofs);
    }
    delete [] interp_data.AEs_stiffm;
    interp_data.AEs_stiffm = NULL;
}

/**
    Obtains the mortar condensed element rhs vectors and maintains some additional data for later use.
*/
static inline
void mortar_condensed_local_rhs(interp_data_t& interp_data, const agg_partitioning_relations_t& agg_part_rels,
                                ElementMatrixProvider *elem_data)
{
    SA_ASSERT(!interp_data.pre_rhs);
    SA_ASSERT(!interp_data.rhs);
    SA_ASSERT(interp_data.invAii);
    SA_ASSERT(interp_data.invAiiAib);
    SA_ASSERT(elem_data);

    const int nparts = agg_part_rels.nparts;

    interp_data.pre_rhs = new DenseMatrix*[nparts];
    interp_data.rhs = new DenseMatrix*[nparts];

    SparseMatrix *AE_rhs;
    Vector src;
    for (int i=0; i < nparts; ++i)
    {
        AE_rhs = elem_data->BuildAEStiff(i);
        SA_ASSERT(AE_rhs);
        SA_ASSERT(AE_rhs->Finalized());
        SA_ASSERT(AE_rhs->Width() == AE_rhs->Height());
        SA_ASSERT(AE_rhs->Height() == agg_part_rels.AE_to_dof->RowSize(i));
        SA_ASSERT(AE_rhs->Height() == AE_rhs->NumNonZeroElems());
        SA_ASSERT(interp_data.invAii[i]);
        const int ndofs = AE_rhs->Height();
        const double * const data = AE_rhs->GetData();
        src.SetSize(interp_data.invAii[i]->Width());
        src = 0.0;
        int intdofs_ctr = 0;
        for (int j=0; j < ndofs; ++j)
        {
            const int gj = agg_num_col_to_glob(*agg_part_rels.AE_to_dof, i, j);
            SA_ASSERT(0 <= gj && gj < agg_part_rels.ND);
            if (!SA_IS_SET_A_FLAG(agg_part_rels.agg_flags[gj], AGG_ON_ESS_DOMAIN_BORDER_FLAG))
            {
                SA_ASSERT(intdofs_ctr < src.Size());
                SA_ASSERT(intdofs_ctr < interp_data.celements_cdofs_offsets[i+1] - interp_data.celements_cdofs_offsets[i]);
                SA_ASSERT(0.0 == src(intdofs_ctr));
                src(intdofs_ctr++) = data[j];
            }
        }
        SA_ASSERT(intdofs_ctr == interp_data.celements_cdofs_offsets[i+1] - interp_data.celements_cdofs_offsets[i]);
        SA_ASSERT(src.Size() == intdofs_ctr + interp_data.invAiiAib[i]->Width());
        delete AE_rhs;
        interp_data.pre_rhs[i] = new DenseMatrix(interp_data.invAii[i]->Height(), 1);
        Vector tmp(interp_data.pre_rhs[i]->GetData(), interp_data.invAii[i]->Height());
        interp_data.invAii[i]->Mult(src, tmp);
        delete interp_data.invAii[i];
        interp_data.invAii[i] = NULL;
        interp_data.rhs[i] = new DenseMatrix(interp_data.invAiiAib[i]->Width(), 1);
        tmp.SetDataAndSize(interp_data.rhs[i]->GetData(), interp_data.invAiiAib[i]->Width());
        interp_data.invAiiAib[i]->MultTranspose(src, tmp);
        interp_data.rhs[i]->Neg();
    }
    delete [] interp_data.invAii;
    interp_data.invAii = NULL;
}

HypreParVector *mortar_assemble_condensed_rhs(interp_data_t& interp_data,
                                              const agg_partitioning_relations_t& agg_part_rels,
                                              ElementMatrixProvider *elem_data)
{
    mortar_condensed_local_rhs(interp_data, agg_part_rels, elem_data);
    SA_ASSERT(interp_data.rhs);
    SA_ASSERT(interp_data.cfaces_cdofs_offsets);
    SA_ASSERT(agg_part_rels.cface_TruecDof_cDof);

    // Assemble the local (on processor) portion of the rhs.
    HypreParVector lrhs(*agg_part_rels.cface_TruecDof_cDof);
    SA_ASSERT(lrhs.Size() == interp_data.cfaces_cdofs_offsets[interp_data.num_cfaces]);
    lrhs = 0.0;
    for (int i=0; i < interp_data.nparts; ++i)
    {
        SA_ASSERT(interp_data.rhs[i]);
        const int ncfaces = agg_part_rels.AE_to_cface->RowSize(i);
        const int * const cfaces = agg_part_rels.AE_to_cface->GetRow(i);
        int brow = 0;
        for (int j=0; j < ncfaces; ++j)
        {
            const int cface = cfaces[j];
            SA_ASSERT(0 <= cface && cface < interp_data.num_cfaces);
            const int ndofs = interp_data.cfaces_cdofs_offsets[cface+1] - interp_data.cfaces_cdofs_offsets[cface];
            const int offset = interp_data.cfaces_cdofs_offsets[cface];
            SA_ASSERT(offset + ndofs <= lrhs.Size());
            for (int k=0; k < ndofs; ++k, ++brow)
                lrhs(offset + k) += (*interp_data.rhs[i])(brow, 0);
        }
        SA_ASSERT(interp_data.rhs[i]->Height() == brow);
        delete interp_data.rhs[i];
        interp_data.rhs[i] = NULL;
    }
    delete [] interp_data.rhs;
    interp_data.rhs = NULL;

    // Assemble the global (in terms of true DoFs on cfaces) rhs.
    HypreParVector *trhs = new HypreParVector(*agg_part_rels.cface_TruecDof_cDof, 1);
    agg_part_rels.cface_TruecDof_cDof->Mult(lrhs, *trhs);
    mbox_make_owner_partitioning(*trhs);

    return trhs;
}

void mortar_discretization(tg_data_t& tg_data, agg_partitioning_relations_t& agg_part_rels,
                           ElementMatrixProvider *elem_data, const Array<Vector *> *face_targets)
{
    tg_data.elem_data = elem_data;
    tg_data.doing_spectral = true;

    // Prepare "identity" element basis and "interior" stiffness matrix, removing all essential BCs' DoFs.
    nonconf_eliminate_boundary_full_element_basis(*tg_data.interp_data, agg_part_rels, elem_data);

    // Obtain coarse/agglomerated face basis.
    ContribTent cfaces_bases_contrib(agg_part_rels.ND);
    SA_ASSERT(NULL == tg_data.interp_data->cfaces_bases);
    if (NULL == face_targets)
        tg_data.interp_data->cfaces_bases = cfaces_bases_contrib.contrib_cfaces_const(agg_part_rels);
    else
        tg_data.interp_data->cfaces_bases = cfaces_bases_contrib.contrib_cfaces_targets(agg_part_rels, *face_targets);
    tg_data.interp_data->num_cfaces = agg_part_rels.num_cfaces;
    tg_data.interp_data->coarse_truedof_offset = cfaces_bases_contrib.get_coarse_truedof_offset();

    // Obtain the local (on CPU) "interpolant" and "restriction".
    cfaces_bases_contrib.insert_from_cfaces_celems_bases(tg_data.interp_data->nparts, tg_data.interp_data->num_cfaces,
                                                         tg_data.interp_data->cut_evects_arr,
                                                         tg_data.interp_data->cfaces_bases, agg_part_rels, false);
    for (int i=0; i < agg_part_rels.nparts; ++i)
        delete tg_data.interp_data->cut_evects_arr[i];
    delete [] tg_data.interp_data->cut_evects_arr;
    tg_data.interp_data->cut_evects_arr = NULL;

    delete tg_data.ltent_interp;
    delete tg_data.ltent_restr;
    delete tg_data.tent_interp;
    tg_data.tent_interp = NULL;
    delete tg_data.interp;
    delete tg_data.restr;
    tg_data.ltent_interp = cfaces_bases_contrib.contrib_tent_finalize();
    tg_data.ltent_restr = NULL;
    tg_data.interp_data->celements_cdofs = cfaces_bases_contrib.get_celements_cdofs();
    tg_data.interp_data->celements_cdofs_offsets = cfaces_bases_contrib.get_celements_cdofs_offsets();
    tg_data.interp_data->cfaces_truecdofs_offsets = cfaces_bases_contrib.get_cfaces_truecdofs_offsets();
    tg_data.interp_data->cfaces_cdofs_offsets = cfaces_bases_contrib.get_cfaces_cdofs_offsets();

    // Make the "interpolant" and "restriction" global via hypre.
    tg_data.interp_data->tent_interp_offsets.SetSize(0);
    tg_data.interp = interp_global_tent_assemble(agg_part_rels, *tg_data.interp_data,
                                                 tg_data.ltent_interp);
    tg_data.restr = tg_data.interp->Transpose();

    // Obtain the remaining necessary relations.
    agg_create_cface_cDof_TruecDof_relations(agg_part_rels, tg_data.interp_data->coarse_truedof_offset,
                                             tg_data.interp_data->cfaces_bases, tg_data.interp_data->celements_cdofs, false);

    mortar_condensed_local_matrices(*tg_data.interp_data, agg_part_rels);

    for (int i=0; i < agg_part_rels.num_cfaces; ++i)
        delete tg_data.interp_data->cfaces_bases[i];
    delete [] tg_data.interp_data->cfaces_bases;
    tg_data.interp_data->cfaces_bases = NULL;

    // Construct the operator as though it is a "coarse" matrix.
    delete tg_data.Ac;
    tg_data.Ac = nonconf_assemble_schur_matrix(*tg_data.interp_data, agg_part_rels,
                                                      *agg_part_rels.cface_cDof_TruecDof);
    for (int i=0; i < agg_part_rels.nparts; ++i)
        delete tg_data.interp_data->schurs[i];
    delete [] tg_data.interp_data->schurs;
    tg_data.interp_data->schurs = NULL;
}

HypreParVector *mortar_reverse_condensation(const HypreParVector& mortar_sol, const tg_data_t& tg_data,
                                            const agg_partitioning_relations_t& agg_part_rels)
{
    SA_ASSERT(agg_part_rels.cface_cDof_TruecDof);
    SA_ASSERT(tg_data.interp);
    SA_ASSERT(tg_data.interp_data);

    const interp_data_t& interp_data = *tg_data.interp_data;
    HypreParVector lmsol(*agg_part_rels.cface_cDof_TruecDof, 1);
    HypreParVector rec(*tg_data.interp);

    SA_ASSERT(mortar_sol.Size() == agg_part_rels.cface_cDof_TruecDof->Width());
    SA_ASSERT(rec.Size() == interp_data.celements_cdofs + interp_data.cfaces_truecdofs_offsets[interp_data.num_cfaces]);
    SA_ASSERT(mortar_sol.Size() == interp_data.cfaces_truecdofs_offsets[interp_data.num_cfaces]);
    SA_ASSERT(lmsol.Size() == interp_data.cfaces_cdofs_offsets[interp_data.num_cfaces]);

    agg_part_rels.cface_cDof_TruecDof->Mult(const_cast<HypreParVector&>(mortar_sol), lmsol);

    // Copy the face values in the true dof sense.
    for (int i = 0; i < mortar_sol.Size(); ++i)
        rec(interp_data.celements_cdofs + i) = mortar_sol(i);

    // Update (backward substitute) the interior portions of the solution vector.
    SA_ASSERT(interp_data.pre_rhs);
    SA_ASSERT(interp_data.invAiiAib);
    Vector bdr;
    Vector total;
    for (int i=0; i < interp_data.nparts; ++i)
    {
        SA_ASSERT(interp_data.pre_rhs[i]);
        SA_ASSERT(interp_data.invAiiAib[i]);
        SA_ASSERT(interp_data.celements_cdofs_offsets[i] < interp_data.celements_cdofs);
        SA_ASSERT(interp_data.celements_cdofs_offsets[i+1] <= interp_data.celements_cdofs);
        const int interior_size = interp_data.celements_cdofs_offsets[i+1] - interp_data.celements_cdofs_offsets[i];
        const int total_size = interp_data.invAiiAib[i]->Height();
        const int bdr_size = interp_data.invAiiAib[i]->Width();
        SA_ASSERT(total_size == interior_size + bdr_size);
        SA_ASSERT(total_size == interp_data.pre_rhs[i]->Height());
        Vector rec_interior(rec.GetData() + interp_data.celements_cdofs_offsets[i], interior_size);
        total.SetSize(total_size);
        total = interp_data.pre_rhs[i]->GetData();

        // Add the contribution of the actual backward substitution.

        // Localize the face solution on the faces of the element in the respective order.
        bdr.SetSize(bdr_size);
        const int ncfaces = agg_part_rels.AE_to_cface->RowSize(i);
        const int * const cfaces = agg_part_rels.AE_to_cface->GetRow(i);
        int brow = 0;
        for (int j=0; j < ncfaces; ++j)
        {
            const int cface = cfaces[j];
            SA_ASSERT(0 <= cface && cface < interp_data.num_cfaces);
            const int ndofs = interp_data.cfaces_cdofs_offsets[cface+1] - interp_data.cfaces_cdofs_offsets[cface];
            const int offset = interp_data.cfaces_cdofs_offsets[cface];
            SA_ASSERT(offset + ndofs <= lmsol.Size());
            SA_ASSERT(brow + ndofs <= bdr.Size());
            for (int k=0; k < ndofs; ++k, ++brow)
                bdr(brow) = lmsol(offset + k);
        }
        SA_ASSERT(bdr.Size() == brow);

        // Obtain the total contribution.
        interp_data.invAiiAib[i]->AddMult_a(-1.0, bdr, total);

        // Take only the interior of the total contribution.
        for (int j=0; j < interior_size; ++j)
            rec_interior(j) = total(j);
    }

    // Obtain the H^1 function in true dofs.
    HypreParVector *sol = new HypreParVector(*tg_data.interp, 1);
    SA_ASSERT(sol->Size() == agg_part_rels.Dof_TrueDof->Width());
    tg_data.interp->Mult(rec, *sol);
    mbox_make_owner_partitioning(*sol);

    return sol;
}

HypreParVector *mortar_assemble_schur_rhs(const interp_data_t& interp_data,
                    const agg_partitioning_relations_t& agg_part_rels, const Vector& rhs)
{
    SA_ASSERT(interp_data.invAiiAib);
    SA_ASSERT(interp_data.celements_cdofs_offsets);
    SA_ASSERT(interp_data.cfaces_cdofs_offsets);
    SA_ASSERT(agg_part_rels.cface_TruecDof_cDof);
    SA_ASSERT(rhs.Size() == interp_data.celements_cdofs + agg_part_rels.cface_TruecDof_cDof->Height());

    // Assemble the local (on processor) portion of the rhs.
    HypreParVector lrhs(*agg_part_rels.cface_TruecDof_cDof);
    SA_ASSERT(lrhs.Size() == interp_data.cfaces_cdofs_offsets[interp_data.num_cfaces]);
    lrhs = 0.0;
    Vector bdr;
    Vector total;
    for (int i=0; i < interp_data.nparts; ++i)
    {
        SA_ASSERT(interp_data.invAiiAib[i]);
        SA_ASSERT(interp_data.celements_cdofs_offsets[i] < interp_data.celements_cdofs);
        SA_ASSERT(interp_data.celements_cdofs_offsets[i+1] <= interp_data.celements_cdofs);
        const int interior_size = interp_data.celements_cdofs_offsets[i+1] - interp_data.celements_cdofs_offsets[i];
        const int total_size = interp_data.invAiiAib[i]->Height();
        const int bdr_size = interp_data.invAiiAib[i]->Width();
        SA_ASSERT(total_size == interior_size + bdr_size);
        const Vector rhs_interior(rhs.GetData() + interp_data.celements_cdofs_offsets[i], interior_size);
        total.SetSize(total_size);
        total = 0.0;
        for (int j=0; j < interior_size; ++j)
            total(j) = rhs_interior(j);

        bdr.SetSize(bdr_size);
        interp_data.invAiiAib[i]->MultTranspose(total, bdr);
        const int ncfaces = agg_part_rels.AE_to_cface->RowSize(i);
        const int * const cfaces = agg_part_rels.AE_to_cface->GetRow(i);
        int brow = 0;
        for (int j=0; j < ncfaces; ++j)
        {
            const int cface = cfaces[j];
            SA_ASSERT(0 <= cface && cface < interp_data.num_cfaces);
            const int ndofs = interp_data.cfaces_cdofs_offsets[cface+1] - interp_data.cfaces_cdofs_offsets[cface];
            const int offset = interp_data.cfaces_cdofs_offsets[cface];
            SA_ASSERT(offset + ndofs <= lrhs.Size());
            for (int k=0; k < ndofs; ++k, ++brow)
                lrhs(offset + k) -= bdr(brow);
        }
        SA_ASSERT(bdr.Size() == brow);
    }

    // Assemble the global (in terms of true DoFs on cfaces) rhs.
    HypreParVector *trhs = new HypreParVector(*agg_part_rels.cface_TruecDof_cDof, 1);
    agg_part_rels.cface_TruecDof_cDof->Mult(lrhs, *trhs);
    mbox_make_owner_partitioning(*trhs);

    return trhs;
}

void mortar_schur_recover(const interp_data_t& interp_data,
                    const agg_partitioning_relations_t& agg_part_rels,
                    const Vector& rhs, const Vector& facev, Vector& x)
{
    SA_ASSERT(agg_part_rels.cface_cDof_TruecDof);
    SA_ASSERT(interp_data.invAii);
    SA_ASSERT(interp_data.invAiiAib);
    SA_ASSERT(interp_data.celements_cdofs_offsets);
    SA_ASSERT(interp_data.cfaces_cdofs_offsets);
    SA_ASSERT(rhs.Size() == interp_data.celements_cdofs + agg_part_rels.cface_cDof_TruecDof->Width());
    SA_ASSERT(facev.Size() == agg_part_rels.cface_cDof_TruecDof->Width());

    // Get the face vector from true face DoFs to local (repeated) DoFs.
    HypreParVector lfacev(*agg_part_rels.cface_cDof_TruecDof, 1);
    agg_part_rels.cface_cDof_TruecDof->Mult(facev, lfacev);

    // Allocate the global solution vector (in all true DoFs, including the interiors)
    // and copy the available face solution from facev.
    SA_ASSERT(x.Size() == rhs.Size());
    SA_ASSERT(facev.Size() + interp_data.celements_cdofs == x.Size());
    for (int i = 0; i < facev.Size(); ++i)
        x(i + interp_data.celements_cdofs) = facev(i);

    // Update (backward substitute) the interior portions of the solution vector.
    Vector bdr;
    Vector totalr;
    Vector total;
    for (int i=0; i < interp_data.nparts; ++i)
    {
        SA_ASSERT(interp_data.invAii[i]);
        SA_ASSERT(interp_data.invAiiAib[i]);
        SA_ASSERT(interp_data.celements_cdofs_offsets[i] < interp_data.celements_cdofs);
        SA_ASSERT(interp_data.celements_cdofs_offsets[i+1] <= interp_data.celements_cdofs);
        const int interior_size = interp_data.celements_cdofs_offsets[i+1] - interp_data.celements_cdofs_offsets[i];
        const int total_size = interp_data.invAiiAib[i]->Height();
        const int bdr_size = interp_data.invAiiAib[i]->Width();
        SA_ASSERT(total_size == interior_size + bdr_size);
        Vector x_interior(x.GetData() + interp_data.celements_cdofs_offsets[i], interior_size);

        const Vector rhs_interior(rhs.GetData() + interp_data.celements_cdofs_offsets[i], interior_size);
        totalr.SetSize(total_size);
        totalr = 0.0;
        for (int j=0; j < interior_size; ++j)
            totalr(j) = rhs_interior(j);

        total.SetSize(total_size);
        // Put the direct inversion of the portion of the rhs.
        interp_data.invAii[i]->Mult(totalr, total);

        // Add the contribution of the actual backward substitution.

        // Localize the face solution on the faces of the element in the respective order.
        bdr.SetSize(bdr_size);
        const int ncfaces = agg_part_rels.AE_to_cface->RowSize(i);
        const int * const cfaces = agg_part_rels.AE_to_cface->GetRow(i);
        int brow = 0;
        for (int j=0; j < ncfaces; ++j)
        {
            const int cface = cfaces[j];
            SA_ASSERT(0 <= cface && cface < interp_data.num_cfaces);
            const int ndofs = interp_data.cfaces_cdofs_offsets[cface+1] - interp_data.cfaces_cdofs_offsets[cface];
            const int offset = interp_data.cfaces_cdofs_offsets[cface];
            SA_ASSERT(offset + ndofs <= lfacev.Size());
            SA_ASSERT(brow + ndofs <= bdr.Size());
            for (int k=0; k < ndofs; ++k, ++brow)
                bdr(brow) = lfacev(offset + k);
        }
        SA_ASSERT(bdr.Size() == brow);

        // Obtain the final contribution.
        interp_data.invAiiAib[i]->AddMult_a(-1.0, bdr, total);

        // Take only the interior of the total contribution.
        for (int j=0; j < interior_size; ++j)
            x_interior(j) = total(j);
    }
}

void MortarSchurSolver::Mult(const mfem::Vector &x, mfem::Vector &y) const
{
    // Elimination
    HypreParVector *schur_rhs = mortar_assemble_schur_rhs(interp_data,
                                                          agg_part_rels, x);
    // Solve Schur system
    HypreParVector eb(*schur_rhs);
    eb = 0.0;
    solver.Mult(*schur_rhs, eb);
    delete schur_rhs;

    // Backward substitution
    mortar_schur_recover(interp_data, agg_part_rels, x, eb, y);
}

} // namespace saamge
