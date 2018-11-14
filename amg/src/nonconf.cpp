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
#include "nonconf.hpp"
#include <mfem.hpp>
#include "xpacks.hpp"
#include "mbox.hpp"
#include "aggregates.hpp"
#include "interp.hpp"
#include "helpers.hpp"

namespace saamge
{
using namespace mfem;

/**
    Obtains the Schur complements (on interface faces) of the modified coarse/agglomerate element matrix,
    when invoking first coarsening.

    1/delta is the scaling parameter for the interface term.

    Fills in arrays in the interpolation data. They will be freed with the interpolation data.
*/
static inline
void nonconf_ip_first_coarse_schur_matrices(interp_data_t& interp_data,
                  const agg_partitioning_relations_t& agg_part_rels, const Vector * const *evals,
                  double delta, bool onlytheblocks=false, bool full_space=false)
{
    SA_ASSERT(NULL != interp_data.cfaces_bases);
    const DenseMatrix * const * const cfaces_bases = interp_data.cfaces_bases;

    // Assemble the blocks of the modified coarse/agglomerate element matrix and obtain the Schur complement
    // on coarse/agglomerate faces for each coarse/agglomerate element.
    SA_ASSERT(NULL == interp_data.Aii);
    SA_ASSERT(NULL == interp_data.Abb);
    SA_ASSERT(NULL == interp_data.Aib);
    SA_ASSERT(NULL == interp_data.invAii);
    SA_ASSERT(NULL == interp_data.invAiiAib);
    SA_ASSERT(NULL == interp_data.AbiinvAii);
    SA_ASSERT(NULL == interp_data.schurs);
    if (onlytheblocks)
    {
        interp_data.Aii = (Matrix **)new DenseMatrix*[agg_part_rels.nparts];
        interp_data.Abb = (Matrix **)new DenseMatrix*[agg_part_rels.nparts];
        interp_data.Aib = (Matrix **)new DenseMatrix*[agg_part_rels.nparts];
    } else
    {
        interp_data.invAii = new DenseMatrix*[agg_part_rels.nparts];
        interp_data.invAiiAib = new DenseMatrix*[agg_part_rels.nparts];
        interp_data.AbiinvAii = new DenseMatrix*[agg_part_rels.nparts];
        interp_data.schurs = new DenseMatrix*[agg_part_rels.nparts];
    }
    for (int i=0; i < agg_part_rels.nparts; ++i)
    {
        SA_ASSERT(NULL != interp_data.cut_evects_arr[i]);
        SA_ASSERT(NULL != interp_data.rhs_matrices_arr[i]);
        SA_ASSERT(interp_data.rhs_matrices_arr[i]->Height() == interp_data.rhs_matrices_arr[i]->Width());
        SA_ASSERT(interp_data.rhs_matrices_arr[i]->Height() == agg_part_rels.AE_to_dof->RowSize(i));
        const int num_cfaces = agg_part_rels.AE_to_cface->RowSize(i);
        const int * const cfaces = agg_part_rels.AE_to_cface->GetRow(i);
        const int interior_size = interp_data.cut_evects_arr[i]->Width();
        int bdr_size = 0;
        for (int j=0; j < num_cfaces; ++j)
        {
            SA_ASSERT(cfaces_bases[cfaces[j]]);
            bdr_size += cfaces_bases[cfaces[j]]->Width();
        }
        DenseMatrix *Aii = new DenseMatrix(interior_size);
        DenseMatrix *Abb = new DenseMatrix(bdr_size);
        DenseMatrix *Aib = new DenseMatrix(interior_size, bdr_size);

        // Loop over the coarse/agglomerate faces of the current agglomerate/coarse element and
        // obtain the respective entries of the matrices.
        const Vector diag(interp_data.rhs_matrices_arr[i]->GetData(), interp_data.rhs_matrices_arr[i]->Height());
        int cface_offset = 0;
        for (int j=0; j < num_cfaces; ++j)
        {
            const int cface = cfaces[j];
            SA_ASSERT(0 <= cface && cface < agg_part_rels.num_cfaces);
            const int cface_basis_size = cfaces_bases[cface]->Width();
            SA_ASSERT(cface_offset + cface_basis_size <= bdr_size);
            SA_ASSERT(cfaces_bases[cface]->Height() == agg_part_rels.cfaces_dof_size[cface]);
            DenseMatrix cut_evects_cface;
            Vector cface_diag;
            SA_ASSERT(agg_part_rels.cface_to_dof->RowSize(cface) == agg_part_rels.cfaces_dof_size[cface]);

            agg_restrict_to_agg_enforce(i, agg_part_rels, agg_part_rels.cfaces_dof_size[cface],
                                        agg_part_rels.cface_to_dof->GetRow(cface),
                                        *(interp_data.cut_evects_arr[i]), cut_evects_cface);
            agg_restrict_vec_to_agg_enforce(i, agg_part_rels, agg_part_rels.cfaces_dof_size[cface],
                                            agg_part_rels.cface_to_dof->GetRow(cface),
                                            diag, cface_diag);

            for (int n=0; n < cface_basis_size; ++n)
            {
                const Vector pn(const_cast<double *>(cfaces_bases[cface]->GetColumn(n)), agg_part_rels.cfaces_dof_size[cface]);
                for (int m=0; m < cface_basis_size; ++m)
                {
                    const Vector pm(const_cast<double *>(cfaces_bases[cface]->GetColumn(m)), agg_part_rels.cfaces_dof_size[cface]);
                    (*Abb)(cface_offset + m, cface_offset + n) = (1./delta) * mbox_energy_inner_prod_diag(cface_diag, pm, pn);
                }
                for (int l=0; l < interior_size; ++l)
                {
                    const Vector ql(cut_evects_cface.GetColumn(l), agg_part_rels.cfaces_dof_size[cface]);
                    (*Aib)(l, cface_offset + n) = -(1./delta) * mbox_energy_inner_prod_diag(cface_diag, ql, pn);
                }
            }

            for (int n=0; n < interior_size; ++n)
            {
                const Vector qn(cut_evects_cface.GetColumn(n), agg_part_rels.cfaces_dof_size[cface]);
                for (int m=0; m < interior_size; ++m)
                {
                    const Vector qm(cut_evects_cface.GetColumn(m), agg_part_rels.cfaces_dof_size[cface]);
                    (*Aii)(m, n) += (1./delta) * mbox_energy_inner_prod_diag(cface_diag, qm, qn);
                }
            }

            cface_offset += cface_basis_size;
        }

        if (full_space)
        {
            SA_ASSERT(interp_data.AEs_stiffm[i]);
            for (int l=0; l < interior_size; ++l)
            {
                const Vector ql(interp_data.cut_evects_arr[i]->GetColumn(l), interp_data.cut_evects_arr[i]->Height());
                for (int k=0; k < interior_size; ++k)
                {
                    const Vector qk(interp_data.cut_evects_arr[i]->GetColumn(k), interp_data.cut_evects_arr[i]->Height());
                    (*Aii)(k, l) += mbox_energy_inner_prod_sparse(*interp_data.AEs_stiffm[i], ql, qk);
                }
            }
        } else
        {
            SA_ASSERT(NULL != evals);
            SA_ASSERT(NULL != evals[i]);
            SA_ASSERT(interior_size <= evals[i]->Size());
            for (int l=0; l < interior_size; ++l)
                (*Aii)(l, l) += evals[i]->Elem(l);
        }

        // Compute the Schur complement.
        if (onlytheblocks)
        {
            interp_data.Aii[i] = Aii;
            interp_data.Abb[i] = Abb;
            interp_data.Aib[i] = Aib;
        } else
        {
            DenseMatrix *invAii = new DenseMatrix;
            xpacks_calc_spd_inverse_dense(*Aii, *invAii);
            DenseMatrix *invAiiAib = new DenseMatrix;
            invAiiAib->SetSize(invAii->Height(), Aib->Width());
            Mult(*invAii, *Aib, *invAiiAib);
            DenseMatrix AbiinvAiiAib;
            AbiinvAiiAib.SetSize(Aib->Width(), invAiiAib->Width());
            MultAtB(*Aib, *invAiiAib, AbiinvAiiAib);
            *Abb -= AbiinvAiiAib;
            interp_data.schurs[i] = Abb;
            interp_data.invAii[i] = invAii;
            interp_data.invAiiAib[i] = invAiiAib;
            interp_data.AbiinvAii[i] = new DenseMatrix;
            interp_data.AbiinvAii[i]->SetSize(Aib->Width(), invAii->Width());
            MultAtB(*Aib, *invAii, *(interp_data.AbiinvAii[i]));
            delete Aib;
            delete Aii;
        }
    }
}

/**
    Once all coarse element Schur complement matrices (on cfaces) are obtained,
    the global Schur complement matrix is assembled by a standard procedure in this
    routine. The procedure is standard but adapted for the particular data structures and organization.

    The returned global Schur complement matrix must be freed by the caller.
*/
static inline
HypreParMatrix *nonconf_assemble_coarse_schur_matrix(const interp_data_t& interp_data,
                    const agg_partitioning_relations_t& agg_part_rels,
                    const HypreParMatrix& cface_cDof_TruecDof)
{
    SA_ASSERT(agg_part_rels.AE_to_cface);
    SA_ASSERT(interp_data.schurs);

    // Assemble the local (on processor) portion of the matrix.
    const int nloc_cfacecdofs = interp_data.cfaces_cdofs_offsets[interp_data.num_cfaces];
    SparseMatrix lschur(nloc_cfacecdofs, nloc_cfacecdofs);
    SA_ASSERT(nloc_cfacecdofs == cface_cDof_TruecDof.GetNumRows());

    for (int i=0; i < interp_data.nparts; ++i)
    {
        int bcol = 0;
        const DenseMatrix &elem_matr = *interp_data.schurs[i];
        SA_ASSERT(&elem_matr);
        const int ncfaces = agg_part_rels.AE_to_cface->RowSize(i);
        const int * const cfaces = agg_part_rels.AE_to_cface->GetRow(i);
        for (int j=0; j < ncfaces; ++j)
        {
            const int cfacej = cfaces[j];
            SA_ASSERT(0 <= cfacej && cfacej < interp_data.num_cfaces);
            const int ndofsj = interp_data.cfaces_bases[cfacej]->Width();
            SA_ASSERT(interp_data.cfaces_cdofs_offsets[cfacej+1] - interp_data.cfaces_cdofs_offsets[cfacej] == ndofsj);
            const int offsetj = interp_data.cfaces_cdofs_offsets[cfacej];
            SA_ASSERT(offsetj + ndofsj <= nloc_cfacecdofs);
            SA_ASSERT(bcol + ndofsj <= elem_matr.Width());
            for (int l=0; l < ndofsj; ++l, ++bcol)
            {
                int brow = 0;
                for (int m=0; m < ncfaces; ++m)
                {
                    const int cfacem = cfaces[m];
                    SA_ASSERT(0 <= cfacem && cfacem < interp_data.num_cfaces);
                    const int ndofsm = interp_data.cfaces_bases[cfacem]->Width();
                    SA_ASSERT(interp_data.cfaces_cdofs_offsets[cfacem+1] - interp_data.cfaces_cdofs_offsets[cfacem] == ndofsm);
                    const int offsetm = interp_data.cfaces_cdofs_offsets[cfacem];
                    SA_ASSERT(offsetm + ndofsm <= nloc_cfacecdofs);
                    SA_ASSERT(brow + ndofsm <= elem_matr.Height());
                    for (int k=0; k < ndofsm; ++k, ++brow)
                        lschur.Add(offsetm + k, offsetj + l, elem_matr(brow, bcol));
                }
                SA_ASSERT(elem_matr.Height() == brow);
            }
        }
        SA_ASSERT(elem_matr.Width() == bcol);
    }
    lschur.Finalize();

    // Assemble the global Schur matrix using the local ones and the cface_cDof_TruecDof relation.
    HypreParMatrix tent_gschur(PROC_COMM, cface_cDof_TruecDof.GetGlobalNumRows(), cface_cDof_TruecDof.GetGlobalNumRows(),
                               cface_cDof_TruecDof.GetRowStarts(), cface_cDof_TruecDof.GetRowStarts(), &lschur);
    HypreParMatrix *gschur = RAP(&tent_gschur, &cface_cDof_TruecDof);
    mbox_make_owner_rowstarts_colstarts(*gschur);

    return gschur;
}

/**
    For testing. Assembles global coarse interior penalty matrix using dense local matrices.

    The returned global matrix must be freed by the caller.
*/
static inline
HypreParMatrix *nonconf_ip_dense_assemble_coarse_matrix(const interp_data_t& interp_data,
                    const agg_partitioning_relations_t& agg_part_rels,
                    const HypreParMatrix& cDof_TruecDof)
{
    SA_ASSERT(agg_part_rels.AE_to_cface);
    SA_ASSERT(interp_data.Aii);
    SA_ASSERT(interp_data.Abb);
    SA_ASSERT(interp_data.Aib);

    // Assemble the local (on processor) portion of the matrix.
    const int nloc_cdofs = interp_data.celements_cdofs + interp_data.cfaces_cdofs_offsets[interp_data.num_cfaces];
    SparseMatrix lmat(nloc_cdofs, nloc_cdofs);
    SA_ASSERT(nloc_cdofs == cDof_TruecDof.GetNumRows());

    for (int i=0; i < interp_data.nparts; ++i)
    {
        const DenseMatrix &Aii = *dynamic_cast<DenseMatrix *>(interp_data.Aii[i]);
        const DenseMatrix &Abb = *dynamic_cast<DenseMatrix *>(interp_data.Abb[i]);
        const DenseMatrix &Aib = *dynamic_cast<DenseMatrix *>(interp_data.Aib[i]);
        SA_ASSERT(&Aii);
        SA_ASSERT(&Abb);
        SA_ASSERT(&Aib);
        SA_ASSERT(Aii.Width() == Aii.Height());
        SA_ASSERT(Aii.Width() == Aib.Height());
        SA_ASSERT(Abb.Width() == Abb.Height());
        SA_ASSERT(Aib.Width() == Abb.Height());
        SA_ASSERT(Aii.Width() == interp_data.cut_evects_arr[i]->Width());

        int bcol;
        const int ncintdofs = Aii.Width();
        const int ncfaces = agg_part_rels.AE_to_cface->RowSize(i);
        const int * const cfaces = agg_part_rels.AE_to_cface->GetRow(i);
        for (int k=0; k < ncintdofs; ++k)
        {
            const int gk = interp_data.celements_cdofs_offsets[i] + k;
            SA_ASSERT(0 <= gk && gk < nloc_cdofs);
            for (int j=0; j < ncintdofs; ++j)
            {
                const int gj = interp_data.celements_cdofs_offsets[i] + j;
                SA_ASSERT(0 <= gj && gj < nloc_cdofs);
                double &rjk = lmat.SearchRow(gj, gk);
                SA_ASSERT(0.0 == rjk);
                rjk += Aii(j, k);
//                lmat.Add(gj, gk, Aii(j, k));
            }

            bcol = 0;
            for (int j=0; j < ncfaces; ++j)
            {
                const int cfacej = cfaces[j];
                SA_ASSERT(0 <= cfacej && cfacej < interp_data.num_cfaces);
                const int ndofsj = interp_data.cfaces_bases[cfacej]->Width();
                SA_ASSERT(interp_data.cfaces_cdofs_offsets[cfacej+1] - interp_data.cfaces_cdofs_offsets[cfacej] == ndofsj);
                const int offsetj = interp_data.celements_cdofs + interp_data.cfaces_cdofs_offsets[cfacej];
                SA_ASSERT(offsetj + ndofsj <= nloc_cdofs);
                SA_ASSERT(bcol + ndofsj <= Aib.Width());
                for (int l=0; l < ndofsj; ++l, ++bcol)
                {
                    SA_ASSERT(gk != offsetj + l);
                    double &rkl = lmat.SearchRow(gk, offsetj + l);
                    SA_ASSERT(0.0 == rkl);
                    rkl += Aib(k, bcol);
                    double &rlk = lmat.SearchRow(offsetj + l, gk);
                    SA_ASSERT(0.0 == rlk);
                    rlk += Aib(k, bcol);
//                    lmat.Add(gk, offsetj + l, Aib(k, bcol));
//                    lmat.Add(offsetj + l, gk, Aib(k, bcol));
                }
            }
            SA_ASSERT(Aib.Width() == bcol);
        }

        bcol = 0;
        for (int j=0; j < ncfaces; ++j)
        {
            const int cfacej = cfaces[j];
            SA_ASSERT(0 <= cfacej && cfacej < interp_data.num_cfaces);
            const int ndofsj = interp_data.cfaces_bases[cfacej]->Width();
            SA_ASSERT(interp_data.cfaces_cdofs_offsets[cfacej+1] - interp_data.cfaces_cdofs_offsets[cfacej] == ndofsj);
            const int offsetj = interp_data.celements_cdofs + interp_data.cfaces_cdofs_offsets[cfacej];
            SA_ASSERT(offsetj + ndofsj <= nloc_cdofs);
            SA_ASSERT(bcol + ndofsj <= Abb.Width());
            for (int l=0; l < ndofsj; ++l, ++bcol)
            {
                int brow = 0;
                for (int m=0; m < ncfaces; ++m)
                {
                    const int cfacem = cfaces[m];
                    SA_ASSERT(0 <= cfacem && cfacem < interp_data.num_cfaces);
                    const int ndofsm = interp_data.cfaces_bases[cfacem]->Width();
                    SA_ASSERT(interp_data.cfaces_cdofs_offsets[cfacem+1] - interp_data.cfaces_cdofs_offsets[cfacem] == ndofsm);
                    const int offsetm = interp_data.celements_cdofs + interp_data.cfaces_cdofs_offsets[cfacem];
                    SA_ASSERT(offsetm + ndofsm <= nloc_cdofs);
                    SA_ASSERT(brow + ndofsm <= Abb.Height());
                    for (int k=0; k < ndofsm; ++k, ++brow)
                    {
                        SA_ASSERT(0.0 == Abb(brow, bcol) || j == m);
                        lmat.Add(offsetm + k, offsetj + l, Abb(brow, bcol));
                    }
                }
                SA_ASSERT(Abb.Height() == brow);
            }
        }
        SA_ASSERT(Abb.Width() == bcol);
    }
    lmat.Finalize();

    // Assemble the global matrix using the local ones and the cDof_TruecDof relation.
    HypreParMatrix tent_gmat(PROC_COMM, cDof_TruecDof.GetGlobalNumRows(), cDof_TruecDof.GetGlobalNumRows(),
                             cDof_TruecDof.GetRowStarts(), cDof_TruecDof.GetRowStarts(), &lmat);
    HypreParMatrix *gmat = RAP(&tent_gmat, &cDof_TruecDof);
    mbox_make_owner_rowstarts_colstarts(*gmat);

    return gmat;
}

/**
    Assembles the global rhs coming from eliminating the interior DoFs. The output vector
    is represented in terms of true cface DoFs (i.e., defined only on cface DoFs)
    and must be freed by the caller. The input vector is in terms of true DoFs that also include the interior DoFs.
*/
static inline
HypreParVector *nonconf_assemble_coarse_schur_rhs(const interp_data_t& interp_data,
                    const agg_partitioning_relations_t& agg_part_rels,
                    const HypreParMatrix& cface_TruecDof_cDof, const Vector& rhs)
{
    SA_ASSERT(interp_data.AbiinvAii);
    SA_ASSERT(interp_data.celements_cdofs_offsets);
    SA_ASSERT(interp_data.cfaces_cdofs_offsets);
    SA_ASSERT(interp_data.cfaces_bases);
    SA_ASSERT(rhs.Size() == interp_data.celements_cdofs + cface_TruecDof_cDof.Height());

    // Assemble the local (on processor) portion of (a component of) the rhs.
    Vector lrhs(cface_TruecDof_cDof.Width());
    SA_ASSERT(lrhs.Size() == interp_data.cfaces_cdofs_offsets[interp_data.num_cfaces]);
    lrhs = 0.0;
    Vector bdr;
    for (int i=0; i < interp_data.nparts; ++i)
    {
        SA_ASSERT(interp_data.AbiinvAii[i]);
        SA_ASSERT(interp_data.celements_cdofs_offsets[i] < interp_data.celements_cdofs);
        SA_ASSERT(interp_data.celements_cdofs_offsets[i+1] <= interp_data.celements_cdofs);
        const int interior_size = interp_data.celements_cdofs_offsets[i+1] - interp_data.celements_cdofs_offsets[i];
        SA_ASSERT(interp_data.cut_evects_arr[i]->Width() == interior_size);
        SA_ASSERT(interp_data.AbiinvAii[i]->Width() == interior_size);
        const Vector interior(rhs.GetData() + interp_data.celements_cdofs_offsets[i], interior_size);

        bdr.SetSize(interp_data.AbiinvAii[i]->Height());
        interp_data.AbiinvAii[i]->Mult(interior, bdr);
        const int ncfaces = agg_part_rels.AE_to_cface->RowSize(i);
        const int * const cfaces = agg_part_rels.AE_to_cface->GetRow(i);
        int brow = 0;
        for (int j=0; j < ncfaces; ++j)
        {
            const int cface = cfaces[j];
            SA_ASSERT(0 <= cface && cface < interp_data.num_cfaces);
            const int ndofs = interp_data.cfaces_bases[cface]->Width();
            SA_ASSERT(interp_data.cfaces_cdofs_offsets[cface+1] - interp_data.cfaces_cdofs_offsets[cface] == ndofs);
            const int offset = interp_data.cfaces_cdofs_offsets[cface];
            SA_ASSERT(offset + ndofs <= lrhs.Size());
            for (int k=0; k < ndofs; ++k, ++brow)
                lrhs(offset + k) -= bdr(brow);
        }
        SA_ASSERT(bdr.Size() == brow);
    }

    // Assemble the global (in terms of true DoFs on cfaces) "elimination" component of the rhs.
    Vector ltrhs(cface_TruecDof_cDof.Height());
    cface_TruecDof_cDof.Mult(lrhs, ltrhs);

    // Extract and add the remaining component of the rhs (in terms of true DoFs on cfaces) that needs no modification.
    HypreParVector *trhs = mbox_restrict_vec_to_faces(rhs, interp_data.celements_cdofs, cface_TruecDof_cDof.GetRowStarts(),
                                                      cface_TruecDof_cDof.GetGlobalNumRows());
    *trhs += ltrhs;

    return trhs;
}

/**
    Performs the backward substitution from the block elimination. It takes the full (including interiors) original
    rhs in true DoFs and the face portion of the (obtained by inverting the Schur complement) solution in face true DoFs (excluding interiors).
    The returned vector (i.e., x) is in terms of all (including interiors) true DoFs.
*/
static inline
void nonconf_coarse_schur_update_interior(const interp_data_t& interp_data,
                    const agg_partitioning_relations_t& agg_part_rels, const HypreParMatrix& cface_cDof_TruecDof,
                    const Vector& rhs, const HypreParVector& facev, Vector& x)
{
    SA_ASSERT(interp_data.invAii);
    SA_ASSERT(interp_data.invAiiAib);
    SA_ASSERT(interp_data.celements_cdofs_offsets);
    SA_ASSERT(interp_data.cfaces_cdofs_offsets);
    SA_ASSERT(interp_data.cfaces_bases);
    SA_ASSERT(rhs.Size() == interp_data.celements_cdofs + cface_cDof_TruecDof.Width());
    SA_ASSERT(facev.Size() == cface_cDof_TruecDof.Width());

    // Get the face vector from true face DoFs to local (repeated) DoFs.
    HypreParVector lfacev(cface_cDof_TruecDof, 1);
    cface_cDof_TruecDof.Mult(facev, lfacev);

    // Allocate the global solution vector (in all true DoFs, including the interiors)
    // and copy the available face solution from facev.
    SA_ASSERT(x.Size() == rhs.Size());
    SA_ASSERT(facev.Size() + interp_data.celements_cdofs <= x.Size());
    for (int i = 0; i < facev.Size(); ++i)
        x(i + interp_data.celements_cdofs) = facev(i);

    // Update (backward substitute) the interior portions of the solution vector.
    Vector bdr;
    for (int i=0; i < interp_data.nparts; ++i)
    {
        SA_ASSERT(interp_data.invAii[i]);
        SA_ASSERT(interp_data.invAiiAib[i]);
        SA_ASSERT(interp_data.celements_cdofs_offsets[i] < interp_data.celements_cdofs);
        SA_ASSERT(interp_data.celements_cdofs_offsets[i+1] <= interp_data.celements_cdofs);
        const int interior_size = interp_data.celements_cdofs_offsets[i+1] - interp_data.celements_cdofs_offsets[i];
        SA_ASSERT(interp_data.cut_evects_arr[i]->Width() == interior_size);
        SA_ASSERT(interp_data.invAii[i]->Width() == interior_size);
        SA_ASSERT(interp_data.invAii[i]->Height() == interior_size);
        SA_ASSERT(interp_data.invAiiAib[i]->Height() == interior_size);
        const Vector rhs_interior(rhs.GetData() + interp_data.celements_cdofs_offsets[i], interior_size);
        Vector x_interior(x.GetData() + interp_data.celements_cdofs_offsets[i], interior_size);

        // Put the direct inversion of the portion of the rhs.
        interp_data.invAii[i]->Mult(rhs_interior, x_interior);

        // Add the contribution of the actual backward substitution.

        // Localize the face solution on the faces of the element in the respective order.
        bdr.SetSize(interp_data.invAiiAib[i]->Width());
        const int ncfaces = agg_part_rels.AE_to_cface->RowSize(i);
        const int * const cfaces = agg_part_rels.AE_to_cface->GetRow(i);
        int brow = 0;
        for (int j=0; j < ncfaces; ++j)
        {
            const int cface = cfaces[j];
            SA_ASSERT(0 <= cface && cface < interp_data.num_cfaces);
            const int ndofs = interp_data.cfaces_bases[cface]->Width();
            SA_ASSERT(interp_data.cfaces_cdofs_offsets[cface+1] - interp_data.cfaces_cdofs_offsets[cface] == ndofs);
            const int offset = interp_data.cfaces_cdofs_offsets[cface];
            SA_ASSERT(offset + ndofs <= lfacev.Size());
            SA_ASSERT(brow + ndofs <= bdr.Size());
            for (int k=0; k < ndofs; ++k, ++brow)
                bdr(brow) = lfacev(offset + k);
        }
        SA_ASSERT(bdr.Size() == brow);

        // Obtain the final contribution.
        interp_data.invAiiAib[i]->AddMult_a(-1.0, bdr, x_interior);
    }
}

void SchurSolver::Mult(const mfem::Vector &x, mfem::Vector &y) const
{
    // Elimination
    HypreParVector *schur_rhs = nonconf_assemble_coarse_schur_rhs(interp_data, agg_part_rels,
                                                                  cface_TruecDof_cDof, x);
    // Solve Schur system
    HypreParVector eb(*schur_rhs);
    eb = 0.0;
    solver.Mult(*schur_rhs, eb);
    delete schur_rhs;

    // Backward substitution
    nonconf_coarse_schur_update_interior(interp_data, agg_part_rels, cface_cDof_TruecDof, x, eb, y);
}

void nonconf_ip_coarsen_finest(tg_data_t& tg_data, agg_partitioning_relations_t& agg_part_rels,
                               ElementMatrixProvider *elem_data, double theta, double delta,
                               bool schur, bool full_space)
{
    tg_data.elem_data = elem_data;
    tg_data.doing_spectral = true;

    Vector **evals;

    // Solve local (on coarse/agglomerated elements) eigenvalue problems and obtain the respective low eigenvectors.
    //TODO: This function is a candidate to be replaced by a matrix-free version, if having a matrix assembled on AE
    //      is deemed suboptimal.
    evals = interp_compute_vectors_nostore(agg_part_rels, *tg_data.interp_data, elem_data, theta, full_space);

    // Having the local eigenvectors, distribute them on faces (taking into account different contributions and
    // filtering out linear dependences) to obtain coarse/agglomerated face basis. This involves communicating the
    // necessary vector and basis information between coarse/agglomerated elements and the respective processors.
    ContribTent cfaces_bases_contrib(agg_part_rels.ND);
    SA_ASSERT(NULL == tg_data.interp_data->cfaces_bases);
    if (full_space)
        tg_data.interp_data->cfaces_bases = cfaces_bases_contrib.contrib_cfaces_full(agg_part_rels);
    else
        tg_data.interp_data->cfaces_bases = cfaces_bases_contrib.contrib_cfaces(agg_part_rels, tg_data.interp_data->cut_evects_arr, false);
    tg_data.interp_data->num_cfaces = agg_part_rels.num_cfaces;
    tg_data.interp_data->coarse_truedof_offset = cfaces_bases_contrib.get_coarse_truedof_offset();

    // Compute Schur complements and other matrices necessary for the algorithm.
    nonconf_ip_first_coarse_schur_matrices(*tg_data.interp_data, agg_part_rels, evals, delta, !schur, full_space);

    for (int i=0; i < agg_part_rels.nparts; ++i)
        delete evals[i];
    delete [] evals;

    // Obtain the local (on CPU) "interpolant" and "restriction".
    cfaces_bases_contrib.insert_from_cfaces_celems_bases(tg_data.interp_data->nparts, tg_data.interp_data->num_cfaces,
                                                         tg_data.interp_data->cut_evects_arr,
                                                         tg_data.interp_data->cfaces_bases, agg_part_rels);
    delete tg_data.ltent_interp;
    delete tg_data.ltent_restr;
    delete tg_data.tent_interp;
    delete tg_data.interp;
    delete tg_data.restr;
    tg_data.ltent_interp = cfaces_bases_contrib.contrib_tent_finalize();
//    tg_data.ltent_restr = cfaces_bases_contrib.contrib_restr_finalize();
    tg_data.interp_data->celements_cdofs = cfaces_bases_contrib.get_celements_cdofs();
    tg_data.interp_data->celements_cdofs_offsets = cfaces_bases_contrib.get_celements_cdofs_offsets();
    tg_data.interp_data->cfaces_truecdofs_offsets = cfaces_bases_contrib.get_cfaces_truecdofs_offsets();
    tg_data.interp_data->cfaces_cdofs_offsets = cfaces_bases_contrib.get_cfaces_cdofs_offsets();

    // Make the "interpolant" and "restriction" global via hypre.
    tg_data.interp_data->tent_interp_offsets.SetSize(0);
    tg_data.interp = interp_global_tent_assemble(agg_part_rels, *tg_data.interp_data,
                                                 tg_data.ltent_interp);
//    tg_data.restr = interp_global_restr_assemble(agg_part_rels, *tg_data.interp_data,
//                                                 tg_data.ltent_restr);
    tg_data.restr = tg_data.interp->Transpose();

    // Obtain the remaining necessary relations.
    agg_create_cface_cDof_TruecDof_relations(agg_part_rels, tg_data.interp_data->coarse_truedof_offset,
                                             tg_data.interp_data->cfaces_bases, tg_data.interp_data->celements_cdofs, !schur);

    // Construct the coarse operator, which is a coarse Schur complement on the coarse faces.
    delete tg_data.Ac;
    if (schur)
        tg_data.Ac = nonconf_assemble_coarse_schur_matrix(*tg_data.interp_data, agg_part_rels,
                        *agg_part_rels.cface_cDof_TruecDof);
    else
        tg_data.Ac = nonconf_ip_dense_assemble_coarse_matrix(*tg_data.interp_data, agg_part_rels,
                        *agg_part_rels.cface_cDof_TruecDof);
}

/**
    Obtains the interior penalty element matrices (their blocks).

    1/delta is the scaling parameter for the interface term.

    Fills in arrays in the interpolation data. They will be freed with the interpolation data.
*/
static inline
void nonconf_ip_discretization_matrices(interp_data_t& interp_data,
                  const agg_partitioning_relations_t& agg_part_rels, double delta)
{
    SA_ASSERT(NULL != interp_data.cfaces_bases);
    const DenseMatrix * const * const cfaces_bases = interp_data.cfaces_bases;

    // Assemble the blocks of the interior penalty coarse/agglomerate element matrix for each coarse/agglomerate element.
    SA_ASSERT(NULL == interp_data.Aii);
    SA_ASSERT(NULL == interp_data.Abb);
    SA_ASSERT(NULL == interp_data.Aib);
    interp_data.Aii = (Matrix **)new SparseMatrix*[agg_part_rels.nparts];
    interp_data.Abb = (Matrix **)new SparseMatrix*[agg_part_rels.nparts];
    interp_data.Aib = (Matrix **)new SparseMatrix*[agg_part_rels.nparts];

    for (int i=0; i < agg_part_rels.nparts; ++i)
    {
        SA_ASSERT(NULL != interp_data.cut_evects_arr[i]);
        interp_data.AEs_stiffm[i]->MoveDiagonalFirst();
        SparseMatrix *D = mbox_snd_diagA_sparse_from_sparse(*interp_data.AEs_stiffm[i]);
        SA_ASSERT(D->Height() == D->Width());
        const int ndofs = agg_part_rels.AE_to_dof->RowSize(i);
        const int num_cfaces = agg_part_rels.AE_to_cface->RowSize(i);
        const int * const cfaces = agg_part_rels.AE_to_cface->GetRow(i);
        const int interior_size = interp_data.cut_evects_arr[i]->Width();
        int bdr_size = 0;
        for (int j=0; j < num_cfaces; ++j)
        {
            SA_ASSERT(cfaces_bases[cfaces[j]]);
            bdr_size += cfaces_bases[cfaces[j]]->Width();
        }
        SparseMatrix *Abb = new SparseMatrix(bdr_size);
        SparseMatrix *Aib = new SparseMatrix(interior_size, bdr_size);

        const Vector diag(D->GetData(), D->Height());

        // Loop over the coarse/agglomerate faces of the current agglomerate/coarse element and
        // obtain or finish the respective entries of the matrices.
        Array<int> map(ndofs);
        int intdofs_ctr = 0;
        for (int j=0; j < ndofs; ++j)
        {
            const int gj = agg_num_col_to_glob(*agg_part_rels.AE_to_dof, i, j);
            SA_ASSERT(0 <= gj && gj < agg_part_rels.ND);
            if (!SA_IS_SET_A_FLAG(agg_part_rels.agg_flags[gj], AGG_ON_ESS_DOMAIN_BORDER_FLAG))
            {
                SA_ASSERT(intdofs_ctr < D->Height());
                map[j] = intdofs_ctr;
                ++intdofs_ctr;
            } else
                map[j] = -1;
        }
        SA_ASSERT(intdofs_ctr == D->Height());
        int cface_offset = 0;
        for (int j=0; j < num_cfaces; ++j)
        {
            const int cface = cfaces[j];
            SA_ASSERT(0 <= cface && cface < agg_part_rels.num_cfaces);
            const int cface_basis_size = cfaces_bases[cface]->Width();
            SA_ASSERT(cface_offset + cface_basis_size <= bdr_size);
            SA_ASSERT(cfaces_bases[cface]->Height() == agg_part_rels.cfaces_dof_size[cface]);
            SA_ASSERT(agg_part_rels.cface_to_dof->RowSize(cface) == agg_part_rels.cfaces_dof_size[cface]);
            const int num_dofs = agg_part_rels.cface_to_dof->RowSize(cface);
            const int * const dofs = agg_part_rels.cface_to_dof->GetRow(cface);
            int ctr = 0;
            for (int k=0; k < num_dofs; ++k)
            {
                if (SA_IS_SET_A_FLAG(agg_part_rels.agg_flags[dofs[k]], AGG_ON_ESS_DOMAIN_BORDER_FLAG))
                    continue;
                const int ldof = map[agg_map_id_glob_to_AE(dofs[k], i, agg_part_rels)];
                SA_ASSERT(0 <= ldof && ldof < D->Height());
                SA_ASSERT(interp_data.AEs_stiffm[i]->GetRowColumns(ldof)[0] == ldof);
                interp_data.AEs_stiffm[i]->GetRowEntries(ldof)[0] += (1./delta) * diag(ldof);
                SA_ASSERT(ctr < cface_basis_size && cface_offset + ctr < bdr_size);
                Abb->Set(cface_offset + ctr, cface_offset + ctr, (1./delta) * diag(ldof));
                Aib->Set(ldof, cface_offset + ctr, -(1./delta) * diag(ldof));
                ++ctr;
            }
            SA_ASSERT(cface_basis_size == ctr);
            cface_offset += cface_basis_size;
        }
        SA_ASSERT(cface_offset == bdr_size);
        delete D;
        interp_data.Aii[i] = interp_data.AEs_stiffm[i];
        interp_data.AEs_stiffm[i] = NULL;
        Abb->Finalize();
        Aib->Finalize();
        interp_data.Abb[i] = Abb;
        interp_data.Aib[i] = Aib;
    }
}

/**
    Assembles global coarse interior penalty matrix using sparse local matrices.

    The returned global matrix must be freed by the caller.
*/
static inline
HypreParMatrix *nonconf_ip_discretization_assemble(const interp_data_t& interp_data,
                    const agg_partitioning_relations_t& agg_part_rels,
                    const HypreParMatrix& cDof_TruecDof)
{
    SA_ASSERT(agg_part_rels.AE_to_cface);
    SA_ASSERT(interp_data.Aii);
    SA_ASSERT(interp_data.Abb);
    SA_ASSERT(interp_data.Aib);

    // Assemble the local (on processor) portion of the matrix.
    const int nloc_cdofs = interp_data.celements_cdofs + interp_data.cfaces_cdofs_offsets[interp_data.num_cfaces];
    SparseMatrix lmat(nloc_cdofs, nloc_cdofs);
    SA_ASSERT(nloc_cdofs == cDof_TruecDof.GetNumRows());

    for (int i=0; i < interp_data.nparts; ++i)
    {
        const SparseMatrix &Aii = *dynamic_cast<SparseMatrix *>(interp_data.Aii[i]);
        const SparseMatrix &Abb = *dynamic_cast<SparseMatrix *>(interp_data.Abb[i]);
        const SparseMatrix &Aib = *dynamic_cast<SparseMatrix *>(interp_data.Aib[i]);
        SA_ASSERT(&Aii);
        SA_ASSERT(&Abb);
        SA_ASSERT(&Aib);
        SA_ASSERT(Aii.Width() == Aii.Height());
        SA_ASSERT(Aii.Width() == Aib.Height());
        SA_ASSERT(Abb.Width() == Abb.Height());
        SA_ASSERT(Aib.Width() == Abb.Height());
        SA_ASSERT(Aii.Width() == interp_data.cut_evects_arr[i]->Width());

        const int ncintdofs = Aii.Height();
        const int nbdrdofs = Abb.Height();
        const int ncfaces = agg_part_rels.AE_to_cface->RowSize(i);
        const int * const cfaces = agg_part_rels.AE_to_cface->GetRow(i);

        Array<int> map(nbdrdofs);
        int ctr = 0;
        for (int j=0; j < ncfaces; ++j)
        {
            const int cface = cfaces[j];
            SA_ASSERT(0 <= cface && cface < interp_data.num_cfaces);
            SA_ASSERT(interp_data.cfaces_cdofs_offsets[cface+1] - interp_data.cfaces_cdofs_offsets[cface] == interp_data.cfaces_bases[cface]->Width());
            int offset = interp_data.celements_cdofs + interp_data.cfaces_cdofs_offsets[cface];
            SA_ASSERT(offset + interp_data.cfaces_bases[cface]->Width() <= nloc_cdofs);
            const int num_dofs = agg_part_rels.cface_to_dof->RowSize(cface);
            const int * const dofs = agg_part_rels.cface_to_dof->GetRow(cface);
            for (int k=0; k < num_dofs; ++k)
            {
                if (SA_IS_SET_A_FLAG(agg_part_rels.agg_flags[dofs[k]], AGG_ON_ESS_DOMAIN_BORDER_FLAG))
                    continue;
                SA_ASSERT(ctr < nbdrdofs);
                map[ctr++] = offset++;
            }
            SA_ASSERT(interp_data.celements_cdofs + interp_data.cfaces_cdofs_offsets[cface+1] == offset);
        }
        SA_ASSERT(nbdrdofs == ctr);

        for (int k=0; k < ncintdofs; ++k)
        {
            const int gk = interp_data.celements_cdofs_offsets[i] + k;
            SA_ASSERT(0 <= gk && gk < nloc_cdofs);
        {
            const int * const row = Aii.GetRowColumns(k);
            const double * const entries = Aii.GetRowEntries(k);
            const int row_size = Aii.RowSize(k);
            for (int j=0; j < row_size; ++j)
            {
                const int gj = interp_data.celements_cdofs_offsets[i] + row[j];
                SA_ASSERT(0 <= gj && gj < nloc_cdofs);
                double &rkj = lmat.SearchRow(gk, gj);
                SA_ASSERT(0.0 == rkj);
                rkj += entries[j];
//                lmat.Add(gk, gj, entries[j]);
            }
        }
        {
            const int * const row = Aib.GetRowColumns(k);
            const double * const entries = Aib.GetRowEntries(k);
            const int row_size = Aib.RowSize(k);
            for (int j=0; j < row_size; ++j)
            {
                const int gj = map[row[j]];
                SA_ASSERT(0 <= gj && gj < nloc_cdofs);
                SA_ASSERT(gj != gk);
                double &rkj = lmat.SearchRow(gk, gj);
                SA_ASSERT(0.0 == rkj);
                rkj += entries[j];
//                lmat.Add(gk, gj, entries[j]);
                double &rjk = lmat.SearchRow(gj, gk);
                SA_ASSERT(0.0 == rjk);
                rjk += entries[j];
//                lmat.Add(gj, gk, entries[j]);
            }
        }
        }

        for (int k=0; k < nbdrdofs; ++k)
        {
            const int gk = map[k];
            SA_ASSERT(0 <= gk && gk < nloc_cdofs);
            const int * const row = Abb.GetRowColumns(k);
            const double * const entries = Abb.GetRowEntries(k);
            const int row_size = Abb.RowSize(k);
            for (int j=0; j < row_size; ++j)
            {
                const int gj = map[row[j]];
                SA_ASSERT(0 <= gj && gj < nloc_cdofs);
                lmat.Add(gk, gj, entries[j]);
            }
        }
    }
    lmat.Finalize();

    // Assemble the global matrix using the local ones and the cDof_TruecDof relation.
    HypreParMatrix tent_gmat(PROC_COMM, cDof_TruecDof.GetGlobalNumRows(), cDof_TruecDof.GetGlobalNumRows(),
                             cDof_TruecDof.GetRowStarts(), cDof_TruecDof.GetRowStarts(), &lmat);
    HypreParMatrix *gmat = RAP(&tent_gmat, &cDof_TruecDof);
    mbox_make_owner_rowstarts_colstarts(*gmat);

    return gmat;
}

void nonconf_ip_discretization(tg_data_t& tg_data, agg_partitioning_relations_t& agg_part_rels,
                               ElementMatrixProvider *elem_data, double delta)
{
    tg_data.elem_data = elem_data;
    tg_data.doing_spectral = false;

    // Prepare "identity" basis and "interior" stiffness matrix, removing all essential BCs' DoFs.
    DenseMatrix ** const cut_evects_arr = tg_data.interp_data->cut_evects_arr;
    SA_ASSERT(cut_evects_arr);
    const int nparts = agg_part_rels.nparts;
    SA_ASSERT(0 < nparts);
    SparseMatrix *AE_stiffm;
    SparseMatrix *new_AE_stiffm;
    SA_ASSERT(tg_data.interp_data->AEs_stiffm);
    for (int i=0; i < nparts; ++i)
    {
        AE_stiffm = elem_data->BuildAEStiff(i);
        SA_ASSERT(AE_stiffm);
        SA_ASSERT(AE_stiffm->Width() == AE_stiffm->Height());
        const int ndofs = AE_stiffm->Height();
        int nintdofs = 0;
        int total_nnz = 0;
        for (int j=0; j < ndofs; ++j)
        {
            const int gj = agg_num_col_to_glob(*agg_part_rels.AE_to_dof, i, j);
            SA_ASSERT(0 <= gj && gj < agg_part_rels.ND);
            if (!SA_IS_SET_A_FLAG(agg_part_rels.agg_flags[gj], AGG_ON_ESS_DOMAIN_BORDER_FLAG))
            {
                ++nintdofs;
                total_nnz += AE_stiffm->RowSize(j);
            } else
                SA_ASSERT(1 == AE_stiffm->RowSize(j));
        }
        SA_ASSERT(ndofs >= nintdofs);
        SA_ASSERT(total_nnz == AE_stiffm->NumNonZeroElems() - (ndofs - nintdofs));
        SA_ASSERT(!cut_evects_arr[i]);
        cut_evects_arr[i] = new DenseMatrix(ndofs, nintdofs);
        SA_ASSERT(cut_evects_arr[i]);
        Array<int> map(ndofs);
        int intdofs_ctr = 0;
        for (int j=0; j < ndofs; ++j)
        {
            const int gj = agg_num_col_to_glob(*agg_part_rels.AE_to_dof, i, j);
            SA_ASSERT(0 <= gj && gj < agg_part_rels.ND);
            if (!SA_IS_SET_A_FLAG(agg_part_rels.agg_flags[gj], AGG_ON_ESS_DOMAIN_BORDER_FLAG))
            {
                SA_ASSERT(intdofs_ctr < nintdofs);
                cut_evects_arr[i]->Elem(j, intdofs_ctr) = 1.0;
                map[j] = intdofs_ctr;
                ++intdofs_ctr;
            } else
                map[j] = -1;
        }
        SA_ASSERT(intdofs_ctr == nintdofs);
        new_AE_stiffm = new SparseMatrix(nintdofs);
        intdofs_ctr = 0;
        for (int j=0; j < ndofs; ++j)
        {
            const int gj = agg_num_col_to_glob(*agg_part_rels.AE_to_dof, i, j);
            SA_ASSERT(0 <= gj && gj < agg_part_rels.ND);
            if (!SA_IS_SET_A_FLAG(agg_part_rels.agg_flags[gj], AGG_ON_ESS_DOMAIN_BORDER_FLAG))
            {
                SA_ASSERT(intdofs_ctr < nintdofs);
                Array<int> cols;
                Vector srow;
                AE_stiffm->GetRow(j, cols, srow);
                SA_ASSERT(cols.Size() == srow.Size() && srow.Size() == AE_stiffm->RowSize(j));
                for (int k = 0; k < cols.Size(); ++k)
                {
                    SA_ASSERT(0 <= cols[k] && cols[k] < ndofs);
                    cols[k] = map[cols[k]];
                    SA_ASSERT(0 <= cols[k] && cols[k] < nintdofs);
                }
                SA_ASSERT(map[j] == intdofs_ctr);
                new_AE_stiffm->SetRow(intdofs_ctr, cols, srow);
                ++intdofs_ctr;
            }
        }
        SA_ASSERT(intdofs_ctr == nintdofs);
        delete AE_stiffm;
        new_AE_stiffm->Finalize();
        SA_ASSERT(new_AE_stiffm->NumNonZeroElems() == total_nnz);
        SA_ASSERT(NULL == tg_data.interp_data->AEs_stiffm[i]);
        tg_data.interp_data->AEs_stiffm[i] = new_AE_stiffm;
    }

    // Obtain coarse/agglomerated face basis.
    ContribTent cfaces_bases_contrib(agg_part_rels.ND);
    SA_ASSERT(NULL == tg_data.interp_data->cfaces_bases);
    tg_data.interp_data->cfaces_bases = cfaces_bases_contrib.contrib_cfaces_full(agg_part_rels);
    tg_data.interp_data->num_cfaces = agg_part_rels.num_cfaces;
    tg_data.interp_data->coarse_truedof_offset = cfaces_bases_contrib.get_coarse_truedof_offset();

    // Obtain the local (on CPU) "interpolant" and "restriction".
    cfaces_bases_contrib.insert_from_cfaces_celems_bases(tg_data.interp_data->nparts, tg_data.interp_data->num_cfaces,
                                                         tg_data.interp_data->cut_evects_arr,
                                                         tg_data.interp_data->cfaces_bases, agg_part_rels);
    delete tg_data.ltent_interp;
    delete tg_data.ltent_restr;
    delete tg_data.tent_interp;
    delete tg_data.interp;
    delete tg_data.restr;
    tg_data.ltent_interp = cfaces_bases_contrib.contrib_tent_finalize();
//    tg_data.ltent_restr = cfaces_bases_contrib.contrib_restr_finalize();
    tg_data.interp_data->celements_cdofs = cfaces_bases_contrib.get_celements_cdofs();
    tg_data.interp_data->celements_cdofs_offsets = cfaces_bases_contrib.get_celements_cdofs_offsets();
    tg_data.interp_data->cfaces_truecdofs_offsets = cfaces_bases_contrib.get_cfaces_truecdofs_offsets();
    tg_data.interp_data->cfaces_cdofs_offsets = cfaces_bases_contrib.get_cfaces_cdofs_offsets();

    // Make the "interpolant" and "restriction" global via hypre.
    tg_data.interp_data->tent_interp_offsets.SetSize(0);
    tg_data.interp = interp_global_tent_assemble(agg_part_rels, *tg_data.interp_data,
                                                 tg_data.ltent_interp);
//    tg_data.restr = interp_global_restr_assemble(agg_part_rels, *tg_data.interp_data,
//                                                 tg_data.ltent_restr);
    tg_data.restr = tg_data.interp->Transpose();

    // Obtain the remaining necessary relations.
    agg_create_cface_cDof_TruecDof_relations(agg_part_rels, tg_data.interp_data->coarse_truedof_offset,
                                             tg_data.interp_data->cfaces_bases, tg_data.interp_data->celements_cdofs, true);

    nonconf_ip_discretization_matrices(*tg_data.interp_data, agg_part_rels, delta);

    // Construct the operator as though it is a "coarse" matrix.
    delete tg_data.Ac;
    tg_data.Ac = nonconf_ip_discretization_assemble(*tg_data.interp_data, agg_part_rels,
                    *agg_part_rels.cface_cDof_TruecDof);
}

agg_partitioning_relations_t *
nonconf_create_partitioning(const agg_partitioning_relations_t& agg_part_rels_nonconf,
                            const interp_data_t& interp_data_nonconf)
{
    const int nparts = agg_part_rels_nonconf.nparts;
    const int num_cfaces = agg_part_rels_nonconf.num_cfaces;

    agg_partitioning_relations_t *agg_part_rels =
        new agg_partitioning_relations_t;
    memset(agg_part_rels, 0, sizeof(*agg_part_rels));

    agg_part_rels->nparts = nparts;
    agg_part_rels->Dof_TrueDof = mbox_clone_parallel_matrix(agg_part_rels_nonconf.cface_cDof_TruecDof);
    agg_part_rels->ND = agg_part_rels->Dof_TrueDof->GetNumRows();
    SA_ASSERT(interp_data_nonconf.celements_cdofs
              + interp_data_nonconf.cfaces_cdofs_offsets[num_cfaces] == agg_part_rels->ND);
    agg_part_rels->owns_Dof_TrueDof = true;
    agg_part_rels->partitioning = helpers_copy_int_arr(agg_part_rels_nonconf.partitioning,
                                                       agg_part_rels_nonconf.elem_to_elem->Size());
    agg_part_rels->elem_to_elem = mbox_copy_table(agg_part_rels_nonconf.elem_to_elem);
    agg_part_rels->AE_to_elem = mbox_copy_table(agg_part_rels_nonconf.AE_to_elem);
    agg_part_rels->elem_to_AE = mbox_copy_table(agg_part_rels_nonconf.elem_to_AE);

    // This seems to matter only during local assembly on the finest level but
    // we never do assembly here and essential boundary DoFs are eliminated.
    agg_part_rels->agg_flags = new agg_dof_status_t[agg_part_rels->ND];
    memset(agg_part_rels->agg_flags, 0, sizeof(agg_dof_status_t) * agg_part_rels->ND);

    agg_part_rels->AE_to_dof = new Table();
    Table& AE_to_dof = *agg_part_rels->AE_to_dof;
    AE_to_dof.MakeI(nparts);
    for (int i=0; i < nparts; ++i)
    {
        SA_ASSERT(interp_data_nonconf.cut_evects_arr[i]->Width() == interp_data_nonconf.celements_cdofs_offsets[i+1]
                                                                    - interp_data_nonconf.celements_cdofs_offsets[i]);
        AE_to_dof.AddColumnsInRow(i, interp_data_nonconf.cut_evects_arr[i]->Width());
        const int ncfaces = agg_part_rels_nonconf.AE_to_cface->RowSize(i);
        const int * const cfaces = agg_part_rels_nonconf.AE_to_cface->GetRow(i);
        for (int j=0; j < ncfaces; ++j)
        {
            const int cface = cfaces[j];
            SA_ASSERT(0 <= cface && cface < num_cfaces);
            SA_ASSERT(interp_data_nonconf.cfaces_bases[cface]->Width() == interp_data_nonconf.cfaces_cdofs_offsets[cface+1]
                                                                          - interp_data_nonconf.cfaces_cdofs_offsets[cface]);
            AE_to_dof.AddColumnsInRow(i, interp_data_nonconf.cfaces_bases[cface]->Width());
        }
    }
    AE_to_dof.MakeJ();

    for (int i=0; i < nparts; ++i)
    {
        const int idofs = interp_data_nonconf.cut_evects_arr[i]->Width();
        int ioffset = interp_data_nonconf.celements_cdofs_offsets[i];
        for (int k=0; k < idofs; ++k, ++ioffset)
            AE_to_dof.AddConnection(i, ioffset);
        SA_ASSERT(ioffset == interp_data_nonconf.celements_cdofs_offsets[i+1]);
        const int ncfaces = agg_part_rels_nonconf.AE_to_cface->RowSize(i);
        const int * const cfaces = agg_part_rels_nonconf.AE_to_cface->GetRow(i);
        for (int j=0; j < ncfaces; ++j)
        {
            const int cface = cfaces[j];
            SA_ASSERT(0 <= cface && cface < num_cfaces);
            const int fdofs = interp_data_nonconf.cfaces_bases[cface]->Width();
            int foffset = interp_data_nonconf.celements_cdofs + interp_data_nonconf.cfaces_cdofs_offsets[cface];
            for (int k=0; k < fdofs; ++k, ++foffset)
                AE_to_dof.AddConnection(i, foffset);
            SA_ASSERT(foffset == interp_data_nonconf.celements_cdofs + interp_data_nonconf.cfaces_cdofs_offsets[cface+1]);
        }
    }
    AE_to_dof.ShiftUpI();
    AE_to_dof.Finalize(); //This shouldn't do anything. Here for consistency, if implementation ever changes.

    agg_part_rels->dof_to_AE = new Table();
    Transpose(AE_to_dof, *agg_part_rels->dof_to_AE, agg_part_rels->ND);

    agg_build_glob_to_AE_id_map(*agg_part_rels);
    HypreParMatrix nil;
    agg_produce_mises(nil, *agg_part_rels, NULL, false);

    return agg_part_rels;
}

ElementIPMatrix::ElementIPMatrix(const agg_partitioning_relations_t& agg_part_rels,
                                 const interp_data_t& interp_data_nonconf) :
    ElementMatrixProvider(agg_part_rels),
    interp_data_nonconf(interp_data_nonconf)
{
    is_geometric = false;
    SA_ASSERT(interp_data_nonconf.Aii);
    SA_ASSERT(interp_data_nonconf.Abb);
    SA_ASSERT(interp_data_nonconf.Aib);
}

Matrix *ElementIPMatrix::GetMatrix(int elno, bool& free_matr) const
{
    SA_ASSERT(false); // Makes no sense, since effectively there is no element matrices,
                      // but only AE matrices.
    free_matr = true;
    return new DenseMatrix;
}

SparseMatrix *ElementIPMatrix::BuildAEStiff(int elno) const
{
    SA_ASSERT(0 <= elno && elno < interp_data_nonconf.nparts);
    SparseMatrix *Aii = dynamic_cast<SparseMatrix *>(interp_data_nonconf.Aii[elno]);
    SparseMatrix *Abb = dynamic_cast<SparseMatrix *>(interp_data_nonconf.Abb[elno]);
    SparseMatrix *Aib = dynamic_cast<SparseMatrix *>(interp_data_nonconf.Aib[elno]);
    SA_ASSERT(Aii);
    SA_ASSERT(Abb);
    SA_ASSERT(Aib);
    SA_ASSERT(Aii->Width() == Aii->Height());
    SA_ASSERT(Aii->Width() == Aib->Height());
    SA_ASSERT(Abb->Width() == Abb->Height());
    SA_ASSERT(Aib->Width() == Abb->Height());
    SparseMatrix *Abi = Transpose(*Aib);

    Array<int> blocks(3);
    blocks[0] = 0;
    blocks[1] = Aii->Height();
    blocks[2] = Aii->Height() + Abb->Height();
    BlockMatrix AEmat(blocks);
    AEmat.SetBlock(0, 0, Aii);
    AEmat.SetBlock(0, 1, Aib);
    AEmat.SetBlock(1, 0, Abi);
    AEmat.SetBlock(1, 1, Abb);

    SparseMatrix *ret = AEmat.CreateMonolithic();
    SA_ASSERT(ret->Height() == ret->Width());
    SA_ASSERT(ret->Height() == Aii->Height() + Abb->Height());

    delete Abi;
    return ret;
}

} // namespace saamge
