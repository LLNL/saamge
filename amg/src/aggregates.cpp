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
#include "aggregates.hpp"
#include <cmath>
#include <algorithm>
#include <vector>
#include <mfem.hpp>
#include "part.hpp"
#include "elmat.hpp"
#include "helpers.hpp"
#include "mbox.hpp"
#include "mfem_addons.hpp"
#include "arbitrator.hpp"
using std::fabs;
using std::sqrt;

using namespace mfem;

/* Static Functions */

/*! \brief Computes the value of an element of a local stiffness matrix.

    Computes the value of an element of a local (for an AE) stiffness matrix.

    \param di (IN) The global number of the first DoF.
    \param dj (IN) The global number of the second DoF.
    \param part (IN) The id of the AE.
    \param agg_part_rels (IN) The partitioning relations.
    \param data (IN/OUT) \a elmat_callback specific data, provides element matrices

    \returns The computed value.

    \warning It must be compiled with Run-Time Type Information (RTTI) enabled
             for \em dynamic_cast to work.
*/
static inline
double agg_assemble_value(int di, int dj, int part,
                          const agg_partitioning_relations_t& agg_part_rels,
                          ElementMatrixProvider *data)
{
    SA_ASSERT(&agg_part_rels);
    SA_ASSERT(agg_part_rels.partitioning);
    SA_ASSERT(agg_part_rels.dof_to_elem);
    SA_ASSERT(agg_part_rels.elem_to_dof);
    const int *partitioning = agg_part_rels.partitioning;
    const Table& dof_to_elem = *agg_part_rels.dof_to_elem;
    const Table& elem_to_dof = *agg_part_rels.elem_to_dof;
    double value = 0.;
    Array<int> rowi, rowj;
    SA_ASSERT(di >=0 && dj >= 0);
    SA_ASSERT(di < dof_to_elem.Size() && dj < dof_to_elem.Size());
    const int rsi = dof_to_elem.RowSize(di);
    const int rsj = dof_to_elem.RowSize(dj);
    const Matrix *elem_matr;
    const DenseMatrix *elmat;
    const SparseMatrix *spm;
    int dii, djj;
    bool free_matr;

    SA_ASSERT(di != dj || rsi == rsj);

    dof_to_elem.GetRow(di, rowi);
    if (di != dj)
    {
        dof_to_elem.GetRow(dj, rowj);
        rowi.Sort();
        rowj.Sort();
#if (SA_IS_DEBUG_LEVEL(4))
        for (int i=1; i < rsi; ++i)
            SA_ASSERT(rowi[i] > rowi[i-1]);
        for (int i=1; i < rsj; ++i)
            SA_ASSERT(rowj[i] > rowj[i-1]);
#endif
    }

    int i, k, j = 0;
    for (i=0; i < rsi; ++i)
    {
        const int elno = rowi[i];
        if (partitioning[elno] != part)
            continue;
        if (di != dj)
        {
            while (j < rsj && rowj[j] < elno)
                ++j;
            if (j >= rsj)
                break;
            SA_ASSERT(rowj[j] >= elno);
            if (rowj[j] != elno)
                continue;
            SA_ASSERT(i < rsi && j < rsj);
            SA_ASSERT(rowj[j] == elno);
        }

        const int ndofs = elem_to_dof.RowSize(elno);
        const int * const dofs = elem_to_dof.GetRow(elno);
        SA_ASSERT(0 < ndofs);

        dii = -1; djj = -1;
        for (k=0; k < ndofs && (dii < 0 || djj < 0); ++k)
        {
            SA_ASSERT(0 <= dofs[k] && dofs[k] < dof_to_elem.Size());
            if (dofs[k] == di)
            {
                SA_ASSERT(dii < 0);
                dii = k;
            }
            if (dofs[k] == dj)
            {
                SA_ASSERT(djj < 0);
                djj = k;
            }
        }
        SA_ASSERT(dii >= 0 && djj >= 0);

        // elem_matr = elmat_callback(elno, &agg_part_rels, data, free_matr);
        elem_matr = data->GetMatrix(elno, free_matr);
        if ((elmat = dynamic_cast<const DenseMatrix *>(elem_matr)))
        {  // Dense matrix.
            SA_ASSERT(elmat->Width() == ndofs);
            SA_ASSERT(elmat->Height() == ndofs);
#if (SA_IS_DEBUG_LEVEL(2))
            SA_ALERT_COND_MSG(SA_IS_REAL_EQ((*elmat)(dii, djj),
                                            (*elmat)(djj, dii)),
                              "Non-symmetric element matrix!!! Difference: %g",
                              (*elmat)(dii, djj) - (*elmat)(djj, dii));
#endif
            value += (*elmat)(dii, djj);
            if (free_matr)
                delete elmat;
        } else
        { // Sparse element matrix.
            SA_ASSERT(dynamic_cast<const SparseMatrix *>(elem_matr));
            spm = static_cast<const SparseMatrix *>(elem_matr);
            SA_ASSERT(const_cast<SparseMatrix *>(spm)->Finalized());
            SA_ASSERT(spm->Width() == ndofs);
            SA_ASSERT(spm->Size() == ndofs);
#if (SA_IS_DEBUG_LEVEL(2))
            SA_ALERT_COND_MSG(SA_IS_REAL_EQ((*spm)(dii, djj),
                                            (*spm)(djj, dii)),
                              "Non-symmetric element matrix!!! Difference: %g",
                              (*spm)(dii, djj) - (*spm)(djj, dii));
#endif
            value += (*spm)(dii, djj);
            if (free_matr)
                delete spm;
        }
    }

    return value;
}

/* Functions */

/**
   Adding ATB 4 May 2015 to extract agg_flags (which we use and I think
   is misnamed) from the aggregate/preaggregate stuff, which we want to
   (eventually) remove.

   I am not sure where this fits, when we do it, how we use it.

   As of 5 January 2016 I think we should also set ON_PROC_IFACE_FLAG here
   for coarser levels
*/
void agg_construct_agg_flags(agg_partitioning_relations_t& agg_part_rels,
                             const agg_dof_status_t *bdr_dofs)
{
    const int ND = agg_part_rels.dof_to_AE->Size();
    SA_ASSERT(!agg_part_rels.agg_flags);
    agg_part_rels.agg_flags = new agg_dof_status_t[ND];
    for (int i=0; i<ND; ++i)
    {
        if (bdr_dofs) // note well this is pointer existence, not bdr_dofs[i] value
            agg_part_rels.agg_flags[i] = bdr_dofs[i];
        else
            agg_part_rels.agg_flags[i] = 0;
        if (SA_IS_SET_A_FLAG(agg_part_rels.agg_flags[i], AGG_ON_PROC_IFACE_FLAG) ||
            agg_part_rels.dof_to_AE->RowSize(i) > 1)
        {
            SA_SET_FLAGS(agg_part_rels.agg_flags[i], AGG_BETWEEN_AES_FLAG);
        }
    }
}

int *agg_preconstruct_aggregates(const Table& dof_to_AE, int nparts,
                                 const agg_dof_status_t *bdr_dofs,
                                 int *& preagg_size,
                                 agg_dof_status_t *& agg_flags)
{
    SA_ASSERT(dof_to_AE.Width() == nparts);
    const int ND = dof_to_AE.Size();
    int *preaggregates = new int[ND];
    preagg_size = new int[nparts];
    agg_flags = new agg_dof_status_t[ND];

    SA_ASSERT(preagg_size);
    SA_ASSERT(preaggregates);
    SA_ASSERT(agg_flags);

    memset(preagg_size, 0, sizeof(*preagg_size)*nparts);

    // Loop over all DoFs and distribute the trivial ones. The non-trivial ones
    // get marked.
    for (int i=0; i < ND; ++i)
    {
        // Copy the currently known flags for DoF i.
        if (bdr_dofs)
            agg_flags[i] = bdr_dofs[i];
        else
            agg_flags[i] = 0;

        // If DoF i is in more than one AE.
        if (SA_IS_SET_A_FLAG(agg_flags[i], AGG_ON_PROC_IFACE_FLAG) ||
            dof_to_AE.RowSize(i) > 1)
        {
            preaggregates[i] = -2; // Mark as non-trivial.
            SA_SET_FLAGS(agg_flags[i], AGG_BETWEEN_AES_FLAG);
        } 
        else
        {
            SA_ASSERT(1 == dof_to_AE.RowSize(i));
            int part = dof_to_AE.GetRow(i)[0];
            SA_ASSERT(0 <= part && part < nparts);
            preaggregates[i] = part;
            ++preagg_size[part];
        }
    }

    return preaggregates;
}

/**
   Ensure consistent ordering across processors.

   dofs contain dofs on entry, on exit contains dofs
   sorted according to their *TrueDof* numbers
*/
void SortByTrueDof(Array<int> &dofs, HypreParMatrix& DofTrueDof)
{
    hypre_ParCSRMatrix * h_dof_truedof = DofTrueDof;
    hypre_CSRMatrix * dtd_diag = h_dof_truedof->diag;
    hypre_CSRMatrix * dtd_offd = h_dof_truedof->offd;
    int * dtd_diag_I = dtd_diag->i;
    int * dtd_diag_J = dtd_diag->j;
    int * dtd_offd_I = dtd_offd->i;
    int * dtd_offd_J = dtd_offd->j;

    std::vector<std::pair<int, int> > pairs;

    for (int k=0; k<dofs.Size(); ++k)
    {
        int truedof;
        int dof = dofs[k];
        if (dtd_diag_I[dof+1] != dtd_diag_I[dof])
            truedof = dtd_diag_J[dtd_diag_I[dof]];
        else
            truedof = dtd_offd_J[dtd_offd_I[dof]];
        pairs.push_back(std::make_pair(truedof,dof));
    }
    std::sort(pairs.begin(),pairs.end());

    for (int k=0; k<dofs.Size(); ++k)
        dofs[k] = pairs[k].second;
}

/**
   A (more or less) drop-in replacement for agg_construct_mises_local
   on the coarsest mesh, where we don't need mises, we can
   use the older concept of "aggregates", that is, one
   set per agglomerated element. Note well that we still 
   use MIS terminology and data structures to store aggregate
   information.

   This borrows heavily from DK's agg_construct_aggregates,
   agg_preconstruct_aggregates, and agg_resolve_aggregates

   Need to fill:
     agg_part_rels.mis_to_dof
                  .truemis_to_dof
                  .mises
                  .num_mises
                  .mis_master

   and return:
     number of owned mises local to this processor

   ATB 29 December 2015
*/
int agg_construct_aggregate_mises(
    HypreParMatrix &Aglobal,
    agg_partitioning_relations_t& agg_part_rels,
    HypreParMatrix * Dof_to_gAE,
    const agg_dof_status_t *previous_flags)
{
    SA_ASSERT(agg_part_rels.dof_to_AE);
    const Table& dof_to_AE = *agg_part_rels.dof_to_AE;

    SA_ASSERT(!agg_part_rels.mises);
    SA_ASSERT(!agg_part_rels.mises_size);
    SA_ASSERT(!agg_part_rels.num_mises);
    SA_ASSERT(!agg_part_rels.agg_flags); // ? see agg_construct_agg_flags()

    SA_ASSERT(dof_to_AE.Width() == agg_part_rels.nparts);
    const int nparts = agg_part_rels.nparts;
    const int ND = dof_to_AE.Size();
    SA_ASSERT(Dof_to_gAE->GetNumRows() == ND); 
    int *preaggregates = new int[ND];
    int *preagg_size = new int[nparts];
    agg_dof_status_t *agg_flags = new agg_dof_status_t[ND];

    SA_ASSERT(preagg_size);
    SA_ASSERT(preaggregates);
    SA_ASSERT(agg_flags);

    memset(preagg_size, 0, sizeof(*preagg_size)*nparts);

    // sec is only use sec here to determine ownership of dofs
    SharedEntityCommunication<DenseMatrix> sec(PROC_COMM, *agg_part_rels.Dof_TrueDof);
    // Loop over all (shared) DoFs and distribute the trivial ones. The non-trivial ones
    // get marked.
    for (int i=0; i < ND; ++i)
    {
        // agg_construct_agg_flags() does the same thing as below, probably better... [TODO]
        if (previous_flags) // note well this checks allocation of pointer, not value of previous_flags[i]
            agg_flags[i] = previous_flags[i];
        else
            agg_flags[i] = 0;

        // If DoF i is in more than one AE or does not belong to this processor
        if ((sec.Owner(i) != PROC_RANK) ||
            SA_IS_SET_A_FLAG(agg_flags[i], AGG_ON_PROC_IFACE_FLAG) ||
            dof_to_AE.RowSize(i) > 1)
        {
            preaggregates[i] = -2; // Mark as non-trivial.
            SA_SET_FLAGS(agg_flags[i], AGG_BETWEEN_AES_FLAG);
        }
        else
        {
            SA_ASSERT(1 == dof_to_AE.RowSize(i));
            int part = dof_to_AE.GetRow(i)[0];
            SA_ASSERT(0 <= part && part < nparts);
            preaggregates[i] = part; // this is local...
            ++preagg_size[part];
        }
    }
    // return preaggregates;

    agg_part_rels.num_mises = agg_part_rels.nparts;
    agg_part_rels.mises = new int[agg_part_rels.ND];
    agg_part_rels.mises_size = new int[agg_part_rels.nparts];

    // mises, mises_size get the preaggregate version [TODO: std::copy]
    memcpy(agg_part_rels.mises_size, preagg_size, agg_part_rels.nparts * sizeof(*preagg_size));
    memcpy(agg_part_rels.mises, preaggregates, ND * sizeof(*preaggregates));
    delete [] preagg_size;
    delete [] preaggregates;

    // now try to resolve somehow...
    SA_ASSERT(agg_part_rels.mises);
    SA_ASSERT(agg_part_rels.mises_size);

    Arbitrator arbitrator(Aglobal, agg_part_rels, agg_flags);

    // Loop over all (shared) DoFs.
    for (int i=0; i < ND; ++i)
    {
        bool owned;
        owned = (sec.Owner(i) == PROC_RANK);
        // SA_ASSERT(owned == SA_IS_SET_A_FLAG(agg_flags[i], AGG_OWNED_FLAG)); // TODO actually figure out flags

        // the only reason for the if branching below is assertions
        // If DoF i is marked as non-trivial and is not on a process's interface.
        if (-2 == agg_part_rels.mises[i] &&
            !SA_IS_SET_A_FLAG(agg_flags[i], AGG_ON_PROC_IFACE_FLAG))
        {
            SA_ASSERT(dof_to_AE.RowSize(i) > 1);
            SA_ASSERT(owned);

            const int agg = arbitrator.suggest(i);
            SA_ASSERT(0 <= agg && agg < agg_part_rels.nparts);
            agg_part_rels.mises[i] = agg;
            ++agg_part_rels.mises_size[agg];
        }
        // if DoF i is on a process interface
        else if (SA_IS_SET_A_FLAG(agg_flags[i], AGG_ON_PROC_IFACE_FLAG))
        {
            if (owned)
            {
                const int agg = arbitrator.suggest(i);
                SA_ASSERT(0 <= agg && agg < agg_part_rels.nparts);
                agg_part_rels.mises[i] = agg;
                ++agg_part_rels.mises_size[agg];
            }
            else
            {
                agg_part_rels.mises[i] = -1;
            }
        }
        // DoF i should be trivial (ie, already aggregated)
        else
        {
            SA_ASSERT( (!owned && agg_part_rels.mises[i] == -1) ||
                       (agg_part_rels.mises[i] >= 0 && agg_part_rels.mises[i] < nparts) );
        }
    }

#if (SA_IS_DEBUG_LEVEL(3))
    {
        int nzero = 0;
        for (int i=0; i < agg_part_rels.nparts; ++i)
        {
            if (!agg_part_rels.mises_size[i])
            {
                ++nzero;
                SA_ALERT_MSG("Zero aggregate: %d", i);
            }
        }
        SA_ASSERT(!nzero); // Empty aggregate(s)
    }
#endif

    agg_part_rels.agg_flags = agg_flags; // ??? do we need these? are we replacing something else?

    Table * dof_to_mis = new Table;
    dof_to_mis->MakeI(ND);
    for (int i=0; i<ND; ++i)
    {
        int mis = agg_part_rels.mises[i];
        if (mis >= 0)
            dof_to_mis->AddAColumnInRow(i);
        // else ???
    }
    dof_to_mis->MakeJ();
    for (int i=0; i<ND; ++i)
    {
        int mis = agg_part_rels.mises[i];
        if (mis >= 0)
            dof_to_mis->AddConnection(i,mis);
        // else ???
    }
    dof_to_mis->ShiftUpI(); // I hate the MFEM table interface, really hate it
    dof_to_mis->Finalize();
    agg_part_rels.mis_to_dof = Transpose(*dof_to_mis);
    agg_part_rels.truemis_to_dof = Transpose(*dof_to_mis);
    delete dof_to_mis;

    agg_part_rels.mis_master = new int[agg_part_rels.num_mises];
    for (int i=0; i<agg_part_rels.num_mises; ++i)
        agg_part_rels.mis_master[i] = PROC_RANK;

    return agg_part_rels.nparts;  // seems fairly unnecessary?
}

/**
   Inspired by serial SAAMGE ATB 1 April 2015
   but now heavily modified to work in parallel
   
   Dof_to_gAE has (shared) local Dofs on rows, global AE on columns,
   ie it is Dof_TrueDof (from the ParFES) * TrueDof_to_AE (from agg_construct_mises_parallel)

   The most important outputs are agg_part_rels.mis_to_dof and truemis_to_dof,
   but we construct several other things (maybe we don't need all of them, could think about saving memory)

   Returns local number of mises
*/
int agg_construct_mises_local(agg_partitioning_relations_t& agg_part_rels,
                              HypreParMatrix * Dof_to_gAE)
{
    int num_local_ldofs = Dof_to_gAE->GetNumRows();

    hypre_ParCSRMatrix * hDof_to_gAE = *Dof_to_gAE;
    hypre_CSRMatrix * diag = hypre_ParCSRMatrixDiag(hDof_to_gAE);
    hypre_CSRMatrix * offd = hypre_ParCSRMatrixOffd(hDof_to_gAE);
    int * Dof_to_gAE_diag_I = hypre_CSRMatrixI(diag);
    int * Dof_to_gAE_diag_J = hypre_CSRMatrixJ(diag);
    int * Dof_to_gAE_offd_I = hypre_CSRMatrixI(offd);
    int * Dof_to_gAE_offd_J = hypre_CSRMatrixJ(offd);

    // count and distributed should be Arrays?
    int * count = new int[num_local_ldofs];
    int * distributed = new int[num_local_ldofs];
    agg_part_rels.mises = new int[num_local_ldofs];

    memset(count, 0, sizeof(int) * num_local_ldofs);
    memset(distributed, 0, sizeof(int) * num_local_ldofs);

    HypreParVector ones(*Dof_to_gAE);
    ones = 1.0;
    HypreParVector rowsum_vector(*Dof_to_gAE, 1);
    Dof_to_gAE->Mult(ones, rowsum_vector);

    int * rowsums = new int[num_local_ldofs];
    for (int i=0; i<num_local_ldofs; ++i)
        rowsums[i] = floor(rowsum_vector(i) + 0.5);

    // only use sec here to determine ownership of dofs
    SharedEntityCommunication<DenseMatrix> * sec; // maybe non-pointer is safer and better
    sec = new SharedEntityCommunication<DenseMatrix>(PROC_COMM, *agg_part_rels.Dof_TrueDof);

    std::vector<Array<int>* > rows;
    std::vector<bool> row_is_true;
    int num_total_rows = 0;
    Array<int> mis_master_a;

    // loop over shared local ldofs
    for (int i=0; i<num_local_ldofs; ++i)
    {
        // distributed is only set if we did distribution on this processor
        if (distributed[i]) 
            continue;

        // reset count; this may be necessary if we continue at (a), but not sure it is necessary
        memset(count, 0, sizeof(int) * num_local_ldofs);
        // loop over global AE that contain local Dof i on this processor
        for (int j=Dof_to_gAE_diag_I[i]; j<Dof_to_gAE_diag_I[i+1]; ++j)
        {
            // loop over all ldofs (could look at Transpose to try to only get ones belonging to AE)
            for (int k=0; k<num_local_ldofs; ++k)
            {
                // count every AE that dof i and dof k share
                for (int kj=Dof_to_gAE_diag_I[k]; kj<Dof_to_gAE_diag_I[k+1]; ++kj)
                    if (Dof_to_gAE_diag_J[kj] == Dof_to_gAE_diag_J[j]) 
                        count[k]++;
            }
        }
        // loop over AE that contain local Dof i on other processors
        for (int j=Dof_to_gAE_offd_I[i]; j<Dof_to_gAE_offd_I[i+1]; ++j)
        {
            // loop over all ldofs (here we have no choice, transpose not accessible)
            for (int k=0; k<num_local_ldofs; ++k)
            {
                // count every AE that dof i and dof k share
                for (int kj=Dof_to_gAE_offd_I[k]; kj<Dof_to_gAE_offd_I[k+1]; ++kj)
                    if (Dof_to_gAE_offd_J[kj] == Dof_to_gAE_offd_J[j]) 
                        count[k]++;
            }
        }

        Array<int>* newrow = new Array<int>;
        bool false_mis = false;
        for (int k=0; k<num_local_ldofs; ++k)
        {
            // all dof k that belong in the same MIS, go in the same MIS
            if ((count[k] == rowsums[k]) && (count[k] == rowsums[i]))
            {
                // check which processors share this MIS, and particularly if 
                // the owner should be a different processor
                int master_proc = sec->Owner(k);
                if (newrow->Size() == 0)
                    mis_master_a.Append(master_proc);
                else if (master_proc < mis_master_a.Last())
                {
                    mis_master_a.Last() = master_proc;
                }
                if (master_proc < PROC_RANK)
                {
                    false_mis = true;
                }
                newrow->Append(k);
            }
        }
        for (int k=0; k<newrow->Size(); ++k)
        {
            distributed[newrow->GetData()[k]] = true;
            agg_part_rels.mises[newrow->GetData()[k]] = rows.size();
        }
        SortByTrueDof(*newrow, *agg_part_rels.Dof_TrueDof);

        num_total_rows++;
        rows.push_back(newrow);
        row_is_true.push_back(!false_mis);
    }
    delete sec;

    SA_ASSERT(num_total_rows >= 0 && ((unsigned int) num_total_rows == rows.size()));
    int num_true_rows = 0;
    for (int i=0; i<num_total_rows; ++i)
       if (row_is_true[i]) num_true_rows++;
    agg_part_rels.truemis_to_dof = new Table;
    agg_part_rels.mis_to_dof = new Table;
    agg_part_rels.truemis_to_dof->MakeI(num_true_rows);
    agg_part_rels.mis_to_dof->MakeI(num_total_rows);
    agg_part_rels.num_mises = num_total_rows;
    int truerownum = 0;
    for (int i=0; i<num_total_rows; ++i)
    {
       if (row_is_true[i]) 
          agg_part_rels.truemis_to_dof->AddColumnsInRow(truerownum++, rows[i]->Size());
       agg_part_rels.mis_to_dof->AddColumnsInRow(i, rows[i]->Size());
    }
    agg_part_rels.truemis_to_dof->MakeJ();
    agg_part_rels.mis_to_dof->MakeJ();
    truerownum = 0;
    for (int i=0; i<num_total_rows; ++i)
    {
       if (row_is_true[i])
          agg_part_rels.truemis_to_dof->AddConnections(truerownum++, rows[i]->GetData(), rows[i]->Size());
       agg_part_rels.mis_to_dof->AddConnections(i, rows[i]->GetData(), rows[i]->Size());
    }
    // I hate the MFEM Table interface, with the burning passion of a million suns
    agg_part_rels.truemis_to_dof->ShiftUpI(); 
    agg_part_rels.truemis_to_dof->Finalize();
    agg_part_rels.mis_to_dof->ShiftUpI();
    agg_part_rels.mis_to_dof->Finalize();
    
    agg_part_rels.mis_master = new int[mis_master_a.Size()];
    for (int i=0; i<mis_master_a.Size(); ++i)
       agg_part_rels.mis_master[i] = mis_master_a[i];

    delete[] count;
    delete[] distributed;
    delete[] rowsums;
    for (int i=0; i<num_total_rows; ++i)
       delete rows[i];

    // SA_PRINTF("%s","---agg_construct_mises_local finished---\n");
    return num_true_rows;
}

/**
   At this point I am just throwing all the tables I can produce 
   up on the wall and seeing which ones stick.
   I am afraid I am reinventing Umberto's SharingMap, but at 
   least now I kind of understand it

   Many of these parameters should be const.

   Could probably just take agg_part_rels, the rest are all members of agg_part_rels
   or even roll this into agg_produce_mises()

   what this produces in the end is:
   agg_part_rels.mises_size
                .mis_to_dof
                .truemis_to_dof
                .mis_to_AE
                .mis_truemis
*/
int agg_construct_mises_parallel(HypreParMatrix &Aglobal,
                                 agg_partitioning_relations_t &agg_part_rels,                        
                                 Table& l_AE_to_dof, Table& l_dof_to_AE,
                                 int *& mises, int *& mises_size,
                                 const agg_dof_status_t *previous_flags,
                                 bool do_aggregates)
{
    HypreParMatrix * Dof_TrueDof = agg_part_rels.Dof_TrueDof; // pure pointer
    HypreParMatrix * TrueDof_Dof = Dof_TrueDof->Transpose();
    // hypre_par_matrix_ownership(*TrueDof_Dof, owns_data, owns_row_starts, owns_col_starts);

    // Dof_TrueDof_Dof has rows owned by Dof_TrueDof, cols by TrueDof_Dof
    HypreParMatrix * Dof_TrueDof_Dof = ParMult(Dof_TrueDof, TrueDof_Dof);

    // construct dof_to_gAE
    Array<int> row_offsets;
    int global_num_rows;
    proc_determine_offsets(l_AE_to_dof.Size(), row_offsets, global_num_rows);
    row_offsets.Append(global_num_rows); // I think MFEM handles assumed partition differently from hypre?
    SA_RPRINTF(0,"constructing AE_to_dof of global size %d by %d\n",
               global_num_rows, Dof_TrueDof->M());

    int * col_offsets = new int[3];
    col_offsets[0] = Dof_TrueDof->GetRowStarts()[0];
    col_offsets[1] = Dof_TrueDof->GetRowStarts()[1];
    col_offsets[2] = Dof_TrueDof->M(); // this is ridiculous, Tzanio

    // the local AE_to_dof has no knowledge of AEs on other processors
    HypreParMatrix * AE_to_dof = new HypreParMatrix(PROC_COMM, global_num_rows,
                                                    Dof_TrueDof->M(),
                                                    row_offsets.GetData(),
                                                    col_offsets,
                                                    &l_AE_to_dof);

    // gAE_to_Dof has row starts owned AE_to_dof, col starts owned by TrueDof_Dof
    HypreParMatrix * gAE_to_Dof = ParMult(AE_to_dof, Dof_TrueDof_Dof); 
    hypre_ParCSRMatrixDeleteZeros(*gAE_to_Dof,1.e-10); // if the matrix is square, it has wrong zeros on diagonal sometimes (rare corner case)

    HypreParMatrix * Dof_to_gAE = gAE_to_Dof->Transpose(); // should own everything
    if (agg_part_rels.testmesh)
        Dof_to_gAE->Print("Dof_to_gAE.mat");

    // construct local agg_part_rels.mis_to_dof, truemis_to_dof, other stuff; 
    int out;
    if (do_aggregates)
    {
        out = agg_construct_aggregate_mises(
            Aglobal, agg_part_rels, Dof_to_gAE, previous_flags);
    }
    else
    {
        out = agg_construct_mises_local(agg_part_rels, Dof_to_gAE);
        agg_part_rels.mises_size = new int[agg_part_rels.mis_to_dof->Size()];
        for (int i=0; i<agg_part_rels.mis_to_dof->Size(); ++i)
            agg_part_rels.mises_size[i] = agg_part_rels.mis_to_dof->RowSize(i);
    }

    // construct mis_truemis
    Array<int> row_offsets2; // these get destroyed at end of routine...
    proc_determine_offsets(agg_part_rels.truemis_to_dof->Size(), row_offsets2, global_num_rows);
    row_offsets2.Append(global_num_rows);
    Array<int> col_offsets2(3);
    for (int i=0; i<2; ++i)
        col_offsets2[i] = Dof_TrueDof->GetRowStarts()[i];
    col_offsets2[2] = Dof_TrueDof->M(); // Really don't like the MFEM constructor I'm using, Tzanio, it's the worst.
    // SA_RPRINTF(0,"    Creating truemis_to_dof of global size %d by %d.\n",global_num_rows,Dof_TrueDof->M());
    HypreParMatrix * truemis_to_dof = new HypreParMatrix(PROC_COMM, global_num_rows,
                                                         Dof_TrueDof->M(),
                                                         row_offsets2.GetData(),
                                                         col_offsets2.GetData(),
                                                         agg_part_rels.truemis_to_dof);
    Array<int> row_offsets3; // these get destroyed at end of routine, so the matrix better too.
    proc_determine_offsets(agg_part_rels.mis_to_dof->Size(), row_offsets3, global_num_rows);
    row_offsets3.Append(global_num_rows);
    // SA_RPRINTF(0,"Creating mis_to_dof of global size %d by %d.\n",global_num_rows,Dof_TrueDof->M());
    HypreParMatrix * mis_to_dof = new HypreParMatrix(PROC_COMM, global_num_rows,
                                                     Dof_TrueDof->M(),
                                                     row_offsets3.GetData(),
                                                     col_offsets2.GetData(),
                                                     agg_part_rels.mis_to_dof);

    if (do_aggregates)
        agg_part_rels.mis_to_AE = IdentityTable(agg_part_rels.num_mises);
    else
        agg_part_rels.mis_to_AE = Mult(*agg_part_rels.mis_to_dof, l_dof_to_AE);
    agg_part_rels.AE_to_mis = Transpose(*agg_part_rels.mis_to_AE);
    if (agg_part_rels.testmesh)
    {
        std::stringstream filename;
        filename << "mis_to_AE." << PROC_RANK << ".table";
        std::ofstream out(filename.str().c_str());
        agg_part_rels.mis_to_AE->Print(out);
    }

    // TODO: wait, why can't I just do mis_to_dof * dof_to_truemis? why Dof_TrueDof_Dof?
    HypreParMatrix * temp = ParMult(mis_to_dof, Dof_TrueDof_Dof); // rows owned by row_offsets3, cols by TrueDof_Dof
    HypreParMatrix * dof_to_truemis = truemis_to_dof->Transpose();

    if (do_aggregates)
    {
        hypre_ParCSRMatrix * h_temp = hypre_IdentityParCSRMatrix(
            PROC_COMM, global_num_rows, row_offsets3.GetData());
        agg_part_rels.mis_truemis = new HypreParMatrix(h_temp);
        mbox_make_owner_rowstarts_colstarts(*agg_part_rels.mis_truemis);
    }
    else
    {
        agg_part_rels.mis_truemis = ParMult(temp, dof_to_truemis); // rows owned by row_offsets3, cols by dof_to_truemis
        mbox_make_owner_rowstarts_colstarts(*agg_part_rels.mis_truemis);
    }

    if (agg_part_rels.testmesh) 
       agg_part_rels.mis_truemis->Print("mis_truemis.mat");

    delete AE_to_dof;
    delete TrueDof_Dof;
    delete Dof_TrueDof_Dof;
    delete temp;
    delete dof_to_truemis;

    delete [] col_offsets;

    delete truemis_to_dof;
    delete mis_to_dof;

    delete gAE_to_Dof;
    delete Dof_to_gAE;

    // SA_RPRINTF(0,"%s","--- end agg_construct_mises_parallel() ---\n");

    return out;
}


/**
   This is some kind of mix of agg_construct_aggregates()
   and agg_produce_mises_refaggs() from serial SAAMGE.
   This is how we replaced aggregates with minimal intersection sets.
   Added ATB 30 March 2015

   For usual MIS algorithm, we don't use bdr_dofs at all
   For MIS-aggregate thing on coarsest level, we might want it
   bdr_dofs is not really the right name, it contains flags
   for whether DOF crosses processors, which we might use
   in aggregate arbitration.

   do_aggregates says whether to do aggregates or usual MISes
*/
void agg_produce_mises(HypreParMatrix& Aglobal,
                       agg_partitioning_relations_t& agg_part_rels,
                       const agg_dof_status_t *bdr_dofs,
                       bool do_aggregates)
{
    SA_ASSERT(agg_part_rels.AE_to_dof);
    SA_ASSERT(agg_part_rels.dof_to_AE);

    // MISes
    SA_ASSERT(!agg_part_rels.num_owned_mises);
    SA_ASSERT(!agg_part_rels.mises);
    SA_ASSERT(!agg_part_rels.mises_size);
    SA_ASSERT(!agg_part_rels.truemis_to_dof);

    agg_part_rels.num_owned_mises = agg_construct_mises_parallel(
        Aglobal, agg_part_rels, *(agg_part_rels.AE_to_dof),
        *(agg_part_rels.dof_to_AE), 
        agg_part_rels.mises, agg_part_rels.mises_size,
        bdr_dofs, do_aggregates);

    SA_RPRINTF(0,"Total number of MISes = %d\n",
               agg_part_rels.mis_truemis->GetGlobalNumRows());
}

SparseMatrix *agg_build_AE_stiffm_with_global(const SparseMatrix& A, int part,
    const agg_partitioning_relations_t& agg_part_rels,
    ElementMatrixProvider *data, bool bdr_cond_imposed,
    bool assemble_ess_diag)
{
    const int * const row = agg_part_rels.AE_to_dof->GetRow(part);
    const int rs = agg_part_rels.AE_to_dof->RowSize(part);
    SparseMatrix *AE_stiffm = new SparseMatrix(rs, rs);
    bool *diag = new bool[rs];

    memset(diag, 0, sizeof(*diag)*rs);

    SA_ASSERT(agg_part_rels.partitioning);
    SA_ASSERT(agg_part_rels.agg_flags); 
    SA_ASSERT(AE_stiffm);

    for (int i=0; i < rs; ++i)
    {
        int glob_dof, row_start, row_size;
        int *neighbours;
        double *neigh_data;

        glob_dof = row[i];
        SA_ASSERT(0 <= glob_dof && glob_dof < agg_part_rels.ND);
        row_start = A.GetI()[glob_dof];
        neighbours = &(A.GetJ()[row_start]);
        neigh_data = &(A.GetData()[row_start]);
        row_size = const_cast<SparseMatrix&>(A).RowSize(glob_dof);
        for (int j=0; j < row_size; ++j)
        {
            int glob_neigh, local_neigh;

            glob_neigh = neighbours[j];

            SA_ASSERT(0 <= glob_neigh && glob_neigh < agg_part_rels.ND);
#if (SA_IS_DEBUG_LEVEL(2))
            SA_ASSERT((agg_elem_in_col(glob_neigh, part, *agg_part_rels.dof_to_AE) >= 0 &&
                       0 <= agg_map_id_glob_to_AE(glob_neigh, part, agg_part_rels)) ||
                      (agg_elem_in_col(glob_neigh, part, *agg_part_rels.dof_to_AE) < 0 &&
                       0 > agg_map_id_glob_to_AE(glob_neigh, part, agg_part_rels)));
#endif

            if (agg_elem_in_col(glob_neigh, part, *agg_part_rels.dof_to_AE) < 0)
                continue;

            local_neigh = agg_map_id_glob_to_AE(glob_neigh, part,
                                                agg_part_rels);

            SA_ASSERT(0 <= local_neigh && local_neigh < rs);

            if (SA_IS_SET_A_FLAG(agg_part_rels.agg_flags[glob_dof], AGG_BETWEEN_AES_FLAG) &&
                SA_IS_SET_A_FLAG(agg_part_rels.agg_flags[glob_neigh], AGG_BETWEEN_AES_FLAG) &&

                !(bdr_cond_imposed &&
                  (SA_IS_SET_A_FLAG(agg_part_rels.agg_flags[glob_dof],
                                    AGG_ON_ESS_DOMAIN_BORDER_FLAG) ||
                   SA_IS_SET_A_FLAG(agg_part_rels.agg_flags[glob_neigh],
                                    AGG_ON_ESS_DOMAIN_BORDER_FLAG)) &&
                  !(assemble_ess_diag && glob_neigh == glob_dof))
                )
            {
                double value;

                if (i < local_neigh || (i == local_neigh && !(diag[i])))
                {
                    value = agg_assemble_value(glob_dof, glob_neigh, part,
                                               agg_part_rels, data);
                    if (0. != value)
                        AE_stiffm->Set(i, local_neigh, value);
                    if (i != local_neigh)
                    {
                        SA_ASSERT(i < local_neigh);
                        if (0. != value)
                            AE_stiffm->Set(local_neigh, i, value);
                    } else
                        diag[i] = true;
                }
            } else
            {
                if (0. != neigh_data[j])
                    AE_stiffm->Set(i, local_neigh, neigh_data[j]);
            }
        }
    }

    AE_stiffm->Finalize();
    delete [] diag;
    return AE_stiffm;
}

SparseMatrix **agg_build_AEs_stiffm_with_global(const SparseMatrix& A,
    const agg_partitioning_relations_t& agg_part_rels,
    ElementMatrixProvider *data, bool bdr_cond_imposed,
    bool assemble_ess_diag)
{
    SA_ASSERT(agg_part_rels.nparts > 0);
    SparseMatrix **arr = new SparseMatrix*[agg_part_rels.nparts];

    SA_ASSERT(arr);
    for (int i=0; i < agg_part_rels.nparts; ++i)
        arr[i] = agg_build_AE_stiffm_with_global(A, i, agg_part_rels,
                                                 data,
                                                 bdr_cond_imposed,
                                                 assemble_ess_diag);

    return arr;
}

bool check_diagonal(const SparseMatrix &mat)
{
    bool out = false;
    for (int i=0; i<mat.Height(); ++i)
        if (mat(i,i) <= 0.0)
        {
            SA_PRINTF("  A(diag)[%d] = %e\n", i, mat(i,i));
            out = true;
        }
    return out;
}

SparseMatrix *agg_build_AE_stiffm(int part,
                                  const agg_partitioning_relations_t& agg_part_rels,
                                  ElementMatrixProvider *data)
{
    bool free_matr;
    const int * const AEelems = agg_part_rels.AE_to_elem->GetRow(part);
    const int num_AEelems = agg_part_rels.AE_to_elem->RowSize(part);
    const int num_AEdofs = agg_part_rels.AE_to_dof->RowSize(part);
    SparseMatrix *AE_stiffm = new SparseMatrix(num_AEdofs, num_AEdofs);
    int i, j, k;
    double el;

    // Loop over all elements on the "fine" level and assemble the local
    // stiffness matrix.

    SA_ASSERT(num_AEelems > 0);

    int elem = AEelems[i=0];
    SA_ASSERT(0 <= elem && elem < agg_part_rels.elem_to_dof->Size());
    // elmat_callback is probably elmat_parallel() on coarser levels
    // const Matrix *matr = elmat_callback(elem, &agg_part_rels, data, free_matr);
    const Matrix *matr = data->GetMatrix(elem, free_matr);

    const SparseMatrix *elem_matr = dynamic_cast<const SparseMatrix *>(matr);
    if (elem_matr) //The element matrices are sparse.
    {
        for (;;)
        {
            SA_ASSERT(elem_matr);
            SA_ASSERT(const_cast<SparseMatrix *>(elem_matr)->Finalized());
            const int * const I = elem_matr->GetI();
            const int * const J = elem_matr->GetJ();
            const double * const Data = elem_matr->GetData();
            const int elem_matr_sz = elem_matr->Size();
            SA_ASSERT(elem_matr->Width() == elem_matr_sz);
            SA_ASSERT(agg_part_rels.elem_to_dof->RowSize(elem) == elem_matr_sz);
            const int * const elemdofs =
                agg_part_rels.elem_to_dof->GetRow(elem);

            // Add the contribution of the current element matrix.
            for (k=0; k < elem_matr_sz; ++k)
            {
                SA_ASSERT(0 <= elemdofs[k] && elemdofs[k] < agg_part_rels.ND);
                const int local_k = agg_map_id_glob_to_AE(elemdofs[k], part,
                                                          agg_part_rels);
                SA_ASSERT(0 <= local_k && local_k < num_AEdofs);
                for (j = I[k]; j < I[k+1]; ++j)
                {
                    SA_ASSERT(0 <= J[j] && J[j] < elem_matr_sz);
                    SA_ASSERT(0 <= elemdofs[J[j]] &&
                              elemdofs[J[j]] < agg_part_rels.ND);
                    el = Data[j];
                    if (0. != el)
                    {
                        const int local_j =
                            agg_map_id_glob_to_AE(elemdofs[J[j]], part,
                                                  agg_part_rels);
                        SA_ASSERT(0 <= local_j && local_j < num_AEdofs);
                        AE_stiffm->Add(local_k, local_j, el);
                    }
                }
            }
            if (free_matr)
                delete elem_matr;

            if (++i < num_AEelems)
            {
                elem = AEelems[i];
                SA_ASSERT(0 <= elem &&
                          elem < agg_part_rels.elem_to_dof->Size());
                // matr = elmat_callback(elem, &agg_part_rels, data, free_matr);
                matr = data->GetMatrix(elem, free_matr);
                elem_matr = static_cast<const SparseMatrix *>(matr);
            } else
                break;
        }
    } else //The element matrices are dense.
    {
        const DenseMatrix *elem_dmatr = dynamic_cast<const DenseMatrix *>(matr);
        for (;;)
        {
            SA_ASSERT(elem_dmatr);
            const int elem_matr_sz = elem_dmatr->Height();
            SA_ASSERT(elem_dmatr->Width() == elem_matr_sz);
            SA_ASSERT(agg_part_rels.elem_to_dof->RowSize(elem) == elem_matr_sz);
            const int * const elemdofs =
                agg_part_rels.elem_to_dof->GetRow(elem);

            // Add the contribution of the current element matrix.
            for (k=0; k < elem_matr_sz; ++k)
            {
                SA_ASSERT(0 <= elemdofs[k] && elemdofs[k] < agg_part_rels.ND);
                const int local_k = agg_map_id_glob_to_AE(elemdofs[k], part,
                                                          agg_part_rels);
                SA_ASSERT(0 <= local_k && local_k < num_AEdofs);
                for (j=0; j < elem_matr_sz; ++j)
                {
                    SA_ASSERT(0 <= elemdofs[j] &&
                              elemdofs[j] < agg_part_rels.ND);
                    el = (*elem_dmatr)(k, j);
                    if (0. != el)
                    {
                        const int local_j =
                            agg_map_id_glob_to_AE(elemdofs[j], part,
                                                  agg_part_rels);
                        SA_ASSERT(0 <= local_j && local_j < num_AEdofs);
                        AE_stiffm->Add(local_k, local_j, el);
                    }
                }
            }
            if (free_matr)
                delete elem_dmatr;

            if (++i < num_AEelems)
            {
                elem = AEelems[i];
                SA_ASSERT(0 <= elem &&
                          elem < agg_part_rels.elem_to_dof->Size());
                // matr = elmat_callback(elem, &agg_part_rels, data, free_matr);
                matr = data->GetMatrix(elem, free_matr);
                elem_dmatr = static_cast<const DenseMatrix *>(matr);
            } else
                break;
        }
    }

    AE_stiffm->Finalize();
    return AE_stiffm;
}

SparseMatrix *agg_simple_assemble(
    const agg_partitioning_relations_t& agg_part_rels,
    const ElementMatrixProvider &data, bool assem_skip_zeros,
    bool final_skip_zeros, bool finalize, const agg_dof_status_t *bdr_dofs,
    const Vector *sol, Vector *rhs, bool keep_diag)
{
    SA_PRINTF_L(4, "%s", "Assembling matrix...\n");
    SA_ASSERT(agg_part_rels.elem_to_dof);
    SA_ASSERT(agg_part_rels.dof_to_elem);
    bool free_matr;
    const Table& elem_to_dof = *agg_part_rels.elem_to_dof;
    const int num_dofs = agg_part_rels.dof_to_elem->Size();
    const int elems_num = elem_to_dof.Size();
    int i, k, j;
    double el;
    SA_ASSERT(num_dofs > 0);
    SparseMatrix *glob_matr = new SparseMatrix(num_dofs);

    SA_ASSERT(elems_num > 0);

    // const Matrix *matr = elmat_callback(i=0, &agg_part_rels, data, free_matr);
    i = 0;
    const Matrix *matr = data.GetMatrix(i, free_matr);

    const SparseMatrix *elem_matr = dynamic_cast<const SparseMatrix *>(matr);
    if (elem_matr) //The element matrices are sparse.
    {
        for (;;)
        {
            SA_ASSERT(elem_matr);
            SA_ASSERT(const_cast<SparseMatrix *>(elem_matr)->Finalized());
            SA_ASSERT(elem_matr->Size() == elem_matr->Width());
            const int sz = elem_matr->Size();
            SA_ASSERT(sz <= num_dofs);
            const int * const I = elem_matr->GetI();
            const int * const J = elem_matr->GetJ();
            const double * const Data = elem_matr->GetData();
            SA_ASSERT(elem_to_dof.RowSize(i) == sz);
            const int * const map = elem_to_dof.GetRow(i);

            for (k=0; k < sz; ++k)
            {
                const int glob_k = map[k];
                SA_ASSERT(0 <= glob_k && glob_k < num_dofs);
                for (j = I[k]; j < I[k+1]; ++j)
                {
                    SA_ASSERT(0 <= J[j] && J[j] < sz);
                    SA_ASSERT(0 <= map[J[j]] && map[J[j]] < num_dofs);
                    el = Data[j];
                    if (assem_skip_zeros && 0. == el)
                        continue;
                    glob_matr->Add(glob_k, map[J[j]], el);
                }
            }
            if (free_matr)
                delete elem_matr;

            if (++i < elems_num)
            {
                // matr = elmat_callback(i, &agg_part_rels, data, free_matr);
                matr = data.GetMatrix(i, free_matr);
                SA_ASSERT(dynamic_cast<const SparseMatrix *>(matr));
                elem_matr = static_cast<const SparseMatrix *>(matr);
            } else
                break;
        }
    } else //The element matrices are dense.
    {
        SA_ASSERT(dynamic_cast<const DenseMatrix *>(matr));
        const DenseMatrix *elem_dmatr = static_cast<const DenseMatrix *>(matr);
        for (i=0;;)
        {
            SA_ASSERT(elem_dmatr);
            SA_ASSERT(elem_dmatr->Height() == elem_dmatr->Width());
            const int sz = elem_dmatr->Height();
            SA_ASSERT(sz <= num_dofs);
            SA_ASSERT(elem_to_dof.RowSize(i) == sz);
            const int * const map = elem_to_dof.GetRow(i);

#if 0
            for (k=0; k < sz; ++k)
            {
                const int glob_k = map[k];
                SA_ASSERT(0 <= glob_k && glob_k < num_dofs);
                for (j=0; j < sz; ++j)
                {
                    SA_ASSERT(0 <= map[j] && map[j] < num_dofs);
                    el = (*elem_dmatr)(k, j);
                    if (assem_skip_zeros && 0. == el)
                        continue;
                    glob_matr->Add(glob_k, map[j], el);
                }
            }
#else
            Array<int> rowcol((int *)map, sz);
            glob_matr->AddSubMatrix(rowcol, rowcol, *elem_dmatr,
                                    assem_skip_zeros);
#endif
            if (free_matr)
                delete elem_dmatr;

            if (++i < elems_num)
            {
                // matr = elmat_callback(i, &agg_part_rels, data, free_matr);
                matr = data.GetMatrix(i, free_matr);
                SA_ASSERT(dynamic_cast<const DenseMatrix *>(matr));
                elem_dmatr = static_cast<const DenseMatrix *>(matr);
            } else
                break;
        }
    }

    if (bdr_dofs)
    {
        SA_ASSERT(sol);
        SA_ASSERT(rhs);
        agg_eliminate_essential_bc(*glob_matr, bdr_dofs, *sol, *rhs, keep_diag);
    }

    if (finalize)
        glob_matr->Finalize(final_skip_zeros);
    return glob_matr;
}

SparseMatrix *agg_simple_assemble(const Matrix * const *elem_matrs,
                                  Table& elem_to_dof, Table& dof_to_elem,
                                  bool assem_skip_zeros, bool final_skip_zeros,
                                  bool finalize,
                                  const agg_dof_status_t *bdr_dofs,
                                  const Vector *sol, Vector *rhs,
                                  bool keep_diag)
{
    agg_partitioning_relations_t agg_part_rels;
    memset(&agg_part_rels, 0, sizeof(agg_part_rels));
    agg_part_rels.elem_to_dof = &elem_to_dof;
    agg_part_rels.dof_to_elem = &dof_to_elem;
    ElementMatrixArray emp(agg_part_rels, elem_matrs);
    return agg_simple_assemble(agg_part_rels, 
                               emp, assem_skip_zeros,
                               final_skip_zeros, finalize, bdr_dofs, sol, rhs,
                               keep_diag);
}

/**
  agglomerate restrict to aggregate (not confusing at all)
*/
int *agg_restrict_to_agg(int part, const Table& AE_to_dof,
                         const int *aggregates, int agg_id, int agg_size,
                         DenseMatrix& cut_evects, DenseMatrix& restricted)
{
    const int * const AE_topol = AE_to_dof.GetRow(part);
    const int AE_sz = AE_to_dof.RowSize(part);
    int *restriction = new int[agg_size];
    int i;

    SA_PRINTF_L(9, "AE size (DoFs): %d, aggregate size: %d\n", AE_sz, agg_size);

    SA_ASSERT(agg_size > 0 && AE_sz > 0);
    SA_ASSERT(cut_evects.Height() == AE_sz);

    if (agg_size == AE_sz)
    {
        restricted = cut_evects;
        memcpy(restriction, AE_topol, sizeof(*restriction)*AE_sz);
        return restriction;
    }

    SA_ASSERT(agg_size < AE_sz);
    restricted.Transpose(cut_evects);
    const int mult = restricted.Height();
    int j;
    double * const data = restricted.Data();
    for (i=j=0; i < agg_size; (++i), (++j))
    {
        SA_ASSERT(j < AE_sz);
        while (aggregates[AE_topol[j]] != agg_id)
        {
            ++j;
            SA_ASSERT(j < AE_sz);
        }
        SA_ASSERT(i <= j);
        restriction[i] = AE_topol[j];
        if (i < j)
            memcpy(data + i*mult, data + j*mult, sizeof(*data)*mult);
    }
#if (SA_IS_DEBUG_LEVEL(3))
    for (; j < AE_sz; ++j)
        SA_ASSERT(aggregates[AE_topol[j]] != agg_id);
#endif
    restricted.UseExternalData(data, mult, agg_size);
    restricted.Transpose();
    return restriction;
}

/**
   copied from serial SAAMGE ATB 21 April 2015
*/
void agg_restrict_to_agg_enforce(
    int part,
    const agg_partitioning_relations_t& agg_part_rels, int agg_size,
    const int *restriction, DenseMatrix& cut_evects,
    DenseMatrix& restricted)
{
    int i;
    SA_ASSERT(agg_part_rels.AE_to_dof);

    SA_PRINTF_L(9, "AE size (DoFs): %d, aggregate size: %d\n",
                agg_part_rels.AE_to_dof->RowSize(part), agg_size);

    SA_ASSERT(agg_size > 0 &&
              agg_part_rels.AE_to_dof->RowSize(part) >= agg_size);
    SA_ASSERT(cut_evects.Height() == agg_part_rels.AE_to_dof->RowSize(part));

    const int num_vects = cut_evects.Width();
    const DenseMatrix trans_cut_evects(cut_evects, 't');
    restricted.SetSize(num_vects, agg_size);

    const double * const tce_data = trans_cut_evects.Data();
    double *res_data = restricted.Data();

    // Do the restriction.
    for (i=0; i < agg_size; (++i), (res_data += num_vects))
    {
        SA_ASSERT(0 <= restriction[i] && restriction[i] < agg_part_rels.ND);
        const int AE_dof = agg_map_id_glob_to_AE(restriction[i], part,
                                                 agg_part_rels);
        SA_ASSERT(0 <= AE_dof &&
                  AE_dof < agg_part_rels.AE_to_dof->RowSize(part));
        memcpy(res_data, tce_data + AE_dof * num_vects, sizeof(*res_data) *
                                                        num_vects);
    }

    restricted.Transpose();
}

void agg_restrict_vect_to_AE(int part,
                             const agg_partitioning_relations_t& agg_part_rels,
                             const Vector& glob_vect, Vector& restricted)
{
    const int * const AE_topol = agg_part_rels.AE_to_dof->GetRow(part);
    const int AE_sz = agg_part_rels.AE_to_dof->RowSize(part);

    SA_ASSERT(glob_vect.Size() == agg_part_rels.dof_to_AE->Size());

    restricted.SetSize(AE_sz);

    for (int i=0; i < AE_sz; ++i)
    {
        int glob_num;

        glob_num = AE_topol[i];
        SA_ASSERT(glob_num < glob_vect.Size());
        restricted(i) = glob_vect(glob_num);
    }
}

void agg_build_glob_to_AE_id_map(agg_partitioning_relations_t& agg_part_rels)
{
    SA_ASSERT(agg_part_rels.AE_to_dof);
    SA_ASSERT(agg_part_rels.dof_to_AE);
    SA_ASSERT(!agg_part_rels.dof_id_inAE);
    SA_ASSERT(agg_part_rels.AE_to_dof->Size() == agg_part_rels.nparts);
    SA_ASSERT(agg_part_rels.dof_to_AE->Size() == agg_part_rels.ND);
    const Table& AE_to_dof = *agg_part_rels.AE_to_dof;
    const Table& dof_to_AE = *agg_part_rels.dof_to_AE;
    agg_part_rels.dof_id_inAE = new int[dof_to_AE.Size_of_connections()];

#if (SA_IS_DEBUG_LEVEL(3))
    memset(agg_part_rels.dof_id_inAE, -1,
           sizeof(*agg_part_rels.dof_id_inAE) *
                  dof_to_AE.Size_of_connections());
#endif

    const int * const I = dof_to_AE.GetI();
    // loop over AEs
    for (int i=0; i < agg_part_rels.nparts; ++i)
    {
        const int * const row = AE_to_dof.GetRow(i);
        const int rs = AE_to_dof.RowSize(i);
        // loop over dofs in AE
        for (int j=0; j < rs; ++j)
        {
            const int dof = row[j];
            SA_ASSERT(0 <= dof && dof < agg_part_rels.ND);
            SA_ASSERT(agg_elem_in_col(dof, i, dof_to_AE) >= 0);
            const int idx = agg_elem_in_col(dof, i, dof_to_AE) + I[dof];
            SA_ASSERT(0 <= idx && idx < dof_to_AE.Size_of_connections());
#if (SA_IS_DEBUG_LEVEL(3))
            SA_ASSERT(0 > agg_part_rels.dof_id_inAE[idx]);
#endif
            agg_part_rels.dof_id_inAE[idx] = j;
        }
    }

#if (SA_IS_DEBUG_LEVEL(3))
    for (int i=0; i < dof_to_AE.Size_of_connections(); ++i)
        SA_ASSERT(0 <= agg_part_rels.dof_id_inAE[i]);
#endif
}

/**
   modified so we can pass a list of cells to not agglomerate
   (ie, the new part is isolated_cells)
*/
agg_partitioning_relations_t *
agg_create_partitioning_fine_isolate(
    HypreParMatrix& A, int NE, Table *elem_to_dof,
    Table *elem_to_elem, int *partitioning,
    const agg_dof_status_t *bdr_dofs, int *nparts,
    HypreParMatrix *dof_truedof,
    const Array<int>& isolated_cells)
{
    agg_partitioning_relations_t *agg_part_rels =
        new agg_partitioning_relations_t;
    memset(agg_part_rels, 0, sizeof(*agg_part_rels));

    agg_part_rels->testmesh = false;
    // agg_part_rels->fes = fes;
    agg_part_rels->Dof_TrueDof = dof_truedof;
    agg_part_rels->owns_Dof_TrueDof = false;

    agg_part_rels->dof_to_elem = new Table();
    agg_part_rels->AE_to_dof = new Table();
    agg_part_rels->dof_to_AE = new Table();

    SA_ASSERT(0 < NE);

    SA_ASSERT(elem_to_elem);
    agg_part_rels->elem_to_elem = elem_to_elem;

    if (partitioning)
        agg_part_rels->partitioning = partitioning;
    else
    {
        agg_part_rels->partitioning =
            part_generate_partitioning_unweighted(*(agg_part_rels->elem_to_elem), nparts);
    }

    // do a post-processing of partitioning to put wells (or whatever) in their own partitions
    if (true)
    {
        int c_elem = *nparts;
        for (int i(0); i < isolated_cells.Size(); ++i)
        {
            int num = isolated_cells[i];
            agg_part_rels->partitioning[num] = c_elem;
            c_elem++;
        }
        *nparts = c_elem;
    }

    // removing wells may have ruined topology
    Array<int> p_array(agg_part_rels->partitioning, elem_to_elem->Size());
    connectedComponents(p_array, *elem_to_elem);
    // int actual_parts = p_array.Max() + 1;
    // *nparts = actual_parts;
    *nparts = p_array.Max() + 1;

    agg_part_rels->nparts = *nparts;
    SA_RPRINTF_L(0, 3, "Postprocessed number of partitions: %d\n", agg_part_rels->nparts);

    const bool do_aggregates = true; // this is probably a reservoir simulation application
    agg_create_partitioning_tables(agg_part_rels,
                                   A, NE, elem_to_dof,
                                   elem_to_elem, partitioning,
                                   bdr_dofs, nparts,
                                   dof_truedof, do_aggregates);
    return agg_part_rels;
}

agg_partitioning_relations_t *
agg_create_partitioning_fine(
    HypreParMatrix& A, int NE, Table *elem_to_dof,
    Table *elem_to_elem, int *partitioning,
    const agg_dof_status_t *bdr_dofs, int *nparts,
    HypreParMatrix *dof_truedof,
    bool do_aggregates, bool testmesh)
{
    agg_partitioning_relations_t *agg_part_rels =
        new agg_partitioning_relations_t;
    memset(agg_part_rels, 0, sizeof(*agg_part_rels));

    agg_part_rels->testmesh = testmesh;
    // agg_part_rels->fes = fes;
    agg_part_rels->Dof_TrueDof = dof_truedof;
    agg_part_rels->owns_Dof_TrueDof = false;

    agg_part_rels->dof_to_elem = new Table();
    agg_part_rels->AE_to_dof = new Table();
    agg_part_rels->dof_to_AE = new Table();

    SA_ASSERT(0 < NE);

    SA_ASSERT(elem_to_elem);
    agg_part_rels->elem_to_elem = elem_to_elem;

    if (partitioning)
        agg_part_rels->partitioning = partitioning;
    else
    {
        agg_part_rels->partitioning =
            part_generate_partitioning_unweighted(*(agg_part_rels->elem_to_elem), nparts);
    }

    agg_part_rels->nparts = *nparts;
    // SA_PRINTF("in agg_create_partitioning_fine(): nparts = %d\n",*nparts);

    agg_create_partitioning_tables(agg_part_rels,
                                   A, NE, elem_to_dof,
                                   elem_to_elem, partitioning,
                                   bdr_dofs, nparts,
                                   dof_truedof,
                                   do_aggregates);
    return agg_part_rels;
}

void agg_create_partitioning_tables(
    agg_partitioning_relations_t * agg_part_rels,
    HypreParMatrix& Aglobal, int NE, Table *elem_to_dof,
    Table *elem_to_elem, int *partitioning,
    const agg_dof_status_t *bdr_dofs, int *nparts,
    HypreParMatrix *dof_truedof,
    bool do_aggregates)
{

    SA_ASSERT(elem_to_dof);
    agg_part_rels->elem_to_dof = elem_to_dof;
    Transpose(*(agg_part_rels->elem_to_dof), *(agg_part_rels->dof_to_elem));

    agg_part_rels->ND = agg_part_rels->dof_to_elem->Size();

#if (SA_IS_DEBUG_LEVEL(3))
    for (int i=0; i < agg_part_rels->ND; ++i)
        SA_ASSERT(agg_part_rels->dof_to_elem->RowSize(i) > 0);
    agg_part_rels->dof_to_dof = new Table();
    SA_ASSERT(agg_part_rels->dof_to_dof);
    Mult(*(agg_part_rels->dof_to_elem), *(agg_part_rels->elem_to_dof),
         *(agg_part_rels->dof_to_dof));
#endif

    agg_construct_tables_from_arr(agg_part_rels->partitioning, NE,
                                  agg_part_rels->elem_to_AE,
                                  agg_part_rels->AE_to_elem);
    Mult(*(agg_part_rels->AE_to_elem), *(agg_part_rels->elem_to_dof),
         *(agg_part_rels->AE_to_dof));
    Transpose(*(agg_part_rels->AE_to_dof), *(agg_part_rels->dof_to_AE));
    agg_build_glob_to_AE_id_map(*agg_part_rels);

    if (agg_part_rels->testmesh)
    {
        std::stringstream filename;
        filename << "AE_to_elem." << PROC_RANK << ".table";
        std::ofstream out(filename.str().c_str());
        agg_part_rels->AE_to_elem->Print(out);
        filename.str("");
        filename << "AE_to_dof." << PROC_RANK << ".table";
        std::ofstream out2(filename.str().c_str());
        agg_part_rels->AE_to_dof->Print(out2);
        filename.str("");
        filename << "elem_to_dof." << PROC_RANK << ".table";
        std::ofstream out4(filename.str().c_str());
        agg_part_rels->elem_to_dof->Print(out4);
        filename.str("");
        filename << "dof_id_inAE." << PROC_RANK << ".array";
        std::ofstream out3(filename.str().c_str());
        for (int j=0; j<agg_part_rels->dof_to_AE->Size_of_connections(); ++j)
            out3 << agg_part_rels->dof_id_inAE[j] << std::endl;
    }

    agg_produce_mises(Aglobal, *agg_part_rels, bdr_dofs, do_aggregates);
    if (agg_part_rels->testmesh)
    {
        std::stringstream filename;
        filename << "truemis_to_dof." << PROC_RANK << ".table";
        std::ofstream out(filename.str().c_str());
        agg_part_rels->truemis_to_dof->Print(out);
        filename.str("");
        filename << "mis_to_dof." << PROC_RANK << ".table";
        std::ofstream out2(filename.str().c_str());
        agg_part_rels->mis_to_dof->Print(out2);
        filename.str("");
        filename << "mis_master." << PROC_RANK << ".array";
        std::ofstream out4(filename.str().c_str());
        for (int j=0; j<agg_part_rels->num_mises; ++j)
            out4 << agg_part_rels->mis_master[j] << std::endl;
        filename.str("");
        filename << "mises." << PROC_RANK << ".array";
        std::ofstream out5(filename.str().c_str());
        for (int j=0; j<agg_part_rels->mis_to_dof->Width(); ++j)
            out5 << agg_part_rels->mises[j] << std::endl;
    }

    // this is deprecated-ish, moving to MISes? (but we still need agg_flags, or something like it)
    // agg_flags is basically complete in preconstruct_aggregates...
    // agg_construct_aggregates(Alocal, *agg_part_rels, bdr_dofs);
    if (!do_aggregates)
        agg_construct_agg_flags(*agg_part_rels, bdr_dofs);

#if (SA_IS_DEBUG_LEVEL(3))
    for (int i=0; i < agg_part_rels->ND; ++i)
        SA_ASSERT(0 <= agg_part_rels->agg_flags[i] &&
                  agg_part_rels->agg_flags[i] <= AGG_ALL_FLAGS);
#endif
}

Table * agg_create_finedof_to_dof(agg_partitioning_relations_t& agg_part_rels,
                                  const agg_partitioning_relations_t& agg_part_rels_fine,
                                  HypreParMatrix * interp)
{
    SA_RPRINTF(0,"fine.Dof_TrueDof is %d by %d, interp is %d by %d, coarse.Dof_TrueDof is %d by %d\n",
               agg_part_rels_fine.Dof_TrueDof->M(), agg_part_rels_fine.Dof_TrueDof->N(),
               interp->M(), interp->N(),
               agg_part_rels.Dof_TrueDof->M(), agg_part_rels.Dof_TrueDof->N());
    HypreParMatrix * temp = ParMult(agg_part_rels_fine.Dof_TrueDof, interp);
    // temp->Print("temp.mat");
    HypreParMatrix * TrueDof_Dof = agg_part_rels.Dof_TrueDof->Transpose();
    HypreParMatrix * hpm_finedof_dof = ParMult(temp, TrueDof_Dof);

    hypre_ParCSRMatrix * h_finedof_dof = (*hpm_finedof_dof);
    hypre_CSRMatrix * finedof_dof_diag = h_finedof_dof->diag;
    // hypre_CSRMatrix * finedof_dof_offd = h_finedof_dof->offd;
    int * finedof_dof_diag_I = finedof_dof_diag->i;
    int * finedof_dof_diag_J = finedof_dof_diag->j;
    // int * finedof_dof_col_starts = h_finedof_dof->col_starts;
    // int * finedof_dof_offd_I = finedof_dof_offd->i;
    // int * finedof_dof_offd_J = finedof_dof_offd->j;    
    // int * finedof_dof_colmap = h_finedof_dof->col_map_offd;

    // should probably just do out->SetIJ(); with some StealData mojo or something...
    Table * out = new Table;
    out->MakeI(finedof_dof_diag->num_rows);
    for (int i=0; i<finedof_dof_diag->num_rows; ++i)
        out->AddColumnsInRow(i, finedof_dof_diag_I[i+1] - finedof_dof_diag_I[i]);
    out->MakeJ(); // I hate the MFEM Table interface the way Liam Neeson hates kidnappers in the Taken movies
    for (int i=0; i<finedof_dof_diag->num_rows; ++i)
        out->AddConnections(i, &finedof_dof_diag_J[finedof_dof_diag_I[i]], 
                            finedof_dof_diag_I[i+1] - finedof_dof_diag_I[i]);
    out->ShiftUpI(); // have I mentioned how I feel about the MFEM Table interface?
    out->Finalize();

    delete temp;
    delete TrueDof_Dof;
    delete hpm_finedof_dof;

    return out;
}

void agg_create_rels_except_elem_coarse(
    HypreParMatrix * A,
    agg_partitioning_relations_t& agg_part_rels,
    const agg_partitioning_relations_t& agg_part_rels_fine,
    HypreParMatrix * interp,
    bool do_aggregates)
{
    // SA_RPRINTF(0,"%s","--- begin agg_create_rels_except_elem_coarse() ---\n");

    SA_ASSERT(interp);
    SA_ASSERT(&agg_part_rels);
    SA_ASSERT(&agg_part_rels_fine);
    SA_ASSERT(!agg_part_rels.elem_to_dof);
    SA_ASSERT(!agg_part_rels.dof_to_dof);
    SA_ASSERT(!agg_part_rels.dof_to_elem);
    SA_ASSERT(!agg_part_rels.AE_to_dof);
    SA_ASSERT(!agg_part_rels.dof_to_AE);
    // agg_part_rels.dorefaggs = dorefaggs;
    agg_part_rels.elem_to_dof = new Table();
    agg_part_rels.dof_to_elem = new Table();
    agg_part_rels.AE_to_dof = new Table();
    agg_part_rels.dof_to_AE = new Table();

    SA_ASSERT(!agg_part_rels.ND);

    // elem_to_dof and dof_to_elem ... possibly nontrivial in parallel...
    SA_ASSERT(agg_part_rels.elem_to_dof);
    SA_ASSERT(agg_part_rels.dof_to_elem);

    Table * finedof_to_dof = agg_create_finedof_to_dof(agg_part_rels, agg_part_rels_fine, interp);
    agg_part_rels.ND = finedof_to_dof->Width();
    Mult(*agg_part_rels_fine.AE_to_dof, *finedof_to_dof,
         *agg_part_rels.elem_to_dof);
    Transpose(*agg_part_rels.elem_to_dof, *agg_part_rels.dof_to_elem);

    if (agg_part_rels.testmesh)
    {
        std::stringstream filename;
        filename << "finedof_to_dof." << PROC_RANK << ".table";
        std::ofstream out(filename.str().c_str());
        finedof_to_dof->Print(out);
        filename.str("");
        filename << "coarse_elem_to_dof." << PROC_RANK << ".table";
        std::ofstream out2(filename.str().c_str());
        agg_part_rels.elem_to_dof->Print(out2);
    }
    
#if (SA_IS_DEBUG_LEVEL(3))
    for (int i=0; i < agg_part_rels.ND; ++i)
        SA_ASSERT(agg_part_rels.dof_to_elem->RowSize(i) > 0);
    SA_ASSERT(agg_part_rels_fine.dof_to_dof);
    agg_part_rels.dof_to_dof = new Table();
    SA_ASSERT(agg_part_rels.dof_to_dof);
    Table finedof_to_finedof_x_finedof_to_dof, dof_to_finedof;
    Mult(*agg_part_rels_fine.dof_to_dof, *finedof_to_dof,
         finedof_to_finedof_x_finedof_to_dof);
    Transpose(*finedof_to_dof, dof_to_finedof);
    Mult(dof_to_finedof, finedof_to_finedof_x_finedof_to_dof,
         *agg_part_rels.dof_to_dof);
#endif

    // finedof_to_dof->LoseData();
    delete finedof_to_dof;

    // AE_to_dof and dof_to_AE
    SA_ASSERT(agg_part_rels.AE_to_dof);
    SA_ASSERT(agg_part_rels.dof_to_AE);
    Mult(*agg_part_rels.AE_to_elem, *agg_part_rels.elem_to_dof,
         *agg_part_rels.AE_to_dof);
    Transpose(*agg_part_rels.AE_to_dof, *agg_part_rels.dof_to_AE);
    if (agg_part_rels.testmesh)
    {
        std::stringstream filename;
        filename << "coarse_AE_to_dof." << PROC_RANK << ".table";
        std::ofstream out(filename.str().c_str());
        agg_part_rels.AE_to_dof->Print(out);
    }
    agg_build_glob_to_AE_id_map(agg_part_rels); 

    // SA_RPRINTF(0,"%s","--- middle of agg_create_rels_except_elem_coarse() ---\n");

    // no boundary dofs [TODO---I think we should have processor boundary info, though] on coarser levels, I guess...
    agg_produce_mises(*A, agg_part_rels, NULL, do_aggregates);

    if (agg_part_rels.testmesh)
    {
        std::stringstream filename;
        filename << "coarse_truemis_to_dof." << PROC_RANK << ".table";
        std::ofstream out(filename.str().c_str());
        agg_part_rels.truemis_to_dof->Print(out);
        filename.str("");
        filename << "coarse_mis_to_dof." << PROC_RANK << ".table";
        std::ofstream out2(filename.str().c_str());
        agg_part_rels.mis_to_dof->Print(out2);
        filename.str("");
        filename << "coarse_mis_master." << PROC_RANK << ".array";
        std::ofstream out4(filename.str().c_str());
        for (int j=0; j<agg_part_rels.num_mises; ++j)
            out4 << agg_part_rels.mis_master[j] << std::endl;
        filename.str("");
        filename << "coarse_mises." << PROC_RANK << ".array";
        std::ofstream out5(filename.str().c_str());
        for (int j=0; j<agg_part_rels.mis_to_dof->Width(); ++j)
            out5 << agg_part_rels.mises[j] << std::endl;
        /*
        filename.str("");
        filename << "coarse_dof_id_inAE." << PROC_RANK << ".array";
        std::ofstream out6(filename.str().c_str());
        for (int j=0; j<agg_part_rels.dof_to_AE->Size_of_connections(); ++j)
            out6 << agg_part_rels.dof_id_inAE[j] << std::endl;
        */
    }

    if (!do_aggregates)
        agg_construct_agg_flags(agg_part_rels, NULL);

#if (SA_IS_DEBUG_LEVEL(3))
    for (int i=0; i < agg_part_rels.ND; ++i)
        SA_ASSERT(0 <= agg_part_rels.agg_flags[i] &&
                  agg_part_rels.agg_flags[i] <= AGG_ALL_FLAGS);
#endif
}

/**
   This routine separated from contrib_mises ATB 30 April 2015

   this is a very important routine for recursion, but I wonder if we should
   do something like (mis_truemis_mis) (mis_coarsedof) instead?
*/
void agg_build_coarse_Dof_TrueDof(agg_partitioning_relations_t &agg_part_rels_coarse,
                                  const agg_partitioning_relations_t &agg_part_rels_fine,
                                  int coarse_truedof_offset, int * mis_numcoarsedof,
                                  DenseMatrix ** mis_tent_interps)
{ 
    SharedEntityCommunication<DenseMatrix> sec(PROC_COMM,
                                               *agg_part_rels_fine.mis_truemis);    

    sec.Broadcast(mis_tent_interps);

    // ----------
    // PHASE 4: figure out coarse dofs, communicate the offsets and counts
    // ----------
    int num_mises = agg_part_rels_fine.num_mises;
    int * mis_truedof_offsets = new int[num_mises];
    int truedof_counter = 0;
    for (int mis=0; mis<num_mises; ++mis)
    {
        int owner = agg_part_rels_fine.mis_master[mis];
        if (owner == PROC_RANK)
        {
            mis_truedof_offsets[mis] = coarse_truedof_offset + truedof_counter;
            truedof_counter += mis_numcoarsedof[mis];
        }
        else
        {
            mis_truedof_offsets[mis] = -1;
        }
    }
    sec.BroadcastFixedSize(mis_truedof_offsets, 1);

    // ----------
    // PHASE 6: assemble the coarse Dof_TrueDof matrix
    // ----------
    int dof_counter = 0;
    Array<int> I;
    Array<int> J;
    Array<int> coarsedof_mis_array;
    int nnz = 0;
    I.Append(0);
    Array<int> dof_masterproc_a;
    for (int mis=0; mis<num_mises; ++mis)
    {
        int owner = agg_part_rels_fine.mis_master[mis];
        // int truemis = sec.GetTrueEntity(mis);
        if (owner == PROC_RANK)
        {
            // SA_PRINTF("on [owned] MIS %d (truemis %d), dof_counter = %d\n", mis, truemis, dof_counter);
            for (int i=0; i<mis_numcoarsedof[mis]; ++i)
            {
                // SA_PRINTF("  (coarse) (locally numbered) dof %d is truedof %d\n", dof_counter + i, mis_truedof_offsets[mis] + i);
                J.Append(mis_truedof_offsets[mis] + i);
                I.Append(++nnz);
                coarsedof_mis_array.Append(mis);
                dof_masterproc_a.Append(owner);
            }
            dof_counter += mis_numcoarsedof[mis];
        }
        else
        {
            // SA_PRINTF("on [shared] MIS %d (truemis %d), dof_counter = %d\n", mis, truemis, dof_counter);
            for (int i=0; i<mis_tent_interps[mis]->Width(); ++i)
            {
                // SA_PRINTF("  (coarse) (locally numbered) dof %d is truedof %d\n", dof_counter + i, mis_truedof_offsets[mis] + i);
                J.Append(mis_truedof_offsets[mis] + i);
                I.Append(++nnz);
                coarsedof_mis_array.Append(mis);
                dof_masterproc_a.Append(owner);
            }
            mis_numcoarsedof[mis] = mis_tent_interps[mis]->Width();
            dof_counter += mis_tent_interps[mis]->Width();
        }
    }
    delete [] mis_truedof_offsets;
    SA_ASSERT(nnz == dof_counter); // may not actually need both...
    SA_ASSERT(J.Size() == dof_counter);

    // build mis_coarsedofoffsets
    int coarsedofoffset = 0;
    agg_part_rels_coarse.mis_coarsedofoffsets = new int[num_mises+1];
    for (int mis=0; mis<num_mises; ++mis)
    {
        agg_part_rels_coarse.mis_coarsedofoffsets[mis] = coarsedofoffset;
        coarsedofoffset += mis_numcoarsedof[mis];
    }
    agg_part_rels_coarse.mis_coarsedofoffsets[num_mises] = coarsedofoffset;

    // build coarse dof_masterproc
    agg_part_rels_coarse.dof_masterproc = new int[dof_counter];
    for (int i=0; i<dof_counter; ++i)
        agg_part_rels_coarse.dof_masterproc[i] = dof_masterproc_a[i];

    // build the actual Dof_TrueDof HypreParMatrix
    double * data = new double[nnz];
    for (int i=0; i<nnz; ++i)
        data[i] = 1.0;
    Array<int> dof_offsets;
    Array<int> truedof_offsets;
    int total_dof, total_truedof;
    proc_determine_offsets(dof_counter, dof_offsets, total_dof);
    proc_determine_offsets(truedof_counter, truedof_offsets, total_truedof);
    // dof_offsets.Append(total_dof); // is this necessray?
    // truedof_offsets.Append(total_truedof);

    /** Creates a general parallel matrix from a local CSR matrix on each
        processor described by the I, J and data arrays. The local matrix should
        be of size (local) nrows by (global) glob_ncols. The parallel matrix
        contains copies of the rows and cols arrays (so they can be deleted). */
    /*
      HypreParMatrix(MPI_Comm comm, int nrows, int glob_nrows, int glob_ncols,
          int *I, int *J, double *data, int *rows, int *cols);
    */
    // probably going to run into some malloc/free mismatch garbage...
    agg_part_rels_coarse.Dof_TrueDof = 
        new HypreParMatrix(PROC_COMM, dof_counter, total_dof, total_truedof,
                           I.GetData(), J.GetData(), data, dof_offsets, truedof_offsets);

    delete [] data;

    // return Dof_TrueDof;
}

/**
   Don't think we're gonna need the global matrix A...
*/
agg_partitioning_relations_t *
agg_create_partitioning_coarse(
    HypreParMatrix* A,
    const agg_partitioning_relations_t& agg_part_rels_fine,
    int coarse_truedof_offset,
    int * mis_numcoarsedof,
    DenseMatrix ** mis_tent_interps,
    HypreParMatrix * interp,
    int *nparts,
    bool do_aggregates)
{
    SA_ASSERT(interp);
    SA_ASSERT(&agg_part_rels_fine);
    agg_partitioning_relations_t *agg_part_rels =
        new agg_partitioning_relations_t;
    memset(agg_part_rels, 0, sizeof(*agg_part_rels));
    agg_part_rels->testmesh = agg_part_rels_fine.testmesh;

    agg_build_coarse_Dof_TrueDof(*agg_part_rels,
                                 agg_part_rels_fine,
                                 coarse_truedof_offset,
                                 mis_numcoarsedof,
                                 mis_tent_interps);
    agg_part_rels->owns_Dof_TrueDof = true;
    if (agg_part_rels->testmesh)
    {
        agg_part_rels->Dof_TrueDof->Print("coarse_Dof_TrueDof.mat");
    }

    Table tmptbl;
    agg_part_rels->elem_to_elem = new Table();

    // elem_to_elem
    Mult(*agg_part_rels_fine.AE_to_elem,
         *agg_part_rels_fine.elem_to_elem, tmptbl);
    Mult(tmptbl, *agg_part_rels_fine.elem_to_AE,
         *agg_part_rels->elem_to_elem);
    SA_ASSERT(agg_part_rels->elem_to_elem->Size() ==
              agg_part_rels->elem_to_elem->Width());
    SA_ASSERT(agg_part_rels->elem_to_elem->Size() == agg_part_rels_fine.nparts);

    // AEs and elements
    if (agg_part_rels->testmesh)
    {
        if (PROC_NUM == 1)
        {
            agg_part_rels->partitioning = new int[4];
            agg_part_rels->partitioning[0] = agg_part_rels->partitioning[1] = 0;
            agg_part_rels->partitioning[2] = agg_part_rels->partitioning[3] = 1;
        }
        else if (PROC_NUM == 2)
        {
            // same partitioning on both ranks...
            agg_part_rels->partitioning = new int[2];
            agg_part_rels->partitioning[0] = 0;
            agg_part_rels->partitioning[1] = 0;
        }
        else
            SA_ASSERT(false);
    }
    else
    {
        const int num_elem = agg_part_rels_fine.nparts;
        SA_ASSERT(agg_part_rels_fine.AE_to_dof->Size() == num_elem);
        int * weights = new int[num_elem];
        for (int i=0; i<num_elem; ++i)
            weights[i] = agg_part_rels_fine.AE_to_dof->RowSize(i);
        agg_part_rels->partitioning =
            part_generate_partitioning(*(agg_part_rels->elem_to_elem), weights, nparts);
        delete [] weights;
    }
    agg_part_rels->nparts = *nparts;
    agg_construct_tables_from_arr(agg_part_rels->partitioning,
                                  agg_part_rels_fine.nparts,
                                  agg_part_rels->elem_to_AE,
                                  agg_part_rels->AE_to_elem);

    if (agg_part_rels->testmesh) // we also output this in ML... remove one of them, I think
    {
        std::stringstream filename;
        filename << "coarse_AE_to_elem." << PROC_RANK << ".table";
        std::ofstream out(filename.str().c_str());
        agg_part_rels->AE_to_elem->Print(out);
    }

    if (SA_IS_OUTPUT_LEVEL(10))
    {
        SA_PRINTF("%s","AE sizes (elems): ");
        for (int i=0; i < *nparts; ++i)
            SA_PRINTF_NOTS("%d:%d ", i, agg_part_rels->AE_to_elem->RowSize(i));
        SA_PRINTF_NOTS("\n");
    }

    agg_create_rels_except_elem_coarse(A, *agg_part_rels, agg_part_rels_fine,
                                       interp, do_aggregates);

    return agg_part_rels;
}

void agg_free_partitioning(agg_partitioning_relations_t *agg_part_rels)
{
    if (!agg_part_rels) return;

    delete agg_part_rels->dof_to_elem;
    delete agg_part_rels->dof_to_dof;
    delete agg_part_rels->elem_to_dof;
    delete agg_part_rels->AE_to_dof;
    delete agg_part_rels->dof_to_AE;

    // new ATB 7 April 2015
    delete agg_part_rels->truemis_to_dof;
    delete agg_part_rels->mis_to_dof;
    delete agg_part_rels->mis_to_AE;
    delete agg_part_rels->AE_to_mis;
    delete agg_part_rels->mis_truemis;
    delete [] agg_part_rels->mis_master;
    delete [] agg_part_rels->mises_size;
    delete [] agg_part_rels->mises;

    delete [] agg_part_rels->dof_id_inAE;
    delete [] agg_part_rels->agg_flags;
    delete [] agg_part_rels->partitioning;
    delete agg_part_rels->AE_to_elem;
    delete agg_part_rels->elem_to_AE;
    delete agg_part_rels->elem_to_elem;

    if (agg_part_rels->owns_Dof_TrueDof)
    {
        delete [] agg_part_rels->mis_coarsedofoffsets;
        delete agg_part_rels->Dof_TrueDof;
        // delete agg_part_rels->dof_proc;
        delete [] agg_part_rels->dof_masterproc;
    }

    delete agg_part_rels;
}

agg_partitioning_relations_t
    *agg_copy_partitioning(const agg_partitioning_relations_t *src)
{
    if (!src) return NULL;
    agg_partitioning_relations_t *dst = new agg_partitioning_relations_t;
    SA_ASSERT(src->elem_to_elem);

    dst->ND = src->ND;
    dst->nparts = src->nparts;
    dst->partitioning = helpers_copy_int_arr(src->partitioning,
                                             src->elem_to_elem->Size());
    dst->dof_to_elem = mbox_copy_table(src->dof_to_elem);

    dst->dof_to_dof = mbox_copy_table(src->dof_to_dof);
    dst->elem_to_dof = mbox_copy_table(src->elem_to_dof);
    dst->AE_to_elem = mbox_copy_table(src->AE_to_elem);
    dst->elem_to_AE = mbox_copy_table(src->elem_to_AE);
    dst->elem_to_elem = mbox_copy_table(src->elem_to_elem);

    dst->AE_to_dof = mbox_copy_table(src->AE_to_dof);
    dst->dof_to_AE = mbox_copy_table(src->dof_to_AE);
    SA_ASSERT(src->dof_to_AE);
    dst->dof_id_inAE = helpers_copy_int_arr(src->dof_id_inAE,
                           src->dof_to_AE->Size_of_connections());

    dst->agg_flags = new agg_dof_status_t[src->ND];
    std::copy(src->agg_flags, src->agg_flags + src->ND, dst->agg_flags);

    return dst;
}
