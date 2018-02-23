/*! \file
    \brief Handles arbitration of degrees of freedom, to decide what 
           aggregates they should go in.

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
#include "arbitrator.hpp"
#include <mfem.hpp>

using namespace mfem;

Arbitrator::Arbitrator(
    mfem::HypreParMatrix& Aglobal, // want to be const, but won't be
    const agg_partitioning_relations_t &agg_part_rels,
    const agg_dof_status_t * const agg_flags)
    :
    Aglobal(Aglobal),
    dof_truedof(*agg_part_rels.Dof_TrueDof),
    aggregates(agg_part_rels.mises),
    agg_size(agg_part_rels.mises_size),
    agg_flags(agg_flags),
    dof_to_AE(*agg_part_rels.dof_to_AE),
    nparts(agg_part_rels.nparts),
    num_dof(agg_part_rels.ND)
{
    hypre_ParCSRMatrix * h_dof_truedof = (hypre_ParCSRMatrix*) dof_truedof;
    dof_truedof_diag = h_dof_truedof->diag;
    
    // 0 below means don't actually transpose data, just graph
    int ierr = hypre_CSRMatrixTranspose(dof_truedof_diag, &truedof_dof_diag, 0); 
    SA_ASSERT(!ierr);    

    // Collect the diagonal of A.
    Aglobal.GetDiag(local_diag);
}

Arbitrator::~Arbitrator()
{
    hypre_CSRMatrixDestroy(truedof_dof_diag);
}

/*! \brief Suggests in which local aggregate a DoF should go.

    The approach is greedy and based on the strength of connections.

    TODO: It can be improved. Also, another strength-based approach may be
          better.

    \param A (IN) The global stiffness matrix.
    \param diag (IN) An array with the values on the diagonal of \a A.
    \param i (IN) The DoF to distribute, in local overlapped numbering
    \param locali (IN) the DoF to distribute, in local non-overlapped numbering

    following parameters changed/added ATB 30 December 2015
    \param aggregates corresponds to agg_part_rels.mises
    \param agg_size corresponds to agg_part_rels.mises_size
    \param agg_flags tells you whether dof is between AEs, 
                     on processor interface, on boundary
    \param dof_to_AE local table, agg_part_rels.dof_to_AE
    \param nparts number of agglomerated elements (locally)
    \param ND number of dofs (locally)

    \returns The ID of the aggregate that is suggested.

    \warning It has issues with pathological cases. For example, AEs with empty
             inner parts (w.r.t. trivially assigned DoFs) cannot be handled.
*/
int Arbitrator::suggest(int i)
{
    int ret = -1000;
    int row_start, neigh, neightrue, max_agg = -1;
    int *neighbours;
    // int row_size = const_cast<SparseMatrix&>(A).RowSize(i);
    double *neigh_data;
    double strength, max_stren = -1.;

    SA_ASSERT(agg_size);
    SA_ASSERT(aggregates);
    SA_ASSERT(agg_flags);
    SA_ASSERT(SA_IS_SET_A_FLAG(agg_flags[i], AGG_BETWEEN_AES_FLAG));

    double * diag = local_diag.GetData();
    SA_ASSERT(diag);

    // TODO these next too could be class private members
    hypre_ParCSRMatrix * h_A = (hypre_ParCSRMatrix*) Aglobal;
    const hypre_CSRMatrix *h_Adiag = h_A->diag;

    // asserting dof i is a truedof, that is local
    SA_ASSERT(dof_truedof_diag->i[i+1] != dof_truedof_diag->i[i]); 
    int itrue = dof_truedof_diag->j[dof_truedof_diag->i[i]];

    int row_size = h_Adiag->i[itrue+1] - h_Adiag->i[itrue];
    row_start = h_Adiag->i[itrue];
    neighbours = &(h_Adiag->j[row_start]);
    neigh_data = &(h_Adiag->data[row_start]);

    SA_ASSERT(row_size > 0);
    // If DoF i is connected to at least one other DoF.
    if (row_size > 1)
    {
        // Find the strongest (local to this processor) connection.
        for (int j=0; j < row_size; ++j) // this is in truedof numbering, kinda
        {
            neightrue = neighbours[j]; // this is in truedof numbering
            neigh = truedof_dof_diag->j[truedof_dof_diag->i[neightrue]];
            SA_ASSERT(0 <= neigh && neigh < num_dof);
            SA_ASSERT(-2 <= aggregates[neigh] &&
                      aggregates[neigh] < nparts);
            SA_ASSERT(neightrue >= 0);
            if (neightrue != itrue && 0 <= aggregates[neigh] &&
                agg_elem_in_col(i, aggregates[neigh], dof_to_AE) >= 0)
            {
                SA_ASSERT(0 <= aggregates[neigh] &&
                          aggregates[neigh] < nparts);
                SA_ASSERT(diag[itrue] > 0. && diag[neightrue] > 0.);
                strength = fabs(neigh_data[j]) /
                                sqrt(diag[itrue] * diag[neightrue]);
                SA_ASSERT(strength >= 0.);
                if (strength > max_stren)
                {
                    max_stren = strength;
                    max_agg = aggregates[neigh];
                }
            }
        }
        if (max_stren >= 0.)
        {
            SA_ASSERT(0 <= max_agg && max_agg < nparts);
            SA_ASSERT(ret < 0);
            ret = max_agg;
        } 
        else   // All neighbors are not distributed or the distributed
        {      // ones are in aggregates where 'i' cannot go.
#if (SA_IS_DEBUG_LEVEL(3))
            for (int j=0; j < row_size; ++j)
            {
                neightrue = neighbours[j];
                neigh = truedof_dof_diag->j[truedof_dof_diag->i[neightrue]];
                SA_ASSERT(0 <= neigh && neigh < num_dof);
                SA_ASSERT(-2 == aggregates[neigh] || -1 == aggregates[neigh] ||
                          agg_elem_in_col(i, aggregates[neigh], dof_to_AE) < 0);
            }
#endif
            row_size = 1; // Fall back to the 'no neighbours' case.
        }
    }
    SA_ASSERT(row_size > 0);
    if (1 >= row_size) // ie, row_size == 1
    {
        int num_AEs = dof_to_AE.RowSize(i);
        SA_ASSERT(num_AEs >= 1);
        int min_agg;
        const int *parts = dof_to_AE.GetRow(i);

        SA_ASSERT(parts[0] < nparts && parts[0] >= 0);
        min_agg = parts[0];
        // Find the aggregate with minimal number of DoFs.
        for (int j=1; j < num_AEs; ++j)
        {
            SA_ASSERT(parts[j] < nparts && parts[j] >= 0);
            if (agg_size[min_agg] > agg_size[parts[j]])
                min_agg = parts[j];
        }
        SA_ASSERT(ret < 0);
        ret = min_agg;
    }
    SA_ASSERT(ret >= 0);

    return ret;
}
