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
#include "aggregates.hpp"
#include <cmath>
#include <algorithm>
#include <mfem.hpp>
#include "part.hpp"
#include "elmat.hpp"
#include "helpers.hpp"
#include "mbox.hpp"
using std::fabs;
using std::sqrt;
using std::copy;

/* Options */

CONFIG_DEFINE_CLASS(AGGREGATES);

/* Static Functions */

/*! \brief Computes the value of an element of a local stiffness matrix.

    Computes the value of an element of a local (for an AE) stiffness matrix.

    \param di (IN) The global number of the first DoF.
    \param dj (IN) The global number of the second DoF.
    \param part (IN) The id of the AE.
    \param agg_part_rels (IN) The partitioning relations.
    \param elmat_callback (IN) A callback that returns element matrices.
    \param data (IN/OUT) \a elmat_callback specific data.

    \returns The computed value.

    \warning It must be compiled with Run-Time Type Information (RTTI) enabled
             for \em dynamic_cast to work.
*/
static inline
double agg_assemble_value(int di, int dj, int part,
                          const agg_partititoning_relations_t& agg_part_rels,
                          agg_elmat_callback_ft elmat_callback, void *data)
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

        elem_matr = elmat_callback(elno, &agg_part_rels, data, free_matr);
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

/*! \brief Suggests in which local aggregate a DoF should go.

    The approach is greedy and based on the strength of connections.

    TODO: It can be improved. Also, another strength-based approach may be
          better.

    \param A (IN) The global stiffness matrix.
    \param diag (IN) An array with the values on the diagonal of \a A.
    \param i (IN) The DoF to distribute.
    \param agg_part_rels (IN) The partitioning relations.

    \returns The ID of the aggregate that is suggested.

    \warning It has issues with pathological cases. For example, AEs with empty
             inner parts (w.r.t. trivially assigned DoFs) cannot be handled.
*/
static inline
int agg_simple_strength_suggest(const SparseMatrix& A, const double *diag,
                                int i, const agg_partititoning_relations_t&
                                                                 agg_part_rels)
{
#ifdef SA_ASSERTS
    const int ND = agg_part_rels.ND;
#endif
    const int * const aggregates = agg_part_rels.aggregates;
    const int * const agg_size = agg_part_rels.agg_size;
    const agg_dof_status_t * const agg_flags = agg_part_rels.agg_flags;
    const Table& dof_to_AE = *(agg_part_rels.dof_to_AE);
    int ret = -1000;
    int row_start, neigh, max_agg = -1;
    int *neighbours;
    int row_size = const_cast<SparseMatrix&>(A).RowSize(i);
    double *neigh_data;
    double strength, max_stren = -1.;

    SA_ASSERT(agg_size);
    SA_ASSERT(aggregates);
    SA_ASSERT(SA_IS_SET_A_FLAG(agg_flags[i], AGG_BETWEEN_AES_FLAG));

    row_start = A.GetI()[i];
    neighbours = &(A.GetJ()[row_start]);
    neigh_data = &(A.GetData()[row_start]);
    SA_ASSERT(row_size > 0);
    // If DoF i is connected to at least one other DoF.
    if (row_size > 1)
    {
        // Find the strongest connection.
        for (int j=0; j < row_size; ++j)
        {
            neigh = neighbours[j];
            SA_ASSERT(0 <= neigh && neigh < ND);
            SA_ASSERT(-2 <= aggregates[neigh] &&
                      aggregates[neigh] < agg_part_rels.nparts);
            if (neigh != i && 0 <= aggregates[neigh] &&
                agg_elem_in_col(i, aggregates[neigh], dof_to_AE) >= 0)
            {

                SA_ASSERT(0 <= aggregates[neigh] &&
                            aggregates[neigh] < agg_part_rels.nparts);
                SA_ASSERT(diag[i] > 0. && diag[neigh] > 0.);
                strength = fabs(neigh_data[j]) /
                                sqrt(diag[i] * diag[neigh]);
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
            SA_ASSERT(0 <= max_agg && max_agg < agg_part_rels.nparts);
            SA_ASSERT(ret < 0);
            ret = max_agg;
        } else // All neighbors are not distributed or the distributed
        {      // ones are in aggregates where 'i' cannot go.
#if (SA_IS_DEBUG_LEVEL(3))
            for (int j=0; j < row_size; ++j)
            {
                neigh = neighbours[j];
                SA_ASSERT(0 <= neigh && neigh < ND);
                SA_ASSERT(-2 == aggregates[neigh] || -1 == aggregates[neigh] ||
                          agg_elem_in_col(i, aggregates[neigh], dof_to_AE) < 0);
            }
#endif
            row_size = 1; //Fall back to the 'no neighbours' case.
        }
    }
    SA_ASSERT(row_size > 0);
    if (1 >= row_size)
    {
        SA_ASSERT(const_cast<SparseMatrix&>(A).RowSize(i) > 1 ||
                  neighbours[0] == i);
        int num_AEs = dof_to_AE.RowSize(i);
        SA_ASSERT(num_AEs >= 1);
        int min_agg;
        const int *parts = dof_to_AE.GetRow(i);

        SA_ASSERT(parts[0] < agg_part_rels.nparts && parts[0] >= 0);
        min_agg = parts[0];
        // Find the aggregate with minimal number of DoFs.
        for (int j=1; j < num_AEs; ++j)
        {
            SA_ASSERT(parts[j] < agg_part_rels.nparts && parts[j] >= 0);
            if (agg_size[min_agg] > agg_size[parts[j]])
                min_agg = parts[j];
        }
        SA_ASSERT(ret < 0);
        ret = min_agg;
    }
    SA_ASSERT(ret >= 0);
    return ret;
}

/* Functions */

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
        } else
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

bool agg_simple_strength_local_resolve(const SparseMatrix& A,
                                       const agg_partititoning_relations_t&
                                                                 agg_part_rels,
                                       bool first_time)
{
    const int ND = agg_part_rels.ND;
    const int * const preaggregates = agg_part_rels.preaggregates;
    const int * const preagg_size = agg_part_rels.preagg_size;
    int * const aggregates = agg_part_rels.aggregates;
    int * const agg_size = agg_part_rels.agg_size;
    const agg_dof_status_t * const agg_flags = agg_part_rels.agg_flags;
#ifdef SA_ASSERTS
    const Table& dof_to_AE = *(agg_part_rels.dof_to_AE);
#endif
    SA_ASSERT(dof_to_AE.Size() == ND);
    SA_ASSERT(dof_to_AE.Width() == agg_part_rels.nparts);
    int i;

    SA_ASSERT(agg_size);
    SA_ASSERT(aggregates);
    SA_ASSERT(preagg_size);
    SA_ASSERT(preaggregates);

    memcpy(agg_size, preagg_size, agg_part_rels.nparts * sizeof(*agg_size));
    memcpy(aggregates, preaggregates, ND * sizeof(*aggregates));

#if (SA_IS_DEBUG_LEVEL(3))
    {
        int nzero = 0;
        for (i=0; i < agg_part_rels.nparts; ++i)
        {
            if (!agg_size[i])
            {
                ++nzero;
                SA_ALERT_MSG("Aggregate with zero inner part: %d", i);
            }
        }
        SA_ASSERT(!nzero); // Empty inner parts of aggregate(s)
    }
#endif

    // Collect the diagonal of A.
    SA_ASSERT(A.Size() == ND);
    SA_ASSERT(A.Width() == ND);
    double *diag = new double[ND];
    for (i=0; i < ND; ++i)
        diag[i] = A(i,i);

    // Loop over all DoFs.
    for (i=0; i < ND; ++i)
    {
        // If DoF i is marked as non-trivial and is not on a processes'
        // interface.
        if (-2 == aggregates[i] &&
            !SA_IS_SET_A_FLAG(agg_flags[i], AGG_ON_PROC_IFACE_FLAG))
        {
            SA_ASSERT(dof_to_AE.RowSize(i) > 1);
            const int agg =
                agg_simple_strength_suggest(A, diag, i, agg_part_rels);
            SA_ASSERT(0 <= agg && agg < agg_part_rels.nparts);
            aggregates[i] = agg;
            ++agg_size[agg];
        } else
        {
            SA_ASSERT(SA_IS_SET_A_FLAG(agg_flags[i], AGG_ON_PROC_IFACE_FLAG) ||
                      1 == dof_to_AE.RowSize(i));
            SA_ASSERT(SA_IS_SET_A_FLAG(agg_flags[i], AGG_ON_PROC_IFACE_FLAG) ||
                      *(dof_to_AE.GetRow(i)) == aggregates[i]);
        }
        SA_ASSERT((-2 == aggregates[i] &&
                   SA_IS_SET_A_FLAG(agg_flags[i], AGG_ON_PROC_IFACE_FLAG)) ||
                  (aggregates[i] < agg_part_rels.nparts && aggregates[i] >= 0));
    }

    delete [] diag;

#if (SA_IS_DEBUG_LEVEL(3))
    {
        for (i=0; i < ND; ++i)
        {
            SA_ASSERT((-2 == aggregates[i] &&
                       SA_IS_SET_A_FLAG(agg_flags[i], AGG_ON_PROC_IFACE_FLAG))
                      ||
                      (aggregates[i] < agg_part_rels.nparts &&
                       aggregates[i] >= 0));
            SA_ASSERT(-2 == aggregates[i] ||
                      agg_elem_in_col(i, aggregates[i], dof_to_AE) >= 0);
        }
        int nzero = 0;
        for (i=0; i < agg_part_rels.nparts; ++i)
        {
            if (!agg_size[i])
            {
                ++nzero;
                SA_ALERT_MSG("Zero aggregate: %d", i);
            }
        }
        SA_ASSERT(!nzero); // Empty aggregate(s)
    }
#endif

    return true;
}

bool agg_simple_strength_iface_resolve(const SparseMatrix& A,
                                       const agg_partititoning_relations_t&
                                                                 agg_part_rels,
                                       bool first_time)
{
    const int ND = agg_part_rels.ND;
    int * const aggregates = agg_part_rels.aggregates;
    int * const agg_size = agg_part_rels.agg_size;
    const agg_dof_status_t * const agg_flags = agg_part_rels.agg_flags;
    int i;

    SA_ASSERT(agg_size);
    SA_ASSERT(aggregates);

    // Collect the diagonal of A.
    SA_ASSERT(A.Size() == ND);
    SA_ASSERT(A.Width() == ND);
    double *diag = new double[ND];
    for (i=0; i < ND; ++i)
        diag[i] = A(i,i);

    // Loop over all DoFs.
    for (i=0; i < ND; ++i)
    {
        // If DoF i is marked as non-trivial and is on a processes' interface.
        if (-2 == aggregates[i])
        {
            SA_ASSERT(SA_IS_SET_A_FLAG(agg_flags[i], AGG_ON_PROC_IFACE_FLAG));
            if (SA_IS_SET_A_FLAG(agg_flags[i], AGG_OWNED_FLAG))
            {
                const int agg =
                    agg_simple_strength_suggest(A, diag, i, agg_part_rels);
                SA_ASSERT(0 <= agg && agg < agg_part_rels.nparts);
                aggregates[i] = agg;
                ++agg_size[agg];
            } else
                aggregates[i] = -1;
        } else
            SA_ASSERT(!SA_IS_SET_A_FLAG(agg_flags[i], AGG_ON_PROC_IFACE_FLAG));

        SA_ASSERT((-1 == aggregates[i] &&
                   SA_IS_SET_A_FLAG(agg_flags[i], AGG_ON_PROC_IFACE_FLAG) &&
                   SA_IS_SET_A_FLAG(agg_flags[i], AGG_BETWEEN_AES_FLAG) &&
                   !SA_IS_SET_A_FLAG(agg_flags[i], AGG_OWNED_FLAG)) ||
                  (aggregates[i] < agg_part_rels.nparts && aggregates[i] >= 0));
#if (SA_IS_DEBUG_LEVEL(3))
        SA_ASSERT(-1 == aggregates[i] ||
                  agg_elem_in_col(i, aggregates[i],
                                  *(agg_part_rels.dof_to_AE)) >= 0);
#endif

    }

    delete [] diag;

#if (SA_IS_DEBUG_LEVEL(3))
    {
        int nzero = 0;
        for (i=0; i < agg_part_rels.nparts; ++i)
        {
            if (!agg_size[i])
            {
                ++nzero;
                SA_ALERT_MSG("Zero aggregate: %d", i);
            }
        }
        SA_ASSERT(!nzero); // Empty aggregate(s)
    }
#endif

    return true;
}

void agg_construct_aggregates(const SparseMatrix& A,
                              agg_partititoning_relations_t& agg_part_rels,
                              const agg_dof_status_t *bdr_dofs)
{
    SA_ASSERT(agg_part_rels.dof_to_AE);

    SA_ASSERT(!agg_part_rels.preaggregates);
    SA_ASSERT(!agg_part_rels.preagg_size);
    SA_ASSERT(!agg_part_rels.agg_flags);
    SA_ASSERT(!agg_part_rels.aggregates);
    SA_ASSERT(!agg_part_rels.agg_size);
    agg_part_rels.preaggregates =
        agg_preconstruct_aggregates(*(agg_part_rels.dof_to_AE),
                                    agg_part_rels.nparts, bdr_dofs,
                                    agg_part_rels.preagg_size,
                                    agg_part_rels.agg_flags);

    agg_part_rels.aggregates = new int[agg_part_rels.ND];
    agg_part_rels.agg_size = new int[agg_part_rels.nparts];
    agg_resolve_aggregates(A, agg_part_rels, true);

#if 0 //(SA_IS_DEBUG_LEVEL(4))
    // Check the integrity of the aggregates.
    SA_ASSERT(agg_part_rels.dof_to_dof);
    SA_PRINTF("--------- aggregates { ---------\n");
    part_check_partitioning(*agg_part_rels.dof_to_dof,
                            agg_part_rels.aggregates);
    SA_PRINTF("--------- } aggregates ---------\n");
#endif
}

SparseMatrix *agg_build_AE_stiffm_with_global(const SparseMatrix& A, int part,
    const agg_partititoning_relations_t& agg_part_rels,
    agg_elmat_callback_ft elmat_callback, void *data, bool bdr_cond_imposed,
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
                                               agg_part_rels, elmat_callback,
                                               data);
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
    const agg_partititoning_relations_t& agg_part_rels,
    agg_elmat_callback_ft elmat_callback, void *data, bool bdr_cond_imposed,
    bool assemble_ess_diag)
{
    SA_ASSERT(agg_part_rels.nparts > 0);
    SparseMatrix **arr = new SparseMatrix*[agg_part_rels.nparts];

    SA_ASSERT(arr);
    for (int i=0; i < agg_part_rels.nparts; ++i)
        arr[i] = agg_build_AE_stiffm_with_global(A, i, agg_part_rels,
                                                 elmat_callback, data,
                                                 bdr_cond_imposed,
                                                 assemble_ess_diag);

    return arr;
}

SparseMatrix *agg_simple_assemble(
    const agg_partititoning_relations_t& agg_part_rels,
    agg_elmat_callback_ft elmat_callback, void *data, bool assem_skip_zeros,
    bool final_skip_zeros, bool finalize, const agg_dof_status_t *bdr_dofs,
    const Vector *sol, Vector *rhs, bool keep_diag)
{
    SA_PRINTF_L(4, "%s", "Assembling matrix...\n");
    SA_ASSERT(elmat_callback);
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

    const Matrix *matr = elmat_callback(i=0, &agg_part_rels, data, free_matr);

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
                matr = elmat_callback(i, &agg_part_rels, data, free_matr);
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
                matr = elmat_callback(i, &agg_part_rels, data, free_matr);
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
    agg_partititoning_relations_t agg_part_rels;
    memset(&agg_part_rels, 0, sizeof(agg_part_rels));
    agg_part_rels.elem_to_dof = &elem_to_dof;
    agg_part_rels.dof_to_elem = &dof_to_elem;
    return agg_simple_assemble(agg_part_rels, elmat_from_array,
                               (void *)elem_matrs, assem_skip_zeros,
                               final_skip_zeros, finalize, bdr_dofs, sol, rhs,
                               keep_diag);
}

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

void agg_restrict_vect_to_AE(int part,
                             const agg_partititoning_relations_t& agg_part_rels,
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

void agg_build_glob_to_AE_id_map(agg_partititoning_relations_t& agg_part_rels)
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
    for (int i=0; i < agg_part_rels.nparts; ++i)
    {
        const int * const row = AE_to_dof.GetRow(i);
        const int rs = AE_to_dof.RowSize(i);
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

agg_partititoning_relations_t *
agg_create_partitioning_fine(const SparseMatrix& A, int NE, Table *elem_to_dof,
                             Table *elem_to_elem, int *partitioning,
                             const agg_dof_status_t *bdr_dofs, int *nparts,
                             ParFiniteElementSpace *fes)
{
    agg_partititoning_relations_t *agg_part_rels =
        new agg_partititoning_relations_t;
    memset(agg_part_rels, 0, sizeof(*agg_part_rels));

    SA_ASSERT(0 <= CONFIG_ACCESS_OPTION(AGGREGATES, resolve_aggragates) &&
              CONFIG_ACCESS_OPTION(AGGREGATES, resolve_aggragates) <
                  AGG_RES_AGGS_MAX);
    agg_part_rels->resolve_aggragates =
        CONFIG_ACCESS_OPTION(AGGREGATES, resolve_aggragates);

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
            part_generate_partitioning(*(agg_part_rels->elem_to_elem), nparts);
    }

    agg_part_rels->nparts = *nparts;

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

    agg_construct_aggregates(A, *agg_part_rels, bdr_dofs);

#if (SA_IS_DEBUG_LEVEL(3))
    for (int i=0; i < agg_part_rels->ND; ++i)
        SA_ASSERT(0 <= agg_part_rels->agg_flags[i] &&
                  agg_part_rels->agg_flags[i] <= AGG_ALL_FLAGS);
#endif

    if (SA_IS_OUTPUT_LEVEL(10))
    {
        PROC_STR_STREAM << "Aggregates sizes: ";
        for (int i=0; i < *nparts; ++i)
            PROC_STR_STREAM << i << ":" << agg_part_rels->agg_size[i] << " ";
        PROC_STR_STREAM << "\n";
        SA_PRINTF("%s", PROC_STR_STREAM.str().c_str());
        PROC_CLEAR_STR_STREAM;
        PROC_STR_STREAM << "AE sizes (elems): ";
        for (int i=0; i < *nparts; ++i)
            PROC_STR_STREAM << i << ":"
                            << agg_part_rels->AE_to_elem->RowSize(i) << " ";
        PROC_STR_STREAM << "\n";
        SA_PRINTF("%s", PROC_STR_STREAM.str().c_str());
        PROC_CLEAR_STR_STREAM;
    }

    agg_part_rels->fes = fes;

    return agg_part_rels;
}

void agg_free_partitioning(agg_partititoning_relations_t *agg_part_rels)
{
    if (!agg_part_rels) return;

    delete agg_part_rels->dof_to_elem;
    delete agg_part_rels->dof_to_dof;
    delete agg_part_rels->elem_to_dof;
    delete agg_part_rels->AE_to_dof;
    delete agg_part_rels->dof_to_AE;
    delete [] agg_part_rels->dof_id_inAE;
    delete [] agg_part_rels->preaggregates;
    delete [] agg_part_rels->preagg_size;
    delete [] agg_part_rels->aggregates;
    delete [] agg_part_rels->agg_size;
    delete [] agg_part_rels->agg_flags;
    delete [] agg_part_rels->partitioning;
    delete agg_part_rels->AE_to_elem;
    delete agg_part_rels->elem_to_AE;
    delete agg_part_rels->elem_to_elem;

    delete agg_part_rels;
}

agg_partititoning_relations_t
    *agg_copy_partitioning(const agg_partititoning_relations_t *src)
{
    if (!src) return NULL;
    agg_partititoning_relations_t *dst = new agg_partititoning_relations_t;
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

    dst->preaggregates = helpers_copy_int_arr(src->preaggregates, src->ND);
    dst->preagg_size = helpers_copy_int_arr(src->preagg_size, src->nparts);
    dst->resolve_aggragates = src->resolve_aggragates;

    dst->aggregates = helpers_copy_int_arr(src->aggregates, src->ND);
    dst->agg_size = helpers_copy_int_arr(src->agg_size, src->nparts);

    dst->agg_flags = new agg_dof_status_t[src->ND];
    copy(src->agg_flags, src->agg_flags + src->ND, dst->agg_flags);

    dst->fes = src->fes;

    return dst;
}
