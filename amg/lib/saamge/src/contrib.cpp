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
#include "contrib.hpp"
#include <mfem.hpp>
#include "aggregates.hpp"
#include "xpacks.hpp"

/* Options */

CONFIG_DEFINE_CLASS(CONTRIB);

/* Static Functions */

/*! \brief Loops over all big aggregates and fills in the tentative interpolant.

    \param tent_int_struct (IN) The structure returned by a call to
                                \b contrib_tent_init.
    \param agg_part_rels (IN) The partitioning relations.
    \param cut_evects_arr (IN) The vectors from all local eigenvalue problems.
    \param eps (IN) Tolerance for the SVD.
    \param do_svd (IN) Whether to perform SVD after restricting to aggregates.
*/
static inline
void contrib_big_aggs(contrib_tent_struct_t *tent_int_struct,
                      const agg_partititoning_relations_t& agg_part_rels,
                      DenseMatrix * const *cut_evects_arr, double eps,
                      bool do_svd)
{
    const bool avoid_ess_brd_dofs = CONFIG_ACCESS_OPTION(CONTRIB,
                                                         avoid_ess_brd_dofs);
    SA_ASSERT(tent_int_struct);
    SA_ASSERT(cut_evects_arr);
    SA_ASSERT(agg_part_rels.AE_to_dof);
    SA_ASSERT(agg_part_rels.aggregates);
    SA_ASSERT(agg_part_rels.agg_size);
    DenseMatrix restricted, lsvects, loc_tent_interp;
    Vector svals;
    int *restriction;

    SA_PRINTF_L(4, "%s", "---------- contrib_big_aggs { ---------------"
                         "------\n");
    SA_PRINTF_L(5, "do_svd: %d\n", do_svd);

    // Loop over agglomerates and big aggregates.
    for (int i=0; i < agg_part_rels.nparts; ++i)
    {
        SA_PRINTF_L(6, "%d +++++++++++++++++++++++++++++++++++++++++++++++++++"
                       "+++++++++++++++++++++++++++++++++++++++++++++\n", i);
        SA_ASSERT(cut_evects_arr[i]);
        // Restrict agglomerate vectors to aggregates
        restriction = agg_restrict_to_agg(i, *(agg_part_rels.AE_to_dof),
                                          agg_part_rels.aggregates, i,
                                          agg_part_rels.agg_size[i],
                                          *(cut_evects_arr[i]), restricted);

        if (avoid_ess_brd_dofs)
        {
            const int dim = restricted.Height();
            bool interior_dofs = false;
            for (int j=0; j < dim; ++j)
            {
                const int row = restriction[j];
                SA_ASSERT(tent_int_struct->rows > row);
                if (!agg_is_dof_on_essential_border(agg_part_rels, row))
                {
                    interior_dofs = true;
                    break;
                }
            }
            if (!interior_dofs)
            {
                if (SA_IS_OUTPUT_LEVEL(5))
                    SA_ALERT_PRINTF("All DoFs are on essential boundary."
                                    " Ignoring the entire contribution"
                                    " introducing not more than %d vector(s) on"
                                    " an aggregate of size %d!",
                                    restricted.Width(), dim);
                delete [] restriction;
                continue;
            }
        }

        if (do_svd)
        {
            // Compute SVD of basis. Build local tentative interpolation
            // operator using the linear independent ones.
            xpack_svd_dense_arr(&restricted, 1, lsvects, svals);
            xpack_orth_set(lsvects, svals, loc_tent_interp, eps);
        }

        // Insert the just built local tentative interpolant into the global
        // one.
        contrib_tent_insert_from_local(tent_int_struct, agg_part_rels,
                                       do_svd ? loc_tent_interp : restricted,
                                       restriction);
        delete [] restriction;
    }
    SA_PRINTF_L(6,  "%s", "end +++++++++++++++++++++++++++++++++++++++++++++++"
                          "+++++++++++++++++++++++++++++++++++++++++++"
                          "++++++\n");
    SA_PRINTF_L(4, "%s", "---------- } contrib_big_aggs ---------------"
                         "------\n");
}

/* Functions */

contrib_tent_struct_t *contrib_tent_init(int ND)
{
    contrib_tent_struct_t *tent_int_struct = new contrib_tent_struct_t;
    tent_int_struct->rows = ND;
    tent_int_struct->filled_cols = 0;
    tent_int_struct->tent_interp = new SparseMatrix(tent_int_struct->rows);

    return tent_int_struct;
}

SparseMatrix *contrib_tent_finalize(contrib_tent_struct_t *tent_int_struct)
{
    SparseMatrix *tent_interp;
    SA_ASSERT(tent_int_struct->filled_cols > 0);
    tent_int_struct->tent_interp->Finalize();
    tent_interp = new SparseMatrix(tent_int_struct->tent_interp->GetI(),
                                   tent_int_struct->tent_interp->GetJ(),
                                   tent_int_struct->tent_interp->GetData(),
                                   tent_int_struct->tent_interp->Size(),
                                   tent_int_struct->filled_cols);
    tent_int_struct->tent_interp->LoseData();
    delete (tent_int_struct->tent_interp);
    SA_ASSERT(tent_interp->GetI() && tent_interp->GetJ() &&
              tent_interp->GetData());
    SA_ASSERT(tent_interp->Size() == tent_int_struct->rows);
    SA_ASSERT(tent_interp->Width() == tent_int_struct->filled_cols);
    delete tent_int_struct;
    return tent_interp;
}

void contrib_tent_insert_from_local(contrib_tent_struct_t *tent_int_struct,
                            const agg_partititoning_relations_t& agg_part_rels,
                            const DenseMatrix& local, const int *restriction)
{
    const bool avoid_ess_brd_dofs = CONFIG_ACCESS_OPTION(CONTRIB,
                                                         avoid_ess_brd_dofs);
    const int vects = local.Width();
    const int dim = local.Height();
    double *data = local.Data();
    bool atleast_one;
    int col, i, j;

    SA_ASSERT(vects > 0);
    SA_ASSERT(dim >= vects);

    col = tent_int_struct->filled_cols;
    for (i=0; i < vects; (++i), (data += dim))
    {
        atleast_one = false;
        for (j=0; j < dim; ++j)
        {
            SA_ASSERT(data + j < local.Data() + dim*vects);
            const int row = restriction[j];
            const double a = data[j];

            SA_ASSERT(tent_int_struct->rows > row);
            SA_ASSERT(tent_int_struct->rows > col);
            SA_ASSERT(dim > 1 || 1. == a);
            if (0. == a ||
                (avoid_ess_brd_dofs &&
                 agg_is_dof_on_essential_border(agg_part_rels, row)))
            {
                if (SA_IS_OUTPUT_LEVEL(7) && 0. != a)
                    SA_ALERT_PRINTF("Non-zero DoF on essential boundary."
                                    " Ignoring entry: %g!", a);
                continue;
            }

            tent_int_struct->tent_interp->Set(row, col, a);
            atleast_one = true;
        }
        if (!atleast_one)
        {
            if (SA_IS_OUTPUT_LEVEL(5))
                SA_ALERT_PRINTF("%s", "Zero tentative prolongator column."
                                      " Ignoring column!");
        } else
            ++col;
    }

    tent_int_struct->filled_cols = col;
}


void contrib_big_aggs_svd(contrib_tent_struct_t *tent_int_struct,
                          const agg_partititoning_relations_t& agg_part_rels,
                          DenseMatrix * const *cut_evects_arr, double eps)
{
    contrib_big_aggs(tent_int_struct, agg_part_rels, cut_evects_arr, eps, true);
}

void contrib_big_aggs_nosvd(contrib_tent_struct_t *tent_int_struct,
                            const agg_partititoning_relations_t& agg_part_rels,
                            DenseMatrix * const *cut_evects_arr, double eps)
{
    contrib_big_aggs(tent_int_struct, agg_part_rels, cut_evects_arr, eps,
                     false);
}
