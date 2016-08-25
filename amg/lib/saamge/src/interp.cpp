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
#include "interp.hpp"
#include <cfloat>
#include <mfem.hpp>
#include "aggregates.hpp"
#include "elmat.hpp"
#include "spectral.hpp"
#include "contrib.hpp"
#include "helpers.hpp"
#include "mbox.hpp"
#include "process.hpp"

/* Options */

CONFIG_DEFINE_CLASS(INTERP);

/* Static Functions */

/*! \brief Generates data used for interpolant smoothing.

    Computes \f$ -D^{-1}A \f$, where D is a diagonal s.p.d. matrix s.t.
    \f$ \| I - D^{-1}A \|_A \leq 1 \f$.

    \param A (IN) The global (among all processes) stiffness matrix.
    \param Dinv_neg (IN) A diagonal that is precisely \f$ -D^{-1} \f$.

    \returns The result is \f$ -D^{-1}A \f$ to be used for producing the final
             (actual) interpolant.

    \warning The returned matrix must be freed by the caller.
    \warning Usually D is the weighted l1-smoother.
*/
static inline
HypreParMatrix *interp_simple_smoother(HypreParMatrix& A,
                                       HypreParVector& Dinv_neg)
{
    SA_ASSERT(A.GetGlobalNumRows() == A.GetGlobalNumCols());

#if (SA_IS_DEBUG_LEVEL(3))
    for (int i=0; i < Dinv_neg.Size(); ++i)
        SA_ASSERT(Dinv_neg(i) < 0.);
#endif

    HypreParMatrix *Dinv_neg_mat = mbox_create_diag_parallel_matrix(Dinv_neg);
    SA_ASSERT(A.GetGlobalNumRows() == Dinv_neg_mat->GetGlobalNumCols());
    HypreParMatrix *smoother = ParMult(Dinv_neg_mat, &A);
    mbox_make_owner_rowstarts_colstarts(*smoother);
    delete Dinv_neg_mat;

    return smoother;
}

/* Functions */

HypreParMatrix *interp_smooth(int degree, double *roots,
                              int times_apply_smoother, HypreParMatrix& A,
                              HypreParMatrix& tent, HypreParVector& Dinv_neg)
{
    HypreParMatrix *smoother_matr = interp_simple_smoother(A, Dinv_neg);
    HypreParMatrix *interp, *iter_matr, *new_interp;

    SA_ASSERT(0 <= degree);
    SA_ASSERT(roots || 0 == degree);
    SA_ASSERT(smoother_matr->GetGlobalNumRows() ==
              smoother_matr->GetGlobalNumCols());
    SA_ASSERT(smoother_matr->GetGlobalNumCols() == tent.GetGlobalNumRows());

    SA_PRINTF_L(4, "%s", "Smoothing tentative prolongator...\n");
    if (SA_IS_OUTPUT_LEVEL(5))
    {
        PROC_STR_STREAM << "Interp roots: ";
        for (int k=0; k < degree; ++k)
            PROC_STR_STREAM << roots[k] << " ";
        PROC_STR_STREAM << "\n";
        SA_PRINTF("%s", PROC_STR_STREAM.str().c_str());
        PROC_CLEAR_STR_STREAM;
    }

    interp = mbox_clone_parallel_matrix(&tent);
    for (int k=0; k < degree; ++k)
    {
        iter_matr = mbox_scale_clone_parallel_matrix(*smoother_matr,
                                                     1./roots[k]);
        mbox_add_daig_parallel_matrix(*iter_matr, 1.);

        SA_ASSERT(0 <= times_apply_smoother);
        for (int i=0; i < times_apply_smoother; ++i)
        {
            new_interp = ParMult(iter_matr, interp);
            mbox_make_owner_rowstarts_colstarts(*new_interp);
            delete interp;
            interp = new_interp;
        }
        delete iter_matr;
    }

    delete smoother_matr;

    return interp;
}

interp_data_t *interp_init_data(
    const agg_partititoning_relations_t& agg_part_rels, double nu)
{
    const int nparts = agg_part_rels.nparts;
    interp_data_t *interp_data = new interp_data_t;
    SA_ASSERT(interp_data);
    memset(interp_data, 0, sizeof(*interp_data));

    SA_ASSERT(CONFIG_ACCESS_OPTION(INTERP, local_prob_solve));
    interp_data->local_prob_solve =
        CONFIG_ACCESS_OPTION(INTERP, local_prob_solve);

    SA_ASSERT(CONFIG_ACCESS_OPTION(INTERP, contrib_agg));
    interp_data->contrib_agg = CONFIG_ACCESS_OPTION(INTERP, contrib_agg);

    SA_ASSERT(CONFIG_ACCESS_OPTION(INTERP, eps_svd) >= 0. &&
              CONFIG_ACCESS_OPTION(INTERP, eps_lin) >= 0.);
    interp_data->eps_svd = CONFIG_ACCESS_OPTION(INTERP, eps_svd);
    interp_data->eps_lin = CONFIG_ACCESS_OPTION(INTERP, eps_lin);

    SA_ASSERT(CONFIG_ACCESS_OPTION(INTERP, finest_elmat_callback));
    interp_data->finest_elmat_callback =
        CONFIG_ACCESS_OPTION(INTERP, finest_elmat_callback);

    interp_data->nparts = nparts;
    SA_ASSERT(0 < interp_data->nparts);
    interp_data->rhs_matrices_arr = new SparseMatrix*[nparts];
    interp_data->cut_evects_arr = new DenseMatrix*[nparts];
    interp_data->AEs_stiffm = new SparseMatrix*[nparts];
    for (int i=0; i < nparts; ++i)
    {
        interp_data->rhs_matrices_arr[i] = NULL;
        interp_data->cut_evects_arr[i] = NULL;
        interp_data->AEs_stiffm[i] = NULL;
    }

    SA_ASSERT(nu >= 0);
    SA_ASSERT(CONFIG_ACCESS_OPTION(INTERP, smoother_roots));
    interp_data->nu_interp = nu;
    interp_data->interp_smoother_roots =
        CONFIG_ACCESS_OPTION(INTERP, smoother_roots)(interp_data->nu_interp,
                                        &(interp_data->interp_smoother_degree));
    interp_data->times_apply_smoother = CONFIG_ACCESS_OPTION(INTERP,
                                            times_apply_smoother);

    if (SA_IS_OUTPUT_LEVEL(5))
    {
        SA_PRINTF("interp_smoother_degree: %d\n",
                  interp_data->interp_smoother_degree);
        PROC_STR_STREAM << "interp_smoother_roots: ";
        for (int i=0; i < interp_data->interp_smoother_degree; ++i)
            PROC_STR_STREAM << interp_data->interp_smoother_roots[i] << " ";
        PROC_STR_STREAM << "\n";
        SA_PRINTF("%s", PROC_STR_STREAM.str().c_str());
        PROC_CLEAR_STR_STREAM;
        SA_PRINTF("times_apply_smoother: %d\n",
                  interp_data->times_apply_smoother);
    }

    return interp_data;
}

void interp_free_data(interp_data_t *interp_data)
{
    if (!interp_data) return;
    mbox_free_matr_arr((Matrix **)interp_data->rhs_matrices_arr,
                       interp_data->nparts);
    mbox_free_matr_arr((Matrix **)interp_data->cut_evects_arr,
                       interp_data->nparts);
    mbox_free_matr_arr((Matrix **)interp_data->AEs_stiffm, interp_data->nparts);
    delete [] interp_data->interp_smoother_roots;
    delete interp_data;
}

interp_data_t *interp_copy_data(const interp_data_t *src)
{
    if (!src) return NULL;
    interp_data_t *dst = new interp_data_t;

    dst->nparts = src->nparts;
    dst->rhs_matrices_arr = mbox_copy_sparse_matr_arr(src->rhs_matrices_arr,
                                                      src->nparts);
    dst->cut_evects_arr = mbox_copy_dense_matr_arr(src->cut_evects_arr,
                                                   src->nparts);
    dst->AEs_stiffm = mbox_copy_sparse_matr_arr(src->AEs_stiffm, src->nparts);

    dst->finest_elmat_callback = src->finest_elmat_callback;

    dst->nu_interp = src->nu_interp;
    dst->interp_smoother_degree = src->interp_smoother_degree;
    dst->interp_smoother_roots =
        helpers_copy_dbl_arr(src->interp_smoother_roots,
                             src->interp_smoother_degree);
    dst->times_apply_smoother = src->times_apply_smoother;

    dst->local_prob_solve = src->local_prob_solve;

    dst->contrib_agg = src->contrib_agg;
    dst->eps_svd = src->eps_svd;
    dst->eps_lin = src->eps_lin;

    src->tent_interp_offsets.Copy(dst->tent_interp_offsets);

    return dst;
}

void interp_compute_vectors(const SparseMatrix& A,
    const agg_partititoning_relations_t& agg_part_rels,
    const interp_data_t& interp_data, void *elem_data_finest, double& tol,
    double& theta, bool *xbad_lin_indep, bool *vector_added, const Vector *xbad,
    bool transf, bool readapting, bool all_eigens, bool spect_update)
{
    const bool bdr_cond_imposed = CONFIG_ACCESS_OPTION(INTERP,
                                                       bdr_cond_imposed);
    const bool assemble_ess_diag = CONFIG_ACCESS_OPTION(INTERP,
                                                        assemble_ess_diag);
    const int nparts = agg_part_rels.nparts;

    double sum_skip = 0.;
    double min_skip = DBL_MAX;
    int skipctr = 0;

    DenseMatrix ** const cut_evects_arr = interp_data.cut_evects_arr;
    SparseMatrix ** const rhs_matrices_arr = interp_data.rhs_matrices_arr;
    SparseMatrix ** const AEs_stiffm = interp_data.AEs_stiffm;
    const spect_local_prob_solve_sparse_ft local_prob_solve =
                interp_data.local_prob_solve;

    bool xbad_lin_indep_local = false;
    bool vector_added_local = false;

    SA_ASSERT(0 < nparts);
    SA_ASSERT(rhs_matrices_arr);
    SA_ASSERT(cut_evects_arr);
    SA_ASSERT(AEs_stiffm);
    SA_ASSERT(!readapting || transf);
    SA_ASSERT(readapting || spect_update);
    SA_ASSERT(transf || spect_update);

    // If not a transformed problem (i.e. in a subspace) will be solved or not
    // readapting, the spectral method is necessary.
    if (!transf || !readapting)
        spect_update = true;

    SA_PRINTF_L(5, "theta: %g, tol: %g\n", theta, tol);

    // Loop over AEs.
    for (int i=0; i < nparts; ++i)
    {
        bool local_added = false;
        const SparseMatrix *AE_stiffm;
        Vector xbad_AE;
        DenseMatrix *Tt = NULL;
        double theta_local;
        theta_local = theta;

        SA_PRINTF_L(6, "%d ---------------------------------------------------"
                       "---------------------------------------------\n", i);

        if (!readapting)
        {
            // Build agglomerate stiffness matrix.
            SA_PRINTF_L(9, "%s", "Assembling local stiffness matrix...\n");
            SA_ASSERT(elem_data_finest);
            if (transf)
            {
                SA_ASSERT(AEs_stiffm[i]);
                delete AEs_stiffm[i];
                AEs_stiffm[i] = NULL;
            }
            SA_ASSERT(!AEs_stiffm[i]);
            AEs_stiffm[i] =
                agg_build_AE_stiffm_with_global(A, i, agg_part_rels,
                    interp_data.finest_elmat_callback, elem_data_finest,
                    bdr_cond_imposed, assemble_ess_diag);
        }
        AE_stiffm = AEs_stiffm[i];
        SA_ASSERT(AE_stiffm);

        SA_PRINTF_L(9, "AE size (DoFs): %d, big aggregate size: %d\n",
                    agg_part_rels.AE_to_dof->RowSize(i),
                    agg_part_rels.agg_size[i]);

        if (transf)
        {
            // The hierarchy is being adapted. So old cut vectors exist and
            // also the old r.h.s. matrices for the eigenvalue problems.
            SA_ASSERT(cut_evects_arr[i]);
            SA_ASSERT(rhs_matrices_arr[i]);

            // Restrict global badguy to AE.
            agg_restrict_vect_to_AE(i, agg_part_rels, *xbad, xbad_AE);

            Tt = new DenseMatrix;
            SA_ASSERT(Tt);

            bool local_lin_indep;
            if (spect_update)
            {
                // Orthogonalize coarse basis and badguy on AE.
                // Resulting orthogonal vectors stored in Tt.
                // Also, note whether a linear independent vector was
                // introduced.
                double ltol =
                    interp_data.eps_lin
                        /* * mbox_energy_norm_sparse(*(rhs_matrices_arr[i]),
                                                     xbad_AE) */;
                local_lin_indep =
                    mbox_orthogonalize_sparse(xbad_AE, *(cut_evects_arr[i]),
                                              *(rhs_matrices_arr[i]),
                                              *(rhs_matrices_arr[i]), ltol,
                                              *Tt);

                if (!readapting)
                {
                    delete rhs_matrices_arr[i];
                    rhs_matrices_arr[i] = NULL;
                }
            } else
            {
                SA_ASSERT(readapting);
                double denom = mbox_energy_norm_sparse(*AE_stiffm, xbad_AE);
                double ltol = tol * denom;
                SA_PRINTF_L(9, "xbad_AE norm: %g, tol * [xbad_AE norm]: %g\n",
                            denom, ltol);
                local_added =
                    mbox_orthogonalize_sparse(xbad_AE, *(cut_evects_arr[i]),
                                              *(rhs_matrices_arr[i]),
                                              *AE_stiffm, ltol, *Tt);
                if (local_added)
                    SA_SWAP(cut_evects_arr[i], Tt, DenseMatrix*);

                local_lin_indep = local_added;
            }

            xbad_lin_indep_local = xbad_lin_indep_local || local_lin_indep;
            SA_PRINTF_L(9, "Is a new vector introduced: %d\n", local_lin_indep);
        } else
        {
            // Simply allocate memory for the matrix of cut vectors.
            // This is in case the hierarchy is being built from scratch.
            SA_ASSERT(!cut_evects_arr[i]);
            cut_evects_arr[i] = new DenseMatrix;
            SA_ASSERT(!rhs_matrices_arr[i]);
            SA_ASSERT(spect_update);
        }
        SA_ASSERT(cut_evects_arr[i]);

        // Solve local eigenvalue problem and note if a vector was added.
        SA_ASSERT(rhs_matrices_arr[i] || !readapting);
        SA_ASSERT(rhs_matrices_arr[i] || spect_update);
        SA_ASSERT(!rhs_matrices_arr[i] || !spect_update || readapting);
        if (spect_update)
        {
            local_added =
                local_prob_solve(*AE_stiffm, rhs_matrices_arr[i], i, i,
                                 agg_part_rels.agg_size[i],
                                 agg_part_rels.aggregates, agg_part_rels,
                                 theta_local, *(cut_evects_arr[i]), Tt, transf,
                                 all_eigens);
        }

        if (transf)
        {
            vector_added_local =
                spect_update ? vector_added_local || local_added :
                               xbad_lin_indep_local;

            // Free the subspace matrices.
            SA_ASSERT(Tt);
            delete Tt;
        }

        // Compute the sum of skipped over aggs.
        SA_ASSERT(0. <= theta_local);
        if (spect_update)
        {
            sum_skip += theta_local;
            ++skipctr;
            if (theta_local < min_skip)
            {
                min_skip = theta_local;
            }
        }
    }

    SA_PRINTF_L(6, "%s", "end ------------------------------------------------"
                         "------------------------------------------------\n");
    if (transf)
    {
        SA_ASSERT(xbad_lin_indep);
        *xbad_lin_indep = xbad_lin_indep_local;
        SA_ASSERT(vector_added);
        *vector_added = vector_added_local;
        if (SA_IS_OUTPUT_LEVEL(5))
            SA_PRINTF("vector added (true/false): %d\n", *vector_added);
    }

    // Suggest a new theta.
    if (spect_update)
    {
        if (SA_IS_OUTPUT_LEVEL(5))
        {
            SA_PRINTF("Vectors skipped on %d agglomerates.\n", skipctr);
            SA_PRINTF("Average skipped over all agglomerates: %g\n",
                      sum_skip / (double)skipctr);
            SA_PRINTF("Min skipped over all agglomerates: %g\n", min_skip);
        }
        double thetap = sum_skip / (double)skipctr;
//        double thetap = min_skip; // This one or the one above.
        SA_ASSERT(skipctr == nparts);
        double eta = 0.5; //(double)skipctr / (double)nparts;

        // Compute a weighted average of the old theta and a proposal.
        if (skipctr > 0)
            theta = (1. - eta) * theta + eta * thetap;
        SA_PRINTF_L(5, "Suggested theta: %g\n", theta);
    }
}

SparseMatrix *interp_sparse_tent_build(const SparseMatrix& A,
    const agg_partititoning_relations_t& agg_part_rels,
    const interp_data_t& interp_data, void *elem_data_finest, double& tol,
    double& theta, bool *xbad_lin_indep, bool *vector_added, const Vector *xbad,
    bool transf, bool readapting, bool all_eigens, bool spect_update)
{
    SA_PRINTF_L(4, "%s", "---------- interp_compute_vectors { ----------------"
                         "---\n");

    interp_compute_vectors(A, agg_part_rels, interp_data, elem_data_finest, tol,
                           theta, xbad_lin_indep, vector_added, xbad, transf,
                           readapting, all_eigens, spect_update);

    if (SA_IS_OUTPUT_LEVEL(4))
    {
        SA_PRINTF("%s", "---------- } interp_compute_vectors ----------------"
                        "---\n");
        SA_PRINTF("%s", "---------- interp_sparse_tent_assemble { -----------"
                        "---\n");
    }

    SparseMatrix *tent_interp;
    tent_interp = interp_sparse_tent_assemble(agg_part_rels, interp_data);

    SA_PRINTF_L(4, "%s", "---------- } interp_sparse_tent_assemble -----------"
                         "---\n");

    return tent_interp;
}

SparseMatrix *interp_sparse_tent_assemble(
     const agg_partititoning_relations_t& agg_part_rels,
     const interp_data_t& interp_data)
{
    contrib_tent_struct_t *tent_int_struct;
    SparseMatrix *tent_interp;

    // Initialize the structure for building the tentative interpolator.
    tent_int_struct = contrib_tent_init(agg_part_rels.ND);

    // Input aggregates contributions.
    interp_data.contrib_agg(tent_int_struct, agg_part_rels,
                            interp_data.cut_evects_arr, interp_data.eps_svd);

    // Produce the tentative interpolant in its final form.
    tent_interp = contrib_tent_finalize(tent_int_struct);
    SA_ASSERT(tent_interp);
    SA_ASSERT(tent_interp->Finalized());

    return tent_interp;
}

HypreParMatrix *interp_global_tent_assemble(
     const agg_partititoning_relations_t& agg_part_rels,
     interp_data_t& interp_data, SparseMatrix *local_tent_interp)
{
    proc_allgather_offsets(local_tent_interp->Width(),
                           interp_data.tent_interp_offsets);

    HypreParMatrix *tent_interp_dof =
        new HypreParMatrix(agg_part_rels.fes->GlobalVSize(),
                           interp_data.tent_interp_offsets[PROC_NUM],
                           agg_part_rels.fes->GetDofOffsets(),
                           (int *)(interp_data.tent_interp_offsets),
                           local_tent_interp);

    HypreParMatrix *tdof_to_dof =
        agg_part_rels.fes->Dof_TrueDof_Matrix()->Transpose();
    SA_ASSERT(tdof_to_dof->GetGlobalNumCols() ==
              tent_interp_dof->GetGlobalNumRows());

    HypreParMatrix *tent_interp_tdof = ParMult(tdof_to_dof, tent_interp_dof);
    mbox_make_owner_rowstarts_colstarts(*tent_interp_tdof);

    delete tdof_to_dof;
    delete tent_interp_dof;

    return tent_interp_tdof;
}
