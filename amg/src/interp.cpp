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

namespace saamge
{
using namespace mfem;

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

void AltThresholdLocal(int m, double threshold,
                       HYPRE_Int * i_in, HYPRE_Int * j_in,
                       double * data_in, HYPRE_Int ** i_out_ptr,
                       HYPRE_Int ** j_out_ptr, double ** data_out_ptr)
{
    *i_out_ptr = new HYPRE_Int[m+1];
    HYPRE_Int * i_out = *i_out_ptr;
    i_out[0] = 0;

    std::vector<int> newj;
    std::vector<double> newdata;
    for (int i=0; i<m; ++i)
    {
        int thisrowcount = 0;
        for (int jp=i_in[i]; jp<i_in[i+1]; ++jp)
        {
            int j = j_in[jp];
            double val = data_in[jp];
            if (fabs(val) > threshold)
            {
                thisrowcount++;
                newj.push_back(j);
                newdata.push_back(val);
            }
        }
        i_out[i+1] = i_out[i] + thisrowcount;
    }
    unsigned int nnz = i_out[m];
    SA_ASSERT(nnz == newj.size());
    SA_ASSERT(nnz == newdata.size());

    *j_out_ptr = new HYPRE_Int[i_out[m]];
    *data_out_ptr = new double[i_out[m]];
    HYPRE_Int * j_out = *j_out_ptr;
    double * data_out = *data_out_ptr;
    for (unsigned int q=0; q<newj.size(); ++q)
    {
        j_out[q] = newj[q];
        data_out[q] = newdata[q];
    }
}

/**
   For some reason mfem::HypreParMatrix::Threshold()
   takes forever ever ever...

   also that Threshold does not seem to support assumed partition
 */
HypreParMatrix * AltThreshold(HypreParMatrix& mat, double val)
{
    hypre_ParCSRMatrix * hmat = mat;
    hypre_CSRMatrix * diag = hmat->diag;
    hypre_CSRMatrix * offd = hmat->offd;

    HYPRE_Int * diag_i, *diag_j, *offd_i, *offd_j;
    double * diag_data, *offd_data;

    AltThresholdLocal(diag->num_rows, val,
                      diag->i, diag->j, diag->data,
                      &diag_i, &diag_j, &diag_data);
    AltThresholdLocal(offd->num_rows, val,
                      offd->i, offd->j, offd->data,
                      &offd_i, &offd_j, &offd_data);

    HYPRE_Int offd_num_cols = offd->num_cols;

   /** Creates general (rectangular) parallel matrix. The new HypreParMatrix
       takes ownership of all input arrays, except col_starts and row_starts. */
        /*
   HypreParMatrix(MPI_Comm comm,
                  HYPRE_Int global_num_rows, HYPRE_Int global_num_cols,
                  HYPRE_Int *row_starts, HYPRE_Int *col_starts,
                  HYPRE_Int *diag_i, HYPRE_Int *diag_j, double *diag_data,
                  HYPRE_Int *offd_i, HYPRE_Int *offd_j, double *offd_data,
                  HYPRE_Int offd_num_cols, HYPRE_Int *offd_col_map);
        */
    HypreParMatrix * newmat = new HypreParMatrix(
        mat.GetComm(), mat.M(), mat.N(),
        hmat->row_starts, hmat->col_starts, diag_i, diag_j, diag_data,
        offd_i, offd_j, offd_data, offd_num_cols, hmat->col_map_offd);
    newmat->CopyRowStarts();
    newmat->CopyColStarts();

    return newmat;
}

HypreParMatrix *interp_smooth(
    int degree, double *roots, double drop_tol,
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

    SA_RPRINTF_L(0, 4, "%s", "Smoothing tentative prolongator...\n");
    if (SA_IS_OUTPUT_LEVEL(5))
    {
        PROC_STR_STREAM << "Interp roots: ";
        for (int k=0; k < degree; ++k)
            PROC_STR_STREAM << roots[k] << " ";
        PROC_STR_STREAM << "\n";
        SA_RPRINTF(0, "%s", PROC_STR_STREAM.str().c_str());
        PROC_CLEAR_STR_STREAM;
    }

    interp = mbox_clone_parallel_matrix(&tent);
    for (int k=0; k < degree; ++k)
    {
        iter_matr = mbox_scale_clone_parallel_matrix(*smoother_matr,
                                                     1./roots[k]);
        mbox_add_diag_parallel_matrix(*iter_matr, 1.);

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

    // something is going wrong here in parallel with drop_tol > 0.0

    if (drop_tol == 0.0)
    {
        return interp;
    }
    else
    {
        new_interp = AltThreshold(*interp, drop_tol);
        delete interp;
        return new_interp;
    }
}

interp_data_t *interp_init_data(
    const agg_partitioning_relations_t& agg_part_rels, int nu_pro, 
    bool use_arpack, bool scaling_P)
{
    const int nparts = agg_part_rels.nparts;
    interp_data_t *interp_data = new interp_data_t;
    SA_ASSERT(interp_data);
    memset(interp_data, 0, sizeof(*interp_data));

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

    SA_ASSERT(nu_pro >= 0);
    interp_data->nu_pro = nu_pro;
    interp_data->interp_smoother_roots = smpr_sa_poly_roots(
        interp_data->nu_pro, &(interp_data->interp_smoother_degree));
    interp_data->times_apply_smoother = 1;

    interp_data->use_arpack = use_arpack;
    interp_data->scaling_P = scaling_P;
    interp_data->drop_tol = 0.0;

    if (SA_IS_OUTPUT_LEVEL(5))
    {
        SA_RPRINTF(0,"interp_smoother_degree: %d\n",
                   interp_data->interp_smoother_degree);
        PROC_STR_STREAM << "interp_smoother_roots: ";
        for (int i=0; i < interp_data->interp_smoother_degree; ++i)
            PROC_STR_STREAM << interp_data->interp_smoother_roots[i] << " ";
        PROC_STR_STREAM << "\n";
        SA_RPRINTF(0,"%s", PROC_STR_STREAM.str().c_str());
        PROC_CLEAR_STR_STREAM;
        SA_RPRINTF(0,"times_apply_smoother: %d\n",
                   interp_data->times_apply_smoother);
    }

    return interp_data;
}

void interp_free_data(interp_data_t *interp_data,
                      bool doing_spectral,
                      double theta)
{
    if (!interp_data) return;

    if (doing_spectral)
    {
        mbox_free_matr_arr((Matrix **)interp_data->rhs_matrices_arr,
                           interp_data->nparts);
        mbox_free_matr_arr((Matrix **)interp_data->cut_evects_arr,
                           interp_data->nparts);
        mbox_free_matr_arr((Matrix **)interp_data->AEs_stiffm,
                           interp_data->nparts);
    }
    else
    {
        delete [] interp_data->rhs_matrices_arr;
        delete [] interp_data->cut_evects_arr;
        delete [] interp_data->AEs_stiffm;
    }
    delete [] interp_data->interp_smoother_roots;
    delete interp_data->local_coarse_one_representation;
    delete [] interp_data->mis_numcoarsedof;

    mbox_free_matr_arr((Matrix**) interp_data->mis_tent_interps,
                       interp_data->num_mises);

    mbox_free_matr_arr((Matrix **)interp_data->Aii, interp_data->nparts);
    mbox_free_matr_arr((Matrix **)interp_data->Abb, interp_data->nparts);
    mbox_free_matr_arr((Matrix **)interp_data->Aib, interp_data->nparts);
    mbox_free_matr_arr((Matrix **)interp_data->invAii, interp_data->nparts);
    mbox_free_matr_arr((Matrix **)interp_data->invAiiAib, interp_data->nparts);
    mbox_free_matr_arr((Matrix **)interp_data->AbiinvAii, interp_data->nparts);
    mbox_free_matr_arr((Matrix **)interp_data->schurs, interp_data->nparts);
    mbox_free_matr_arr((Matrix **)interp_data->cfaces_bases, interp_data->num_cfaces);

    delete [] interp_data->celements_cdofs_offsets;
    delete [] interp_data->cfaces_truecdofs_offsets;
    delete [] interp_data->cfaces_cdofs_offsets;

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

    // dst->finest_elmat_callback = src->finest_elmat_callback;

    dst->nu_pro = src->nu_pro;
    dst->interp_smoother_degree = src->interp_smoother_degree;
    dst->interp_smoother_roots =
        helpers_copy_dbl_arr(src->interp_smoother_roots,
                             src->interp_smoother_degree);
    dst->times_apply_smoother = src->times_apply_smoother;

    src->tent_interp_offsets.Copy(dst->tent_interp_offsets);
    dst->total_cols_interp = src->total_cols_interp;

    dst->Aii = mbox_copy_matr_arr(src->Aii, src->nparts);
    dst->Abb = mbox_copy_matr_arr(src->Abb, src->nparts);
    dst->Aib = mbox_copy_matr_arr(src->Aib, src->nparts);
    dst->invAii = mbox_copy_dense_matr_arr(src->invAii, src->nparts);
    dst->invAiiAib = mbox_copy_dense_matr_arr(src->invAiiAib, src->nparts);
    dst->AbiinvAii = mbox_copy_dense_matr_arr(src->AbiinvAii, src->nparts);
    dst->schurs = mbox_copy_dense_matr_arr(src->schurs, src->nparts);
    dst->cfaces_bases = mbox_copy_dense_matr_arr(src->cfaces_bases, src->num_cfaces);
    dst->num_cfaces = src->num_cfaces;

    dst->celements_cdofs = src->celements_cdofs;
    dst->celements_cdofs_offsets = helpers_copy_int_arr(src->celements_cdofs_offsets,
                                                        src->nparts + 1);
    dst->cfaces_truecdofs_offsets = helpers_copy_int_arr(src->cfaces_truecdofs_offsets,
                                                         src->num_cfaces + 1);
    dst->cfaces_cdofs_offsets = helpers_copy_int_arr(src->cfaces_cdofs_offsets,
                                                     src->num_cfaces + 1);

    return dst;
}

/**
   Actually solve the local eigenvalue problems

   we sometimes run out of memory in this routine...
   don't know if it's the local matrix assembly or the eigenvector solve
*/
void interp_compute_vectors(
    const agg_partitioning_relations_t& agg_part_rels,
    const interp_data_t& interp_data, ElementMatrixProvider *elem_data,
    double tol, double& theta, bool *xbad_lin_indep, bool *vector_added,
    const Vector *xbad, bool transf, bool readapting,
    bool all_eigens, bool spect_update, bool bdr_cond_imposed)
{
    // const bool assemble_ess_diag = true;
    const int nparts = agg_part_rels.nparts;

    double sum_skip = 0.;
    double min_skip = DBL_MAX;
    int skipctr = 0;

    DenseMatrix ** const cut_evects_arr = interp_data.cut_evects_arr;
    SparseMatrix ** const rhs_matrices_arr = interp_data.rhs_matrices_arr;
    SparseMatrix ** const AEs_stiffm = interp_data.AEs_stiffm;

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

    SA_RPRINTF_L(0, 5, "theta: %g, tol: %g\n", theta, tol);

    int arpack_size_threshold;
    if (interp_data.use_arpack)
        arpack_size_threshold = ARPACK_SIZE_THRESHOLD; // ??? 64, but no good reason for that
    else
        arpack_size_threshold = std::numeric_limits<int>::max();
    Eigensolver eigensolver(agg_part_rels.mises, agg_part_rels,
                            arpack_size_threshold);

    // Loop over AEs.
    for (int i=0; i<nparts; ++i)
    {
        if (nparts < 10 || i % (nparts / 10) == 0)
            SA_RPRINTF_L(0, 5, "  local eigenvalue problem %d / %d\n", i, nparts);
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
            SA_ASSERT(elem_data);
            if (transf)
            {
                SA_ASSERT(AEs_stiffm[i]);
                delete AEs_stiffm[i];
                AEs_stiffm[i] = NULL;
            }
            SA_ASSERT(!AEs_stiffm[i]); // we demand to assemble these ourselves
            AEs_stiffm[i] = elem_data->BuildAEStiff(i);
        }
        AE_stiffm = AEs_stiffm[i];
        SA_ASSERT(AE_stiffm);
        if (agg_part_rels.testmesh &&
            !elem_data->IsGeometric())
        {
            std::stringstream filename;
            filename << "AE_stiffm_" << i << "." << PROC_RANK << ".mat";
            std::ofstream out(filename.str().c_str());
            AE_stiffm->Print(out);
        }

        SA_PRINTF_L(9, "AE size (DoFs): %d, MIS size: %d\n",
                    agg_part_rels.AE_to_dof->RowSize(i),
                    agg_part_rels.mises_size[i]);

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
                double ltol = INTERP_LINEAR_TOLERANCE;
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
            } 
            else
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
        } 
        else
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
            int agg_size = -1; // this only has any effect if we are doing the schur eigenproblem...
            if (agg_part_rels.mises_size != NULL)
                agg_size = agg_part_rels.mises_size[i];
            local_added = eigensolver.Solve(
                *AE_stiffm, rhs_matrices_arr[i], i, i,
                agg_size,
                theta_local, *(cut_evects_arr[i]));
        }

        // test routine for mltest, put an extra eigenvector on AE 0 [on processor 0]
        if (agg_part_rels.testmesh && i == 0 && PROC_RANK == 0)
        {
            int h = cut_evects_arr[i]->Height();
            int w = cut_evects_arr[i]->Width() + 1;

            double * temp = new double[h * w];
            memcpy(temp, cut_evects_arr[i]->Data(), h * (w-1) * sizeof(double));
            for (int j=0; j<h; ++j)
                temp[h * (w-1) + j] = 1.0;
            delete cut_evects_arr[i];
            cut_evects_arr[i] = new DenseMatrix(h,w);
            memcpy(cut_evects_arr[i]->Data(), temp, h * w * sizeof(double));
            delete [] temp;
        }

        if (agg_part_rels.testmesh)
        {
            std::stringstream filename;
            filename << "cut_evects_arr_" << i << "." << PROC_RANK << ".densemat";
            std::ofstream out(filename.str().c_str());
            cut_evects_arr[i]->Print(out);
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
            SA_RPRINTF(0,"Vectors skipped on %d agglomerates.\n", skipctr);
            SA_RPRINTF(0,"Average skipped over all agglomerates: %g\n",
                       sum_skip / (double)skipctr);
            SA_RPRINTF(0,"Min skipped over all agglomerates: %g\n", min_skip);
        }
        double thetap = sum_skip / (double)skipctr;
//        double thetap = min_skip; // This one or the one above.
        SA_ASSERT(skipctr == nparts);
        double eta = 0.5; //(double)skipctr / (double)nparts;

        // Compute a weighted average of the old theta and a proposal.
        if (skipctr > 0)
            theta = (1. - eta) * theta + eta * thetap;
        SA_RPRINTF_L(0, 5, "Suggested theta: %g\n", theta);
    }

    if (SA_IS_OUTPUT_LEVEL(5))
        eigensolver.PrintStatistics();
}

Vector **interp_compute_vectors_nostore(const agg_partitioning_relations_t& agg_part_rels,
    const interp_data_t& interp_data, ElementMatrixProvider *elem_data, double theta, bool full_space)
{
    const int nparts = agg_part_rels.nparts;

    DenseMatrix ** const cut_evects_arr = interp_data.cut_evects_arr;
    SparseMatrix *rhs_matrix;
    SparseMatrix *AE_stiffm;
    Vector **evals = new Vector*[nparts];

    SA_ASSERT(0 < nparts);
    SA_ASSERT(cut_evects_arr);

    SA_RPRINTF_L(0, 5, "theta: %g\n", theta);

    int arpack_size_threshold;
    if (interp_data.use_arpack)
        arpack_size_threshold = ARPACK_SIZE_THRESHOLD; // Some hard-coded parameter.
    else
        arpack_size_threshold = std::numeric_limits<int>::max();
    Eigensolver eigensolver(NULL, agg_part_rels, arpack_size_threshold);

    // Loop over AEs.
    for (int i=0; i < nparts; ++i)
    {
        if (nparts < 10 || i % (nparts / 10) == 0)
            SA_RPRINTF_L(0, 5, "  local eigenvalue problem %d / %d\n", i, nparts);

        SA_PRINTF_L(6, "%d ---------------------------------------------------"
                    "---------------------------------------------\n", i);

        SA_PRINTF_L(9, "%s", "Assembling local stiffness matrix...\n");
        SA_ASSERT(elem_data);
        SA_ASSERT(interp_data.AEs_stiffm);
        SA_ASSERT(!interp_data.AEs_stiffm[i]);
        AE_stiffm = elem_data->BuildAEStiff(i);
        if (full_space)
            interp_data.AEs_stiffm[i] = AE_stiffm;
        SA_ASSERT(AE_stiffm);
        SA_ASSERT(AE_stiffm->Width() == AE_stiffm->Height());
        SA_ASSERT(NULL != interp_data.rhs_matrices_arr);
        SA_ASSERT(NULL == interp_data.rhs_matrices_arr[i]);
        interp_data.rhs_matrices_arr[i] = mbox_snd_diagA_sparse_from_sparse(*AE_stiffm);
        if (agg_part_rels.testmesh &&
            !elem_data->IsGeometric())
        {
            std::stringstream filename;
            filename << "AE_stiffm_" << i << "." << PROC_RANK << ".mat";
            std::ofstream out(filename.str().c_str());
            AE_stiffm->Print(out);
        }

        SA_PRINTF_L(9, "AE size (DoFs): %d\n", agg_part_rels.AE_to_dof->RowSize(i));

        // Simply allocate memory for the matrix of cut vectors.
        // This is in case the hierarchy is being built from scratch, which is the only case in this routine.
        SA_ASSERT(!cut_evects_arr[i]);
        cut_evects_arr[i] = new DenseMatrix;
        SA_ASSERT(cut_evects_arr[i]);

        // Solve local eigenvalue problem.
        if (full_space)
        {
            int nintdofs = 0;
            for (int j=0; j < AE_stiffm->Width(); ++j)
            {
                const int gj = agg_num_col_to_glob(*agg_part_rels.AE_to_dof, i, j);
                SA_ASSERT(0 <= gj && gj < agg_part_rels.ND);
                if (!SA_IS_SET_A_FLAG(agg_part_rels.agg_flags[gj], AGG_ON_ESS_DOMAIN_BORDER_FLAG))
                    ++nintdofs;
            }
            cut_evects_arr[i]->SetSize(AE_stiffm->Width(), nintdofs);
            nintdofs = 0;
            for (int j=0; j < AE_stiffm->Width(); ++j)
            {
                const int gj = agg_num_col_to_glob(*agg_part_rels.AE_to_dof, i, j);
                SA_ASSERT(0 <= gj && gj < agg_part_rels.ND);
                if (!SA_IS_SET_A_FLAG(agg_part_rels.agg_flags[gj], AGG_ON_ESS_DOMAIN_BORDER_FLAG))
                {
                    SA_ASSERT(nintdofs < cut_evects_arr[i]->Width());
                    cut_evects_arr[i]->Elem(j, nintdofs) = 1.0;
                    ++nintdofs;
                }
            }
            //cut_evects_arr[i]->Diag(1.0, AE_stiffm->Width());
            evals[i] = NULL;
        } else
        {
            rhs_matrix = NULL;
            eigensolver.Solve(*AE_stiffm, rhs_matrix, i, -1, -1, theta, *(cut_evects_arr[i]));
            delete rhs_matrix;
            delete AE_stiffm;
            evals[i] = eigensolver.StealEigenvalues();
            SA_ASSERT(NULL != evals[i]);
        }

        // test routine for mltest, put an extra eigenvector on AE 0 [on processor 0]
        if (agg_part_rels.testmesh && i == 0 && PROC_RANK == 0)
        {
            int h = cut_evects_arr[i]->Height();
            int w = cut_evects_arr[i]->Width() + 1;

            double * temp = new double[h * w];
            memcpy(temp, cut_evects_arr[i]->Data(), h * (w-1) * sizeof(double));
            for (int j=0; j<h; ++j)
                temp[h * (w-1) + j] = 1.0;
            delete cut_evects_arr[i];
            cut_evects_arr[i] = new DenseMatrix(h,w);
            memcpy(cut_evects_arr[i]->Data(), temp, h * w * sizeof(double));
            delete [] temp;
        }

        if (agg_part_rels.testmesh)
        {
            std::stringstream filename;
            filename << "cut_evects_arr_" << i << "." << PROC_RANK << ".densemat";
            std::ofstream out(filename.str().c_str());
            cut_evects_arr[i]->Print(out);
        }
    }

    SA_PRINTF_L(6, "%s", "end ------------------------------------------------"
                         "------------------------------------------------\n");

    if (SA_IS_OUTPUT_LEVEL(5))
        eigensolver.PrintStatistics();

    return evals;
}

/**
   Idea here is to build spectral + polynomial space.
*/
SparseMatrix * interp_build_composite(
    const agg_partitioning_relations_t &agg_part_rels,
    interp_data_t& interp_data, ElementMatrixProvider *elem_data, double theta,
    int spatial_dimension, int num_nodes, const Vector& coords, int order,
    bool avoid_ess_bdr_dofs, bool use_spectral)
{
    const double tol = 0.0; // ? for readapting or something
    bool *xbad_lin_indep = NULL; // TODO const
    bool *vector_added = NULL; // TODO const
    const Vector *xbad = NULL;
    const bool transf = false;
    const bool readapting = false;
    const bool all_eigens = false;
    const bool spect_update = true;

    // this fills interp_data.cut_evects_arr
    if (use_spectral)
    {
        interp_compute_vectors(
            agg_part_rels, interp_data, elem_data, tol, theta, xbad_lin_indep,
            vector_added, xbad, transf, readapting, all_eigens, spect_update,
            avoid_ess_bdr_dofs);
    }

    SparseMatrix *tent_interp;
    ContribTent tent_int_struct(agg_part_rels.ND, avoid_ess_bdr_dofs);
    if (use_spectral)
    {
        tent_int_struct.contrib_composite(
            agg_part_rels, interp_data.cut_evects_arr, order, spatial_dimension,
            num_nodes, coords);
    }
    else
    {
        if (order == 0)
        {
            tent_int_struct.contrib_ones(agg_part_rels);
        }
        else if (order == 1)
        {
            SA_ASSERT(agg_part_rels.ND == num_nodes); // not elasticity...
            tent_int_struct.contrib_linears(
                agg_part_rels, spatial_dimension, num_nodes, coords);
        }
        else
        {
            SA_ASSERT(false);
        }
    }

    // Below local_coarse_one_representation, mis_numcoardof, mis_tent_interps
    // are all copying pointers, with the expectation that interp_data will
    // delete them later on
    interp_data.local_coarse_one_representation =
        tent_int_struct.get_local_coarse_one_representation();
    interp_data.coarse_truedof_offset =
        tent_int_struct.get_coarse_truedof_offset();
    interp_data.mis_numcoarsedof = tent_int_struct.get_mis_numcoarsedof();
    interp_data.mis_tent_interps = tent_int_struct.get_mis_tent_interps();
    interp_data.num_mises = agg_part_rels.num_mises;

    // Produce the tentative interpolant in its (local) almost final form.
    // (this builds local_tent_interp and copies/moves some data structures and
    // pointers around)
    tent_interp = tent_int_struct.contrib_tent_finalize();
    SA_ASSERT(tent_interp);
    SA_ASSERT(tent_interp->Finalized());

    return tent_interp;
}

/**
   Trying to build an interpolant with no spectral information and
   only a polynomial coarse space. This should be a drop-in
   replacement for interp_sparse_tent_build() and should be (very)
   similar to it when theta = 0 and order = 0, but faster.
*/
SparseMatrix *interp_build_polynomial(
    const agg_partitioning_relations_t &agg_part_rels,
    interp_data_t& interp_data, int spatial_dimension,
    int num_nodes, const Vector& coords, int order, bool avoid_ess_bdr_dofs)
{
    return interp_build_composite(
        agg_part_rels, interp_data, NULL, 0.0, spatial_dimension, num_nodes,
        coords, order, avoid_ess_bdr_dofs, false);
}

SparseMatrix *interp_build_minimal(
    const agg_partitioning_relations_t &agg_part_rels,
    interp_data_t& interp_data)
{
    Vector dummy;
    return interp_build_polynomial(agg_part_rels, interp_data,
                                   -1, -1, dummy, 0, true);
}

SparseMatrix *interp_sparse_tent_build(
    const agg_partitioning_relations_t& agg_part_rels,
    interp_data_t& interp_data, ElementMatrixProvider *elem_data, double& tol,
    double& theta, bool *xbad_lin_indep, bool *vector_added, const Vector *xbad,
    bool transf, bool readapting, bool all_eigens, bool spect_update,
    bool avoid_ess_bdr_dofs)
{
    SA_RPRINTF_L(0,4, "%s", "---------- interp_compute_vectors { --------------"
                 "-----\n");

    bool bdr_cond_imposed = avoid_ess_bdr_dofs;
    interp_compute_vectors(
        agg_part_rels, interp_data, elem_data, tol, theta, xbad_lin_indep,
        vector_added, xbad, transf, readapting, all_eigens, spect_update,
        bdr_cond_imposed);

    if (SA_IS_OUTPUT_LEVEL(4))
    {
        SA_RPRINTF(0,"%s", "---------- } interp_compute_vectors ----------------"
                   "---\n");
        SA_RPRINTF(0,"%s", "---------- interp_sparse_tent_assemble { -----------"
                   "---\n");
    }

    SparseMatrix *tent_interp;
    tent_interp = interp_sparse_tent_assemble(agg_part_rels, interp_data,
                                              avoid_ess_bdr_dofs);

    SA_RPRINTF_L(0,4, "%s", "---------- } interp_sparse_tent_assemble -----------"
                 "---\n");

    return tent_interp;
}

SparseMatrix *interp_sparse_tent_assemble(
    const agg_partitioning_relations_t& agg_part_rels,
    interp_data_t& interp_data, bool avoid_ess_bdr_dofs)
{
    SparseMatrix *tent_interp;

    // Initialize the structure for building the tentative interpolator.
    ContribTent tent_int_struct(agg_part_rels.ND, avoid_ess_bdr_dofs);

    // Input aggregates [mises] contributions.
    // on modern parallel multilevel branches this should be contrib_mises()
    // in original Delyan code it is probably contrib_big_aggs_nosvd() 
    tent_int_struct.contrib_mises(agg_part_rels,
                                  interp_data.cut_evects_arr,
                                  interp_data.scaling_P);

    interp_data.local_coarse_one_representation = tent_int_struct.get_local_coarse_one_representation(); // copying a pointer (interp_data deletes)
    interp_data.coarse_truedof_offset = tent_int_struct.get_coarse_truedof_offset();
    interp_data.mis_numcoarsedof = tent_int_struct.get_mis_numcoarsedof(); // copying a pointer (interp_data deletes)
    interp_data.num_mises = agg_part_rels.num_mises;
    // SA_RPRINTF(PROC_NUM-1,"coarse_truedofoffset = %d, mises on this processor = %d\n",
    //         interp_data.coarse_truedof_offset, interp_data.num_mises);
    interp_data.mis_tent_interps = tent_int_struct.get_mis_tent_interps(); // copying a pointer (interp_data deletes)

    // Produce the tentative interpolant in its final form.
    // (this just copies/moves some data structures and pointers around)
    tent_interp = tent_int_struct.contrib_tent_finalize();
    SA_ASSERT(tent_interp);
    SA_ASSERT(tent_interp->Finalized());

    return tent_interp;
}

HypreParMatrix *interp_global_tent_assemble(
     const agg_partitioning_relations_t& agg_part_rels,
     interp_data_t& interp_data, SparseMatrix *local_tent_interp)
{
    int total_columns;
    if (interp_data.tent_interp_offsets.Size() > 0)
        total_columns = interp_data.total_cols_interp;
    else
    {
        proc_determine_offsets(local_tent_interp->Width(),
                               interp_data.tent_interp_offsets,
                               total_columns);
        interp_data.total_cols_interp = total_columns;
    }

    int * dof_offsets = agg_part_rels.Dof_TrueDof->RowPart();
    // may need to append the total number because Tzanio is a fool

    HypreParMatrix *tent_interp_dof = new HypreParMatrix(
        PROC_COMM, agg_part_rels.Dof_TrueDof->GetGlobalNumRows(),
        total_columns, dof_offsets, (int *)(interp_data.tent_interp_offsets),
        local_tent_interp);

    HypreParMatrix *tdof_to_dof = 
        agg_part_rels.Dof_TrueDof->Transpose();
    SA_ASSERT(tdof_to_dof->GetGlobalNumCols() ==
              tent_interp_dof->GetGlobalNumRows());

    HypreParMatrix *tent_interp_tdof = ParMult(tdof_to_dof, tent_interp_dof);
    mbox_make_owner_rowstarts_colstarts(*tent_interp_tdof);

    delete tdof_to_dof;
    delete tent_interp_dof;

    return tent_interp_tdof;
}

HypreParMatrix *interp_global_restr_assemble(
     const agg_partitioning_relations_t& agg_part_rels,
     interp_data_t& interp_data, SparseMatrix *local_tent_restr)
{
    int total_rows;
    if (interp_data.tent_interp_offsets.Size() > 0)
        total_rows = interp_data.total_cols_interp;
    else
    {
        proc_determine_offsets(local_tent_restr->Height(),
                               interp_data.tent_interp_offsets,
                               total_rows);
        interp_data.total_cols_interp = total_rows;
    }

    int * dof_offsets = agg_part_rels.Dof_TrueDof->RowPart();
    // May need to append the total number. XXX: This is a weird thing.

    HypreParMatrix *tent_restr_dof = new HypreParMatrix(
        PROC_COMM, total_rows, agg_part_rels.Dof_TrueDof->GetGlobalNumRows(),
        (int *)(interp_data.tent_interp_offsets), dof_offsets,
        local_tent_restr);

    SA_ASSERT(tent_restr_dof->GetGlobalNumCols() ==
              agg_part_rels.Dof_TrueDof->GetGlobalNumRows());

    HypreParMatrix *tent_restr_tdof = ParMult(tent_restr_dof, agg_part_rels.Dof_TrueDof);
    mbox_make_owner_rowstarts_colstarts(*tent_restr_tdof);

    delete tent_restr_dof;

    return tent_restr_tdof;
}

/**
   ATB 7 October 2014, follows pattern of interp_global_tent_assemble

   memory use, malloc etc. here is a nightmare...

   also build HypreParVector version of local_coarse_one_representation 

   DEPRECATED
*/
HypreParVector *interp_global_coarse_one_assemble(
    const agg_partitioning_relations_t& agg_part_rels,
    interp_data_t& interp_data, SparseMatrix *local_tent_interp,
    Array<double> * local_coarse_one_representation)
{
    // may not need to redo this if we just did interp_global_tent_assemble()
    // proc_allgather_offsets(local_tent_interp->Width(),
    //                     interp_data.tent_interp_offsets);
    int total;
    proc_determine_offsets(local_tent_interp->Width(),
                           interp_data.tent_interp_offsets,
                           total);
    
    // double * data = (double*) malloc(local_coarse_one_representation->Size() * sizeof(double));
    // SA_ASSERT(data);
    const double * olddata = local_coarse_one_representation->GetData();
    // memcpy(data,olddata,local_coarse_one_representation->Size()*sizeof(double));

    // TODO---should probably just get conforming vector from matrix...
    HypreParVector * coarse_one_representation = 
        new HypreParVector(PROC_COMM, total,
                           (int*) interp_data.tent_interp_offsets);
    hypre_ParVector * hnco = (hypre_ParVector*) (*coarse_one_representation);
    // coarse_one_representation->BecomeVectorOwner(); // commented out ATB 19 December 2014, may cause memory leak
    memcpy(hypre_ParVectorLocalVector(hnco)->data,olddata,
           local_coarse_one_representation->Size()*sizeof(double));
    hypre_ParVectorOwnsData(hnco) = 1;
    hypre_ParVectorOwnsPartitioning(hnco) = 0; // offsets are owned by the interpolation matrix
    // TODO probably need to do some more specific Hypre owns_col_starts garbage as well as the above line...

    return coarse_one_representation;
}

/**
   directly assemble the scaling matrix P from the local_coarse_one_representation

   eventually replace the interp_global_coarse_one_assemble() routine above
   and also the build_scaling_P() routine in spe10.cpp

   ATB 20 February 2015
*/
HypreParMatrix * interp_scaling_P_assemble(
    const agg_partitioning_relations_t& agg_part_rels,
    interp_data_t& interp_data, SparseMatrix *local_tent_interp,
    Array<double> * local_coarse_one_representation)
{
    // make a local matrix, with rows: local_coarse_one_representation->Size() and columns: number of local AE [-> now MISes]
    int num_mises = agg_part_rels.num_mises;

    int num_local_rows = local_coarse_one_representation->Size();
    // the following assertion fails if you try to do linears in your coarse
    // space with corrected_nullspace on the same level (ie, two-level)
    // also fails for --correct-nulspace with constants (not necessarily linears)
    // also fails for --correct-nulspace with elasticity and rigid body modes
    SA_ASSERT(num_local_rows == local_tent_interp->Width());
    
    int num_local_cols = 0;
    for (int j=0; j<num_mises; ++j)
        if (agg_part_rels.mis_master[j] == PROC_RANK &&
            interp_data.mis_numcoarsedof[j] > 0)
            num_local_cols++;

    SA_RPRINTF_L(0, 8, "local scaling_P is %d by %d\n",
                 num_local_rows,num_local_cols); 
    SparseMatrix * serial_out = new SparseMatrix(num_local_rows, num_local_cols);

    // put the entries in local matrix
    int running_total = 0;
    int col = 0;
    for (int j=0; j<num_mises; ++j)
        if (agg_part_rels.mis_master[j] == PROC_RANK)
        {
            // this assertion fails if we are doing testmesh, because we add extra coarse dofs for debugging but don't put them in local_coarse_one_representation
            // SA_ASSERT(running_total + interp_data.mis_numcoarsedof[j] < num_local_rows);
            for (int i=0; i<interp_data.mis_numcoarsedof[j]; ++i)
            {
                double val = local_coarse_one_representation->GetData()[running_total + i];
                serial_out->Set(running_total + i, col, val);
            }
            if (interp_data.mis_numcoarsedof[j] > 0) col++;
            running_total += interp_data.mis_numcoarsedof[j];
        }
    serial_out->Finalize();
    SA_ASSERT(num_local_cols == col);
    SA_ASSERT(num_local_rows == running_total);

    // put together the parallel matrix
    int global_num_rows;
    proc_determine_offsets(num_local_rows,
                           interp_data.tent_interp_offsets,
                           global_num_rows);
    int global_num_cols;
    Array<int> cols;
    proc_determine_offsets(num_local_cols,
                           cols, global_num_cols);
    
    int * modifyJ = serial_out->GetJ();
    for (int q=0; q<serial_out->NumNonZeroElems(); ++q)
        modifyJ[q] = modifyJ[q] + cols[0];

    // we use this constructor because it copies the row_part and col_part
    HypreParMatrix * out = new HypreParMatrix(
        PROC_COMM, num_local_rows,  global_num_rows, global_num_cols,
        serial_out->GetI(), modifyJ, serial_out->GetData(),
        interp_data.tent_interp_offsets.GetData(), cols.GetData());

    delete serial_out;
    return out;
}

} // namespace saamge
