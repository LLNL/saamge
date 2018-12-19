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
#include "spectral.hpp"
#include <climits>
#include <mfem.hpp>
#include "aggregates.hpp"
#include "xpacks.hpp"
#include "arpacks.hpp"
#include "helpers.hpp"
#include "mbox.hpp"

namespace saamge
{
using namespace mfem;

Eigensolver::Eigensolver(
    const int * aggregates, 
    const agg_partitioning_relations_t &agg_part_rels,
    int threshold)
    :
    aggregates(aggregates),
    agg_part_rels(agg_part_rels),
    threshold(threshold),
    transf(false),
    all_eigens(false),
    max_arpack_vectors(10),
    count_solves(0),
    count_direct_solves(0),
    count_max_used(0),
    smallest_eigenvalue_skipped(std::numeric_limits<double>::max()),
    eigenvalues(NULL)
{
}

Eigensolver::~Eigensolver()
{
    delete eigenvalues;
}

void Eigensolver::GetStatistics(
    int &o_count_solves, int &o_count_direct_solves,
    int &o_count_max_used, double &o_smallest_eigenvalue_skipped)
{
    o_count_solves = count_solves;
    o_count_direct_solves = count_direct_solves;
    o_count_max_used = count_max_used;
    o_smallest_eigenvalue_skipped = smallest_eigenvalue_skipped;
}

void Eigensolver::PrintStatistics()
{
    if (PROC_RANK == 0)
    {
        std::cout << "  [" << PROC_RANK << "] count_solves = " 
                  << count_solves << std::endl;
        std::cout << "  [" << PROC_RANK << "] count_direct_solves = " 
                  << count_direct_solves << std::endl;
        std::cout << "  [" << PROC_RANK << "] count_max_used = " 
                  << count_max_used << std::endl;
        std::cout << "  [" << PROC_RANK << "] smallest_eigenvalue_skipped = " 
                  << smallest_eigenvalue_skipped << std::endl;
    }
}

bool Eigensolver::Solve(
    const mfem::SparseMatrix& A, mfem::SparseMatrix *& B, 
    int part, int agg_id, int aggregate_size, double& theta,
    mfem::DenseMatrix& cut_evects, int fixed_num)
{
    int problem_size = A.Width();
    count_solves++;
    if (problem_size <= threshold)
    {
        count_direct_solves++;
        return SolveDirect(A, B, part, agg_id, aggregate_size,
                           theta, cut_evects, 
                           NULL, transf, all_eigens, fixed_num);
    }
    else
    {
#if SAAMGE_USE_ARPACK
        return SolveIterative(A, B, part, agg_id, aggregate_size,
                              theta, cut_evects, 
                              NULL, transf, all_eigens);
#else
        count_direct_solves++;
        return SolveDirect(A, B, part, agg_id, aggregate_size,
                           theta, cut_evects, 
                           NULL, transf, all_eigens, fixed_num);
#endif
    }
}

/**
   This is the standard eigenvalue solver for multilevel runs.
   It is better to use ARPACK for larger problems, we are still
   experimenting with this and what "larger problems" means and
   how this affects accuracy etc.
*/
bool Eigensolver::SolveDirect(
    const mfem::SparseMatrix& A,
    mfem::SparseMatrix *& B, int part, int agg_id, int agg_size,
    double& theta, mfem::DenseMatrix& cut_evects,
    const mfem::DenseMatrix *Tt,
    bool transf, bool all_eigens, int fixed_num)
{
    DenseMatrix *cut_ptr, cut_helper;
    DenseMatrix deA, deB;
    delete eigenvalues;
    eigenvalues = new Vector;
    Vector &evals = *eigenvalues;
    const double lmax = 1.; // Special choice which is good when the weighted
                            // l1-smoother is used
    const int cut_evects_num_beg = cut_evects.Width(); // Number of vectors in
                                                       // old basis (w/o xbad)
    bool vector_added = false;
    double skipped = 0.;

    SA_ASSERT(A.Width() == A.Size());

    SA_ASSERT(SA_REAL_ALMOST_LE(theta, lmax));
    SA_ASSERT(theta >= 0.);
    // Build the weighted l1-smoother for the eigenvalue problem if not given.
    if (!B) 
        B = mbox_snd_D_sparse_from_sparse(A);

    SA_ASSERT(B->Width() == B->Size());
    SA_ASSERT(B->Width() == A.Width());
    if (transf)
    {
        // Transform the matrices for the eigenproblem.
        SA_ASSERT(Tt);
        DenseMatrix T(*Tt, 't');
        mbox_transform_sparse(A, *Tt, deA);
        mbox_transform_diag(T, *B, deB);
        cut_ptr = &cut_helper;
    } 
    else
    {
        // Take the matrices without transforming them.
        mbox_convert_sparse_to_dense(A, deA);
        mbox_convert_sparse_to_dense(*B, deB);
        cut_ptr = &cut_evects;
    }
    SA_ASSERT(deA.Width() == deA.Height());
    SA_ASSERT(deB.Width() == deB.Height());
    SA_ASSERT(deB.Width() == deA.Width());
    SA_ASSERT((transf && &cut_helper == cut_ptr) ||
              (!transf && &cut_evects == cut_ptr));

    if (all_eigens)
    {
        DenseMatrix evects;

        // Solve the local eigenvalue problem computing all eigenvalues and
        // eigenvectors
        xpacks_calc_all_gen_eigens_dense(deA, evals, evects, deB);
#if (SA_IS_DEBUG_LEVEL(12))
        helpers_write_vector_for_gnuplot(part, evals);
#endif

        // Take only the eigenvectors with eigenvalue <= theta * lmax
        // Store them in *cut_ptr
        skipped = xpack_cut_evects_small(evals, evects, theta * evals(evals.Size() - 1), *cut_ptr);
        SA_PRINTF_L(9, "skipped = %g, largest: %g\n", skipped,
                    evals(evals.Size() - 1));
        SA_ASSERT(SA_REAL_ALMOST_LE(skipped, evals(evals.Size() - 1)));
        SA_ASSERT(SA_REAL_ALMOST_LE(0., skipped));
    } 
    else
    {
        // Solve the local eigenvalue problem computing the necessary
        // eigenvalues and eigenvectors
        xpacks_calc_lower_eigens_dense(deA, evals, *cut_ptr, deB, theta * lmax,
                                       true, fixed_num);
    }
    if (SA_IS_OUTPUT_LEVEL(9))
    {
        SA_PRINTF("theta * lmax: %g\n", theta * lmax);
        SA_PRINTF("total eigens: %d, taken: %d\n", deA.Size(),
                  cut_ptr->Width());
        // SA_PRINTF("%s","evalues: ");
        // for (int j=0; j<cut_ptr->Width(); ++j)
        //    SA_PRINTF("%e  ",evals[j]);
        // SA_PRINTF("%s","\n");
    }

    // If width of matrix *cut_ptr is greater than the original number
    // of vects, then throw vector_added flag
    vector_added = (cut_evects_num_beg < cut_ptr->Width());
    SA_PRINTF_L(9, "cut_evects before: %d, after: %d, added (true/false): %d\n",
                cut_evects_num_beg, cut_ptr->Width(), vector_added);
    SA_ALERT_COND_MSG(cut_evects_num_beg <= cut_ptr->Width(),
                      "Dimension decreased for %d!", part);
    SA_ASSERT(vector_added || transf);

    if (transf)
    {
        // Transform back the computed eigenvectors
        SA_PRINTF_L(9, "%s", "Transforming back eigenvectors...\n");
        SA_ASSERT(Tt);
        SA_ASSERT(&cut_helper == cut_ptr);
        mbox_transform_vects(*Tt, cut_helper, cut_evects);
    }

    // Return the skipped eigenvalue using the argument (variable) theta
    if (all_eigens)
        theta = skipped;
    SA_ASSERT(SA_REAL_ALMOST_LE(theta, lmax));
    SA_ASSERT(SA_REAL_ALMOST_LE(0., theta));
    if (theta < 0.)
        theta = 0.;

    return vector_added;
}

#if SAAMGE_USE_ARPACK
bool Eigensolver::SolveIterative(
    const mfem::SparseMatrix& A,
    mfem::SparseMatrix *& B, int part, int agg_id, int agg_size,
    double& theta, mfem::DenseMatrix& cut_evects,
    const mfem::DenseMatrix *Tt,
    bool transf, bool all_eigens)
{
    SA_ASSERT(!transf); // not implemented, but possible if you want
    SA_ASSERT(!all_eigens); // not implemented, and difficult with ARPACK

    DenseMatrix *cut_ptr, cut_helper;
    delete eigenvalues;
    eigenvalues = new Vector;
    Vector &evals = *eigenvalues;
    const double lmax = 1.; // Special choice which is good when the weighted
                            // l1-smoother is used
    const int cut_evects_num_beg = cut_evects.Width(); // Number of vectors in
                                                       // old basis (w/o xbad)
    bool vector_added = false;

    SA_ASSERT(A.Width() == A.Size());

    SA_ASSERT(SA_REAL_ALMOST_LE(theta, lmax));
    SA_ASSERT(theta >= 0.);
    // Build the weighted l1-smoother for the eigenvalue problem if not given.
    if(!B) B = mbox_snd_D_sparse_from_sparse(A);

    SA_ASSERT(B->Width() == B->Size());
    SA_ASSERT(B->Width() == A.Width());

    // Solve the local eigenvalue problem computing the necessary
    // eigenvalues and eigenvectors

    int min_vectors = 1;
    double arpack_tol = 1.e-4;
    int max_arpack_its = 200;
    int num_arnoldi = (A.Width() < 4*max_arpack_vectors) ? A.Width() : 4*max_arpack_vectors;
    if (A.Width() < max_arpack_vectors) max_arpack_vectors = A.Width();
    cut_ptr = &cut_helper;
    int numvectors = arpacks_calc_portion_eigens_sparse_diag(
        A, evals, *cut_ptr, *B, max_arpack_vectors, true,
        max_arpack_its, num_arnoldi, arpack_tol);
    int vectors_got = min_vectors;
    for (int ev=min_vectors; ev<max_arpack_vectors; ++ev)
    {
        if (evals[ev] < theta)
            vectors_got++;
    }
    if (vectors_got == max_arpack_vectors)
    {
        count_max_used++;
    }
    else
    {
        // this may not be quite the right eigenvalue, what with smoothers / preconditioners / transformations...
        double eigenvalue_skipped = evals[vectors_got];
        smallest_eigenvalue_skipped = 
            (eigenvalue_skipped < smallest_eigenvalue_skipped) ? eigenvalue_skipped : smallest_eigenvalue_skipped;
    }   
    cut_evects.SetSize(A.Height(), vectors_got);
    memcpy(cut_evects.Data(), cut_ptr->Data(), sizeof(double) * A.Height() * vectors_got);

    if (SA_IS_OUTPUT_LEVEL(9))
    {
        SA_PRINTF("theta * lmax: %g\n", theta * lmax);
        SA_PRINTF("system size: %d, eigens computed: %d, eigens taken: %d\n",
                  A.Height(), numvectors, vectors_got);
    }

    // If width of matrix *cut_ptr is greater than the original number
    // of vects, then throw vector_added flag
    vector_added = (cut_evects_num_beg < cut_ptr->Width());
    SA_PRINTF_L(9, "cut_evects before: %d, after: %d, added (true/false): %d\n",
                cut_evects_num_beg, cut_ptr->Width(), vector_added);
    SA_ALERT_COND_MSG(cut_evects_num_beg <= cut_ptr->Width(),
                      "Dimension decreased for %d!", part);
    SA_ASSERT(vector_added || transf);

    SA_ASSERT(SA_REAL_ALMOST_LE(theta, lmax));
    SA_ASSERT(SA_REAL_ALMOST_LE(0., theta));
    if (theta < 0.)
        theta = 0.;

    return vector_added;
}
#endif

void spect_schur_augment_transf(
    const DenseMatrix& Tt, int part, int agg_id,
    int agg_size, const int *aggregates, const Table& AE_to_dof,
    DenseMatrix& augTt)
{
    const int h = Tt.Height();
    const int w = Tt.Width();
    const int AE_minus_agg = h - agg_size;
    const int W = w + AE_minus_agg;
    const int * const AE_dofs = AE_to_dof.GetRow(part);
    int i, j;
    SA_ASSERT(AE_to_dof.RowSize(part) == h);
    SA_ASSERT(W <= h);

    augTt.SetSize(h, W);

    const double *ptr = Tt.Data();
    double *aptr = augTt.Data();

    // Restrict the base vectors to the aggregate.
    for (i=0; i < w; ++i)
    {
        for (j=0; j < h; (++j), (++ptr), (++aptr))
        {
            if (aggregates[AE_dofs[j]] == agg_id)
                *aptr = *ptr;
            else
                *aptr = 0.;
        }
    }
    SA_ASSERT(Tt.Data() + h*w == ptr);
    SA_ASSERT(augTt.Data() + h*w == aptr);

    // Take identity outside the aggregate.
    SA_ASSERT(w == i);
    for (int k=0; k < AE_minus_agg; ++k)
    {
        int row = 0;
#ifdef SA_ASSERTS
        int p = 0;
#endif
        SA_ASSERT(i < W);
        SA_ASSERT(i - w == k);
        for (j=0; j < h; (++j), (++aptr))
        {
            if (aggregates[AE_dofs[j]] != agg_id)
            {
                SA_ASSERT(row < h - agg_size);
                if (k == row)
                {
                    *aptr = 1.;
#ifdef SA_ASSERTS
                    ++p;
#endif
                } else
                    *aptr = 0.;
                ++row;
            } else
                *aptr = 0.;
        }
        SA_ASSERT(AE_minus_agg == row);
        SA_ASSERT(1 == p);
#ifdef SA_ASSERTS
        ++i;
#endif
    }
    SA_ASSERT(W == i);
    SA_ASSERT(augTt.Data() + h*W == aptr);
}

/**
   Not normally used, was used for two-level SAAMGe when
   number of aggregates == number of agglomerates but 
   agglomerates overlapped and aggregates didn't; now we
   use MISes instead, generally.

   (In principle you can still do a Schur-like thing with
   MISes but it gets pretty complicated and we have not gone 
   there.)
*/
bool spect_schur_local_prob_solve_sparse(
    const SparseMatrix& A,
    SparseMatrix *& B, int part, int agg_id, int agg_size,
    const int *aggregates,
    const agg_partitioning_relations_t& agg_part_rels,
    double& theta, DenseMatrix& cut_evects,
    const DenseMatrix *Tt,
    bool transf, bool all_eigens)
{
    DenseMatrix *cut_ptr, cut_helper;
    DenseMatrix deA, deB, augTt;
    Vector evals;
    const double lmax = 1.; // Special choice which is good when the weighted
                            // l1-smoother is used
    const int cut_evects_num_beg = cut_evects.Width(); // Number of vectors in
                                                       // old basis (w/o xbad)
    bool vector_added = false;
    double skipped = 0.;
    const double *sqnorms;
    int meaningful_size = 0;

    SA_ASSERT(A.Width() == A.Size());

    SA_ASSERT(SA_REAL_ALMOST_LE(theta, lmax));
    SA_ASSERT(theta >= 0.);

    // Compute the lower bound for sigma (the eigenvalues of the modified
    // problem). They are always not larger than 1.
    const double bound = 1. / (1. + lmax * theta);

    // Build the restricted weighted l1-smoother for the eigenvalue problem if
    // not given.
    if (!B)
    {
        B = mbox_restr_snd_D_sparse_from_sparse(A, aggregates,
                *agg_part_rels.AE_to_dof, agg_id, part, agg_size);
    }
    SA_ASSERT(B->Width() == B->Size());
    SA_ASSERT(B->Width() == A.Width());

    SparseMatrix *newA = mbox_add_diag_to_sparse(*B, A);

    if (transf)
    {
        // Transform the matrices for the eigenproblem.
        SA_ASSERT(Tt);
        meaningful_size = Tt->Width();
        spect_schur_augment_transf(*Tt, part, agg_id, agg_size, aggregates,
                                   *agg_part_rels.AE_to_dof, augTt);
        Tt = &augTt;
        DenseMatrix T(*Tt, 't');
        mbox_transform_sparse(*newA, *Tt, deA);
        mbox_transform_diag(T, *B, deB);
        cut_ptr = &cut_helper;
    } else
    {
        // Take the matrices without transforming them.
        mbox_convert_sparse_to_dense(*newA, deA);
        mbox_convert_sparse_to_dense(*B, deB);
        cut_ptr = &cut_evects;
        meaningful_size = agg_size;
    }
    delete newA;
    SA_ASSERT(deA.Width() == deA.Height());
    SA_ASSERT(deB.Width() == deB.Height());
    SA_ASSERT(deB.Width() == deA.Width());
    SA_ASSERT((transf && &cut_helper == cut_ptr) ||
              (!transf && &cut_evects == cut_ptr));
    SA_PRINTF_L(9, "Meaningful number of vectors: %d\n", meaningful_size);
    SA_ASSERT(0 < meaningful_size);

    if (all_eigens)
    {
        DenseMatrix evects;

        // Solve the local eigenvalue problem computing all eigenvalues and
        // eigenvectors
        xpacks_calc_all_gen_eigens_dense(deB, evals, evects, deA);
#if (SA_IS_DEBUG_LEVEL(12))
        helpers_write_vector_for_gnuplot(part, evals);
#endif

        // Take only the eigenvectors with eigenvalue >= bound.
        // Store them in *cut_ptr
        skipped = xpack_cut_evects_large(evals, evects, bound, *cut_ptr);
        SA_PRINTF_L(9, "skipped = %g, smallest: %g\n", skipped, evals(0));

        SA_ASSERT(evals.Size() >= meaningful_size);
        SA_ASSERT(cut_ptr->Width() <= meaningful_size);
        // Fix this skipped value so it is meaningful.
        if (cut_ptr->Width() == meaningful_size)
            skipped = evals(evals.Size() - meaningful_size);
        SA_PRINTF_L(9, "meaningful skipped: %g, meaningful smallest: %g\n",
                    skipped, evals(evals.Size() - meaningful_size));

        SA_ASSERT(SA_REAL_ALMOST_LE(skipped, 1.));
        SA_ASSERT(SA_REAL_ALMOST_LE(0., skipped));

        // Get the beginning of the eigenvalues for the taken vectors.
        SA_ASSERT(evals.Size() - cut_ptr->Width() >= 0);
        sqnorms = evals.GetData() + (evals.Size() - cut_ptr->Width());
    } else
    {
        // Solve the local eigenvalue problem computing the necessary
        // eigenvalues and eigenvectors
        xpacks_calc_upper_eigens_dense(deB, evals, *cut_ptr, deA, bound, true);
        SA_ASSERT(cut_ptr->Width() <= meaningful_size);

        // Get the beginning of the eigenvalues for the taken vectors.
        sqnorms = evals.GetData();
    }
    if (SA_IS_OUTPUT_LEVEL(9))
    {
        PROC_STR_STREAM << "Actual Evals = [ ";
        for (int i=1; i <= meaningful_size && i <= evals.Size(); ++i)
            PROC_STR_STREAM << 1./evals(evals.Size() - i) - 1. << " ";
        PROC_STR_STREAM << "]\n";
        SA_PRINTF("%s", PROC_STR_STREAM.str().c_str());
        PROC_CLEAR_STR_STREAM;
    }

    // Normalize the taken vectors.
    mbox_sqnormalize_vects(*cut_ptr, sqnorms);

    if (SA_IS_OUTPUT_LEVEL(9))
    {
        SA_PRINTF("theta * lmax: %g, bound: %g\n", theta * lmax, bound);
        SA_PRINTF("total eigens: %d, taken: %d\n", deA.Size(),
                  cut_ptr->Width());
    }

    // If width of matrix *cut_ptr is greater than the original number
    // of vects, then throw vector_added flag
    vector_added = (cut_evects_num_beg < cut_ptr->Width());
    SA_PRINTF_L(9, "cut_evects before: %d, after: %d, added (true/false): %d\n",
                cut_evects_num_beg, cut_ptr->Width(), vector_added);
    SA_ALERT_COND_MSG(cut_evects_num_beg <= cut_ptr->Width(),
                      "Dimension decreased for %d!", part);
    SA_ASSERT(vector_added || transf);

    if (transf)
    {
        // Transform back the computed eigenvectors
        SA_PRINTF_L(9, "%s", "Transforming back eigenvectors...\n");
        SA_ASSERT(Tt);
        SA_ASSERT(&cut_helper == cut_ptr);
        mbox_transform_vects(*Tt, cut_helper, cut_evects);
    }

    // Return the skipped eigenvalue using the argument (variable) theta
    if (all_eigens)
    {
        theta = 1. / skipped - 1.;
        SA_PRINTF_L(9, "Locally suggested theta: %g\n", theta);
    }
    SA_ASSERT(SA_REAL_ALMOST_LE(theta, lmax));
    SA_ASSERT(SA_REAL_ALMOST_LE(0.,theta));
    if (theta < 0.)
        theta = 0.;

    return vector_added;
}

} // namespace saamge
