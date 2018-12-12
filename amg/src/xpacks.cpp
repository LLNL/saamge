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
#include "xpacks.hpp"
#include <mfem.hpp>
extern "C"
{
    int dpotrf_(char *uplo, int *n, double *a, int *
                lda, int *info);

    int dpotri_(char *uplo, int *n, double *a, int *
                lda, int *info);

    int dsyev_(char *jobz, char *uplo, int *n, double *a, 
               int *lda, double *w, double *work, int *lwork, 
               int *info);

    int dsygv_(int *itype, char *jobz, char *uplo, int *
               n, double *a, int *lda, double *b, int *ldb, 
               double *w, double *work, int *lwork, int *info);

    double dlamch_(char *cmach);

    int dsygvx_(int *itype, char *jobz, char *range, char *
                uplo, int *n, double *a, int *lda, double *b, int 
                *ldb, double *vl, double *vu, int *il, int *iu, 
                double *abstol, int *m, double *w, double *z__, 
                int *ldz, double *work, int *lwork, int *iwork, 
                int *ifail, int *info);

    int dgesvd_(char *jobu, char *jobvt, int *m, int *n, 
                double *a, int *lda, double *s, double *u, int *
                ldu, double *vt, int *ldvt, double *work, int *lwork, 
                int *info);

    int dgels_(char *trans, int *m, int *n, int *
               nrhs, double *a, int *lda, double *b, int *ldb, 
               double *work, int *lwork, int *info);

    int dposv_(char *uplo, int *n, int *nrhs, double 
               *a, int *lda, double *b, int *ldb, int *info);

}

namespace saamge
{
using namespace mfem;

/* Functions */

void xpacks_calc_spd_inverse_dense(const DenseMatrix& Ain, DenseMatrix& invA)
{
    char uplo = 'U';
    int n = Ain.Height();
    int lda = n;
    int info;
    SA_ASSERT(Ain.Height() == Ain.Width());
    invA.SetSize(n);
    SA_ASSERT(invA.Width() == invA.Height());
    SA_ASSERT(invA.Width() == Ain.Width());
    double *A = invA.Data();

    memcpy(A, Ain.Data(), sizeof(*A) * n * n);

    dpotrf_(&uplo, &n, A, &lda, &info);
    SA_ASSERT(!info);

    dpotri_(&uplo, &n, A, &lda, &info);
    SA_ASSERT(!info);

    for (int i=1; i < n; ++i)
        for (int j=0; j < i; ++j)
            invA(i, j) = invA(j, i);
}

void xpacks_calc_all_eigens_dense(const DenseMatrix& Ain, Vector& evals,
                                  DenseMatrix& evects)
{
    char jobz = 'V';
    char uplo = 'U';
    int n = Ain.Height();
    int lda = n;
    int lwork = 3*n;
//    double *work = new double[lwork];
    int info;
    double *A;
    double *w;

    SA_ASSERT(n > 0);

    evals.SetSize(n);
    w = (double *)evals.GetData();
    evects.SetSize(n);
    A = (double *)evects.Data();

    SA_ASSERT(Ain.Height() == Ain.Width());
    SA_ASSERT(evects.Height() == evects.Width());
    SA_ASSERT(evects.Height() == Ain.Height());

    memcpy(A, Ain.Data(), sizeof(*A) * n * n);

    lwork = -1;
    double qwork;
    dsyev_(&jobz, &uplo, &n, A, &lda, w, &qwork, &lwork, &info);
    SA_ASSERT(!info);
    lwork = (int)qwork + 1;
    double *work = new double[lwork];

    dsyev_(&jobz, &uplo, &n, A, &lda, w, work, &lwork, &info);
    SA_ASSERT(!info);

    delete [] work;
}

bool xpacks_fix_spsd_dense(DenseMatrix& A)
{
    DenseMatrix evects;
    Vector evals;
    bool touched = false;

    SA_ASSERT(A.Height() == A.Width());

    xpacks_calc_all_eigens_dense(A, evals, evects);
    SA_ASSERT(A.Width() == evects.Width());
    SA_ASSERT(evects.Height() == evects.Width());
    SA_ASSERT(evals.Size() == evects.Width());

    const int n = evals.Size();
    for (int i=0; i < n; ++i)
    {
        if (evals(i) < 0.)
        {
            evals(i) = 0.;
            touched = true;
        } else
            break;
    }

    if (touched)
        MultADAt(evects, evals, A);

    return touched;
}

void xpacks_calc_all_gen_eigens_dense(const DenseMatrix& Ain, Vector& evals,
                                      DenseMatrix& evects,
                                      const DenseMatrix& Bin)
{
    int itype = 1;
    char jobz = 'V';
    char uplo = 'U';
    int n = Ain.Height();
    int lda = n;
    int ldb = n;
    int lwork = 3*n;
//    double *work = new double[lwork];
    int info;
    double *A;
    double *B = new double[n*n];
    double *w;

    SA_ASSERT(n > 0);

    evals.SetSize(n);
    w = (double *)evals.GetData();
    evects.SetSize(n);
    A = (double *)evects.Data();

    SA_ASSERT(Ain.Height() == Ain.Width());
    SA_ASSERT(evects.Height() == evects.Width());
    SA_ASSERT(Bin.Height() == Bin.Width());
    SA_ASSERT(evects.Height() == Ain.Height());
    SA_ASSERT(Bin.Height() == Ain.Height());

    memcpy(B, Bin.Data(), sizeof(*B) * n * n);
    memcpy(A, Ain.Data(), sizeof(*A) * n * n);

    lwork = -1;
    double qwork;
    dsygv_(&itype, &jobz, &uplo, &n, A, &lda, B, &ldb, w, &qwork, &lwork,
           &info);
    SA_ASSERT(!info);
    lwork = (int)qwork + 1;
    double *work = new double[lwork];

    dsygv_(&itype, &jobz, &uplo, &n, A, &lda, B, &ldb, w, work, &lwork, &info);
    SA_ASSERT(!info);

    delete [] B;
    delete [] work;
}

int xpacks_calc_lower_eigens_dense(const DenseMatrix& Ain, Vector& evals,
                                   DenseMatrix& evects, const DenseMatrix& Bin,
                                   double upper, bool atleast_one, int fixed_num)
{
    int itype = 1;
    char jobz = 'V';
    char range = fixed_num > 0 ? 'I' : 'V'; // could use 'I' here to get a fixed number of eigenvalues, replace NULL, NULL with ints il, iu for which eigenvectors to take
    char uplo = 'U';
    int n = Ain.Height();
    int lda = n;
    int ldb = n;
    double vl = -1.;
    double vu = upper;
    int il = 1;
    int iu = fixed_num;
    char cmach = 'S';
    double abstol = 2. * dlamch_(&cmach);
    int m;
    int ldz = n;
    int lwork = 8*n;
//    double *work = new double[lwork];
    int *iwork = new int[5*n];
    int *ifail = new int[n];
    int info;
    double *A = new double[n*n];
    double *B = new double[n*n];
    double *w = new double[n];
    double *z = new double[n*n];

    SA_ASSERT(n > 0);

    SA_ASSERT(Ain.Height() == Ain.Width());
    SA_ASSERT(Bin.Height() == Bin.Width());
    SA_ASSERT(Bin.Height() == Ain.Height());

    memcpy(B, Bin.Data(), sizeof(*B) * n * n);
    memcpy(A, Ain.Data(), sizeof(*A) * n * n);

    lwork = -1;
    double qwork;
    dsygvx_(&itype, &jobz, &range, &uplo, &n, A, &lda, B, &ldb, &vl, &vu, &il,
            &iu, &abstol, &m, w, z, &ldz, &qwork, &lwork, iwork, ifail, &info);
    SA_ASSERT(!info);
    lwork = (int)qwork + 1;
    double *work = new double[lwork];

    dsygvx_(&itype, &jobz, &range, &uplo, &n, A, &lda, B, &ldb, &vl, &vu, &il,
            &iu, &abstol, &m, w, z, &ldz, work, &lwork, iwork, ifail, &info);
    SA_ASSERT(!info);

    if (atleast_one && 0 >= m)
    {
#if (SA_IS_DEBUG_LEVEL(6))
        SA_ALERT(0 < m);
#endif
        // Far from good
        int il = 1;
        int iu = 1;
        range = 'I';

        memcpy(B, Bin.Data(), sizeof(*B) * n * n);
        memcpy(A, Ain.Data(), sizeof(*A) * n * n);

        dsygvx_(&itype, &jobz, &range, &uplo, &n, A, &lda, B, &ldb, NULL, NULL,
                &il, &iu, &abstol, &m, w, z, &ldz, work, &lwork, iwork, ifail,
                &info);
        SA_ASSERT(!info);
        SA_ASSERT(1 == m);
    }

    evals.SetSize(m);
    memcpy(evals.GetData(), w, sizeof(*w) * m);
    evects.SetSize(n, m);
    memcpy(evects.Data(), z, sizeof(*z) * n * m);

    if (SA_IS_OUTPUT_LEVEL(9))
    {
        PROC_STR_STREAM << "lower_eigens_dense: Evals = [ ";
        for (int i=0; i < evals.Size(); ++i)
            PROC_STR_STREAM << evals(i) << " ";
        PROC_STR_STREAM << "]\n";
        SA_PRINTF("%s", PROC_STR_STREAM.str().c_str());
        PROC_CLEAR_STR_STREAM;
    }

    delete [] z;
    delete [] w;
    delete [] B;
    delete [] A;
    delete [] ifail;
    delete [] iwork;
    delete [] work;

    return m;
}

int xpacks_calc_upper_eigens_dense(const DenseMatrix& Ain, Vector& evals,
                                   DenseMatrix& evects, const DenseMatrix& Bin,
                                   double lower, bool atleast_one)
{
    int itype = 1;
    char jobz = 'V';
    char range = 'V'; // could use 'I' here to get a fixed number of eigenvalues, replace NULL, NULL with ints il, iu for which eigenvectors to take
    char uplo = 'U';
    int n = Ain.Height();
    int lda = n;
    int ldb = n;
    double vl = lower;
    double vu = 2.;
    char cmach = 'S';
    double abstol = 2. * dlamch_(&cmach);
    int m;
    int ldz = n;
    int lwork = 8*n;
//    double *work = new double[lwork];
    int *iwork = new int[5*n];
    int *ifail = new int[n];
    int info;
    double *A = new double[n*n];
    double *B = new double[n*n];
    double *w = new double[n];
    double *z = new double[n*n];

    SA_ASSERT(n > 0);

    SA_ASSERT(Ain.Height() == Ain.Width());
    SA_ASSERT(Bin.Height() == Bin.Width());
    SA_ASSERT(Bin.Height() == Ain.Height());

    memcpy(B, Bin.Data(), sizeof(*B) * n * n);
    memcpy(A, Ain.Data(), sizeof(*A) * n * n);

    lwork = -1;
    double qwork;
    dsygvx_(&itype, &jobz, &range, &uplo, &n, A, &lda, B, &ldb, &vl, &vu, NULL,
            NULL, &abstol, &m, w, z, &ldz, &qwork, &lwork, iwork, ifail, &info);
    SA_ASSERT(!info);
    lwork = (int)qwork + 1;
    double *work = new double[lwork];

    dsygvx_(&itype, &jobz, &range, &uplo, &n, A, &lda, B, &ldb, &vl, &vu, NULL,
            NULL, &abstol, &m, w, z, &ldz, work, &lwork, iwork, ifail, &info);
    SA_ASSERT(!info);

    if (atleast_one && 0 >= m)
    {
#if (SA_IS_DEBUG_LEVEL(6))
        SA_ALERT(0 < m);
#endif
        // Far from good
        int il = n;
        int iu = n;
        range = 'I';

        memcpy(B, Bin.Data(), sizeof(*B) * n * n);
        memcpy(A, Ain.Data(), sizeof(*A) * n * n);

        dsygvx_(&itype, &jobz, &range, &uplo, &n, A, &lda, B, &ldb, NULL, NULL,
                &il, &iu, &abstol, &m, w, z, &ldz, work, &lwork, iwork, ifail,
                &info);
        SA_ASSERT(!info);
        SA_ASSERT(1 == m);
    }

    evals.SetSize(m);
    memcpy(evals.GetData(), w, sizeof(*w) * m);
    evects.SetSize(n, m);
    memcpy(evects.Data(), z, sizeof(*z) * n * m);

    if (SA_IS_OUTPUT_LEVEL(9))
    {
        PROC_STR_STREAM << "upper_eigens_dense: Evals = [ ";
        for (int i=0; i < evals.Size(); ++i)
            PROC_STR_STREAM << evals(i) << " ";
        PROC_STR_STREAM << "]\n";
        SA_PRINTF("%s", PROC_STR_STREAM.str().c_str());
        PROC_CLEAR_STR_STREAM;
    }

    delete [] z;
    delete [] w;
    delete [] B;
    delete [] A;
    delete [] ifail;
    delete [] iwork;
    delete [] work;

    return m;
}

double xpack_cut_evects_small(const Vector& evals, const DenseMatrix& evects,
                              double bound, DenseMatrix& cut_evects)
{
    const int evals_sz = evals.Size();
    const int height = evects.Height();
    int i;
    double skipped = 0.;

    if (SA_IS_OUTPUT_LEVEL(9))
    {
        PROC_STR_STREAM << "cut_evects_small: Evals = [ ";
        for (i=0; i < evals_sz; ++i)
            PROC_STR_STREAM << evals(i) << " ";
        PROC_STR_STREAM << "]\n";
        SA_PRINTF("%s", PROC_STR_STREAM.str().c_str());
        PROC_CLEAR_STR_STREAM;
    }

    for (i=0; i < evals_sz && evals(i) <= bound; ++i);
    SA_ALERT(0 < i);

    if (i < evals_sz)
        skipped = evals(i);
    else
        skipped = evals(evals_sz - 1);

    if (0 >= i) i = 1;

    SA_PRINTF_L(9, "cut_evects cuts: %d, takes: %d, total: %d\n", evals_sz - i,
                i, evals_sz);

    SA_ASSERT(i <= evects.Width() && evects.Width() == evals_sz);

    cut_evects.SetSize(height, i);
    memcpy(cut_evects.Data(), evects.Data(), sizeof(double) * i * height);

    return skipped;
}

double xpack_cut_evects_large(const Vector& evals, const DenseMatrix& evects,
                              double bound, DenseMatrix& cut_evects)
{
    const int evals_sz = evals.Size();
    const int height = evects.Height();
    int i;
    double skipped = 0.;

    if (SA_IS_OUTPUT_LEVEL(9))
    {
        PROC_STR_STREAM << "cut_evects_large: Evals = [ ";
        for (i=0; i < evals_sz; ++i)
            PROC_STR_STREAM << evals(i) << " ";
        PROC_STR_STREAM << "]\n";
        SA_PRINTF("%s", PROC_STR_STREAM.str().c_str());
        PROC_CLEAR_STR_STREAM;
    }

    SA_ASSERT(0 < evals_sz);
    for (i = evals_sz; i > 0 && evals(i-1) >= bound; --i);
    SA_ALERT(evals_sz > i);

    if (0 < i)
        skipped = evals(i-1);
    else
        skipped = evals(0);

    if (evals_sz <= i) i = evals_sz - 1;
    SA_ASSERT(0 <= i);

    const int taken = evals_sz - i;
    SA_PRINTF_L(9, "cut_evects cuts: %d, takes: %d, total: %d\n", i, taken,
                evals_sz);
    SA_ASSERT(0 < taken && taken <= evals_sz);

    SA_ASSERT(taken <= evects.Width() && evects.Width() == evals_sz);
    SA_ASSERT(taken + i == evects.Width());

    cut_evects.SetSize(height, taken);
    memcpy(cut_evects.Data(), evects.Data() + i * height,
           sizeof(double) * taken * height);

    return skipped;
}

void xpack_svd_dense_arr(const DenseMatrix *arr, int arr_size,
                         DenseMatrix& lsvects, Vector& svals)
{
    SA_ASSERT(arr);
    SA_ASSERT(0 < arr_size);
    int i,j;
    char jobu = 'S';
    char jobvt = 'N';
    int m = arr[0].Height();
    int n = arr[0].Width();

    for (i=1; i < arr_size; ++i)
    {
        SA_ASSERT(arr[i].Height() == m);
        n += arr[i].Width();
    }

    int lda = m;
    int ldu = m;
    int ldvt = n;
    double *vt = NULL;
    int lwork = -1;
    int info;
    double qwork;
    double *a = new double[m*n];
    double *s;
    double *u;
    int minimal = std::min(m, n);

    if (SA_IS_OUTPUT_LEVEL(9))
    {
        SA_PRINTF("array size: %d\n", arr_size);
        SA_PRINTF("total vectors: %d\n", n);
        SA_PRINTF("dimension: %d\n", m);
        SA_PRINTF("number of singulars: %d\n", minimal);
    }

    if (minimal <= 0)
        SA_PRINTF("%s","ERROR: empty eigenvalue array!\n");
    SA_ASSERT(minimal > 0);

    double *ptr = a;
    Vector vect(NULL, m);
    if (SA_IS_OUTPUT_LEVEL(9))
        PROC_STR_STREAM << "Norms = [ ";
    for (i=0; i < arr_size; ++i)
    {
        for (j=0; j<arr[i].Width(); ++j)
        {
            vect.SetData(arr[i].Data() + j*m);
            const double norm = vect.Norml2();
            if (SA_IS_OUTPUT_LEVEL(9))
                PROC_STR_STREAM << norm << " ";
            if (SA_REAL_ALMOST_LE(norm, 0.))
            {
                SA_PRINTF_L(6, "      WARNING: zero eigenvector. arr_size = %d, minimal = %d, "
                          "m = %d, n = %d, i = %d, norm = %e\n",
                          arr_size, minimal, m, n, i, norm);
                n = n - 1;
            }
            else
            {
                vect /= norm;
                memcpy(ptr, arr[i].Data() + j*m, sizeof(*ptr)*m);
                ptr += m;
            }
        }
    }
    if (SA_IS_OUTPUT_LEVEL(9))
    {
        PROC_STR_STREAM << "]\n";
        SA_PRINTF("%s", PROC_STR_STREAM.str().c_str());
        PROC_CLEAR_STR_STREAM;
    }
    minimal = std::min(m, n);

    svals.SetSize(minimal);
    s = (double *)svals.GetData();
    lsvects.SetSize(m, minimal);
    u = lsvects.Data();

    dgesvd_(&jobu, &jobvt, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, &qwork,
            &lwork, &info);
    SA_ASSERT(!info);

    lwork = (int)qwork + 1;
    SA_ASSERT(lwork >= 1);
    if (lwork < std::max(3 * minimal + std::max(m, n), 5 * minimal))
        lwork = std::max(3 * minimal + std::max(m, n), 5 * minimal);
    double *work = new double[lwork];

    dgesvd_(&jobu, &jobvt, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work,
            &lwork, &info);
    SA_ASSERT(!info);

    delete [] work;
    delete [] a;
}

void xpack_orth_set(const DenseMatrix& lsvects, const Vector& svals,
                    DenseMatrix& orth_set, double eps)
{
    const int h = lsvects.Height();
    int i;

    if (SA_IS_OUTPUT_LEVEL(9))
    {
        PROC_STR_STREAM << "Singular values = [ ";
        for (i=0; i < svals.Size(); ++i)
            PROC_STR_STREAM << svals(i) << " ";
        PROC_STR_STREAM << "]\n";
        SA_PRINTF("%s", PROC_STR_STREAM.str().c_str());
        PROC_CLEAR_STR_STREAM;
    }

    SA_ASSERT(svals.Size());
    SA_ASSERT(lsvects.Width() == svals.Size());
    eps *= svals(0);
    for (i=0; i < svals.Size() && svals(i) > eps; ++i);

    SA_PRINTF_L(9, "xpack_orth_set cuts: %d, takes: %d, total: %d, eps: %g,"
                   " max sval: %g\n", lsvects.Width() - i, i, lsvects.Width(),
                eps, svals(0));

    SA_ASSERT(i);

    orth_set.SetSize(h, i);
    memcpy(orth_set.Data(), lsvects.Data(), sizeof(double)*i*h);
}

/**
   this routine added ATB 7 October 2014

   to solve linear least squares problem
*/
void xpack_solve_lls(const DenseMatrix& A, const Vector &rhs, Vector &x)
{
    char trans = 'N';
    int m = A.Height();
    int n = A.Width();
    int nrhs = 1;
    double *a = new double[m*n];
    int lda = m;
    double *b = new double[m];
    int ldb = m;
    int info;
    int lwork = 2*(m+n);
    double *work = new double[2*(m+n)];
    
    SA_ASSERT(A.Height() == rhs.Size());
    SA_ASSERT(A.Width() == x.Size());

    memcpy(b, rhs.GetData(), sizeof(*b)*m);
    memcpy(a, A.Data(), sizeof(*a)*m*n);

    dgels_(&trans, &m, &n, &nrhs, a, &lda, b, &ldb, work, &lwork, &info);
    SA_ASSERT(!info);
    
    for (int i=0; i<n; ++i)
        x(i) = b[i];
    delete[] b;
    delete[] work;
    delete[] a;
}

void xpack_solve_spd_Cholesky(const DenseMatrix& A, const Vector &rhs,
                              Vector &x)
{
    char uplo = 'U';
    int n = A.Height();
    int nrhs = 1;
    double *a = A.Data();
    int lda = n;
    double *b = new double[n];
    int ldb = n;
    int info;

    SA_ASSERT(A.Width() == A.Height());
    SA_ASSERT(rhs.Size() == n);

    memcpy(b, rhs.GetData(), sizeof(*b)*n);

    dposv_(&uplo, &n, &nrhs, a, &lda, b, &ldb, &info);
    SA_ASSERT(!info);

    x.Destroy();
    x.SetDataAndSize(b, n);
    x.MakeDataOwner();
}

} // namespace saamge
