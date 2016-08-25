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
#include "xpacks.hpp"
#include <mfem.hpp>
extern "C" {
#include <f2c.h>
#include <clapack.h>
}

/* Functions */

void xpacks_calc_spd_inverse_dense(const DenseMatrix& Ain, DenseMatrix& invA)
{
    char uplo = 'U';
    integer n = Ain.Height();
    integer lda = n;
    integer info;
    SA_ASSERT(Ain.Height() == Ain.Width());
    invA.SetSize(n);
    SA_ASSERT(invA.Width() == invA.Height());
    SA_ASSERT(invA.Width() == Ain.Width());
    doublereal *A = invA.Data();

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
    integer n = Ain.Height();
    integer lda = n;
    integer lwork = 3*n;
//    doublereal *work = new doublereal[lwork];
    integer info;
    doublereal *A;
    doublereal *w;

    SA_ASSERT(n > 0);

    evals.SetSize(n);
    w = (doublereal *)evals.GetData();
    evects.SetSize(n);
    A = (doublereal *)evects.Data();

    SA_ASSERT(Ain.Height() == Ain.Width());
    SA_ASSERT(evects.Height() == evects.Width());
    SA_ASSERT(evects.Height() == Ain.Height());

    memcpy(A, Ain.Data(), sizeof(*A) * n * n);

    lwork = -1;
    doublereal qwork;
    dsyev_(&jobz, &uplo, &n, A, &lda, w, &qwork, &lwork, &info);
    SA_ASSERT(!info);
    lwork = (integer)qwork + 1;
    doublereal *work = new doublereal[lwork];

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
    integer itype = 1;
    char jobz = 'V';
    char uplo = 'U';
    integer n = Ain.Height();
    integer lda = n;
    integer ldb = n;
    integer lwork = 3*n;
//    doublereal *work = new doublereal[lwork];
    integer info;
    doublereal *A;
    doublereal *B = new doublereal[n*n];
    doublereal *w;

    SA_ASSERT(n > 0);

    evals.SetSize(n);
    w = (doublereal *)evals.GetData();
    evects.SetSize(n);
    A = (doublereal *)evects.Data();

    SA_ASSERT(Ain.Height() == Ain.Width());
    SA_ASSERT(evects.Height() == evects.Width());
    SA_ASSERT(Bin.Height() == Bin.Width());
    SA_ASSERT(evects.Height() == Ain.Height());
    SA_ASSERT(Bin.Height() == Ain.Height());

    memcpy(B, Bin.Data(), sizeof(*B) * n * n);
    memcpy(A, Ain.Data(), sizeof(*A) * n * n);

    lwork = -1;
    doublereal qwork;
    dsygv_(&itype, &jobz, &uplo, &n, A, &lda, B, &ldb, w, &qwork, &lwork,
           &info);
    SA_ASSERT(!info);
    lwork = (integer)qwork + 1;
    doublereal *work = new doublereal[lwork];

    dsygv_(&itype, &jobz, &uplo, &n, A, &lda, B, &ldb, w, work, &lwork, &info);
    SA_ASSERT(!info);

    delete [] B;
    delete [] work;
}

int xpacks_calc_lower_eigens_dense(const DenseMatrix& Ain, Vector& evals,
                                   DenseMatrix& evects, const DenseMatrix& Bin,
                                   double upper, bool atleast_one)
{
    integer itype = 1;
    char jobz = 'V';
    char range = 'V';
    char uplo = 'U';
    integer n = Ain.Height();
    integer lda = n;
    integer ldb = n;
    doublereal vl = -1.;
    doublereal vu = upper;
    char cmach = 'S';
    doublereal abstol = 2. * dlamch_(&cmach);
    integer m;
    integer ldz = n;
    integer lwork = 8*n;
//    doublereal *work = new doublereal[lwork];
    integer *iwork = new integer[5*n];
    integer *ifail = new integer[n];
    integer info;
    doublereal *A = new doublereal[n*n];
    doublereal *B = new doublereal[n*n];
    doublereal *w = new doublereal[n];
    doublereal *z = new doublereal[n*n];

    SA_ASSERT(n > 0);

    SA_ASSERT(Ain.Height() == Ain.Width());
    SA_ASSERT(Bin.Height() == Bin.Width());
    SA_ASSERT(Bin.Height() == Ain.Height());

    memcpy(B, Bin.Data(), sizeof(*B) * n * n);
    memcpy(A, Ain.Data(), sizeof(*A) * n * n);

    lwork = -1;
    doublereal qwork;
    dsygvx_(&itype, &jobz, &range, &uplo, &n, A, &lda, B, &ldb, &vl, &vu, NULL,
            NULL, &abstol, &m, w, z, &ldz, &qwork, &lwork, iwork, ifail, &info);
    SA_ASSERT(!info);
    lwork = (integer)qwork + 1;
    doublereal *work = new doublereal[lwork];

    dsygvx_(&itype, &jobz, &range, &uplo, &n, A, &lda, B, &ldb, &vl, &vu, NULL,
            NULL, &abstol, &m, w, z, &ldz, work, &lwork, iwork, ifail, &info);
    SA_ASSERT(!info);

    if (atleast_one && 0 >= m)
    {
#if (SA_IS_DEBUG_LEVEL(6))
        SA_ALERT(0 < m);
#endif
        // Far from good
        integer il = 1;
        integer iu = 1;
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
    integer itype = 1;
    char jobz = 'V';
    char range = 'V';
    char uplo = 'U';
    integer n = Ain.Height();
    integer lda = n;
    integer ldb = n;
    doublereal vl = lower;
    doublereal vu = 2.;
    char cmach = 'S';
    doublereal abstol = 2. * dlamch_(&cmach);
    integer m;
    integer ldz = n;
    integer lwork = 8*n;
//    doublereal *work = new doublereal[lwork];
    integer *iwork = new integer[5*n];
    integer *ifail = new integer[n];
    integer info;
    doublereal *A = new doublereal[n*n];
    doublereal *B = new doublereal[n*n];
    doublereal *w = new doublereal[n];
    doublereal *z = new doublereal[n*n];

    SA_ASSERT(n > 0);

    SA_ASSERT(Ain.Height() == Ain.Width());
    SA_ASSERT(Bin.Height() == Bin.Width());
    SA_ASSERT(Bin.Height() == Ain.Height());

    memcpy(B, Bin.Data(), sizeof(*B) * n * n);
    memcpy(A, Ain.Data(), sizeof(*A) * n * n);

    lwork = -1;
    doublereal qwork;
    dsygvx_(&itype, &jobz, &range, &uplo, &n, A, &lda, B, &ldb, &vl, &vu, NULL,
            NULL, &abstol, &m, w, z, &ldz, &qwork, &lwork, iwork, ifail, &info);
    SA_ASSERT(!info);
    lwork = (integer)qwork + 1;
    doublereal *work = new doublereal[lwork];

    dsygvx_(&itype, &jobz, &range, &uplo, &n, A, &lda, B, &ldb, &vl, &vu, NULL,
            NULL, &abstol, &m, w, z, &ldz, work, &lwork, iwork, ifail, &info);
    SA_ASSERT(!info);

    if (atleast_one && 0 >= m)
    {
#if (SA_IS_DEBUG_LEVEL(6))
        SA_ALERT(0 < m);
#endif
        // Far from good
        integer il = n;
        integer iu = n;
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
    int i;
    char jobu = 'S';
    char jobvt = 'N';
    integer m = arr[0].Height();
    integer n = arr[0].Width();

    for (i=1; i < arr_size; ++i)
    {
        SA_ASSERT(arr[i].Height() == m);
        n += arr[i].Width();
    }

    integer lda = m;
    integer ldu = m;
    integer ldvt = n;
    doublereal *vt = NULL;
    integer lwork = -1;
    integer info;
    doublereal qwork;
    doublereal *a = new doublereal[m*n];
    doublereal *s;
    doublereal *u;
    const int minimal = min(m, n);

    if (SA_IS_OUTPUT_LEVEL(9))
    {
        SA_PRINTF("array size: %d\n", arr_size);
        SA_PRINTF("number of singulars: %d\n", minimal);
    }

    SA_ASSERT(minimal > 0);

    doublereal *ptr = a;
    for (i=0; i < arr_size; ++i)
    {
        SA_ASSERT(a + m*n > ptr);
        const int sz = arr[i].Width() * m;
        SA_ASSERT(a + m*n >= ptr + sz);
        memcpy(ptr, arr[i].Data(), sizeof(*ptr)*sz);
        ptr += sz;
    }
    SA_ASSERT(a + m*n == ptr);

    if (SA_IS_OUTPUT_LEVEL(9))
        PROC_STR_STREAM << "Norms = [ ";
    Vector vect(NULL, m);
    for ((i=0), (ptr=a); i < n; (++i), (ptr += m))
    {
        vect.SetData(ptr);
        const double norm = vect.Norml2();
        if (SA_IS_OUTPUT_LEVEL(9))
            PROC_STR_STREAM << norm << " ";
        SA_ALERT(!SA_REAL_ALMOST_LE(norm, 0.));
        vect /= norm;
    }
    if (SA_IS_OUTPUT_LEVEL(9))
    {
        PROC_STR_STREAM << "]\n";
        SA_PRINTF("%s", PROC_STR_STREAM.str().c_str());
        PROC_CLEAR_STR_STREAM;
    }

    svals.SetSize(minimal);
    s = (doublereal *)svals.GetData();
    lsvects.SetSize(m, minimal);
    u = lsvects.Data();

    dgesvd_(&jobu, &jobvt, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, &qwork,
            &lwork, &info);
    SA_ASSERT(!info);

    lwork = (integer)qwork + 1;
    SA_ASSERT(lwork >= 1);
    if (lwork < max(3 * minimal + max(m, n), 5 * minimal))
        lwork = max(3 * minimal + max(m, n), 5 * minimal);
    doublereal *work = new doublereal[lwork];

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

void xpack_solve_spd_Cholesky(const DenseMatrix& A, const Vector &rhs,
                              Vector &x)
{
    char uplo = 'U';
    integer n = A.Height();
    integer nrhs = 1;
    doublereal *a = A.Data();
    integer lda = n;
    doublereal *b = new doublereal[n];
    integer ldb = n;
    integer info;

    SA_ASSERT(A.Width() == A.Height());
    SA_ASSERT(rhs.Size() == n);

    memcpy(b, rhs.GetData(), sizeof(*b)*n);

    dposv_(&uplo, &n, &nrhs, a, &lda, b, &ldb, &info);
    SA_ASSERT(!info);

    x.Destroy();
    x.SetDataAndSize(b, n);
    x.MakeDataOwner();
}
