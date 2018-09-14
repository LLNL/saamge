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
#include "mbox.hpp"
#include <fstream>
#include <cmath>
#include <seq_mv.h>
#include <_hypre_parcsr_mv.h>
#include <_hypre_parcsr_ls.h>
#include <mfem.hpp>
#include "xpacks.hpp"

namespace saamge
{
using namespace mfem;
using std::fabs;
using std::sqrt;

/* Functions */

void hypre_par_matrix_ownership(
    HypreParMatrix &mat, bool &data, bool &row_starts, bool &col_starts)
{
    hypre_ParCSRMatrix * hmat = (hypre_ParCSRMatrix*) mat;
    data = hmat->owns_data;
    row_starts = hmat->owns_row_starts;
    col_starts = hmat->owns_col_starts;
}

/**
   Copied from Parelag hypreExtension/hypre_CSRFactory.c
*/
hypre_CSRMatrix * hypre_ZerosCSRMatrix( HYPRE_Int nrows, HYPRE_Int ncols)
{
    hypre_CSRMatrix * A = hypre_CSRMatrixCreate( nrows, ncols, 0);
    hypre_CSRMatrixInitialize( A );

    HYPRE_Int * i_A = hypre_CSRMatrixI(A);

    HYPRE_Int i;
    for( i = 0; i < nrows+1; ++i)
        i_A[i] = 0;

    return A;
}

/**
   Copied from Parelag hypreExtension/hypre_CSRFactory.c
*/
hypre_CSRMatrix * hypre_IdentityCSRMatrix( HYPRE_Int nrows)
{
    hypre_CSRMatrix * A = hypre_CSRMatrixCreate( nrows, nrows, nrows);
    hypre_CSRMatrixInitialize( A );

    HYPRE_Int * i_A = hypre_CSRMatrixI(A);
    HYPRE_Int * j_A = hypre_CSRMatrixJ(A);
    double    * a_A = hypre_CSRMatrixData(A);

    HYPRE_Int i;
    for( i = 0; i < nrows+1; ++i)
        i_A[i] = i;

    for( i = 0; i < nrows; ++i)
    {
        j_A[i] = i;
        a_A[i] = 1.;
    }

    return A;
}

/**
   Copied from Parelag hypreExtension/hypre_CSRFactory.c
*/
hypre_ParCSRMatrix * hypre_IdentityParCSRMatrix( 
    MPI_Comm comm, HYPRE_Int global_num_rows, HYPRE_Int * row_starts)
{
    HYPRE_Int num_nonzeros_diag;
    if(HYPRE_AssumedPartitionCheck())
    {
        num_nonzeros_diag = row_starts[1] - row_starts[0];
        // hypre_assert(row_starts[2] == global_num_rows);
    }
    else
    {
        HYPRE_Int pid;
        HYPRE_Int np;

        MPI_Comm_rank(comm, &pid);
        MPI_Comm_size(comm, &np);

        num_nonzeros_diag = row_starts[pid+1] - row_starts[pid];
        hypre_assert(row_starts[np] == global_num_rows);
    }
    hypre_ParCSRMatrix * mat = hypre_ParCSRMatrixCreate ( 
        comm , global_num_rows , global_num_rows ,
        row_starts , row_starts , 0 , num_nonzeros_diag, 0 );

    hypre_ParCSRMatrixSetRowStartsOwner(mat, 0);
    hypre_ParCSRMatrixSetColStartsOwner(mat, 0);

    hypre_ParCSRMatrixColMapOffd(mat) = NULL;

    hypre_CSRMatrixDestroy( hypre_ParCSRMatrixDiag(mat) );
    hypre_CSRMatrixDestroy( hypre_ParCSRMatrixOffd(mat) );

    hypre_ParCSRMatrixDiag(mat) = hypre_IdentityCSRMatrix(num_nonzeros_diag);
    hypre_ParCSRMatrixOffd(mat) = hypre_ZerosCSRMatrix(num_nonzeros_diag, 0);

    hypre_ParCSRMatrixSetDataOwner(mat, 1);

    return mat;
}

/**
   Copied from Parelag hypreExtension/deleteZeros.c

   probably better to use mfem::HypreParMatrix::Threshold (but see MFEM Issue #290)
*/
HYPRE_Int hypre_ParCSRMatrixDeleteZeros( hypre_ParCSRMatrix *A , double tol )
{
    hypre_CSRMatrix * diag =
        hypre_CSRMatrixDeleteZeros( hypre_ParCSRMatrixDiag(A), tol );
    hypre_CSRMatrix * offd =
        hypre_CSRMatrixDeleteZeros( hypre_ParCSRMatrixOffd(A), tol );

    if (diag)
    {
        hypre_CSRMatrixDestroy( hypre_ParCSRMatrixDiag(A) );
        hypre_ParCSRMatrixDiag(A) = diag;
    }

    if (offd)
    {
        hypre_CSRMatrixDestroy( hypre_ParCSRMatrixOffd(A) );
        hypre_ParCSRMatrixOffd(A) = offd;
    }

    if (hypre_ParCSRMatrixCommPkg(A))
    {
        hypre_MatvecCommPkgDestroy(hypre_ParCSRMatrixCommPkg(A));
        hypre_MatvecCommPkgCreate(A);
    }

    if (hypre_ParCSRMatrixCommPkgT(A))
    {
        hypre_MatvecCommPkgDestroy(hypre_ParCSRMatrixCommPkgT(A));
    }

    hypre_ParCSRMatrixSetNumNonzeros(A);

    return 0;
}

double mbox_energy_inner_prod_sparse(const SparseMatrix& A, const Vector& x,
                                     const Vector& y)
{
    SA_ASSERT(A.Width() == A.Size());
    SA_ASSERT(A.Width() == x.Size());
    SA_ASSERT(A.Width() == y.Size());

    return A.InnerProduct(x, y);
}

double mbox_energy_inner_prod_dense(const DenseMatrix& A, const Vector& x,
                                    const Vector& y)
{
    SA_ASSERT(A.Width() == A.Height());
    SA_ASSERT(A.Width() == x.Size());
    SA_ASSERT(A.Width() == y.Size());

    return A.InnerProduct(x, y);
}

double mbox_energy_inner_prod_diag(const Vector& D, const Vector& x,
                                   const Vector& y)
{
    SA_ASSERT(D.Size() == x.Size());
    SA_ASSERT(D.Size() == y.Size());

    double prod = 0.0;
    for (int i=0; i < D.Size(); ++i)
        prod += D(i)*x(i)*y(i);

    return prod;
}

double mbox_gen_rayleigh_quot_sparse(const SparseMatrix& A,
                                     const SparseMatrix& B, const Vector& x)
{
    SA_ASSERT(A.Width() == A.Size());
    SA_ASSERT(B.Width() == B.Size());
    SA_ASSERT(A.Size()  == B.Size());
    SA_ASSERT(A.Width() == x.Size());

    double denom = B.InnerProduct(x, x);
    SA_ASSERT(denom > 0.);
    double num = A.InnerProduct(x, x);

    return (num / denom);
}

double mbox_gen_rayleigh_quot_dense(const DenseMatrix& A, const DenseMatrix& B,
                                    const Vector& x)
{
    double denom = mbox_energy_inner_prod_dense(B, x, x);
    SA_ASSERT(denom > 0.);
    double num = mbox_energy_inner_prod_dense(A, x, x);

    return (num / denom);
}

double mbox_energy_norm_sparse(const SparseMatrix& A, const Vector& x)
{
    double innerprod = mbox_energy_inner_prod_sparse(A, x, x);
    SA_ASSERT(innerprod >= 0.);
    return sqrt(innerprod);
}

double mbox_energy_norm_dense(const DenseMatrix& A, const Vector& x)
{
    double innerprod = mbox_energy_inner_prod_dense(A, x, x);
    SA_ASSERT(innerprod >= 0.);
    return sqrt(innerprod);
}

Table *mbox_copy_table(const Table *src)
{
    if (!src) return NULL;
    const int sz = src->Size();
    const int non_zero = src->Size_of_connections();
    int *I = new int[sz+1];
    int *J = new int[non_zero];

    memcpy(I, src->GetI(), sizeof(*I)*(sz+1));
    memcpy(J, src->GetJ(), sizeof(*J)*non_zero);
    SA_ASSERT(non_zero == I[sz]);

    Table *dst = new Table();
    dst->SetIJ(I, J, sz);
    return dst;
}

void mbox_free_matr_arr(Matrix **arr, int n)
{
    if (!arr) return;
    SA_ASSERT(n >= 0);
    for (int i=0; i < n; ++i)
        delete arr[i];
    delete [] arr;
}

SparseMatrix *mbox_copy_sparse_matr(const SparseMatrix *src)
{
    if (!src) return NULL;
    SA_ASSERT(const_cast<SparseMatrix *>(src)->Finalized());
    const int sz = src->Size();
    const int non_zero = src->NumNonZeroElems();
    int *I = new int[sz+1];
    int *J = new int[non_zero];
    double *Data = new double[non_zero];

    memcpy(I, src->GetI(), sizeof(*I)*(sz+1));
    memcpy(J, src->GetJ(), sizeof(*J)*non_zero);
    memcpy(Data, src->GetData(), sizeof(*Data)*non_zero);
    SA_ASSERT(non_zero == I[sz]);

    return (new SparseMatrix(I, J, Data, sz, src->Width()));
}

SparseMatrix **mbox_copy_sparse_matr_arr(SparseMatrix **src, int n)
{
    if (!src) return NULL;
    SA_ASSERT(n > 0);
    SparseMatrix **dst = new SparseMatrix*[n];
    for (int i=0; i < n; ++i)
        dst[i] = mbox_copy_sparse_matr(src[i]);
    return dst;
}

DenseMatrix **mbox_copy_dense_matr_arr(DenseMatrix **src, int n)
{
    if (!src) return NULL;
    SA_ASSERT(n > 0);
    DenseMatrix **dst = new DenseMatrix*[n];
    for (int i=0; i < n; ++i)
        dst[i] = new DenseMatrix(*(src[i]));
    return dst;
}

Matrix **mbox_copy_matr_arr(Matrix **src, int n)
{
    if (!src) return NULL;
    SA_ASSERT(n > 0);
    if (dynamic_cast<DenseMatrix *>(src[0]))
        return (Matrix **)mbox_copy_dense_matr_arr((DenseMatrix **)src, n);
    return (Matrix **)mbox_copy_sparse_matr_arr((SparseMatrix **)src, n);
}

Table *mbox_read_table(const char *filename)
{
    int size, i_size, j_size, *I, *J;
    std::ifstream itbl(filename, std::ifstream::binary);
    SA_ASSERT(itbl);
    itbl.read((char *)&size, sizeof(size));
    itbl.read((char *)&j_size, sizeof(j_size));
    i_size = size + 1;
    I = new int[i_size];
    J = new int[j_size];
    itbl.read((char *)I, sizeof(*I) * i_size);
    itbl.read((char *)J, sizeof(*J) * j_size);
    itbl.close();
    SA_ASSERT(j_size == I[size]);
    Table *tbl = new Table;
    tbl->SetIJ(I, J, size);
    return tbl;
}

void mbox_write_table(const char *filename, const Table& tbl)
{
    int size, i_size, j_size;
    std::ofstream otbl(filename, std::ofstream::binary);
    SA_ASSERT(otbl);
    size = tbl.Size();
    i_size = size + 1;
    j_size = tbl.Size_of_connections();
    otbl.write((char *)&size, sizeof(size));
    otbl.write((char *)&j_size, sizeof(j_size));
    otbl.write((char *)tbl.GetI(), sizeof(*(tbl.GetI())) * i_size);
    otbl.write((char *)tbl.GetJ(), sizeof(*(tbl.GetJ())) * j_size);
    otbl.close();
}

SparseMatrix *mbox_read_sparse_matr(const char *filename)
{
    std::ifstream ispm(filename, std::ifstream::binary);
    SA_ASSERT(ispm);
    SparseMatrix *spm = mbox_read_sparse_matr(ispm);
    ispm.close();
    return spm;
}

SparseMatrix *mbox_read_sparse_matr(std::ifstream& ispm)
{
    int size, width, i_size, j_size, *I, *J;
    double *data;
    SA_ASSERT(ispm);
    ispm.read((char *)&size, sizeof(size));
    ispm.read((char *)&width, sizeof(width));
    ispm.read((char *)&j_size, sizeof(j_size));
    i_size = size + 1;
    I = new int[i_size];
    J = new int[j_size];
    data = new double[j_size];
    ispm.read((char *)I, sizeof(*I) * i_size);
    ispm.read((char *)J, sizeof(*J) * j_size);
    ispm.read((char *)data, sizeof(*data) * j_size);
    SA_ASSERT(j_size == I[size]);
    return (new SparseMatrix(I, J, data, size, width));
}

void mbox_write_sparse_matr(const char *filename, const SparseMatrix& spm)
{
    ofstream ospm(filename, ofstream::binary);
    SA_ASSERT(ospm);
    mbox_write_sparse_matr(ospm, spm);
    ospm.close();
}

void mbox_write_sparse_matr(ofstream& ospm, const SparseMatrix& spm)
{
    int size, width, i_size, j_size;
    SA_ASSERT(ospm);
    SA_ASSERT(const_cast<SparseMatrix&>(spm).Finalized());
    size = spm.Size();
    width = spm.Width();
    i_size = size + 1;
    j_size = spm.NumNonZeroElems();
    ospm.write((char *)&size, sizeof(size));
    ospm.write((char *)&width, sizeof(width));
    ospm.write((char *)&j_size, sizeof(j_size));
    ospm.write((char *)spm.GetI(), sizeof(*(spm.GetI())) * i_size);
    ospm.write((char *)spm.GetJ(), sizeof(*(spm.GetJ())) * j_size);
    ospm.write((char *)spm.GetData(), sizeof(*(spm.GetData())) * j_size);
}

DenseMatrix *mbox_read_dense_matr(const char *filename)
{
    ifstream idem(filename, ifstream::binary);
    SA_ASSERT(idem);
    DenseMatrix *dem = mbox_read_dense_matr(idem);
    idem.close();
    return dem;
}

DenseMatrix *mbox_read_dense_matr(ifstream& idem)
{
    int height, width, data_sz;
    double *data;
    SA_ASSERT(idem);
    idem.read((char *)&height, sizeof(height));
    idem.read((char *)&width, sizeof(width));
    data_sz = height * width;
    data = new double[data_sz];
    idem.read((char *)data, sizeof(*data) * data_sz);
    return (new DenseMatrix(data, height, width));
}

void mbox_write_dense_matr(const char *filename, const DenseMatrix& dem)
{
    ofstream odem(filename, ofstream::binary);
    SA_ASSERT(odem);
    mbox_write_dense_matr(odem, dem);
    odem.close();
}

void mbox_write_dense_matr(ofstream& odem, const DenseMatrix& dem)
{
    int height, width, data_sz;
    SA_ASSERT(odem);
    odem.write((char *)&height, sizeof(height));
    odem.write((char *)&width, sizeof(width));
    data_sz = height * width;
    odem.write((char *)dem.Data(), sizeof(*(dem.Data())) * data_sz);
}

SparseMatrix **mbox_read_sparse_matr_arr(const char *filename, int *n)
{
    ifstream ispm(filename, ifstream::binary);
    SA_ASSERT(ispm);
    ispm.read((char *)n, sizeof(*n));
    SparseMatrix **arr = new SparseMatrix*[*n];
    for (int i=0; i < *n; ++i)
        arr[i] = mbox_read_sparse_matr(ispm);
    return arr;
}

void mbox_write_sparse_matr_arr(const char *filename, SparseMatrix **arr,
                                int n)
{
    SA_ASSERT(arr);
    SA_ASSERT(n > 0);
    ofstream ospm(filename, ofstream::binary);
    SA_ASSERT(ospm);
    ospm.write((char *)&n, sizeof(n));
    for (int i=0; i < n; ++i)
        mbox_write_sparse_matr(ospm, *(arr[i]));
    ospm.close();
}

DenseMatrix **mbox_read_dense_matr_arr(const char *filename, int *n)
{
    ifstream idem(filename, ifstream::binary);
    SA_ASSERT(idem);
    idem.read((char *)n, sizeof(*n));
    DenseMatrix **arr = new DenseMatrix*[*n];
    for (int i=0; i < *n; ++i)
        arr[i] = mbox_read_dense_matr(idem);
    return arr;
}

void mbox_write_dense_matr_arr(const char *filename, DenseMatrix **arr,
                               int n)
{
    SA_ASSERT(arr);
    SA_ASSERT(n > 0);
    ofstream odem(filename, ofstream::binary);
    SA_ASSERT(odem);
    odem.write((char *)&n, sizeof(n));
    for (int i=0; i < n; ++i)
        mbox_write_dense_matr(odem, *(arr[i]));
    odem.close();
}

void mbox_convert_sparse_to_dense(const SparseMatrix& Sp, DenseMatrix& D)
{
    SA_ASSERT(const_cast<SparseMatrix&>(Sp).Finalized());
    const int h = Sp.Size();
    const int w = Sp.Width();
    int *I = Sp.GetI();
    int *J = Sp.GetJ();
    double *Data = Sp.GetData();

    D.SetSize(0);
    D.SetSize(h, w);

    for(int i=0; i < h; ++i)
    {
        for (int j=I[i]; j < I[i+1]; ++j)
        {
            SA_ASSERT(J[j] < w);
            D(i, J[j]) = Data[j];
        }
    }
}

SparseMatrix *mbox_convert_dense_to_sparse(const DenseMatrix& D)
{
    SparseMatrix *sp = new SparseMatrix(D.Height(), D.Width());
    int i;
    Array<int> rows(D.Height());
    Array<int> cols(D.Width());

    for (i=0; i < D.Height(); ++i)
        rows[i] = i;
    for (i=0; i < D.Width(); ++i)
        cols[i] = i;
    sp->SetSubMatrix(rows, cols, D);

    sp->Finalize();
    return sp;
}

double *mbox_give_upper_trian(const DenseMatrix& A)
{
    SA_ASSERT(A.Height() == A.Width());
    const int n = A.Height();
    double *ut = new double[(n*(n + 1)) >> 1];
    int p;

    p = 0;
    for (int i=0; i < n; ++i)
    {
        for (int j=0; j <= i; ++j)
        {
            SA_ASSERT(p < (n*(n + 1)) >> 1);
            ut[p++] = A(j,i);
        }
    }
    SA_ASSERT((n*(n + 1)) >> 1 == p);

    return ut;
}

void mbox_swap_data_dense(DenseMatrix& f, DenseMatrix& s)
{
    int fh = f.Height();
    int fw = f.Width();
    double *fdata = f.Data();

    f.UseExternalData(s.Data(), s.Height(), s.Width());
    s.UseExternalData(fdata, fh, fw);
}

void mbox_add_sparse_to_dense(DenseMatrix& A, const SparseMatrix& Sp)
{
    SA_ASSERT(Sp.Size() == A.Height());
    SA_ASSERT(A.Width() == Sp.Width());
    SA_ASSERT(const_cast<SparseMatrix&>(Sp).Finalized());

    const int n = Sp.Size();
    const int * const I = Sp.GetI();
    const int * const J = Sp.GetJ();
    const double * const Data = Sp.GetData();

    for(int i=0; i < n; ++i)
    {
        for (int j=I[i]; j < I[i+1]; ++j)
        {
            SA_ASSERT(J[j] < A.Width());
            A(i, J[j]) += Data[j];
        }
    }
}

SparseMatrix *mbox_add_diag_to_sparse(const SparseMatrix& D,
                                      const SparseMatrix& Sp)
{
    SA_ASSERT(Sp.Size() == Sp.Width());
    SA_ASSERT(D.Size() == D.Width());
    SA_ASSERT(Sp.Size() == D.Width());

    const int n = Sp.Size();
    SparseMatrix *ret = mbox_copy_sparse_matr(&Sp);
    const int * const I = ret->GetI();
    const int * const J = ret->GetJ();
    double * const Data = ret->GetData();
#ifdef SA_ASSERTS
    int p = 0;
#endif

    for (int i=0; i < n; ++i)
    {
        for (int j = I[i]; j < I[i+1]; ++j)
        {
            if (J[j] == i)
            {
                Data[j] += D(i, i);
#ifdef SA_ASSERTS
                ++p;
#endif
            }
        }
    }
    SA_ASSERT(n == p);

    return ret;
}

void mbox_mult_sparse_to_dense(const SparseMatrix& A, const DenseMatrix& B,
                               DenseMatrix& AB)
{
    const int width = B.Width();
    const int heightout = A.Size();
    const int heightin = B.Height();
    Vector vin, vout;
    double *datain = B.Data(), *dataout;

    SA_ASSERT(A.Width() == B.Height());

    AB.SetSize(heightout, width);
    dataout = AB.Data();
    for (int i=0; i < width; (++i), (datain += heightin),
                             (dataout += heightout))
    {
        vin.SetDataAndSize(datain, heightin);
        vout.SetDataAndSize(dataout, heightout);
        A.Mult(vin, vout);
    }
}

SparseMatrix *mbox_build_Dinv_neg(const SparseMatrix& A)
{
    SA_ASSERT(A.Size() == A.Width() &&
              const_cast<SparseMatrix&>(A).Finalized());
    const int n = A.Size();
    int *I = new int[n+1];
    int *J = new int[n];
    double *Data = new double[n];
    double sum;

    for (int i=0; i < n; ++i)
    {
        int *row, beg;
        double *a, diag;

        J[i] = I[i] = i;

        sum = 0.;
        diag = A(i, i);
        beg = A.GetI()[i];
        row = A.GetJ() + beg;
        a = A.GetData() + beg;
        SA_ASSERT(diag > 0.);
        for (int j=0; j < const_cast<SparseMatrix&>(A).RowSize(i); ++j)
        {
            SA_ASSERT(A(row[j], row[j]) > 0.);
            sum -= fabs(a[j]) * sqrt(diag / A(row[j], row[j]));
        }
        SA_ASSERT(sum < 0.);
        Data[i] = 1. / sum;
    }
    I[n] = n;

    return (new SparseMatrix(I, J, Data, n, n));
}

DenseMatrix *mbox_snd_id_dense_from_dense(const DenseMatrix& A)
{
    const int n = A.Height();
    SA_ASSERT(A.Width() == n);
    DenseMatrix *B = new DenseMatrix;

    B->Diag(1., n);

    return B;
}

DenseMatrix *mbox_snd_id_dense_from_sparse(const SparseMatrix& A)
{
    const int n = A.Size();
    SA_ASSERT(A.Width() == n);
    DenseMatrix *B = new DenseMatrix;

    B->Diag(1., n);

    return B;
}

SparseMatrix *mbox_snd_id_sparse_from_sparse(const SparseMatrix& A)
{
    const int n = A.Size();
    SA_ASSERT(A.Width() == n);
    int *I = new int[n+1];
    int *J = new int[n];
    double *Data = new double[n];

    for (int i=0; i < n; ++i)
    {
        J[i] = I[i] = i;
        Data[i] = 1.;
    }
    I[n] = n;

    return (new SparseMatrix(I, J, Data, n, n));
}

SparseMatrix *mbox_snd_id_sparse_from_dense(const DenseMatrix& A)
{
    const int n = A.Height();
    SA_ASSERT(A.Width() == n);
    int *I = new int[n+1];
    int *J = new int[n];
    double *Data = new double[n];

    for (int i=0; i < n; ++i)
    {
        J[i] = I[i] = i;
        Data[i] = 1.;
    }
    I[n] = n;

    return (new SparseMatrix(I, J, Data, n, n));
}

DenseMatrix *mbox_snd_diagA_dense_from_dense(const DenseMatrix& A)
{
    const int n = A.Height();
    SA_ASSERT(A.Width() == n);
    DenseMatrix *B = new DenseMatrix(n);

    for (int i=0; i < n; ++i)
    {
        (*B)(i,i) = A(i,i);
        SA_ASSERT((*B)(i,i) > 0.);
    }

    return B;
}

DenseMatrix *mbox_snd_diagA_dense_from_sparse(const SparseMatrix& A)
{
    const int n = A.Size();
    SA_ASSERT(A.Width() == n);
    DenseMatrix *B = new DenseMatrix(n);

    for (int i=0; i < n; ++i)
    {
        (*B)(i,i) = A(i,i);
        SA_ASSERT((*B)(i,i) > 0.);
    }

    return B;
}

SparseMatrix *mbox_snd_diagA_sparse_from_sparse(const SparseMatrix& A)
{
    const int n = A.Size();
    SA_ASSERT(A.Width() == n);
    int *I = new int[n+1];
    int *J = new int[n];
    double *Data = new double[n];

    for (int i=0; i < n; ++i)
    {
        J[i] = I[i] = i;
        SA_ASSERT(A(i,i) > 0.);
        Data[i] = A(i,i);
    }
    I[n] = n;

    return (new SparseMatrix(I, J, Data, n, n));
}

SparseMatrix *mbox_snd_diagA_sparse_from_dense(const DenseMatrix& A)
{
    const int n = A.Height();
    SA_ASSERT(A.Width() == n);
    int *I = new int[n+1];
    int *J = new int[n];
    double *Data = new double[n];

    for (int i=0; i < n; ++i)
    {
        J[i] = I[i] = i;
        SA_ASSERT(A(i,i) > 0.);
        Data[i] = A(i,i);
    }
    I[n] = n;

    return (new SparseMatrix(I, J, Data, n, n));
}

DenseMatrix *mbox_snd_diagAinv_dense_from_dense(const DenseMatrix& A)
{
    const int n = A.Height();
    SA_ASSERT(A.Width() == n);
    DenseMatrix *B = new DenseMatrix(n);

    for (int i=0; i < n; ++i)
    {
        (*B)(i,i) = 1. / A(i,i);
        SA_ASSERT((*B)(i,i) > 0.);
    }

    return B;
}

DenseMatrix *mbox_snd_diagAinv_dense_from_sparse(const SparseMatrix& A)
{
    const int n = A.Size();
    SA_ASSERT(A.Width() == n);
    DenseMatrix *B = new DenseMatrix(n);

    for (int i=0; i < n; ++i)
    {
        (*B)(i,i) = 1. / A(i,i);
        SA_ASSERT((*B)(i,i) > 0.);
    }

    return B;
}

SparseMatrix *mbox_snd_diagAinv_sparse_from_sparse(const SparseMatrix& A)
{
    const int n = A.Size();
    SA_ASSERT(A.Width() == n);
    int *I = new int[n+1];
    int *J = new int[n];
    double *Data = new double[n];

    for (int i=0; i < n; ++i)
    {
        J[i] = I[i] = i;
        SA_ASSERT(A(i,i) > 0.);
        Data[i] = 1. / A(i,i);
    }
    I[n] = n;

    return (new SparseMatrix(I, J, Data, n, n));
}

SparseMatrix *mbox_snd_diagAinv_sparse_from_dense(const DenseMatrix& A)
{
    const int n = A.Height();
    SA_ASSERT(A.Width() == n);
    int *I = new int[n+1];
    int *J = new int[n];
    double *Data = new double[n];

    for (int i=0; i < n; ++i)
    {
        J[i] = I[i] = i;
        SA_ASSERT(A(i,i) > 0.);
        Data[i] = 1. / A(i,i);
    }
    I[n] = n;

    return (new SparseMatrix(I, J, Data, n, n));
}

DenseMatrix *mbox_snd_D_dense_from_dense(const DenseMatrix& A)
{
    const int n = A.Height();
    SA_ASSERT(A.Width() == n);
    DenseMatrix *B = new DenseMatrix(n);
    double sum;

    for (int i=0; i < n; ++i)
    {
        double diag = A(i,i);
        SA_ASSERT(diag > 0.);
        sum = 0.;
        for (int j=0; j < n; ++j)
            sum += fabs(A(i,j)) * sqrt(diag / A(j,j));
        (*B)(i,i) = sum;
        SA_ASSERT((*B)(i,i) > 0.);
    }

    return B;
}

DenseMatrix *mbox_snd_D_dense_from_sparse(const SparseMatrix& A)
{
    const int n = A.Size();
    SA_ASSERT(A.Width() == n);
    DenseMatrix *B = new DenseMatrix(n);
    double sum;

    for (int i=0; i < n; ++i)
    {
        int *row, beg;
        double *a, diag;

        diag = A(i, i);
        beg = A.GetI()[i];
        row = A.GetJ() + beg;
        a = A.GetData() + beg;
        SA_ASSERT(diag > 0.);
        sum = 0.;
        for (int j=0; j < const_cast<SparseMatrix&>(A).RowSize(i); ++j)
        {
            SA_ASSERT(A(row[j], row[j]) > 0.);
            sum += fabs(a[j]) * sqrt(diag / A(row[j], row[j]));
        }
        (*B)(i,i) = sum;
        SA_ASSERT((*B)(i,i) > 0.);
    }

    return B;
}

SparseMatrix *mbox_snd_D_sparse_from_sparse(const SparseMatrix& A)
{
    const int n = A.Size();
    SA_ASSERT(A.Width() == n);
    int *I = new int[n+1];
    int *J = new int[n];
    double *Data = new double[n];
    double sum;

    for (int i=0; i < n; ++i)
    {
        int *row, beg;
        double *a, diag;

        J[i] = I[i] = i;

        sum = 0.;
        diag = A(i, i);
        beg = A.GetI()[i];
        row = A.GetJ() + beg;
        a = A.GetData() + beg;
        // if (diag <= 0.0) SA_PRINTF("!!! diag = %e\n", diag);
        SA_ASSERT(diag > 0.);
        const int a_rsz = const_cast<SparseMatrix&>(A).RowSize(i);
        for (int j=0; j < a_rsz; ++j)
        {
            // if (A(row[j], row[j]) <= 0.0) SA_PRINTF("!!! j diag = %e\n", A(row[j], row[j]));
            SA_ASSERT(A(row[j], row[j]) > 0.);
            sum += fabs(a[j]) * sqrt(diag / A(row[j], row[j]));
        }
        SA_ASSERT(sum > 0.);
        Data[i] = sum;
    }
    I[n] = n;

    return (new SparseMatrix(I, J, Data, n, n));
}

SparseMatrix *mbox_snd_D_sparse_from_dense(const DenseMatrix& A)
{
    const int n = A.Height();
    SA_ASSERT(A.Width() == n);
    int *I = new int[n+1];
    int *J = new int[n];
    double *Data = new double[n];
    double sum;

    for (int i=0; i < n; ++i)
    {
        double diag = A(i,i);
        SA_ASSERT(diag > 0.);
        J[i] = I[i] = i;
        sum = 0.;
        for (int j=0; j < n; ++j)
            sum += fabs(A(i,j)) * sqrt(diag / A(j,j));
        SA_ASSERT(sum > 0.);
        Data[i] = sum;
    }
    I[n] = n;

    return (new SparseMatrix(I, J, Data, n, n));
}

SparseMatrix *mbox_restr_snd_D_sparse_from_sparse(const SparseMatrix& A,
                  const int *elem_to_smallcol, const Table& bigcol_to_elem,
                  int small_id, int big_id, int small_sz)
{
    SA_ASSERT(A.Size() == A.Width());

    const int * const bigcol = bigcol_to_elem.GetRow(big_id);
    const int bigcol_sz = bigcol_to_elem.RowSize(big_id);
    SA_ASSERT(A.Size() == bigcol_sz);

    int *I = new int[bigcol_sz + 1];
    int *J = new int[small_sz];
    double *Data = new double[small_sz];
    double sum;

    int filled = 0;
    for (int i=0; i < bigcol_sz; ++i)
    {
        int *row, beg;
        double *a, diag;

        SA_ASSERT(filled <= small_sz);

        I[i] = filled;

        if (elem_to_smallcol[bigcol[i]] != small_id)
            continue;

        SA_ASSERT(filled < small_sz);

        sum = 0.;
        diag = A(i, i);
        beg = A.GetI()[i];
        row = A.GetJ() + beg;
        a = A.GetData() + beg;
        SA_ASSERT(diag > 0.);
        const int a_rsz = const_cast<SparseMatrix&>(A).RowSize(i);
        for (int j=0; j < a_rsz; ++j)
        {
            SA_ASSERT(A(row[j], row[j]) > 0.);
            sum += fabs(a[j]) * sqrt(diag / A(row[j], row[j]));
        }
        SA_ASSERT(sum > 0.);
        Data[filled] = sum;
        J[filled++] = i;
    }
    I[bigcol_sz] = filled;
    SA_ASSERT(small_sz == I[bigcol_sz]);

    return (new SparseMatrix(I, J, Data, bigcol_sz, bigcol_sz));
}

DenseMatrix *mbox_snd_Dinv_dense_from_dense(const DenseMatrix& A)
{
    const int n = A.Height();
    SA_ASSERT(A.Width() == n);
    DenseMatrix *B = new DenseMatrix(n);
    double sum;

    for (int i=0; i < n; ++i)
    {
        double diag = A(i,i);
        SA_ASSERT(diag > 0.);
        sum = 0.;
        for (int j=0; j < n; ++j)
            sum += fabs(A(i,j)) * sqrt(diag / A(j,j));
        (*B)(i,i) = 1. / sum;
        SA_ASSERT((*B)(i,i) > 0.);
    }

    return B;
}

DenseMatrix *mbox_snd_Dinv_dense_from_sparse(const SparseMatrix& A)
{
    const int n = A.Size();
    SA_ASSERT(A.Width() == n);
    DenseMatrix *B = new DenseMatrix(n);
    double sum;

    for (int i=0; i < n; ++i)
    {
        int *row, beg;
        double *a, diag;

        diag = A(i, i);
        beg = A.GetI()[i];
        row = A.GetJ() + beg;
        a = A.GetData() + beg;
        SA_ASSERT(diag > 0.);
        sum = 0.;
        for (int j=0; j < const_cast<SparseMatrix&>(A).RowSize(i); ++j)
        {
            SA_ASSERT(A(row[j], row[j]) > 0.);
            sum += fabs(a[j]) * sqrt(diag / A(row[j], row[j]));
        }
        (*B)(i,i) = 1. / sum;
        SA_ASSERT((*B)(i,i) > 0.);
    }

    return B;
}

SparseMatrix *mbox_snd_Dinv_sparse_from_sparse(const SparseMatrix& A)
{
    SA_ASSERT(A.Size() == A.Width() &&
              const_cast<SparseMatrix&>(A).Finalized());
    const int n = A.Size();
    int *I = new int[n+1];
    int *J = new int[n];
    double *Data = new double[n];
    double sum;

    for (int i=0; i < n; ++i)
    {
        int *row, beg;
        double *a, diag;

        J[i] = I[i] = i;

        sum = 0.;
        diag = A(i, i);
        beg = A.GetI()[i];
        row = A.GetJ() + beg;
        a = A.GetData() + beg;
        SA_ASSERT(diag > 0.);
        for (int j=0; j < const_cast<SparseMatrix&>(A).RowSize(i); ++j)
        {
            SA_ASSERT(A(row[j], row[j]) > 0.);
            sum += fabs(a[j]) * sqrt(diag / A(row[j], row[j]));
        }
        SA_ASSERT(sum > 0.);
        Data[i] = 1. / sum;
    }
    I[n] = n;

    return (new SparseMatrix(I, J, Data, n, n));
}

SparseMatrix *mbox_snd_Dinv_sparse_from_dense(const DenseMatrix& A)
{
    const int n = A.Height();
    SA_ASSERT(A.Width() == n);
    int *I = new int[n+1];
    int *J = new int[n];
    double *Data = new double[n];
    double sum;

    for (int i=0; i < n; ++i)
    {
        double diag = A(i,i);
        SA_ASSERT(diag > 0.);
        J[i] = I[i] = i;
        sum = 0.;
        for (int j=0; j < n; ++j)
            sum += fabs(A(i,j)) * sqrt(diag / A(j,j));
        SA_ASSERT(sum > 0.);
        Data[i] = 1. / sum;
    }
    I[n] = n;

    return (new SparseMatrix(I, J, Data, n, n));
}

Matrix **mbox_produce_snd_arr(Matrix **in_arr, int n, mbox_snd_ft snd_gen)
{
    Matrix **out_arr = new Matrix*[n];

    for (int i=0; i < n; ++i)
        out_arr[i] = snd_gen(*(in_arr[i]));

    return out_arr;
}

void mbox_get_diag_of_sparse(const SparseMatrix& Sp, Vector& d)
{
    SA_ASSERT(Sp.Width() == Sp.Size());
    const int n = Sp.Size();

    d.SetSize(n);
    for (int i=0; i < n; ++i)
        d(i) = Sp(i, i);
}

void mbox_transform_diag(const DenseMatrix& T, const SparseMatrix& D,
                         DenseMatrix& TdTt)
{
    SA_ASSERT(D.Width() == D.Size());
    SA_ASSERT(T.Width() == D.Size());

    Vector d;
    mbox_get_diag_of_sparse(D, d);
    TdTt.SetSize(T.Height());

    MultADAt(T, d, TdTt);
}

void mbox_transform_dense(const DenseMatrix& T, const DenseMatrix& A,
                          DenseMatrix& TATt)
{
    SA_ASSERT(A.Width() == A.Height());
    SA_ASSERT(T.Width() == A.Height());

    const int t_sz = T.Height();
    DenseMatrix ATt(A.Height(), t_sz);
    TATt.SetSize(t_sz);

    MultABt(A, T, ATt);
    Mult(T, ATt, TATt);
}

void mbox_transform_sparse(const SparseMatrix& A, const DenseMatrix& Tt,
                           DenseMatrix& TATt)
{
    SA_ASSERT(A.Width() == A.Size());
    SA_ASSERT(Tt.Height() == A.Width());

    DenseMatrix ATt;
    TATt.SetSize(Tt.Width());

    mbox_mult_sparse_to_dense(A, Tt, ATt);
    MultAtB(Tt, ATt, TATt);
}

void mbox_transform_vects(const DenseMatrix& Tt, const DenseMatrix& vects,
                          DenseMatrix& trans_vects)
{
    SA_ASSERT(Tt.Width() == vects.Height());
    trans_vects.SetSize(Tt.Height(), vects.Width());
    Mult(Tt, vects, trans_vects);
}

void mbox_sqnormalize_vects(DenseMatrix& vects, const double *sqnorms)
{
    SA_ASSERT(sqnorms);
    SA_ASSERT(vects.Height() > 0 && vects.Width() > 0);

    const int dim = vects.Height();
    const int vects_num = vects.Width();
    double *ptr;
    int i;
    Vector vect(NULL, dim);

    for ((i=0), (ptr=vects.Data()); i < vects_num; (++i), (ptr += dim))
    {
        vect.SetData(ptr);
        vect /= sqrt(sqnorms[i]);
    }
}

bool mbox_orthogonalize(const Vector& x, const DenseMatrix& vects,
                        const Matrix& D, const Matrix& A, double tol,
                        DenseMatrix& orth_vects, mbox_norm_ft energy_norm,
                        mbox_inner_prod_ft energy_inner_prod)
{
    const int size = vects.Height();
    const int n = vects.Width();
    SA_ASSERT(x.Size() == size);
    int i;
    Vector v;
    double *data = vects.Data();
    Vector res(x);
    double res_norm;

#if (SA_IS_DEBUG_LEVEL(8))
    Vector v1;
    for (i=0; i < n; ++i)
    {
        v.SetDataAndSize(data + i*size, size);
        SA_ALERT(SA_IS_REAL_EQ(energy_norm(D, v), 1.));
        for (int j=i+1; j < n; ++j)
        {
            v1.SetDataAndSize(data + j*size, size);
            SA_ALERT(SA_IS_REAL_EQ(energy_inner_prod(D, v, v1), 0.));
        }
    }
#endif

    const double badguy_norm = energy_norm(A, x);
    SA_PRINTF_L(9, "badguy norm: %g\n", badguy_norm);

    for (i=0; i < n; (++i), (data += size))
    {
        v.SetDataAndSize(data, size);
        add(res, -energy_inner_prod(D, x, v), v, res);
    }
    res_norm = energy_norm(A, res);
    if (SA_IS_OUTPUT_LEVEL(9))
    {
        SA_PRINTF("candidate vector norm: %g\n", res_norm);
        SA_PRINTF("[candidate norm] / [badguy norm] = %g",
                  res_norm / badguy_norm);
    }
    if (res_norm < tol)
    {
        orth_vects = vects;
        return false; //no new vectors
    }

#if (SA_IS_DEBUG_LEVEL(8))
    data = vects.Data();
    for (i=0; i < n; ++i)
    {
        v.SetDataAndSize(data + i*size, size);
        SA_ALERT(SA_IS_REAL_EQ(energy_inner_prod(D, v, res), 0.));
    }
#endif

    res /= energy_norm(D, res);
    SA_ALERT(SA_IS_REAL_EQ(energy_norm(D, res), 1.));

    orth_vects.SetSize(size, n+1);
    data = orth_vects.Data();
    const int amount = n*size;
    SA_ASSERT(res.Size() == size);
    memcpy(data, vects.Data(), amount * sizeof(*data));
    memcpy(data + amount, res.GetData(), size * sizeof(*data));

    return true; //new vector
}

bool mbox_are_equal_dense(const DenseMatrix& f, const DenseMatrix& s,
                          double tol)
{
    SA_PRINTF_L(3, "%s", "--------- compare_dense { ---------\n");

    if (SA_IS_OUTPUT_LEVEL(2))
    {
        SA_PRINTF("first: %d x %d\n", f.Height(), f.Width());
        SA_PRINTF("second: %d x %d\n", s.Height(), s.Width());
    }

    if (f.Height() != s.Height() || f.Width() != s.Width())
    {
        SA_PRINTF_L(2, "%s", "NOT matching dimensions!\n");
        SA_PRINTF_L(2, "%s", "Equal: NO!\n");
        SA_PRINTF_L(3, "%s", "--------- } compare_dense ---------\n");
        return false;
    }

    bool equal = true;

    for (int i=0; i < f.Height(); ++i)
        for (int j=0; j < f.Width(); ++j)
        {
            const double diff = f(i, j) - s(i, j);
            if (fabs(diff) > tol)
            {
                equal = false;
                SA_PRINTF_L(2, "(%d, %d): f: %g, s: %g, diff: %g\n", i, j,
                            f(i, j), s(i, j), diff);
            }
        }

    SA_PRINTF_L(2, "Equal: %s\n", (equal?"YES!":"NO!"));

    SA_PRINTF_L(3, "%s", "--------- } compare_dense ---------\n");
    return equal;
}

bool mbox_are_equal_sparse(const SparseMatrix& f, const SparseMatrix& s,
                           double tol, const int *bdr_dofs/*=NULL*/)
{
    SA_PRINTF_L(3, "%s", "--------- compare_sparse { ---------\n");

    SA_ASSERT(const_cast<SparseMatrix&>(f).Finalized() &&
              const_cast<SparseMatrix&>(s).Finalized());

    if (SA_IS_OUTPUT_LEVEL(2))
    {
        SA_PRINTF("first: %d x %d, NNZ: %d\n", f.Size(), f.Width(),
                  f.NumNonZeroElems());
        SA_PRINTF("second: %d x %d, NNZ: %d\n", s.Size(), s.Width(),
                  s.NumNonZeroElems());
    }

    if (f.Size() != s.Size() || f.Width() != s.Width())
    {
        SA_PRINTF_L(2, "%s", "NOT matching dimensions!\n");
        SA_PRINTF_L(2, "%s", "Equal: NO!\n");
        SA_PRINTF_L(3, "%s", "--------- } compare_sparse ---------\n");
        return false;
    }

    bool equal = true;

    {
        const int * const I = f.GetI();
        const int * const J = f.GetJ();
        const double * const Data = f.GetData();

        for (int i=0; i < f.Size(); ++i)
            for (int j = I[i]; j < I[i+1]; ++j)
            {
                const double ff = Data[j];
                const double ss = s(i, J[j]);
                const double diff = ff - ss;
                if (fabs(diff) > tol)
                {
                    equal = false;
                    if (SA_IS_OUTPUT_LEVEL(2))
                    {
                        PROC_STR_STREAM << "f: (" << i << ", " << J[j]
                                        << "): f: " << ff << ", s: " << ss
                                        << ", diff: " << diff;
                        if (bdr_dofs)
                            PROC_STR_STREAM << " -> (" << (int)bdr_dofs[i]
                                            << ", " << (int)bdr_dofs[J[j]]
                                            << ")";
                        PROC_STR_STREAM << "\n";
                        SA_PRINTF("%s", PROC_STR_STREAM.str().c_str());
                        PROC_CLEAR_STR_STREAM;
                    }
                }
            }
    }

    {
        const int * const I = s.GetI();
        const int * const J = s.GetJ();
        const double * const Data = s.GetData();

        for (int i=0; i < s.Size(); ++i)
            for (int j = I[i]; j < I[i+1]; ++j)
            {
                const double ff = f(i, J[j]);
                const double ss = Data[j];
                const double diff = ff - ss;
                if (fabs(diff) > tol)
                {
                    equal = false;
                    if (SA_IS_OUTPUT_LEVEL(2))
                    {
                        PROC_STR_STREAM << "s: (" << i << ", " << J[j]
                                        << "): f: " << ff << ", s: " << ss
                                        << ", diff: " << diff;
                        if (bdr_dofs)
                            PROC_STR_STREAM << " -> (" << (int)bdr_dofs[i]
                                            << ", " << (int)bdr_dofs[J[j]]
                                            << ")";
                        PROC_STR_STREAM << "\n";
                        SA_PRINTF("%s", PROC_STR_STREAM.str().c_str());
                        PROC_CLEAR_STR_STREAM;
                    }
                }
            }
    }

    SA_PRINTF_L(2, "Equal: %s\n", (equal?"YES!":"NO!"));

    SA_PRINTF_L(3, "%s", "--------- } compare_sparse ---------\n");
    return equal;
}

void mbox_set_zero_diag_outside_set_sparse(SparseMatrix& D,
                                           const int *elem_to_smallcol,
                                           const Table& bigcol_to_elem,
                                           int small_id, int big_id)
{
    SA_ASSERT(D.Size() == D.Width());
    SA_ASSERT(D.NumNonZeroElems() == D.Size());
#if (SA_IS_DEBUG_LEVEL(4))
    for (int i=0; i < D.NumNonZeroElems(); ++i)
    {
        SA_ASSERT(D.GetI()[i] == i);
        SA_ASSERT(D.GetJ()[i] == i);
    }
#endif

    const int * const bigcol = bigcol_to_elem.GetRow(big_id);
    const int bigcol_sz = bigcol_to_elem.RowSize(big_id);
    double * const Data = D.GetData();
    SA_ASSERT(D.Size() == bigcol_sz);

    for (int i=0; i < bigcol_sz; ++i)
    {
        if (elem_to_smallcol[bigcol[i]] != small_id)
            Data[i] = 0.;
    }
}

SparseMatrix *mbox_filter_zeros_sparse(const SparseMatrix& A)
{
    SA_ASSERT(&A);
    SA_ASSERT(const_cast<SparseMatrix&>(A).Finalized());
    const int * const in_I = A.GetI();
    const double * const in_Data = A.GetData();
    const int h = A.Size();
    const int w = A.Width();
    const int in_nnz = A.NumNonZeroElems();
    const double *cdata;
    int i, j;

    int out_nnz = 0;
    for ((i=0), (cdata = in_Data); i < in_nnz; (++i), (++cdata))
    {
        if (0. != *cdata)
            ++out_nnz;
    }

    int * const out_I = new int[h + 1];
    int * const out_J = new int[out_nnz];
    double * const out_Data = new double[out_nnz];

    int idxi = 0;
    const int *cidxj = A.GetJ();
    int *idxj = out_J;
    double *data = out_Data;
    cdata = in_Data;
    for (i=0; i < h; ++i)
    {
        const int endr = in_I[i+1];
        SA_ASSERT(idxi <= out_nnz);
        out_I[i] = idxi;
        for (j = in_I[i]; j < endr; (++j), (++cdata), (++cidxj))
        {
            if (0. != *cdata)
            {
                *(data++) = *cdata;
                *(idxj++) = *cidxj;
                ++idxi;
            }
        }
    }
    SA_ASSERT(idxi == out_nnz);
    out_I[i] = idxi;

    return (new SparseMatrix(out_I, out_J, out_Data, h, w));
}

void mbox_build_schur_from_sparse(const SparseMatrix& A, DenseMatrix& S,
                                  const Array<int>& set)
{
    int i;
    SA_ASSERT(set);
    SA_ASSERT(A.Size() == A.Width());
    SA_ASSERT(0 < set.Size() && set.Size() < A.Size());
    S.SetSize(set.Size());
    DenseMatrix A22(A.Size() - set.Size()), invA22;
    DenseMatrix A12(S.Height(), A22.Width());
    Array<int> antiset(A22.Width());

    bool *inset = new bool[A.Size()];
    memset(inset, 0, sizeof(*inset) * A.Size());
    for (i=0; i < set.Size(); ++i)
    {
        SA_ASSERT(0 <= set[i] && set[i] < A.Size());
        SA_ASSERT(!inset[set[i]]);
        inset[set[i]] = true;
    }
    int j = 0;
    for (i=0; i < A.Size(); ++i)
    {
        if(!inset[i])
        {
            SA_ASSERT(j < A22.Width());
            antiset[j++] = i;
        }
    }
    SA_ASSERT(A22.Width() == j);
    delete [] inset;

    const_cast<SparseMatrix&>(A).GetSubMatrix(set, set, S);
    const_cast<SparseMatrix&>(A).GetSubMatrix(antiset, antiset, A22);
    xpacks_calc_spd_inverse_dense(A22, invA22);
    const_cast<SparseMatrix&>(A).GetSubMatrix(set, antiset, A12);
    DenseMatrix A12_invA22(A12.Height(), invA22.Width());
    Mult(A12, invA22, A12_invA22);
    DenseMatrix A12_invA22_A21(A12.Height());
    MultABt(A12_invA22, A12, A12_invA22_A21);
    SA_ASSERT(A12_invA22_A21.Height() == S.Height());
    S.Add(-1., A12_invA22_A21);
}

SparseMatrix *mbox_create_diag_sparse_for_hypre(Vector& diag)
{
    const int n = diag.Size();
    int *I = (int *) hypre_CTAlloc(HYPRE_Int, n + 1);
    int *J = (int *) hypre_CTAlloc(HYPRE_Int, n);
    double *Data = hypre_CTAlloc(double, n + 1);;

    for (int i=0; i < n; ++i)
    {
        I[i] = i;
        J[i] = i;
        Data[i] = diag(i);
    }
    I[n] = n;

    return new SparseMatrix(I, J, Data, n, n);
}

HypreParMatrix *mbox_clone_parallel_matrix(HypreParMatrix *A)
{
    if (!A)
        return NULL;

    hypre_ParCSRMatrix *hA = (hypre_ParCSRMatrix *)(*A);
    hypre_ParCSRMatrix *copy = hypre_ParCSRMatrixCompleteClone(hA);
    if (hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixDiag(hA)))
    {
        SA_ASSERT(hypre_CSRMatrixData(hypre_ParCSRMatrixDiag(copy)));
        memcpy(hypre_CSRMatrixData(hypre_ParCSRMatrixDiag(copy)),
               hypre_CSRMatrixData(hypre_ParCSRMatrixDiag(hA)),
               sizeof(*hypre_CSRMatrixData(hypre_ParCSRMatrixDiag(hA))) *
                   hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixDiag(hA)));
    }
    if (hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixOffd(hA)))
    {
        SA_ASSERT(hypre_CSRMatrixData(hypre_ParCSRMatrixOffd(copy)));
        memcpy(hypre_CSRMatrixData(hypre_ParCSRMatrixOffd(copy)),
               hypre_CSRMatrixData(hypre_ParCSRMatrixOffd(hA)),
               sizeof(*hypre_CSRMatrixData(hypre_ParCSRMatrixOffd(hA))) *
                   hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixOffd(hA)));
    }
    hypre_MatvecCommPkgCreate(copy);

    HypreParMatrix *copyA = new HypreParMatrix(copy);
    mbox_make_owner_rowstarts_colstarts(*copyA);

    return copyA;
}

void mbox_compute_abs_parallel_matrix(HypreParMatrix& A)
{
    hypre_ParCSRMatrix *hA = (hypre_ParCSRMatrix *)A;

    double * const diag_data = hypre_CSRMatrixData(hypre_ParCSRMatrixDiag(hA));
    const HYPRE_Int diag_nnz =
                        hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixDiag(hA));

    double * const offd_data = hypre_CSRMatrixData(hypre_ParCSRMatrixOffd(hA));
    const HYPRE_Int offd_nnz =
                        hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixOffd(hA));

    for (HYPRE_Int i=0; i < diag_nnz; ++i)
        diag_data[i] = fabs(diag_data[i]);

    for (HYPRE_Int i=0; i < offd_nnz; ++i)
        offd_data[i] = fabs(offd_data[i]);
}

HypreParMatrix *mbox_abs_clone_parallel_matrix(HypreParMatrix& A)
{
    HypreParMatrix *copy = mbox_clone_parallel_matrix(&A);
    mbox_compute_abs_parallel_matrix(*copy);

    return copy;
}

void mbox_scale_parallel_matrix(HypreParMatrix& A, double c)
{
    hypre_ParCSRMatrix *hA = (hypre_ParCSRMatrix *)A;

    double * const diag_data = hypre_CSRMatrixData(hypre_ParCSRMatrixDiag(hA));
    const HYPRE_Int diag_nnz =
                        hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixDiag(hA));

    double * const offd_data = hypre_CSRMatrixData(hypre_ParCSRMatrixOffd(hA));
    const HYPRE_Int offd_nnz =
                        hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixOffd(hA));

    for (HYPRE_Int i=0; i < diag_nnz; ++i)
        diag_data[i] *= c;

    for (HYPRE_Int i=0; i < offd_nnz; ++i)
        offd_data[i] *= c;
}

HypreParMatrix *mbox_scale_clone_parallel_matrix(HypreParMatrix& A, double c)
{
    HypreParMatrix *copy = mbox_clone_parallel_matrix(&A);
    mbox_scale_parallel_matrix(*copy, c);

    return copy;
}

void mbox_invert_daig_parallel_matrix(HypreParMatrix& A)
{
    SA_ASSERT(A.GetGlobalNumRows() == A.GetGlobalNumCols());
    hypre_ParCSRMatrix *hA = (hypre_ParCSRMatrix *)A;
    SA_ASSERT(hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(hA)) ==
              hypre_CSRMatrixNumCols(hypre_ParCSRMatrixDiag(hA)));
    const HYPRE_Int diag_size =
                            hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(hA));
    double * const diag_data = hypre_CSRMatrixData(hypre_ParCSRMatrixDiag(hA));
    HYPRE_Int * const diag_i = hypre_CSRMatrixI(hypre_ParCSRMatrixDiag(hA));

    for (HYPRE_Int i=0; i < diag_size; ++i)
    {
        SA_ASSERT(i == hypre_CSRMatrixJ(hypre_ParCSRMatrixDiag(hA))[diag_i[i]]);
        diag_data[diag_i[i]] = 1. / diag_data[diag_i[i]];
    }
}

void mbox_negate_daig_parallel_matrix(HypreParMatrix& A)
{
    SA_ASSERT(A.GetGlobalNumRows() == A.GetGlobalNumCols());
    hypre_ParCSRMatrix *hA = (hypre_ParCSRMatrix *)A;
    SA_ASSERT(hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(hA)) ==
              hypre_CSRMatrixNumCols(hypre_ParCSRMatrixDiag(hA)));
    const HYPRE_Int diag_size =
                            hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(hA));
    double * const diag_data = hypre_CSRMatrixData(hypre_ParCSRMatrixDiag(hA));
    HYPRE_Int * const diag_i = hypre_CSRMatrixI(hypre_ParCSRMatrixDiag(hA));

    for (HYPRE_Int i=0; i < diag_size; ++i)
    {
        SA_ASSERT(i == hypre_CSRMatrixJ(hypre_ParCSRMatrixDiag(hA))[diag_i[i]]);
        diag_data[diag_i[i]] = -diag_data[diag_i[i]];
    }
}

void mbox_sqrt_daig_parallel_matrix(HypreParMatrix& A)
{
    SA_ASSERT(A.GetGlobalNumRows() == A.GetGlobalNumCols());
    hypre_ParCSRMatrix *hA = (hypre_ParCSRMatrix *)A;
    SA_ASSERT(hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(hA)) ==
              hypre_CSRMatrixNumCols(hypre_ParCSRMatrixDiag(hA)));
    const HYPRE_Int diag_size =
                            hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(hA));
    double * const diag_data = hypre_CSRMatrixData(hypre_ParCSRMatrixDiag(hA));
    HYPRE_Int * const diag_i = hypre_CSRMatrixI(hypre_ParCSRMatrixDiag(hA));

    for (HYPRE_Int i=0; i < diag_size; ++i)
    {
        SA_ASSERT(i == hypre_CSRMatrixJ(hypre_ParCSRMatrixDiag(hA))[diag_i[i]]);
        diag_data[diag_i[i]] = sqrt(diag_data[diag_i[i]]);
    }
}

void mbox_add_diag_parallel_matrix(HypreParMatrix& A, double c)
{
    SA_ASSERT(A.GetGlobalNumRows() == A.GetGlobalNumCols());
    hypre_ParCSRMatrix *hA = (hypre_ParCSRMatrix *)A;
    SA_ASSERT(hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(hA)) ==
              hypre_CSRMatrixNumCols(hypre_ParCSRMatrixDiag(hA)));
    const HYPRE_Int diag_size =
                            hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(hA));
    double * const diag_data = hypre_CSRMatrixData(hypre_ParCSRMatrixDiag(hA));
    HYPRE_Int * const diag_i = hypre_CSRMatrixI(hypre_ParCSRMatrixDiag(hA));

    for (HYPRE_Int i=0; i < diag_size; ++i)
    {
        SA_ASSERT(i == hypre_CSRMatrixJ(hypre_ParCSRMatrixDiag(hA))[diag_i[i]]);
        diag_data[diag_i[i]] += c;
    }
}

void mbox_make_owner_rowstarts_colstarts(HypreParMatrix& A)
{
    hypre_ParCSRMatrix *hA = (hypre_ParCSRMatrix *)A;

    if (!hypre_ParCSRMatrixOwnsRowStarts(hA))
    {
        SA_ASSERT(hypre_ParCSRMatrixRowStarts(hA));
        HYPRE_Int *row_starts = hypre_CTAlloc(HYPRE_Int, 2);
        SA_ASSERT(row_starts);
        memcpy(row_starts, hypre_ParCSRMatrixRowStarts(hA),
               sizeof(*row_starts) * (2));
        hypre_ParCSRMatrixRowStarts(hA) = row_starts;
        hypre_ParCSRMatrixSetRowStartsOwner(hA, 1);
    }

    if (!hypre_ParCSRMatrixOwnsColStarts(hA))
    {
        SA_ASSERT(hypre_ParCSRMatrixColStarts(hA));
        HYPRE_Int *col_starts = hypre_CTAlloc(HYPRE_Int, 2);
        SA_ASSERT(col_starts);
        memcpy(col_starts, hypre_ParCSRMatrixColStarts(hA),
               sizeof(*col_starts) * (2));
        hypre_ParCSRMatrixColStarts(hA) = col_starts;
        hypre_ParCSRMatrixSetColStartsOwner(hA, 1);
    }
}

void mbox_make_owner_partitioning(HypreParVector& v)
{
    hypre_ParVector *hv = (hypre_ParVector *)v;

    if (!hypre_ParVectorOwnsPartitioning(hv))
    {
        SA_ASSERT(hypre_ParVectorPartitioning(hv));
        HYPRE_Int *partitioning = hypre_CTAlloc(HYPRE_Int, 2);
        SA_ASSERT(partitioning);
        memcpy(partitioning, hypre_ParVectorPartitioning(hv),
               sizeof(*partitioning) * (2));
        hypre_ParVectorPartitioning(hv) = partitioning;
        hypre_ParVectorSetPartitioningOwner(hv, 1);
    }
}

void mbox_entry_mult_vector(Vector& f, const Vector& s)
{
    SA_ASSERT(f.Size() == s.Size());
    const int n = f.Size();
    for (int i=0; i < n; ++i)
        f(i) *= s(i);
}

void mbox_invert_vector(Vector& v)
{
    const int n = v.Size();
    for (int i=0; i < n; ++i)
        v(i) = 1. / v(i);
}

void mbox_sqrt_vector(Vector& v)
{
    const int n = v.Size();
    for (int i=0; i < n; ++i)
        v(i) = sqrt(v(i));
}

HypreParVector *mbox_get_diag_parallel_matrix(HypreParMatrix& A)
{
    SA_ASSERT(A.GetGlobalNumRows() == A.GetGlobalNumCols());
    Vector ldiag;
    mbox_vector_initialize_for_hypre(ldiag,
        hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag((hypre_ParCSRMatrix *)
                                                      A)));
    /*
    std::cout << "[" << PROC_RANK << "] A.ColPart()[0] = " << A.ColPart()[0] << std::endl;
    std::cout << "[" << PROC_RANK << "] ldiag.Size() = " << ldiag.Size() 
              << ", A.ColPart()[PROC_RANK + 1] = " << A.ColPart()[PROC_RANK + 1] 
              << ", A.ColPart()[PROC_RANK] = " << A.ColPart()[PROC_RANK] << std::endl;
    */
    SA_ASSERT(ldiag.Size() == A.ColPart()[1] -
                              A.ColPart()[0]);
    SparseMatrix sdiag;
    A.GetDiag(sdiag);
    sdiag.GetDiag(ldiag);
    SA_ASSERT(ldiag.Size() == A.ColPart()[1] -
                              A.ColPart()[0]);
    double *p;
    ldiag.StealData(&p);
    HypreParVector *diag =
        new HypreParVector(PROC_COMM, A.GetGlobalNumCols(), p, A.ColPart());
    hypre_SeqVectorSetDataOwner(hypre_ParVectorLocalVector((hypre_ParVector *)
                                                           *diag), 1);
    mbox_make_owner_partitioning(*diag);
    return diag;
}

HypreParMatrix *mbox_create_diag_parallel_matrix(HypreParVector& diag)
{
    hypre_ParVector *hdiag = (hypre_ParVector *)diag;
    SparseMatrix *diag_sp = mbox_create_diag_sparse_for_hypre(diag);
    HypreParMatrix *diag_mat =
        new HypreParMatrix(PROC_COMM, hypre_ParVectorGlobalSize(hdiag),
                           hypre_ParVectorPartitioning(hdiag), diag_sp);
    diag_sp->LoseData();
    delete diag_sp;
    hypre_CSRMatrixSetDataOwner(
        hypre_ParCSRMatrixDiag((hypre_ParCSRMatrix *) *diag_mat), 1);

    diag_mat->SetOwnerFlags(-1,-1,-1); // new ATB 8 March 2016 for MFEM 3.1

    return diag_mat;
}

/**
   this is the l1 smoother, which we use as something spectrally equivalent
   to the diagonal of A, also with other nice properties
*/
HypreParVector *mbox_build_Dinv_neg_parallel_matrix(HypreParMatrix& A)
{
    HypreParMatrix *Aabs = mbox_abs_clone_parallel_matrix(A);
    HypreParVector *diag1 = mbox_get_diag_parallel_matrix(*Aabs);
    HypreParVector *diag2 = mbox_get_diag_parallel_matrix(*Aabs);

    mbox_sqrt_vector(*diag1);
    mbox_sqrt_vector(*diag2);
    mbox_invert_vector(*diag1);

    Vector y(diag1->Size());
    Aabs->Mult(*diag1, y);

    mbox_entry_mult_vector(*diag2, y);
    mbox_invert_vector(*diag2);
    *diag2 *= -1.;
    mbox_make_owner_partitioning(*diag2);

    delete diag1;
    delete Aabs;

    return diag2;
}

HypreParVector *mbox_clone_parallel_vector(HypreParVector *v)
{
    if (!v)
        return NULL;

    hypre_ParVector *hv = (hypre_ParVector *)(*v);
    hypre_ParVector *hclone = hypre_ParVectorCloneShallow(hv);
    hypre_SeqVectorDestroy(hypre_ParVectorLocalVector(hclone));
    hypre_ParVectorLocalVector(hclone) =
        hypre_SeqVectorCloneDeep(hypre_ParVectorLocalVector(hv));
    HypreParVector *clone = new HypreParVector((HYPRE_ParVector)hclone);
    // clone->BecomeVectorOwner(); // ATB 19 December 2014 [this change very likely causes a memory leak!]
    mbox_make_owner_partitioning(*clone);
    return clone;
}

void mbox_project_parallel(HypreParMatrix& interp, HypreParVector& v)
{
    SA_ASSERT(interp.GetGlobalNumRows() == mbox_parallel_vector_size(v));
    HypreParMatrix *restr = interp.Transpose();
    HypreParMatrix *coarse = ParMult(restr, &interp);
    SA_ASSERT(coarse->GetGlobalNumRows() == coarse->GetGlobalNumCols());
    SA_ASSERT(coarse->GetGlobalNumRows() == interp.GetGlobalNumCols());
    HypreSolver *amg = new HypreBoomerAMG(*coarse);
    HyprePCG *pcg = new HyprePCG(*coarse);

    Vector lrv(mbox_rows_in_current_process(*restr));
    restr->Mult(v, lrv);
    HypreParVector RV(PROC_COMM, interp.GetGlobalNumCols(), lrv.GetData(),
                      interp.GetColStarts());
    pcg->SetTol(1e-12);
    pcg->SetMaxIter(10 * coarse->GetGlobalNumRows());
    pcg->SetPrintLevel(1);
    pcg->SetPreconditioner(*amg);
    Vector lcv(mbox_rows_in_current_process(*restr));
    HypreParVector CV(PROC_COMM, interp.GetGlobalNumCols(), lcv.GetData(),
                      interp.GetColStarts());
    pcg->Mult(RV, CV);
    interp.Mult(CV, v);
    delete pcg;
    delete amg;
    delete coarse;
    delete restr;
}

void mbox_project_parallel(HypreParMatrix& A, HypreParMatrix& interp,
                           HypreParVector& v)
{
    SA_ASSERT(A.GetGlobalNumCols() == A.GetGlobalNumRows());
    SA_ASSERT(interp.GetGlobalNumRows() == A.GetGlobalNumRows());
    SA_ASSERT(interp.GetGlobalNumRows() == mbox_parallel_vector_size(v));
    HypreParMatrix *restr = interp.Transpose();
    HypreParMatrix *coarse = RAP(&A, &interp);
    SA_ASSERT(coarse->GetGlobalNumRows() == coarse->GetGlobalNumCols());
    SA_ASSERT(coarse->GetGlobalNumRows() == interp.GetGlobalNumCols());
    HypreSolver *amg = new HypreBoomerAMG(*coarse);
    HyprePCG *pcg = new HyprePCG(*coarse);

    Vector lbv(v.Size());
    A.Mult(v, lbv);
    Vector lrv(mbox_rows_in_current_process(*restr));
    restr->Mult(lbv, lrv);
    HypreParVector RV(PROC_COMM, interp.GetGlobalNumCols(), lrv.GetData(),
                      interp.GetColStarts());
    pcg->SetTol(1e-12);
    pcg->SetMaxIter(10 * coarse->GetGlobalNumRows());
    pcg->SetPrintLevel(1);
    pcg->SetPreconditioner(*amg);
    Vector lcv(mbox_rows_in_current_process(*restr));
    HypreParVector CV(PROC_COMM, interp.GetGlobalNumCols(), lcv.GetData(),
                      interp.GetColStarts());
    pcg->Mult(RV, CV);
    interp.Mult(CV, v);
    delete pcg;
    delete amg;
    delete coarse;
    delete restr;
}

HypreParVector *mbox_restrict_vec_to_faces(const Vector& vec, int elements_dofs,
                                           HYPRE_Int *new_processor_offsets, int new_glob_size)
{
    HypreParVector *restr_vec = new HypreParVector(PROC_COMM, new_glob_size, new_processor_offsets);
    mbox_make_owner_partitioning(*restr_vec);
    SA_ASSERT(vec.Size() - elements_dofs == restr_vec->Size());
    for (int i=0; i < restr_vec->Size(); i++)
        (*restr_vec)(i) = vec(elements_dofs + i);

    return restr_vec;
}

} // namespace saamge
