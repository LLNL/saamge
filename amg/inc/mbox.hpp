/*! \file
    \brief Matrix related functionality.

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

#pragma once
#ifndef _MBOX_HPP
#define _MBOX_HPP

#include "common.hpp"
#include <fstream>
#include <seq_mv.h>
#include <_hypre_parcsr_mv.h>
#include <_hypre_parcsr_ls.h>
#include <mfem.hpp>
using std::ofstream;
using std::ifstream;
using namespace mfem;


/* Types */
/*! \brief A function type for generators of a matrix from another matrix.

    This type is for generating a dense matrix from another dense matrix.
    If we consider the generalized eigenvalue problem:
        \f$ A\mathbf{x} = \lambda B \mathbf{x}\f$,
    then this type of functions generates dense B from given dense \a A.

    \param A (IN) The first matrix.

    \returns The second (generated) matrix.

    \warning Usually the returned dense matrix must be freed by the caller.
*/
typedef DenseMatrix * (*mbox_snd_dense_from_dense_ft)(const DenseMatrix& A);

/*! \brief A function type for generators of a matrix from another matrix.

    This type is for generating a dense matrix from another sparse matrix.
    If we consider the generalized eigenvalue problem:
        \f$ A\mathbf{x} = \lambda B \mathbf{x}\f$,
    then this type of functions generates dense B from given sparse \a A.

    \param A (IN) The first matrix.

    \returns The second (generated) matrix.

    \warning Usually the returned dense matrix must be freed by the caller.
*/
typedef DenseMatrix * (*mbox_snd_dense_from_sparse_ft)(const SparseMatrix& A);

/*! \brief A function type for generators of a matrix from another matrix.

    This type is for generating a sparse matrix from another sparse matrix.
    If we consider the generalized eigenvalue problem:
        \f$ A\mathbf{x} = \lambda B \mathbf{x}\f$,
    then this type of functions generates sparse B from given sparse \a A.

    \param A (IN) The first matrix.

    \returns The second (generated) matrix.

    \warning Usually the returned sparse matrix must be freed by the caller.
*/
typedef SparseMatrix * (*mbox_snd_sparse_from_sparse_ft)(const SparseMatrix& A);

/*! \brief A function type for generators of a matrix from another matrix.

    This type is for generating a sparse matrix from another dense matrix.
    If we consider the generalized eigenvalue problem:
        \f$ A\mathbf{x} = \lambda B \mathbf{x}\f$,
    then this type of functions generates sparse B from given dense \a A.

    \param A (IN) The first matrix.

    \returns The second (generated) matrix.

    \warning Usually the returned sparse matrix must be freed by the caller.
*/
typedef SparseMatrix * (*mbox_snd_sparse_from_dense_ft)(const DenseMatrix& A);

/*! \brief A function type for generators of a matrix from another matrix.

    This type is for generating a matrix from another matrix. The matrix type
    is generic not related to sparse or dense. This generalization makes it
    easier to define functions with more generic argument types (like
    \b mbox_produce_snd_arr). This approach will work only with virtual
    functions and may require virtual destructors.

    If we consider the generalized eigenvalue problem:
        \f$ A\mathbf{x} = \lambda B \mathbf{x}\f$,
    then this type of functions generates B from given \a A.

    \param A (IN) The first matrix.

    \returns The second (generated) matrix.

    \warning Usually the returned matrix must be freed by the caller. Virtual
             destructors are needed in this case or the caller must know the
             type of the returned matrix when freeing it.
*/
typedef Matrix * (*mbox_snd_ft)(const Matrix& A);

/*! \brief A function type for generators of a dioagonal from another matrix.

    \param A (IN) The matrix.

    \returns The generated diagonal.

    \warning Usually the returned vector must be freed by the caller.
*/
typedef HypreParVector * (*mbox_snd_vec_from_mat_par_ft)(HypreParMatrix& A);

/*! \brief A function type for norms.

    Computes the norm of \a x.

    \param A (IN) A matrix.
    \param x (IN) The vector the norm of which will be computed.

    \returns The norm of \a x.

    \warning The matrix type is general and specifications for sparse or dense
             may be done. One should be careful with type castings and type
             mismatches.
*/
typedef double (*mbox_norm_ft)(const Matrix& A, const Vector& x);

/*! \brief A function type for inner products.

    Computes the inner product of \a x and \a y.

    \param A (IN) A matrix.
    \param x (IN) One vector.
    \param y (IN) Another vector.

    \returns The inner product of \a x and \a y.

    \warning The matrix type is general and specifications for sparse or dense
             may be done. One should be careful with type castings and type
             mismatches.
*/
typedef double (*mbox_inner_prod_ft)(const Matrix& A, const Vector& x,
                                     const Vector& y);

/* Functions */

void hypre_par_matrix_ownership(
    HypreParMatrix &mat, bool &data, bool &row_starts, bool &col_starts);

/*! Copied from Parelag hypreExtension/hypre_CSRFactory.c */
hypre_ParCSRMatrix * hypre_IdentityParCSRMatrix( 
    MPI_Comm comm, HYPRE_Int global_num_rows, HYPRE_Int * row_starts);

/*! copied from Parelag hypreExtension/deleteZeros.c */
HYPRE_Int hypre_ParCSRMatrixDeleteZeros(hypre_ParCSRMatrix *A , double tol);

/*! \brief Computes the energy inner product with sparse \a A.

    \a y'\a A\a x.

    \param A (IN) A sparse matrix.
    \param x (IN) One vector.
    \param y (IN) Another vector.

    \returns The energy inner product of \a x and \a y.

    \warning \a A must be s.p.d.
*/
double mbox_energy_inner_prod_sparse(const SparseMatrix& A, const Vector& x,
                                     const Vector& y);

/*! \brief Computes the energy inner product with dense \a A.

    \a y'\a A\a x.

    \param A (IN) A dense matrix.
    \param x (IN) One vector.
    \param y (IN) Another vector.

    \returns The energy inner product of \a x and \a y.

    \warning \a A must be s.p.d.
*/
double mbox_energy_inner_prod_dense(const DenseMatrix& A, const Vector& x,
                                    const Vector& y);

/*! \brief Computes the generalized Rayleigh quotient of matrix pair (A,B).

    Computes the generalized Rayleigh quotient of matrix pair (A,B) for vector
    \a x.

    \a x'\a A\a x / \a x'\a B\a x

    \param A (IN) A sparse matrix.
    \param B (IN) B sparse matrix.
    \param x (IN) One vector.

    \returns The generalized Rayleigh quotient of matrix pair (A,B) for vector
             \a x.
*/
double mbox_gen_rayleigh_quot_sparse(const SparseMatrix& A,
                                     const SparseMatrix& B, const Vector& x);

/*! \brief Computes the generalized Rayleigh quotient of matrix pair (A,B).

    Computes the generalized Rayleigh quotient of matrix pair (A,B) for vector
    \a x.

    \a x'\a A\a x / \a x'\a B\a x

    \param A (IN) A dense matrix.
    \param B (IN) B dense matrix.
    \param x (IN) One vector.

    \returns The generalized Rayleigh quotient of matrix pair (A,B) for vector
             \a x.
*/

double mbox_gen_rayleigh_quot_dense(const DenseMatrix& A, const DenseMatrix& B,
                                    const Vector& x);

/*! \brief Computes the energy norm with sparse \a A.

    sqrt(\a x'\a A\a x).

    \param A (IN) A sparse matrix.
    \param x (IN) The vector the norm of which will be computed.

    \returns The energy norm of \a x.

    \warning \a A must be s.p.d.
*/
double mbox_energy_norm_sparse(const SparseMatrix& A, const Vector& x);

/*! \brief Computes the energy norm with dense \a A.

    sqrt(\a x'\a A\a x).

    \param A (IN) A dense matrix.
    \param x (IN) The vector the norm of which will be computed.

    \returns The energy norm of \a x.

    \warning \a A must be s.p.d.
*/
double mbox_energy_norm_dense(const DenseMatrix& A, const Vector& x);

/*! \brief Creates a copy of a table.

    \param src (IN) The table to be copied.

    \returns The copy. If \a src is NULL, it returns NULL.

    \warning The returned table must be freed by the caller.
*/
Table *mbox_copy_table(const Table *src);

/*! \brief Frees an array of matrices.

    \param arr (IN) The array of pointers to matrices.
    \param n (IN) The number of elements of the array.

    \warning Virtual destructors are likely to be needed.
*/
void mbox_free_matr_arr(Matrix **arr, int n);

/*! \brief Creates a copy of a finalized sparse matrix.

    \param src (IN) The sparse matrix to be copied.

    \returns The copy. If \a src is NULL, it returns NULL.

    \warning The returned sparse matrix must be freed by the caller.
    \warning \a src must be finalized.
*/
SparseMatrix *mbox_copy_sparse_matr(const SparseMatrix *src);

/*! \brief Copies an array of sparse matrices.

    \param src (IN) The array of pointers to sparse matrices to be copied.
    \param n (IN) The number of elements in the array.

    \returns The copy. If \a src is NULL, it returns NULL.

    \warning The returned array must be freed by the caller using
             \b mbox_free_matr_arr.
*/
SparseMatrix **mbox_copy_sparse_matr_arr(SparseMatrix **src, int n);

/*! \brief Copies an array of dense matrices.

    \param src (IN) The array of pointers to dense matrices to be copied.
    \param n (IN) The number of elements in the array.

    \returns The copy. If \a src is NULL, it returns NULL.

    \warning The returned array must be freed by the caller using
             \b mbox_free_matr_arr.
*/
DenseMatrix **mbox_copy_dense_matr_arr(DenseMatrix **src, int n);

/*! \brief Loads a table from a file.

    \a filename is a binary file with format:
    <number of rows><number of connections><the I array><the J array>.

    \param filename (IN) The name of the file with the table.

    \returns The loaded table.

    \warning The returned table must be freed by the caller.
*/
Table *mbox_read_table(const char *filename);

/*! \brief Writes a table to a file.

    \a filename is a binary file with format:
    <number of rows><number of connections><the I array><the J array>.

    \param filename (IN) The name of the file to write. If it exists, it will
                         be erased prior to writing.
    \param tbl (IN) The table to be written.
*/
void mbox_write_table(const char *filename, const Table& tbl);

/*! \brief Loads a finalized sparse matrix from a file.

    \a filename is a binary file with format:
    <number of rows><number of columns><number of non-zero elements><the I array><the J array><the A array>.

    \param filename (IN) The name of the file with the sparse matrix.

    \returns The loaded sparse matrix.

    \warning The returned sparse matrix must be freed by the caller.
*/
SparseMatrix *mbox_read_sparse_matr(const char *filename);

/*! \brief Loads a finalized sparse matrix from an input stream.

    \a ispm must give a binary stream with format:
    <number of rows><number of columns><number of non-zero elements><the I array><the J array><the A array>.

    \param ispm (IN) The input stream.

    \returns The loaded sparse matrix.

    \warning The returned sparse matrix must be freed by the caller.
*/
SparseMatrix *mbox_read_sparse_matr(ifstream& ispm);

/*! \brief Writes a finalized sparse matrix to a file.

    \a filename is a binary file with format:
    <number of rows><number of columns><number of non-zero elements><the I array><the J array><the A array>.

    \param filename (IN) The name of the file to write. If it exists, it will
                         be erased prior to writing.
    \param spm (IN) The sparse matrix to be written.

    \warning \a spm must be finalized.
*/
void mbox_write_sparse_matr(const char *filename, const SparseMatrix& spm);

/*! \brief Writes a finalized sparse matrix to an output stream.

    The output to \a ospm is binary with format:
    <number of rows><number of columns><number of non-zero elements><the I array><the J array><the A array>.

    \param ospm (IN) The output stream.
    \param spm (IN) The sparse matrix to be written.

    \warning \a spm must be finalized.
*/
void mbox_write_sparse_matr(ofstream& ospm, const SparseMatrix& spm);

/*! \brief Loads a dense matrix from a file.

    \a filename is a binary file with format:
    <number of rows><number of columns><the matrix elements (data)>.

    \param filename (IN) The name of the file with the dense matrix.

    \returns The loaded sparse matrix.

    \warning The returned dense matrix must be freed by the caller.
*/
DenseMatrix *mbox_read_dense_matr(const char *filename);

/*! \brief Loads a dense matrix from an input stream.

    \a idem must give a binary stream with format:
    <number of rows><number of columns><the matrix elements (data)>.

    \param idem (IN) The input stream.

    \returns The loaded dense matrix.

    \warning The returned dense matrix must be freed by the caller.
*/
DenseMatrix *mbox_read_dense_matr(ifstream& idem);

/*! \brief Writes a dense matrix to a file.

    \a filename is a binary file with format:
    <number of rows><number of columns><the matrix elements (data)>.

    \param filename (IN) The name of the file to write. If it exists, it will
                         be erased prior to writing.
    \param dem (IN) The dense matrix to be written.
*/
void mbox_write_dense_matr(const char *filename, const DenseMatrix& dem);

/*! \brief Writes a dense matrix to an output stream.

    The output to \a odem is binary with format:
    <number of rows><number of columns><the matrix elements (data)>.

    \param odem (IN) The output stream.
    \param dem (IN) The dense matrix to be written.
*/
void mbox_write_dense_matr(ofstream& odem, const DenseMatrix& dem);

/*! \brief Loads an array of sparse matrices from a file.

    \a filename is a binary file with format:
    <n (number of matrices)><sparse matrix 0><sparse matrix 1>...<sparse matrix n-1>.
    The format for a single sparse matrix is like in
    \b mbox_read_sparse_matr.

    \param filename (IN) The name of the file with the array of sparse
                         matrices.
    \param n (OUT) The number of sparse matrices in the array.

    \returns The loaded array of pointers to sparse matrices.

    \warning The returned array must be freed by the caller using
             \b mbox_free_matr_arr.
*/
SparseMatrix **mbox_read_sparse_matr_arr(const char *filename, int *n);

/*! \brief Writes an array of sparse matrices to a file.

    \a filename is a binary file with format:
    <n (number of matrices)><sparse matrix 0><sparse matrix 1>...<sparse matrix n-1>.
    The format for a single sparse matrix is like in
    \b mbox_write_sparse_matr.

    \param filename (IN) The name of the file to write. If it exists, it will
                         be erased prior to writing.
    \param arr (IN) The array to be written.
    \param n (IN) The number of sparse matrices in the array.
*/
void mbox_write_sparse_matr_arr(const char *filename, SparseMatrix **arr,
                                int n);

/*! \brief Loads an array of dense matrices from a file.

    \a filename is a binary file with format:
    <n (number of matrices)><dense matrix 0><dense matrix 1>...<dense matrix n-1>.
    The format for a single dense matrix is like in \b mbox_read_dense_matr.

    \param filename (IN) The name of the file with the array of dense
                         matrices.
    \param n (OUT) The number of dense matrices in the array.

    \returns The loaded array of pointers to dense matrices.

    \warning The returned array must be freed by the caller using
            \b mbox_free_matr_arr.
*/
DenseMatrix **mbox_read_dense_matr_arr(const char *filename, int *n);

/*! \brief Writes an array of dense matrices to a file.

    \a filename is a binary file with format:
    <n (number of matrices)><dense matrix 0><dense matrix 1>...<dense matrix n-1>
    The format for a single dense matrix is like in \b mbox_write_dense_matr.

    \param filename (IN) The name of the file to write. If it exists, it will
                         be erased prior to writing.
    \param arr (IN) The array to be written.
    \param n (IN) The number of dense matrices in the array.
*/
void mbox_write_dense_matr_arr(const char *filename, DenseMatrix **arr,
                               int n);

/*! \brief Generates (converts) a dense matrix from a sparse matrix.

    \param Sp (IN) The sparse matrix to be copied (converted).
    \param D (OUT) The generated dense matrix.
*/
void mbox_convert_sparse_to_dense(const SparseMatrix& Sp, DenseMatrix& D);

/*! \brief Generates (converts) a sparse matrix from a dense matrix.

    \param D (IN) The dense matrix to be copied (converted).

    \returns The generated sparse matrix.

    \warning The returned sparse matrix must be freed by the caller.
*/
SparseMatrix *mbox_convert_dense_to_sparse(const DenseMatrix& D);

/*! \brief Gives the upper triangular part of a square dense matrix.

    It includes the main diagonal. The format is "column by column". In fact,
    if \a A is symmetric, it is equivalent to taking the lower triangular part
    in format "row by row".

    \param A (IN) The square dense matrix. Usually symmetric (to make sense).

    \returns The upper triangular part of \a A. It has n(n+1)/2 elements, where
             n is the height (width) of \a A.

    \warning The returned array must be freed by the caller.
*/
double *mbox_give_upper_trian(const DenseMatrix& A);

/*! \brief Swaps the data of two dense matrices.

    \param f (IN/OUT) The first dense matrix.
    \param s (IN/OUT) The first dense matrix.
*/
void mbox_swap_data_dense(DenseMatrix& f, DenseMatrix& s);

/*! \brief Adds a sparse matrix to a dense one destructively.

    Destructively means that the dense matrix gets modified.

    \param A (IN/OUT) The dense matrix.
    \param Sp (IN) the sparse matrix.
*/
void mbox_add_sparse_to_dense(DenseMatrix& A, const SparseMatrix& Sp);

/*! \brief Adds the diagonal of a sparse matrix to another sparse matrix.

    Adds the diagonal of \a D to \a Sp (it is expected to have a full diagonal)
    and returns the result.

    \param D (IN) See the description.
    \param Sp (IN) See the description.

    \returns The sum. See the description.

    \warning The returned sparse matrix must be freed by the caller.
*/
SparseMatrix *mbox_add_diag_to_sparse(const SparseMatrix& D,
                                      const SparseMatrix& Sp);

/*! \brief Multiplies a sparse matrix to a dense matrix.

    It computes \a AB column by column multiplying \a A to each column vector
    of \a B, one by one.

    \param A (IN) A sparse matrix.
    \param B (IN) A dense matrix.
    \param AB (OUT) A dense matrix equal to \a A * \a B
*/
void mbox_mult_sparse_to_dense(const SparseMatrix& A, const DenseMatrix& B,
                               DenseMatrix& AB);

/*! \brief Generates the negative of the inverse of the weighted l1-smoother.

    Good for structures of type \b smpr_poly_data_t.

    \param A (IN) The input matrix: \f$ A = \left( a_{ij} \right) \f$.

    \returns The negative of the inverse of the weighted l1-smoother for \a A.
             That is, it returns \f$ -D^{-1} \f$, where
             \f$ D^{-1} = \text{diag}\left( 1 / d_i \right) \f$ and
             \f$ d_i = \sum\limits_j |a_{ij}| \sqrt{\frac{a_{ii}}{a_{jj}}} \f$.

    \warning The returned sparse matrix must be freed by the caller.
*/
SparseMatrix *mbox_build_Dinv_neg(const SparseMatrix& A);

/*! \brief A function generating a matrix from another matrix.

    It generates a dense matrix from another dense matrix.
    If we consider the generalized eigenvalue problem:
        \f$ A\mathbf{x} = \lambda B \mathbf{x}\f$,
    then this function generate dense B from given dense \a A.
    Here B is the identity matrix with the size of \a A.

    \param A (IN) The first matrix.

    \returns The second (generated) matrix.

    \warning The returned dense matrix must be freed by the caller.
*/
DenseMatrix *mbox_snd_id_dense_from_dense(const DenseMatrix& A);

/*! \brief A function generating a matrix from another matrix.

    It generates a dense matrix from another sparse matrix.
    If we consider the generalized eigenvalue problem:
        \f$ A\mathbf{x} = \lambda B \mathbf{x}\f$,
    then this function generates dense B from given sparse \a A.
    Here B is the identity matrix with the size of \a A.

    \param A (IN) The first matrix.

    \returns The second (generated) matrix.

    \warning The returned dense matrix must be freed by the caller.
*/
DenseMatrix *mbox_snd_id_dense_from_sparse(const SparseMatrix& A);

/*! \brief A function generating a matrix from another matrix.

    It generates a sparse matrix from another sparse matrix.
    If we consider the generalized eigenvalue problem:
        \f$ A\mathbf{x} = \lambda B \mathbf{x}\f$,
    then this function generates sparse B from given sparse \a A.
    Here B is the identity matrix with the size of \a A.

    \param A (IN) The first matrix.

    \returns The second (generated) matrix.

    \warning The returned sparse matrix must be freed by the caller.
*/
SparseMatrix *mbox_snd_id_sparse_from_sparse(const SparseMatrix& A);

/*! \brief A function generating a matrix from another matrix.

    It generates a sparse matrix from another dense matrix.
    If we consider the generalized eigenvalue problem:
        \f$ A\mathbf{x} = \lambda B \mathbf{x}\f$,
    then this function generates sparse B from given dense \a A.
    Here B is the identity matrix with the size of \a A.

    \param A (IN) The first matrix.

    \returns The second (generated) matrix.

    \warning The returned sparse matrix must be freed by the caller.
*/
SparseMatrix *mbox_snd_id_sparse_from_dense(const DenseMatrix& A);

/*! \brief A function generating a matrix from another matrix.

    It generates a dense matrix from another dense matrix.
    If we consider the generalized eigenvalue problem:
        \f$ A\mathbf{x} = \lambda B \mathbf{x}\f$,
    then this function generate dense B from given dense \a A.
    Here B is the diagonal of \a A.

    \param A (IN) The first matrix.

    \returns The second (generated) matrix.

    \warning The returned dense matrix must be freed by the caller.
*/
DenseMatrix *mbox_snd_diagA_dense_from_dense(const DenseMatrix& A);

/*! \brief A function generating a matrix from another matrix.

    It generates a dense matrix from another sparse matrix.
    If we consider the generalized eigenvalue problem:
        \f$ A\mathbf{x} = \lambda B \mathbf{x}\f$,
    then this function generates dense B from given sparse \a A.
    Here B is the diagonal of \a A.

    \param A (IN) The first matrix.

    \returns The second (generated) matrix.

    \warning The returned dense matrix must be freed by the caller.
*/
DenseMatrix *mbox_snd_diagA_dense_from_sparse(const SparseMatrix& A);

/*! \brief A function generating a matrix from another matrix.

    It generates a sparse matrix from another sparse matrix.
    If we consider the generalized eigenvalue problem:
        \f$ A\mathbf{x} = \lambda B \mathbf{x}\f$,
    then this function generates sparse B from given sparse \a A.
    Here B is the diagonal of \a A.

    \param A (IN) The first matrix.

    \returns The second (generated) matrix.

    \warning The returned sparse matrix must be freed by the caller.
*/
SparseMatrix *mbox_snd_diagA_sparse_from_sparse(const SparseMatrix& A);

/*! \brief A function generating a matrix from another matrix.

    It generates a sparse matrix from another dense matrix.
    If we consider the generalized eigenvalue problem:
        \f$ A\mathbf{x} = \lambda B \mathbf{x}\f$,
    then this function generates sparse B from given dense \a A.
    Here B is the diagonal of \a A.

    \param A (IN) The first matrix.

    \returns The second (generated) matrix.

    \warning The returned sparse matrix must be freed by the caller.
*/
SparseMatrix *mbox_snd_diagA_sparse_from_dense(const DenseMatrix& A);

/*! \brief A function generating a matrix from another matrix.

    It generates a dense matrix from another dense matrix.
    If we consider the generalized eigenvalue problem:
        \f$ A\mathbf{x} = \lambda B \mathbf{x}\f$,
    then this function generate dense B from given dense \a A.
    Here B is the inverse of the diagonal of \a A.

    \param A (IN) The first matrix.

    \returns The second (generated) matrix.

    \warning The returned dense matrix must be freed by the caller.
*/
DenseMatrix *mbox_snd_diagAinv_dense_from_dense(const DenseMatrix& A);

/*! \brief A function generating a matrix from another matrix.

    It generates a dense matrix from another sparse matrix.
    If we consider the generalized eigenvalue problem:
        \f$ A\mathbf{x} = \lambda B \mathbf{x}\f$,
    then this function generates dense B from given sparse \a A.
    Here B is the inverse of the diagonal of \a A.

    \param A (IN) The first matrix.

    \returns The second (generated) matrix.

    \warning The returned dense matrix must be freed by the caller.
*/
DenseMatrix *mbox_snd_diagAinv_dense_from_sparse(const SparseMatrix& A);

/*! \brief A function generating a matrix from another matrix.

    It generates a sparse matrix from another sparse matrix.
    If we consider the generalized eigenvalue problem:
        \f$ A\mathbf{x} = \lambda B \mathbf{x}\f$,
    then this function generates sparse B from given sparse \a A.
    Here B is the inverse of the diagonal of \a A.

    \param A (IN) The first matrix.

    \returns The second (generated) matrix.

    \warning The returned sparse matrix must be freed by the caller.
*/
SparseMatrix *mbox_snd_diagAinv_sparse_from_sparse(const SparseMatrix& A);

/*! \brief A function generating a matrix from another matrix.

    It generates a sparse matrix from another dense matrix.
    If we consider the generalized eigenvalue problem:
        \f$ A\mathbf{x} = \lambda B \mathbf{x}\f$,
    then this function generates sparse B from given dense \a A.
    Here B is the inverse of the diagonal of \a A.

    \param A (IN) The first matrix.

    \returns The second (generated) matrix.

    \warning The returned sparse matrix must be freed by the caller.
*/
SparseMatrix *mbox_snd_diagAinv_sparse_from_dense(const DenseMatrix& A);

/*! \brief A function generating a matrix from another matrix.

    It generates a dense matrix from another dense matrix.
    If we consider the generalized eigenvalue problem:
        \f$ A\mathbf{x} = \lambda B \mathbf{x}\f$,
    then this function generate dense B from given dense \a A.
    Here B is the weighted l1-smoother i.e. \f$ B = \text{diag}(d_i) \f$ with
    entries \f$ d_i = \sum\limits_j |a_{ij}| \sqrt{\frac{a_{ii}}{a_{jj}}} \f$.

    \param A (IN) The first matrix.

    \returns The second (generated) matrix.

    \warning The returned dense matrix must be freed by the caller.
*/
DenseMatrix *mbox_snd_D_dense_from_dense(const DenseMatrix& A);

/*! \brief A function generating a matrix from another matrix.

    It generates a dense matrix from another sparse matrix.
    If we consider the generalized eigenvalue problem:
        \f$ A\mathbf{x} = \lambda B \mathbf{x}\f$,
    then this function generates dense B from given sparse \a A.
    Here B is the weighted l1-smoother i.e. \f$ B = \text{diag}(d_i) \f$ with
    entries \f$ d_i = \sum\limits_j |a_{ij}| \sqrt{\frac{a_{ii}}{a_{jj}}} \f$.

    \param A (IN) The first matrix.

    \returns The second (generated) matrix.

    \warning The returned dense matrix must be freed by the caller.
*/
DenseMatrix *mbox_snd_D_dense_from_sparse(const SparseMatrix& A);

/*! \brief A function generating a matrix from another matrix.

    It generates a sparse matrix from another sparse matrix.
    If we consider the generalized eigenvalue problem:
        \f$ A\mathbf{x} = \lambda B \mathbf{x}\f$,
    then this function generates sparse B from given sparse \a A.
    Here B is the weighted l1-smoother i.e. \f$ B = \text{diag}(d_i) \f$ with
    entries \f$ d_i = \sum\limits_j |a_{ij}| \sqrt{\frac{a_{ii}}{a_{jj}}} \f$.

    \param A (IN) The first matrix.

    \returns The second (generated) matrix.

    \warning The returned sparse matrix must be freed by the caller.
*/
SparseMatrix *mbox_snd_D_sparse_from_sparse(const SparseMatrix& A);

/*! \brief A function generating a matrix from another matrix.

    It generates a sparse matrix from another dense matrix.
    If we consider the generalized eigenvalue problem:
        \f$ A\mathbf{x} = \lambda B \mathbf{x}\f$,
    then this function generates sparse B from given dense \a A.
    Here B is the weighted l1-smoother i.e. \f$ B = \text{diag}(d_i) \f$ with
    entries \f$ d_i = \sum\limits_j |a_{ij}| \sqrt{\frac{a_{ii}}{a_{jj}}} \f$.

    \param A (IN) The first matrix.

    \returns The second (generated) matrix.

    \warning The returned sparse matrix must be freed by the caller.
*/
SparseMatrix *mbox_snd_D_sparse_from_dense(const DenseMatrix& A);

/*! \brief A function generating a matrix from another matrix.

    It generates a sparse matrix from another sparse matrix.

    Generates the weighted l1-smoother (see \b mbox_snd_D_sparse_from_sparse)
    but a restricted version of it.

    We have small non-intersecting collections of elements (in general) and big
    (possibly intersecting) collections of elements. Each small collection is
    entirely contained in one or more big collections.

    The sparse matrix \a A is defined on a big collection. A small collection,
    contained entirely in the big one is, given. This function generates the
    weighted l1-smoother such that it is the same as before but the diagonal
    entries corresponding to elements outside the small collection are left
    empty.

    \param A (IN) The first matrix.
    \param elem_to_smallcol (IN) An array matching each element to its unique
                                 small collection.
    \param bigcol_to_elem (IN) A relation table relating big collections to
                               elements.
    \param small_id (IN) The number of the small collection.
    \param small_sz (IN) The size of the small collection.

    \returns The second (generated) matrix.

    \warning The returned sparse matrix must be freed by the caller.
*/
SparseMatrix *mbox_restr_snd_D_sparse_from_sparse(const SparseMatrix& A,
                  const int *elem_to_smallcol, const Table& bigcol_to_elem,
                  int small_id, int big_id, int small_sz);

/*! \brief A function generating a matrix from another matrix.

    It generates a dense matrix from another dense matrix.
    If we consider the generalized eigenvalue problem:
        \f$ A\mathbf{x} = \lambda B \mathbf{x}\f$,
    then this function generate dense B from given dense \a A.
    Here B is the inverse of the weighted l1-smoother i.e.
    \f$ B = \text{diag}\left( 1 / d_i \right) \f$, where
    \f$ d_i = \sum\limits_j |a_{ij}| \sqrt{\frac{a_{ii}}{a_{jj}}} \f$.

    \param A (IN) The first matrix.

    \returns The second (generated) matrix.

    \warning The returned dense matrix must be freed by the caller.
*/
DenseMatrix *mbox_snd_Dinv_dense_from_dense(const DenseMatrix& A);

/*! \brief A function generating a matrix from another matrix.

    It generates a dense matrix from another sparse matrix.
    If we consider the generalized eigenvalue problem:
        \f$ A\mathbf{x} = \lambda B \mathbf{x}\f$,
    then this function generates dense B from given sparse \a A.
    Here B is the inverse of the weighted l1-smoother i.e.
    \f$ B = \text{diag}\left( 1 / d_i \right) \f$, where
    \f$ d_i = \sum\limits_j |a_{ij}| \sqrt{\frac{a_{ii}}{a_{jj}}} \f$.

    \param A (IN) The first matrix.

    \returns The second (generated) matrix.

    \warning The returned dense matrix must be freed by the caller.
*/
DenseMatrix *mbox_snd_Dinv_dense_from_sparse(const SparseMatrix& A);

/*! \brief A function generating a matrix from another matrix.

    It generates a sparse matrix from another sparse matrix.
    If we consider the generalized eigenvalue problem:
        \f$ A\mathbf{x} = \lambda B \mathbf{x}\f$,
    then this function generates sparse B from given sparse \a A.
    Here B is the inverse of the weighted l1-smoother i.e.
    \f$ B = \text{diag}\left( 1 / d_i \right) \f$, where
    \f$ d_i = \sum\limits_j |a_{ij}| \sqrt{\frac{a_{ii}}{a_{jj}}} \f$.

    \param A (IN) The first matrix.

    \returns The second (generated) matrix.

    \warning The returned sparse matrix must be freed by the caller.
*/
SparseMatrix *mbox_snd_Dinv_sparse_from_sparse(const SparseMatrix& A);

/*! \brief A function generating a matrix from another matrix.

    It generates a sparse matrix from another dense matrix.
    If we consider the generalized eigenvalue problem:
        \f$ A\mathbf{x} = \lambda B \mathbf{x}\f$,
    then this function generates sparse B from given dense \a A.
    Here B is the inverse of the weighted l1-smoother i.e.
    \f$ B = \text{diag}\left( 1 / d_i \right) \f$, where
    \f$ d_i = \sum\limits_j |a_{ij}| \sqrt{\frac{a_{ii}}{a_{jj}}} \f$.

    \param A (IN) The first matrix.

    \returns The second (generated) matrix.

    \warning The returned sparse matrix must be freed by the caller.
*/
SparseMatrix *mbox_snd_Dinv_sparse_from_dense(const DenseMatrix& A);

/*! \brief Generates an array of matrices from an array of matrices.

    If we consider the generalized eigenvalue problem:
        \f$ A\mathbf{x} = \lambda B \mathbf{x}\f$,
    then this function generates an array of B's from a given array of A's.
    The method, for producing every single B form every single A, is given by
    \a snd_gen. The function uses generic matrix type and is not related to
    sparse or dense matrices but the given input and the expected output types
    must match the choice of \a snd_gen.

    \param in_arr (IN) The array of pointers to first matrices (A's).
    \param n (IN) The number of matrices is \a in_arr (and also in the output).
    \param snd_gen (IN) The method of generating the second matrices from the
                        first ones. It may be of type
                        mbox_snd_<de|sp>_from_<de|sp>_ft after a proper cast.

    \returns An array of \a n pointers to second (generated) matrices (B's).

    \warning The returned array must be freed by the caller using
             \b mbox_free_matr_arr.
*/
Matrix **mbox_produce_snd_arr(Matrix **in_arr, int n, mbox_snd_ft snd_gen);

/*! \brief Gets the diagonal of a sparse matrix.

    \param Sp (IN) The sparse matrix
    \param d (OUT) The diagonal.
*/
void mbox_get_diag_of_sparse(const SparseMatrix& Sp, Vector& d);

/*! \brief Transforms the diagonal of a diagonal sparse matrix.

    Computes \a T \a d \a T^t, where d is simply the diagonal of \a D.

    \param T (IN) The dense full-rank matrix which is the transpose of the
                  transformation matrix Tt = T^t.
    \param D (IN) The (possibly diagonal) sparse matrix to be transformed.
    \param TdTt (OUT) The dense resulting matrix of the transformation.
*/
void mbox_transform_diag(const DenseMatrix& T, const SparseMatrix& D,
                         DenseMatrix& TdTt);

/*! \brief Transforms a dense matrix.

    Computes \a T \a A \a T^t.

    \param T (IN) The dense full-rank matrix which is the transpose of the
                  transformation matrix Tt = T^t.
    \param A (IN) The dense matrix to be transformed.
    \param TATt (OUT) The dense resulting matrix of the transformation.
*/
void mbox_transform_dense(const DenseMatrix& T, const DenseMatrix& A,
                          DenseMatrix& TATt);

/*! \brief Transforms a sparse matrix.

    Computes T \a A \a Tt, where T = \a (Tt)^t.

    \param A (IN) The sparse matrix to be transformed.
    \param Tt (IN) The dense full-rank matrix of the transformation.
    \param TATt (OUT) The dense resulting matrix of the transformation.
*/
void mbox_transform_sparse(const SparseMatrix& A, const DenseMatrix& Tt,
                           DenseMatrix& TATt);

/*! \brief Transforms a group of vectors.

    Transforms a group of vectors presented as columns of a dense matrix.
    Computes \a Tt \a vects.

    \param Tt (IN) The dense full-rank matrix of the transformation.
    \param vects (IN) The vectors to be transformed.
    \param trans_vects (OUT) The transformed vectors.
*/
void mbox_transform_vects(const DenseMatrix& Tt, const DenseMatrix& vects,
                          DenseMatrix& trans_vects);

/*! \brief Normalizes a group of vectors presented as columns of a dense matrix.

    \param vects (IN/OUT) The vectors to be normalized.
    \param sqnorms (IN) An array with the squares of the norms of the vectors.
*/
void mbox_sqnormalize_vects(DenseMatrix& vects, const double *sqnorms);

/*! \brief Orthogonalizes a vector to a group of orthonormal vectors.

    The orthogonalization (and normalization) is with respect to the inner
    product determined by \a D, \a energy_inner_prod, and the induced norm
    \a energy_norm.

    \param x (IN) The vector to be orthogonalized and normalized.
    \param vects (IN) The group of orthonormal (with respect to
                      \a energy_inner_prod, \a D, \a energy_norm) vectors.
    \param D (IN) A matrix for the orthonormalization. It might be
                  dense or sparse and this must match the choice of
                  \a energy_inner_prod and \a energy_norm.
    \param A (IN) A matrix for the acceptance criteria. It might be
                  dense or sparse and this must match the choice of
                  \a energy_inner_prod and \a energy_norm.
    \param tol (IN) The tolerance for the acceptance criteria.
    \param orth_vects (OUT) The group of all vectors -- \a vects together with
                            the orthogonalized and normalized \a x (if \a x is
                            accepted, otherwise it is simply a copy of
                            \a vects). The resulting vectors are orthonormal
                            (with respect to \a D).
    \param energy_norm (IN) The energy norm. It may be for dense or sparse \a D
                            but it has to match the choice of \a D and
                            \a energy_inner_prod and an appropriate cast will
                            be necessary.
    \param energy_inner_prod (IN) The energy inner product. It may be for dense
                                  or sparse \a D but it has to match the choice
                                  of \a D and \a energy_inner_prod and an
                                  appropriate cast will be necessary.

    \returns \em true if new vector was introduced. \em false if no new vector
             was introduced.

    \warning \a energy_norm must be the norm induced by \a energy_inner_prod.
*/
bool mbox_orthogonalize(const Vector& x, const DenseMatrix& vects,
                        const Matrix& D, const Matrix& A, double tol,
                        DenseMatrix& orth_vects, mbox_norm_ft energy_norm,
                        mbox_inner_prod_ft energy_inner_prod);

/*! \brief Checks if two dense matrices are equal.

    \param f (IN) The first matrix.
    \param s (IN) The second matrix.
    \param tol (IN) The tolerance for the comparison.

    \returns If they are equal.
*/
bool mbox_are_equal_dense(const DenseMatrix& f, const DenseMatrix& s,
                          double tol);

/*! \brief Checks if two sparse matrices are equal.

    \param f (IN) The first matrix.
    \param s (IN) The second matrix.
    \param tol (IN) The tolerance for the comparison.
    \param bdr_dofs (IN) An array to indicate DoFs' correspondence to essential
                         boundary conditions. It can be NULL. Only provides
                         additional output.

    \returns If they are equal.
*/
bool mbox_are_equal_sparse(const SparseMatrix& f, const SparseMatrix& s,
                           double tol, const int *bdr_dofs=NULL);

/*! \brief Restricts a diagonal matrix.

    Restricts a diagonal matrix defined on a big collection to a small one.

    We have small non-intersecting collections of elements (in general) and big
    (possibly intersecting) collections of elements. Each small collection is
    entirely contained in one or more big collections.

    The sparse matrix \a D is diagonal (with a full diagonal) and defined on a
    big collection. A small collection, contained entirely in the big one is,
    given. This function sets to zero all entries of the diagonal of \a D
    corresponding to elements outside the small collection

    \param D (IN/OUT) See the description.
    \param elem_to_smallcol (IN) An array matching each element to its unique
                                 small collection.
    \param bigcol_to_elem (IN) A relation table relating big collections to
                               elements.
    \param small_id (IN) The number of the small collection.
    \param big_id (IN) The number of the big collection.
*/
void mbox_set_zero_diag_outside_set_sparse(SparseMatrix& D,
                                           const int *elem_to_smallcol,
                                           const Table& bigcol_to_elem,
                                           int small_id, int big_id);

/*! \brief Creates a copy of a finalized sparse matrix and removes the zeros.

    Removes the zero elements in a sparse CSR matrix.

    \param A (IN) The sparse matrix to be cleaned. MUST be finalized.

    \returns The cleaned copy of \a A.

    \warning The returned sparse matrix must be freed by the caller.
*/
SparseMatrix *mbox_filter_zeros_sparse(const SparseMatrix& A);

/*! \brief Computes the Schur complement for a given \a set of indices.

    \param A (IN) The sparse matrix whose Schur complement is compute.
    \param S (OUT) The dense Schur complement.
    \param set (IN) An array denoting the set of indices (subset of the whole
                    index set for \a A) for which the Schur complement will be
                    computed. It must not be a trivial set (empty one or the
                    entire set of all indices). The Schur complement will be
                    computed for the indices in the array keeping the provided
                    order in the array.
*/
void mbox_build_schur_from_sparse(const SparseMatrix& A, DenseMatrix& S,
                                  const Array<int>& set);

/*! \brief Creates a diagonal sparse matrix using HYPRE allocations.

    Calls HYPRE directly.

    This is to be used when we make a HYPRE structure an owner of the matrix
    data.

    \param diag (IN) The diagonal as a vector.

    \returns The diagonal sparse matrix.

    \warning The returned sparse matrix must be freed by the caller.
*/
SparseMatrix *mbox_create_diag_sparse_for_hypre(Vector& diag);

/*! \brief Makes a copy of a parallel matrix.

    Calls HYPRE directly.

    \param A (IN) The matrix to copy.

    \returns The copy of the matrix.

    \warning The returned matrix must be freed by the caller.
*/
HypreParMatrix *mbox_clone_parallel_matrix(HypreParMatrix *A);

/*! \brief Replaces all data entries in the matrix by their absolute values.

    Calls HYPRE directly.

    \param A (IN/OUT) The matrix to replace its entries.
*/
void mbox_compute_abs_parallel_matrix(HypreParMatrix& A);

/*! \brief Copies a matrix and replaces all entries by their absolute values.

    \param A (IN) The matrix to copy.

    \returns The copy of the matrix.

    \warning The returned matrix must be freed by the caller.
*/
HypreParMatrix *mbox_abs_clone_parallel_matrix(HypreParMatrix& A);

/*! \brief Scales a matrix by a constant

    Calls HYPRE directly.

    \param A (IN/OUT) The matrix to replace its entries.
    \param c (IN) The constant.
*/
void mbox_scale_parallel_matrix(HypreParMatrix& A, double c);

/*! \brief Copies a matrix and scales all entries by a constant.

    \param A (IN) The matrix to copy.
    \param c (IN) The constant.

    \returns The copy of the matrix.

    \warning The returned matrix must be freed by the caller.
*/
HypreParMatrix *mbox_scale_clone_parallel_matrix(HypreParMatrix& A, double c);

/*! \brief Inverts the diagonal of a matrix.

    Calls HYPRE directly.

    \param A (IN/OUT) The matrix.
*/
void mbox_invert_daig_parallel_matrix(HypreParMatrix& A);

/*! \brief Negates the diagonal of a matrix.

    Calls HYPRE directly.

    \param A (IN/OUT) The matrix.
*/
void mbox_negate_daig_parallel_matrix(HypreParMatrix& A);

/*! \brief Square-roots the diagonal of a matrix.

    Calls HYPRE directly.

    \param A (IN/OUT) The matrix.
*/
void mbox_sqrt_daig_parallel_matrix(HypreParMatrix& A);

/*! \brief Adds a constant to the diagonal of a matrix.

    Calls HYPRE directly.

    \param A (IN/OUT) The matrix.
    \param c (IN) The constant.
*/
void mbox_add_daig_parallel_matrix(HypreParMatrix& A, double c);

/*! \brief If not an owner of offsets, "hard" copies them and becomes an owner.

    Calls HYPRE directly.

    \param A (IN/OUT) The matrix.
*/
void mbox_make_owner_rowstarts_colstarts(HypreParMatrix& A);

/*! \brief If not an owner of offsets, "hard" copies them and becomes an owner.

    Calls HYPRE directly. Also assumes HYPRE is compiled with global
    partitioning of the parallel vector.

    \param v (IN/OUT) The vector.
*/
void mbox_make_owner_partitioning(HypreParVector& v);

/*! \brief Computes \a f(i) *= \a s(i).

    \param f (IN/OUT) The first vector.
    \param s (IN) The second vector.
*/
void mbox_entry_mult_vector(Vector& f, const Vector& s);

/*! \brief Computes \a v(i) = \a v(i)^{-1}.

    \param v (IN/OUT) The vector.
*/
void mbox_invert_vector(Vector& v);

/*! \brief Computes \a v(i) = \a v(i)^{1/2}.

    \param v (IN/OUT) The vector.
*/
void mbox_sqrt_vector(Vector& v);

/*! \brief Returns the diagonal of a parallel matrix.

    Calls HYPRE directly. Also assumes HYPRE is compiled with global
    partitioning of the parallel vector.

    \param A (IN) The matrix.

    \returns The diagonal of the matrix as a vector.

    \warning The returned vector must be freed by the caller.
*/
HypreParVector *mbox_get_diag_parallel_matrix(HypreParMatrix& A);

/*! \brief Creates a diagonal parallel matrix.

    Calls HYPRE directly. Also assumes HYPRE is compiled with global
    partitioning of the parallel vector.

    \param diag (IN) The diagonal as a vector.

    \returns The diagonal matrix.

    \warning The returned matrix must be freed by the caller.
*/
HypreParMatrix *mbox_create_diag_parallel_matrix(HypreParVector& diag);

/*! \brief Generates the negative of the inverse of the weighted l1-smoother.

    \param A (IN) The input matrix: \f$ A = \left( a_{ij} \right) \f$.

    \returns The negative of the inverse of the weighted l1-smoother for \a A.
             That is, it returns \f$ -D^{-1} \f$, where
             \f$ D^{-1} = \text{diag}\left( 1 / d_i \right) \f$ and
             \f$ d_i = \sum\limits_j |a_{ij}| \sqrt{\frac{a_{ii}}{a_{jj}}} \f$.

    \warning The returned matrix must be freed by the caller.
*/
HypreParVector *mbox_build_Dinv_neg_parallel_matrix(HypreParMatrix& A);

/*! \brief Makes a copy of a parallel vector.

    Calls HYPRE directly.

    \param v (IN) The vector to copy.

    \returns The copy of the vector.

    \warning The returned vector must be freed by the caller.
*/
HypreParVector *mbox_clone_parallel_vector(HypreParVector *v);

/*! \brief Computes the Euclidean orthogonal projection.

    \param interp (IN) The interpolant from the subspace.
    \param v (IN/OUT) The vector to project as input, and the projection of the
                      vector as output.
*/
void mbox_project_parallel(HypreParMatrix& interp, HypreParVector& v);

/*! \brief Computes the A-orthogonal projection.

    \param A (IN) The SPD matrix of the inner product.
    \param interp (IN) The interpolant from the subspace.
    \param v (IN/OUT) The vector to project as input, and the projection of the
                      vector as output.

    \returns The energy norm.
*/
void mbox_project_parallel(HypreParMatrix& A, HypreParMatrix& interp,
                           HypreParVector& v);

/* Inline Functions */
/*! \brief A wrapper of \b mbox_orthogonalize for sparse \a D.

    \b mbox_energy_norm_sparse and \b mbox_energy_inner_prod_sparse are used.

    \param x (IN) The vector to be orthogonalized and normalized.
    \param vects (IN) The group of orthonormal (with respect to \a D) vectors.
    \param D (IN) A sparse matrix for the orthogonalization.
    \param A (IN) A sparse matrix for the acceptance criteria.
    \param tol (IN) The tolerance for the acceptance criteria.
    \param orth_vects (OUT) The group of all vectors -- \a vects together with
                            the orthogonalized and normalized \a x (if \a x is
                            accepted, otherwise it is simply a copy of
                            \a vects). The resulting vectors are orthonormal
                            (with respect to \a D).

    \returns \em true if new vector was introduced. \em false if no new vector
             was introduced.
*/
static inline
bool mbox_orthogonalize_sparse(const Vector& x, const DenseMatrix& vects,
                               const SparseMatrix& D, const SparseMatrix& A,
                               double tol, DenseMatrix& orth_vects);

/*! \brief A wrapper of \b mbox_orthogonalize for dense \a D.

    \b mbox_energy_norm_dense and \b mbox_energy_inner_prod_dense are used.

    \param x (IN) The vector to be orthogonalized and normalized.
    \param vects (IN) The group of orthonormal (with respect to \a D) vectors.
    \param D (IN) A dense matrix for the energy inner product and the energy
                  norm.
    \param A (IN) A sparse matrix for the acceptance criteria.
    \param tol (IN) The tolerance for the acceptance criteria.
    \param orth_vects (OUT) The group of all vectors -- \a vects together with
                            the orthogonalized and normalized \a x (if \a x is
                            accepted, otherwise it is simply a copy of
                            \a vects). The resulting vectors are orthonormal
                            (with respect to \a D).

    \returns \em true if new vector was introduced. \em false if no new vector
             was introduced.
*/
static inline
bool mbox_orthogonalize_dense(const Vector& x, const DenseMatrix& vects,
                              const DenseMatrix& D, const DenseMatrix& A,
                              double tol, DenseMatrix& orth_vects);

/*! \brief Returns the global size of a parallel vector.

    Calls HYPRE directly.

    \param v (IN) The vector.

    \returns The global size of the vector.
*/
static inline
int mbox_parallel_vector_size(HypreParVector &v);

/*! \brief Returns the number of rows in the current process.

    Assumes HYPRE is compiled with global partitioning of the parallel vector.

    \param A (IN) The matrix.

    \returns See the description.
*/
static inline
int mbox_rows_in_current_process(HypreParMatrix& A);

/*! \brief Returns the number of columns in the current process.

    Assumes HYPRE is compiled with global partitioning of the parallel vector.

    \param A (IN) The matrix.

    \returns See the description.
*/
static inline
int mbox_cols_in_current_process(HypreParMatrix& A);

/*! \brief Sets the vector as an owner of its local data.

    Calls HYPRE directly.

    \param v (IN) The vector.
*/
static inline
void mbox_make_owner_data(HypreParVector &v);

/*! \brief Computes the dot product of two parallel vectors.

    Calls HYPRE directly.

    \param v1 (IN) The first vector.
    \param v2 (IN) The second vector.

    \returns The dot product.
*/
static inline
double mbox_parallel_inner_product(HypreParVector &v1, HypreParVector &v2);

/*! \brief Computes the energy inner product of two parallel vectors.

    \param A (IN) A SPD matrix.
    \param x (IN) The first vector.
    \param y (IN) The second vector.

    \returns The energy inner product product.
*/
static inline
double mbox_energy_inner_product_parallel(HypreParMatrix& A, HypreParVector& x,
                                          HypreParVector& y);

/*! \brief Computes the energy norm of a parallel vector.

    \param A (IN) A SPD matrix.
    \param x (IN) The vector.

    \returns The energy norm.
*/
static inline
double mbox_energy_norm_parallel(HypreParMatrix& A, HypreParVector& x);

/*! \brief Allocates the memory in the vector using HYPRE allocations.

    Calls HYPRE directly.

    This is to be used when we make a HYPRE structure an owner of the vector
    data.

    \param A (IN/OUT) The vector.
    \param size (IN) Desired vector size.
*/
static inline
void mbox_vector_initialize_for_hypre(Vector& v, int size);

/* Inline Functions Definitions */
static inline
bool mbox_orthogonalize_sparse(const Vector& x, const DenseMatrix& vects,
                               const SparseMatrix& D, const SparseMatrix& A,
                               double tol, DenseMatrix& orth_vects)
{
    return mbox_orthogonalize(x, vects, D, A, tol, orth_vects,
               (mbox_norm_ft)mbox_energy_norm_sparse,
               (mbox_inner_prod_ft)mbox_energy_inner_prod_sparse);
}

static inline
bool mbox_orthogonalize_dense(const Vector& x, const DenseMatrix& vects,
                              const DenseMatrix& D, const DenseMatrix& A,
                              double tol, DenseMatrix& orth_vects)
{
    return mbox_orthogonalize(x, vects, D, A, tol, orth_vects,
               (mbox_norm_ft)mbox_energy_norm_dense,
               (mbox_inner_prod_ft)mbox_energy_inner_prod_dense);
}

static inline
int mbox_parallel_vector_size(HypreParVector &v)
{
    return hypre_ParVectorGlobalSize((hypre_ParVector *)v);
}

static inline
int mbox_parallel_vector_size(const HypreParVector &v)
{
    return hypre_ParVectorGlobalSize((hypre_ParVector *)v);
}

static inline
int mbox_rows_in_current_process(HypreParMatrix& A)
{
    return A.RowPart()[1] - A.RowPart()[0];
}

static inline
int mbox_cols_in_current_process(HypreParMatrix& A)
{
    return A.ColPart()[1] - A.ColPart()[0];
}

static inline
void mbox_make_owner_data(HypreParVector &v)
{
    hypre_SeqVectorSetDataOwner(hypre_ParVectorLocalVector((hypre_ParVector *)
                                                           v), 1);
}

static inline
double mbox_parallel_inner_product(HypreParVector &v1, HypreParVector &v2)
{
    SA_ASSERT(mbox_parallel_vector_size(v1) == mbox_parallel_vector_size(v2));
    return hypre_ParVectorInnerProd((hypre_ParVector *)v1,
                                    (hypre_ParVector *)v2);
}

static inline
double mbox_energy_inner_product_parallel(HypreParMatrix& A, HypreParVector& x,
                                          HypreParVector& y)
{
    SA_ASSERT(A.GetGlobalNumRows() == A.GetGlobalNumCols());
    Vector tmp(x.Size());
    A.Mult(x, tmp);
    HypreParVector TMP(PROC_COMM, A.GetGlobalNumRows(), tmp.GetData(),
                       A.GetRowStarts());
    return mbox_parallel_inner_product(TMP, y);
}

static inline
double mbox_energy_norm_parallel(HypreParMatrix& A, HypreParVector& x)
{
    return sqrt(mbox_energy_inner_product_parallel(A, x, x));
}

static inline
void mbox_vector_initialize_for_hypre(Vector& v, int size)
{
    SA_ASSERT(size >= 0);
    v.Destroy();
    v.SetDataAndSize(hypre_CTAlloc(double, size), size);
}

#endif // _MBOX_HPP
