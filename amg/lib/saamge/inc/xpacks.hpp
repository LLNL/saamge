/*! \file
    \brief Functions that use LAPACK and ARPACK (ARPACK++) and a bit more.

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
#ifndef _XPACKS_HPP
#define _XPACKS_HPP

#include "common.hpp"
#include <mfem.hpp>

using namespace mfem;

/* Functions */
/*! \brief Computes the inverse of an s.p.d. dense matrix.

    \param Ain (IN) The s.p.d. matrix to be inverted.
    \param invA (OUT) The inverse matrix.
*/
void xpacks_calc_spd_inverse_dense(const DenseMatrix& Ain, DenseMatrix& invA);

/*! \brief Computes all eigenpairs of a dense matrix.

    The eigenvalue problem is \f$ A\mathbf{x} = \lambda \mathbf{x}\f$, where
    A is symmetric. Uses LAPACK.

    \param Ain (IN) This is A.
    \param evals (OUT) The eigenvalues.
    \param evects (OUT) The eigenvectors as columns of a dense matrix.

    \warning A is symmetric.
*/
void xpacks_calc_all_eigens_dense(const DenseMatrix& Ain, Vector& evals,
                                  DenseMatrix& evects);

/*! \brief "Fixes" a symmetric positive semi-definite matrix.

    The function eigendecomposes \a A, changes all negative eigenvalues (if
    any) to 0, and multiplies back to restore a "fixed" version of \a A.

    \param Ain (IN/OUT) The s.p.sd. matrix to "fix" (as input) and the "fixed"
                        matrix (as output).

    \returns Whether the matrix \a A was actually changed. If \em false, no
             eigenvalue is negative and \a A was not changed. If \em true, at
             least one eigenvalue was negative and an actual "fixing" procedure
             was performed.
*/
bool xpacks_fix_spsd_dense(DenseMatrix& A);

/*! \brief Computes all (generalized) eigenpairs of dense matrices.

    The eigenvalue problem is \f$ A\mathbf{x} = \lambda B \mathbf{x}\f$, where
    A is symmetric and B is s.p.d. Uses LAPACK.

    \param Ain (IN) This is A.
    \param evals (OUT) The eigenvalues.
    \param evects (OUT) The eigenvectors as columns of a dense matrix.
    \param Bin (IN) This is B.

    \warning A is symmetric and B is s.p.d.
*/
void xpacks_calc_all_gen_eigens_dense(const DenseMatrix& Ain, Vector& evals,
                                      DenseMatrix& evects,
                                      const DenseMatrix& Bin);

/*! \brief Computes the lower eigenvalues and eigenvectors of dense matrices.

    Computes the eigenvalues in (-1,\a upper] and the corresponding
    eigenvectors.
    The eigenvalue problem is \f$ A\mathbf{x} = \lambda B \mathbf{x}\f$, where
    A is symmetric and B is s.p.d. Uses LAPACK.

    TODO: When \a atleast_one is set recomputing is not the most efficient
          approach. Needs improvement which will most likely result in not as
          simple function as we have now.

    \param Ain (IN) This is A.
    \param evals (OUT) The eigenvalues.
    \param evects (OUT) The eigenvectors as columns of a dense matrix.
    \param Bin (IN) This is B.
    \param upper (IN) The upper bound for the eigenvalues.
    \param atleast_one (IN) If set, the function will try to recompute at least
                            one eigenvalue and eigenvector in case all
                            eigenvalues are above \a upper.

    \returns The number of eigenvalues and eigenvectors computed.

    \warning A is symmetric and B is s.p.d.
*/
int xpacks_calc_lower_eigens_dense(const DenseMatrix& Ain, Vector& evals,
                                   DenseMatrix& evects, const DenseMatrix& Bin,
                                   double upper, bool atleast_one/*=1*/);

/*! \brief Computes the upper eigenvalues and eigenvectors of dense matrices.

    Computes the eigenvalues in (\a lower, 2] and the corresponding
    eigenvectors.
    The eigenvalue problem is \f$ A\mathbf{x} = \lambda B \mathbf{x}\f$, where
    A is symmetric and B is s.p.d. Uses LAPACK.

    TODO: When \a atleast_one is set recomputing is not the most efficient
          approach. Needs improvement which will most likely result in not as
          simple function as we have now.

    \param Ain (IN) This is A.
    \param evals (OUT) The eigenvalues.
    \param evects (OUT) The eigenvectors as columns of a dense matrix.
    \param Bin (IN) This is B.
    \param lower (IN) The upper bound for the eigenvalues.
    \param atleast_one (IN) If set, the function will try to recompute at least
                            one eigenvalue and eigenvector in case all
                            eigenvalues are below \a lower.

    \returns The number of eigenvalues and eigenvectors computed.

    \warning A is symmetric and B is s.p.d.
*/
int xpacks_calc_upper_eigens_dense(const DenseMatrix& Ain, Vector& evals,
                                   DenseMatrix& evects, const DenseMatrix& Bin,
                                   double lower, bool atleast_one/*=1*/);

/*! \brief Remove eigenvectors for large eigenvalues (takes the small).

    All eigenvectors with eigenvalues larger than \a bound are
    removed.

    \param evals (IN) The eigenvalues.
    \param evects (IN) The eigenvectors as columns of a dense matrix.
    \param bound (IN) The upper bound for choosing the cut vectors.
    \param cut_evects (OUT) Contains the eigenvectors (as column vectors) with
                            eigenvalues smaller or equal to \a bound.

    \returns The eigenvalue of the first cut vector in case at least one vector
             is cut. Otherwise (no vectors cut), the maximal eigenvalue
             is returned.

    \warning It leaves at least one vector not cut.
*/
double xpack_cut_evects_small(const Vector& evals, const DenseMatrix& evects,
                              double bound, DenseMatrix& cut_evects);

/*! \brief Remove eigenvectors for small eigenvalues (takes the large).

    All eigenvectors with eigenvalues smaller than \a bound are
    removed.

    \param evals (IN) The eigenvalues.
    \param evects (IN) The eigenvectors as columns of a dense matrix.
    \param bound (IN) The lower bound for choosing the cut vectors.
    \param cut_evects (OUT) Contains the eigenvectors (as column vectors) with
                            eigenvalues larger or equal to \a bound.

    \returns The eigenvalue of the first cut vector in case at least one vector
             is cut. Otherwise (no vectors cut), the minimal eigenvalue
             is returned.

    \warning It leaves at least one vector not cut.
*/
double xpack_cut_evects_large(const Vector& evals, const DenseMatrix& evects,
                              double bound, DenseMatrix& cut_evects);


/*! \brief Computes the singular values and left singular vectors.

    Computes the singular values and left singular vectors of an array of dense
    matrices.

    All matrices must have the same height. It simply computes SVD of a matrix
    whose columns are the union of the columns of all matrices.

    \param arr (IN) An array of dense matrices.
    \param arr_size (IN) The number of entries in the array.
    \param lsvects (OUT) The left singular vectors as columns of a dense
                         matrix.
    \param svals (OUT) This singular values.
*/
void xpack_svd_dense_arr(const DenseMatrix *arr, int arr_size,
                         DenseMatrix& lsvects, Vector& svals);

/*! \brief Cuts the left singular vectors with close to zero singular values.

    Thus the resulting set of vectors are orthogonal (linearly independent) and
    span the column space of the matrix they are computed from.
    The removed left singular vectors are those with singular values less or
    equal to \a eps * <the largest singular value>. Uses LAPACK.

    \param lsvects (IN) The left singular vectors as columns of a dense matrix.
    \param svals (IN) This singular values.
    \param orth_set (OUT) The orthogonal set of left singular vectors.
    \param eps (IN) The threshold for removing the left singular vectors with
                    effectively zero singular values.
*/
void xpack_orth_set(const DenseMatrix& lsvects, const Vector& svals,
                    DenseMatrix& orth_set, double eps);

/*! \brief solves least squares problem Ax = b for possibly rectangular A.
 */
void xpack_solve_lls(const DenseMatrix& A, const Vector &rhs, Vector &x);

/*! \brief Solves a linear system using the Cholesky decomposition.

    For s.p.d. dense matrices. Uses LAPACK.

    \param A (IN) The matrix of the system.
    \param rhs (IN) The right-hand side.
    \param x (OUT) The solution.

    \warning \a A is s.p.d.
*/
void xpack_solve_spd_Cholesky(const DenseMatrix& A, const Vector &rhs,
                              Vector &x);

#endif // _XPACKS_HPP
