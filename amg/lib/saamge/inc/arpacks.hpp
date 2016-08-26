/*! \file
    \brief Functions that use ARPACK (ARPACK++) for eigenpair problems.

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
#ifndef _ARPACKS_HPP
#define _ARPACKS_HPP

#include "common.hpp"
#include <mfem.hpp>

/* Functions */
/*! \brief Computes a part of eigenvalues and eigenvectors of sparse matrices.

    Depending on the choice of \a lower it computes either the lower or the
    upper portion of the spectrum.

    The eigenvalue problem is \f$ A\mathbf{x} = \lambda B \mathbf{x}\f$, where
    A is sparse symmetric and B is s.p.d. and DIAGONAL. Uses ARPACK (ARPACK++).

    All parameters are generally set to default and a few are accessible.

    \param Ain (IN) This is A.
    \param evals (OUT) The eigenvalues.
    \param evects (OUT) The eigenvectors as columns of a dense matrix.
    \param Bin (IN) This is B.
    \param num_evects (IN) The desired number of eigenvalues and eigenvectors.
    \param lower (IN) If \em true, then it computes the lower portion of the
                      spectrum. Otherwise, if \em false, it computes the upper
                      portion of the spectrum. Default is \em true.
    \param max_iters (IN) The maximal number of iteration performed by ARPACK.
                          Value of 0 means to use some default value in
                          ARPACK++.
    \param ncv (IN) The number of Arnoldi vectors on each iteration in ARPACK.
                    Value of 0 means to use some default value in ARPACK++.
    \param tol (IN) The tolerance in ARPACK. Value of 0.0 means to use some
                    default value in ARPACK++.

    \returns The number of eigenvalues and eigenvectors computed.

    \warning A is symmetric and B is s.p.d and DIAGONAL.
*/
int arpacks_calc_portion_eigens_sparse_diag(const mfem::SparseMatrix& Ain,
                                            mfem::Vector& evals,
                                            mfem::DenseMatrix& evects,
                                            const mfem::SparseMatrix& Bin,
                                            int num_evects, bool lower=true,
                                            int max_iters=0, int ncv=0,
                                            int tol=0.);

/*! \brief Computes a part of eigenvalues and eigenvectors of sparse matrices.

    Depending on the choice of \a lower it computes either the lower or the
    upper portion of the spectrum.

    The eigenvalue problem is \f$ A\mathbf{x} = \lambda B \mathbf{x}\f$, where
    A is sparse symmetric and B is sparse s.p.d. Uses ARPACK (ARPACK++).

    All parameters are generally set to default and a few are accessible.

    \param Ain (IN) This is A.
    \param evals (OUT) The eigenvalues.
    \param evects (OUT) The eigenvectors as columns of a dense matrix.
    \param Bin (IN) This is B.
    \param num_evects (IN) The desired number of eigenvalues and eigenvectors.
    \param lower (IN) If \em true, then it computes the lower portion of the
                      spectrum. Otherwise, if \em false, it computes the upper
                      portion of the spectrum. Default is \em true.
    \param max_iters (IN) The maximal number of iteration performed by ARPACK.
                          Value of 0 means to use some default value in
                          ARPACK++.
    \param ncv (IN) The number of Arnoldi vectors on each iteration in ARPACK.
                    Value of 0 means to use some default value in ARPACK++.
    \param tol (IN) The tolerance in ARPACK. Value of 0.0 means to use some
                    default value in ARPACK++.

    \returns The number of eigenvalues and eigenvectors computed.

    \warning A is symmetric and B is s.p.d.
*/
/*
int arpacks_calc_portion_eigens_sparse(const SparseMatrix& Ain, Vector& evals,
                                       DenseMatrix& evects,
                                       const SparseMatrix& Bin, int num_evects,
                                       bool lower=true, int max_iters=0,
                                       int ncv=0, int tol=0.);
*/

#endif // _ARPACKS_HPP
