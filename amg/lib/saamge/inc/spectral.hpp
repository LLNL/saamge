/*! \file
    \brief Solving eigenproblems as part of constructing a prolongator.

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

    XXX Notes:

    It turns out that situations exist where for the same spectral tolerance
    \b spect_simple_local_prob_solve_sparse will accept a bit more vectors than
    \b spect_schur_local_prob_solve_sparse resulting in dramatically different
    convergence rate. This happened when coarsening from level 1 to level 2
    in a 3 level solver (level 0 is the finest) using the general tentative
    prolongator (without refined aggregates). We have level 1 constructed
    using the Schur complements and when constructing level 2 we use the Schur
    complements again but we change the method just for one AE (on level 1).
    Namely in case (I) we use the Schur complement and in case (II) we use
    the simple approach provided by \b spect_simple_local_prob_solve_sparse
    (i.e. the local stiffness matrix itself) but only for this single AE. In
    case (I) we accept 2 vectors from this AE resulting in total 99 DoFs on
    level 2 (the coarsest level), while in case (II) we accept 3 vectors
    resulting in total 100 DoFs on the coarsest level (in all cases the
    spectral tolerance is 0.001). In case (I) the convergence factor is 0.999
    and in case (II) it is a bit above 0.829 which is a dramatic difference and
    it entirely due to this one local vector difference. If we force in case (I)
    3 vectors to be accepted on this single AE resulting again in total of 100
    DoFs on the coarsest level, then the convergence also improves to around
    0.8. That is, it seems these methods really behave very closely but the
    dramatic difference is that this 3rd vector in case (I) has an eigenvalue a
    bit larger than 0.07 (i.e. a tolerance above 0.07 would be required), while
    in case (II) the 3rd vector had an eigenvalue smaller than 5e-7 making it
    acceptable in this weird case with much smaller tolerances. It is well
    verified now and it is NOT due to a bug in
    \b spect_schur_local_prob_solve_sparse.

    TODO: It might be a good idea to add more flexibility allowing the use of
          other right-hand-side matrices in the generalized eigenvalue problems.
          Then, implementation of a procedure determining the maximal
          eigenvalue would be necessary.
*/

#pragma once
#ifndef _SPECTRAL_HPP
#define _SPECTRAL_HPP

#include "common.hpp"
#include <mfem.hpp>
#include "aggregates.hpp"

/* Types */
/*! \brief Solves a local eigenproblem.

    Computes the near-kernel vectors of a local matrix.

    \param A (IN) Usually this is the local stiffness matrix of the AE in local
                  DoF numbering.
    \param B (IN/OUT) If NULL is passed, a smoother is returned here and it
                      must be freed by the caller. Otherwise, the parameter is
                      used as input giving a smoother.
    \param part (IN) The AE on which the local matrix was assembled.
    \param agg_id (IN) The aggregate where the problem is solved .
    \param agg_size (IN) The size of the aggregate.
    \param aggregates (IN) A relation that relates DoFs to aggregates.
    \param agg_part_rels (IN) The structure for mesh relations.
    \param theta (IN) Spectral tolerance. This argument is also used to return
                      a new local theta suggestion.
    \param cut_evects (IN/OUT) These are the near-kernel vectors. When
                               adaptation is being done this is used as an
                               input of the previous vectors from the previous
                               adaptation cycle or the vectors when the
                               hierarchy was initially generated.
    \param Tt (IN) The subspace basis (when adapting). It may be NULL if no
                   transformation is used.
    \param transf (IN) If the eigenproblem has to be solved in a subspace.
    \param all_eigens (IN) If set, all eigenvalues and eigenvectors will be
                           computed and then only the
                           eigenvectors with eigenvalues smaller or equal to
                           \a theta * \em lmax will be taken. If not set
                           then only the needed eigenvectors will be computed.
                           In case no eigenvector fulfills the requirement
                           above, only one vector will be used -- an
                           eigenvector with a minimal eigenvalue.

    \returns \em true if the eigensolve returned more vectors. It makes sense
             only in case consequent adaptation is being done.

    \warning The returned sparse smoother (in \a B) will be freed by the caller.
*/
typedef bool (*spect_local_prob_solve_sparse_ft)(const SparseMatrix& A,
         SparseMatrix *& B, int part, int agg_id, int agg_size,
         const int *aggregates,
         const agg_partititoning_relations_t& agg_part_rels,
         double& theta, DenseMatrix& cut_evects,
         const DenseMatrix *Tt,
         bool transf, bool all_eigens);

/* Functions */
/*! \brief Solves a local eigenproblem on an AE. Non-Schur version.

    Let's consider the generalized eigenvalue problem
    \f$ F\mathbf{x} = \lambda K \mathbf{x}\f$. If \a transf is \em false then
    F is \a A and K is the weighted l1-smoother, \a B. Otherwise,
    F = \a T \a A \a Tt and K = \a T \a B \a Tt, where \a B is the weighted
    l1-smoother. Moreover, if \a transf is \em true, then the resulting
    eigenvectors are transformed by applying \a Tt.

    Only the eigenvectors with eigenvalues smaller or equal to
    \a theta * \em lmax are used for generating the local tentative
    interpolant. \em lmax is the maximal eigenvalue returned when solving the
    eigenvalue problem when no transformation is used. When transformation is
    applied \em lmax is the maximal eigenvalue of the eigenvalue problem before
    the transformation i.e. with F being \a A and K being the weighted
    l1-smoother. Since we use the weighted l1-smoother, we take \em lmax = 1.

    \param A (IN) See the description. Usually this is the local stiffness
                  matrix of the AE in local DoF numbering.
    \param B (IN/OUT) If NULL is passed, the weighted l1-smoother is returned
                      here and it must be freed by the caller. Otherwise, the
                      parameter is used as input giving the weighted
                      l1-smoother.
    \param part (IN) The aggregate and AE on which the local tentative
                     interpolant is being built.
    \param agg_id (IN) Not used.
    \param agg_size (IN) Not used.
    \param aggregates (IN) Not used.
    \param agg_part_rels (IN) Not used.
    \param theta (IN) See the description. This argument is also used to return
                      a new local theta suggestion.
    \param cut_evects (IN/OUT) These are the eigenvectors from the eigenvalue
                               problem in the description with eigenvalues
                               smaller or equal to \a theta * \em lmax. When
                               adaptation is being done this is used as an
                               input of the previous vectors from the previous
                               adaptation cycle or the vectors when the
                               hierarchy was initially generated.
    \param Tt (IN) See the description. It may be NULL if no transformation is
                  used.
    \param transf (IN) See the description.
    \param all_eigens (IN) If set, all eigenvalues and eigenvectors (see the
                           description) will be computed and then only the
                           eigenvectors with eigenvalues smaller or equal to
                           \a theta * \em lmax will be taken. If not set
                           then only the needed eigenvectors will be computed.
                           In case no eigenvector fulfills the requirement
                           above, only one vector will be used -- an
                           eigenvector with a minimal eigenvalue.

    \returns \em true if the eigensolve returned more vectors. It makes sense
             only in case consequent adaptation is being done.

    \warning The returned sparse weighted l1-smoother (in \a B) must be freed
             by the caller.
*/
bool spect_simple_local_prob_solve_sparse(const SparseMatrix& A,
         SparseMatrix *& B, int part, int agg_id, int agg_size,
         const int *aggregates,
         const agg_partititoning_relations_t& agg_part_rels,
         double& theta, DenseMatrix& cut_evects,
         const DenseMatrix *Tt/*=NULL*/,
         bool transf/*=0*/, bool all_eigens/*=0*/);

/*! \brief Augments a subspace matrix for the Schur complement eigenproblem.

    \param Tt (IN) The subspace matrix.
    \param part (IN) The number of the AE.
    \param agg_id (IN) The number of the (refined) aggregate.
    \param agg_size (IN) The size of the (refined) aggregate.
    \param aggregates (IN) Array relating DoFs to (refined) aggregates.
    \param AE_to_dof (IN) Table that relates AEs to DoFs.
    \param augTt (OUT) The augmented subspace matrix.
*/
void spect_schur_augment_transf(const DenseMatrix& Tt, int part, int agg_id,
         int agg_size, const int *aggregates, const Table& AE_to_dof,
         DenseMatrix& augTt);

/*! \brief Solves a local eigenproblem on an aggregate. Schur version.

    See \b spect_simple_local_prob_solve_sparse.

    Only the eigenvectors with eigenvalues smaller or equal to
    \a theta * \em lmax are used for generating the local tentative
    interpolant. Since we use the weighted l1-smoother, we take \em lmax = 1.

    Although the returned \a cut_evects are defined on the entire AE, only
    their restrictions to the aggregate are actually the eigenvectors of the
    Schur complement. Inside the AE but outside the aggregates \a cut_evects
    are "minimal energy" extensions of the respective eigenvectors of the Schur
    complement.

    SVD is NOT needed and NOT advised after restricting the eigenvectors to
    the aggregate.

    The approach here allows avoiding explicit construction of the Schur
    complement on the aggregates. Conversely, we solve sparse problems on
    the full size of the AE. If Schur's complement was to be constructed we
    would need to solve smaller (on the aggregate, not on the generally bigger
    corresponding AE) but dense problems.

    XXX: A potential disadvantage arises when solving in subspace (adapting) is
         performed. While generally this would lead to smaller problems
         (compared to solving in full space), the approach here (at least its
         current implementation) will project to the subspace only the part of
         the matrix corresponding to the available eigenvectors. Since these
         eigenvectors come from a Schur complement on the aggregate, only the
         part of the matrix (which "lives" on the entire AE) corresponding to
         the aggregate will be shrunk. The part of the matrix corresponding to
         DoFs in the AE but outside the aggregate will not be shrunk but rather
         will be kept the same. Namely, the DoFs outside the aggregate will be
         transported to the subspace where we solve. This reduces the speed
         gain from solving in subspace. Sometimes on coarse levels the number
         of DoFs in the AE but outside the aggregate is pretty big, effectively
         keeping the size of the eigenproblem also big.

    \param A (IN) Usually this is the local stiffness matrix of the AE in local
                  DoF numbering.
    \param B (IN/OUT) If NULL is passed, the restricted weighted l1-smoother is
                      returned here and it must be freed by the caller.
                      Otherwise, the parameter is used as input giving the
                      restricted weighted l1-smoother.
    \param part (IN) The AE on which the local matrix was assembled.
    \param agg_id (IN) The aggregate for the Schur complement.
    \param agg_size (IN) The size of the aggregate.
    \param aggregates (IN) A relation that relates DoFs to aggregates.
    \param agg_part_rels (IN) The structure for mesh relations.
    \param theta (IN) See the description. This argument is also used to return
                      a new local theta suggestion.
    \param cut_evects (IN/OUT) These are the eigenvectors from the eigenvalue
                               problem in the description with eigenvalues
                               smaller or equal to \a theta * \em lmax. When
                               adaptation is being done this is used as an
                               input of the previous vectors from the previous
                               adaptation cycle or the vectors when the
                               hierarchy was initially generated.
    \param Tt (IN) The subspace basis (when adapting). It may be NULL if no
                   transformation is used.
    \param transf (IN) If the eigenproblem has to be solved in a subspace.
    \param all_eigens (IN) If set, all eigenvalues and eigenvectors (see the
                           description) will be computed and then only the
                           eigenvectors with eigenvalues smaller or equal to
                           \a theta * \em lmax will be taken. If not set
                           then only the needed eigenvectors will be computed.
                           In case no eigenvector fulfills the requirement
                           above, only one vector will be used -- an
                           eigenvector with a minimal eigenvalue.

    \returns \em true if the eigensolve returned more vectors. It makes sense
             only in case consequent adaptation is being done.

    \warning The returned sparse restricted weighted l1-smoother (in \a B) must
             be freed by the caller.
*/
bool spect_schur_local_prob_solve_sparse(const SparseMatrix& A,
         SparseMatrix *& B, int part, int agg_id, int agg_size,
         const int *aggregates,
         const agg_partititoning_relations_t& agg_part_rels,
         double& theta, DenseMatrix& cut_evects,
         const DenseMatrix *Tt/*=NULL*/,
         bool transf/*=0*/, bool all_eigens/*=0*/);

#endif // _SPECTRAL_HPP
