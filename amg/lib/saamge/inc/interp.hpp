/*! \file
    \brief Interpolant related functionality.

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

#pragma once
#ifndef _INTERP_HPP
#define _INTERP_HPP

#include "common.hpp"
#include <mfem.hpp>
#include "contrib.hpp"
#include "mbox.hpp"
#include "aggregates.hpp"
#include "spectral.hpp"
#include "smpr.hpp"
#include "elmat.hpp"

/* Types */
/*! \brief Parameters and data for the interpolant.
*/
typedef struct {
    int nparts; /*!< The number of AEs and big aggregates. */
    SparseMatrix **rhs_matrices_arr; /*!< See \b interp_sparse_tent_build. The
                                          B matrices in the eigenvalue problems
                                          \f$ A\mathbf{x} = \lambda B \mathbf{x}\f$
                                          for all AEs. */
    DenseMatrix **cut_evects_arr; /*!< See \b interp_sparse_tent_build. These
                                       are the \em cut_evects used for building
                                       the current hierarchy. */
    SparseMatrix **AEs_stiffm; /*!< See \b interp_sparse_tent_build. Here the
                                    local stiffness matrices are saved and if
                                    necessary reused. */

    agg_elmat_callback_ft finest_elmat_callback; /*!< What element callback
                                                      will be used for the
                                                      finest (usually
                                                      geometric) level when
                                                      assembling local (AE)
                                                      stiffness matrices. */

    int nu_interp; /*!< Nu for the interpolant smoother. */
    int interp_smoother_degree; /*!< The degree of the polynomial of the
                                     interpolant smoother. */
    double *interp_smoother_roots; /*!< The roots of the polynomial of the
                                        interpolant smoother. */
    int times_apply_smoother; /*!< How many times the prolongator smoother is
                                   to be applied. */

    spect_local_prob_solve_sparse_ft local_prob_solve; /*!< The method used
                                                            for finding
                                                            near-kernel
                                                            vectors. */

    contrib_agg_ft contrib_agg; /*!< The function distributing vectors among
                                     (big or refined) aggregates. */

    double eps_svd; /*!< A parameter of \a contrib_agg. Usually SVD
                         tolerance. */
    double eps_lin; /*!< This is the tolerance when orthogonalizing and
                         checking for linear dependence during adaptation. */

    Array<int> tent_interp_offsets; /*!< Offsets of the columns of the local
                                         (within the process) tentative
                                         prolongator w.r.t. the global
                                         numbering across all processes. */
} interp_data_t;

/* Options */

/*! \brief The configuration class of this module.

    \warning Have in mind that some of the default options' values are set to
             suit the two-level case. The multilevel case may uses its own
             options.
*/
CONFIG_BEGIN_CLASS_DECLARATION(INTERP)

    /*! The method used for finding near-kernel vectors.

        \warning This parameter is used during the construction of objects
                 (structure instances) so if modified it will only have effect
                 for new objects and will NOT modify the behavior of existing
                 ones. For altering the option for existing objects look at the
                 corresponding fields in the respective structure(s). */
    CONFIG_DECLARE_OPTION(spect_local_prob_solve_sparse_ft, local_prob_solve);

    /*! The function distributing vectors among (big or refined) aggregates.

        \warning This parameter is used during the construction of objects
                 (structure instances) so if modified it will only have effect
                 for new objects and will NOT modify the behavior of existing
                 ones. For altering the option for existing objects look at the
                 corresponding fields in the respective structure(s). */
    CONFIG_DECLARE_OPTION(contrib_agg_ft, contrib_agg);

    /*! A parameter of \b interp_data_t::contrib_agg. Usually SVD tolerance.

        \warning This parameter is used during the construction of objects
                 (structure instances) so if modified it will only have effect
                 for new objects and will NOT modify the behavior of existing
                 ones. For altering the option for existing objects look at the
                 corresponding fields in the respective structure(s). */
    CONFIG_DECLARE_OPTION(double, eps_svd);

    /*! This is the tolerance when orthogonalizing and checking for linear
        dependence during adaptation.

        \warning This parameter is used during the construction of objects
                 (structure instances) so if modified it will only have effect
                 for new objects and will NOT modify the behavior of existing
                 ones. For altering the option for existing objects look at the
                 corresponding fields in the respective structure(s). */
    CONFIG_DECLARE_OPTION(double, eps_lin);

    /*! Determines what roots (and thus what polynomial) to be used for
        smoothing the tentative prolongator.

        \warning This parameter is used during the construction of objects
                 (structure instances) so if modified it will only have effect
                 for new objects and will NOT modify the behavior of existing
                 ones. For altering the option for existing objects look at the
                 corresponding fields in the respective structure(s). */
    CONFIG_DECLARE_OPTION(smpr_roots_ft, smoother_roots);

    /*! How many times the prolongator smoother is to be applied.

        \warning This parameter is used during the construction of objects
                 (structure instances) so if modified it will only have effect
                 for new objects and will NOT modify the behavior of existing
                 ones. For altering the option for existing objects look at the
                 corresponding fields in the respective structure(s). */
    CONFIG_DECLARE_OPTION(unsigned int, times_apply_smoother);

    /*! Whether the essential boundary conditions are imposed on the global
        stiffness matrix on the finest (geometric) level. */
    CONFIG_DECLARE_OPTION(bool, bdr_cond_imposed);

    /*! Causes the diagonal elements corresponding to DoFs lying simultaneously
        on AEs' interfaces and the essential part of the boundary to be
        assembled (while building the local (AE) stiffness matrices) instead of
        copied from the global stiffness matrix. It is only useful when
        \b bdr_cond_imposed is \em true. It is meaningful in the cases when the
        global matrix has its diagonal entries kept, instead of set to 1,
        during the global imposition of essential boundary conditions. */
    CONFIG_DECLARE_OPTION(bool, assemble_ess_diag);

    /*! What element callback will be used for the finest (usually geometric)
        level when assembling local (AE) stiffness matrices.

        \warning This parameter is used during the construction of objects
                 (structure instances) so if modified it will only have effect
                 for new objects and will NOT modify the behavior of existing
                 ones. For altering the option for existing objects look at the
                 corresponding fields in the respective structure(s). */
    CONFIG_DECLARE_OPTION(agg_elmat_callback_ft, finest_elmat_callback);

CONFIG_END_CLASS_DECLARATION(INTERP)

CONFIG_BEGIN_INLINE_CLASS_DEFAULTS(INTERP)
    CONFIG_DEFINE_OPTION_DEFAULT(local_prob_solve,
                                 spect_schur_local_prob_solve_sparse),
    CONFIG_DEFINE_OPTION_DEFAULT(contrib_agg, contrib_big_aggs_nosvd),
    CONFIG_DEFINE_OPTION_DEFAULT(eps_svd, 1e-10),
    CONFIG_DEFINE_OPTION_DEFAULT(eps_lin, 1e-12),
    CONFIG_DEFINE_OPTION_DEFAULT(smoother_roots, smpr_sa_poly_roots),
    CONFIG_DEFINE_OPTION_DEFAULT(times_apply_smoother, 1),
    CONFIG_DEFINE_OPTION_DEFAULT(bdr_cond_imposed, true),
    CONFIG_DEFINE_OPTION_DEFAULT(assemble_ess_diag, true),
    CONFIG_DEFINE_OPTION_DEFAULT(finest_elmat_callback,
                                 elmat_standard_geometric_dense)
CONFIG_END_CLASS_DEFAULTS

/* Functions */
/*! \brief Smooths the tentative interpolant producing the final interpolant.

    If S is \f$ -D^{-1}A \f$ and \f$ \tau_i, i = 0, \dots, \text{degree}-1 \f$
    is the contents of \a roots, then this function returns
    \f$ \prod_i \left( I + \frac{1}{\tau_i}S \right) P \f$, where P is the
    tentative interpolant.

    \param degree (IN) The degree of the polynomial smoother.
    \param roots (IN) The roots of the polynomial smoother.
    \param times_apply_smoother (IN) How many times the prolongator smoother is
                                     to be applied.
    \param A (IN) The global (among all processes) stiffness matrix.
    \param tent (IN) The tentative interpolant.
    \param Dinv_neg (IN) A diagonal that is precisely \f$ -D^{-1} \f$.

    \returns The final (actual) interpolant.

    \warning The returned matrix must be freed by the caller.
    \warning Usually D is the weighted l1-smoother.
*/
HypreParMatrix *interp_smooth(int degree, double *roots,
                              int times_apply_smoother, HypreParMatrix& A,
                              HypreParMatrix& tent, HypreParVector& Dinv_neg);

/*! \brief Initializes interpolant data.

    \param agg_part_rels (IN) The partitioning relations.
    \param nu (IN) The degree of the polynomial smoother if the tentative
                   interpolant will be smoothed by a polynomial.

    \returns A structure with the interpolant data.

    \warning The returned structure must be freed by the caller using
             \b interp_free_data.
    \warning It uses the options \b contrib_agg to initialize the method to be
             used for distributing vectors to aggregates.
    \warning It uses the option \b local_prob_solve the method to be used for
             computing near-kernel vectors.
    \warning It uses the options \b eps_svd and \b eps_svd to initialize to
             tolerances in the returned structure instance.
    \warning It uses the option \b smoother_roots to compute the roots of the
             prolongator smoothing polynomial.
    \warning It uses the option \b times_apply_smoother to initialize the
             number of time the prolongator smoother will be applied every time
             it is used for smoothing the tentative prolongator.
    \warning It uses the options \b finest_elmat_callback.
*/
interp_data_t *interp_init_data(
    const agg_partititoning_relations_t& agg_part_rels, double nu);

/*! \brief Frees interpolant data.

    \param interp_data (IN) To be freed.
*/
void interp_free_data(interp_data_t *interp_data);

/*! \brief Copies interpolant data.

    \param src (IN) To be copied.

    \returns A copy.

    \warning The returned structure must be freed by the caller using
             \b interp_free_data.
*/
interp_data_t *interp_copy_data(const interp_data_t *src);

/*! \brief Computes the vectors used for building the tentative interpolant.

    Computes the vectors used later for building the tentative interpolant.

    When adaptation is used then \a transf must be set. In that case the
    transformation matrices \a T and \a Tt  for the i-th AE are produced using
    the vectors in \a interp_data_t::cut_evects_arr[i] and the orthonormalized
    (using \a interp_data_t::rhs_matrices_arr[i]) restriction of \a xbad.

    When no transformation is used, \a interp_data_t::cut_evects_arr gets
    filled with the new \em cut_evects. Otherwise (when transformation is used),
    \a interp_data_t::cut_evects_arr contains the former \em cut_evects as
    input and they are overwritten by the new ones.
    \a interp_data_t::cut_evects_arr must always have as many elements as the
    number of AEs and, in case no transformation is used, the elements must be
    NULL and also freed by the caller later using \b mbox_free_matr_arr since
    they will be allocated in this function.

    Similarly, \a interp_data_t::rhs_matrices_arr must always have as many
    elements as the number of AEs and, in case no transformation is used, the
    elements must be NULL and also freed by the caller later using
    \b mbox_free_matr_arr since they will be allocated in this function. When
    no transformation is used, \a interp_data_t::rhs_matrices_arr gets filled
    with the new r.h.s. matrices for the eigenvalue problems. Otherwise (when
    transformation is used), \a interp_data_t::rhs_matrices_arr contains the
    former r.h.s. matrices as input and they are overwritten by the new ones.

    Also, \a interp_data_t::AEs_stiffm must always have as many elements as the
    number of AEs and, in case \a readapting is \em true, the local local
    stiffness matrices in this array are used, otherwise they are computed and
    overwritten. In case the hierarchy is being built from scratch, the
    elements of \a interp_data_t::AEs_stiffm must be NULL.

    \param A (IN) The global stiffness matrix (with imposed boundary
                  conditions if on geometric level).
    \param agg_part_rels (IN) The partitioning relations.
    \param interp_data (IN/OUT) Parameters and data for the interpolant.
    \param elem_data_finest (IN) Data corresponding to the finest element
                                 matrices callback
                                 \a interp_data.finest_elmat_callback. If the
                                 current level is not the finest this MUST be
                                 NULL to invoke coarse level construction.
                                 Usually this is a MFEM \em BilinearForm used
                                 for assembling \a A (on a geometric level).
    \param tol (IN/OUT) The tolerance for adding vectors when
                        \a spect_update = \em false. A suggestion for future
                        tolerance is returned here.
    \param theta (IN/OUT) Spectral tolerance. It is also
                          used for returning suggestions for future theta.
    \param xbad_lin_indep (OUT) \em false if \a xbad is linearly dependant
                                on all AEs. It may be NULL if no transformation
                                is used.
    \param vector_added (OUT) \em true if at least one vector was added locally
                              somewhere. It may be NULL if no transformation is
                              used.
    \param xbad (IN) See the description. It may be NULL if no transformation
                     is used.
    \param transf (IN) See the description.
    \param readapting (IN) Can be \em true only when \a transf is \em true. It
                           determines whether a readaptation is being done, i.e.
                           adaptation for an unchanged matrix.
    \param all_eigens (IN) Whether to compute all eigenvectors or just
                           the desired part of them.
    \param spect_update (IN) If \a spect_update = \em true then we proceed as
                             usual. That is, coarse basis vectors are found in
                             the near kernel of T*A*Tt subject to a spectral
                             threshold. If \a spect_update = \em false, then
                             the columns of Tt are the new coarse basis and
                             should be input into the tentative prolongator
                             without further modification. It makes sense to be
                             \em false only when readaptation is being done
                             (i.e. when adapting for an unchanged matrix).

    \warning \a rhs_matrices_arr and \a cut_evects_arr must be freed by the
             caller (see the description).
    \warning See the warning in \b agg_build_AE_stiffm_with_global related to
             the zero elements in \a A.
    \warning It uses the option \b bdr_cond_imposed to determine whether the
             essential boundary conditions are imposed on the global stiffness
             matrix on the finest (geometric) level (see
             \b agg_build_AE_stiffm_with_global).
    \warning \a A and \a elem_data_* must correspond to each other.
*/
void interp_compute_vectors(const SparseMatrix& A,
    const agg_partititoning_relations_t& agg_part_rels,
    const interp_data_t& interp_data, void *elem_data_finest, double& tol,
    double& theta, bool *xbad_lin_indep/*=NULL*/, bool *vector_added/*=NULL*/, const Vector *xbad/*=NULL*/,
    bool transf/*=0*/, bool readapting/*=0*/, bool all_eigens/*=0*/, bool spect_update/*=1*/);

/*! \brief Generates the tentative interpolant.

    \param A (IN) The global stiffness matrix (with imposed boundary
                  conditions if on geometric level).
    \param agg_part_rels (IN) The partitioning relations.
    \param interp_data (IN) Parameters and data for the interpolant.
    \param elem_data_finest (IN/OUT) See \b interp_compute_vectors.
    \param tol (IN/OUT) See \b interp_compute_vectors.
    \param theta (IN/OUT) See \b interp_compute_vectors.
    \param xbad_lin_indep (OUT) See \b interp_compute_vectors.
    \param vector_added (OUT) See \b interp_compute_vectors.
    \param xbad (IN) See \b interp_compute_vectors.
    \param transf (IN) See \b interp_compute_vectors.
    \param readapting (IN) See \b interp_compute_vectors.
    \param all_eigens (IN) Whether to compute all eigenvectors or just
                           the desired part of them.
    \param spect_update (IN) See \b interp_compute_vectors.

    \returns The tentative interpolant.

    \warning The returned sparse matrix must be freed by the caller.
    \warning See the warning in \b agg_build_AE_stiffm_with_global related to
             the zero elements in \a A.
    \warning \a A and \a elem_data_* must correspond to each other.
*/
SparseMatrix *interp_sparse_tent_build(const SparseMatrix& A,
    const agg_partititoning_relations_t& agg_part_rels,
    const interp_data_t& interp_data, void *elem_data_finest, double& tol,
    double& theta, bool *xbad_lin_indep/*=NULL*/, bool *vector_added/*=NULL*/, const Vector *xbad/*=NULL*/,
    bool transf/*=0*/, bool readapting/*=0*/, bool all_eigens/*=0*/, bool spect_update/*=1*/);

/*! \brief Simply (re)assembles the tentative interpolant.

    No eigenvalue problems are solved. The eigenvectors must be already there
    from a previous interpolant computation or simply precomputed by a call to
    \b interp_compute_vectors.

    This function is useful, e.g., if the aggregates get modified and we simply
    reconstruct the tentative interpolant with the already present eigenvectors.

    \param agg_part_rels (IN) The partitioning relations.
    \param interp_data (IN) Parameters and data for the interpolant.

    \returns The tentative interpolant.

    \warning The returned sparse matrix must be freed by the caller.
*/
SparseMatrix *interp_sparse_tent_assemble(
     const agg_partititoning_relations_t& agg_part_rels,
     const interp_data_t& interp_data);

/*! \brief Assembles the global parallel tentative interpolant.

    \param agg_part_rels (IN) The partitioning relations.
    \param interp_data (IN) Parameters and data for the interpolant.
    \param local_tent_interp (IN) The assembled local portion (on the process)
                                  of the tentative interpolant.

    \returns The global parallel tentative interpolant.

    \warning The returned parallel sparse matrix must be freed by the caller.
*/
HypreParMatrix *interp_global_tent_assemble(
     const agg_partititoning_relations_t& agg_part_rels,
     interp_data_t& interp_data, SparseMatrix *local_tent_interp);

/* Inline Functions */
/*! \brief Smooths the tentative interpolant to produce the final one.

    \param A (IN) The global (among all processes) stiffness matrix.
    \param interp_data (IN) Parameters and data for the interpolant.
    \param tent_interp (IN) The tentative interpolant.
    \param Dinv_neg (IN) A diagonal that is precisely \f$ -D^{-1} \f$.

    \returns The final smoothed prolongator.

    \warning The returned matrix must be freed by the caller.
    \warning Usually D is the weighted l1-smoother.
*/
static inline
HypreParMatrix *interp_smooth_interp(HypreParMatrix& A,
                                     const interp_data_t& interp_data,
                                     HypreParMatrix& tent_interp,
                                     HypreParVector& Dinv_neg);

/* Inline Functions Definitions */
static inline
HypreParMatrix *interp_smooth_interp(HypreParMatrix& A,
                                     const interp_data_t& interp_data,
                                     HypreParMatrix& tent_interp,
                                     HypreParVector& Dinv_neg)
{
    return interp_smooth(interp_data.interp_smoother_degree,
                         interp_data.interp_smoother_roots,
                         interp_data.times_apply_smoother, A, tent_interp,
                         Dinv_neg);
}

#endif // _INTERP_HPP
