/*! \file
    \brief Interpolant related functionality.

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

#pragma once
#ifndef _INTERP_HPP
#define _INTERP_HPP

#include "common.hpp"
#include <mfem.hpp>
#include "contrib.hpp"
#include "mbox.hpp"
#include "aggregates.hpp"
#include "spectral.hpp"

namespace saamge
{

// forward declaration
class ElementMatrixProvider;

/* Types */
/*! \brief Parameters and data for the interpolant.
*/
typedef struct {
    int nparts; /*!< The number of AEs and big aggregates. */
    mfem::SparseMatrix **rhs_matrices_arr; /*!< See \b interp_sparse_tent_build. The
                                          B matrices in the eigenvalue problems
                                          \f$ A\mathbf{x} = \lambda B \mathbf{x}\f$
                                          for all AEs. */
    /** See \b interp_sparse_tent_build. These are the \em cut_evects used for
        building the current hierarchy. */
    mfem::DenseMatrix **cut_evects_arr;
    /** See \b interp_sparse_tent_build. Here the local stiffness matrices are
        saved and if necessary reused. */
    mfem::SparseMatrix **AEs_stiffm;

    /** Some matrices and bases for the nonconforming method. */
    mfem::Matrix **Aii;
    mfem::Matrix **Abb;
    mfem::Matrix **Aib;
    mfem::DenseMatrix **invAii;
    mfem::DenseMatrix **invAiiAib;
    mfem::DenseMatrix **AbiinvAii;
    mfem::DenseMatrix **schurs;
    mfem::DenseMatrix **cfaces_bases;
    int num_cfaces;

    /** Some information to map the coarse DoFs for the first refinement on a processor in the nonconforming method.
        Particularly, celements_cdofs is the number of coarse DoFs associated with all
        coarse (agglomerated) elements on the processor. The coarse DoFs on the processor
        are ordered in such a way that all DoFs associated with elements come first and
        ordered according to the elements' order and their local (within an element) ordering.
        Then, all coarse faces' DoFs come also ordered according to the cfaces' order and
        the local (within a face) ordering. This principle would hold for both (local) DoFs and
        true DoFs. In fact, only cface DoFs can be shared and make sense to be seen as local and true
        DoFs, while element DoFs are always true DoFs. In particular, true DoFs follow the ordering principle but
        include only cfaces owned by the processor, while local DoFs include all cfaces known to the processor.
        This is the ordering of the columns in the "prolongator" and the rows in the "restriction".
        Here, we have the offsets of the cDoFs for every single coarse element and the offsets (starting with zero, i.e. the
        actual offset is obtained by adding celements_cdofs) of both local and true cDoFs for the faces. If a cface is not owned,
        its true offset is -1. */
    int celements_cdofs;
    int *celements_cdofs_offsets;
    int *cfaces_truecdofs_offsets;
    int *cfaces_cdofs_offsets;

    int nu_pro; /*!< Nu for the interpolant smoother. */
    int interp_smoother_degree; /*!< The degree of the polynomial of the
                                     interpolant smoother. */
    double *interp_smoother_roots; /*!< The roots of the polynomial of the
                                        interpolant smoother. */
    int times_apply_smoother; /*!< How many times the prolongator smoother is
                                   to be applied. */

    mfem::Array<int> tent_interp_offsets; /*!< Offsets of the columns of the local
                                         (within the process) tentative
                                         prolongator w.r.t. the global
                                         numbering across all processes. */
    int total_cols_interp; /*!< Total number of columns in the global interpolant. */

    mfem::Array<double> * local_coarse_one_representation; /*! ATB building coarse_one_representation on the fly */

    bool use_arpack; /*! ATB whether to use ARPACK or do direct eigensolver */
    bool scaling_P; /*! ATB whether or not to do the coarse_one_representation, scaling_P business */

    int * mis_numcoarsedof; /*!< How many true coarse DoFs a MIS contributed, i.e., how many columns of the interpolation matrix. */

    int coarse_truedof_offset; /*!< TODO this may be the same as tent_interp_offsets... (Yes, it is for the MIS method, but not for the non-conforming. DZK)
                                    In the non-conforming method, this is the offset of the coarse faces' (true) coarse DoFs (i.e., ignoring the coarse elements' coarse DoFs)
                                    for the current processor in the global true coarse DoF numbering, restricted to coarse faces. */
    int num_mises;
    /** tentative interpolants localized to each MIS,
        (a) a waste of memory if we also have tent_interp and
        (b) maybe belongs in interp_data_t or contrib_ or something? */
    mfem::DenseMatrix ** mis_tent_interps;
    /**
       After smoothing interpolation matrix, we drop entries from P that are
       smaller than this. This has no theoretical foundation but has been shown
       to have moderate effect on operator complexity with very little effect
       on accuracy, 1.e-3 is a typical number.
    */
    double drop_tol;
} interp_data_t;

/* Options */

const int ARPACK_SIZE_THRESHOLD = 64;

const double INTERP_LINEAR_TOLERANCE = 1.e-12;

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
mfem::HypreParMatrix *interp_smooth(
    int degree, double *roots, double drop_tol, int times_apply_smoother,
    mfem::HypreParMatrix& A, mfem::HypreParMatrix& tent,
    mfem::HypreParVector& Dinv_neg);

/*! \brief Initializes interpolant data.

    \param agg_part_rels (IN) The partitioning relations.
    \param nu (IN) The degree of the polynomial smoother if the tentative
                   interpolant will be smoothed by a polynomial.
    \param scaling_P whether or not to build the coarse_ones_representation, scaling_P business

    \returns A structure with the interpolant data.

    \warning The returned structure must be freed by the caller using
             \b interp_free_data.
    \warning It uses the options \b contrib_agg to initialize the method to be
             used for distributing vectors to aggregates.
    \warning It uses the option \b local_prob_solve the method to be used for
             computing near-kernel vectors.
    \warning It uses the option \b times_apply_smoother to initialize the
             number of time the prolongator smoother will be applied every time
             it is used for smoothing the tentative prolongator.
    \warning It uses the options \b finest_elmat_callback.
*/
interp_data_t *interp_init_data(
    const agg_partitioning_relations_t& agg_part_rels, int nu_pro, 
    bool use_arpack, bool scaling_P = false);

/*! \brief Frees interpolant data.

    \param interp_data (IN) To be freed.
*/
void interp_free_data(interp_data_t *interp_data,
                      bool doing_spectral, double theta);

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

    \param agg_part_rels (IN) The partitioning relations.
    \param interp_data (IN/OUT) Parameters and data for the interpolant.
    \param elem_data_finest (IN) Data corresponding to the finest element
                                 matrices callback
                                 \a interp_data.finest_elmat_callback. If the
                                 current level is not the finest this MUST be
                                 NULL to invoke coarse level construction.
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
*/
void interp_compute_vectors(
    const agg_partitioning_relations_t& agg_part_rels,
    const interp_data_t& interp_data, ElementMatrixProvider *elem_data,
    double tol, double& theta, bool *xbad_lin_indep, bool *vector_added,
    const mfem::Vector *xbad, bool transf, bool readapting,
    bool all_eigens, bool spect_update, bool bdr_cond_imposed);

/*! \brief A simple version of \a interp_compute_vectors() that does not store all matrix information.

    Since it stores no local matrix information it is not suitable for adaptivity and, therefore, it provides no such
    functionality. It serves only for building a hierarchy from scratch that will not be subject to any adaptation.

    The main purpose of this function is to be used in the nonconforming hierarchies. Therefore, it fills in
    interp_data.rhs_matrices_arr with the diagonals of the respective local (on AE) stiffness matrices.

    Returns an array of the eigenvalues computed on all AEs. Must be freed by the caller.
*/
mfem::Vector **interp_compute_vectors_nostore(const agg_partitioning_relations_t& agg_part_rels,
    const interp_data_t& interp_data, ElementMatrixProvider *elem_data, double theta, bool full_space=false);

/**
   Build composite coarse space with both spectral and polynomial
   degrees of freedom.
*/
mfem::SparseMatrix * interp_build_composite(
    const agg_partitioning_relations_t &agg_part_rels,
    interp_data_t& interp_data, ElementMatrixProvider *elem_data, double theta,
    int spatial_dimension, int num_nodes, const mfem::Vector& coords, int order,
    bool avoid_ess_bdr_dofs, bool use_spectral);

/**
   Build interpolation based on polynomials and geometric information, rather
   than eigenvalue problems.

   Should be a more-or-less drop in replacement for interp_sparse_tent_build()

   Only implemented for order == 0 and order == 1.
*/
mfem::SparseMatrix *interp_build_polynomial(
    const agg_partitioning_relations_t &agg_part_rels,
    interp_data_t& interp_data, int spatial_dimension,
    int num_nodes, const mfem::Vector& coords, int order,
    bool avoid_ess_bdr_dofs);

/**
   Wrapper for interp_build_polynomial with order = 0,
   for backwards compatibility / simplicity.
*/
mfem::SparseMatrix *interp_build_minimal(
    const agg_partitioning_relations_t &agg_part_rels,
    interp_data_t& interp_data);

/*! \brief Generates the tentative interpolant.

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
    \param avoid_ess_bdr_dofs (IN) determines how interpolant handles essential boundary dofs, see contrib.cpp

    \returns The tentative interpolant.

    \warning The returned sparse matrix must be freed by the caller.
    \warning See the warning in \b agg_build_AE_stiffm_with_global related to
             the zero elements in \a A.
*/
mfem::SparseMatrix *interp_sparse_tent_build(
    const agg_partitioning_relations_t& agg_part_rels,
    interp_data_t& interp_data, 
    ElementMatrixProvider * elem_data, double& tol,
    double& theta, bool *xbad_lin_indep/*=NULL*/, bool *vector_added/*=NULL*/,
    const mfem::Vector *xbad/*=NULL*/, bool transf/*=0*/, bool readapting/*=0*/,
    bool all_eigens/*=0*/, bool spect_update/*=1*/, bool avoid_ess_bdr_dofs);

/*! \brief Simply (re)assembles the tentative interpolant.

    No eigenvalue problems are solved. The eigenvectors must be already there
    from a previous interpolant computation or simply precomputed by a call to
    \b interp_compute_vectors.

    This function is useful, e.g., if the aggregates get modified and we simply
    reconstruct the tentative interpolant with the already present eigenvectors.

    \param agg_part_rels (IN) The partitioning relations.
    \param interp_data (IN) Parameters and data for the interpolant.
    \param avoid_ess_bdr_dofs (IN) Determines how interpolant handles essential boundary dofs.

    \returns The tentative interpolant.

    \warning The returned sparse matrix must be freed by the caller.
*/
mfem::SparseMatrix *interp_sparse_tent_assemble(
     const agg_partitioning_relations_t& agg_part_rels,
     interp_data_t& interp_data, bool avoid_ess_bdr_dofs);

/*! \brief Assembles the global parallel tentative interpolant.

    \param agg_part_rels (IN) The partitioning relations.
    \param interp_data (IN) Parameters and data for the interpolant.
    \param local_tent_interp (IN) The assembled local portion (on the process)
                                  of the tentative interpolant.

    \returns The global parallel tentative interpolant.

    \warning The returned parallel sparse matrix must be freed by the caller.
*/
mfem::HypreParMatrix *interp_global_tent_assemble(
     const agg_partitioning_relations_t& agg_part_rels,
     interp_data_t& interp_data, mfem::SparseMatrix *local_tent_interp);

/*! \brief Similar to \b interp_global_tent_assemble but for the restriction.
*/
mfem::HypreParMatrix *interp_global_restr_assemble(
     const agg_partitioning_relations_t& agg_part_rels,
     interp_data_t& interp_data, mfem::SparseMatrix *local_tent_restr);

/*! \brief assembles global coarse representation of ones (ie, near null space)

  DEPRECATED

  added ATB 7 October 2014
*/
mfem::HypreParVector *interp_global_coarse_one_assemble(
    const agg_partitioning_relations_t& agg_part_rels,
    interp_data_t& interp_data, mfem::SparseMatrix *local_tent_interp,
    mfem::Array<double> * local_coarse_one_representation);

/*! \brief assembles scaling P 

  added ATB 20 February 2015
*/
mfem::HypreParMatrix * interp_scaling_P_assemble(
    const agg_partitioning_relations_t& agg_part_rels,
    interp_data_t& interp_data, mfem::SparseMatrix *local_tent_interp,
    mfem::Array<double> * local_coarse_one_representation);

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
mfem::HypreParMatrix *interp_smooth_interp(
    mfem::HypreParMatrix& A, const interp_data_t& interp_data,
    mfem::HypreParMatrix& tent_interp, mfem::HypreParVector& Dinv_neg);

/* Inline Functions Definitions */
static inline
mfem::HypreParMatrix *interp_smooth_interp(
    mfem::HypreParMatrix& A, const interp_data_t& interp_data,
    mfem::HypreParMatrix& tent_interp, mfem::HypreParVector& Dinv_neg)
{
    return interp_smooth(interp_data.interp_smoother_degree,
                         interp_data.interp_smoother_roots, interp_data.drop_tol,
                         interp_data.times_apply_smoother, A, tent_interp,
                         Dinv_neg);
}

} // namespace saamge

#endif // _INTERP_HPP
