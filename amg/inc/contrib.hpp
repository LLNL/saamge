/*! \file
    \brief Routines that assemble the tentative prolongator from local ones.

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
#ifndef _CONTRIB_HPP
#define _CONTRIB_HPP

#include "common.hpp"
#include <mfem.hpp>
#include "aggregates.hpp"

#include "SharedEntityCommunication.hpp"

namespace saamge
{

/* Types */

/*! \brief Make contrib_tent_struct_t a bit more object oriented.

    We want to manage optional behavior, such as having a
    coordinate vector that defines polynomial coarse spaces,
    a little better.

    And we want to get rid of the CONFIG_CLASS declarations.
*/
class ContribTent
{
public:
    /*! \brief Initiates the process of building the tentative interpolant.

      \param ND (IN) The number of fine DoFs.

      \returns The structure needed in \b contrib_tent_insert_from_local and
          freed/finalized by \b contrib_tent_finalize

      \warning The returned structure must be finalized and freed by the caller
          using contrib_tent_finalize
    */
    ContribTent(int ND, bool avoid_ess_bdr_dofs=true);
    ~ContribTent();

    /*! \brief Produces the final tentative interpolant.

      This is tha last phase of the tentative interpolant construction. It's
      called after \b contrib_tent_init and all \b contrib_tent_insert_from_local.
      \a tent_int_struct gets freed.

      \returns The local tentative interpolant as a finalized sparse matrix.

      \warning After calling this function no more calls of
             \b contrib_tent_insert_from_local can be made with
             \a tent_int_struct as a parameter.
    */
    mfem::SparseMatrix *contrib_tent_finalize();

    /*! \brief Similar to \b contrib_tent_finalize but for the restriction.
    */
    mfem::SparseMatrix *contrib_restr_finalize();

    /*!
      Like contrib_ones, except contrib spatial_dimension linear functions
      instead of constants.

      coords probably comes from Mesh.GetCoords() or Mesh.GetVertices(), at
      least on finest level.
    */
    void contrib_linears(
        const agg_partitioning_relations_t& agg_part_rels,
        int spatial_dimension, int num_nodes, const mfem::Vector& coords);

    /*!
      Like contrib_linears, get rigid body modes for elasticity
    */
    void contrib_rbms(
        const agg_partitioning_relations_t& agg_part_rels,
        int spatial_dimension, int num_nodes, const mfem::Vector& coords);

    /*!
      like contrib_mises, assume no eigenvalue problem, just use constant
      vector
    */
    void contrib_ones(const agg_partitioning_relations_t& agg_part_rels);

    /*! \brief Visits all MISes and fills in the tentative interpolant.

      \param agg_part_rels (IN) The partitioning relations.
      \param cut_evects_arr (IN) The vectors from all local eigenvalue problems.
      \param scaling_P (IN) if set, also build scaling_P for corrected nullspace
    */
    void contrib_mises(
        const agg_partitioning_relations_t& agg_part_rels,
        mfem::DenseMatrix * const *cut_evects_arr, bool scaling_P);

    /*! \brief Visits all coarse (agglomerate) faces and obtains the coarse face basis functions using "cut vectors" in H1 dofs.

      XXX: It would be good to combine the functions with similar functionality.
           However, I currently prefer to not interfere with existing code, but instead to simply augment it.

      \param agg_part_rels (IN) The partitioning relations.
      \param cut_evects_arr (IN) The vectors from all local eigenvalue problems.

      \returns An array of coarse faces bases on all coarse faces on the current processor. Must be freed by the caller.

    */
    mfem::DenseMatrix **contrib_cfaces(const agg_partitioning_relations_t& agg_part_rels,
                                       mfem::DenseMatrix * const *cut_evects_arr, bool full_space=false);

    /*! \brief Visits all coarse (agglomerate) faces and obtains the coarse face basis functions.

      Uses provided global targets.

      \returns An array of coarse faces bases on all coarse faces on the current processor. Must be freed by the caller.

    */
    mfem::DenseMatrix **contrib_cfaces_targets(const agg_partitioning_relations_t& agg_part_rels,
                                               const mfem::Array<mfem::Vector *>& face_targets);

    /*! \brief Visits all coarse (agglomerate) faces and obtains the coarse face basis functions for the "full" space.

      \param agg_part_rels (IN) The partitioning relations.

      \returns An array of coarse faces bases on all coarse faces on the current processor. Must be freed by the caller.

    */
    mfem::DenseMatrix **contrib_cfaces_full(const agg_partitioning_relations_t& agg_part_rels);

    /*! \brief Visits all coarse (agglomerate) faces and obtains the coarse face constant basis functions.

      \param agg_part_rels (IN) The partitioning relations.

      \returns An array of coarse faces bases on all coarse faces on the current processor. Must be freed by the caller.

    */
    mfem::DenseMatrix **contrib_cfaces_const(const agg_partitioning_relations_t& agg_part_rels);

    /*! \brief Visits all coarse (agglomerate) faces and obtains the constant basis functions that are constant on a fine face.

      XXX: The implementation is not great.

      \param agg_part_rels (IN) The partitioning relations.

      \returns An array of coarse faces bases on all coarse faces on the current processor. Must be freed by the caller.

    */
    mfem::DenseMatrix **contrib_ffaces_const(const agg_partitioning_relations_t& agg_part_rels);

    /*! \brief Inserts (embeds) the bases for the nonconforming method.

        Sets the entries of the local "interpolation" and "restriction" matrices
        for the nonconforming case using the bases on coarse (agglomerated) faces and
        coarse (agglomerated) elements.
    */
    void insert_from_cfaces_celems_bases(int nparts, int num_cfaces, const mfem::DenseMatrix * const *cut_evects_arr,
                                         const mfem::DenseMatrix * const *cfaces_bases, const agg_partitioning_relations_t& agg_part_rels, bool face_avg=true);

    /**
       Do both spectral and linears.
    */
    void contrib_composite(
        const agg_partitioning_relations_t& agg_part_rels,
        mfem::DenseMatrix * const *cut_evects_arr, int polynomial_order,
        int spatial_dimension, int num_nodes, const mfem::Vector& coords);

    // some getters
    mfem::Array<double> * get_local_coarse_one_representation() 
    {
        mfem::Array<double> * local_coarse_one_representatio_ = local_coarse_one_representation;
        local_coarse_one_representation = NULL;
        return local_coarse_one_representatio_;
    }
    int get_coarse_truedof_offset() {return coarse_truedof_offset;}
    int * get_mis_numcoarsedof() {return mis_numcoarsedof;}
    mfem::DenseMatrix ** get_mis_tent_interps() {return mis_tent_interps;}
    void set_threshold(double val) {threshold_ = val;}
    int get_celements_cdofs() {return celements_cdofs;}
    int *get_celements_cdofs_offsets()
    {
        int *celements_cdofs_offsets_ = celements_cdofs_offsets;
        celements_cdofs_offsets = NULL;
        return celements_cdofs_offsets_;
    }
    int *get_cfaces_truecdofs_offsets()
    {
        int *cfaces_truecdofs_offsets_ = cfaces_truecdofs_offsets;
        cfaces_truecdofs_offsets = NULL;
        return cfaces_truecdofs_offsets_;
    }
    int *get_cfaces_cdofs_offsets()
    {
        int *cfaces_cdofs_offsets_ = cfaces_cdofs_offsets;
        cfaces_cdofs_offsets = NULL;
        return cfaces_cdofs_offsets_;
    }

private:
    /*! \brief Deals with essential boundary conditions in the interpolator.

        \param restriction (IN) usually a row of mis_to_dof, used to identify dofs
               in local with dofs in larger matrix
    */
    void contrib_filter_boundary(const agg_partitioning_relations_t& agg_part_rels,
                                 mfem::DenseMatrix& local, 
                                 const int *restriction);

    void contrib_tent_insert_simple(const agg_partitioning_relations_t& agg_part_rels,
                                    mfem::DenseMatrix& local, 
                                    const int *restriction);

    /*! \brief Inserts (embeds) the local tentative interpolant in the global one.

      DEPRECATED, we only use insert_simple now.

      \param agg_part_rels (IN) The partitioning relations.
      \param local (IN/OUT) The local tentative interpolant, this may also be modified by boundary conditions
      \param restriction (IN) The array that for each DoF of the aggregate
             maps its global number (as returned by
             \b agg_restrict_to_agg).
    */
    void contrib_tent_insert_from_local(const agg_partitioning_relations_t& agg_part_rels,
                                        mfem::DenseMatrix& local, const int *restriction);

    /**
       Take cut_evects_arr, which probably came from interp_compute_vectors(),
       and do the appropriate communication on shared minimal intersection
       sets, returning the synchronized matrices of eigenvectors that are now
       ready for SVD.

       Really this is just some code extracted from contrib_mises to make it 
       more "modular"
    */
    mfem::DenseMatrix ** CommunicateEigenvectors(
        const agg_partitioning_relations_t& agg_part_rels,
        mfem::DenseMatrix * const *cut_evects_arr,
        SharedEntityCommunication<mfem::DenseMatrix>& sec);

    /**
       Takes cut_evects_arr, which are vectors defined on local AEs,
       and do the appropriate communication on shared coarse (agglomerate) faces,
       returning the synchronized matrices of restricted eigenvectors that are now
       ready for SVD.

       XXX: It would be good to combine the functions with similar functionality.
            However, I currently prefer to not interfere with existing code, but instead to simply augment it.
    */
    mfem::DenseMatrix ** CommunicateEigenvectorsCFaces(
        const agg_partitioning_relations_t& agg_part_rels,
        mfem::DenseMatrix * const *cut_evects_arr,
        SharedEntityCommunication<mfem::DenseMatrix>& sec);

    /**
       Given a received_mats array, either from CommunicateEigenvectors() or
       possibly basically empty, add constant functions to it.
    */
    void ExtendWithConstants(mfem::DenseMatrix ** received_mats,
                             const agg_partitioning_relations_t& agg_part_rels);

    /**
       Given a received_mats array, either from CommunicateEigenvectors() or
       possibly basically empty, add polynomial functions to it.

       Right now only implemented for constants and linears.
    */
    void ExtendWithPolynomials(
        mfem::DenseMatrix ** received_mats,
        const agg_partitioning_relations_t& agg_part_rels,
        int order, int spatial_dimension, 
        int num_nodes, const mfem::Vector& coords);

    /**
       For elasticity, extend spectral degrees of freedom with rigid body
       modes.
    */
    void ExtendWithRBMs(
        mfem::DenseMatrix ** received_mats,
        const agg_partitioning_relations_t& agg_part_rels,
        int spatial_dimension,
        int num_nodes, const mfem::Vector& coords);

    /**
       Take received_mats, deal with essential boundary conditions, do SVDs,
       insert into the tentative prolongator, and then destroy received_mats.

       Note well that received_mats is totally deleted by this routine.
    */
    void SVDInsert(const agg_partitioning_relations_t& agg_part_rels,
                   mfem::DenseMatrix ** received_mats, int * row_sizes,
                   bool scaling_P);

    /**
       Take received_mats, do SVDs on coarse faces, and then destroy received_mats.

       Returns the obtained coarse face bases on owned coarse faces. On the faces that are not owned
       an empty DenseMatrix is defined. Needs to be freed by the caller.

       Does not deal with essential boundary conditions, only basic linear dependence filtering.

       Note well that received_mats is totally deleted by this routine.
    */
    mfem::DenseMatrix **CFacesSVD(const agg_partitioning_relations_t& agg_part_rels,
                                  mfem::DenseMatrix **received_mats, int *row_sizes, bool full_space=false);

    /**
       Take span_mats, do SVDs on coarse faces, and then destroy span_mats.

       Returns the obtained coarse face bases on known coarse faces. Needs to be freed by the caller.

       span_mats is an array of singleton dense matrices (one matrix on each entry).

       No communication is performed, so it is applicable in the cases where known
       global targets are used.

       Does not deal with essential boundary conditions, only basic linear dependence filtering.

       Note well that span_mats is totally deleted by this routine.
    */
    mfem::DenseMatrix **CFacesSVDnocomm(const agg_partitioning_relations_t& agg_part_rels,
                                        mfem::DenseMatrix **span_mats);

    /*! building coarse_one_representation on the fly (we are going to just
      copy this pointer to interp_data) */
    mfem::Array<double> * local_coarse_one_representation; 
    int coarse_truedof_offset;
    int * mis_numcoarsedof;
    mfem::DenseMatrix ** mis_tent_interps;


    /*! See \b interp_data_t for explaining these variables. */
    int celements_cdofs;
    int *celements_cdofs_offsets;
    int *cfaces_truecdofs_offsets;
    int *cfaces_cdofs_offsets;

    const int rows; /*!< The rows of the interpolant matrix. */
    int filled_cols; /*!< How many columns are currently introduced in the
                          interpolant matrix. */
    mfem::SparseMatrix *tent_interp_; /*!< The partially built matrix of the tentative
                                    interpolant. */
    mfem::SparseMatrix *tent_restr_; /*!< The partially built matrix of the tentative
                                    restriction. */

    bool avoid_ess_bdr_dofs;
    double svd_eps; // tolerance for SVD calculation
    double threshold_; // do not insert values smaller than this into P
};

} // namespace saamge

#endif // _CONTRIB_HPP
