/*! \file
    \brief Aggregates and agglomerated elements (AEs) related functionality.

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

    1. We count on the boundary conditions (if imposed) to be imposed only on
       the global matrix and NOT on the element matrices.
    2. Only domain integrators are considered in the assembly procedure.
    3. When the simple aggregation resolution is used, some aggregates may end
       up empty! This is an issue that is not solved yet. The program cannot
       continue with empty aggregates.
    4. The strength aggregation resolution has issues with pathological cases.
       For example, AEs with empty inner parts (w.r.t. trivially assigned DoFs)
       cannot be handled.
    5. For coarse levels there are at least two ways to compute cdof_to_cdof.
       (here, f=fine and c=coarse).
       The first way:
        cdof_to_cdof = cdof_to_celement x celement_to_cdof =
                     = cdof_to_fdof x fdof_to_fAE x fAE_to_fdof x fdof_to_cdof.
       The second way:
        cdof_to_cdof = cdof_to_fdof x fdof_to_fdof x fdof_to_cdof.
       Clearly, they do not produce the same graph in the general case.
       The first method does not provide the sparsity pattern of the coarse
       stiffness matrix. In the case when fdof_to_cdof comes from the tentative
       prolongators, the second method gives the sparsity pattern of the coarse
       operators in the tentative hierarchy. When fdof_to_cdof comes from the
       final smoothed prolongators, the second method gives the sparsity
       pattern of the coarse operators in the final (smoothed) hierarchy.
       XXX: Currently, cdof_to_cdof is used only for debug. Please, have this
       in mind in case you want to use it for "non-debug". For the debug
       purposes, the second way above for attaining cdof_to_cdof defines also a
       more adequate notion of connectivity.

    TODO: Make it possible to somehow configure the methodology of creating
          partitions of elem_to_elem, i.e. creating AEs. Currently it is
          hardcoded which makes it not very flexible. Also, think of handling
          the case when the number of generated partitions is different from
          the desired number.
*/

#pragma once
#ifndef _AGGREGATES_HPP
#define _AGGREGATES_HPP

#include "common.hpp"
#include <mfem.hpp>
#include "part.hpp"

/* Defines */
/*! \def AGG_BETWEEN_AES_FLAG
    \brief A flag denoting that a DoF is on the border between AEs.
*/
/*! \def AGG_ON_ESS_DOMAIN_BORDER_FLAG
    \brief A flag denoting that a DoF is on the "essential" domain border.
*/
/*! \def AGG_ON_PROC_IFACE_FLAG
    \brief A flag denoting that a DoF is on the interface between processes.
*/
/*! \def AGG_OWNED_FLAG
    \brief A flag denoting that a DoF is owned (in some sense) by the process.
*/
/*! \def AGG_ALL_FLAGS
    \brief All flags set.
*/
#define AGG_BETWEEN_AES_FLAG            0x01
#define AGG_ON_ESS_DOMAIN_BORDER_FLAG   0x02
#define AGG_ON_PROC_IFACE_FLAG          0x04
#define AGG_OWNED_FLAG                  0x08

#define AGG_ALL_FLAGS               (AGG_BETWEEN_AES_FLAG | \
                                     AGG_ON_ESS_DOMAIN_BORDER_FLAG | \
                                     AGG_ON_PROC_IFACE_FLAG | \
                                     AGG_OWNED_FLAG)

/* Types */
/*! \brief Ways of assigning the arguable DoFs when aggregating.
*/
typedef enum {
    AGG_RES_AGGS_SIMPLE_STRENGTH, /*!< The approach is greedy and based on the
                                       strength of connections. */

    AGG_RES_AGGS_MAX /*!< Sentinel. */
} agg_resolve_aggragates_t;

/*! \brief Status of a DoF.
*/
typedef char agg_dof_status_t;

/*! \brief Structure for partitioning parameters and relations.
*/
typedef struct {
    int ND; /*!< The number of DoFs of the finite element space on the fine
                 level. */
    int nparts; /*!< The number of parts (AEs and big aggregates). */
    int *partitioning; /*!< The partitioning of the dual graph. */
    Table *dof_to_elem; /*!< The relation table that relates DoFs to
                             elements. */
    Table *dof_to_dof; /*!< The relation table that relates DoFs to DoFs.
                            XXX: Currently it is used only when SA_DEBUG_LEVEL
                                 is sufficiently high! */

    Table *elem_to_dof; /*!< The relation table that relates elements to
                             DoFs. */
    Table *AE_to_elem; /*!< The relation table that relates AEs to elements. */
    Table *elem_to_AE; /*!< The relation table that relates elements to AEs. */
    Table *elem_to_elem; /*!< The relation table that relates elements to
                              elements. */

    Table *AE_to_dof; /*!< The relation table that relates AEs to DoFs. */
    Table *dof_to_AE; /*!< The relation table that relates DoFs to AEs. */
    int *dof_id_inAE; /*!< An array used for mapping global DoF IDs to local
                           DoF IDs within any AE.
                           XXX: This array alone does not provide enough
                                information. It is combined with \b dof_to_AE
                                to get the final result. */

    int *preaggregates; /*!< An array that for each DoF matches the aggregate
                             it surely belongs to or has -2 for unresolved
                             DoFs. */
    int *preagg_size; /*!< An array with the sizes of the aggregates in number
                           of DoFs before resolving arguable DoFs. */
    agg_resolve_aggragates_t resolve_aggragates; /*!< Determines how to assign
                                                      the arguable DoFs when
                                                      aggregating. */

    int *aggregates; /*!< An array that for every DoF matches the aggregate it
                          belongs to. */
    int *agg_size; /*!< An array with the sizes of the aggregates in number of
                        DoFs. */
    agg_dof_status_t *agg_flags; /*!< An array of flags for every DoF. */

    ParFiniteElementSpace *fes; /*!< The parallel finite element space we are
                                     on. Will not be freed when the structure
                                     is being freed.
                                     XXX: We actually do not need finite
                                          element information. We use this for
                                          the topological relation structures
                                          and communicators it provides. This
                                          is a candidate to modify so that the
                                          method would be really general. */

} agg_partititoning_relations_t;

/*! \brief A callback returning desired element matrices.

    The numbering of the element DoFs (for the returned matrix) must correspond
    to the numbering given by \b elem_to_dof in \a agg_part_rels.

    See \em elmat.hpp.

    XXX: If OpenMP (or similar) would be used, these functions MUST be
         thread-safe.

    \param elno (IN) The element number.
    \param agg_part_rels (IN) The partitioning relations.
    \param data (IN/OUT) Function specific data.
    \param free_matr (OUT) Indicates whether the returned matrix must be freed
                           by the caller.

    \returns An element matrix (sparse or dense, it must be
             \em dynamic_cast-ed). Only \b SparseMatrix and \b DenseMatrix
             derivatives of \b Matrix are supported.

    \warning It must be compiled with Run-Time Type Information (RTTI) enabled
             for \em dynamic_cast to work.
    \warning Any function of this type must return always sparse or always
             dense matrices, no mixing allowed.
*/
typedef Matrix *(*agg_elmat_callback_ft)(int elno,
    const agg_partititoning_relations_t *agg_part_rels, void *data,
    bool& free_matr);

/* Options */

/*! \brief The configuration class of this module.
*/
CONFIG_BEGIN_CLASS_DECLARATION(AGGREGATES)

    /*! Determines how to assign the arguable DoFs when aggregating.

        \warning This parameter is used during the construction of objects
                 (structure instances) so if modified it will only have effect
                 for new objects and will NOT modify the behavior of existing
                 ones. For altering the option for existing objects look at the
                 corresponding fields in the respective structure(s). */
    CONFIG_DECLARE_OPTION(agg_resolve_aggragates_t, resolve_aggragates);

CONFIG_END_CLASS_DECLARATION(AGGREGATES)

CONFIG_BEGIN_INLINE_CLASS_DEFAULTS(AGGREGATES)
    CONFIG_DEFINE_OPTION_DEFAULT(resolve_aggragates,
                                 AGG_RES_AGGS_SIMPLE_STRENGTH)
CONFIG_END_CLASS_DEFAULTS

/* Functions */
/*! \brief Generates the preaggregates and related structures.

    The preaggregates resolve only the obvious DoF assignments. The arguable
    DoFs are left unassigned.

    \param dof_to_AE (IN) Relation table matching DoFs to AEs.
    \param nparts (IN) The number of AEs in the partitioning.
    \param bdr_dofs (IN) An array that shows if a DoF i is on essential domain
                         border. It can be NULL.
    \param preagg_size (OUT) An array with the sizes of the preaggregates in
                             number of DoFs. MUST BE FREED BY THE CALLER!
    \param agg_flags (OUT) An array of flags for every DoF.  MUST BE FREED BY
                           THE CALLER!

    \returns An array that contains for every non-arguable DoF the number (id)
             of the aggregate it belongs to. For the arguable DoFs it contains
             -2. It relates DoFs to preaggregates.

    \warning The returned array must be freed by the caller.
*/
int *agg_preconstruct_aggregates(const Table& dof_to_AE, int nparts,
                                 const agg_dof_status_t *bdr_dofs,
                                 int *& preagg_size,
                                 agg_dof_status_t *& agg_flags);

/*! \brief Resolves the arguable DoFs locally.

    The DoFs on processes' interfaces remain unresolved in the end. Only the
    guys in the local group are resolved. The interface DoFs remain marked as
    -2 in \em agg_part_rels.aggregates.

    The approach is greedy and based on the strength of connections.

    TODO: It can be improved. Also, another strength-based approach may be
          better.

    \param A (IN) The global stiffness matrix.
    \param agg_part_rels (IN/OUT) The partitioning relations.
    \param first_time (IN) Whether this is the first time aggregates are
                           resolved. Actually, it is NOT used here.

    \returns If anything was actually changed.

    \warning \em agg_part_rels.aggregates and \em agg_part_rels.agg_size must
             be allocated prior to calling this function.
    \warning It has issues with pathological cases. For example, AEs with empty
             inner parts (w.r.t. trivially assigned DoFs) cannot be handled.
*/
bool agg_simple_strength_local_resolve(const SparseMatrix& A,
                                       const agg_partititoning_relations_t&
                                                                 agg_part_rels,
                                       bool first_time);

/*! \brief Resolves the arguable DoFs on processes' interfaces.

    The approach is greedy and based on the strength of connections.

    TODO: It can be improved. Also, another strength-based approach may be
          better. Now it is very basic. Simply gives the DoF to the master of
          the group (the process with lowest rank according to MFEM) and then
          uses strength of connections to distribute it within the master
          process.

    \param A (IN) The global stiffness matrix.
    \param agg_part_rels (IN/OUT) The partitioning relations.
    \param first_time (IN) Whether this is the first time aggregates are
                           resolved. Actually, it is NOT used here.

    \returns If anything was actually changed.

    \warning It has issues with pathological cases. For example, AEs with empty
             inner parts (w.r.t. trivially assigned DoFs) cannot be handled.
*/
bool agg_simple_strength_iface_resolve(const SparseMatrix& A,
                                       const agg_partititoning_relations_t&
                                                                 agg_part_rels,
                                       bool first_time);

/*! \brief Full initial construction of the aggregates.

    Includes preconstruction and resolution of non-trivial DoFs.

    \param A (IN) The global stiffness matrix.
    \param agg_part_rels (IN/OUT) The partitioning relations.
    \param bdr_dofs (IN) An array that shows if a DoF i is on essential domain
                         border. It can be NULL.
*/
void agg_construct_aggregates(const SparseMatrix& A,
                              agg_partititoning_relations_t& agg_part_rels,
                              const agg_dof_status_t *bdr_dofs);

/*! \brief Assembles the local stiffness matrix for an AE on the finest mesh.

    Uses entries in the global stiffness matrix for better performance. Also,
    takes care of essential boundary conditions.

    Initially implemented for the finest (geometric) mesh.

    \param A (IN) The global stiffness matrix.
    \param part (IN) The id (number) of the AE in question.
    \param agg_part_rels (IN) The partitioning relations.
    \param elmat_callback (IN) A callback that returns element matrices.
    \param data (IN/OUT) \a elmat_callback specific data.
    \param bdr_cond_imposed (IN) States if the border conditions are imposed on
                                 \a A.
    \param assemble_ess_diag (IN) Causes the diagonal elements corresponding
                                  to DoFs lying simultaneously on AEs'
                                  interfaces and the essential part of the
                                  boundary to be assembled instead of copied
                                  from the global matrix \a A. It is only
                                  useful when \a bdr_cond_imposed is \em true.
                                  It is meaningful in the cases when the global
                                  matrix has its diagonal entries kept, instead
                                  of set to 1, during the global imposition of
                                  essential boundary conditions.

    \returns The local stiffness matrix for the AE in question.

    \warning The returned sparse matrix must be freed by the caller.
    \warning If the matrix \a A represents exactly the topology of the mesh in
             the sense of the relation \b dof_to_dof, then this algorithm is
             expected to work properly. That is, in the \em BilinearForm case,
             the caller should build it using Assemble(0) and Finalize(0).
             Actually, Assemble(1) will also work and, also, cutting off the
             connections to the DoFs with essential boundary conditions will
             work. The only issue is when the global matrix \A shows no
             connection between two DoFs (globally) but locally (in the element
             stiffness matrices) such a connection may exist. In such case it
             may happen that a local matrix (for an AE) will have a zero
             element where it shouldn't be zero. In the \a agg_simple_assemble
             case, just setting \em final_skip_zeros to \em false should be
             sufficient.
*/
SparseMatrix *agg_build_AE_stiffm_with_global(const SparseMatrix& A, int part,
    const agg_partititoning_relations_t& agg_part_rels,
    agg_elmat_callback_ft elmat_callback, void *data, bool bdr_cond_imposed,
    bool assemble_ess_diag);

/*! \brief Assembles the local stiffness matrices for all AEs.

    Uses entries in the global stiffness matrix for better performance. Also,
    takes care of essential boundary conditions.

    Initially implemented for the finest (geometric) mesh.

    \param A (IN) The global stiffness matrix.
    \param agg_part_rels (IN) The partitioning relations.
    \param elmat_callback (IN) A callback that returns element matrices.
    \param data (IN/OUT) \a elmat_callback specific data.
    \param bdr_cond_imposed (IN) States if the border conditions are imposed on
                                 \a A.
    \param assemble_ess_diag (IN) Causes the diagonal elements corresponding
                                  to DoFs lying simultaneously on AEs'
                                  interfaces and the essential part of the
                                  boundary to be assembled instead of copied
                                  from the global matrix \a A. It is only
                                  useful when \a bdr_cond_imposed is \em true.
                                  It is meaningful in the cases when the global
                                  matrix has its diagonal entries kept, instead
                                  of set to 1, during the global imposition of
                                  essential boundary conditions.

    \returns An array of pointers to local stiffness matrices for all AEs.

    \warning The returned array of pointers to sparse matrices must be freed by
             the caller using \b mbox_free_matr_arr.
    \warning See the warning in \b agg_build_AE_stiffm_with_global related to
             the zero elements in \a A.
*/
SparseMatrix **agg_build_AEs_stiffm_with_global(const SparseMatrix& A,
    const agg_partititoning_relations_t& agg_part_rels,
    agg_elmat_callback_ft elmat_callback, void *data, bool bdr_cond_imposed,
    bool assemble_ess_diag);

/*! \brief Assembles a global matrix from element matrices.

    Can also impose essential boundary conditions (BCs).

    \param agg_part_rels (IN) The partitioning relations. Only \em elem_to_dof
                              and \em dof_to_elem are used here. More might be
                              required depending ot \a elmat_callback.
    \param elmat_callback (IN) A callback that returns element matrices.
    \param data (IN/OUT) \a elmat_callback specific data.
    \param assem_skip_zeros (IN) Skip matrix entries with zero value when
                                 assembling. Skips zeroes in the element
                                 matrices when adding them to the global matrix.
    \param final_skip_zeros (IN) Skip matrix entries with zero value when
                                 finalizing. Not relevant if \a finalize is
                                 \em false.
    \param finalize (IN) Finalize the sparse matrix before returning it.
    \param bdr_dofs (IN) An array that shows if a DoF i is on essential domain
                         border. If it is not NULL essential BCs. If it is NULL,
                         no BCs are imposed.
    \param sol (IN) A vector (like a solution) that fulfils the essential BCs.
                    Not used and can be NULL if \a bdr_dofs is NULL.
    \param rhs (IN/OUT) The right hand side without imposed essential BCs. It
                        will be modified to impose the essential BCs. Not used
                        and can be NULL if \a bdr_dofs is NULL.
    \param keep_diag (IN) Whether to keep the diagonal matrix entry when
                          imposing essential boundary conditions. If not kept,
                          it will be set to 1. Not used if \a bdr_dofs is NULL.

    \returns The assembled global matrix.

    \warning The returned sparse matrix must be freed by the caller.
*/
SparseMatrix *agg_simple_assemble(
    const agg_partititoning_relations_t& agg_part_rels,
    agg_elmat_callback_ft elmat_callback, void *data, bool assem_skip_zeros/*=true*/,
    bool final_skip_zeros/*=true*/, bool finalize/*=true*/, const agg_dof_status_t *bdr_dofs,
    const Vector *sol, Vector *rhs, bool keep_diag);

/*! \brief Assembles a global matrix from element matrices.

    Can also impose essential boundary conditions (BCs).

    The element matrices must be ordered according to the numbering given by
    \a elem_to_dof.

    \param elem_matrs (IN) An array of \b Matrix pointers (which are either
                           \b SparseMatrix pointers or \b DenseMatrix pointers).
    \param elem_to_dof (IN) A table that relates elements to DoFs.
    \param dof_to_elem (IN) A table that relates DoFs to elements.
    \param assem_skip_zeros (IN) Skip matrix entries with zero value when
                                 assembling. Skips zeroes in the element
                                 matrices when adding them to the global matrix.
    \param final_skip_zeros (IN) Skip matrix entries with zero value when
                                 finalizing. Not relevant if \a finalize is
                                 \em false.
    \param finalize (IN) Finalize the sparse matrix before returning it.

    \returns The assembled global matrix.

    \warning The returned sparse matrix must be freed by the caller.
*/
SparseMatrix *agg_simple_assemble(const Matrix * const *elem_matrs,
                                  Table& elem_to_dof, Table& dof_to_elem,
                                  bool assem_skip_zeros/*=true*/, bool final_skip_zeros/*=true*/,
                                  bool finalize/*=true*/,
                                  const agg_dof_status_t *bdr_dofs,
                                  const Vector *sol, Vector *rhs,
                                  bool keep_diag);

/*! \brief Restricts a group of vectors to an aggregate.

    Restricts a group of vectors presented as a dense matrix (each column is a
    vector), defined on an AE, to a specified aggregate.

    \param part (IN) The id (number) of the AE in question.
    \param AE_to_dof (IN) How AEs are related to DoFs.
    \param aggregates (IN) Array that maps DoFs to aggregates.
    \param agg_id (IN) The id (number) of the aggregate to restrict to.
    \param agg_size (IN) The size of the aggregate (in number of DoFs).
    \param cut_evects (IN) The vectors, defined on the AE, that will be
                           restricted to the aggregate.
    \param restricted (OUT) The restricted, to the aggregate, vectors.

    \returns An array that for each DoF of the aggregate (as indexed in the
             \a restricted vectors) maps its global number.

    \warning The returned array must be freed by the caller.
    \warning The aggregate must be contained in the AE.
*/
int *agg_restrict_to_agg(int part, const Table& AE_to_dof,
                         const int *aggregates, int agg_id, int agg_size,
                         DenseMatrix& cut_evects, DenseMatrix& restricted);

/*! \brief Restricts a globally defined vector to an AE.

    \param part (IN) The id (number) of the AE in question.
    \param agg_part_rels (IN) The partitioning relations.
    \param glob_vect (IN) The vector, defined globally, that will be restricted
                          to the AE.
    \param restricted (OUT) The restricted, to the AE, vector.
*/
void agg_restrict_vect_to_AE(int part,
                             const agg_partititoning_relations_t& agg_part_rels,
                             const Vector& glob_vect, Vector& restricted);

/*! \brief Builds structures for mapping global DoF IDs to local (AE) IDs.

    \param agg_part_rels (IN) The partitioning relations.
*/
void agg_build_glob_to_AE_id_map(agg_partititoning_relations_t& agg_part_rels);

/*! \brief Creates all relations on the finest (usually geometric) mesh.

    The function is a wrapper that uses several other functions to do the
    partitioning; generate AEs, corresponding aggregates and all needed
    relations and structures.

    \param A (IN) The global stiffness matrix.
    \param NE (IN) The number of element.
    \param elem_to_dof (IN) The relation relating elements to DoFs. It becomes
                            owned by the returned partitioning structure and
                            it will be freed automatically with that structure.
                            MUST NOT be freed or modified separately by the
                            caller.
    \param elem_to_elem (IN) The relation relating elements to elements. It
                             becomes owned by the returned partitioning
                             structure and it will be freed automatically with
                             that structure. MUST NOT be freed or modified
                             separately by the caller.
    \param partitioning (IN) If NULL, the partitioning will be generated in
                             this function. If the pointer is valid, the given
                             partitioning will be used. In that case, it
                             becomes owned by the returned partitioning
                             structure and it will be freed automatically with
                             that structure. MUST NOT be freed or modified
                             separately by the caller.
    \param bdr_dofs (IN) An array that shows if a DoF i is on the domain border.
    \param nparts (IN/OUT) The number of partitions (AEs, aggregates) in the
                           partitioning. It inputs the desired number of
                           partitions and outputs the actual number of
                           generated partitions.
    \param fes (IN) The used finite element space. A pointer to it will stay in
                    the returned structure. However, the caller owns the finite
                    element space and must free it when the returned structure
                    is freed. See also the description of
                    \em agg_partititoning_relations_t::fes.

    \returns A structure with the partitioning relations.

    \warning The returned structure must be freed by the caller by calling
             \b agg_free_partitioning.
*/
agg_partititoning_relations_t *
agg_create_partitioning_fine(const SparseMatrix& A, int NE, Table *elem_to_dof,
                             Table *elem_to_elem, int *partitioning,
                             const agg_dof_status_t *bdr_dofs, int *nparts,
                             ParFiniteElementSpace *fes);

/*! \brief Frees a partitioning relations structure.

    \param agg_part_rels (IN) The structure to be freed.
*/
void agg_free_partitioning(agg_partititoning_relations_t *agg_part_rels);

/*! \brief Makes a copy of partitioning data.

    \param src (IN) The partitioning data to be copied.

    \returns A copy of \a src.

    \warning The returned structure must be freed by the caller using
             \b agg_free_partitioning.
*/
agg_partititoning_relations_t
    *agg_copy_partitioning(const agg_partititoning_relations_t *src);

/* Inline Functions */
/*! \brief Determines if a DoF is on "essential boundary".

    Determines if a DoF is on the part of the boundary with essential boundary
    conditions.

    \param agg_part_rels (IN) The partitioning relations.
    \param dof_id (IN) The DoF in question.

    \returns Whether the DoF is on the part of the boundary with essential
             boundary conditions.
*/
static inline
bool agg_is_dof_on_essential_border(const agg_partititoning_relations_t&
                                                                  agg_part_rels,
                                    int dof_id);

/*! \brief Generates the relation table that correspond to \a A_to_B_arr.

    The array \a A_to_B_arr describes a simple relation (each A is related to
    a single B, i.e. the B-s partition the set of A-s).

    This function generates the relation tables corresponding to the relations
    in array \a A_to_B_arr.

    \param A_to_B_arr (IN) See the description.
    \param num_A (IN) The number of A-s.
    \param A_to_B (OUT) Simply a table version of the input \a A_to_B_arr.
                        MUST BE FREED BY THE CALLER!
    \param B_to_A (OUT) The transposed of \a A_to_B. MUST BE FREED BY THE
                        CALLER!
*/
static inline
void agg_construct_tables_from_arr(const int *A_to_B_arr, int num_A,
                                   Table *& A_to_B, Table *& B_to_A);

/*! \brief Gives the global index of an element in a collection (general).

    Gives the global number of an element from its local number in a collection.

    \param col_to_elem (IN) The relation table that relates collections to
                            their elements.
    \param col (IN) The ID (number) of the collection in question.
    \param loc_num (IN) The local ID (number) of the element inside the
                        collection.

    \returns The global id (number) of the element in question.
*/
static inline
int agg_num_col_to_glob(const Table& col_to_elem, int col, int loc_num);

/*! \brief Returns if an element (general) is in a given collection (general).

    \param elem (IN) The id of the element.
    \param col (IN) The id of the collection.
    \param elem_to_col (IN) Table that relates elements to collections.

    \returns If the element is in the collection it returns the order of the
             'collection' in the 'element' row in the CSR representation of
             \a elem_to_col. If the element is NOT in the collection, a
             negative number is returned.
*/
static inline
int agg_elem_in_col(int elem, int col, const Table& elem_to_col);

/*! \brief Returns the local (AE) ID of a DoFs, given the global ID.

    \param glob_id (IN) The global ID of the DoF.
    \param part (IN) The AE of interest.
    \param agg_part_rels (IN) The partitioning relations.

    \returns If the DoF is in the AE of interest its local ID is returned.
             Otherwise, a negative number is returned.
*/
static inline
int agg_map_id_glob_to_AE(int glob_id, int part,
                          const agg_partititoning_relations_t& agg_part_rels);

/*! \brief Resolves the arguable DoFs.

    \param A (IN) The global stiffness matrix.
    \param agg_part_rels (IN/OUT) The partitioning relations.
    \param first_time (IN) Whether this is the first time aggregates are
                           resolved.

    \returns If anything was actually changed.

    \warning agg_part_rels.aggregates and agg_part_rels.agg_size must be
             allocated prior to calling this function.
*/
static inline
bool agg_resolve_aggregates(const SparseMatrix& A,
                            const agg_partititoning_relations_t& agg_part_rels,
                            bool first_time);

/*! \brief Eliminates essential boundary conditions (BCs) in \a A and \a rhs.

    \param A (IN/OUT) The global stiffness matrix without imposed essential BCs.
    \param bdr_dofs (IN) An array that shows if a DoF i is on essential domain
                         border.
    \param sol (IN) A vector (like a solution) that fulfils the essential BCs.
    \param rhs (IN/OUT) The right hand side without imposed essential BCs.
    \param keep_diag (IN) Whether to keep the diagonal matrix entry when
                          imposing essential boundary conditions. If not kept,
                          it will be set to 1.
*/
static inline
void agg_eliminate_essential_bc(SparseMatrix& A,
                                const agg_dof_status_t *bdr_dofs,
                                const Vector& sol, Vector& rhs, bool keep_diag);

/*! \brief Prints information about the partitionings.

    \param agg_part_rels (IN) The partitioning relations.
*/
static inline
void agg_print_data(const agg_partititoning_relations_t& agg_part_rels);

/* Inline Functions Definitions */
static inline
bool agg_is_dof_on_essential_border(const agg_partititoning_relations_t&
                                                                  agg_part_rels,
                                    int dof_id)
{
    return SA_IS_SET_A_FLAG(agg_part_rels.agg_flags[dof_id],
                            AGG_ON_ESS_DOMAIN_BORDER_FLAG);
}

static inline
void agg_construct_tables_from_arr(const int *A_to_B_arr, int num_A,
                                   Table *& A_to_B, Table *& B_to_A)
{
    SA_ASSERT(A_to_B_arr);
    A_to_B = new Table(num_A, const_cast<int *>(A_to_B_arr));
    B_to_A = new Table();
    Transpose(*A_to_B, *B_to_A);
}

static inline
int agg_num_col_to_glob(const Table& col_to_elem, int col, int loc_num)
{
    SA_ASSERT(loc_num < col_to_elem.RowSize(col));
    return col_to_elem.GetRow(col)[loc_num];
}

static inline
int agg_elem_in_col(int elem, int col, const Table& elem_to_col)
{
    SA_ASSERT(0 <= elem && elem < elem_to_col.Size());
    SA_ASSERT(0 <= col);
#if (SA_IS_DEBUG_LEVEL(10))
    SA_ASSERT(col < elem_to_col.Width()); //XXX: Way too slow!
#endif
    const int *row = elem_to_col.GetRow(elem);
    const int rowsz = elem_to_col.RowSize(elem);

    for (int i=0; i < rowsz; ++i)
    {
        if (*(row++) == col)
            return i;
    }

    return -1;
}

static inline
int agg_map_id_glob_to_AE(int glob_id, int part,
                          const agg_partititoning_relations_t& agg_part_rels)
{
    SA_ASSERT(agg_part_rels.AE_to_dof);
    SA_ASSERT(agg_part_rels.dof_to_AE);
    SA_ASSERT(agg_part_rels.dof_id_inAE);
    SA_ASSERT(0 <= part && part < agg_part_rels.AE_to_dof->Size());
    SA_ASSERT(0 <= glob_id && glob_id < agg_part_rels.ND);

    const Table& dof_to_AE = *agg_part_rels.dof_to_AE;
    const int order = agg_elem_in_col(glob_id, part, dof_to_AE);
    SA_ASSERT(0 > order ||
              (0 <= order + dof_to_AE.GetI()[glob_id] &&
               order + dof_to_AE.GetI()[glob_id] < dof_to_AE.Size_of_connections() &&
               dof_to_AE.GetJ()[order + dof_to_AE.GetI()[glob_id]] == part &&
               agg_part_rels.dof_id_inAE[order + dof_to_AE.GetI()[glob_id]] >= 0));

    return (0 > order ? order :
            agg_part_rels.dof_id_inAE[order + dof_to_AE.GetI()[glob_id]]);
}

static inline
bool agg_resolve_aggregates(const SparseMatrix& A,
                            const agg_partititoning_relations_t& agg_part_rels,
                            bool first_time)
{
    SA_ASSERT(0 <= agg_part_rels.resolve_aggragates &&
              agg_part_rels.resolve_aggragates < AGG_RES_AGGS_MAX);
    SA_ASSERT(agg_part_rels.aggregates);
    SA_ASSERT(agg_part_rels.agg_size);
    bool ret = true;
    switch (agg_part_rels.resolve_aggragates)
    {
        default:
            SA_ASSERT(false);
            // Fall through to get to some default behavior in bad cases.

        case AGG_RES_AGGS_SIMPLE_STRENGTH:
            ret = agg_simple_strength_local_resolve(A, agg_part_rels,
                                                    first_time);
            ret = agg_simple_strength_iface_resolve(A, agg_part_rels,
                                                    first_time)
                  || ret;
            break;
    }
    return ret;
}

static inline
void agg_eliminate_essential_bc(SparseMatrix& A,
                                const agg_dof_status_t *bdr_dofs,
                                const Vector& sol, Vector& rhs, bool keep_diag)
{
    SA_PRINTF_L(4, "%s", "Imposing boundary conditions...\n");
    SA_ASSERT(A.Width() == A.Size());
    SA_ASSERT(rhs.Size() == A.Size());
    SA_ASSERT(sol.Size() == A.Size());
    SA_ASSERT(bdr_dofs);

    const int n = A.Size();
    for (int i=0; i < n; ++i)
    {
        if (SA_IS_SET_A_FLAG(bdr_dofs[i], AGG_ON_ESS_DOMAIN_BORDER_FLAG))
            A.EliminateRowCol(i, sol(i), rhs, (int)keep_diag);
    }
}

static inline
void agg_print_data(const agg_partititoning_relations_t& agg_part_rels)
{
    int i, min, max;
    SA_PRINTF("%s", ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
                    ">>>>>>>>>>>>>>>>>>>>>>>>>>>>\n");
    SA_PRINTF("%s", "\tPartitions (aggregates and agglomerates):\n");
    SA_PRINTF("Agglomerates and (big) aggregates: %d\n",
              agg_part_rels.nparts);

    SA_ASSERT(agg_part_rels.agg_size);
    SA_ASSERT(agg_part_rels.nparts > 0);
    min = max = agg_part_rels.agg_size[0];
    for(i=1; i < agg_part_rels.nparts; ++i)
    {
        const int curr_size = agg_part_rels.agg_size[i];
        if (max < curr_size)
            max = curr_size;
        if (min > curr_size)
            min = curr_size;
    }
    SA_PRINTF("Size of (big) aggregates, in DoFs, (min | avg | max):"
              " (%d | %g | %d)\n", min,
              (double)agg_part_rels.ND / (double)agg_part_rels.nparts, max);

    SA_ASSERT(agg_part_rels.elem_to_AE && agg_part_rels.AE_to_elem);
    min = max = agg_part_rels.AE_to_elem->RowSize(0);
    for(i=1; i < agg_part_rels.nparts; ++i)
    {
        const int curr_size = agg_part_rels.AE_to_elem->RowSize(i);
        if (max < curr_size)
            max = curr_size;
        if (min > curr_size)
            min = curr_size;
    }
    SA_PRINTF("Size of agglomerates, in elements, (min | avg | max):"
              " (%d | %g | %d)\n", min,
              (double)agg_part_rels.elem_to_AE->Size() /
              (double)agg_part_rels.nparts, max);

    SA_ASSERT(agg_part_rels.AE_to_dof);
    int total = 0;
    min = max = agg_part_rels.AE_to_dof->RowSize(0);
    for (i=0; i < agg_part_rels.nparts; ++i)
    {
        const int curr_size = agg_part_rels.AE_to_dof->RowSize(i);
        total += curr_size;
        if (max < curr_size)
            max = curr_size;
        if (min > curr_size)
            min = curr_size;
    }
    SA_PRINTF("Size of agglomerates, in DoFs, (min | avg | max):"
              " (%d | %g | %d)\n", min,
              (double)total / (double)agg_part_rels.nparts, max);
    SA_PRINTF("%s", ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
                    ">>>>>>>>>>>>>>>>>>>>>>>>>>>>\n");
}

#endif // _AGGREGATES_HPP
