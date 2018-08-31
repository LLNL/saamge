/*! \file
    \brief Aggregates and agglomerated elements (AEs) related functionality.

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

#include "SharedEntityCommunication.hpp"

namespace saamge
{

// let's try a forward declaration just for fun...
class ElementMatrixProvider;
class Arbitrator;

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
    mfem::Table *dof_to_elem; /*!< The relation table that relates DoFs to
                             elements. */
    mfem::Table *dof_to_dof; /*!< The relation table that relates DoFs to DoFs.
                            XXX: Currently it is used only when SA_DEBUG_LEVEL
                                 is sufficiently high! */

    mfem::Table *elem_to_dof; /*!< The relation table that relates elements to
                             DoFs. */
    mfem::Table *AE_to_elem; /*!< The relation table that relates AEs to elements. */
    mfem::Table *elem_to_AE; /*!< The relation table that relates elements to AEs. */
    mfem::Table *elem_to_elem; /*!< The relation table that relates elements to
                                    elements. */

    // Face relations are used in the nonconforming coarsening algorithms.
    // Coarse faces can be seen as MISes in terms of faces instead of DoFs. That is, MISes in the respective RT0 space.
    // In fact, once elements and faces are "broken" with the first coarsening, the usual DoF MISes and face MISes should
    // mark the same further coarse faces.
    mfem::Table *elem_to_face; /*!< The relation table that relates elements to
                                    their faces. */
    mfem::Table *face_to_dof; /*!< The relation table that relates faces to
                                   their DoFs. */
    mfem::Table *AE_to_face; /*!< The relation table that relates AEs to (fine) faces. */
    mfem::Table *cface_to_face; /*!< The relation table that relates coarse faces to fine faces. */
    mfem::Table *AE_to_cface; /*!< The relation table that relates AEs to coarse faces. */
    mfem::Table *cface_to_dof; /*!< The relation table that relates coarse faces to fine DoFs. */
    mfem::Table *dof_to_cface; /*!< The relation table that relates fine DoFs to coarse faces. */
    mfem::HypreParMatrix *face_to_trueface; /*!< On the finest level, this should be obtained from MFEM
                                                 otherwise it is built on coarse level, where coarse faces (agglomerated faces)
                                                 take the role of faces. */
    int num_owned_cfaces; /*!< The number of coarse faces locally owned on this processor. */
    int num_cfaces; /*!< number of coarse faces we know about on this processor, including some we share with others */
    int *cfaces; /*!< An array that for every (fine) face matches the coarse face it belongs to.
                      -1 is for internal faces that do not belong to any coarse face. */
    int *cfaces_size; /*!< An array with the sizes of the coarse (agglomerated) faces in number of (fine) faces. */
    int *cfaces_dof_size; /*!< An array with the sizes of the coarse (agglomerated) faces in number of (fine) DoFs. */
    int *cface_master; /*!< Tells you the master processor for this coarse face. */
    mfem::HypreParMatrix *cface_to_truecface; /*!< Connects local coarse faces to true global coarse faces. */
    mfem::HypreParMatrix *cface_cDof_TruecDof; /*!< Coarse DoF to coarse true DoF but restricted only to coarse faces. */
    mfem::HypreParMatrix *cface_TruecDof_cDof; /*!< Thee transposed of cface_cDof_TruecDof. */
    int *dof_num_gcfaces; /*!< An array that contains the number of global (across all CPUs) number of coarse faces that contain a DoF. */

    mfem::Table *AE_to_dof; /*!< The relation table that relates AEs to DoFs. */
    mfem::Table *dof_to_AE; /*!< The relation table that relates DoFs to AEs. */
    int *dof_id_inAE; /*!< An array used for mapping global DoF IDs to local
                           DoF IDs within any AE.
                           XXX: This array alone does not provide enough
                                information. It is combined with \b dof_to_AE
                                to get the final result. */
    int *dof_num_gAEs; /*!< An array that contains the number of global (across all CPUs) number of AEs that contain a DoF. */


    // TODO TODO TODO
    // to get three level aggregates in parallel, need to figure out agg_flags, figure out
    //     how to know when a dof is on processor boundary


    agg_dof_status_t *agg_flags; /*!< An array of flags for every DoF. (not related to aggregates, really...) */

    // ------ the following are proposed replacement for aggregates, with MISes ATB 30 March 2015
    mfem::Table *truemis_to_dof; /*!< The relation table that relates MISes to DoFs. */ 
    mfem::Table *mis_to_dof; /*!< not sure we need both of these */
    int *mis_master; /*!< tells you the master processor for this mis */ // our choice is basically lowest index...

    mfem::Table * mis_to_AE; // we like this one in contrib_mises and for building Dof_TrueDof
    mfem::Table * AE_to_mis; // we like this one in elmat_parallel()
    mfem::HypreParMatrix * mis_truemis;
    int num_owned_mises; /*!< The number of MISes. (locally owned on this processor) */ 
    int num_mises; /*!< number of MISes we know about on this processor, including some we share with others */
    int *mises; /*!< An array that for every DoF matches the MIS it belongs
                  to. */ 
    int *mises_size; /*!< An array with the sizes of the MISes in number of
                       (fine) DoFs. */ 

    int *mis_coarsedofoffsets; /*!< keep track of which MIS has which coarse dof, might be more useful to have a full mis_coarsedof mfem::Table (this lives on coarse agg_part_rels, empty on the finest level) */

    int *dof_masterproc;
    // ------ end proposed replacement

    mfem::HypreParMatrix * Dof_TrueDof; /*!< on the finest level, just points to fes->Dof_TrueDof_Matrix();
                                    otherwise we build it. */
    bool owns_Dof_TrueDof;
    bool testmesh; /*!< whether to use the little 12-element test mesh we sometimes use for debugging */

} agg_partitioning_relations_t;

/* Functions */

void agg_construct_agg_flags(agg_partitioning_relations_t& agg_part_rels,
                             const agg_dof_status_t *bdr_dofs);

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
int *agg_preconstruct_aggregates(const mfem::Table& dof_to_AE, int nparts,
                                 const agg_dof_status_t *bdr_dofs,
                                 int *& preagg_size,
                                 agg_dof_status_t *& agg_flags);

/*! \brief Full initial construction of the aggregates.

    Includes preconstruction and resolution of non-trivial DoFs.

    \param A (IN) The global stiffness matrix.
    \param agg_part_rels (IN/OUT) The partitioning relations.
    \param bdr_dofs (IN) An array that shows if a DoF i is on essential domain
                         border. It can be NULL.
*/
void agg_construct_aggregates(const mfem::SparseMatrix& A,
                              agg_partitioning_relations_t& agg_part_rels,
                              const agg_dof_status_t *bdr_dofs);

/*! \brief Assembles the local stiffness matrix for an AE on the finest mesh.

    Uses entries in the global stiffness matrix for better performance. Also,
    takes care of essential boundary conditions.

    Initially implemented for the finest (geometric) mesh.

    Note that "global" here is slightly misleading in that the matrix we look
    at is local to this processor, but is global in the sense of containing a 
    bunch of AEs, while the local stiffness matrix that comes out is really
    local to one AE.

    \param A (IN) The global stiffness matrix.
    \param part (IN) The id (number) of the AE in question.
    \param agg_part_rels (IN) The partitioning relations.
    \param data (IN/OUT) \a elmat_callback specific data, provides element matrices
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
mfem::SparseMatrix *agg_build_AE_stiffm_with_global(
    const mfem::SparseMatrix& A, int part,
    const agg_partitioning_relations_t& agg_part_rels,
    const ElementMatrixProvider *data, bool bdr_cond_imposed,
    bool assemble_ess_diag);

/*! \brief Assembles the local stiffness matrix for an AE.

    Uses only element matrices and not the global stiffness matrix.

    Initially implemented for the coarse (non-geometric) meshes. No boundary
    conditions are taken into account.

    \param part (IN) The id (number) of the AE in question.
    \param agg_part_rels (IN) The partitioning relations.
    \param data (IN/OUT) \a elmat_callback specific data, provides element matrices

    \returns The local stiffness matrix for the AE in question.

    \warning The returned sparse matrix must be freed by the caller.
*/
mfem::SparseMatrix *agg_build_AE_stiffm(
    int part, const agg_partitioning_relations_t& agg_part_rels,
    const ElementMatrixProvider *data);

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
int *agg_restrict_to_agg(int part, const mfem::Table& AE_to_dof,
                         const int *aggregates, int agg_id, int agg_size,
                         mfem::DenseMatrix& cut_evects, mfem::DenseMatrix& restricted);

/*! \brief Restricts a group of vectors to an aggregate,
  given an already computed restriction (returned from agg_restrict_to_agg()
*/
void agg_restrict_to_agg_enforce(int part,
                                 const agg_partitioning_relations_t& agg_part_rels, int agg_size,
                                 const int *restriction, mfem::DenseMatrix& cut_evects,
                                 mfem::DenseMatrix& restricted);

/*! \brief The same as agg_restrict_to_agg_enforce but only for a single vector.
*/
void agg_restrict_vec_to_agg_enforce(
    int part,
    const agg_partitioning_relations_t& agg_part_rels, int agg_size,
    const int *restriction, const mfem::Vector& vec,
    mfem::Vector& restricted);

/*! \brief Restricts a globally defined vector to an AE.

    \param part (IN) The id (number) of the AE in question.
    \param agg_part_rels (IN) The partitioning relations.
    \param glob_vect (IN) The vector, defined globally, that will be restricted
                          to the AE.
    \param restricted (OUT) The restricted, to the AE, vector.
*/
void agg_restrict_vect_to_AE(int part,
                             const agg_partitioning_relations_t& agg_part_rels,
                             const mfem::Vector& glob_vect, mfem::Vector& restricted);

/*! \brief Builds structures for mapping global DoF IDs to local (AE) IDs.

    \param agg_part_rels (IN) The partitioning relations.
*/
void agg_build_glob_to_AE_id_map(agg_partitioning_relations_t& agg_part_rels);

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
    \param dof_truedof Table for parallel implementation, on fine level (ie, this
                       function) caller owns this and must free it.
    \param do_aggregates whether to do aggregates or the usual MISes. If
                       you choose aggregates, you cannot coarsen further.

    \returns A structure with the partitioning relations.

    \warning The returned structure must be freed by the caller by calling
             \b agg_free_partitioning.
*/
agg_partitioning_relations_t *
agg_create_partitioning_fine(mfem::HypreParMatrix& A, int NE, mfem::Table *elem_to_dof,
                             mfem::Table *elem_to_elem, int *partitioning,
                             const agg_dof_status_t *bdr_dofs, int *nparts,
                             mfem::HypreParMatrix *dof_truedof,
                             bool do_aggregates, bool testmesh=false);

/*! \brief like agg_create_partitioning_fine,

  but isolate some cells in their own partition
*/
agg_partitioning_relations_t *
agg_create_partitioning_fine_isolate(
    mfem::HypreParMatrix& A, int NE, mfem::Table *elem_to_dof,
    mfem::Table *elem_to_elem, int *partitioning,
    const agg_dof_status_t *bdr_dofs, int *nparts,
    mfem::HypreParMatrix *dof_truedof,
    const mfem::Array<int>& isolated_cells);

/*! \brief Finish agg_create_partitioning_fine

  Added this just to split agg_create_partitioning_fine into pieces
  and avoid copying too much code.
*/
void agg_create_partitioning_tables(
    agg_partitioning_relations_t * agg_part_rels,
    mfem::HypreParMatrix& Alocal, int NE, mfem::Table *elem_to_dof,
    mfem::Table *elem_to_elem, int *partitioning,
    const agg_dof_status_t *bdr_dofs, int *nparts,
    mfem::HypreParMatrix *dof_truedof, bool do_aggregates);

/*! \brief Constructs only the face relations.

  This function counts on already built relation tables (like AEs) and only
  fills in the face and coarse face relations by constructing the coarse faces.
  The arguments should be obtained from MFEM on the finest level, while on the coarsest level they are produced
  by replacing element with AE and face with coarse face (coarse DoF also should appear at some point).
  The arguments become possessions of the agg_part_rels structure.
*/
void agg_build_face_relations(agg_partitioning_relations_t *agg_part_rels, mfem::Table *elem_to_face, mfem::Table *face_to_dof, mfem::HypreParMatrix *face_to_trueface);

/*! \brief builds finedof_to_dof, a local Table, based on some parallel relations

    fairly experimental
*/
mfem::Table * agg_create_finedof_to_dof(agg_partitioning_relations_t& agg_part_rels,
                                  const agg_partitioning_relations_t& agg_part_rels_fine,
                                  mfem::HypreParMatrix * interp);

/*! \brief Creates all relations except a few element and AE related.

    The relations that are not created here must exist prior to calling this
    function.

    \param A (IN) The global stiffness matrix (on the current "fine" level that
                  is to be coarsened).
    \param agg_part_rels (IN/OUT) The relations structure to be filled in.
    \param agg_part_rels_fine (IN) The relations from the level right above the
                                   considered one.
    \param interp (IN) The (tentative) interpolation operator that produced the
                       currently considered level.
    \param dorefaggs (IN) Whether to build MISes and refined aggregates.
*/
void agg_create_rels_except_elem_coarse(
    mfem::HypreParMatrix * A,
    agg_partitioning_relations_t& agg_part_rels,
    const agg_partitioning_relations_t& agg_part_rels_fine,
    mfem::HypreParMatrix * interp);

void agg_build_coarse_Dof_TrueDof(agg_partitioning_relations_t &agg_part_rels_coarse,
                                  const agg_partitioning_relations_t &agg_part_rels_fine,
                                  int coarse_truedof_offset, int * mis_numcoarsedof,
                                  mfem::DenseMatrix ** mis_tent_interps);

/*! Builds and returns the coarse DoF to coarse true DoF but restricted only to coarse faces
    (unless the fullrelations flag is set, which enforces the inclusion of interior DoFs).
    This is suited for nonconforming methods, where the coarse faces are the only ones that have shared DoFs.
*/
mfem::HypreParMatrix *agg_build_cface_cDof_TruecDof(const agg_partitioning_relations_t &agg_part_rels,
                                                    int coarse_truedof_offset, mfem::DenseMatrix **cfaces_bases,
                                                    int celements_cdofs=0, bool fullrelations=false);

/*! A wrapper for \b agg_build_cface_cDof_TruecDof, that simply fills in the respective
    relations inside \a agg_part_rels.
*/
void agg_create_cface_cDof_TruecDof_relations(agg_partitioning_relations_t &agg_part_rels,
                                              int coarse_truedof_offset, mfem::DenseMatrix **cfaces_bases,
                                              int celements_cdofs=0, bool fullrelations=false);

/*! \brief Creates all relations for a non-geometric level.

    The function is a wrapper that uses several other functions to do the
    partitioning; generate AEs, corresponding aggregates and all needed
    relations and structures.

    It is only useful when coarsening is desired i.e. for a level which is not
    the coarsest one.

    \param A (IN) The global stiffness matrix (on the current "fine" level that
                  is to be coarsened).
    \param agg_part_rels_fine (IN) The relations from the level right above the
                                   considered one.
    \param interp (IN) The (tentative) interpolation operator that produced the
                       currently considered level.
    \param nparts (IN/OUT) The number of partitions (AEs, aggregates) in the
                           partitioning. It inputs the desired number of
                           partitions and outputs the actual number of
                           generated partitions.
    \param do_aggregates (IN) Whether to do aggregates or the usual MISes
                         Note that if you do aggregates, you cannot
                         coarsen any further.

    \returns A structure with the partitioning relations.

    \warning The returned structure must be freed by the caller by calling
             \b agg_free_partitioning.
*/
agg_partitioning_relations_t *
agg_create_partitioning_coarse(
    mfem::HypreParMatrix* A,
    const agg_partitioning_relations_t& agg_part_rels_fine,
    int coarse_truedof_offset,
    int * mis_numcoarsedof,
    mfem::DenseMatrix ** mis_tent_interps,
    mfem::HypreParMatrix * interp,
    int *nparts,
    bool do_aggregates);

/*! \brief Frees a partitioning relations structure.

    \param agg_part_rels (IN) The structure to be freed.
*/
void agg_free_partitioning(agg_partitioning_relations_t *agg_part_rels);

/*! \brief Makes a copy of partitioning data.

    \param src (IN) The partitioning data to be copied.

    \returns A copy of \a src.

    \warning The returned structure must be freed by the caller using
             \b agg_free_partitioning.
*/
agg_partitioning_relations_t
    *agg_copy_partitioning(const agg_partitioning_relations_t *src);

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
bool agg_is_dof_on_essential_border(const agg_partitioning_relations_t&
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
                                   mfem::Table *& A_to_B, mfem::Table *& B_to_A);

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
int agg_num_col_to_glob(const mfem::Table& col_to_elem, int col, int loc_num);

/*! \brief Returns if an element (general) is in a given collection (general).

    \param elem (IN) The id of the element.
    \param col (IN) The id of the collection.
    \param elem_to_col (IN) mfem::Table that relates elements to collections.

    \returns If the element is in the collection it returns the order of the
             'collection' in the 'element' row in the CSR representation of
             \a elem_to_col. If the element is NOT in the collection, a
             negative number is returned.
*/
static inline
int agg_elem_in_col(int elem, int col, const mfem::Table& elem_to_col);

/*! \brief Returns the local (AE) ID of a DoFs, given the global ID.

    \param glob_id (IN) The global ID of the DoF.
    \param part (IN) The AE of interest.
    \param agg_part_rels (IN) The partitioning relations.

    \returns If the DoF is in the AE of interest its local ID is returned.
             Otherwise, a negative number is returned.
*/
static inline
int agg_map_id_glob_to_AE(int glob_id, int part,
                          const agg_partitioning_relations_t& agg_part_rels);

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
void agg_eliminate_essential_bc(mfem::SparseMatrix& A,
                                const agg_dof_status_t *bdr_dofs,
                                const mfem::Vector& sol, mfem::Vector& rhs, bool keep_diag);

/*! \brief Prints information about the partitionings.

    \param agg_part_rels (IN) The partitioning relations.
*/
static inline
void agg_print_data(const agg_partitioning_relations_t& agg_part_rels);

/* Inline Functions Definitions */
static inline
bool agg_is_dof_on_essential_border(
    const agg_partitioning_relations_t& agg_part_rels,int dof_id)
{
    return SA_IS_SET_A_FLAG(agg_part_rels.agg_flags[dof_id],
                            AGG_ON_ESS_DOMAIN_BORDER_FLAG);
}

static inline
void agg_construct_tables_from_arr(const int *A_to_B_arr, int num_A,
                                   mfem::Table *& A_to_B, mfem::Table *& B_to_A)
{
    SA_ASSERT(A_to_B_arr);
    A_to_B = new mfem::Table(num_A, const_cast<int *>(A_to_B_arr));
    B_to_A = new mfem::Table();
    Transpose(*A_to_B, *B_to_A);
}

static inline
int agg_num_col_to_glob(const mfem::Table& col_to_elem, int col, int loc_num)
{
    SA_ASSERT(loc_num < col_to_elem.RowSize(col));
    return col_to_elem.GetRow(col)[loc_num];
}

static inline
int agg_elem_in_col(int elem, int col, const mfem::Table& elem_to_col)
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
                          const agg_partitioning_relations_t& agg_part_rels)
{
    SA_ASSERT(agg_part_rels.AE_to_dof);
    SA_ASSERT(agg_part_rels.dof_to_AE);
    SA_ASSERT(agg_part_rels.dof_id_inAE);
    SA_ASSERT(0 <= part && part < agg_part_rels.AE_to_dof->Size());
    SA_ASSERT(0 <= glob_id && glob_id < agg_part_rels.ND);

    const mfem::Table& dof_to_AE = *agg_part_rels.dof_to_AE;
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
void agg_eliminate_essential_bc(mfem::SparseMatrix& A,
                                const agg_dof_status_t *bdr_dofs,
                                const mfem::Vector& sol, mfem::Vector& rhs, bool keep_diag)
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

/**
   Not sure this is printing the right thing since MIS -> aggregate 
   change.
*/
static inline
void agg_print_data(const agg_partitioning_relations_t& agg_part_rels)
{
    int i, min, max;
    SA_RPRINTF(0,"%s", ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
               ">>>>>>>>>>>>>>>>>>>>>>>>>>>>\n");
    SA_RPRINTF(0,"%s", "\tPartitions (aggregates and agglomerates):\n");
    SA_RPRINTF(0,"Local agglomerates: %d\n",
               agg_part_rels.nparts);

    int global_agglomerates;
    int nparts = agg_part_rels.nparts;
    MPI_Reduce(&nparts, &global_agglomerates, 1, 
               MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    SA_RPRINTF(0,"Global agglomerates: %d\n", global_agglomerates);

    if (agg_part_rels.mises_size)
    {
        SA_RPRINTF(0,"Local MISes: %d\n", agg_part_rels.num_owned_mises);
        min = max = agg_part_rels.mises_size[0];
        for (i=1; i< agg_part_rels.num_owned_mises; ++i)
        {
            const int curr_size = agg_part_rels.mises_size[i];
            if (max < curr_size)
                max = curr_size;
            if (min > curr_size)
                min = curr_size;
        }
        SA_RPRINTF(0, "Size of MISes, in DoFs, (min | avg | max):"
                   " (%d | %g | %d)\n", min,
                   (double)agg_part_rels.ND / (double)agg_part_rels.num_owned_mises, max);
    }
    SA_ASSERT(agg_part_rels.elem_to_AE && agg_part_rels.AE_to_elem);
    min = max = agg_part_rels.AE_to_elem->RowSize(0);
    for (i=1; i < agg_part_rels.nparts; ++i)
    {
        const int curr_size = agg_part_rels.AE_to_elem->RowSize(i);
        if (max < curr_size)
            max = curr_size;
        if (min > curr_size)
            min = curr_size;
    }
    SA_RPRINTF(0,"Size of agglomerates, in elements, (min | avg | max):"
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
    SA_RPRINTF(0,"Size of agglomerates, in DoFs, (min | avg | max):"
              " (%d | %g | %d)\n", min,
              (double)total / (double)agg_part_rels.nparts, max);
    SA_RPRINTF(0, "%s", ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
               ">>>>>>>>>>>>>>>>>>>>>>>>>>>>\n");
}

} // namespace saamge

#endif // _AGGREGATES_HPP
