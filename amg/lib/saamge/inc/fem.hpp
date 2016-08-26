/*! \file
    \brief Finite elements (FE) related functionality and a bit more.

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
#ifndef _FEM_HPP
#define _FEM_HPP

#include "common.hpp"
#include <mfem.hpp>
#include "mfem_addons.hpp"
#include "aggregates.hpp"
#include "elmat.hpp"

using namespace mfem;

/* Options */

/*! \brief The configuration class of this module.
*/
CONFIG_BEGIN_CLASS_DECLARATION(FEM)

    /*! The port where GLVIS listens. */
    CONFIG_DECLARE_OPTION(unsigned int, glvis_port);

    /*! Whether to keep the diagonal matrix entry when imposing essential
        boundary conditions. If not kept, it will be set to 1.*/
    CONFIG_DECLARE_OPTION(bool, keep_diag);

CONFIG_END_CLASS_DECLARATION(FEM)

CONFIG_BEGIN_INLINE_CLASS_DEFAULTS(FEM)
    CONFIG_DEFINE_OPTION_DEFAULT(glvis_port, 19916),
    CONFIG_DEFINE_OPTION_DEFAULT(keep_diag, true)
CONFIG_END_CLASS_DEFAULTS

/* Defines */
/*! \def FEM_ALLOC_FEC_FES
    \brief Allocations of FE space and collection.

    Creates and allocates the FE collection, with name given by \a fec_ptr_out,
    and FE space, with name given by \a fes_ptr_out. These names are names of
    pointers. The type of the FE collection is taken from \a fe_coll_type_in.

    \param fe_coll_type_in (IN) The type of the FE collection
                                (e.g LinearFECollection).
    \param mesh_ptr_in (IN) A pointer to the mesh on which the space and
                            collection will be defined.
    \param fec_ptr_out (IN/OUT) A name of the pointer to the created
                                collection.
    \param fes_ptr_out (IN/OUT) A name of the pointer to the created
                                space.

    \warning \a fec_ptr_out and \a fes_ptr_out must be freed by the caller
             using \b FEM_FREE_FEC_FES.
*/
/*! \def FEM_ALLOC_PWC_FEC_FES
    \brief Allocations of piece-wise constant FE space and collection.

    Creates and allocates the FE collection, with name given by \a fec_ptr_out,
    and FE space, with name given by \a fes_ptr_out. These names are names of
    pointers.

    \param mesh_ptr_in (IN) A pointer to the mesh on which the space and
                            collection will be defined.
    \param fec_ptr_out (IN/OUT) A name of the pointer to the created
                                collection.
    \param fes_ptr_out (IN/OUT) A name of the pointer to the created
                                space.

    \warning \a fec_ptr_out and \a fes_ptr_out must be freed by the caller
             using \b FEM_FREE_FEC_FES.
*/
/*! \def FEM_FREE_FEC_FES
    \brief Frees FE space and collection.

    \param fec_ptr (IN) A name of the pointer to the previously created
                        collection.
    \param fes_ptr (IN) A name of the pointer to the previously created space.

    \warning \a fec_ptr_out and \a fes_ptr_out are usually created using
             \b FEM_ALLOC_LIN_FEC_FES.
*/
#define FEM_ALLOC_FEC_FES(fe_coll_type_in, mesh_ptr_in, fec_ptr_out, \
                          fes_ptr_out) \
    FiniteElementCollection * fec_ptr_out = new fe_coll_type_in; \
    ParFiniteElementSpace * fes_ptr_out = \
        new ParFiniteElementSpace(mesh_ptr_in, fec_ptr_out)

#define FEM_ALLOC_PWC_FEC_FES(mesh_ptr_in, fec_ptr_out, fes_ptr_out) \
    FiniteElementCollection * fec_ptr_out = (mesh_ptr_in->Dimension() == 2 ? \
                  (FiniteElementCollection *)new Const2DFECollection : \
                  (FiniteElementCollection *)new Const3DFECollection); \
    ParFiniteElementSpace * fes_ptr_out = \
        new ParFiniteElementSpace(mesh_ptr_in, fec_ptr_out)

#define FEM_FREE_FEC_FES(fec_ptr, fes_ptr) \
do { \
    delete fes_ptr; \
    delete fec_ptr; \
} while (0)

/* Functions */
/*! \brief Refines mesh to a certain upper bound for the number of elements.

    \param max_num_elems (IN) An upper bound for the number of elements.
    \param mesh (IN/OUT) The mesh to be refined.

    \warning \a mesh should be created prior to calling this function.
*/
void fem_refine_mesh(int max_num_elems, Mesh& mesh);

/*! \brief Refines mesh a certain number of times.

    \param times (IN) Number of times to refine the mesh.
    \param mesh (IN/OUT) The mesh to be refined.

    \warning \a mesh should be created prior to calling this function.
*/
void fem_refine_mesh_times(int times, Mesh& mesh);

/*! \brief Initializes \a x on space \a fespace with \a bdr_coeff.

    \param x (OUT) The grid function to be initialized.
    \param fespace (IN) The FE space to be used.
    \param bdr_coeff (IN) The coefficient to be projected on the grid function
                          \a x. It defines a border condition.
*/
void fem_init_with_bdr_cond(ParGridFunction& x, ParFiniteElementSpace *fespace,
                            Coefficient& bdr_coeff);

/*! \brief Finds the border DoFs with essential boundary conditions.

    Finds the border DoFs (the ones on the boundary of the domain) with
    essential (Dirichlet) boundary conditions.

    \param fes (IN) The finite element space.
    \param ess_bdr (IN) An array of flags. If the i-th element of the array is
                        \em true then the border elements with attribute (i+1)
                        are considered to have essential (Dirichlet) boundary
                        conditions on them. If it's \em false, the boundary
                        with that attribute is considered free.

    \returns A array mapping on each DoF a non-zero value if it is on a part of
             the border with essential boundary conditions or on the interface
             between processes.

    \warning The returned array must be freed by the caller.
*/
agg_dof_status_t *fem_find_bdr_dofs(ParFiniteElementSpace& fes,
                                    Array<int> *ess_bdr);

/*! \brief Generates a right-hand side linear form.

    \param fespace (IN) The finite element space.
    \param rhs (IN) The coefficient defining the right-hand side.

    \returns The linear form for the right-hand side.

    \warning The returned linear form must be freed by the caller.
*/
ParLinearForm *fem_assemble_rhs(ParFiniteElementSpace *fespace,
                                Coefficient& rhs);

/*! \brief Visualizes a grid function using GLVIS in serial.

    \param mesh (IN) The mesh on which the grid function is defined.
    \param x (IN) The grid function to be visualized.
    \param keys (IN) Instructions for GLVIS (as keys to "press").

    \warning GLVIS daemon must be running on localhost. The port it is bound to
             is given by the \b glvis_port option.
    \warning The precision of the output is determined by the \b prec option.
*/
void fem_serial_visualize_gf(const Mesh& mesh, GridFunction& x,
                             const char *keys="");

/*! \brief Visualizes a grid function using GLVIS in parallel.

    \param mesh (IN) The mesh on which the grid function is defined.
    \param x (IN) The grid function to be visualized.
    \param keys (IN) Instructions for GLVIS (as keys to "press").

    \warning GLVIS daemon must be running on localhost. The port it is bound to
             is given by the \b glvis_port option.
    \warning The precision of the output is determined by the \b prec option.
*/
void fem_parallel_visualize_gf(const ParMesh& mesh, ParGridFunction& x,
                               const char *keys="");

/*! \brief Visualizes piece-wise constant coefficient using GLVIS in serial.

    \param mesh (IN) The mesh on which the grid function is defined.
    \param coef (IN) The coefficient.
    \param keys (IN) Instructions for GLVIS (as keys to "press").

    \warning GLVIS daemon must be running on localhost. The port it is bound to
             is given by the \b glvis_port option.
    \warning The precision of the output is determined by the \b prec option.
*/
void fem_serial_visualize_pwc_coef(Mesh& mesh, Coefficient& coef,
                                   const char *keys="");

/*! \brief Visualizes piece-wise constant coefficient using GLVIS in parallel.

    \param mesh (IN) The mesh on which the grid function is defined.
    \param coef (IN) The coefficient.
    \param keys (IN) Instructions for GLVIS (as keys to "press").

    \warning GLVIS daemon must be running on localhost. The port it is bound to
             is given by the \b glvis_port option.
    \warning The precision of the output is determined by the \b prec option.
*/
void fem_parallel_visualize_pwc_coef(ParMesh& mesh, Coefficient& coef,
                                     const char *keys="");

/*! \brief Visualizes the partitioning of a mesh in serial.

    Shows the partitions themselves in different colors.

    \param mesh (IN) The mesh.
    \param partitioning (IN) The partitioning (in the format given by
                             \b fem_create_partitioning).
    \param keys (IN) Instructions for GLVIS (as keys to "press").

    \warning GLVIS daemon must be running on localhost. The port it is bound to
             is given by the \b glvis_port option.
    \warning The precision of the output is determined by the \b prec option.
*/
void fem_serial_visualize_partitioning(Mesh& mesh, int *partitioning,
                                       const char *keys="");

/*! \brief Visualizes the partitioning of a mesh in parallel.

    Shows the partitions themselves in different colors.

    \param mesh (IN) The mesh.
    \param parts (IN) The number of parts in the current process.
    \param partitioning (IN) The partitioning (in the format given by
                             \b fem_create_partitioning).
    \param keys (IN) Instructions for GLVIS (as keys to "press").

    \warning GLVIS daemon must be running on localhost. The port it is bound to
             is given by the \b glvis_port option.
    \warning The precision of the output is determined by the \b prec option.
*/
void fem_parallel_visualize_partitioning(ParMesh& mesh, int *partitioning,
                                         int parts, const char *keys="");

/*! \brief Visualizes the aggregates in serial.

    Shows the aggregates in different colors.

    \param fes (IN) The FE space used.
    \param aggregates (IN) The aggregates (as given by
                           \b fem_create_partitioning or
                           \b agg_construct_aggregates).
    \param keys (IN) Instructions for GLVIS (as keys to "press").

    \warning GLVIS daemon must be running on localhost. The port it is bound to
             is given by the \b glvis_port option.
    \warning The precision of the output is determined by the \b prec option.
*/
void fem_serial_visualize_aggregates(FiniteElementSpace *fes, int *aggregates,
                                     const char *keys="");

/*! \brief Visualizes the aggregates in parallel.

    Shows the aggregates in different colors.

    \param fes (IN) The FE space used.
    \param aggregates (IN) The aggregates (as given by
                           \b fem_create_partitioning or
                           \b agg_construct_aggregates).
    \param parts (IN) The number of parts in the current process.
    \param keys (IN) Instructions for GLVIS (as keys to "press").

    \warning GLVIS daemon must be running on localhost. The port it is bound to
             is given by the \b glvis_port option.
    \warning The precision of the output is determined by the \b prec option.
*/
void fem_parallel_visualize_aggregates(ParFiniteElementSpace *fes,
                                       int *aggregates, int parts,
                                       const char *keys="");

/*! \brief Loads a mesh from a file. The format is determined by MFEM.

    \param filename (IN) The name of the file with the mesh.

    \returns The loaded mesh.

    \warning The returned mesh must be freed by the caller.
*/
Mesh *fem_read_mesh(const char *filename);

/*! \brief Writes a mesh to a file. The format is determined by MFEM.

    \param filename (IN) The name of the file to write. If it exists, it will
                         be erased prior to writing.
    \param mesh (IN) The mesh to be written.

    \warning The precision of the output is determined by the \b prec option.
*/
void fem_write_mesh(const char *filename, const Mesh& mesh);

/*! \brief Loads a grid function from a file. The format is determined by MFEM.

    \param filename (IN) The name of the file with the grid function.
    \param mesh (IN) The mesh on which the grid function will be defined.

    \returns The loaded grid function.

    \warning The returned grid function must be freed by the caller.
*/
GridFunction *fem_read_gf(const char *filename, Mesh *mesh);

/*! \brief Writes a grid function to a file. The format is determined by MFEM.

    \param filename (IN) The name of the file to write. If it exists, it will
                         be erased prior to writing.
    \param gf (IN) The grid function to be written.

    \warning The precision of the output is determined by the \b prec option.
*/
void fem_write_gf(const char *filename, GridFunction& gf);

/* Function Templates */
/*! \brief Assembles the global stiffness matrix by generating a bilinear form.

    \em T is usually \b Coefficient or \b MatrixCoefficient, or any of their
    derivative classes.

    \param fespace (IN) The finite element space.
    \param coeff (IN) The coefficient \f$k\f$ in
                      \f$-\textrm{div}\left( k \nabla p \right)\f$.
    \param x (IN) The initial solution initialized with the boundary
                  conditions (if to be imposed). It needs to have \a fespace as
                  a FE space.
    \param b (IN/OUT) The right-hand side. It is going to be updated as the
                      boundary conditions are being imposed (if to be imposed).
                      This usually comes from \b fem_assemble_rhs.
    \param bdr_cond_impose (IN) A flag whether to impose the boundary
                                conditions on the global stiffness matrix.
    \param ess_bdr (IN) An array of flags. If the i-th element of the array is
                        \em true then the border elements with attribute (i+1)
                        are considered to have essential (Dirichlet) boundary
                        conditions on them. If it's \em false, the boundary
                        with that attribute is considered free. May be NULL if
                        \a bdr_cond_impose is \em false.

    \returns The bilinear form for global stiffness matrix.

    \warning The returned bilinear form must be freed by the caller.
    \warning This function is imposing everywhere essential boundary condition
             on the global stiffness matrix.
*/
template <class T>
ParBilinearForm *fem_assemble_stiffness(ParFiniteElementSpace *fespace,
                                        T& coeff, Vector& x, Vector& b,
                                        bool bdr_cond_impose,
                                        Array<int> *ess_bdr);

/*! \brief Build everything needed for the solver to be used.

    This is a wrapper function that calls several other functions to assemble
    the right-hand side, the global stiffness matrix etc. This is all that is
    necessary for the solver to be called.

    \em T is usually \b Coefficient or \b MatrixCoefficient, or any of their
    derivative classes.

    \param fespace (IN) The finite element space.
    \param rhs (IN) The coefficient defining the right-hand side.
    \param bdr_coeff (IN) The coefficient to be projected on the grid function
                          \a x. It defines a border condition.
    \param coeff (IN) The coefficient \f$k\f$ in
                      \f$-\textrm{div}\left( k \nabla p \right)\f$.
    \param bdr_cond_impose (IN) A flag whether to impose the boundary
                                conditions on the global stiffness matrix and
                                whether to initialize x with
    \param x (OUT) This may be used as an initial guess. It gets initialized
                   with the border condition (if to be imposed). It's FE space
                   is set to fespace.
    \param b (OUT) The right-hand side. MUST BE FREED BY THE CALLER!
    \param a (OUT) The bilinear form for the global stiffness matrix. MUST BE
                   FREED BY THE CALLER!
    \param ess_bdr (IN) An array of flags. If the i-th element of the array is
                        \em true then the border elements with attribute (i+1)
                        are considered to have essential (Dirichlet) boundary
                        conditions on them. If it's \em false, the boundary
                        with that attribute is considered free. May be NULL if
                        \a bdr_cond_impose is \em false.

    \warning This function is imposing everywhere essential boundary condition
             on the global stiffness matrix.
*/
template <class T>
void fem_build_discrete_problem(ParFiniteElementSpace *fespace,
                                Coefficient& rhs, Coefficient& bdr_coeff,
                                T& coeff, bool bdr_cond_impose,
                                ParGridFunction& x,
                                ParLinearForm*& b, ParBilinearForm*& a,
                                Array<int> *ess_bdr);

/* Inline Functions */
/*! \brief Partitions the unweighted dual graph of the mesh.

    \param mesh (IN) The mesh to partition.
    \param parts (IN/OUT) The desired number of partitions in the partitioning,
                          as input. As output: the number of non-empty
                          partitions, which is the number of actually generated
                          partitions.

    \returns The partitioning of the dual graph of the mesh.

    \warning The returned array must be freed by the caller.
*/
static inline
int *fem_partition_mesh(Mesh& mesh, int *nparts);

/*! \brief Utility routine, to deal with vector-valued problems.

    Converts an mfem elem_to_dof table to one that SAAMGe can use
    algebraically without knowing about the vector structure.
*/
Table* vector_valued_elem_to_dof(const Table& mfem_elem_to_dof,
                                 const int vdim, const int ordering);

/*! \brief Partitions a Cartesian 2D mesh into rectangles.

    This function is used to do element agglomeration on slices of
    the SPE10 test problem.
*/
int *fem_partition_dual_simple_2D(Mesh& mesh, int *nparts, int *nparts_x,
                                  int *nparts_y);

/*! \brief Creates all relations on the finest (geometric) mesh.

    The function is a wrapper that uses several other functions to do the
    partitioning; generate AEs, corresponding aggregates and all needed
    relations and structures.

    \param A (IN) The global stiffness matrix.
    \param fes (IN) The used finite element space. A pointer to it will stay in
                    the returned structure. However, the caller owns the finite
                    element space and must free it when the returned structure
                    is freed.
    \param bdr_dofs (IN) An array that shows if a DoF i is on the domain border.
    \param nparts (IN/OUT) The number of partitions (AEs, aggregates) in the
                           partitioning. It inputs the desired number of
                           partitions and outputs the actual number of
                           generated partitions.

    \returns A structure with the partitioning relations.

    \warning The returned structure must be freed by the caller by calling
             \b agg_free_partitioning.
*/
agg_partitioning_relations_t *
fem_create_partitioning(HypreParMatrix& A, ParFiniteElementSpace& fes,
                        const agg_dof_status_t *bdr_dofs, int *nparts,
                        bool do_aggregates);

agg_partitioning_relations_t *
fem_create_partitioning_from_matrix(const SparseMatrix& A,
                                    int *nparts,
                                    HypreParMatrix *dof_truedof,
                                    Array<int>& isolated_cells);

/* Function Templates Definitions */
template <class T>
ParBilinearForm *fem_assemble_stiffness(ParFiniteElementSpace *fespace,
                                        T& coeff, Vector& x, Vector& b,
                                        bool bdr_cond_impose,
                                        Array<int> *ess_bdr)
{
    SA_RPRINTF_L(0, 4, "%s", "Assembling global stiffness matrix...\n");
    ParBilinearForm *a = new ParBilinearForm(fespace);
    a->AddDomainIntegrator(new DiffusionIntegrator(coeff));
    a->Assemble(/*0*/);
    if (bdr_cond_impose)
    {
        SA_RPRINTF_L(0, 4, "%s", "Imposing boundary conditions...\n");
        /* Imposing Dirichlet boundary conditions. */
        SA_ASSERT(ess_bdr->Size() == fespace->GetMesh()->bdr_attributes.Max());
        Array<int> ess_dofs;
        fespace->GetEssentialVDofs(*ess_bdr, ess_dofs);
        a->EliminateEssentialBCFromDofs(ess_dofs, x, b,
                                    (int)CONFIG_ACCESS_OPTION(FEM, keep_diag));
    }
    SA_RPRINTF_L(0, 4, "%s", "Finalizing global stiffness matrix...\n");
    a->Finalize(0);

    return a;
}

template <class T>
void fem_build_discrete_problem(ParFiniteElementSpace *fespace,
                                Coefficient& rhs, Coefficient& bdr_coeff,
                                T& coeff, bool bdr_cond_impose,
                                ParGridFunction& x,
                                ParLinearForm*& b, ParBilinearForm*& a,
                                Array<int> *ess_bdr)
{
    SA_RPRINTF_L(0, 4, "%s", "Building discrete problem...\n");

    // Define the solution vector x as a finite element grid function
    // corresponding to fespace. Initialize x in such a way that it satisfies
    // the boundary conditions.
    if (bdr_cond_impose)
    {
        fem_init_with_bdr_cond(x, fespace, bdr_coeff);
    } 
    else
    {
        x.Update(fespace);
    }

    // Set up the linear form b(.) which corresponds to the right-hand side
    // of the FEM linear system, which in this case is (rhs_func,phi_i), where
    // phi_i are the basis functions in the finite element space, fespace.
    b = fem_assemble_rhs(fespace, rhs);

    // Set up the bilinear form a(.,.) on the finite element space
    // corresponding to the Laplacian operator -Delta, by adding the Diffusion
    // domain integrator.
    a = fem_assemble_stiffness(fespace, coeff, x, *b, bdr_cond_impose, ess_bdr);
}

/* Inline Functions Definitions */
static inline
int *fem_partition_mesh(Mesh& mesh, int *nparts)
{
    //XXX: This will stay allocated in MESH till the end.
    return part_generate_partitioning_unweighted(mesh.ElementToElementTable(), nparts);
}

#endif // _FEM_HPP
