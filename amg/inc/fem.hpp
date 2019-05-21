/*! \file
    \brief Finite elements (FE) related functionality and a bit more.

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
#ifndef _FEM_HPP
#define _FEM_HPP

#include "common.hpp"
#include <mfem.hpp>
#include <cmath>
#include "mfem_addons.hpp"
#include "aggregates.hpp"
#include "elmat.hpp"

namespace saamge
{

/* Options */

const int GLVIS_PORT = 19916;

/* Classes */

/*! \brief A convenience for computing a basic monomial

    Taken from ParElag.
*/
class MonomialCoefficient : public mfem::Coefficient
{
public:
    MonomialCoefficient(int orderx, int ordery, int orderz = 0):
        order_x(orderx),
        order_y(ordery),
        order_z(orderz)
    {}

    virtual double Eval(mfem::ElementTransformation &T,
                        const mfem::IntegrationPoint &ip)
    {
        using std::pow;
        double x[3];
        mfem::Vector transip(x, 3);

        T.Transform(ip, transip);

        return pow(x[0],order_x)*pow(x[1],order_y)*pow(x[2],order_z);
    }
    virtual void Read(std::istream &in){ in >> order_x >> order_y >> order_z;}

private:
    int order_x;
    int order_y;
    int order_z;
};

/* Functions */
/*! \brief Refines mesh to a certain upper bound for the number of elements.

    \param max_num_elems (IN) An upper bound for the number of elements.
    \param mesh (IN/OUT) The mesh to be refined.

    \warning \a mesh should be created prior to calling this function.
*/
void fem_refine_mesh(int max_num_elems, mfem::Mesh& mesh);

/*! \brief Refines mesh a certain number of times.

    \param times (IN) Number of times to refine the mesh.
    \param mesh (IN/OUT) The mesh to be refined.

    \warning \a mesh should be created prior to calling this function.
*/
void fem_refine_mesh_times(int times, mfem::Mesh& mesh);

/*! \brief Initializes \a x on space \a fespace with \a bdr_coeff.

    \param x (OUT) The grid function to be initialized.
    \param fespace (IN) The FE space to be used.
    \param bdr_coeff (IN) The coefficient to be projected on the grid function
                          \a x. It defines a border condition.
*/
void fem_init_with_bdr_cond(mfem::ParGridFunction& x, mfem::ParFiniteElementSpace *fespace,
                            mfem::Coefficient& bdr_coeff);

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
agg_dof_status_t *fem_find_bdr_dofs(mfem::ParFiniteElementSpace& fes,
                                    mfem::Array<int> *ess_bdr);

/*! \brief Generates a right-hand side linear form.

    \param fespace (IN) The finite element space.
    \param rhs (IN) The coefficient defining the right-hand side.

    \returns The linear form for the right-hand side.

    \warning The returned linear form must be freed by the caller.
*/
mfem::ParLinearForm *fem_assemble_rhs(mfem::ParFiniteElementSpace *fespace,
                                mfem::Coefficient& rhs);

/*! \brief Visualizes a grid function using GLVIS in serial.

    \param mesh (IN) The mesh on which the grid function is defined.
    \param x (IN) The grid function to be visualized.
    \param keys (IN) Instructions for GLVIS (as keys to "press").

    \warning GLVIS daemon must be running on localhost. The port it is bound to
             is given by the \b glvis_port option.
    \warning The precision of the output is determined by the \b prec option.
*/
void fem_serial_visualize_gf(const mfem::Mesh& mesh, mfem::GridFunction& x,
                             const char *keys="");

/*! \brief Visualizes a grid function using GLVIS in parallel.

    \param mesh (IN) The mesh on which the grid function is defined.
    \param x (IN) The grid function to be visualized.
    \param keys (IN) Instructions for GLVIS (as keys to "press").

    \warning GLVIS daemon must be running on localhost. The port it is bound to
             is given by the \b glvis_port option.
    \warning The precision of the output is determined by the \b prec option.
*/
void fem_parallel_visualize_gf(const mfem::ParMesh& mesh, mfem::ParGridFunction& x,
                               const char *keys="");

/*! \brief Visualizes piece-wise constant coefficient using GLVIS in serial.

    \param mesh (IN) The mesh on which the grid function is defined.
    \param coef (IN) The coefficient.
    \param keys (IN) Instructions for GLVIS (as keys to "press").

    \warning GLVIS daemon must be running on localhost. The port it is bound to
             is given by the \b glvis_port option.
    \warning The precision of the output is determined by the \b prec option.
*/
void fem_serial_visualize_pwc_coef(mfem::Mesh& mesh, mfem::Coefficient& coef,
                                   const char *keys="");

/*! \brief Visualizes piece-wise constant coefficient using GLVIS in parallel.

    \param mesh (IN) The mesh on which the grid function is defined.
    \param coef (IN) The coefficient.
    \param keys (IN) Instructions for GLVIS (as keys to "press").

    \warning GLVIS daemon must be running on localhost. The port it is bound to
             is given by the \b glvis_port option.
    \warning The precision of the output is determined by the \b prec option.
*/
void fem_parallel_visualize_pwc_coef(mfem::ParMesh& mesh, mfem::Coefficient& coef,
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
void fem_serial_visualize_partitioning(mfem::Mesh& mesh, int *partitioning,
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
void fem_parallel_visualize_partitioning(mfem::ParMesh& mesh, int *partitioning,
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
void fem_serial_visualize_aggregates(
    mfem::FiniteElementSpace *fes, int *aggregates, const char *keys="");

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
void fem_parallel_visualize_aggregates(mfem::ParFiniteElementSpace *fes,
                                       int *aggregates, int parts,
                                       const char *keys="");

/*! \brief Loads a mesh from a file. The format is determined by MFEM.

    \param filename (IN) The name of the file with the mesh.

    \returns The loaded mesh.

    \warning The returned mesh must be freed by the caller.
*/
mfem::Mesh *fem_read_mesh(const char *filename);

/*! \brief Writes a mesh to a file. The format is determined by MFEM.

    \param filename (IN) The name of the file to write. If it exists, it will
                         be erased prior to writing.
    \param mesh (IN) The mesh to be written.

    \warning The precision of the output is determined by the \b prec option.
*/
void fem_write_mesh(const char *filename, const mfem::Mesh& mesh);

/*! \brief Loads a grid function from a file. The format is determined by MFEM.

    \param filename (IN) The name of the file with the grid function.
    \param mesh (IN) The mesh on which the grid function will be defined.

    \returns The loaded grid function.

    \warning The returned grid function must be freed by the caller.
*/
mfem::GridFunction *fem_read_gf(const char *filename, mfem::Mesh *mesh);

/*! \brief Writes a grid function to a file. The format is determined by MFEM.

    \param filename (IN) The name of the file to write. If it exists, it will
                         be erased prior to writing.
    \param gf (IN) The grid function to be written.

    \warning The precision of the output is determined by the \b prec option.
*/
void fem_write_gf(const char *filename, mfem::GridFunction& gf);

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
mfem::ParBilinearForm *fem_assemble_stiffness(
    mfem::ParFiniteElementSpace *fespace, T& coeff, mfem::Vector& x,
    mfem::Vector& b, bool bdr_cond_impose, mfem::Array<int> *ess_bdr);

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
void fem_build_discrete_problem(mfem::ParFiniteElementSpace *fespace,
                                mfem::Coefficient& rhs, mfem::Coefficient& bdr_coeff,
                                T& coeff, bool bdr_cond_impose,
                                mfem::ParGridFunction& x,
                                mfem::ParLinearForm*& b, mfem::ParBilinearForm*& a,
                                mfem::Array<int> *ess_bdr);

/*! \brief Constructs only the face relations.

  This function counts on already built relation tables (like AEs) and only
  fills in the face and coarse face relations by constructing the coarse faces.
  It essentially obtains from MFEM the arguments on the finest level for agg_build_face_relations.
*/
void fem_build_face_relations(agg_partitioning_relations_t *agg_part_rels, mfem::ParFiniteElementSpace& fes);


/*! \brief Obtains all targets up to certain order.
    Borrows heavily from ParElag.
*/
void fem_polynomial_targets(mfem::FiniteElementSpace *fespace, mfem::Array<mfem::Vector *>& targets, int order);

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
int *fem_partition_mesh(mfem::Mesh& mesh, int *nparts);

/*! \brief Utility routine, to deal with vector-valued problems.

    Converts an mfem elem_to_dof table to one that SAAMGe can use
    algebraically without knowing about the vector structure.
*/
mfem::Table* vector_valued_elem_to_dof(const mfem::Table& mfem_elem_to_dof,
                                 const int vdim, const int ordering);

/*! \brief Partitions a Cartesian 2D mesh into rectangles.
*/
int *fem_partition_dual_simple_2D(mfem::Mesh& mesh, int *nparts, int *nparts_x,
                                  int *nparts_y);

/*! \brief Partitions a Cartesian 3D mesh into rectangles.
*/
int *fem_partition_dual_simple_3D(mfem::Mesh& mesh, int *nparts, int *nparts_x,
                                  int *nparts_y, int *nparts_z);

/*! \brief One fine element to one coarse element - possibly useful at high
           order.
*/
agg_partitioning_relations_t *
fem_create_partitioning_identity(
    mfem::HypreParMatrix& A, mfem::ParFiniteElementSpace& fes,
    const agg_dof_status_t *bdr_dofs, int *nparts, bool do_mises=true);

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
fem_create_partitioning(mfem::HypreParMatrix& A, mfem::ParFiniteElementSpace& fes,
                        const agg_dof_status_t *bdr_dofs, int *nparts,
                        bool do_aggregates, bool do_mises=true);

agg_partitioning_relations_t *
fem_create_partitioning_from_matrix(const mfem::SparseMatrix& A,
                                    int *nparts,
                                    mfem::HypreParMatrix *dof_truedof,
                                    mfem::Array<int>& isolated_cells);

/*! \brief Computes the center of an element

    \param mesh (IN) A mesh.
    \param elno (IN) The number of the element.
    \param center (OUT) The center of the element.
*/
void fem_get_element_center(const mfem::Mesh& mesh, int elno, mfem::Vector& center);

/*!
    Obtains the interpolant of a monomial.
*/
static inline
void fem_monomial_target(mfem::FiniteElementSpace *fespace, mfem::Vector& target,
                         int orderx, int ordery, int orderz=0);

/*!
    Frees targets.
*/
static inline
void fem_free_targets(mfem::Array<mfem::Vector *>& targets);

/* Function Templates Definitions */
template <class T>
mfem::ParBilinearForm *fem_assemble_stiffness(
    mfem::ParFiniteElementSpace *fespace,
    T& coeff, mfem::Vector& x, mfem::Vector& b,
    bool bdr_cond_impose, mfem::Array<int> *ess_bdr)
{
    SA_RPRINTF_L(0, 4, "%s", "Assembling global stiffness matrix...\n");
    mfem::ParBilinearForm *a = new mfem::ParBilinearForm(fespace);
    a->AddDomainIntegrator(new mfem::DiffusionIntegrator(coeff));
    a->Assemble(/*0*/);
    if (bdr_cond_impose)
    {
        SA_RPRINTF_L(0, 4, "%s", "Imposing boundary conditions...\n");
        /* Imposing Dirichlet boundary conditions. */
        SA_ASSERT(ess_bdr->Size() == fespace->GetMesh()->bdr_attributes.Max());
        mfem::Array<int> ess_dofs;
        fespace->GetEssentialVDofs(*ess_bdr, ess_dofs);
        const bool keep_diag = true;
        a->EliminateEssentialBCFromDofs(ess_dofs, x, b, keep_diag);
    }
    SA_RPRINTF_L(0, 4, "%s", "Finalizing global stiffness matrix...\n");
    a->Finalize(0);

    return a;
}

template <class T>
void fem_build_discrete_problem(
    mfem::ParFiniteElementSpace *fespace,
    mfem::Coefficient& rhs, mfem::Coefficient& bdr_coeff,
    T& coeff, bool bdr_cond_impose, mfem::ParGridFunction& x,
    mfem::ParLinearForm*& b, mfem::ParBilinearForm*& a,
    mfem::Array<int> *ess_bdr)
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
        x.SetSpace(fespace);
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
int *fem_partition_mesh(mfem::Mesh& mesh, int *nparts)
{
    //XXX: This will stay allocated in MESH till the end.
    return part_generate_partitioning_unweighted(mesh.ElementToElementTable(), nparts);
}

static inline
void fem_monomial_target(mfem::FiniteElementSpace *fespace, mfem::Vector& target,
                         int orderx, int ordery, int orderz)
{
    SA_ASSERT(orderx >= 0 && ordery >= 0 && orderz >= 0);
    MonomialCoefficient mc(orderx, ordery, orderz);
    target.SetSize(fespace->GetVSize());
    mfem::GridFunction gf;
    gf.MakeRef(fespace, target, 0);
    gf.ProjectCoefficient(mc);
}

static inline
void fem_free_targets(mfem::Array<mfem::Vector *>& targets)
{
    for (int i=0; i < targets.Size(); ++i)
        delete targets[i];
    targets.DeleteAll();
}

} // namespace saamge

#endif // _FEM_HPP
