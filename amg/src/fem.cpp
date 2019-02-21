/*! \file

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

#include "common.hpp"
#include "fem.hpp"
#define _USE_MATH_DEFINES
#include <cmath>
#include <fstream>
#include <sstream>
#include <cfloat>
#include <algorithm>
#include <mfem.hpp>
#include "part.hpp"
#include "helpers.hpp"
using std::floor;
#if (__cplusplus > 199711L)
using std::round;
#endif
using std::sqrt;
using std::stringstream;

namespace saamge
{
using namespace mfem;

/* Functions */

void fem_refine_mesh(int max_num_elems, Mesh& mesh)
{
    SA_RPRINTF_L(0, 4, "%s", "Refining mesh...\n");
    const int ref_levels =
        (int)floor(log((double)max_num_elems / mesh.GetNE()) /
                   (M_LN2 * mesh.Dimension()));
    for (int i=0; i < ref_levels; ++i)
    {
        SA_PRINTF_L(4, "Refining %d...\n", i);
        mesh.UniformRefinement();
    }
}

void fem_refine_mesh_times(int times, Mesh& mesh)
{
    SA_RPRINTF_L(0, 4, "%s", "Refining mesh...\n");
    for (int i=0; i < times; ++i)
    {
        SA_RPRINTF_L(0, 4, "Refining %d...\n", i);
        mesh.UniformRefinement();
    }
}

void fem_init_with_bdr_cond(ParGridFunction& x, ParFiniteElementSpace *fes,
                            Coefficient& bdr_coeff)
{
    SA_RPRINTF_L(0, 4, "%s", "Initializing vector with boundary conditions...\n");
    x.SetSpace(fes);
    x.ProjectCoefficient(bdr_coeff);
}

agg_dof_status_t *fem_find_bdr_dofs(ParFiniteElementSpace& fes,
                                    Array<int> *ess_bdr)
{
    // const int ND = fes.GetNDofs();
    const int ND = fes.GetVSize();
    agg_dof_status_t *bdr_dofs = new agg_dof_status_t[ND];
    SA_ASSERT(fes.GetNV() <= ND);

    std::fill(bdr_dofs, bdr_dofs + ND, 0);

    Array<int> ess_dofs;
    fes.GetEssentialVDofs(*ess_bdr, ess_dofs);
    SA_ASSERT(ess_dofs.Size() == ND);

    Table &group_ldof = fes.GroupComm().GroupLDofTable();
    const int ngroups = group_ldof.Size();
    // const int n_ldofs = group_ldof.Width(); // = 0 in serial

    // std::cout << "<<<< ND = fes.GetVSize() = " << ND << ", fes.GetNDofs() = " << fes.GetNDofs()
    //        << ", n_ldofs = " << n_ldofs << std::endl;

    // int count = 0;
    for (int i=0; i < ND; ++i)
    {
        if (ess_dofs[i])
        {
            SA_SET_FLAGS(bdr_dofs[i], AGG_ON_ESS_DOMAIN_BORDER_FLAG);
            // std::cout << "      <<<< set ess flag on dof " << i << std::endl;
            // count++;
        }
        if (-1 != fes.GetLocalTDofNumber(i))
            SA_SET_FLAGS(bdr_dofs[i], AGG_OWNED_FLAG);
    }
    // std::cout << "<<<< count " << count << " bdr dofs" << std::endl;

    int max_ldof = -1;
    // std::cout << "<<<< ngroups = " << ngroups << std::endl;
    for (int gr=1; gr < ngroups; ++gr)
    {
        const int *ldofs = group_ldof.GetRow(gr);
        const int nldofs = group_ldof.RowSize(gr);
        for (int i=0; i < nldofs; ++i)
        {
            // despite previous doubts, ldof does seem to correspond to ND
            if (ldofs[i] > max_ldof)
                max_ldof = ldofs[i];
            SA_ASSERT(0 <= ldofs[i] && ldofs[i] < ND);
            SA_SET_FLAGS(bdr_dofs[ldofs[i]], AGG_ON_PROC_IFACE_FLAG);
        }
    }
    // std::cout << "<<<< max_ldof = " << max_ldof << std::endl;

    return bdr_dofs;
}

ParLinearForm *fem_assemble_rhs(ParFiniteElementSpace *fespace,
                                Coefficient& rhs)
{
    SA_RPRINTF_L(0, 4, "%s", "Assembling global right-hand side...\n");
    ParLinearForm *b = new ParLinearForm(fespace);

    // elasticity?
    SA_ASSERT(fespace->GetVDim() == 1);
    b->AddDomainIntegrator(new DomainLFIntegrator(rhs));
    b->Assemble();

    return b;
}

void fem_serial_visualize_gf(const Mesh& mesh, GridFunction& x,
                             const char *keys/*=""*/)
{
    SA_PRINTF_L(4, "%s", "Visualizing grid function in serial...\n");
    char vishost[] = "localhost";
    int visport = GLVIS_PORT;
    // osockstream sol_sock(visport, vishost);
    socketstream sol_sock(vishost, visport);

    if (!sol_sock.is_open())
        return;

    if (2 == mesh.Dimension())
        sol_sock << "fem2d_gf_data_keys\n";
    else
        sol_sock << "fem3d_gf_data_keys\n";
    sol_sock.precision(CONFIG_ACCESS_OPTION(GLOBAL, prec));
    mesh.Print(sol_sock);
    x.Save(sol_sock);
    sol_sock << keys;
    // sol_sock.send();
    sol_sock.close();
}

void fem_parallel_visualize_gf(const ParMesh& mesh, ParGridFunction& x,
                               const char *keys/*=""*/)
{
    SA_PRINTF_L(4, "%s", "Visualizing grid function in parallel...\n");
    char vishost[] = "localhost";
    int visport = GLVIS_PORT;
    // osockstream sol_sock(visport, vishost);
    socketstream sol_sock(vishost, visport);

    if (!sol_sock.is_open())
        return;

    sol_sock << "parallel " << PROC_NUM << " " << PROC_RANK << std::endl;

    if (2 == mesh.Dimension())
        sol_sock << "fem2d_gf_data_keys\n";
    else
        sol_sock << "fem3d_gf_data_keys\n";
    sol_sock.precision(CONFIG_ACCESS_OPTION(GLOBAL, prec));
    mesh.Print(sol_sock);
    x.Save(sol_sock);
    sol_sock << keys;
    // sol_sock.send();
    sol_sock.close();
}

void fem_serial_visualize_pwc_coef(Mesh& mesh, Coefficient& coef,
                                   const char *keys/*=""*/)
{
    SA_PRINTF_L(4, "%s", "Preparing piece-wise constant visualization "
                         "in serial...\n");
    char vishost[] = "localhost";
    int visport = GLVIS_PORT;
    // osockstream sol_sock(visport, vishost);
    socketstream sol_sock(vishost, visport);
    if (!sol_sock.is_open())
        return;
    sol_sock.close();

    FiniteElementCollection *pfec;
    if (mesh.Dimension() == 2)
        pfec = new Const2DFECollection;
    else
        pfec = new Const3DFECollection;
    FiniteElementSpace *pfes = new FiniteElementSpace(&mesh, pfec);
    GridFunction p(pfes);
    p.ProjectCoefficient(coef);
    stringstream strs;
    strs << "f" << (mesh.Dimension() == 2?"Rj":"") << keys;
    fem_serial_visualize_gf(mesh, p, strs.str().c_str());
    delete pfes;
    delete pfec;
}

void fem_parallel_visualize_pwc_coef(ParMesh& mesh, Coefficient& coef,
                                     const char *keys/*=""*/)
{
    SA_PRINTF_L(4, "%s", "Preparing piece-wise constant visualization "
                         "in parallel...\n");
    char vishost[] = "localhost";
    int visport = GLVIS_PORT;
    // osockstream sol_sock(visport, vishost);
    socketstream sol_sock(vishost, visport);
    if (!sol_sock.is_open())
        return;
    sol_sock.close();

    FiniteElementCollection *pfec;
    if (mesh.Dimension() == 2)
        pfec = new Const2DFECollection;
    else
        pfec = new Const3DFECollection;
    ParFiniteElementSpace *pfes =
        new ParFiniteElementSpace(&mesh, pfec, mesh.Dimension(), 0);
    ParGridFunction p(pfes);
    p.ProjectCoefficient(coef);
    stringstream strs;
    strs << "f" << (mesh.Dimension() == 2?"Rj":"") << keys;
    fem_parallel_visualize_gf(mesh, p, strs.str().c_str());
    delete pfes;
    delete pfec;
}

void fem_serial_visualize_partitioning(Mesh& mesh, int *partitioning,
                                       const char *keys/*=""*/)
{
    SA_PRINTF_L(4, "%s", "Visualizing partition in serial...\n");
    char vishost[] = "localhost";
    int visport = GLVIS_PORT;
    // osockstream sol_sock(visport, vishost);
    socketstream sol_sock(vishost, visport);

    if (!sol_sock.is_open())
        return;

    SA_ASSERT(partitioning);
    FiniteElementCollection *pfec;
    if (mesh.Dimension() == 2)
        pfec = new Const2DFECollection;
    else
        pfec = new Const3DFECollection;
    FiniteElementSpace *pfes  = new FiniteElementSpace(&mesh, pfec);
    GridFunction p(pfes);
    const int NE = mesh.GetNE();
    for (int i=0; i < NE; ++i)
    {
        p(i) = (double)partitioning[i];
    }
    if (2 == mesh.Dimension())
        sol_sock << "fem2d_gf_data_keys\n";
    else
        sol_sock << "fem3d_gf_data_keys\n";
    sol_sock.precision(CONFIG_ACCESS_OPTION(GLOBAL, prec));
    mesh.PrintWithPartitioning(partitioning, sol_sock);
    p.Save(sol_sock);
    sol_sock << "f" << (2 == mesh.Dimension()?"Rjl":"") << keys;
    // sol_sock.send();
    sol_sock.close();
    delete pfes;
    delete pfec;
}

void fem_parallel_visualize_partitioning(ParMesh& mesh, int *partitioning,
                                         int parts, const char *keys/*=""*/)
{
    // SA_PRINTF_L(4, "%s", "Visualizing partition in parallel...\n");
    char vishost[] = "localhost";
    int visport = GLVIS_PORT;
    // osockstream sol_sock(visport, vishost);
    socketstream sol_sock(vishost, visport);

    if (!sol_sock.is_open())
        return;

    SA_ASSERT(partitioning);
    FiniteElementCollection *pfec;
    if (mesh.Dimension() == 2)
        pfec = new Const2DFECollection;
    else
        pfec = new Const3DFECollection;
    ParFiniteElementSpace *pfes =
        new ParFiniteElementSpace(&mesh, pfec, mesh.Dimension(), 0);
    ParGridFunction p(pfes);
    const int NE = mesh.GetNE();
    int *lpartitioning = new int[NE];
    Array<int> offsets;
    // proc_allgather_offsets(parts, offsets);
    // SA_ASSERT(offsets[PROC_RANK] >= 0);
    int total;
    proc_determine_offsets(parts, offsets, total);
    SA_ASSERT(offsets[0] >= 0);
    for (int i=0; i < NE; ++i)
    {
        lpartitioning[i] = partitioning[i] + offsets[0];
        p(i) = (double)lpartitioning[i];
        // SA_PRINTF("element %d, partitioning = %d, offset = %d, lpartitioning = %d, p = %f\n",
        //        i, partitioning[i], offsets[0], lpartitioning[i], p(i));
    }

    if (PROC_NUM > 1)
        sol_sock << "parallel " << PROC_NUM << " " << PROC_RANK << std::endl;

    if (2 == mesh.Dimension())
        sol_sock << "fem2d_gf_data_keys\n";
    else
        sol_sock << "fem3d_gf_data_keys\n";
    sol_sock.precision(CONFIG_ACCESS_OPTION(GLOBAL, prec));
    mesh.PrintWithPartitioning(lpartitioning, sol_sock);
    p.Save(sol_sock);
    sol_sock << "f" << (2 == mesh.Dimension()?"Rjl":"") << keys;
    // sol_sock.send();
    sol_sock.close();
    delete [] lpartitioning;
    delete pfes;
    delete pfec;
}

void fem_serial_visualize_aggregates(FiniteElementSpace *fes, int *aggregates,
                                     const char *keys/*=""*/)
{
    SA_PRINTF_L(4, "%s", "Visualizing aggregates in serial...\n");
    SA_ASSERT(fes);
    SA_ASSERT(aggregates);
    GridFunction x(fes);
    const int Ndofs = fes->GetNDofs();

    char vishost[] = "localhost";
    int visport = GLVIS_PORT;
    socketstream sol_sock(vishost, visport);

    if (!sol_sock.is_open())
        return;

    for (int i=0; i < Ndofs; ++i)
    {
        x(i) = aggregates[i];
    }
    if (2 == fes->GetMesh()->Dimension())
        sol_sock << "fem2d_gf_data_keys\n";
    else
        sol_sock << "fem3d_gf_data_keys\n";
    sol_sock.precision(CONFIG_ACCESS_OPTION(GLOBAL, prec));
    fes->GetMesh()->Print(sol_sock);
    x.Save(sol_sock);
    sol_sock << "Rj" << keys;
    // sol_sock.send();
    sol_sock.close();
}

void fem_parallel_visualize_aggregates(ParFiniteElementSpace *fes,
                                       int *aggregates, int parts,
                                       const char *keys/*=""*/)
{
    SA_PRINTF_L(4, "%s", "Visualizing aggregates in parallel...\n");
    SA_ASSERT(fes);
    SA_ASSERT(aggregates);
    ParGridFunction x(fes);
    const int Ndofs = fes->GetNDofs();

    char vishost[] = "localhost";
    int visport = GLVIS_PORT;
    // osockstream sol_sock(visport, vishost);
    socketstream sol_sock(vishost, visport);

    if (!sol_sock.is_open())
        return;

    Array<int> offsets;
    // proc_allgather_offsets(parts, offsets);
    // SA_ASSERT(offsets[PROC_RANK] >= 0);
    int total;
    proc_determine_offsets(parts, offsets, total);
    SA_ASSERT(offsets[0] >= 0);
    for (int i=0; i < Ndofs; ++i)
    {
        x(i) = (double)(aggregates[i] +
                        (aggregates[i] < 0 ? 0 : offsets[0]));
    }

    sol_sock << "parallel " << PROC_NUM << " " << PROC_RANK << std::endl;

    if (2 == fes->GetMesh()->Dimension())
        sol_sock << "fem2d_gf_data_keys\n";
    else
        sol_sock << "fem3d_gf_data_keys\n";
    sol_sock.precision(CONFIG_ACCESS_OPTION(GLOBAL, prec));
    fes->GetMesh()->Print(sol_sock);
    x.Save(sol_sock);
    sol_sock << "Rj" << keys;
    // sol_sock.send();
    sol_sock.close();
}

Mesh *fem_read_mesh(const char *filename)
{
    SA_RPRINTF_L(0, 4, "%s", "Loading mesh...\n");
    std::ifstream imesh(filename);
    SA_ASSERT(imesh);
    Mesh *mesh = new Mesh(imesh, 1, 1);
    imesh.close();
    return mesh;
}

void fem_write_mesh(const char *filename, const Mesh& mesh)
{
    std::ofstream omesh(filename);
    SA_ASSERT(omesh);
    omesh.precision(CONFIG_ACCESS_OPTION(GLOBAL, prec));
    mesh.Print(omesh);
    omesh.close();
}

GridFunction *fem_read_gf(const char *filename, Mesh *mesh)
{
    std::ifstream igf(filename);
    SA_ASSERT(igf);
    GridFunction *gf = new GridFunction(mesh, igf);
    igf.close();
    return gf;
}

void fem_write_gf(const char *filename, GridFunction& gf)
{
    std::ofstream ogf(filename);
    SA_ASSERT(ogf);
    ogf.precision(CONFIG_ACCESS_OPTION(GLOBAL, prec));
    gf.Save(ogf);
    ogf.close();
}

/** 
    MFEM defines a "Dof" as a nodal point for the scalar problem,
    if we are doing a vector problem (i.e. elasticity), our SAAMGe
    definition of a dof is different. We define a "Dof" here as a
    scalar number, so for us in SAAMGe there are multiple Dofs per
    nodal point, while in MFEM there is only one.

    As a result, we need to modify MFEM's elem_to_dof table.
*/
Table* vector_valued_elem_to_dof(const Table& mfem_elem_to_dof,
                                 const int vdim, const int ordering)
{
    Table * out = new Table;
    Array<int> mfem_row;

    const int numrows = mfem_elem_to_dof.Size();
    const int num_mfemdof = mfem_elem_to_dof.Width();

    out->MakeI(numrows);
    for (int i=0; i<numrows; ++i)
    {
        out->AddColumnsInRow(i, vdim * mfem_elem_to_dof.RowSize(i));
    }
    out->MakeJ();
    if (ordering == Ordering::byVDIM)
    {
        for (int i=0; i<numrows; ++i)
        {
            mfem_elem_to_dof.GetRow(i,mfem_row);
            Array<int> newrow(mfem_row.Size() * vdim);
            int k = 0;
            for (int mdof=0; mdof<mfem_row.Size(); ++mdof)
                for (int dim=0; dim<vdim; ++dim)
                    newrow[k++] = mfem_row[mdof]*vdim + dim;
            out->AddConnections(i, newrow.GetData(), newrow.Size());
        }
    }
    else if (ordering == Ordering::byNODES)
    {
        for (int i=0; i<numrows; ++i)
        {
            mfem_elem_to_dof.GetRow(i,mfem_row);
            Array<int> newrow(mfem_row.Size() * vdim);
            int k = 0;
            for (int dim=0; dim<vdim; ++dim)
                for (int mdof=0; mdof<mfem_row.Size(); ++mdof)
                    newrow[k++] = (dim*num_mfemdof) + mfem_row[mdof];
            out->AddConnections(i, newrow.GetData(), newrow.Size());
        }
    }
    else
    {
        SA_ASSERT(false);
    }
    // the MFEM Table interface is the spawn of Satan, and I despise it with every fiber of my being
    out->ShiftUpI();
    out->Finalize();

    return out;
}

void fem_get_element_center(const Mesh& mesh, int elno, Vector& center)
{
    DenseMatrix Pts;
    const int Dim = mesh.Dimension();

    mesh.GetPointMatrix(elno, Pts);
    center.SetSize(Dim);
    center = 0.;
    for (int i = 0; i < Pts.Width(); ++i)
        for (int j = 0; j < Dim; ++j)
            center(j) += Pts(j, i);
    center /= (double)Pts.Width();
}

void fem_get_element_max_vertex(const Mesh& mesh, int elno, Vector& maxv)
{
    DenseMatrix Pts;

    mesh.GetPointMatrix(elno, Pts);
    Pts.GetColumn(0, maxv);
    for (int i = 1; i < Pts.Width(); ++i)
    {
        for (int j = 0; j < mesh.Dimension(); ++j)
        {
            if (maxv(j) < Pts(j, i))
                maxv(j) = Pts(j, i);
        }
    }
}

void fem_get_element_min_vertex(const Mesh& mesh, int elno, Vector& minv)
{
    DenseMatrix Pts;

    mesh.GetPointMatrix(elno, Pts);
    Pts.GetColumn(0, minv);
    for (int i = 1; i < Pts.Width(); ++i)
    {
        for (int j = 0; j < mesh.Dimension(); ++j)
        {
            if (minv(j) > Pts(j, i))
                minv(j) = Pts(j, i);
        }
    }
}

int *fem_partition_dual_simple_2D(Mesh& mesh, int *nparts, int *nparts_x,
                                  int *nparts_y)
{
//    SA_ASSERT(PROC_NUM == 1);

    SA_ASSERT(nparts);
    SA_ASSERT(nparts_x);
    SA_ASSERT(nparts_y);
    const int NE = mesh.GetNE();
    int *partitioning = new int[NE];

    SA_ASSERT(2 == mesh.Dimension());
    SA_PRINTF_L(5, "Number of Partitions: desired: %d", *nparts);

    // Check to make sure square number of partitions.
    if (*nparts_x <= 0 && *nparts_y <= 0)
    {
        SA_ASSERT(*nparts > 0);
        *nparts_x = *nparts_y = (int)round(sqrt(*nparts));
    } 
    else if (*nparts_x <= 0)
    {
        SA_ASSERT(*nparts > 0);
        SA_ASSERT(*nparts_y > 0);
        *nparts_x = (int)round((double)*nparts / (double)*nparts_y);
    } 
    else if ((*nparts_y <= 0))
    {
        SA_ASSERT(*nparts > 0);
        SA_ASSERT(*nparts_x > 0);
        *nparts_y = (int)round((double)*nparts / (double)*nparts_x);
    }

    *nparts = *nparts_x * *nparts_y;
    SA_PRINTF_NOTS_L(5, ", will generate: %d, in x direction: %d, "
                        "in y direction: %d\n", *nparts, *nparts_x, *nparts_y);
    SA_ASSERT(*nparts_y > 0);
    SA_ASSERT(*nparts > 0);
    SA_ASSERT(*nparts_x > 0);

    double sx=0., sy=0.;
    Vector maximal_point; // top right vertex
    for (int i=0; i < NE; ++i)
    {
        fem_get_element_max_vertex(mesh, i, maximal_point);
        if (sx < maximal_point(0))
            sx = maximal_point(0);
        if (sy < maximal_point(1))
            sy = maximal_point(1);
    }
    SA_ASSERT(sx > 0.);
    SA_ASSERT(sy > 0.);

    double lx=sx, ly=sy;
    Vector minimal_point; // bottom left vertex
    for (int i=0; i < NE; ++i)
    {
        fem_get_element_min_vertex(mesh, i, minimal_point);
        if (lx > minimal_point(0))
            lx = minimal_point(0);
        if (ly > minimal_point(1))
            ly = minimal_point(1);
    }
    SA_ASSERT(sx > lx);
    SA_ASSERT(sy > ly);

    for (int i=0; i < NE; ++i)
    {
        int x, y;
        double xmax, ymax;
#if 0
        fem_get_element_max_vertex(mesh, i, maximal_point);
#else
        fem_get_element_center(mesh, i, maximal_point);
#endif

        xmax = maximal_point(0);
        ymax = maximal_point(1);

        y = (int)((ymax - ly) * (double)*nparts_y / (sy - ly));
        x = (int)((xmax - lx) * (double)*nparts_x / (sx - lx));
        if (x == *nparts_x) --x;
        if (y == *nparts_y) --y;
        SA_ASSERT(0 <= x && x < *nparts_x);
        SA_ASSERT(0 <= y && y < *nparts_y);

        partitioning[i] = y * *nparts_x + x;

        SA_ASSERT(0 <= partitioning[i] && partitioning[i] < *nparts);
    }

#if (SA_IS_DEBUG_LEVEL(4))
    SA_RPRINTF(0,"%s","AEs { ---------\n");
    part_check_partitioning(mesh.ElementToElementTable(), partitioning);
    SA_RPRINTF(0,"%s","} AEs ---------\n");
#endif

    return partitioning;
}

agg_partitioning_relations_t *
fem_create_partitioning_identity(HypreParMatrix& A, ParFiniteElementSpace& fes,
                                 const agg_dof_status_t *bdr_dofs, int *nparts, bool do_mises)
{
    Table *elem_to_dof, *elem_to_elem;
    Mesh *mesh = fes.GetMesh();

    *nparts = mesh->GetNE();

    //XXX: This will stay allocated in MESH till the end.
    elem_to_elem = mbox_copy_table(&(mesh->ElementToElementTable()));

    if (fes.GetVDim() == 1)
    {
        // scalar problem
        elem_to_dof = mbox_copy_table(&(fes.GetElementToDofTable()));
    }
    else
    {
        // elasticity
        elem_to_dof = vector_valued_elem_to_dof(
            fes.GetElementToDofTable(), fes.GetVDim(), fes.GetOrdering());
    }

    int * partitioning = new int[mesh->GetNE()]; // copied to agg_part_rels, deleted there
    for (int i=0; i<mesh->GetNE(); ++i)
        partitioning[i] = i;

    // in what follows, bdr_dofs is only used as info to copy onto coarser level, 
    // does not actually affect partitioning
    const bool do_aggregates = false;
    agg_partitioning_relations_t *agg_part_rels = agg_create_partitioning_fine(
        A, fes.GetNE(), elem_to_dof, elem_to_elem, partitioning, bdr_dofs, nparts,
        fes.Dof_TrueDof_Matrix(), do_aggregates, false, do_mises);

    SA_ASSERT(agg_part_rels);
    return agg_part_rels;
}

agg_partitioning_relations_t *
fem_create_partitioning(HypreParMatrix& A, ParFiniteElementSpace& fes,
                        const agg_dof_status_t *bdr_dofs, int *nparts,
                        bool do_aggregates, bool do_mises)
{
    Table *elem_to_dof, *elem_to_elem;
    Mesh *mesh = fes.GetMesh();

    //XXX: This will stay allocated in MESH till the end.
    elem_to_elem = mbox_copy_table(&(mesh->ElementToElementTable()));

    if (fes.GetVDim() == 1)
    {
        // scalar problem
        elem_to_dof = mbox_copy_table(&(fes.GetElementToDofTable()));
    }
    else
    {
        // elasticity
        elem_to_dof = vector_valued_elem_to_dof(
            fes.GetElementToDofTable(), fes.GetVDim(), fes.GetOrdering());
    }

    // in what follows, bdr_dofs is only used as info to copy onto coarser level, 
    // does not actually affect partitioning
    agg_partitioning_relations_t *agg_part_rels = agg_create_partitioning_fine(
        A, fes.GetNE(), elem_to_dof, elem_to_elem, NULL, bdr_dofs, nparts,
        fes.Dof_TrueDof_Matrix(), do_aggregates, false, do_mises);

    SA_ASSERT(agg_part_rels);
    return agg_part_rels;
}

agg_partitioning_relations_t *
fem_create_partitioning_from_matrix(const SparseMatrix& A, int *nparts,
                                    HypreParMatrix *dof_truedof,
                                    Array<int>& isolated_cells)
{
    const bool do_aggregates = true;
    int *partitioning = NULL;

    // elem_to_elem should be just the graph of A
    // elem_to_dof should be an identity matrix
    // (should rename to "cell" or "volume" for clarity)
    Table *elem_to_elem = TableFromSparseMatrix(A);
    Table *elem_to_dof = IdentityTable(A.Size());
    SA_ASSERT(elem_to_dof);
    SA_ASSERT(elem_to_elem);

    char * bdr_dofs = new char[A.Size()];
    memset(bdr_dofs, 0, sizeof(char) * A.Size());

    agg_partitioning_relations_t *agg_part_rels;

    HypreParMatrix * fakeAparallel = FakeParallelMatrix(&A);
    // fakeAparallel->Print("fakeAparallel.mat");

    if (isolated_cells.Size() == 0)
    {
        agg_part_rels = agg_create_partitioning_fine(
            *fakeAparallel, A.Size(), elem_to_dof, elem_to_elem,
            partitioning, bdr_dofs, nparts, dof_truedof, do_aggregates);
    }
    else
    {
        // do_aggregates is always true for this call
        agg_part_rels = agg_create_partitioning_fine_isolate(
            *fakeAparallel, A.Size(), elem_to_dof, elem_to_elem, partitioning,
            bdr_dofs, nparts, dof_truedof, isolated_cells);
    }
    delete[] bdr_dofs;

    delete fakeAparallel;
    
    SA_ASSERT(agg_part_rels);
    return agg_part_rels;
}

void fem_build_face_relations(agg_partitioning_relations_t *agg_part_rels, ParFiniteElementSpace& fes)
{
    Mesh *mesh = fes.GetMesh();
    SA_ASSERT(2 <= mesh->Dimension() && mesh->Dimension() <= 4);
    const int num_faces = (3 <= mesh->Dimension() ? mesh->GetNFaces() : mesh->GetNEdges());
    SA_RPRINTF(0, "pFaces: %d\n", num_faces);

    // Element to facet relation obtained from MFEM.
    Table *elem_to_face = mbox_copy_table(3 <= mesh->Dimension() ?
                                          &(mesh->ElementToFaceTable()) :
                                          &(mesh->ElementToEdgeTable()));
    SA_ASSERT(elem_to_face->Width() == num_faces);

    // Facet to DoF relation required a bit more attention. It is reasonable to assume that
    // all facets have the same number of DoFs.
    Array<int> dofs;
    fes.GetFaceDofs(0, dofs);
    const int num_face_dofs = dofs.Size();
    SA_ASSERT(num_face_dofs > 0);
    Table *face_to_dof = new Table(num_faces, num_face_dofs);
    for (int j=0; j < num_face_dofs; ++j)
        SA_VERIFY(face_to_dof->Push(0, dofs[j]) >= 0);
    for (int i=1; i < num_faces; ++i)
    {
        fes.GetFaceDofs(i, dofs);
        SA_ASSERT(dofs.Size() == num_face_dofs);
        for (int j=0; j < num_face_dofs; ++j)
            SA_VERIFY(face_to_dof->Push(i, dofs[j]) >= 0);
    }

    // The facet to true facet relation is obtained as in ParElag.
    // I'm not a big fan of this gymnastics with RT0 spaces, but it works.
    // It constructs a parallel RT0 finite element space on the parallel mesh
    // and the facet to true facet relation is DoF to true DoF in that space.
    // Signs in the relation that represent some orientation of facets are currently unimportant.
    ParMesh *pmesh = fes.GetParMesh();
    RT_FECollection RT0_fec(0, pmesh->Dimension());
    ParFiniteElementSpace RT0_fes(pmesh, &RT0_fec);
    // XXX: This is very weird! I'm not sure if memory is well managed here.
    signed char diagOwner = RT0_fes.Dof_TrueDof_Matrix()->OwnsDiag();
    signed char offdOwner = RT0_fes.Dof_TrueDof_Matrix()->OwnsOffd();
    signed char colMapOwner = RT0_fes.Dof_TrueDof_Matrix()->OwnsColMap();
    RT0_fes.Dof_TrueDof_Matrix()->SetOwnerFlags(-1,-1,-1);
    hypre_ParCSRMatrix *mat = RT0_fes.Dof_TrueDof_Matrix()->StealData();
    HypreParMatrix *face_to_trueface = new HypreParMatrix(mat);
    face_to_trueface->SetOwnerFlags(diagOwner, offdOwner, colMapOwner);
    mbox_make_owner_rowstarts_colstarts(*face_to_trueface);

    // Finally, call the aggregation routine.
    agg_build_face_relations(agg_part_rels, elem_to_face, face_to_dof, face_to_trueface);
}

void fem_polynomial_targets(FiniteElementSpace *fespace, Array<Vector *>& targets, int order)
{
    SA_ASSERT(order >= 0);
    const int dim = fespace->GetMesh()->Dimension();
    int size = 0;
    int ctr = 0;
    int order_x, order_y, order_z;

    switch(dim)
    {
    case 2:

        size = (order+1)*(order+2)/2;
        targets.SetSize(size);
        for(int order_max = 0; order_max <= order; ++order_max)
            for(order_x = 0; order_x <= order_max; ++order_x)
            {
                SA_ASSERT(ctr < size);
                targets[ctr] = new Vector;
                order_y = order_max - order_x;
                fem_monomial_target(fespace, *targets[ctr++], order_x, order_y);
            }
        SA_ASSERT(ctr == size);

    break;
    case 3:

        for(int order_max = 0; order_max <= order; ++order_max)
            for(order_x = 0; order_x <= order_max; ++order_x)
                for(order_y = 0; order_y <= order_max-order_x; ++order_y)
                    ++size;
        targets.SetSize(size);
        for(int order_max = 0; order_max <= order; ++order_max)
            for(order_x = 0; order_x <= order_max; ++order_x)
                for(order_y = 0; order_y <= order_max-order_x; ++order_y)
                {
                    SA_ASSERT(ctr < size);
                    targets[ctr] = new Vector;
                    order_z = order_max - order_x - order_y;
                    fem_monomial_target(fespace, *targets[ctr++], order_x, order_y, order_z);
                }
        SA_ASSERT(ctr == size);

    break;
    default:

        SA_ASSERT(false);
    }
}

} // namespace saamge
