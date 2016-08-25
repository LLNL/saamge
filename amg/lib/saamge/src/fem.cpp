/*! \file

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
using std::ofstream;
using std::ifstream;
using std::stringstream;
using std::fill;

/* Options */

CONFIG_DEFINE_CLASS(FEM);

/* Functions */

void fem_refine_mesh(int max_num_elems, Mesh& mesh)
{
    SA_PRINTF_L(4, "%s", "Refining mesh...\n");
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
    SA_PRINTF_L(4, "%s", "Refining mesh...\n");
    for (int i=0; i < times; ++i)
    {
        SA_PRINTF_L(4, "Refining %d...\n", i);
        mesh.UniformRefinement();
    }
}

void fem_init_with_bdr_cond(ParGridFunction& x, ParFiniteElementSpace *fes,
                            Coefficient& bdr_coeff)
{
    SA_PRINTF_L(4, "%s", "Initializing vector with boundary conditions...\n");
    x.Update(fes);
    x.ProjectCoefficient(bdr_coeff);
}

agg_dof_status_t *fem_find_bdr_dofs(ParFiniteElementSpace& fes,
                                    Array<int> *ess_bdr)
{
    Array<int> be_dofs;
    const int ND = fes.GetNDofs();
    agg_dof_status_t *bdr_dofs = new agg_dof_status_t[ND];
    SA_ASSERT(fes.GetNV() <= ND);

    fill(bdr_dofs, bdr_dofs + ND, 0);

    Array<int> ess_dofs;
    fes.GetEssentialVDofs(*ess_bdr, ess_dofs);
    SA_ASSERT(ess_dofs.Size() == ND);

    Table &group_ldof = fes.GroupComm().GroupLDofTable();
    const int ngroups = group_ldof.Size();

    for (int i=0; i < ND; ++i)
    {
        if (ess_dofs[i])
            SA_SET_FLAGS(bdr_dofs[i], AGG_ON_ESS_DOMAIN_BORDER_FLAG);
        if (-1 != fes.GetLocalTDofNumber(i))
            SA_SET_FLAGS(bdr_dofs[i], AGG_OWNED_FLAG);
    }

    for (int gr=1; gr < ngroups; ++gr)
    {
        const int *ldofs = group_ldof.GetRow(gr);
        const int nldofs = group_ldof.RowSize(gr);
        for (int i=0; i < nldofs; ++i)
        {
            SA_ASSERT(0 <= ldofs[i] && ldofs[i] < ND);
            SA_SET_FLAGS(bdr_dofs[ldofs[i]], AGG_ON_PROC_IFACE_FLAG);
        }
    }

    return bdr_dofs;
}

ParLinearForm *fem_assemble_rhs(ParFiniteElementSpace *fespace,
                                Coefficient& rhs)
{
    SA_PRINTF_L(4, "%s", "Assembling global right-hand side...\n");
    ParLinearForm *b = new ParLinearForm(fespace);
    b->AddDomainIntegrator(new DomainLFIntegrator(rhs));
    b->Assemble();

    return b;
}

void fem_serial_visualize_gf(const Mesh& mesh, GridFunction& x,
                             const char *keys/*=""*/)
{
    SA_PRINTF_L(4, "%s", "Visualizing grid function in serial...\n");
    char vishost[] = "localhost";
    int  visport   = CONFIG_ACCESS_OPTION(FEM, glvis_port);
    osockstream sol_sock(visport, vishost);

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
    sol_sock.send();
    sol_sock.close();
}

void fem_parallel_visualize_gf(const ParMesh& mesh, ParGridFunction& x,
                               const char *keys/*=""*/)
{
    SA_PRINTF_L(4, "%s", "Visualizing grid function in parallel...\n");
    char vishost[] = "localhost";
    int  visport   = CONFIG_ACCESS_OPTION(FEM, glvis_port);
    osockstream sol_sock(visport, vishost);

    if (!sol_sock.is_open())
        return;

    sol_sock << "parallel " << PROC_NUM << " " << PROC_RANK << endl;

    if (2 == mesh.Dimension())
        sol_sock << "fem2d_gf_data_keys\n";
    else
        sol_sock << "fem3d_gf_data_keys\n";
    sol_sock.precision(CONFIG_ACCESS_OPTION(GLOBAL, prec));
    mesh.Print(sol_sock);
    x.Save(sol_sock);
    sol_sock << keys;
    sol_sock.send();
    sol_sock.close();
}

void fem_serial_visualize_pwc_coef(Mesh& mesh, Coefficient& coef,
                                   const char *keys/*=""*/)
{
    SA_PRINTF_L(4, "%s", "Preparing piece-wise constant visualization "
                         "in serial...\n");
    char vishost[] = "localhost";
    int  visport   = CONFIG_ACCESS_OPTION(FEM, glvis_port);
    osockstream sol_sock(visport, vishost);
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
    int  visport   = CONFIG_ACCESS_OPTION(FEM, glvis_port);
    osockstream sol_sock(visport, vishost);
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
    int  visport   = CONFIG_ACCESS_OPTION(FEM, glvis_port);
    osockstream sol_sock(visport, vishost);

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
    sol_sock.send();
    sol_sock.close();
    delete pfes;
    delete pfec;
}

void fem_parallel_visualize_partitioning(ParMesh& mesh, int *partitioning,
                                         int parts, const char *keys/*=""*/)
{
    SA_PRINTF_L(4, "%s", "Visualizing partition in parallel...\n");
    char vishost[] = "localhost";
    int  visport   = CONFIG_ACCESS_OPTION(FEM, glvis_port);
    osockstream sol_sock(visport, vishost);

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
    proc_allgather_offsets(parts, offsets);
    SA_ASSERT(offsets[PROC_RANK] >= 0);
    for (int i=0; i < NE; ++i)
    {
        lpartitioning[i] = partitioning[i] + offsets[PROC_RANK];
        p(i) = (double)lpartitioning[i];
    }

    sol_sock << "parallel " << PROC_NUM << " " << PROC_RANK << endl;

    if (2 == mesh.Dimension())
        sol_sock << "fem2d_gf_data_keys\n";
    else
        sol_sock << "fem3d_gf_data_keys\n";
    sol_sock.precision(CONFIG_ACCESS_OPTION(GLOBAL, prec));
    mesh.PrintWithPartitioning(lpartitioning, sol_sock);
    p.Save(sol_sock);
    sol_sock << "f" << (2 == mesh.Dimension()?"Rjl":"") << keys;
    sol_sock.send();
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
    int  visport   = CONFIG_ACCESS_OPTION(FEM, glvis_port);
    osockstream sol_sock(visport, vishost);

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
    sol_sock.send();
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
    int  visport   = CONFIG_ACCESS_OPTION(FEM, glvis_port);
    osockstream sol_sock(visport, vishost);

    if (!sol_sock.is_open())
        return;

    Array<int> offsets;
    proc_allgather_offsets(parts, offsets);
    SA_ASSERT(offsets[PROC_RANK] >= 0);
    for (int i=0; i < Ndofs; ++i)
    {
        x(i) = (double)(aggregates[i] +
                        (aggregates[i] < 0 ? 0 : offsets[PROC_RANK]));
    }

    sol_sock << "parallel " << PROC_NUM << " " << PROC_RANK << endl;

    if (2 == fes->GetMesh()->Dimension())
        sol_sock << "fem2d_gf_data_keys\n";
    else
        sol_sock << "fem3d_gf_data_keys\n";
    sol_sock.precision(CONFIG_ACCESS_OPTION(GLOBAL, prec));
    fes->GetMesh()->Print(sol_sock);
    x.Save(sol_sock);
    sol_sock << "Rj" << keys;
    sol_sock.send();
    sol_sock.close();
}

Mesh *fem_read_mesh(const char *filename)
{
    SA_PRINTF_L(4, "%s", "Loading mesh...\n");
    ifstream imesh(filename);
    SA_ASSERT(imesh);
    Mesh *mesh = new Mesh(imesh, 1, 1);
    imesh.close();
    return mesh;
}

void fem_write_mesh(const char *filename, const Mesh& mesh)
{
    ofstream omesh(filename);
    SA_ASSERT(omesh);
    omesh.precision(CONFIG_ACCESS_OPTION(GLOBAL, prec));
    mesh.Print(omesh);
    omesh.close();
}

GridFunction *fem_read_gf(const char *filename, Mesh *mesh)
{
    ifstream igf(filename);
    SA_ASSERT(igf);
    GridFunction *gf = new GridFunction(mesh, igf);
    igf.close();
    return gf;
}

void fem_write_gf(const char *filename, GridFunction& gf)
{
    ofstream ogf(filename);
    SA_ASSERT(ogf);
    ogf.precision(CONFIG_ACCESS_OPTION(GLOBAL, prec));
    gf.Save(ogf);
    ogf.close();
}
