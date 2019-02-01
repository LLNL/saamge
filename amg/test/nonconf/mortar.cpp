/*
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

/**
    Nonconforming SAAMGe as an statically-condensed mortar discretization appearing as a "coarse" space.
    It is intended for solver setting, which means that we consider essential BCs that
    can only be zero. By tradition, BCs are messy in this code.
*/

#include <mfem.hpp>
#include <mpi.h>
#include <saamge.hpp>

using namespace mfem;
using namespace saamge;

double checkboard_coef(Vector& x)
{
    SA_ASSERT(2 <= x.Size() && x.Size() <= 3);
    double d = (double)10;

    if ((x.Size() == 2 &&
         ((int)ceil(x(0)*d) & 1) == ((int)ceil(x(1)*d) & 1)) ||
        (x.Size() == 3 &&
         ((((int)ceil(x(2)*d) & 1) &&
           ((int)ceil(x(0)*d) & 1) == ((int)ceil(x(1)*d) & 1)) ||
          (!((int)ceil(x(2)*d) & 1) &&
           ((int)ceil(x(0)*d) & 1) != ((int)ceil(x(1)*d) & 1)))))
    {
        return 1e6;
    }
    else
    {
        return 1e0;
    }
}

double rhs_func(Vector& x)
{
    SA_ASSERT(2 <= x.Size() && x.Size() <= 3);
    return 1.0;
//    return 2.0 * (x[0]*(1.0-x[0]) + x[1]*(1.0-x[1]));
}

double ex_func(Vector& x)
{
    SA_ASSERT(2 <= x.Size() && x.Size() <= 2);
    return x[0]*(1.0-x[0])*x[1]*(1.0-x[1]);
}

double bdr_cond(Vector& x)
{
    SA_ASSERT(2 <= x.Size() && x.Size() <= 3);
    return 0.0;
}

int main(int argc, char *argv[])
{
    // Initialize process related stuff.
    MPI_Init(&argc, &argv);
    proc_init(MPI_COMM_WORLD);

    Mesh *mesh;
    ParGridFunction x, x1, err;
    ParLinearForm *b;
    ParBilinearForm *a;
    agg_partitioning_relations_t *agg_part_rels;
    tg_data_t *tg_data;

    OptionsParser args(argc, argv);

    const char *mesh_file = "";
    args.AddOption(&mesh_file, "-m", "--mesh",
                   "Mesh file to use.", true);
    bool visualize = true;
    args.AddOption(&visualize, "-vis", "--visualization", "-no-vis",
                   "--no-visualization",
                   "Enable or disable GLVis visualization.");
    int serial_times_refine = 0;
    args.AddOption(&serial_times_refine, "-sr", "--serial-refine",
                   "How many times to refine mesh before parallel partition.");
    int times_refine = 0;
    args.AddOption(&times_refine, "-r", "--refine", 
                   "How many times to refine the mesh (in parallel).");
    int order = 1;
    args.AddOption(&order, "-o", "--order",
                   "Polynomial order of finite element space.");
    int elems_per_agg = 8;
    args.AddOption(&elems_per_agg, "-e", "--elems-per-agg",
                   "Number of elements per agglomerated element.");
    bool coarse_direct = false;
    args.AddOption(&coarse_direct, "--coarse-direct", "--coarse-direct",
                   "--coarse-amg", "--coarse-amg",
                   "Use a direct solver on coarsest level or AMG V-cycle.");
    bool indirect_rhs = false;
    args.AddOption(&indirect_rhs, "-ind", "--indirect-rhs",
                   "---no-ind", "--no-indirect-rhs",
                   "Obtain RHS via the transfer operator.");

    args.Parse();
    if (!args.Good())
    {
        if (PROC_RANK == 0)
            args.PrintUsage(cout);
        MPI_Finalize();
        return 1;
    }
    if (PROC_RANK == 0)
        args.PrintOptions(cout);

    MPI_Barrier(PROC_COMM); // try to make MFEM's debug element orientation prints not mess up the parameters above

    // Read the mesh from the given mesh file.
//    mesh = fem_read_mesh(mesh_file);
    mesh = new Mesh(elems_per_agg >> 1, elems_per_agg >> 1, Element::TRIANGLE, 1);
    fem_refine_mesh_times(serial_times_refine, *mesh);
    // Serial mesh.
    SA_RPRINTF(0,"NV: %d, NE: %d\n", mesh->GetNV(), mesh->GetNE());

    // Parallel mesh and finite elements stuff.
    Array<int> ess_bdr(mesh->bdr_attributes.Max());
    ess_bdr = 1;
    ess_bdr[3] = 1;

    int nprocs = PROC_NUM;
    int *proc_partitioning;
    proc_partitioning = fem_partition_mesh(*mesh, &nprocs);
//    int nprocs_x = 2;
//    int nprocs_y = nprocs_x;
//    proc_partitioning = fem_partition_dual_simple_2D(*mesh, &nprocs, &nprocs_x, &nprocs_y);
    SA_ASSERT(PROC_NUM == nprocs);
    if (0 == PROC_RANK && visualize)
        fem_serial_visualize_partitioning(*mesh, proc_partitioning);
    ParMesh pmesh(MPI_COMM_WORLD, *mesh, proc_partitioning);
    delete [] proc_partitioning;
    fem_refine_mesh_times(times_refine, pmesh);

    H1_FECollection fec(order);
    ParFiniteElementSpace fes(&pmesh, &fec);
    const int pNV = pmesh.GetNV();
    const int pNE = pmesh.GetNE();
    const int pND = fes.GetNDofs();
    const int ND = fes.GlobalTrueVSize();
    SA_RPRINTF(0,"pNV: %d, pNE: %d, pND: %d, ND: %d\n", pNV, pNE, pND, ND);

    L2_FECollection cfec(0, pmesh.Dimension());
    ParFiniteElementSpace cfes(&pmesh, &cfec);

    FunctionCoefficient bdr_coeff(bdr_cond);
    FunctionCoefficient rhs(rhs_func);
    FunctionCoefficient conduct_func(checkboard_coef);
    ParGridFunction conductivity(&cfes);
    conductivity.ProjectCoefficient(conduct_func);
    GridFunctionCoefficient conduct_coeff(&conductivity);

    if (visualize)
        fem_parallel_visualize_gf(pmesh, conductivity, pmesh.Dimension() == 2?"jfR":"f");

    fem_build_discrete_problem(&fes, rhs, bdr_coeff, conduct_coeff, true, x, b, a, &ess_bdr);
    x1.SetSpace(&fes);
    err.SetSpace(&fes);

    SparseMatrix& Al = a->SpMat();
    HypreParMatrix *Ag = a->ParallelAssemble();
    HypreParVector *bg = b->ParallelAssemble();
    HypreParVector *hxg = x.ParallelAverage();

    // Actual AMGe stuff

    int nparts;
    agg_dof_status_t *bdr_dofs = fem_find_bdr_dofs(fes, &ess_bdr);
    nparts = pmesh.GetNE() / elems_per_agg;
    if (nparts == 0)
        nparts = 1;
//    agg_part_rels = fem_create_partitioning(*Ag, fes, bdr_dofs, &nparts, false, false);
//    agg_part_rels = fem_create_partitioning_identity(*Ag, fes, bdr_dofs, &nparts, false);

    int nparts_x = 1 << (serial_times_refine + times_refine + 1);
    int nparts_y = nparts_x;
    int *partitioning = fem_partition_dual_simple_2D(pmesh, &nparts, &nparts_x, &nparts_y);
    agg_part_rels = agg_create_partitioning_fine(
        *Ag, fes.GetNE(), mbox_copy_table(&(fes.GetElementToDofTable())), mbox_copy_table(&(pmesh.ElementToElementTable())), partitioning, bdr_dofs, &nparts,
        fes.Dof_TrueDof_Matrix(), false, false, false);

    delete [] bdr_dofs;
    fem_build_face_relations(agg_part_rels, fes);
    if (visualize)
        fem_parallel_visualize_partitioning(pmesh, agg_part_rels->partitioning, nparts);

    ElementMatrixStandardGeometric *emp = new ElementMatrixStandardGeometric(*agg_part_rels, Al, a);

    tg_data = tg_init_data(*Ag, *agg_part_rels, 0, 1, 1.0, false, 0.0, false);
    tg_data->polynomial_coarse_space = -1;

    mortar_discretization(*tg_data, *agg_part_rels, emp);
    tg_print_data(*Ag, tg_data);

    mfem::Solver *solver;
    mfem::Solver *fsolver;
    if (coarse_direct)
    {
        solver = new HypreDirect(*tg_data->Ac);
        fsolver = new HypreDirect(*Ag);
    } else
    {
        solver = new AMGSolver(*tg_data->Ac, false, 1e-16, 1000);
        fsolver = new AMGSolver(*Ag, false);
    }
    tg_data->coarse_solver = solver;

    // Obtain the usual solution.
    fsolver->Mult(*bg, *hxg);
    x = *hxg;
//    FunctionCoefficient uex(ex_func);
//    x.ProjectCoefficient(uex);
    if (visualize)
        fem_parallel_visualize_gf(pmesh, x);

    // Obtain mortar solution.
    HypreParVector *hx1g;
    HypreParVector *cbg;
    HypreParVector *precbg=NULL;
    HypreParVector cx(*tg_data->Ac);
    cx = 0.0;

    if (!indirect_rhs)
    {
        ElementDomainLFVectorStandardGeometric rhsp(*agg_part_rels, new DomainLFIntegrator(rhs), &fes);
        cbg = mortar_assemble_condensed_rhs(*tg_data->interp_data, *agg_part_rels, &rhsp);
    } else
    {
        precbg = new HypreParVector(*tg_data->interp);
        tg_data->restr->Mult(*bg, *precbg);
        cbg = mortar_assemble_schur_rhs(*tg_data->interp_data, *agg_part_rels, *precbg);
    }

    tg_data->coarse_solver->Mult(*cbg, cx);
    if (!indirect_rhs)
        hx1g = mortar_reverse_condensation(cx, *tg_data, *agg_part_rels);
    else
    {
        HypreParVector postcx(*tg_data->interp);
        mortar_schur_recover(*tg_data->interp_data, *agg_part_rels, *precbg, cx, postcx);
        hx1g = new HypreParVector(*tg_data->interp, 1);
        tg_data->interp->Mult(postcx, *hx1g);
    }
    x1 = *hx1g;
    if (visualize)
        fem_parallel_visualize_gf(pmesh, x1);

    // Error
    GridFunctionCoefficient xgf(&x);
    const double l2err = x1.ComputeL2Error(xgf);
    const double maxerr = x1.ComputeMaxError(xgf);
    err = x;
    err -= x1;
    if (visualize)
        fem_parallel_visualize_gf(pmesh, err);
    HypreParVector *gx = err.ParallelProject();
    const double energyerr = mbox_energy_norm_parallel(*Ag, *gx);
    SA_RPRINTF(0, "ERROR: L2 = %g; Linf = %g; ENERGY = %g\n", l2err, maxerr, energyerr);

    delete gx;
    delete cbg;
    delete precbg;
    tg_free_data(tg_data);
    delete fsolver;
    agg_free_partitioning(agg_part_rels);
    delete hx1g;
    delete hxg;
    delete bg;
    delete Ag;
    delete a;
    delete b;
    delete mesh;
    MPI_Finalize();
    return 0;
}
