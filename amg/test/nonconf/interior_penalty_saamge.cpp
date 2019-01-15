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
    Nonconforming SAAMGe as an interior penalty discretization appearing as a "coarse" space.
    It is intended for solver setting, which means that we consider essential BCs that
    can only be zero. By tradition, BCs are messy in this code.

    The idea is to use the usual SAAMGe for the interior penalty problem.
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
    ParGridFunction x, x1;
    ParLinearForm *b;
    ParBilinearForm *a;
    agg_partitioning_relations_t *agg_part_rels, *agg_part_rels_saamge;
    tg_data_t *tg_data, *tg_data_saamge=NULL;
    ml_data_t *ml_data_saamge;
    int *nparts_arr;

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
    double theta = 0.003;
    args.AddOption(&theta, "-t", "--theta",
                   "Tolerance for eigenvalue problems.");
    int nu_relax = 4;
    args.AddOption(&nu_relax, "-n", "--nu-relax",
                   "Degree for smoother in the relaxation.");
    double delta = 1.0;
    args.AddOption(&delta, "-d", "--delta",
                   "The reciprocal of the interface term weight.");
    int elems_per_agg = 8;
    args.AddOption(&elems_per_agg, "-e", "--elems-per-agg",
                   "Number of elements per agglomerated element.");
    bool coarse_direct = false;
    args.AddOption(&coarse_direct, "--coarse-direct", "--coarse-direct",
                   "--coarse-amg", "--coarse-amg",
                   "Use a direct solver on coarsest level or AMG V-cycle.");
    bool direct_eigensolver = true;
    args.AddOption(&direct_eigensolver, "-q", "--direct-eigensolver",
                   "-nq", "--no-direct-eigensolver",
                   "Use direct eigensolver from LAPACK or ARPACK.");
    bool zero_rhs = false;
    args.AddOption(&zero_rhs, "-z", "--zero-rhs",
                   "-nz", "--no-zero-rhs",
                   "Solve with zero RHS and random initial guess.");
    bool ml = false;
    args.AddOption(&ml, "-ml", "--multilevel",
                   "-nml", "--no-multilevel",
                   "Multilevel instead of two-level.");
    int nl = 2;
    args.AddOption(&nl, "-nl", "--num-levels",
                   "Number of levels.");
    bool aux = false;
    args.AddOption(&aux, "-au", "--auxiliary",
                   "-nau", "--no-auxiliary",
                   "Auxiliary solver.");
    int nu_relax_aux = 2;
    args.AddOption(&nu_relax_aux, "-na", "--nu-relax-aux",
                   "Degree for smoother in the relaxation for the auxiliary solver.");

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

    tg_data = tg_init_data(*Ag, *agg_part_rels, 0, nu_relax_aux, 1.0, false, 0.0, !direct_eigensolver);
    tg_data->polynomial_coarse_space = -1;

    nonconf_ip_discretization(*tg_data, *agg_part_rels, emp, delta);
    tg_print_data(*Ag, tg_data);

    agg_part_rels_saamge = nonconf_create_partitioning(*agg_part_rels, *tg_data->interp_data);
    ElementIPMatrix *emp_ip = new ElementIPMatrix(*agg_part_rels_saamge, *tg_data->interp_data);
    if (ml)
    {
        nparts_arr = new int[nl-1];
        nparts_arr[0] = nparts;
        for (int i=1; i < nl-1; ++i)
        {
            nparts_arr[i] = (int) round((double) nparts_arr[i-1] / (double) (elems_per_agg >> 1));
            if (nparts_arr[i] < 1) nparts_arr[i] = 1;
        }
        MultilevelParameters mlp(nl-1, nparts_arr, 0, 0, nu_relax, theta,
                                 theta, -1, false, !direct_eigensolver, false);
        if (coarse_direct)
            mlp.set_coarse_direct(true);
        ml_data_saamge = ml_produce_data(*tg_data->Ac, agg_part_rels_saamge, emp_ip, mlp);
    } else
    {
        tg_data_saamge = tg_produce_data(*tg_data->Ac, *agg_part_rels_saamge, 0, nu_relax, emp_ip, theta, false, -1,
                                         !direct_eigensolver, false);
        tg_fillin_coarse_operator(*tg_data->Ac, tg_data_saamge, false);

        mfem::Solver *solver;
        if (coarse_direct)
            solver = new HypreDirect(*tg_data_saamge->Ac);
        else
            solver = new AMGSolver(*tg_data_saamge->Ac, false, 1e-16, 1000);
        tg_data_saamge->coarse_solver = solver;
        tg_print_data(*tg_data->Ac, tg_data_saamge);
    }
    mfem::Solver *fsolver;
    if (coarse_direct)
        fsolver = new HypreDirect(*Ag);
    else
        fsolver = new AMGSolver(*Ag, false);

    if (zero_rhs)
    {
        x = 0.0;
        SA_RPRINTF(0, "%s", "\n");
        SA_RPRINTF(0, "%s", "\t\t\tRUNNING STATIONARY ITERATION WITH RANDOM INITIAL GUESS AND ZERO R.H.S:\n");
        SA_RPRINTF(0, "%s", "\n");
    }
    else
    {
        // Obtain the usual solution.
        fsolver->Mult(*bg, *hxg);
        x = *hxg;
//        FunctionCoefficient uex(ex_func);
//        x.ProjectCoefficient(uex);
        if (visualize)
            fem_parallel_visualize_gf(pmesh, x);

        SA_RPRINTF(0, "%s", "\n");
        SA_RPRINTF(0, "%s", "\t\t\tSOLVING THE PROBLEM USING STATIONARY ITERATION:\n");
        SA_RPRINTF(0, "%s", "\n");
    }

    // Obtain IP solution.
    HypreParVector *hx1g = x1.ParallelAverage();
    HypreParVector *cbg;
    HypreParVector cx(*tg_data->interp);

//    *(tg_data->interp) = 1.0;
//    SparseMatrix *restr = Transpose(*(tg_data_saamge->ltent_interp));
//    for (int i=0; i < restr->Height(); ++i)
//    {
//        cx = 0.0;
//        for (int j=0; j < restr->RowSize(i); ++j)
//            cx(restr->GetRowColumns(i)[j]) = 1.0;
//        tg_data->interp->Mult(cx, *hx1g);
//        x = *hx1g;
//        fem_parallel_visualize_gf(pmesh, x);
//        int foo;
//        std::cin >> foo;
//    }
//    delete restr;

//    *(tg_data->interp) = 1.0;
//    for (int i=0; i < agg_part_rels_saamge->num_mises; ++i)
//    {
//        cx = 0.0;
//        for (int j=0; j < agg_part_rels_saamge->mis_to_dof->RowSize(i); ++j)
//            cx(agg_part_rels_saamge->mis_to_dof->GetRow(i)[j]) = 1.0;
//        tg_data->interp->Mult(cx, *hx1g);
//        x = *hx1g;
//        fem_parallel_visualize_gf(pmesh, x);
//        int foo;
//        std::cin >> foo;
//    }

    cx = 0.0;
//        cbg = new HypreParVector(*tg_data->interp);
//        tg_data->restr->Mult(*bg, *cbg);
    ElementDomainLFVectorStandardGeometric rhsp(*agg_part_rels, new DomainLFIntegrator(rhs), &fes);
    cbg = nonconf_ip_discretization_rhs(*tg_data->interp_data, *agg_part_rels, &rhsp);

    if (aux)
    {
//        tg_data->coarse_solver = new HypreDirect(*tg_data->Ac);
        tg_data->coarse_solver = new VCycleSolver((ml ? levels_list_get_level(ml_data_saamge->levels_list, 0)->tg_data :
                                                        tg_data_saamge), false);
        tg_data->coarse_solver->SetOperator(*tg_data->Ac);
        tg_run(*Ag, agg_part_rels, *hxg, *bg, 1000, 1e-12, 1e-24, 1.0, tg_data, zero_rhs, true);
        x1 = *hxg;
    } else
    {
        tg_run(*tg_data->Ac, agg_part_rels_saamge, cx, *cbg, 1000, 1e-12, 1e-24, 1.0,
               (ml ? levels_list_get_level(ml_data_saamge->levels_list, 0)->tg_data :
                     tg_data_saamge), zero_rhs, true);
        tg_data->interp->Mult(cx, *hx1g);
        x1 = *hx1g;
    }
    if (visualize)
        fem_parallel_visualize_gf(pmesh, x1);

    // Error
    GridFunctionCoefficient xgf(&x);
    const double l2err = x1.ComputeL2Error(xgf);
    const double maxerr = x1.ComputeMaxError(xgf);
    if (zero_rhs)
        x = x1;
    else
    {
        x -= x1;
        if (visualize)
            fem_parallel_visualize_gf(pmesh, x);
    }
    HypreParVector *gx = x.ParallelProject();
    const double energyerr = mbox_energy_norm_parallel(*Ag, *gx);
    SA_RPRINTF(0, "ERROR: L2 = %g; Linf = %g; ENERGY = %g\n", l2err, maxerr, energyerr);

    delete gx;
    delete cbg;
    tg_free_data(tg_data);
    if (ml)
    {
        ml_free_data(ml_data_saamge);
        delete [] nparts_arr;
    } else
        tg_free_data(tg_data_saamge);
    delete fsolver;
    agg_free_partitioning(agg_part_rels);
    agg_free_partitioning(agg_part_rels_saamge);
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
