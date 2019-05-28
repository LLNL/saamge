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
    Nonconforming interior penalty (IP) AMGe in two-level setting applied to
    an elliptic problem as a solver.

    This example starts with an H1 problem and produces an agglomeration.
    Based on the agglomerates it builds IP spaces (on the agglomerate
    elements and agglomerate faces) and formulation together with transition operators (using averaging)
    between H1 and the constructed IP spaces. The IP spaces can be fine-scale or coarse
    (constructed via eigenvalue problems for the H1 agglomerate matrices).
    If they are coarse the transition operators are straight between fine H1 and coarse IP spaces.
    The IP problem can be the entire formulation or condensed to the agglomerate faces
    (utilizing a Schur complement) for both fine- and coarse-scale IP formulations.

    In the end, it uses the transition operators and the IP formulation to obtain, by utilizing
    the standard two-level V-cycle, a multiplicative auxiliary space preconditioner
    for the H1 problem. The IP problem is inverted "exactly" in the preconditioner.
*/

#include <mfem.hpp>
#include <mpi.h>
#include <saamge.hpp>

using namespace mfem;
using namespace saamge;

double checkboard_coef(Vector& x)
{
    SA_ASSERT(2 <= x.Size() && x.Size() <= 3);
    double d = (double)8;

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
    ParGridFunction x;
    ParLinearForm *b;
    ParBilinearForm *a;
    agg_partitioning_relations_t *agg_part_rels;
    tg_data_t *tg_data;

    OptionsParser args(argc, argv);

    int dim = 2;
    args.AddOption(&dim, "-dim", "--dimension",
                   "The domain dimension.");
    bool visualize = false;
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
                   "Tolerance for eigenvalue problems for the IP spaces.");
    int nu_relax = 2;
    args.AddOption(&nu_relax, "-n", "--nu-relax",
                   "Degree for smoother in the relaxation for the auxiliary-space solver. "
                   "That is for the smoother on the H1 form before and after invoking the auxiliary "
                   "IP correction.");
    bool full_space = true;
    args.AddOption(&full_space, "-f", "--full-space",
                   "-nf", "--no-full-space",
                   "Build the full IP space instead of using eigensolvers.");
    bool schur = false;
    args.AddOption(&schur, "-s", "--schur",
                   "-ns", "--no-schur",
                   "Whether to use the Schur complement on the IP problem.");
    double delta = 1.0;
    args.AddOption(&delta, "-d", "--delta",
                   "The reciprocal of the interface term weight.");
    bool global_diag = true;
    args.AddOption(&global_diag, "-gd", "--global-diagonal",
                   "-ngd", "--no-global-diagonal",
                   "Use the global diagonal for face penalties in the IP formulation.");
    int elems_per_agg = 2;
    args.AddOption(&elems_per_agg, "-e", "--elems-per-agg",
                   "Number of rectangular partitions in one direction that constitute an agglomerated element.");
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
    bool stat_it = false;
    args.AddOption(&stat_it, "-si", "--stat-it",
                   "-nsi", "--no-stat-it",
                   "Whether to perform stationary iterations. "
                   "PCG is always invoked, independently of this parameter.");

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

    SA_ASSERT(2 <= dim && dim <= 3);

    // Build mesh.
    if (2 == dim)
        mesh = new Mesh(elems_per_agg, elems_per_agg, Element::TRIANGLE, 1);
    else
        mesh = new Mesh(elems_per_agg, elems_per_agg, elems_per_agg, Element::TETRAHEDRON, 1);
    fem_refine_mesh_times(serial_times_refine, *mesh);
    const int init_intervals_on_side = elems_per_agg << serial_times_refine;
    const int init_aggs_on_side = 1 << serial_times_refine;
    // Serial mesh.
    SA_RPRINTF(0,"NV: %d, NE: %d\n", mesh->GetNV(), mesh->GetNE());

    // Parallel mesh and finite elements stuff.
    Array<int> ess_bdr(mesh->bdr_attributes.Max());
    ess_bdr = 1;
    ess_bdr[3] = 1;

    int nprocs = PROC_NUM;
    int avg_procs_on_side = (int)((2 == dim ? sqrt((double)nprocs) : cbrt((double)nprocs)) + 1e-10);
    SA_ASSERT(avg_procs_on_side >= 1);
    int procs_on_side[dim];
    for (int i=0; i < dim; ++i)
        procs_on_side[i] = avg_procs_on_side;
    bool ret;
    ret = helpers_factorize(nprocs, dim, procs_on_side);
    if (!ret)
        SA_ALERT_RPRINTF(0, "Failed to factorize %d in %d factors!", nprocs, dim);
    int *proc_partitioning;
    if (2 == dim)
        proc_partitioning = fem_partition_dual_simple_2D(*mesh, &nprocs, &procs_on_side[0], &procs_on_side[1]);
    else
        proc_partitioning = fem_partition_dual_simple_3D(*mesh, &nprocs, &procs_on_side[0], &procs_on_side[1], &procs_on_side[2]);
    if (0 == PROC_RANK && visualize)
        fem_serial_visualize_partitioning(*mesh, proc_partitioning);
    for (int i=0; i < dim && 0 == PROC_RANK; ++i)
    {
        SA_ASSERT(init_intervals_on_side % procs_on_side[i] == 0);
        SA_ASSERT(init_aggs_on_side % procs_on_side[i] == 0);
    }
    MPI_Barrier(PROC_COMM);
    SA_ASSERT(PROC_NUM == nprocs);
    ParMesh pmesh(MPI_COMM_WORLD, *mesh, proc_partitioning);
    delete [] proc_partitioning;
    fem_refine_mesh_times(times_refine, pmesh);
    int intervals_on_side[dim];
    int aggs_on_side[dim];
    for (int i=0; i < dim; ++i)
    {
        intervals_on_side[i] = (init_intervals_on_side / procs_on_side[i]) << times_refine;
        aggs_on_side[i] = (init_aggs_on_side / procs_on_side[i]) << times_refine;
        SA_ASSERT(intervals_on_side[i] = aggs_on_side[i] * elems_per_agg);
    }

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

    SparseMatrix& Al = a->SpMat();
    HypreParMatrix *Ag = a->ParallelAssemble();
    HypreParVector *bg = b->ParallelAssemble();
    HypreParVector *hxg = x.ParallelAverage();
    HypreParVector *pxg = x.ParallelAverage();

    // Actual AMGe stuff

    int nparts;
    agg_dof_status_t *bdr_dofs = fem_find_bdr_dofs(fes, &ess_bdr);
    if (2 == dim)
        nparts = elems_per_agg * elems_per_agg * 2;
    else
        nparts = elems_per_agg * elems_per_agg * elems_per_agg * 6;
    nparts = pmesh.GetNE() / nparts;
    if (nparts == 0)
        nparts = 1;
    int *partitioning;
    if (2 == dim)
        partitioning = fem_partition_dual_simple_2D(pmesh, &nparts, &aggs_on_side[0], &aggs_on_side[1]);
    else
        partitioning = fem_partition_dual_simple_3D(pmesh, &nparts, &aggs_on_side[0], &aggs_on_side[1], &aggs_on_side[2]);
    agg_part_rels = agg_create_partitioning_fine(
        *Ag, fes.GetNE(), mbox_copy_table(&(fes.GetElementToDofTable())), mbox_copy_table(&(pmesh.ElementToElementTable())), partitioning, bdr_dofs, &nparts,
        fes.Dof_TrueDof_Matrix(), false, false, false);
    delete [] bdr_dofs;
    fem_build_face_relations(agg_part_rels, fes);
    if (visualize)
        fem_parallel_visualize_partitioning(pmesh, agg_part_rels->partitioning, nparts);

    ElementMatrixStandardGeometric *emp = new ElementMatrixStandardGeometric(*agg_part_rels, Al, a);
    //MultilevelParameters mlp(1, &nparts, 0, 0, nu_relax, theta, theta, -1, false, !direct_eigensolver, false);
    //mlp.set_coarse_direct(coarse_direct);

    //const bool avoid_ess_bdr_dofs = true;
    //tg_data = tg_produce_data(*Ag, *agg_part_rels, 3, nu_relax, emp, theta, false, -1,
    //                          !direct_eigensolver, avoid_ess_bdr_dofs);
    //tg_fillin_coarse_operator(*Ag, tg_data, false);

    tg_data = tg_init_data(*Ag, *agg_part_rels, 0, nu_relax, theta, false, 0.0, !direct_eigensolver);
    tg_data->polynomial_coarse_space = -1;

    Vector diag;
    if (global_diag)
        mbox_obtain_global_diagonal(*Ag, *(agg_part_rels->Dof_TrueDof), diag);

    if (full_space)
        nonconf_ip_discretization(*tg_data, *agg_part_rels, emp, delta, global_diag?&diag:NULL, schur);
    else
        nonconf_ip_coarsen_finest_h1(*tg_data, *agg_part_rels, emp, theta, delta, global_diag?&diag:NULL, schur, full_space);
    tg_print_data(*Ag, tg_data);

    Solver *solver;
    if (coarse_direct)
        solver = new HypreDirect(*tg_data->Ac);
    else
        solver = new AMGSolver(*tg_data->Ac, false, 1e-16, 1000);
    if (schur)
        tg_data->coarse_solver = new SchurSolver(*tg_data->interp_data, *agg_part_rels, *agg_part_rels->cface_cDof_TruecDof,
                                                 *agg_part_rels->cface_TruecDof_cDof, *solver);
    else
    {
        tg_data->coarse_solver = solver;
        solver = NULL;
    }

    if (stat_it)
    {
        if (zero_rhs)
        {
            SA_RPRINTF(0, "%s", "\n");
            SA_RPRINTF(0, "%s", "\t\t\tRUNNING STATIONARY ITERATION WITH RANDOM INITIAL GUESS AND ZERO R.H.S:\n");
            SA_RPRINTF(0, "%s", "\n");
        }
        else
        {
            SA_RPRINTF(0, "%s", "\n");
            SA_RPRINTF(0, "%s", "\t\t\tSOLVING THE PROBLEM USING STATIONARY ITERATION:\n");
            SA_RPRINTF(0, "%s", "\n");
        }

        tg_run(*Ag, agg_part_rels, *hxg, *bg, 1000, 1e-12, 1e-24, 1.0, tg_data, zero_rhs, true);

        x = *hxg;
        if (visualize)
            fem_parallel_visualize_gf(pmesh, x);
    }
    delete hxg;

    if (zero_rhs)
    {
        SA_RPRINTF(0, "%s", "\n");
        SA_RPRINTF(0, "%s", "\t\t\tRUNNING PCG WITH RANDOM INITIAL GUESS AND ZERO R.H.S:\n");
        SA_RPRINTF(0, "%s", "\n");
        pxg->Randomize(0);
        *bg = 0.0;
    }
    else
    {
        SA_RPRINTF(0, "%s", "\n");
        SA_RPRINTF(0, "%s", "\t\t\tSOLVING THE PROBLEM USING PCG:\n");
        SA_RPRINTF(0, "%s", "\n");
    }

    int iterations = -1;
    int converged = -1;

    VCycleSolver Bprec(tg_data, false);
    Bprec.SetOperator(*Ag);

    CGSolver hpcg(MPI_COMM_WORLD);
    hpcg.SetOperator(*Ag);
    hpcg.SetRelTol(1e-6); // MFEM squares this.
    hpcg.SetMaxIter(1000);
    hpcg.SetPrintLevel(1);
    hpcg.SetPreconditioner(Bprec);
    hpcg.Mult(*bg, *pxg);
    iterations = hpcg.GetNumIterations();
    converged = hpcg.GetConverged();
    x = *pxg;
    delete pxg;
    if (visualize)
        fem_parallel_visualize_gf(pmesh, x);
    delete bg;
    delete Ag;
    delete a;
    delete b;
    if (converged)
        SA_RPRINTF(0, "Outer PCG converged in %d iterations.\n", iterations);
    else
        SA_RPRINTF(0, "Outer PCG failed to converge after %d iterations!\n", iterations);

    tg_free_data(tg_data);
    delete solver;
    agg_free_partitioning(agg_part_rels);
    delete mesh;
    MPI_Finalize();
    return 0;
}
