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
    an elliptic problem as an auxiliary space solver, using SAAMGe for the IP problem.

    This example starts with an H1 problem and produces an agglomeration.
    Based on the agglomerates it builds IP spaces (on the agglomerate
    elements and agglomerate faces) and formulation together with transition operators (using averaging)
    between H1 and the constructed IP spaces. The IP spaces can be fine-scale or coarse
    (constructed via eigenvalue problems for the H1 agglomerate matrices).
    If they are coarse the transition operators are straight between fine H1 and coarse IP spaces.
    The IP problem can be the entire formulation or condensed to the agglomerate faces
    (utilizing a Schur complement) for both fine- and coarse-scale IP formulations.

    The matrix of the IP system is obtained via assembly. The constructed IP problem is preconditioned
    by SAAMGe (two- or multi-level). This configuration can be used as an auxiliary solver for the
    H1 problem or SAAMGe can be tested for solving the IP problem alone. Whenever the IP problem is
    solved alone and in accordance with the need, the right-hand side of the IP system can be assembled
    (in a fine-scale IP formulation) or obtained via the transition operators (in a coarse-scale IP formulation).
    This does NOT result in equal right-hand sides but seems to provide similar results.

    It is intended for solver settings, which means that we consider essential BCs (boundary conditions) that
    can only be zero in the IP formulation. Essential BCs are strongly enforced in the IP formulation by considering
    basis (both the ones associated with the agglomerates and the agglomerate faces) functions that are
    NOT supported on the respective portion of the boundary.

    When a condensed (Schur complement) version is utilized, the IP system is reduced only on the agglomerate faces.
    However, the averaging transition operator (between H1 and IP) is for the whole IP space. Therefore, some additional
    transitions are utilized before and after invoking the condensed IP solver/preconditioner.

    While stationary iteration is supported here, for the auxiliary-space preconditioner, it is mostly intended to be
    used in a Krylov solver (e.g., CG).
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
    ml_data_t *ml_data_saamge=NULL;

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
    double theta_saamge = 0.003;
    args.AddOption(&theta_saamge, "-ts", "--theta-saamge",
                   "Tolerance for eigenvalue problems for the SAAMGe preconditioner of the IP matrix for the first coarsening.");
    double theta_saamge_c = theta_saamge;
    args.AddOption(&theta_saamge_c, "-tsc", "--theta-saamge-c",
                   "Tolerance for eigenvalue problems for the SAAMGe preconditioner of the IP matrix for the rest of the coarsenings.");
    int nu_relax = 2;
    args.AddOption(&nu_relax, "-n", "--nu-relax",
                   "Degree for smoother in the relaxation when preconditioning the IP problem.");
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
    bool ml = false;
    args.AddOption(&ml, "-ml", "--multilevel",
                   "-nml", "--no-multilevel",
                   "Multilevel instead of two-level procedure.");
    int nl = 2;
    args.AddOption(&nl, "-nl", "--num-levels",
                   "Number of levels when multilevel is activated. "
                   "If 2, this is similar to NOT activating multilevel at all.");
    bool aux = true;
    args.AddOption(&aux, "-au", "--auxiliary",
                   "-nau", "--no-auxiliary",
                   "Auxiliary solver. If NOT set, then the resulting IP problem is solved, "
                   "whether condensed to agglomerate faces or NOT.");
    int nu_relax_aux = 2;
    args.AddOption(&nu_relax_aux, "-na", "--nu-relax-aux",
                   "Degree for smoother in the relaxation for the auxiliary-space solver. "
                   "That is for the smoother on the H1 form before and after invoking the auxiliary "
                   "IP correction.");
    bool compute_errors = false;
    args.AddOption(&compute_errors, "-ce", "--compute-errors",
                   "-nce", "--no-compute-errors",
                   "Whether to compute errors comparing solutions obtained here to the H1 solution.");
    double tol = 1e-12;
    args.AddOption(&tol, "-tol", "--tolerance",
                   "Relative tolerance for solver convergence.");

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
    x1.SetSpace(&fes);

    SparseMatrix& Al = a->SpMat();
    HypreParMatrix *Ag = a->ParallelAssemble();
    HypreParVector *bg = b->ParallelAssemble();
    HypreParVector *hxg = x.ParallelAverage();
    HypreParVector *pxg = x.ParallelAverage();
    HypreParVector *xg = x.ParallelAverage();

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

    tg_data = tg_init_data(*Ag, *agg_part_rels, 0, nu_relax_aux, theta, false, 0.0, !direct_eigensolver);
    tg_data->polynomial_coarse_space = -1;

    Vector diag;
    if (global_diag)
        mbox_obtain_global_diagonal(*Ag, *(agg_part_rels->Dof_TrueDof), diag);

    if (full_space)
        nonconf_ip_discretization(*tg_data, *agg_part_rels, emp, delta, global_diag?&diag:NULL, schur);
    else
        nonconf_ip_coarsen_finest_h1(*tg_data, *agg_part_rels, emp, theta, delta, global_diag?&diag:NULL, schur, full_space);
    tg_print_data(*Ag, tg_data);

    Array<Matrix *> *elmats;
    ElementMatrixProvider *emp_ip;
    if (full_space)
    {
        agg_part_rels_saamge = nonconf_create_partitioning(*agg_part_rels, *tg_data->interp_data);
        emp_ip = new ElementIPMatrix(*agg_part_rels_saamge, *tg_data->interp_data);
    }
    else
    {
        Table cface_to_AE;
        Transpose(*agg_part_rels->AE_to_cface, cface_to_AE, agg_part_rels->num_cfaces);
        Table *elem_to_elem = Mult(*agg_part_rels->AE_to_cface, cface_to_AE);

        Table *elem_to_dof = nonconf_create_AE_to_dof(*agg_part_rels, *tg_data->interp_data);

        agg_dof_status_t *bdr_dofs = new agg_dof_status_t[tg_data->Ac->GetNumRows()]();
        const int lev_elems_per_agg = (2 == dim ? elems_per_agg * elems_per_agg * 2 :
                                                  elems_per_agg * elems_per_agg * elems_per_agg * 6);
        nparts = (int) round((double) nparts / (double) lev_elems_per_agg);
        agg_part_rels_saamge = agg_create_partitioning_fine(*tg_data->Ac, agg_part_rels->nparts, elem_to_dof, elem_to_elem, NULL, bdr_dofs, &nparts, agg_part_rels->cface_cDof_TruecDof, false);
        delete [] bdr_dofs;

        elmats = new Array<Matrix *>(agg_part_rels->nparts);
        for (int i=0; i < agg_part_rels->nparts; ++i)
        {
            (*elmats)[i] = nonconf_AE_matrix(*tg_data->interp_data, i);
            SA_ASSERT((*elmats)[i]);
        }
        emp_ip = new ElementMatrixArray(*agg_part_rels_saamge, *elmats);
    }
    if (ml)
    {
        int nparts_arr[nl-1];
        nparts_arr[0] = nparts;
        const int lev_elems_per_agg = (2 == dim ? elems_per_agg * elems_per_agg * 2 :
                                                  elems_per_agg * elems_per_agg * elems_per_agg * 6);
        for (int i=1; i < nl-1; ++i)
        {
            nparts_arr[i] = (int) round((double) nparts_arr[i-1] / (double) lev_elems_per_agg);
            if (nparts_arr[i] < 1) nparts_arr[i] = 1;
        }
        MultilevelParameters mlp(nl-1, nparts_arr, 0, 0, nu_relax, theta_saamge,
                                 theta_saamge_c, -1, false, !direct_eigensolver, false);
        if (coarse_direct)
            mlp.set_coarse_direct(true);
        ml_data_saamge = ml_produce_data(*tg_data->Ac, agg_part_rels_saamge, emp_ip, mlp);
    } else
    {
        tg_data_saamge = tg_produce_data(*tg_data->Ac, *agg_part_rels_saamge, 0, nu_relax, emp_ip, theta_saamge, false, -1,
                                         !direct_eigensolver, false);
        tg_fillin_coarse_operator(*tg_data->Ac, tg_data_saamge, false);

        Solver *solver;
        if (coarse_direct)
            solver = new HypreDirect(*tg_data_saamge->Ac);
        else
            solver = new AMGSolver(*tg_data_saamge->Ac, false, 1e-16, 1000);
        tg_data_saamge->coarse_solver = solver;
        tg_print_data(*tg_data->Ac, tg_data_saamge);
    }
    if (!full_space)
    {
        for (int i=0; i < agg_part_rels->nparts; ++i)
            delete (*elmats)[i];
        delete elmats;
    }

    if (compute_errors && !zero_rhs)
    {
        // Obtain the H1 solution.
        Solver *fsolver;
        if (coarse_direct)
            fsolver = new HypreDirect(*Ag);
        else
            fsolver = new AMGSolver(*Ag, false);
        fsolver->Mult(*bg, *xg);
        delete fsolver;
        x = *xg;
//        FunctionCoefficient uex(ex_func);
//        x.ProjectCoefficient(uex);
        if (visualize)
            fem_parallel_visualize_gf(pmesh, x);
    }

    // Obtain IP solution.
    HypreParVector cx(*tg_data->interp);
    HypreParVector *cbg = NULL;

//    HypreParVector *hx1g = x1.ParallelAverage();
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
//    delete hx1g;

    Solver *solver=NULL;
    if (aux)
    {
        solver = new VCycleSolver((ml ? levels_list_get_level(ml_data_saamge->levels_list, 0)->tg_data :
                                        tg_data_saamge), false);
        solver->SetOperator(*tg_data->Ac);
//        solver = new HypreDirect(*tg_data->Ac);
        if (schur)
            tg_data->coarse_solver = new SchurSolver(*tg_data->interp_data, *agg_part_rels, *agg_part_rels->cface_cDof_TruecDof,
                                                     *agg_part_rels->cface_TruecDof_cDof, *solver);
        else
        {
            tg_data->coarse_solver = solver;
            solver = NULL;
        }
    } else
    {
        if (zero_rhs)
            cbg = new HypreParVector(*tg_data->interp);
        else
        {
            cx = 0.0;
            if (full_space)
            {
                ElementDomainLFVectorStandardGeometric rhsp(*agg_part_rels, new DomainLFIntegrator(rhs), &fes);
                cbg = nonconf_ip_discretization_rhs(*tg_data, *agg_part_rels, &rhsp);
            } else
            {
                cbg = new HypreParVector(*tg_data->interp);
                tg_data->restr->Mult(*bg, *cbg);
            }
        }
    }

    if (stat_it)
    {
        if (zero_rhs)
        {
            SA_RPRINTF(0, "%s", "\n");
            SA_RPRINTF(0, "%s", "\t\t\tRUNNING STATIONARY ITERATION WITH RANDOM INITIAL GUESS AND ZERO R.H.S:\n");
            SA_RPRINTF(0, "%s", "\n");
            x = 0.0;
            if (!aux)
                *cbg = 0.0;
        }
        else
        {
            SA_RPRINTF(0, "%s", "\n");
            SA_RPRINTF(0, "%s", "\t\t\tSOLVING THE PROBLEM USING STATIONARY ITERATION:\n");
            SA_RPRINTF(0, "%s", "\n");
        }

        if (aux)
            tg_run(*Ag, agg_part_rels, *hxg, *bg, 1000, tol, 1e-24, 1.0, tg_data, zero_rhs, true);
        else
        {
            if (schur)
            {
                HypreParVector *schur_rhs = nonconf_assemble_schur_rhs(*tg_data->interp_data, *agg_part_rels,
                                                                       *agg_part_rels->cface_TruecDof_cDof, *cbg);
                HypreParVector eb(*schur_rhs);
                eb = 0.0;
                tg_run(*tg_data->Ac, agg_part_rels_saamge, eb, *schur_rhs, 1000, tol, 1e-24, 1.0,
                       (ml ? levels_list_get_level(ml_data_saamge->levels_list, 0)->tg_data :
                        tg_data_saamge), zero_rhs, true);
                delete schur_rhs;
                nonconf_schur_update_interior(*tg_data->interp_data, *agg_part_rels, *agg_part_rels->cface_cDof_TruecDof, *cbg, eb, cx);
            }
            else
            {
                tg_run(*tg_data->Ac, agg_part_rels_saamge, cx, *cbg, 1000, tol, 1e-24, 1.0,
                       (ml ? levels_list_get_level(ml_data_saamge->levels_list, 0)->tg_data :
                        tg_data_saamge), zero_rhs, true);
            }
            tg_data->interp->Mult(cx, *hxg);
        }
        x1 = *hxg;
        if (visualize)
            fem_parallel_visualize_gf(pmesh, x1);

        // Error
        if (compute_errors)
        {
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
            delete gx;
            SA_RPRINTF(0, "ERROR: L2 = %g; Linf = %g; ENERGY = %g\n", l2err, maxerr, energyerr);
            x = *xg;
        }
        cx = 0.0;
    }
    delete hxg;

    if (zero_rhs)
    {
        SA_RPRINTF(0, "%s", "\n");
        SA_RPRINTF(0, "%s", "\t\t\tRUNNING PCG WITH RANDOM INITIAL GUESS AND ZERO R.H.S:\n");
        SA_RPRINTF(0, "%s", "\n");
        x = 0.0;
        if (aux)
        {
            pxg->Randomize(0);
            *bg = 0.0;
        }
        else
        {
            cx.Randomize(0);
            *cbg = 0.0;
        }
    }
    else
    {
        SA_RPRINTF(0, "%s", "\n");
        SA_RPRINTF(0, "%s", "\t\t\tSOLVING THE PROBLEM USING PCG:\n");
        SA_RPRINTF(0, "%s", "\n");
    }

    int iterations = -1;
    int converged = -1;

    if (aux)
    {
        VCycleSolver Bprec(tg_data, false);
        Bprec.SetOperator(*Ag);

        CGSolver hpcg(MPI_COMM_WORLD);
        hpcg.SetOperator(*Ag);
        hpcg.SetRelTol(sqrt(tol)); // MFEM squares this.
        hpcg.SetMaxIter(1000);
        hpcg.SetPrintLevel(1);
        hpcg.SetPreconditioner(Bprec);
        hpcg.Mult(*bg, *pxg);
        iterations = hpcg.GetNumIterations();
        converged = hpcg.GetConverged();
    }
    else
    {
        VCycleSolver Bprec((ml ? levels_list_get_level(ml_data_saamge->levels_list, 0)->tg_data :
                            tg_data_saamge), false);
        Bprec.SetOperator(*tg_data->Ac);

        CGSolver hpcg(MPI_COMM_WORLD);
        hpcg.SetOperator(*tg_data->Ac);
        hpcg.SetRelTol(sqrt(tol)); // MFEM squares this.
        hpcg.SetMaxIter(1000);
        hpcg.SetPrintLevel(1);
        hpcg.SetPreconditioner(Bprec);
        if (schur)
        {
            SchurSolver SS(*tg_data->interp_data, *agg_part_rels, *agg_part_rels->cface_cDof_TruecDof,
                           *agg_part_rels->cface_TruecDof_cDof, hpcg, zero_rhs);
            SS.Mult(*cbg, cx);
        }
        else
            hpcg.Mult(*cbg, cx);
        iterations = hpcg.GetNumIterations();
        converged = hpcg.GetConverged();

        tg_data->interp->Mult(cx, *pxg);
    }
    x1 = *pxg;
    if (visualize)
        fem_parallel_visualize_gf(pmesh, x1);

    if (converged)
        SA_RPRINTF(0, "Outer PCG converged in %d iterations.\n", iterations);
    else
        SA_RPRINTF(0, "Outer PCG failed to converge after %d iterations!\n", iterations);

    // Error
    if (compute_errors)
    {
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
        delete gx;
        SA_RPRINTF(0, "ERROR: L2 = %g; Linf = %g; ENERGY = %g\n", l2err, maxerr, energyerr);
    }

    delete xg;
    delete cbg;
    delete pxg;
    delete bg;
    delete Ag;
    delete a;
    delete b;
    tg_free_data(tg_data);
    ml_free_data(ml_data_saamge);
    tg_free_data(tg_data_saamge);
    delete solver;
    agg_free_partitioning(agg_part_rels);
    agg_free_partitioning(agg_part_rels_saamge);
    delete mesh;
    MPI_Finalize();
    return 0;
}
