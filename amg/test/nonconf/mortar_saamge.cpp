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
    Nonconforming mortar AMGe in two-level setting applied to
    an elliptic problem as an auxiliary space solver, using SAAMGe for the mortar problem.

    This example starts with an H1 problem and produces an agglomeration.
    Based on the agglomerates it builds non-conforming spaces (on the agglomerate
    elements and agglomerate faces) and a mortar formulation together with transition operators (using averaging only on "element" DoFs)
    between H1 and the constructed non-conforming spaces. The non-conforming spaces are fine-scale on the interior and coarse
    (using polynomials) on the agglomerate faces. The Lagrangian multipliers are not explicitly appearing in the vectors
    and right-hand sides, but the space for them is cloned from the interface non-conforming space.
    The mortar problem is condensed to the agglomerate faces (utilizing a Schur complement)
    via the elimination of the "elements" and Lagrangian multipliers.

    The matrix of the (condensed) mortar system is obtained via assembly. The constructed mortar problem is preconditioned
    by SAAMGe (two- or multi-level). This configuration can be used as an auxiliary solver for the
    H1 problem or SAAMGe can be tested for solving the mortar problem alone. Whenever the mortar problem is
    solved alone and in accordance with the need, the right-hand side of the mortar system
    (excluding Lagrangian multipliers) can be assembled or obtained via the transition
    operators. This does NOT result in equal right-hand sides but seems to provide similar results.

    It is intended for solver settings, which means that we consider essential BCs (boundary conditions) that
    can only be zero in the mortar formulation. Essential BCs are strongly enforced in the mortar formulation by considering
    non-conforming basis (both the ones associated with the agglomerates and the agglomerate faces) functions that are
    NOT supported on the respective portion of the boundary.

    The mortar system is reduced (via Schur complement) only on the agglomerate faces. However, the averaging transition
    operator (between H1 and non-conforming) is for the whole non-conforming space (Lagrangian multipliers are NOT included). Therefore,
    some additional transitions are utilized before and after invoking the condensed morter solver/preconditioner.

    While stationary iteration is supported here, for the auxiliary-space preconditioner, it is mostly intended to be
    used in a Krylov solver (e.g., CG).

    There is also an option to invoke an additive version of the auxiliary space preconditioner.
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
        return 1e4;
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
    agg_partitioning_relations_t *agg_part_rels, *agg_part_rels_saamge=NULL;
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
    int order = 2;
    args.AddOption(&order, "-o", "--order",
                   "Polynomial order of finite element space.");
    int faceorder = 1;
    args.AddOption(&faceorder, "-fo", "--face-order",
                   "Polynomial order of face space.");
    double theta_saamge = 0.003;
    args.AddOption(&theta_saamge, "-ts", "--theta-saamge",
                   "Tolerance for eigenvalue problems for the SAAMGe preconditioner of the mortar matrix for the first coarsening.");
    double theta_saamge_c = theta_saamge;
    args.AddOption(&theta_saamge_c, "-tsc", "--theta-saamge-c",
                   "Tolerance for eigenvalue problems for the SAAMGe preconditioner of the mortar matrix for the rest of the coarsenings.");
    int fixed_num_evecs_saamge = 0;
    args.AddOption(&fixed_num_evecs_saamge, "-fvs", "--fixed-num-evecs-saamge",
                   "Takes a predetermined number of eigenvectors from the eigenvalue problems for the SAAMGe preconditioner "
                   "of the mortar matrix for the first coarsening. A value of 0 disables it and uses theta tolerances.");
    int fixed_num_evecs_saamge_c = fixed_num_evecs_saamge;
    args.AddOption(&fixed_num_evecs_saamge_c, "-fvsc", "--fixed-num-evecs-saamge-c",
                   "Takes a predetermined number of eigenvectors from the eigenvalue problems for the SAAMGe preconditioner "
                   "of the mortar matrix for the rest of the coarsenings. A value of 0 disables it and uses theta tolerances.");
    int nu_relax = 2;
    args.AddOption(&nu_relax, "-n", "--nu-relax",
                   "Degree for smoother in the relaxation when preconditioning the mortar problem.");
    bool global_diag = true;
    args.AddOption(&global_diag, "-gd", "--global-diagonal",
                   "-ngd", "--no-global-diagonal",
                   "Use the global diagonal for face penalties in the mortar formulation.");
    int elems_per_agg = 2;
    args.AddOption(&elems_per_agg, "-e", "--elems-per-agg",
                   "Number of rectangular partitions in one direction that constitute an agglomerated element.");
    bool coarse_direct = false;
    args.AddOption(&coarse_direct, "--coarse-direct", "--coarse-direct",
                   "--coarse-amg", "--coarse-amg",
                   "Use a direct solver on coarsest level or AMG V-cycle.");
    bool indirect_rhs = false;
    args.AddOption(&indirect_rhs, "-ind", "--indirect-rhs",
                   "---no-ind", "--no-indirect-rhs",
                   "When --auxiliary is unset, obtain RHS, for the mortar problem, via the transfer operator. "
                   "Otherwise, construct it consistently with the actual discrete weak form.");
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
                   "Auxiliary solver for the H1 problem. If NOT set, then the resulting mortar problem is solved, "
                   "condensed to agglomerate faces.");
    int nu_relax_aux = 2;
    args.AddOption(&nu_relax_aux, "-na", "--nu-relax-aux",
                   "Degree for smoother in the relaxation for the auxiliary-space solver. "
                   "That is for the smoother on the H1 form before and after invoking the auxiliary "
                   "mortar correction.");
    bool compute_errors = false;
    args.AddOption(&compute_errors, "-ce", "--compute-errors",
                   "-nce", "--no-compute-errors",
                   "Whether to compute errors comparing solutions obtained here to the H1 solution.");
    bool solver_agg = false;
    args.AddOption(&solver_agg, "-sa", "--solver-agglomerate",
                   "-nsa", "--no-solver-agglomerate",
                   "Whether to aggregate when building AMGe for the mortar problem or to remain with "
                   "the original aggregates of the mortar formulation.");
    double tol = 1e-8;
    args.AddOption(&tol, "-tol", "--tolerance",
                   "Relative tolerance for solver convergence.");
    int svd_min_skip = 0;
    args.AddOption(&svd_min_skip, "-sms", "--svd-min-skip",
                   "Minimal number of left singular vectors to remove when using SVD in the AMGe multigrid for the mortar problem.");
    bool use_cg = false;
    args.AddOption(&use_cg, "-cg", "--cg",
                   "-ncg", "--no-cg",
                   "Use CG on the auxiliary space, preconditioned by the SAAMGe (or BoomerAMG). Otherwise, use a basic static iteration of SAAMGe.");
    int iters = 1;
    args.AddOption(&iters, "-iters", "--iterations",
                   "Number of static linear or CG iterations on the auxiliary level.");
    bool use_saamge = true;
    args.AddOption(&use_saamge, "-saamge", "--saamge",
                   "-nsaamge", "--no-saamge",
                   "Use SAAMGe on the auxiliary level. Otherwise, use BoomerAMG.");
    bool additive = false;
    args.AddOption(&additive, "-ad", "--additive",
                   "-nad", "--no-additive",
                   "Use an additive auxiliary space preconditioner. Otherwise, it defaults to a multiplicative one.");
    int times_add_smooth = 2;
    args.AddOption(&times_add_smooth, "-tad", "--times-add-smooth",
                   "Times to apply the smoother in an additive setting.");

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

    if (!aux)
        use_saamge = true;

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
    int avg_procs_on_side = 1 << (int)log2((2 == dim ? sqrt((double)nprocs) : cbrt((double)nprocs)) + 1e-10);
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

    tg_data = tg_init_data(*Ag, *agg_part_rels, 0, nu_relax_aux, 1.0, false, 0.0, false);
    tg_data->polynomial_coarse_space = -1;

    Vector diag;
    if (global_diag)
        mbox_obtain_global_diagonal(*Ag, *(agg_part_rels->Dof_TrueDof), diag);

    Array<Vector *> targets;
    fem_polynomial_targets(&fes, targets, faceorder);

    mortar_discretization(*tg_data, *agg_part_rels, emp, &targets, global_diag?&diag:NULL);
    const double OC_mortar = tg_print_data(*Ag, tg_data);

    fem_free_targets(targets);

    Array<Matrix *> *elmats=NULL;
    ElementMatrixProvider *emp_mortar;
    if (use_saamge && !solver_agg)
    {
        agg_part_rels_saamge = nonconf_create_partitioning(*agg_part_rels, *tg_data->interp_data);
        emp_mortar = new ElementIPMatrix(*agg_part_rels_saamge, *tg_data->interp_data);
    }
    else if (use_saamge)
    {
        Table cface_to_AE;
        Transpose(*agg_part_rels->AE_to_cface, cface_to_AE, agg_part_rels->num_cfaces);
        Table *elem_to_elem = Mult(*agg_part_rels->AE_to_cface, cface_to_AE);

        Table *elem_to_dof = nonconf_create_AE_to_dof(*agg_part_rels, *tg_data->interp_data);

        agg_dof_status_t *bdr_dofs = new agg_dof_status_t[agg_part_rels->cface_cDof_TruecDof->GetNumRows()]();
        const int lev_elems_per_agg = (2 == dim ? elems_per_agg * elems_per_agg :
                                                  elems_per_agg * elems_per_agg * elems_per_agg);
        nparts = (int) round((double) nparts / (double) lev_elems_per_agg);
        agg_part_rels_saamge = agg_create_partitioning_fine(*tg_data->Ac, agg_part_rels->nparts, elem_to_dof,
                                   elem_to_elem, NULL, bdr_dofs, &nparts, agg_part_rels->cface_cDof_TruecDof, false);
        delete [] bdr_dofs;

        elmats = new Array<Matrix *>(agg_part_rels->nparts);
        for (int i=0; i < agg_part_rels->nparts; ++i)
        {
            (*elmats)[i] = nonconf_AE_matrix(*tg_data->interp_data, i);
            SA_ASSERT((*elmats)[i]);
        }
        emp_mortar = new ElementMatrixArray(*agg_part_rels_saamge, *elmats);
    }
    double OC_aux = 0.0;
    if (use_saamge && ml)
    {
        int nparts_arr[nl-1];
        nparts_arr[0] = nparts;
        const int lev_elems_per_agg = (2 == dim ? elems_per_agg * elems_per_agg :
                                                  elems_per_agg * elems_per_agg);
        for (int i=1; i < nl-1; ++i)
        {
            nparts_arr[i] = solver_agg ? (int) round((double) nparts_arr[i-1] / (double) lev_elems_per_agg) : nparts_arr[i-1];
            if (nparts_arr[i] < 1) nparts_arr[i] = 1;
        }
        MultilevelParameters mlp(nl-1, nparts_arr, 0, 0, nu_relax, theta_saamge,
                                 theta_saamge_c, -1, false, !direct_eigensolver, false,
                                 fixed_num_evecs_saamge, fixed_num_evecs_saamge_c);
        mlp.set_svd_min_skip(svd_min_skip);
        if (coarse_direct)
            mlp.set_coarse_direct(true);
        ml_data_saamge = ml_produce_data(*tg_data->Ac, agg_part_rels_saamge, emp_mortar, mlp);
        OC_aux = ml_compute_OC(*tg_data->Ac, *ml_data_saamge);
    } else if (use_saamge)
    {
        tg_data_saamge = tg_produce_data(*tg_data->Ac, *agg_part_rels_saamge, 0, nu_relax, emp_mortar, theta_saamge, false, -1,
                                         !direct_eigensolver, false, fixed_num_evecs_saamge, svd_min_skip);
        tg_fillin_coarse_operator(*tg_data->Ac, tg_data_saamge, false);

        Solver *solver;
        if (coarse_direct)
            solver = new HypreDirect(*tg_data_saamge->Ac);
        else
            solver = new AMGSolver(*tg_data_saamge->Ac, false, 1e-16, 1);
        tg_data_saamge->coarse_solver = solver;
        OC_aux = tg_print_data(*tg_data->Ac, tg_data_saamge);
    }
    SA_RPRINTF(0, "Total OC: %g\n", 1.0 + OC_aux*(OC_mortar - 1.0));
    if (use_saamge && solver_agg)
    {
        SA_ASSERT(elmats);
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

    // Obtain mortar solution.
    HypreParVector cx(*tg_data->Ac);
    HypreParVector *cbg = NULL;
    HypreParVector *precbg=NULL;
    Solver *solver=NULL;
    Solver *solver_iter=NULL;
    if (aux)
    {
        if (use_saamge)
        {
            solver = new VCycleSolver((ml ? levels_list_get_level(ml_data_saamge->levels_list, 0)->tg_data :
                                        tg_data_saamge), false);
            solver->SetOperator(*tg_data->Ac);
        } else
        {
            solver = new HypreBoomerAMG(*tg_data->Ac);
            ((HypreBoomerAMG *)solver)->SetPrintLevel(0);
        }

        if (use_cg)
        {
            CGSolver *hpcg = new CGSolver(MPI_COMM_WORLD);
            hpcg->SetOperator(*tg_data->Ac);
            hpcg->SetRelTol(0.0); // MFEM squares this.
            hpcg->SetMaxIter(iters);
            hpcg->SetPrintLevel(-1);
            hpcg->SetPreconditioner(*solver);
            solver_iter = hpcg;
        } else if (iters > 1)
        {
            SLISolver *sli = new SLISolver(MPI_COMM_WORLD);
            sli->SetOperator(*tg_data->Ac);
            sli->SetRelTol(0.0); // MFEM squares this.
            sli->SetMaxIter(iters);
            sli->SetPrintLevel(-1);
            sli->SetPreconditioner(*solver);
            solver_iter = sli;
        } else
        {
            solver_iter = solver;
            solver = NULL;
        }

        tg_data->coarse_solver = new MortarSchurSolver(*tg_data->interp_data, *agg_part_rels, *solver_iter);
    } else
    {
        if (zero_rhs)
            cbg = new HypreParVector(*tg_data->Ac);
        else
        {
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
            tg_run(*tg_data->Ac, agg_part_rels_saamge, cx, *cbg, 1000, tol, 1e-24, 1.0,
                   (ml ? levels_list_get_level(ml_data_saamge->levels_list, 0)->tg_data :
                   tg_data_saamge), zero_rhs, true);
            delete hxg;
            if (!indirect_rhs || zero_rhs)
                hxg = mortar_reverse_condensation(cx, *tg_data, *agg_part_rels, zero_rhs);
            else
            {
                HypreParVector postcx(*tg_data->interp);
                mortar_schur_recover(*tg_data->interp_data, *agg_part_rels, *precbg, cx, postcx);
                hxg = new HypreParVector(*tg_data->interp, 1);
                tg_data->interp->Mult(postcx, *hxg);
            }
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
        SmootherSolver Sprec(smpr_sym_poly, tg_data->poly_data, false, times_add_smooth);
        Sprec.SetOperator(*Ag);
        AdditiveSolver Aprec(Sprec, Bprec);

        CGSolver hpcg(MPI_COMM_WORLD);
        hpcg.SetOperator(*Ag);
        hpcg.SetRelTol(sqrt(tol)); // MFEM squares this.
        hpcg.SetMaxIter(1000);
        hpcg.SetPrintLevel(1);
        if (additive)
        {
            tg_data->pre_smoother = tg_data->post_smoother = solve_empty;
            hpcg.SetPreconditioner(Aprec);
        } else
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
        hpcg.Mult(*cbg, cx);
        iterations = hpcg.GetNumIterations();
        converged = hpcg.GetConverged();
        delete pxg;
        if (!indirect_rhs || zero_rhs)
            pxg = mortar_reverse_condensation(cx, *tg_data, *agg_part_rels, zero_rhs);
        else
        {
            HypreParVector postcx(*tg_data->interp);
            mortar_schur_recover(*tg_data->interp_data, *agg_part_rels, *precbg, cx, postcx);
            pxg = new HypreParVector(*tg_data->interp, 1);
            tg_data->interp->Mult(postcx, *pxg);
        }
    }
    delete precbg;
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

    if (!use_saamge)
    {
        if (solver)
            hypre_BoomerAMGSetupStats((HYPRE_Solver)(*(HypreBoomerAMG *)solver), (hypre_ParCSRMatrix *)*tg_data->Ac);
        else
            hypre_BoomerAMGSetupStats((HYPRE_Solver)(*(HypreBoomerAMG *)solver_iter), (hypre_ParCSRMatrix *)*tg_data->Ac);
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
    delete solver_iter;
    agg_free_partitioning(agg_part_rels);
    agg_free_partitioning(agg_part_rels_saamge);
    delete mesh;
    MPI_Finalize();
    return 0;
}
