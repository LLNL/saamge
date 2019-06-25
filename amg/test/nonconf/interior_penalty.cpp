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
    Nonconforming interior penalty (IP) AMGe as a discretization for an elliptic problem.

    This example starts with an H1 problem and produces an agglomeration.
    Based on the agglomerates it builds IP spaces (on the agglomerate
    elements and agglomerate faces) and formulation together with transition operators (using averaging)
    between H1 and the constructed IP spaces. The IP spaces can be fine-scale or coarse
    (constructed via eigenvalue problems for the H1 agglomerate matrices or local fine-scale IP matrices).
    If they are coarse the transition operators are straight between fine H1 and coarse IP spaces.
    The IP problem can be the entire formulation or condensed to the agglomerate faces
    (utilizing a Schur complement) for both fine- and coarse-scale IP formulations.

    The matrix of the IP system is obtained via assembly, whereas the right-hand side of the
    IP system can be assembled (in a fine-scale IP formulation) or obtained via the transition
    operators (in a coarse-scale IP formulation). This does NOT result in equal right-hand sides but
    seems to provide similar results.

    In the end, it inverts "exactly" the IP problem and the H1 problem, and uses the transition
    operators to compute errors between the two solutions.

    It is intended for solver settings, which means that we consider essential BCs (boundary conditions) that
    can only be zero. Essential BCs are strongly enforced in the IP formulation by considering basis
    (both the ones associated with the agglomerates and the agglomerate faces) functions that are
    NOT supported on the respective portion of the boundary.

    XXX: If the agglomerates are of fixed size relative to the fine mesh,
         The IP discretization does NOT converge, but it still provides preconditioner properties.

    XXX: One can also test a two-level SAAMGe on the resulting entire or condensed (Schur)
         IP problem, using the same theta and some hard-coded parameters. It considers the agglomerates
         as elements and uses the IP dofs to obtain elem_to_elem and elem_to_dof. From there,
         it calls the standard procedures of SAAMGe that obtain partitions by grouping
         elements (i.e., agglomerates) together, using, e.g., METIS, and constructing the hierarchy.
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
    ParGridFunction x, x1, err;
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
    bool full_space = true;
    args.AddOption(&full_space, "-f", "--full-space",
                   "-nf", "--no-full-space",
                   "Build the full IP space instead of using eigensolvers.");
    bool ip_spectral = false;
    args.AddOption(&ip_spectral, "-ips", "--ip-spectral",
                   "-nips", "--no-ip-spectral",
                   "If not using full space, this selects whether to use spectral construction based on "
                   "local H1 or local fine-scale IP matrices.");
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
    bool saamge = false;
    args.AddOption(&saamge, "-a", "--saamge",
                   "-na", "--no-saamge",
                   "Test a two-level SAAMGe on the resulting entire or condensed (Schur) IP problem, "
                   "using the same theta and some hard-coded parameters.");

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
    err.SetSpace(&fes);

    SparseMatrix& Al = a->SpMat();
    HypreParMatrix *Ag = a->ParallelAssemble();
    HypreParVector *bg = b->ParallelAssemble();

//    MassIntegrator mi(conduct_coeff);
//    ElementTransformation *eltrans;
//    DenseMatrix Mloc;
//    const FiniteElement *fe;
//    eltrans = pmesh.GetEdgeTransformation(10);
//    fe = fes.FEColl()->FiniteElementForGeometry(Geometry::SEGMENT);
//    mi.AssembleElementMatrix(*fe, *eltrans, Mloc);
//    Mloc *= (double)(1 << (serial_times_refine + times_refine));
//    std::ofstream myfile0;
//    myfile0.open("Mloc.out");
//    Mloc.Print(myfile0);
//    myfile0.close();

//    DGDiffusionIntegrator dgi(conduct_coeff, -1.0, 1.0);
//    FaceElementTransformations *tr;
//    DenseMatrix Mloc;
//    tr = pmesh.GetInteriorFaceTransformations(10);
//    dgi.AssembleFaceMatrix(*fes.GetFE(tr->Elem1No), *fes.GetFE(tr->Elem2No), *tr, Mloc);
//    std::ofstream myfile0;
//    myfile0.open("Mloc.out");
//    Mloc.Print(myfile0);
//    myfile0.close();

//    TraceJumpIntegrator dgi;
//    FaceElementTransformations *tr;
//    DenseMatrix Mloc;
//    tr = pmesh.GetFaceElementTransformations(10);
//    dgi.AssembleFaceMatrix(*fes.GetFaceElement(10), *fes.GetFE(tr->Elem1No), *fes.GetFE(tr->Elem2No), *tr, Mloc);
//    Mloc *= (double)(1 << (serial_times_refine + times_refine));
//    std::ofstream myfile0;
//    myfile0.open("Mloc.out");
//    Mloc.Print(myfile0);
//    myfile0.close();

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

    tg_data = tg_init_data(*Ag, *agg_part_rels, 0, 1, theta, false, 0.0, !direct_eigensolver);
    tg_data->polynomial_coarse_space = -1;

    Vector diag;
    if (global_diag)
        mbox_obtain_global_diagonal(*Ag, *(agg_part_rels->Dof_TrueDof), diag);

    if (full_space)
        nonconf_ip_discretization(*tg_data, *agg_part_rels, emp, delta, global_diag?&diag:NULL, schur);
    else if (!ip_spectral)
        nonconf_ip_coarsen_finest_h1(*tg_data, *agg_part_rels, emp, theta, delta, global_diag?&diag:NULL, schur, full_space);
    else
        nonconf_ip_coarsen_finest_ip(*tg_data, *agg_part_rels, emp, theta, delta, global_diag?&diag:NULL, schur);
    tg_print_data(*Ag, tg_data);

    Solver *solver;
    Solver *fsolver;
    if (coarse_direct)
    {
        solver = new HypreDirect(*tg_data->Ac);
        fsolver = new HypreDirect(*Ag);
    } else
    {
        solver = new AMGSolver(*tg_data->Ac, false, 1e-16, 1);
        fsolver = new AMGSolver(*Ag, false);
    }
    if (schur)
        tg_data->coarse_solver = new SchurSolver(*tg_data->interp_data, *agg_part_rels, *agg_part_rels->cface_cDof_TruecDof,
                                                 *agg_part_rels->cface_TruecDof_cDof, *solver);
    else
    {
        tg_data->coarse_solver = solver;
        solver = NULL;
    }

//    SparseMatrix *B=NULL, mat;
//    DenseMatrix evec;
//    double theta_;
//    Eigensolver eigensolver(agg_part_rels->mises, *agg_part_rels, 10000);
//    tg_data->Ac->GetDiag(mat);
//    eigensolver.Solve(mat, B, 0, 0, 0, theta_, evec, 1);
//    delete B;
//    std::ofstream myfile;
//    myfile.open("globalevec.out");
//    Vector vec(evec.GetData(), evec.Height());
//    vec.Print(myfile);
//    myfile.close();
//    HypreParVector *hvec = x1.ParallelAverage();
//    tg_data->interp->Mult(vec, *hvec);
//    x1 = *hvec;
//    fem_parallel_visualize_gf(pmesh, x1);
//    delete hvec;
//
//    B = NULL;
//    evec.SetSize(0);
//    eigensolver.Solve(Al, B, 0, 0, 0, theta_, evec, 1);
//    delete B;
//    myfile.open("globalevec1.out");
//    Vector vec1(evec.GetData(), evec.Height());
//    vec1.Print(myfile);
//    myfile.close();
//    x1 = vec1;
//    fem_parallel_visualize_gf(pmesh, x1);

    // Obtain the H1 solution.
    HypreParVector *hxg = x.ParallelAverage();
    fsolver->Mult(*bg, *hxg);
    delete fsolver;
    x = *hxg;
    delete hxg;
//    FunctionCoefficient uex(ex_func);
//    x.ProjectCoefficient(uex);
    if (visualize)
        fem_parallel_visualize_gf(pmesh, x);

    // Obtain IP solution.
    HypreParVector *cbg;
    HypreParVector cx(*tg_data->interp);
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
    HypreParVector *hx1g = x1.ParallelAverage();
    tg_data->coarse_solver->Mult(*cbg, cx);
    tg_data->interp->Mult(cx, *hx1g);
    x1 = *hx1g;
    delete hx1g;
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
    delete gx;
    SA_RPRINTF(0, "ERROR: L2 = %g; Linf = %g; ENERGY = %g\n", l2err, maxerr, energyerr);

//    std::cout << "celements_cdofs: " << tg_data->interp_data->celements_cdofs << std::endl;
//    std::cout << "coarse_truedof_offset: " << tg_data->interp_data->coarse_truedof_offset << std::endl;
//    for (int i=0; i <= tg_data->interp_data->nparts; ++i)
//        std::cout << i << ": " << tg_data->interp_data->celements_cdofs_offsets[i] << std::endl;
//    std::cout << std::endl;
//    for (int i=0; i <= tg_data->interp_data->num_cfaces; ++i)
//        std::cout << i << ": " << tg_data->interp_data->cfaces_cdofs_offsets[i] << std::endl;
//    std::cout << std::endl;
//    for (int i=0; i <= tg_data->interp_data->num_cfaces; ++i)
//        std::cout << i << ": " << tg_data->interp_data->cfaces_truecdofs_offsets[i] << std::endl;

//    for (int i=0; i < tg_data->interp_data->num_cfaces; ++i)
//        SA_PRINTF("cface %d: %dx%d\n", i, tg_data->interp_data->cfaces_bases[i]->Height(), tg_data->interp_data->cfaces_bases[i]->Width());
//    for (int i=0; i < tg_data->interp_data->nparts; ++i)
//        SA_PRINTF("celem %d: %dx%d\n", i, tg_data->interp_data->cut_evects_arr[i]->Height(), tg_data->interp_data->cut_evects_arr[i]->Width());

//    for (int i=0; i < agg_part_rels->num_cfaces; ++i)
//    {
//        x = 0.0;
//        for (int j=0; j < agg_part_rels->cface_to_dof->RowSize(i); ++j)
//            x(agg_part_rels->cface_to_dof->GetRow(i)[j]) = 1.0;
//        fem_parallel_visualize_gf(pmesh, x);
//        int foo;
//        std::cin >> foo;
//    }

//    *(tg_data->interp) = 1.0;
//    HypreParVector tx(*tg_data->interp);
//    HypreParVector *tmp = x1.ParallelAverage();
//    for (int i=0; i < agg_part_rels->nparts; ++i)
//    {
//        tx = 0.0;
//        for (int j = tg_data->interp_data->celements_cdofs_offsets[i];
//             j < tg_data->interp_data->celements_cdofs_offsets[i+1]; ++j)
//        {
//            tx(j) = cx(j);
//        }
//        tg_data->interp->Mult(tx, *tmp);
//        x = *tmp;
//        fem_parallel_visualize_gf(pmesh, x);
//        int foo;
//        std::cin >> foo;
//    }
//    delete tmp;

//    for (int i=0; i < agg_part_rels->nparts; ++i)
//    {
//        err = 0.0;
//        for (int j=0; j < agg_part_rels->AE_to_dof->RowSize(i); ++j)
//        {
//            const int idx = agg_part_rels->AE_to_dof->GetRow(i)[j];
//            err(idx) = x(idx);
//        }
//        fem_parallel_visualize_gf(pmesh, err);
//        int foo;
//        std::cin >> foo;
//    }

//    *(tg_data->interp) = 1.0;
//    HypreParVector tx(*tg_data->interp);
//    HypreParVector *tmp = x1.ParallelAverage();
//    for (int i=0; i < agg_part_rels->nparts; ++i)
//    {
//        err = 0.0;
//        tx = 0.0;
//        for (int j=0; j < agg_part_rels->AE_to_dof->RowSize(i); ++j)
//        {
//            const int idx = agg_part_rels->AE_to_dof->GetRow(i)[j];
//            err(idx) = x(idx);
//        }
//        for (int j = tg_data->interp_data->celements_cdofs_offsets[i];
//             j < tg_data->interp_data->celements_cdofs_offsets[i+1]; ++j)
//        {
//            tx(j) = cx(j);
//        }
//        tg_data->interp->Mult(tx, *tmp);
//        err -= *tmp;
//        fem_parallel_visualize_gf(pmesh, err);
//        int foo;
//        std::cin >> foo;
//    }

//    HypreParVector onec(*tg_data->Ac), rsumc(*tg_data->Ac, 1);
//    HypreParVector one(*Ag), rsum(*Ag, 1), rsumc1(*Ag, 1);
//    onec = 1.0;
//    one = 1.0;
//    tg_data->Ac->Mult(onec, rsumc);
//    Ag->Mult(one, rsum);
//    *(tg_data->interp) = 1.0;
//    tg_data->interp->Mult(rsumc, rsumc1);
//    for (int i=0; i < agg_part_rels->ND; ++i)
//        if (SA_IS_SET_A_FLAG(agg_part_rels->agg_flags[i], AGG_ON_ESS_DOMAIN_BORDER_FLAG))
//        {
//            rsum(i) = -15.0;
//            rsumc1(i) = -15.0;
//        }
//    rsumc.Print("rsumc.out");
//    rsum.Print("rsum.out");
//    rsumc1.Print("rsumc1.out");

//    cbg->Print("cbg.out");
//    bg->Print("bg.out");
//    HypreParVector bg1(*Ag, 1);
//    *(tg_data->interp) = 1.0;
//    tg_data->interp->Mult(*cbg, bg1);
//    bg1.Print("bg1.out");

//    *(tg_data->restr) = 1.0;
//    HypreParMatrix *Ag1 = RAP(tg_data->Ac, tg_data->restr);
//    Ag->Print("Ag.out");
//    Ag1->Print("Ag1.out");
//    Ag->Add(-1.0, *Ag1);
//    Ag->Print("Ag_diff.out");
//    delete Ag1;
//    HypreParVector one0(*Ag);
//    one0 = 1.0;
//    std::cout << "Total sum: " << mbox_energy_norm_parallel(*Ag, one0) << std::endl;

    delete bg;
    delete Ag;
    delete a;
    delete b;

    if (saamge)
    {
        agg_partitioning_relations_t *saamg_agg_part_rels;
        tg_data_t *saamg_tg_data;

        Table cface_to_AE;
        Transpose(*agg_part_rels->AE_to_cface, cface_to_AE, agg_part_rels->num_cfaces);
        Table *elem_to_elem = Mult(*agg_part_rels->AE_to_cface, cface_to_AE);

        Table *elem_to_dof = nonconf_create_AE_to_dof(*agg_part_rels, *tg_data->interp_data);

        agg_dof_status_t *bdr_dofs = new agg_dof_status_t[agg_part_rels->cface_cDof_TruecDof->GetNumRows()]();
        nparts = (int)(nparts/10);
        saamg_agg_part_rels = agg_create_partitioning_fine(*tg_data->Ac, agg_part_rels->nparts, elem_to_dof, elem_to_elem, NULL, bdr_dofs, &nparts, agg_part_rels->cface_cDof_TruecDof, false);
        delete [] bdr_dofs;

        Array<Matrix *> elmats(agg_part_rels->nparts);
        for (int i=0; i < agg_part_rels->nparts; ++i)
        {
            elmats[i] = nonconf_AE_matrix(*tg_data->interp_data, i);
            SA_ASSERT(elmats[i]);
        }
        ElementMatrixArray *emp = new ElementMatrixArray(*saamg_agg_part_rels, elmats);

        saamg_tg_data = tg_produce_data(*tg_data->Ac, *saamg_agg_part_rels, 0, 3, emp, theta, false, -1, !direct_eigensolver, true);
        tg_fillin_coarse_operator(*tg_data->Ac, saamg_tg_data, false);
        if (coarse_direct)
            saamg_tg_data->coarse_solver = new HypreDirect(*saamg_tg_data->Ac);
        else
            saamg_tg_data->coarse_solver = new AMGSolver(*saamg_tg_data->Ac, false);

        HypreParVector xg(*tg_data->Ac);
        xg = 0.0;
        tg_run(*tg_data->Ac, agg_part_rels, xg, *cbg, 1000, 10e-12, 10e-24, 1., saamg_tg_data, true, true);

        tg_free_data(saamg_tg_data);
        agg_free_partitioning(saamg_agg_part_rels);
        for (int i=0; i < agg_part_rels->nparts; ++i)
            delete elmats[i];
    }

    delete cbg;
    tg_free_data(tg_data);
    delete solver;
    agg_free_partitioning(agg_part_rels);
    delete mesh;
    MPI_Finalize();
    return 0;
}
