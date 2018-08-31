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
    double theta = 0.1;
    args.AddOption(&theta, "-t", "--theta",
                   "Tolerance for eigenvalue problems.");
    bool full_space = true;
    args.AddOption(&full_space, "-f", "--full-space",
                   "-nf", "--no-full-space",
                   "Build the full space instead of using eigensolvers.");
    bool schur = false;
    args.AddOption(&schur, "-s", "--schur",
                   "-ns", "--no-schur",
                   "Whether to use the Schur complement on the IP problem.");
    double delta = 1e-6;
    args.AddOption(&delta, "-d", "--delta",
                   "The reciprocal of the interface term weight.");
    int elems_per_agg = 256;
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
    bool saamge = false;
    args.AddOption(&saamge, "-a", "--saamge",
                   "-na", "--no-saamge",
                   "Test SAAMGe on the IP problem.");

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
    mesh = fem_read_mesh(mesh_file);
    fem_refine_mesh_times(serial_times_refine, *mesh);
    // Serial mesh.
    SA_RPRINTF(0,"NV: %d, NE: %d\n", mesh->GetNV(), mesh->GetNE());

    // Parallel mesh and finite elements stuff.
    Array<int> ess_bdr(mesh->bdr_attributes.Max());
    ess_bdr = 0;
    ess_bdr[3] = 1; // marked as 4 in mltest.mesh, but MFEM subtracts 1

    int nprocs = PROC_NUM;
    int *proc_partitioning;
    proc_partitioning = fem_partition_mesh(*mesh, &nprocs);
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
    agg_part_rels = fem_create_partitioning(*Ag, fes, bdr_dofs, &nparts, false);
    delete [] bdr_dofs;
    fem_build_face_relations(agg_part_rels, fes);
    if (visualize)
        fem_parallel_visualize_partitioning(pmesh, agg_part_rels->partitioning, nparts);

    ElementMatrixStandardGeometric *emp = new ElementMatrixStandardGeometric(*agg_part_rels, Al, a);

//    const bool avoid_ess_bdr_dofs = true;
//    tg_data = tg_produce_data(*Ag, *agg_part_rels, 3, 3, emp, theta, false, -1,
//                              !direct_eigensolver, avoid_ess_bdr_dofs);
//    tg_fillin_coarse_operator(*Ag, tg_data, false);

    tg_data = tg_init_data(*Ag, *agg_part_rels, 0, 1, theta, false, 0.0, !direct_eigensolver);
    tg_data->polynomial_coarse_space = -1;

    nonconf_ip_coarsen_finest(*tg_data, *agg_part_rels, emp, theta, delta, schur, full_space);

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
    if (schur)
        tg_data->coarse_solver = new SchurSolver(*tg_data->interp_data, *agg_part_rels, *agg_part_rels->cface_cDof_TruecDof,
                                                 *agg_part_rels->cface_TruecDof_cDof, *solver);
    else
        tg_data->coarse_solver = solver;

    tg_print_data(*Ag, tg_data);

    // Obtain the usual solution.
    fsolver->Mult(*bg, *hxg);
    x = *hxg;
    if (visualize)
        fem_parallel_visualize_gf(pmesh, x);

    // Obtain IP solution.
    HypreParVector *hx1g = x1.ParallelAverage();
    HypreParVector cbg(*tg_data->interp);
    HypreParVector cx(*tg_data->interp);
    tg_data->restr->Mult(*bg, cbg);
    tg_data->coarse_solver->Mult(cbg, cx);
    tg_data->interp->Mult(cx, *hx1g);
    x1 = *hx1g;
    if (visualize)
        fem_parallel_visualize_gf(pmesh, x1);

    // Error
    GridFunctionCoefficient xgf(&x);
    const double l2err = x1.ComputeL2Error(xgf);
    const double maxerr = x1.ComputeMaxError(xgf);
    x -= x1;
    if (visualize)
        fem_parallel_visualize_gf(pmesh, x);
    HypreParVector *gx = x.ParallelProject();
    const double energyerr = mbox_energy_norm_parallel(*Ag, *gx);
    SA_RPRINTF(0, "ERROR: L2 = %g; Linf = %g; ENERGY = %g\n", l2err, maxerr, energyerr);

//    for (int i=0; i < tg_data->interp_data->num_cfaces; ++i)
//        SA_PRINTF("cface %d: %dx%d\n", i, tg_data->interp_data->cfaces_bases[i]->Height(), tg_data->interp_data->cfaces_bases[i]->Width());
//    for (int i=0; i < tg_data->interp_data->nparts; ++i)
//        SA_PRINTF("celem %d: %dx%d\n", i, tg_data->interp_data->cut_evects_arr[i]->Height(), tg_data->interp_data->cut_evects_arr[i]->Width());

    if (saamge && schur)
    {
        agg_partitioning_relations_t *amg_agg_part_rels;
        tg_data_t *amg_tg_data;

        Table cface_to_AE;
        Transpose(*agg_part_rels->AE_to_cface, cface_to_AE, agg_part_rels->num_cfaces);
        Table *elem_to_elem = Mult(*agg_part_rels->AE_to_cface, cface_to_AE);

        Table cface_to_cdof;
        int *cface_cdofs_J = new int[tg_data->interp_data->cfaces_cdofs_offsets[agg_part_rels->num_cfaces]];
        for (int i=0; i < tg_data->interp_data->cfaces_cdofs_offsets[agg_part_rels->num_cfaces]; ++i)
            cface_cdofs_J[i] = i;
        cface_to_cdof.SetIJ(tg_data->interp_data->cfaces_cdofs_offsets, cface_cdofs_J, agg_part_rels->num_cfaces);
        Table *elem_to_dof = Mult(*agg_part_rels->AE_to_cface, cface_to_cdof);
        cface_to_cdof.LoseData();
        delete [] cface_cdofs_J;

        agg_dof_status_t *bdr_dofs = new agg_dof_status_t[tg_data->interp_data->cfaces_cdofs_offsets[agg_part_rels->num_cfaces]]();
        nparts = (int)(nparts/10);
        amg_agg_part_rels = agg_create_partitioning_fine(*tg_data->Ac, agg_part_rels->nparts, elem_to_dof, elem_to_elem, NULL, bdr_dofs, &nparts, agg_part_rels->cface_cDof_TruecDof, false);
        delete [] bdr_dofs;

        Array<DenseMatrix*> elmats(agg_part_rels->nparts);
        for (int i=0; i < agg_part_rels->nparts; ++i)
            elmats[i] = tg_data->interp_data->schurs[i];
        ElementMatrixDenseArray *emp = new ElementMatrixDenseArray(*amg_agg_part_rels, elmats);

        amg_tg_data = tg_produce_data(*tg_data->Ac, *amg_agg_part_rels, 3, 3, emp, theta, false, -1, !direct_eigensolver, true);
        tg_fillin_coarse_operator(*tg_data->Ac, amg_tg_data, false);

        HypreParVector xg(*tg_data->Ac);
        HypreParVector bg(*tg_data->Ac);
        bg = 0.0;
        tg_run(*tg_data->Ac, agg_part_rels, xg, bg, 1000, 10e-12, 10e-24, 1., amg_tg_data, true, true);

        delete amg_tg_data;
        delete amg_agg_part_rels;
    }

    delete gx;
    tg_free_data(tg_data);
    if (schur)
        delete solver;
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
