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
   Take a canonical spectral coarsening of a fine grid problem, and
   try to compare different strategies for solving the resulting
   coarse problem.

   That is, start on a coarse level instead of trying to develop
   a solver for the fine-level problem.

   Andrew T. Barker
   atb@llnl.gov
   22 September 2015
*/

#include <mfem.hpp>
#include <mpi.h>
#include <saamge.hpp>

#include "InversePermeabilityFunction.hpp"
#include "picojson.h"

using namespace mfem;
using namespace saamge;

/**
  generate hexahedral mesh, usually for SPE10
*/
Mesh * create_hexahedral_mesh(
    unsigned int Nx,
    unsigned int Ny,
    unsigned int Nz,
    double hx,
    double hy,
    double hz)
{
    // build 3d cartesian mesh where each parallelepiped is split into 6 tetrahedrons
    unsigned int nVx=Nx+1;
    unsigned int nVy=Ny+1;
    unsigned int nVz=Nz+1;
    Mesh * mesh = new Mesh(3, nVx*nVy*nVz, Nx*Ny*Nz, 2*(Nx*Ny+Ny*Nz+Nx*Nz));
    double vert_coord[3];
    for (unsigned i=0; i<nVx; i++)
        for (unsigned j=0; j<nVy; j++)
            for (unsigned k=0; k<nVz; k++)
            {
                vert_coord[0] = i*hx;
                vert_coord[1] = j*hy;
                vert_coord[2] = k*hz;
                mesh->AddVertex(vert_coord);
            }
    int vi[8];
    for (unsigned i=0; i<Nx; i++)
        for (unsigned j=0; j<Ny; j++)
            for (unsigned k=0; k<Nz; k++)
            {
                int v000=i*nVz*nVy + (j)*nVz + k;
                int v001=i*nVz*nVy + j*nVz + k+1;
                int v010=i*nVz*nVy + (j+1)*nVz + k;
                int v011=i*nVz*nVy + (j+1)*nVz + k+1;

                int v100=(i+1)*nVz*nVy + (j)*nVz + k;
                int v101=(i+1)*nVz*nVy + (j)*nVz + k+1;
                int v110=(i+1)*nVz*nVy + (j+1)*nVz + k;
                int v111=(i+1)*nVz*nVy + (j+1)*nVz + k+1;

                vi[0]=v000; vi[1]=v100; vi[2]=v110; vi[3]=v010;
                vi[4]=v001; vi[5]=v101; vi[6]=v111; vi[7]=v011;
                mesh->AddHex (vi);

                // add boundary quadrangles
                if (0==i)
                {
                    vi[0]=v000;
                    vi[1]=v001;
                    vi[2]=v011;
                    vi[3]=v010;
                    mesh->AddBdrQuad(vi, 1);
                }
                if (Nx-1==i)
                {
                    vi[0]=v100;
                    vi[1]=v110;
                    vi[2]=v111;
                    vi[3]=v101;
                    mesh->AddBdrQuad(vi, 2);
                }

                if (0==j)
                {
                    vi[0]=v000;
                    vi[1]=v001;
                    vi[2]=v101;
                    vi[3]=v100;
                    mesh->AddBdrQuad(vi, 3);
                }
                if (Ny-1==j)
                {
                    vi[0]=v010;
                    vi[1]=v011;
                    vi[2]=v111;
                    vi[3]=v110;
                    mesh->AddBdrQuad(vi, 4);
                }
                if (0==k)
                {
                    vi[0]=v000;
                    vi[1]=v100;
                    vi[2]=v110;
                    vi[3]=v010;
                    mesh->AddBdrQuad(vi, 5);
                }
                if (Nz-1==k)
                {
                    vi[0]=v001;
                    vi[1]=v101;
                    vi[2]=v111;
                    vi[3]=v011;
                    mesh->AddBdrQuad(vi, 6);
                }
            }
    mesh->FinalizeHexMesh (1, 0); // includes CheckBdrElementOrientation()
    return mesh;
}

double rhs_func(Vector& x)
{
    SA_ASSERT(2 <= x.Size() && x.Size() <= 3);
    return 1;
}

double bdr_cond(Vector& x)
{
    SA_ASSERT(2 <= x.Size() && x.Size() <= 3);
    return 0.;
}

int main(int argc, char *argv[])
{
    // Initialize process related stuff.
    MPI_Init(&argc, &argv);
    proc_init(MPI_COMM_WORLD);

    StopWatch chrono;
    chrono.Clear();
    chrono.Start();

    Mesh *mesh;
    ParMesh *pmesh;
    int *proc_partitioning;
    ParGridFunction x;
    ParLinearForm *b;
    ParBilinearForm *a;
    agg_partitioning_relations_t *agg_part_rels;
    ml_data_t *ml_data;

    const char *perm_file = "/home/barker29/spe10/spe_perm.dat";
    bool visualize = false;

    // parameters for first coarsening, build coarse problem
    // these are intended to be the same all the time, ie, we fix one
    // canonical coarse problem and compare different ways of solving it
    int spe10_scale = 5;    
    int first_nu_pro = 1;
    double first_theta = 1.e-4;    
    int first_elems_per_agg = 512;

    // parameters for coarse solver / multilvel hierarchy
    int nu_pro = 0;
    int nu_relax = 3;
    double theta = 0.003;
    int num_levels = 3;
    int elems_per_agg = 256;
    bool minimal_coarse = false;
    bool correct_nullspace = true;
    // bool double_cycle = false; // eventually may want to try this, or partial_smooth...

    OptionsParser args(argc, argv);
    args.AddOption(&perm_file, "-pf", "--perm",
                   "Permeability data, only relevant with --spe10.");
    args.AddOption(&visualize, "-vis", "--visualization", "-no-vis",
                   "--no-visualization",
                   "Enable or disable GLVis visualization.");
    args.AddOption(&nu_pro, "-p", "--nu-pro",
                   "Degree of the smoother for the smoothed aggregation.");
    args.AddOption(&nu_relax, "-n", "--nu-relax",
                   "Degree for smoother in the relaxation.");
    args.AddOption(&spe10_scale, "-ss", "--spe10-scale",
                   "Scale of SPE10, 5 is full, smaller is smaller, larger does not make sense.");
    args.AddOption(&theta, "-t", "--theta",
                   "Tolerance for eigenvalue problems.");
    args.AddOption(&num_levels, "-l", "--num-levels",
                   "Number of levels in multilevel algorithm.");
    args.AddOption(&elems_per_agg, "-e", "--elems-per-agg",
                   "Number of elements per agglomerated element.");
    args.AddOption(&minimal_coarse, "-mc", "--minimal-coarse",
                   "-nmc", "--no-minimal-coarse",
                   "Minimal coarse space, ie, vector of all ones, for every coarsening except 0th.");
    args.AddOption(&correct_nullspace, "-c", "--correct-nullspace",
                   "-nc", "--no-correct-nullspace",
                   "Use the corrected nullspace technique on coarsest level.");

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

    picojson::object tryjson;
    picojson::object pjargs;
    picojson::object timings;
    tryjson["invocation"] = picojson::value(argv[0]);
    tryjson["processors"] = picojson::value((double) PROC_NUM);
    pjargs["nu-pro"] = picojson::value((double) nu_pro);
    pjargs["nu-relax"] = picojson::value((double) nu_relax);
    pjargs["spe10-scale"] = picojson::value((double) spe10_scale);
    pjargs["theta"] = picojson::value(theta);
    pjargs["num-levels"] = picojson::value((double) num_levels);
    pjargs["elems-per-agg"] = picojson::value((double) elems_per_agg);
    pjargs["minimal-coarse"] = picojson::value(minimal_coarse);
    pjargs["correct-nullspace"] = picojson::value(correct_nullspace);

    tryjson["arguments"] = picojson::value(pjargs);

    MPI_Barrier(PROC_COMM); // try to make MFEM's debug element orientation prints not mess up the parameters above
    // change Nx, Ny, Nz if you want, but not hx, hy, hz
    // int Nx = 60;
    // int Ny = 220;
    // int Nz = 85;
    int Nx = 12 * spe10_scale;
    int Ny = 44 * spe10_scale;
    int Nz = 17 * spe10_scale;
    double hx = 20.0;
    double hy = 10.0;
    double hz = 2.0;
    
    mesh = create_hexahedral_mesh(Nx, Ny, Nz,
                                  hx, hy, hz);

    InversePermeabilityFunction::SetNumberCells(Nx,Ny,Nz);
    InversePermeabilityFunction::SetMeshSizes(hx, hy, hz);
    InversePermeabilityFunction::ReadPermeabilityFile(perm_file);

    // Serial mesh.
    SA_RPRINTF(0,"NV: %d, NE: %d\n", mesh->GetNV(), mesh->GetNE());

    // Parallel mesh and finite elements stuff.
    Array<int> ess_bdr(mesh->bdr_attributes.Max());
    ess_bdr = 1;
    ess_bdr[0] = 0; 
    ess_bdr[1] = 0; 
        
    FunctionCoefficient bdr_coeff(bdr_cond);
    FunctionCoefficient rhs(rhs_func);

    int nprocs = PROC_NUM;
    proc_partitioning = fem_partition_mesh(*mesh, &nprocs);
    if (0 == PROC_RANK && visualize)
        fem_serial_visualize_partitioning(*mesh, proc_partitioning);
    pmesh = new ParMesh(MPI_COMM_WORLD, *mesh, proc_partitioning);
    delete [] proc_partitioning;

    FiniteElementCollection * fec = new LinearFECollection;
    ParFiniteElementSpace * fes = new ParFiniteElementSpace(pmesh, fec);
    int pNV = pmesh->GetNV();
    int pNE = pmesh->GetNE();
    int pND = fes->GetNDofs();
    int ND = fes->GlobalTrueVSize();
    SA_RPRINTF(0,"pNV: %d, pNE: %d, pND: %d, ND: %d\n", 
               pNV, pNE, pND, ND);

    MatrixFunctionCoefficient * matrix_conductivity = NULL;  
    matrix_conductivity = new MatrixFunctionCoefficient(pmesh->Dimension(),
                                                        InversePermeabilityFunction::PermeabilityTensor);
    fem_build_discrete_problem(fes, rhs, bdr_coeff, *matrix_conductivity, true, x, b,
                               a, &ess_bdr);

    SparseMatrix& Al = a->SpMat();
    HypreParMatrix *Ag = a->ParallelAssemble();
    // HypreParVector *bg = b->ParallelAssemble();
    HypreParVector *hxg = x.ParallelAverage();
    HypreParVector *pxg = x.ParallelAverage();
    chrono.Stop();
    SA_RPRINTF(0, "TIMING: fem setup %f seconds.\n", chrono.RealTime());
    timings["fem-setup"] = picojson::value(chrono.RealTime());

    // some actual AMGe stuff
    chrono.Clear();
    chrono.Start();
    int * nparts_arr = new int[num_levels-1];
    agg_dof_status_t *bdr_dofs = fem_find_bdr_dofs(*fes, &ess_bdr);

    nparts_arr[0] = pmesh->GetNE() / first_elems_per_agg;
    const bool do_aggregates = false;
    agg_part_rels = fem_create_partitioning(*Ag, *fes, bdr_dofs, nparts_arr,
                                            do_aggregates);
    for(int i=1; i < num_levels-1; ++i)
    {
        nparts_arr[i] = (int) round((double) nparts_arr[i-1] /
                                    (double) elems_per_agg);
        if (nparts_arr[i] < 1) nparts_arr[i] = 1;
    }
    delete [] bdr_dofs;

    if (visualize)
    {
        fem_parallel_visualize_partitioning(*pmesh, agg_part_rels->partitioning,
                                            nparts_arr[0]);
    }
    ElementMatrixProvider * emp = new ElementMatrixStandardGeometric(
        *agg_part_rels, Al, a);
    const bool use_arpack = false;
    int polynomial_coarse_space;
    if (minimal_coarse)
    {
        polynomial_coarse_space = 0;
        first_theta = 0.0;
        theta = 0.0;
    }
    else
    {
        polynomial_coarse_space = -1;
    }
    MultilevelParameters mlp(
        num_levels-1, nparts_arr, first_nu_pro, nu_pro, nu_relax, first_theta,
        theta, polynomial_coarse_space, correct_nullspace, use_arpack,
        do_aggregates);
    ml_data = ml_produce_data(*Ag, agg_part_rels, emp, mlp);

    chrono.Stop();
    SA_RPRINTF(0,"TIMING: multilevel spectral SA-AMGe setup %f seconds.\n", chrono.RealTime());
    timings["amg-setup"] = picojson::value(chrono.RealTime());

    // build coarse problem
    HypreParMatrix *Adifficult = ml_data->levels_list.finest->tg_data->Ac;
    HypreParVector bdifficult(*Adifficult);

    HypreParVector xdifficult(*Adifficult);
    // std::cout << "Afine is " << Ag->M() << " by " << Ag->N() << std::endl;
    // std::cout << "Acoarse is " << Adifficult->M() << " by " << Adifficult->N() << std::endl;
    tryjson["coarse-dofs"] = picojson::value((double) Adifficult->M());
    tryjson["operator-complexity"] = 
        picojson::value(ml_compute_OC_from_level(*Adifficult, *ml_data, ml_data->levels_list.finest->coarser));
    Array<int> num_dofs;
    picojson::array num_dofs_p;
    ml_get_dims(*ml_data, num_dofs);
    for (int i=0; i<num_dofs.Size(); ++i)
        num_dofs_p.push_back(picojson::value((double) num_dofs[i]));
    tryjson["dof-list"] = picojson::value(num_dofs_p);

    // solve with straight up Hypre
    {
        SA_RPRINTF(0,"%s","\n---\nSOLVING WITH STRAIGHT UP HYPRE:\n---\n\n");
        chrono.Clear();
        chrono.Start();
        HypreBoomerAMG hbamg(*Adifficult);
        bdifficult = 0.0;
        hbamg.SetPrintLevel(0);
        xdifficult.Randomize(0);
        CGSolver pcg(MPI_COMM_WORLD);
        pcg.SetOperator(*Adifficult);
        pcg.SetPreconditioner(hbamg);
        pcg.SetRelTol(1e-6); // for some reason MFEM squares this...
        pcg.SetMaxIter(1000);
        pcg.SetPrintLevel(1);
        pcg.Mult(bdifficult, xdifficult);
        int hypreit = pcg.GetNumIterations();
        int hypreconverged = pcg.GetConverged();
        chrono.Stop();
        if (hypreconverged)
        {
            SA_RPRINTF(0,"Hypre BoomerAMG PCG converged in %d iterations.\n",hypreit);
            tryjson["hypre-its"] = picojson::value((double) hypreit);
        }
        else
            SA_RPRINTF(0,"Hypre BoomerAMG PCG DIVERGED after %d iterations.\n",hypreit);
        SA_RPRINTF(0,"TIMING: Hypre BoomerAMG solution of coarse problem %f seconds.\n", chrono.RealTime());
        timings["hypre-solve"] = picojson::value(chrono.RealTime());
    }

    // solve with our hierarchy
    {
        if (num_levels == 2)
            SA_RPRINTF(0,"%s","\n---\nSOLVING WITH CORRECTED NULLSPACE:\n---\n\n");
        else
            SA_RPRINTF(0,"%s","\n---\nSOLVING WITH HIERARCHY:\n---\n\n");
        chrono.Clear();
        chrono.Start();
        bdifficult = 0.0;
        xdifficult.Randomize(0);
        Solver * Bprec;
        if (num_levels == 2)
        {
            levels_level_t * level = levels_list_get_level(ml_data->levels_list, 0);
            SA_ASSERT(level);
            Bprec = level->tg_data->coarse_solver;
        }
        else
        {
            levels_level_t * level = levels_list_get_level(ml_data->levels_list, 1);
            SA_ASSERT(level);
            Bprec = new VCycleSolver(level->tg_data, false);
        }
        Bprec->SetOperator(*Adifficult);

        CGSolver pcg(MPI_COMM_WORLD);
        pcg.SetOperator(*Adifficult);
        pcg.SetPreconditioner(*Bprec);
        pcg.SetRelTol(1e-6); // for some reason MFEM squares this...
        pcg.SetMaxIter(1000);
        pcg.SetPrintLevel(1);
        pcg.Mult(bdifficult,xdifficult);
        int myiterations = pcg.GetNumIterations();
        int myconverged = pcg.GetConverged();
        if (num_levels != 2)
            delete Bprec;
        if (myconverged)
        {
            SA_RPRINTF(0, "Outer PCG converged in %d iterations.\n", myiterations);
            tryjson["hierarchy-its"] = picojson::value((double) myiterations);
        }
        else
            SA_RPRINTF(0, "Outer PCG FAILED to converge after %d iterations!\n", myiterations);
        chrono.Stop();
        SA_RPRINTF(0,"TIMING: solve with SA-AMGe preconditioned CG %f seconds.\n", chrono.RealTime());
        timings["hierarchy-solve"] = picojson::value(chrono.RealTime());
    }

    ml_free_data(ml_data);
    agg_free_partitioning(agg_part_rels);

    tryjson["timings"] = picojson::value(timings);
    if (PROC_RANK == 0)
    {
        std::cout << picojson::value(tryjson).serialize();
        std::cout << std::endl;
    }

    delete pxg;
    delete hxg;
    // delete bg;
    delete Ag;
    delete a;
    delete b;

    delete fes;
    delete fec;
    delete matrix_conductivity;

    delete [] nparts_arr;
    delete pmesh;
    delete mesh;

    MPI_Finalize();

    return 0;
}
