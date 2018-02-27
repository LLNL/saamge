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

// Wrapping algebraic saamge and applying it to a 2nd order pde
// Patrick V. Zulian
// zulian1@llnl.gov
// 10 August 2016

// ATB: not sure we want this, just demonstrates SAAMGeAlgPC object

#include <memory>
#include <mfem.hpp>
#include <mpi.h>
#include <saamge.hpp>
#include <saamgealgpc.hpp>
#include "AnisotropicDiffusionIntegrator.hpp"

using namespace mfem;
using namespace saamge;
using std::shared_ptr;
using std::make_shared;

void plot(const Mesh &mesh, const GridFunction &x)
{
    std::ostringstream mesh_name, sol_name;
    mesh_name << "mesh." << std::setfill('0') << std::setw(6) << PROC_NUM;
    sol_name << "sol."   << std::setfill('0') << std::setw(6) << PROC_NUM;

    char vishost[] = "localhost";
    int  visport   = 19916;
    socketstream sol_sock(vishost, visport);
    sol_sock.precision(8);
    sol_sock << "solution\n" << mesh << x << std::flush;
    sol_sock << std::flush;
}

double rhs_fun(const Vector &) { return 1.0; }   

Vector make_b_local(const int dim, const bool x_prefer)
{
    Vector b_local(dim);
    b_local = .1;

    for (int i = 0; i < dim; ++i)
    {
        if (x_prefer)
        {
            b_local(dim - 1 - i) = i * 2; 
        }
        else
        {
            b_local(i) = i * 2;    
        }
    }
    return b_local;
}

int main(int argc, char *argv[])
{
    // Initialize process related stuff.
    MPI_Init(&argc, &argv);
    proc_init(MPI_COMM_WORLD);

    {
        StopWatch chrono;
        chrono.Clear();
        chrono.Start();

        const char *mesh_file = "../../../../workspace/stage/mfem/data/inline-tri.mesh";
        bool zero_rhs = false;
        bool x_prefer = 1;
        bool aniso = false;
        double k_squared = -200.0;

        OptionsParser args(argc, argv);
        args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to read.");

        args.Parse();
        if (!args.Good())
        {
            if (PROC_RANK == 0)
                args.PrintUsage(std::cout);
            MPI_Finalize();
            return 1;
        }
        if (PROC_RANK == 0)
            args.PrintOptions(std::cout);

        std::ifstream is(mesh_file);
        auto mesh = make_shared<Mesh>(is, 1, 1);
        is.close();

        for (int i = 0; i < 2; ++i)
        {
            mesh->UniformRefinement();
        }

        const int dim = mesh->Dimension();
        auto fec     = make_shared<H1_FECollection>(1, dim);
        auto fespace = make_shared<FiniteElementSpace>(mesh.get(), fec.get());

        auto b = make_shared<LinearForm>(fespace.get());
        FunctionCoefficient f(&rhs_fun);   
        b->AddDomainIntegrator(new DomainLFIntegrator(f));
        b->Assemble();

        VectorConstantCoefficient b_global(make_b_local(dim, x_prefer));
        auto a = make_shared<BilinearForm>(fespace.get());

        if (aniso)
        {
            std::cout << "using aniso" << std::endl;
            a->AddDomainIntegrator(new AnisotropicDiffusionIntegrator(b_global));
        }
        else
        {
            std::cout << "normal" << std::endl;
            a->AddDomainIntegrator(new AnisotropicDiffusionIntegrator());
        }

        ConstantCoefficient c(k_squared);
        a->AddDomainIntegrator(new MassIntegrator(c));
        a->Assemble();

        Array<int> ess_bdr(mesh->bdr_attributes.Max());
        ess_bdr = 1;
        Array<int> ess_tdof_list;
        fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

        GridFunction x(fespace.get());
        x = 0.0;
        x.Update();

        MPI_Barrier(PROC_COMM); 

        // this algebraic stuff not yet implemented in parallel, should be at some point
        int nprocs = PROC_NUM;
        SA_ASSERT(nprocs == 1);

        shared_ptr<SparseMatrix> mat = make_shared<SparseMatrix>(a->SpMat());

        Vector X, B;
        a->FormLinearSystem(ess_tdof_list, x, *b, *mat, X, B);

        chrono.Stop();
        SA_RPRINTF(0, "TIMING: matrix load and setup %f seconds.\n", chrono.RealTime());

        chrono.Clear();
        chrono.Start();

        SA_RPRINTF(0, "%s", "\n");
        SA_RPRINTF(0, "%s", "\t\t\tSOLVING ORIGINAL FINE SCALE PROBLEM USING HYPRE BOOMERAMG:\n");
        SA_RPRINTF(0, "%s", "\n");

        int row_starts[2];
        row_starts[0] = 0;
        row_starts[1] = mat->Height();

        auto Ag = make_shared<HypreParMatrix>(PROC_COMM, mat->Height(),
                                              row_starts, mat.get());
        auto hbamg = make_shared<HypreBoomerAMG>(*Ag);
        hbamg->SetPrintLevel(0);

        auto pcg = make_shared<CGSolver>(MPI_COMM_WORLD);
        pcg->SetOperator(*Ag);
        pcg->SetRelTol(1e-6); 
        pcg->SetMaxIter(1000);
        pcg->SetPrintLevel(1);
        pcg->SetPreconditioner(*hbamg);

        pcg->Mult(B, X);

        SA_RPRINTF(0, "TIMING: setup and solve with Hypre BoomerAMG "
           "preconditioned CG %f seconds.\n", chrono.RealTime());

        a->RecoverFEMSolution(X, *b, x);

        ////Visualization
        // plot(*mesh,  x);

        chrono.Clear();
        chrono.Start();
        auto saamgealgpc = make_shared<SAAMGeAlgPC>();
        saamgealgpc->Make(Ag, *mat);
        chrono.Stop();
        SA_RPRINTF(0,"TIMING: multilevel spectral SA-AMGe setup %f seconds.\n",
                   chrono.RealTime());

        if (zero_rhs)
        {
            SA_RPRINTF(0, "%s", "\n\t\t\tRUNNING PCG WITH RANDOM INITIAL GUESS AND ZERO"
             " R.H.S:\n\n");
        }
        else
        {
            SA_RPRINTF(0, "%s", "\n\t\t\tSOLVING THE PROBLEM USING PCG:\n\n");
        }

        chrono.Clear();
        chrono.Start();
        X = 0.0;
        CGSolver hpcg(MPI_COMM_WORLD);
        hpcg.SetOperator(*Ag);
        hpcg.SetRelTol(1e-6); 
        hpcg.SetMaxIter(1000);
        hpcg.SetPrintLevel(1);
        hpcg.SetPreconditioner(*saamgealgpc);
        hpcg.Mult(B, X);
        int iterations = hpcg.GetNumIterations();
        int converged  = hpcg.GetConverged();

        if (converged)
            SA_RPRINTF(0, "Outer PCG converged in %d iterations.\n", iterations);
        else
            SA_RPRINTF(0, "Outer PCG failed to converge after %d iterations!\n",
                       iterations);
        chrono.Stop();
        SA_RPRINTF(0,"TIMING: solve with SA-AMGe preconditioned CG %f seconds.\n", chrono.RealTime());

        a->RecoverFEMSolution(X, *b, x);
        plot(*mesh,  x);
    }

    MPI_Finalize();
    return 0;
}
