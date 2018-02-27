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

// Wrapping geometric saamge and applying it to a 2nd order pde
// Patrick V. Zulian
// zulian1@llnl.gov
// 10 August 2016

// ATB: I am not sure of the purpose of this test example, just to show use
// of SAAMGePC object?

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <memory>
#include <assert.h>

#include <mpi.h>
#include <saamge.hpp>

#include "saamgepc.hpp"
#include "AnisotropicDiffusionIntegrator.hpp"

using std::shared_ptr;
using std::make_shared;
using namespace mfem;
using namespace saamge;

double rhs_fun(const Vector &v) { return 1.0; }//sqrt(v(0) * v(0) + v(1) * v(1)); }

Vector make_b_local(const int dim, const bool x_prefer)
{
    Vector b_local(dim);
    b_local = 1.0;

    for (int i = 0; i < dim; ++i)
    {
        if (x_prefer)
        {
            b_local(dim - 1 - i) = i * 10;
        }
        else
        {
            b_local(i) = i * 10;
        }
    }
    return b_local;
}

int main(int argc, char *argv[])
{
    {
        // 1. Initialize MPI.
        int num_procs, myid;
        MPI_Init(&argc, &argv);
        MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
        MPI_Comm_rank(MPI_COMM_WORLD, &myid);
                
        const char *mesh_file = "../../test/mltest.mesh";

        bool aniso = true;
        bool x_prefer = true;
        double k_squared = 200;

        OptionsParser args(argc, argv);
        args.AddOption(&mesh_file, "-m", "--mesh",  "Mesh file to use.");
        args.AddOption(&aniso, "-a", "--aniso", "-na", "--no-aniso",
                       "enable/disable anisotropic example (true by default)");
        args.AddOption(&x_prefer, "-x", "--x", "-y", "--y", "preferential direction");
        args.AddOption(&k_squared, "-k", "--k", "k value in div(grad(u)) + k * u");
        args.Parse();
                
        if (!args.Good())
        {
            if (myid == 0)
            {
                args.PrintUsage(std::cout);
            }
            MPI_Finalize();
            return 1;
        }

        if (myid == 0)
        {
            args.PrintOptions(std::cout);
        }

        ifstream imesh(mesh_file);
                
        auto mesh = make_shared<Mesh>(imesh, 1, 1);
        imesh.close();

        int dim = mesh->Dimension();

        {
            int ref_levels = (int)floor(log(10000./mesh->GetNE())/log(2.)/dim);
            for (int l = 0; l < ref_levels; l++)
            {
                mesh->UniformRefinement();
            }
        }
                
        auto pmesh = make_shared<ParMesh>(MPI_COMM_WORLD, *mesh);
                
        fem_refine_mesh_times(1, *pmesh);

        auto fec = make_shared<H1_FECollection>(1, dim);
        auto fespace = make_shared<ParFiniteElementSpace>(pmesh.get(), fec.get());
        HYPRE_Int size = fespace->GlobalTrueVSize();

        if (myid == 0)
        {
            std::cout << "Number of finite element unknowns: " << size << std::endl;
        }

        // boundary
        Array<int> ess_bdr(pmesh->bdr_attributes.Max());
        ess_bdr = 1;

        auto b = make_shared<ParLinearForm>(fespace.get());
        FunctionCoefficient f(rhs_fun);         
        b->AddDomainIntegrator(new DomainLFIntegrator(f));
        b->Assemble();

        VectorConstantCoefficient b_global(make_b_local(dim, x_prefer));
        auto a = make_shared<ParBilinearForm>(fespace.get());
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

        //////////////////////////////////////////////////////////////////////
        //////////////////////////////////////////////////////////////////////

        ParGridFunction x(fespace.get()); 
        x = 0.0;

        a->Assemble();   
        a->EliminateEssentialBC(ess_bdr, x, *b);     
        a->Finalize();

        shared_ptr<HypreParVector> B(b->ParallelAssemble());
                
        /// SAAMGE begin
        auto saamge_prec = make_shared<SAAMGePC>(fespace, ess_bdr);
        auto &Al = a->SpMat();
        shared_ptr<HypreParMatrix> A(a->ParallelAssemble());
        saamge_prec->Make(a, A, Al);

        // !!! Important, the dofs are reorganized in saamge_prec->Make
        // By fem_create_partitioning
        Vector X(A->Width());
        X = 0.0;

        MPI_Barrier(MPI_COMM_WORLD);
        if(myid == 0) std::cout << "Solving....\n" << std::endl;

        for (int i = 0; i < num_procs; ++i)
        {
            if (i == myid)
            {
                std::cout << "[" << i << "]" << std::endl;
                std::cout << "A: " << A->Height() << ", " << A->Width() << std::endl;
                std::cout << "B: " << B->Size() << ", X: " << X.Size() << std::endl;
            }

            MPI_Barrier(MPI_COMM_WORLD);
        }

        CGSolver hpcg(MPI_COMM_WORLD);
        hpcg.SetOperator(*A);
        hpcg.SetRelTol(1e-6); 
        hpcg.SetMaxIter(1000);
        hpcg.SetPrintLevel(1);
        hpcg.SetPreconditioner(*saamge_prec);
        hpcg.Mult(*B, X);

        MPI_Barrier(MPI_COMM_WORLD);
        if (myid == 0)
            std::cout << "finished solving" << std::endl;

        //SAAMGE end

        int iterations = hpcg.GetNumIterations();
        int converged  = hpcg.GetConverged();

        SA_RPRINTF(0, "Outer PCG %s converged in %d iterations.\n",
                   (converged ? "" : "did NOT"), iterations);

        a->RecoverFEMSolution(X, *b, x);
                
        ////Visualization
        std::ostringstream mesh_name, sol_name;
        mesh_name << "mesh." << std::setfill('0') << std::setw(6) << myid;
        sol_name << "sol." << std::setfill('0') << std::setw(6) << myid;
                
        char vishost[] = "localhost";
        int  visport   = 19916;
        socketstream sol_sock(vishost, visport);
        sol_sock << "parallel " << num_procs << " " << myid << "\n";
        sol_sock.precision(8);
        sol_sock << "solution\n" << *pmesh << x << std::flush;
        sol_sock << std::flush;
    }

    return MPI_Finalize();
}
