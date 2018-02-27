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

// Wrapping algebraic saamge and applying it to a mixed 2nd order pde
// Patrick V. Zulian
// zulian1@llnl.gov
// 10 August 2016

#include "mfem.hpp"
#include <fstream>
#include <iostream>

#include "SecondOrderEllipticIntegrator.hpp"
#include "saamgealgpc.hpp"
#include "LSHelmholtzProblem.hpp"

using namespace mfem;
using namespace saamge;
using std::shared_ptr;
using std::make_shared;

void plot_scalar(ParMesh &mesh, ParGridFunction &x)
{
    int num_procs, myid;
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    char vishost[] = "localhost";
    int  visport   = 19916;
    socketstream sol_sock(vishost, visport);
    sol_sock << "parallel " << num_procs << " " << myid << "\n";
    sol_sock.precision(8);
    sol_sock << "solution\n" << mesh << x << std::flush;
    sol_sock << std::flush;
}

void plot_vector(ParMesh &mesh, ParGridFunction &x)
{
    int num_procs, myid;
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    char vishost[] = "localhost";
    int  visport   = 19916;
    socketstream sol_sock(vishost, visport);
    sol_sock << "parallel " << num_procs << " " << myid << "\n";
    sol_sock.precision(8);
    sol_sock << "vector-solution\n" << mesh << x << std::flush;
    sol_sock << std::flush;
}

/*!
 * To launch it with saamge solver:     ./leastsquarealgebraictest          [-k 5]
 * To launch it with BoomerAMG solver:  ./leastsquarealgebraictest -l -ns   [-k 5]
 */
int main(int argc, char *argv[])
{
    {
        StopWatch chrono;

        int num_procs, myid;
        MPI_Init(&argc, &argv);
        MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
        MPI_Comm_rank(MPI_COMM_WORLD, &myid);
        bool verbose = (myid == 0);

        const char *mesh_file = "../../test/mltest.mesh";
        int order = 2;
        int order_grad = 2;
        int n_refs = 2;
        bool visualization = 1;
        bool use_system = true;
        double k_local = 0.1;
        bool use_saamge_solver = true;
        double beta_val = 0.99;
        bool use_local_blocks = false;

        OptionsParser args(argc, argv);
        args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
        args.AddOption(&order, "-o", "--order",
                       "Finite element order (polynomial degree).");
        args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                       "--no-visualization",
                       "Enable or disable GLVis visualization.");
        args.AddOption(&k_local, "-k", "--k", "the k-coefficient");
        args.AddOption(&use_system, "-v", "--v", "-nv", "--nv",
                       "Use system AMG for gradient");
        args.AddOption(&use_saamge_solver, "-s", "--saamge", "-ns",
                       "--no-saamge", "use saamge solver");
        args.AddOption(&use_local_blocks, "-l", "--localblocks", "-nl",
                       "--no-localblocks", "use local dxd blocks");
        args.AddOption(&beta_val, "-b", "--beta",
                       "pentaly parameter for  beta . (curl(sigma) . curl(q))");
        args.AddOption(&n_refs, "-r", "--n_refs", "number of refinements");
        args.Parse();
        if (!args.Good())
        {
            if (verbose)
            {
                args.PrintUsage(cout);
            }
            MPI_Finalize();
            return 1;
        }
        if (verbose)
        {
            args.PrintOptions(cout);
        }

        Mesh * mesh;
        std::ifstream imesh(mesh_file);
        if (imesh)
            mesh = new Mesh(imesh, 1, 1);
        else
            mesh = new Mesh(2, 2, Element::QUADRILATERAL, 1);

        int dim = mesh->Dimension();
        int ref_levels = (int) floor(log(10000./mesh->GetNE())/log(2.)/dim);
        {
            if (n_refs >= 0)
            {
                ref_levels = n_refs;
            }
            for (int l = 0; l < ref_levels; l++)
            {
                mesh->UniformRefinement();
            }
        }
        auto pmesh = make_shared<ParMesh>(MPI_COMM_WORLD, *mesh);
        delete mesh;

        auto u_coll     = make_shared<H1_FECollection>(order, dim);
        auto gradu_coll = make_shared<H1_FECollection>(order_grad, dim);

        auto U_space    = make_shared<ParFiniteElementSpace>(
            pmesh.get(), u_coll.get(), 1, Ordering::byVDIM);
        auto W_space    = make_shared<ParFiniteElementSpace>(
            pmesh.get(), gradu_coll.get(), dim, Ordering::byVDIM);

        // set-up least squares system forms and matrices
        bool eliminate_bc_dofs = use_saamge_solver || !use_local_blocks; 
        std::cout << "eliminate_bc_dofs: "
                  << (eliminate_bc_dofs ? "true" : "false")
                  << std::endl;
        LSHelmholtzProblem hp(eliminate_bc_dofs);
        hp.Init(U_space, W_space, k_local, beta_val);

        shared_ptr<HypreParMatrix> mnlthc_mat;
        shared_ptr<Vector>         mnlthc_rhs;
        shared_ptr<SparseMatrix>   mnlthc_mat_l;

        if (use_local_blocks)
        {
            hp.LocalBlocksMakeMonolithic(mnlthc_mat, mnlthc_rhs, mnlthc_mat_l);
        }
        else
        {
            hp.MakeMonolithic(mnlthc_mat, mnlthc_rhs, mnlthc_mat_l);
        }

        // mnlthc_mat->Print("lb_mat.txt");

        Vector X(mnlthc_rhs->Size());
        X = 0.0;

        chrono.Start();
        ///////////////////////////////////////////////////////////////
        ///////////////////////////////////////////////////////////////

        shared_ptr<Solver> pc;
        if (!use_saamge_solver)
        {
            auto hbamg = make_shared<HypreBoomerAMG>(*mnlthc_mat);
            if (use_local_blocks)
            {
                hbamg->SetSystemsOptions(dim + 1);
            }
            pc = hbamg;
        }
        else
        {
            auto saamgealgpc = make_shared<SAAMGeAlgPC>();
            saamgealgpc->Make(mnlthc_mat, *mnlthc_mat_l);
            pc = saamgealgpc;
        }

        CGSolver hpcg(MPI_COMM_WORLD);
        // GMRESSolver hpcg(MPI_COMM_WORLD);
        // MINRESSolver hpcg(MPI_COMM_WORLD);

        hpcg.SetOperator(*mnlthc_mat);
        // hpcg.SetRelTol(1e-6); 
        hpcg.SetAbsTol(1e-10); 
        hpcg.SetMaxIter(600);
        hpcg.SetPrintLevel(1);
        hpcg.SetPreconditioner(*pc);

        hpcg.Mult(*mnlthc_rhs, X);

        chrono.Stop();

        //dims,k,n_refs,dofs,n_iterations,final_norm,time,convergence_rate
        std::cout << "csv_data:";    
        std::cout << dim << "," << k_local << "," << ref_levels << ",";
        std::cout << mnlthc_rhs->Size() << "," << hpcg.GetNumIterations() << ",";
        std::cout << hpcg.GetFinalNorm() << "," << chrono.RealTime() << std::endl;

        ///////////////////////////////////////////////////////////////
        ///////////////////////////////////////////////////////////////

        auto u     = make_shared<ParGridFunction>(U_space.get());
        auto gradu = make_shared<ParGridFunction>(W_space.get());

        if (use_local_blocks)
        {
            hp.LocalBlocksRecoverSolution(X, *u, *gradu);
        }
        else
        {
            hp.RecoverSolution(X, *u, *gradu);
        }

        plot_scalar(*pmesh, *u);
        plot_vector(*pmesh, *gradu);
   }

   MPI_Finalize();

   return 0;
}
