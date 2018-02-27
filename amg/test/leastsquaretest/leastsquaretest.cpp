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

// Wrapping mesh based saamge and applying it to a mixed 2nd order pde
// Patrick V. Zulian
// zulian1@llnl.gov
// 10 August 2016

#include "mfem.hpp"
#include <fstream>
#include <iostream>

#include "SecondOrderEllipticIntegrator.hpp"
#include "saamgepc.hpp"
#include "LSHelmholtzProblem.hpp"

using namespace mfem;
using namespace saamge;
using std::shared_ptr;
using std::make_shared;

void make_block_system(
    const std::shared_ptr<mfem::HypreParMatrix> &B11, 
    const std::shared_ptr<mfem::HypreParMatrix> &B12,
    const std::shared_ptr<mfem::HypreParMatrix> &B21,
    const std::shared_ptr<mfem::HypreParMatrix> &B22,
    std::shared_ptr<mfem::BlockOperator> &block_op,
    std::shared_ptr<mfem::BlockVector> &solution,
    std::shared_ptr<mfem::BlockVector> &rhs,
    mfem::Array<int> &offsets)
{
    using namespace mfem;

    offsets.SetSize(3);
    offsets[0] = 0;
    offsets[1] = B11->Height();
    offsets[2] = B22->Height();
    offsets.PartialSum();

    block_op = std::make_shared<BlockOperator>(offsets);
    block_op->SetBlock(0, 0, B11.get());
    block_op->SetBlock(0, 1, B12.get());
    block_op->SetBlock(1, 0, B21.get());
    block_op->SetBlock(1, 1, B22.get());

    solution = std::make_shared<BlockVector>(offsets);
    rhs      = std::make_shared<BlockVector>(offsets);

    (*solution) = 0.0;
    (*rhs)      = 0.0;
}

template<class Operator, class Preconditioner, class RSH, class Solution>
void solve_system(Operator &op, Preconditioner &preconditioner, RSH &rhs,
                  Solution &solution, const int verbosity=-1)
{
    using namespace std;
    using namespace mfem;

    // auto pcg = make_shared<GMRESSolver>(MPI_COMM_WORLD);
    // auto pcg = make_shared<MINRESSolver>(MPI_COMM_WORLD);
    auto pcg = make_shared<CGSolver>(MPI_COMM_WORLD);

    pcg->SetOperator(op);
    pcg->SetMaxIter(1000);
    pcg->SetPrintLevel(verbosity);
    pcg->SetAbsTol (1e-10);
    pcg->SetPreconditioner(preconditioner);
    pcg->Mult(rhs, solution);

    int iterations = pcg->GetNumIterations();
    int converged  = pcg->GetConverged();
    double final_norm = pcg->GetFinalNorm();

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0)
    {
        if (converged)
        {
            printf("n_iterations: %d\n", iterations);
        }
        else
        {
            printf("Outer PCG failed to converge after %d iterations!\n", iterations);
        }

        printf("final_norm: %g\n", final_norm);
    }
}

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
        int order = 1;
        int order_grad = 1;
        bool visualization = 1;
        bool use_system = true;
        double k_local = 1.0;
        bool use_saamge_solver = true;
        double beta_val = 1.0;
        int n_refs = 1;

        OptionsParser args(argc, argv);
        args.AddOption(&mesh_file, "-m", "--mesh",
                       "Mesh file to use.");
        args.AddOption(&order, "-o", "--order",
                       "Finite element order (polynomial degree).");
        args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                       "--no-visualization",
                       "Enable or disable GLVis visualization.");
        args.AddOption(&k_local, "-k", "--k", "the k-coefficient");
        args.AddOption(&use_system, "-v", "--v", "-nv", "--nv",
                       "Use system AMG for gradient");
        args.AddOption(&use_saamge_solver, "-s", "--saamge", "-ns", "--no-saamge",
                       "use saamge solver");
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

        Mesh * mesh = new Mesh(mesh_file, 1, 1);;
        // Mesh *mesh = new Mesh(2, 2, 2, Element::HEXAHEDRON, 1);
        int dim = mesh->Dimension();
        int ref_levels = (int)floor(log(10000./mesh->GetNE())/log(2.)/dim);
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
        LSHelmholtzProblem hp;
        hp.Init(U_space, W_space, k_local, beta_val);

        ////////////////////////////////////////////////////////////////
        // Block system
        ////////////////////////////////////////////////////////////////

        Array<int> block_offsets;
        shared_ptr<BlockOperator> block_op;
        shared_ptr<BlockVector>   trueX;  
        shared_ptr<BlockVector>   trueRhs;

        ////////////////////////////////////////////////////////////////

        MPI_Barrier(MPI_COMM_WORLD);
        chrono.Start();

        if (use_saamge_solver)
        {
            auto prec_u = make_shared<SAAMGePC>(U_space, hp.ess_bdr);
            prec_u->Make(hp.u_bf, hp.M, *hp.M_spMat);

            Array<int> ess_bdr_vec;
            ess_bdr_vec.SetSize(pmesh->bdr_attributes.Size());
            ess_bdr_vec = 0;

            auto prec_gradu = make_shared<SAAMGePC>(W_space, ess_bdr_vec);
            prec_gradu->Make(hp.gradu_bf, hp.G, *hp.G_spMat);

            //To be done after Make due to side effects on the partitioning
            make_block_system(hp.M, hp.BT, hp.B, hp.G, block_op, trueX, trueRhs,
                              block_offsets);

            hp.f_form->ParallelAssemble(trueRhs->GetBlock(0));
            hp.f_dot_diff_form->ParallelAssemble(trueRhs->GetBlock(1));

            auto preconditioner =
                make_shared<BlockDiagonalPreconditioner>(block_offsets);
            preconditioner->SetDiagonalBlock(0, prec_u.get());
            preconditioner->SetDiagonalBlock(1, prec_gradu.get());

            solve_system(*block_op, *preconditioner, *trueRhs, *trueX, 0);
        }
        else
        {
            make_block_system(hp.M, hp.BT, hp.B, hp.G, block_op, trueX, trueRhs,
                              block_offsets);

            hp.f_form->ParallelAssemble(trueRhs->GetBlock(0));
            hp.f_dot_diff_form->ParallelAssemble(trueRhs->GetBlock(1));

            auto prec_u     = make_shared<HypreBoomerAMG>(*hp.M);
            auto prec_gradu = make_shared<HypreBoomerAMG>(*hp.G);
      
            if (use_system)
            {
                prec_gradu->SetSystemsOptions(dim);
            }

            auto preconditioner =
                make_shared<BlockDiagonalPreconditioner>(block_offsets);
            preconditioner->SetDiagonalBlock(0, prec_u.get());
            preconditioner->SetDiagonalBlock(1, prec_gradu.get());

            solve_system(*block_op, *preconditioner, *trueRhs, *trueX);
        }
        MPI_Barrier(MPI_COMM_WORLD);

        chrono.Stop();
        if (myid == 0)
        {
            std::cout << "time: " << chrono.RealTime() << std::endl;
            std::cout << "csv_data:" << dim  << "," << k_local << ","
                      << ref_levels << "," << (hp.M->Height() + hp.G->Height())
                      << std::endl;
        }
        MPI_Barrier(MPI_COMM_WORLD);

        ///////////////////////////////////////////////////////////////
        ///////////////////////////////////////////////////////////////

        auto u     = make_shared<ParGridFunction>(U_space.get());
        auto gradu = make_shared<ParGridFunction>(W_space.get());

        u->Distribute(&(trueX->GetBlock(0)));
        gradu->Distribute(&(trueX->GetBlock(1)));

        plot_scalar(*pmesh, *u);
        plot_vector(*pmesh, *gradu);
    }

    MPI_Finalize();
    return 0;
}
