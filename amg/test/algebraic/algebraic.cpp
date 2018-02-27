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
   Trying to apply SAAMGe algebraically
   to an arbitrary matrix.

   Using "diagonal compensation" or
   "window AMG."

   TODO:
   * make it work with nu_pro not equal to zero
   * implement multilevel
   * implement correct nulspace
   * make it work in parallel (non-trivial)

   Andrew T. Barker
   atb@llnl.gov
   16 September 2015
*/

#include <mfem.hpp>
#include <mpi.h>
#include <saamge.hpp>

using namespace mfem;
using namespace saamge;

/**
   HypreParMatrix has a .Read() method, but I cannot get it
   to work no matter how hard I try.

   I am reading into a SparseMatrix* here because I want to,
   but it is not hard to get a HypreParMatrix if you want.
*/
SparseMatrix * ReadHypreMat(const char * filename)
{
    std::ifstream in(filename);
    SA_ASSERT(in.good());

    int row0, row1, col0, col1;
    in >> row0 >> row1 >> col0 >> col1;
    SA_ASSERT(row0 == 0);
    SA_ASSERT(col0 == 0);
    // std::cout << row0 << row1 << col0 << col1 << std::endl;
    SparseMatrix * out = new SparseMatrix(row1+1,col1+1);
    while (in)
    {
        int i;
        int j;
        double x;
        in >> i >> j >> x;
        out->Add(i,j,x);
    }
    out->Finalize();
    // out->Print(std::cout);
    return out;
}

int main(int argc, char *argv[])
{
    // Initialize process related stuff.
    MPI_Init(&argc, &argv);
    proc_init(MPI_COMM_WORLD);

    StopWatch chrono;
    chrono.Clear();
    chrono.Start();

    agg_partitioning_relations_t *agg_part_rels;
    // ml_data_t *ml_data;
    tg_data_t *tg_data;

    const char *matrix_file = "anisotropic.mat.00000";

    int first_nu_pro = -1;
    int nu_pro = 0;
    int nu_relax = 3;
    double theta = 0.003;
    double first_theta = -1.0;
    int num_levels = 2;
    int elems_per_agg = 256;
    int first_elems_per_agg = -1;

    bool zero_rhs = false;
    bool minimal_coarse = false;
    bool correct_nulspace = true;
    bool window_amg = false;

    OptionsParser args(argc, argv);
    args.AddOption(&matrix_file, "-m", "--matrix",
                   "Matrix file to read.");
    args.AddOption(
        &nu_pro, "-p", "--nu-pro",
        "Degree of the smoother for the smoothed aggregation for first coarsening.");
    args.AddOption(
        &first_nu_pro, "-fp", "--first-nu-pro",
        "Degree of smoother for smoothed aggregation on later coarsenings.");
    args.AddOption(&nu_relax, "-n", "--nu-relax",
                   "Degree for smoother in the relaxation.");
    args.AddOption(&theta, "-t", "--theta",
                   "Tolerance for eigenvalue problems.");
    args.AddOption(
        &first_theta, "-ft", "--first-theta",
        "Tolerance for eigenvalue problems for first (finest) coarsening.");
    /*
    args.AddOption(&num_levels, "-l", "--num-levels",
                   "Number of levels in multilevel algorithm.");
    */
    args.AddOption(&elems_per_agg, "-e", "--elems-per-agg",
                   "Number of elements per agglomerated element.");
    args.AddOption(&first_elems_per_agg, "-fe", "--first-elems-per-agg",
                   "Number of elements per AE for first (finest) coarsening.");
    args.AddOption(&zero_rhs, "-z", "--zero-rhs",
                   "-nz", "--no-zero-rhs",
                   "Solve CG with zero RHS and random initial guess.");
    args.AddOption(&minimal_coarse, "-mc", "--minimal-coarse",
                   "-nmc", "--no-minimal-coarse",
                   "Minimal coarse space, ie, vector of all ones.");
    args.AddOption(&correct_nulspace, "-c", "--correct-nulspace",
                   "-nc", "--no-correct-nulspace",
                   "Use the corrected nulspace technique on coarsest level.");
    args.AddOption(
        &window_amg, "-w", "--window-amg",
        "-nw", "--no-window-amg",
        "Use Henson-Vassilevski window based AMG for constructing local matrices.");

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
    if (first_elems_per_agg < 0) first_elems_per_agg = elems_per_agg;
    if (first_theta < 0.0) first_theta = theta;
    if (first_nu_pro < 0) first_nu_pro = nu_pro;

    SA_ASSERT(!(correct_nulspace && minimal_coarse));

    MPI_Barrier(PROC_COMM); // try to make MFEM's debug element orientation prints not mess up the parameters above

    int nprocs = PROC_NUM;
    SA_ASSERT(nprocs == 1); // this algebraic stuff not yet implemented in parallel, should be at some point

    SparseMatrix * mat = ReadHypreMat(matrix_file);
    int row_starts[2];
    row_starts[0] = 0;
    row_starts[1] = mat->Height();
    HypreParMatrix Ag(PROC_COMM, mat->Height(), row_starts, mat);

    HypreParVector bg(Ag);
    bg = 1.0;
    HypreParVector xg(Ag);
    xg = 1.0;
    HypreParVector hxg(Ag);

    chrono.Stop();
    SA_RPRINTF(0, "TIMING: matrix load and setup %f seconds.\n", chrono.RealTime());

    // basic solver stuff.
    chrono.Clear();
    chrono.Start();
    SA_RPRINTF(0, "%s", "\n");
    SA_RPRINTF(0, "%s", "\t\t\tSOLVING ORIGINAL FINE SCALE PROBLEM USING HYPRE BOOMERAMG:\n");
    SA_RPRINTF(0, "%s", "\n");
    HypreBoomerAMG *hbamg = new HypreBoomerAMG(Ag);
    hbamg->SetPrintLevel(0);

    CGSolver * pcg = new CGSolver(MPI_COMM_WORLD);
    pcg->SetOperator(Ag);
    pcg->SetRelTol(1e-6); // for some reason MFEM squares this...
    pcg->SetMaxIter(1000);
    pcg->SetPrintLevel(1);
    pcg->SetPreconditioner(*hbamg);

    if (zero_rhs)
    {
        hxg.Randomize(0);
        bg = 0.0;
    }
    pcg->Mult(bg, hxg);
    delete pcg;
    delete hbamg;
    xg = hxg;
    SA_RPRINTF(0, "TIMING: setup and solve with Hypre BoomerAMG "
               "preconditioned CG %f seconds.\n", chrono.RealTime());

    // some actual AMGe stuff
    chrono.Clear();
    chrono.Start();
    int * nparts_arr = new int[num_levels-1];

    bool df0eliminated = true;
    SparseMatrix * Al;
    if (df0eliminated)
    {
        Al = new SparseMatrix(mat->Height()-1, mat->Width()-1);
        int * I = mat->GetI();
        int * J = mat->GetJ();
        double * data = mat->GetData();
        for (int i=1; i<mat->Height(); ++i)
        {
            for (int j=I[i]; j<I[i+1]; ++j)
                Al->Add(i-1,J[j]-1,data[j]);
        }
        Al->Finalize();
        // delete mat; // would like to, but Ag is using its data
    } 
    else
    {
        Al = mat;
    }

    nparts_arr[0] = Al->Height() / first_elems_per_agg;
    SparseMatrix * identity = IdentitySparseMatrix(Al->Height());
    int drow_starts[2];
    drow_starts[0] = 0;
    drow_starts[1] = Al->Height();
    HypreParMatrix * dof_truedof = new HypreParMatrix(
        PROC_COMM, identity->Height(), drow_starts, identity);
    Array<int> isolated_cells(0);

    if (window_amg)
        TestWindowSubMatrices();

    agg_part_rels = fem_create_partitioning_from_matrix(
        *Al, nparts_arr, dof_truedof, isolated_cells);
    for (int i=1; i < num_levels-1; ++i)
    {
        nparts_arr[i] = (int) round((double) nparts_arr[i-1] / (double) elems_per_agg);
        if (nparts_arr[i] < 1) nparts_arr[i] = 1;
    }

    // std::cout << "Ag is " << Ag.Height() << " by " << Ag.Width() 
    //           << " and has " << Ag.NNZ() << " nonzeros." << std::endl;
    const bool use_arpack = true;
    const bool avoid_ess_bdr_dofs = true;
    int polynomial_coarse;
    if (minimal_coarse)
        polynomial_coarse = 0;
    else
        polynomial_coarse = -1;
    tg_data = tg_produce_data_algebraic(
        *Al, Ag, *agg_part_rels, first_nu_pro, nu_relax, first_theta,
        (nu_pro > 0), polynomial_coarse, window_amg, use_arpack, avoid_ess_bdr_dofs);

    if (df0eliminated)
    {
        tg_augment_interp_with_identity(*tg_data, 1);
    }
    tg_fillin_coarse_operator(Ag, tg_data, false);
    tg_data->coarse_solver = new AMGSolver(*tg_data->Ac, false);

    chrono.Stop();
    SA_RPRINTF(0,"TIMING: multilevel spectral SA-AMGe setup %f seconds.\n", chrono.RealTime());

    if (zero_rhs)
    {
        SA_RPRINTF(0, "%s", "\n");
        SA_RPRINTF(0, "%s", "\t\t\tRUNNING PCG WITH RANDOM INITIAL GUESS AND ZERO"
                   " R.H.S:\n");
        SA_RPRINTF(0, "%s", "\n");
    }
    else
    {
        SA_RPRINTF(0, "%s", "\n");
        SA_RPRINTF(0, "%s", "\t\t\tSOLVING THE PROBLEM USING PCG:\n");
        SA_RPRINTF(0, "%s", "\n");      
    }

    chrono.Clear();
    chrono.Start();
    HypreParVector pxg(Ag);
    if (zero_rhs)
    {
        // helpers_random_vect(*agg_part_rels, *pxg);
        pxg.Randomize(0);
        bg = 0.0;
    }
    int iterations = -1;
    int converged = -1;
    Solver * Bprec;

    // levels_level_t * level = levels_list_get_level(ml_data->levels_list, 0);
    Bprec = new VCycleSolver(tg_data, false);
    Bprec->SetOperator(Ag);

    CGSolver hpcg(MPI_COMM_WORLD);
    hpcg.SetOperator(Ag);
    hpcg.SetRelTol(1e-6); // for some reason MFEM squares this...
    hpcg.SetMaxIter(1000);
    hpcg.SetPrintLevel(1);
    hpcg.SetPreconditioner(*Bprec);
    hpcg.Mult(bg,pxg);
    iterations = hpcg.GetNumIterations();
    converged = hpcg.GetConverged();
    delete Bprec;
    if (converged)
        SA_RPRINTF(0, "Outer PCG converged in %d iterations.\n", iterations);
    else
        SA_RPRINTF(0, "Outer PCG failed to converge after %d iterations!\n", iterations);
    chrono.Stop();
    SA_RPRINTF(0,"TIMING: solve with SA-AMGe preconditioned CG %f seconds.\n", chrono.RealTime());

    // ml_free_data(ml_data);
    tg_free_data(tg_data);
    agg_free_partitioning(agg_part_rels);

    if (df0eliminated)
        delete mat;
    delete Al;
    delete [] nparts_arr;
    delete identity;
    delete dof_truedof;

    MPI_Finalize();
    return 0;
}
