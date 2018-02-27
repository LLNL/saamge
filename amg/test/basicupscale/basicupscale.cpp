/**
   basic upscaling, trying to test this against the H1 spectral branch of parelag

   Andrew T. Barker
   barker29@llnl.gov
   16 December 2014
*/

/*! \file
    \brief TG driver with example how to modify the coarsest-level solver.

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

#include <mfem.hpp>
#include <mpi.h>
#include <saamge.hpp>

using namespace mfem;
using namespace saamge;

class Arguments
{
public:
    Arguments(int argc, char *argv[]);
    void unknown_arg(std::string & arg);

    bool visualize;
    bool checkerboard;
    std::string meshfile;
    int num_refine;
    int elems_per_agg; 
    int nu_pro;
    int nu_relax;
    double theta;
    std::string solver;
};

Arguments::Arguments(int argc, char *argv[])
{
    visualize = false;
    checkerboard = false;
    meshfile = "/home/barker29/meshes/cube474.mesh3d";
    num_refine = 1;
    elems_per_agg = 200; 
    nu_pro = 3;
    nu_relax = 3;
    theta = 0.003;
    solver = "both";

    int i = 1;
    while (i < argc)
    {
        std::string arg(argv[i]);
        if (arg == "--visualize")
        {
            visualize = true;
            i++;
        }
        else if (arg == "--checkerboard")
        {
            checkerboard = true;
            i++;
        }
        else if (arg == "--meshfile")
        {
            meshfile = argv[i+1];
            i += 2;
        }
        else if (arg == "--refine")
        {
            num_refine = (int) strtol(argv[i+1], NULL, 0);
            i += 2;
        }
        else if (arg == "--elems-per-agg")
        {
            elems_per_agg = (int) strtol(argv[i+1], NULL, 0);
            i += 2;
        }
        else if (arg == "--nu-pro")
        {
            nu_pro = (int) strtol(argv[i+1], NULL, 0);
            i += 2;
        }
        else if (arg == "--nu-relax")
        {
            nu_relax = (int) strtol(argv[i+1], NULL, 0);
            i += 2;
        }
        else if (arg == "--theta")
        {
            theta = strtod(argv[i+1], NULL);
            i += 2;
        }
        else if (arg == "--solver")
        {
            solver = argv[i+1];
            i += 2;
        }
        else
        {         
            unknown_arg(arg);
        }
    }
}

void Arguments::unknown_arg(std::string & arg)
{
    std::cout << "Unknown or incorrectly used argument " << arg << std::endl;
    throw 1;
}

// ----------------------------------------

/* Coefficient, R.H.S. and Boundary Condition Functions */

/*!
    A checkerboard coefficient for unit square or unit cube domain.
*/
double checkboard_coef(Vector& x)
{
    SA_ASSERT(2 <= x.Size() && x.Size() <= 3);
    double d = (double) 2;

    if ((x.Size() == 2 &&
         ((int)ceil(x(0)*d) & 1) == ((int)ceil(x(1)*d) & 1)) ||
        (x.Size() == 3 &&
         ((((int)ceil(x(2)*d) & 1) &&
           ((int)ceil(x(0)*d) & 1) == ((int)ceil(x(1)*d) & 1)) ||
          (!((int)ceil(x(2)*d) & 1) &&
          ((int)ceil(x(0)*d) & 1) != ((int)ceil(x(1)*d) & 1)))))
    {
        return 1e6;
    } else
        return 1e0;
}

/*!
    Right hand side (source term) of the equation.
*/
double rhs_func(Vector& x)
{
    SA_ASSERT(x.Size() <= 3);
    return 1;
}

/*!
    Dirichlet boundary conditions.
*/
double bdr_cond(Vector& x)
{
    SA_ASSERT(x.Size() <= 3);
    return 0.;
}

////////////////////////////////////////////////////////////////////////////

/*!
    Main entry point.
*/
int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    proc_init(MPI_COMM_WORLD);

/// =======================================================================
/// Setting up the problem.
/// =======================================================================

    Mesh *mesh = NULL;
    agg_partitioning_relations_t *agg_part_rels;
    ParGridFunction x;
    ParLinearForm *b;
    ParBilinearForm *a;
    tg_data_t *tg_data;

    Arguments args(argc,argv);
    mesh = fem_read_mesh(args.meshfile.c_str());

    /// pick the parts of the boundary with essential conditions.
    Array<int> ess_bdr(mesh->bdr_attributes.Max());
    ess_bdr = 1;
    // ess_bdr[0] = 0; ///< South in 2D unit square. (Dirchlet, Nuemann) = (1, 0)
    // ess_bdr[2] = 0; ///< North in 2D unit square. (Dirchlet, Nuemann) = (1, 0)
    ess_bdr[0] = 0;
    ess_bdr[5] = 0;

    /// refine the mesh.
    int nprocs = PROC_NUM;
    int * proc_partitioning = fem_partition_mesh(*mesh, &nprocs);

    // if (args.visualize && 0 == PROC_RANK)
    //   fem_serial_visualize_partitioning(*mesh, proc_partitioning);

    // XXX: The parallel mesh will own and deallocate the partitioning.
    ParMesh * pmesh = new ParMesh(MPI_COMM_WORLD, *mesh, proc_partitioning);
    fem_refine_mesh_times(args.num_refine, *pmesh);
    SA_RPRINTF(0,"Dimensions: %d, Number of vertices: %d, Number of elements: %d",
               mesh->Dimension(), mesh->GetNV(), mesh->GetNE());

    FiniteElementCollection * fec = new LinearFECollection;
    ParFiniteElementSpace * fes = new ParFiniteElementSpace(pmesh, fec);
    pmesh->PrintInfo();

    /// Boundary coefficient and r.h.s. coefficient.
    FunctionCoefficient bdr_coeff(bdr_cond);
    FunctionCoefficient rhs(rhs_func);

    Coefficient * conduct_coeff;
    FiniteElementCollection * cfec = (FiniteElementCollection*) new Const3DFECollection;
    ParFiniteElementSpace * cfes = new ParFiniteElementSpace(pmesh, cfec);
    FunctionCoefficient conduct_func(checkboard_coef);
    ParGridFunction conductivity(cfes);
    if (args.checkerboard)
    {
        conductivity.ProjectCoefficient(conduct_func);
        conduct_coeff = new GridFunctionCoefficient(&conductivity);
    }
    else
    {
        conduct_coeff = new ConstantCoefficient(1.0);
    }

    fem_build_discrete_problem(fes, rhs, bdr_coeff, *conduct_coeff, true, x, b,
                               a, &ess_bdr);
    SparseMatrix& Al = a->SpMat();
    HypreParMatrix *Ag = a->ParallelAssemble();
    HypreParVector *bg = b->ParallelAssemble();
    HypreParVector *hxg = x.ParallelAverage();
    HypreParVector *sxg = x.ParallelAverage();
    HypreParVector *pxg = x.ParallelAverage();

    // Solver stuff.
    SA_RPRINTF(0, "%s", "\n\t\t\tSOLVING THE FINE PROBLEM USING HYPRE AND NO COARSE GRID:\n\n");
    HypreSolver *amg = new HypreBoomerAMG(*Ag);
    HyprePCG *pcg = new HyprePCG(*Ag);
    pcg->SetTol(1e-12);
    pcg->SetMaxIter(1000);
    pcg->SetPrintLevel(2);
    pcg->SetPreconditioner(*amg);
    pcg->Mult(*bg, *hxg);
    delete pcg;
    delete amg;
    x = *hxg;

    /// Create create partitioning on the finest level. These are agglomerates,
    /// aggregates, and other structures and relations.
    // CONFIG_ACCESS_OPTION(INTERP, bdr_cond_imposed) = false;
    // CONFIG_ACCESS_OPTION(CONTRIB, avoid_ess_brd_dofs) = false;
    const bool avoid_ess_bdr_dofs = false;

    int pne = pmesh->GetNE(); // this is a local number of elements
    int nparts = (int) round((double) pne / (double) args.elems_per_agg);
    agg_dof_status_t *bdr_dofs = fem_find_bdr_dofs(*fes, &ess_bdr);

    agg_part_rels = fem_create_partitioning(*Ag, *fes, bdr_dofs, &nparts, false);
    delete [] bdr_dofs;
    ElementMatrixStandardGeometric emp(*agg_part_rels, Al, a);
    emp.SetBdrCondImposed(false);
    const int polynomial_coarse = -1; // purely spectral space
    const bool use_arpack = false;
    tg_data = tg_produce_data(*Ag,  *agg_part_rels, args.nu_pro, args.nu_relax,
                              &emp, args.theta, 1, polynomial_coarse,
                              use_arpack, avoid_ess_bdr_dofs);

/// =======================================================================
/// Solving the problem.
/// =======================================================================

    /*
    SA_RPRINTF(0, "%s", "\n");
    SA_RPRINTF(0, "%s", "\t\t\tSOLVING THE PROBLEM USING STATIONARY"
                        " ITERATION:\n");
    SA_RPRINTF(0, "%s", "\n");
    tg_run(*Ag, agg_part_rels, *sxg, *bg, 1000, 1e-12, 0., 1., tg_data, false);
    x = *sxg;
    fem_parallel_visualize_gf(*pmesh, x);
    *sxg -= *hxg;
    x = *sxg;
    fem_parallel_visualize_gf(*pmesh, x);
    */

    /*
    SA_RPRINTF(0, "%s", "\n");
    SA_RPRINTF(0, "%s", "\t\t\tRUNNING STATIONARY ITERATION WITH RANDOM INITIAL"
                        " GUESS AND ZERO R.H.S:\n");
    SA_RPRINTF(0, "%s", "\n");
    tg_run(*Ag, agg_part_rels, *sxg, *bg, 1000, 1e-12, 0., 1., tg_data, true);
    x = *sxg;
    fem_parallel_visualize_gf(*pmesh, x);
    */

    SA_RPRINTF(0, "%s", "\n\t\t\tSOLVING THE PROBLEM USING PCG:\n\n");
    tg_pcg_run(*Ag, agg_part_rels, *pxg, *bg, 1000, 1e-12, 0., tg_data, false);
    x = *pxg;

    Coefficient * zero_coefficient = new ConstantCoefficient(0.0);
    double L2norm = x.ComputeL2Error(&zero_coefficient);
    SA_RPRINTF(0,"    fine solution L2 norm || x ||_L2 = %e\n",L2norm);
    // double fineenergy = sqrt(Ag->InnerProduct(x,x));
    // SA_RPRINTF(0,"    fine solution energy norm = %e\n",fineenergy);

    if (args.visualize) // visualize the solution
        fem_parallel_visualize_gf(*pmesh, x);
    *pxg -= *hxg;
    x = *pxg;
    if (args.visualize) // visualize error with respect to BoomerAMG solution?
        fem_parallel_visualize_gf(*pmesh, x);

    /*
    SA_RPRINTF(0, "%s", "\n");
    SA_RPRINTF(0, "%s", "\t\t\tRUNNING PCG WITH RANDOM INITIAL GUESS AND ZERO"
                        " R.H.S:\n");
    SA_RPRINTF(0, "%s", "\n");
    tg_pcg_run(*Ag, agg_part_rels, *pxg, *bg, 1000, 1e-12, 0., tg_data, true);
    x = *pxg;
    */

/// =======================================================================
/// Do the upscaling test.
/// =======================================================================

    SA_RPRINTF(0,"%s","Doing upscaled solution...\n");
    HypreParVector coarserhs(PROC_COMM, tg_data->restr->M(), tg_data->restr->RowPart());
    HypreParVector coarsesol(PROC_COMM, tg_data->restr->M(), tg_data->restr->RowPart());
    // Vector coarsesolonfinegrid(tg_data->restr->Width());
    ParGridFunction coarsesolonfinegrid(x);
    coarsesolonfinegrid = 0.0;
    tg_data->restr->Mult(*bg,coarserhs);
    // tg_data->restr->Mult(exxonfemrhs,coarserhs);
    if (false)
    {
        std::ofstream outfile("coarse.m");
        tg_data->Ac->PrintMatlab(outfile);
        outfile.close();
    }

    HypreSolver *coarseamg = new HypreBoomerAMG(*tg_data->Ac);
    HyprePCG *coarsepcg = new HyprePCG(*tg_data->Ac);
    coarsepcg->SetTol(1e-12);
    coarsepcg->SetMaxIter(1000);
    coarsepcg->SetPrintLevel(2);
    coarsepcg->SetPreconditioner(*coarseamg);
    coarsepcg->Mult(coarserhs, coarsesol);
    delete coarsepcg;
    delete coarseamg;
    x = *hxg;
    tg_data->interp->Mult(coarsesol,coarsesolonfinegrid);    

    if (args.visualize) // visualize coarse solution
        fem_parallel_visualize_gf(*pmesh, coarsesolonfinegrid);

    ParGridFunction upscaleerror(x);
    add(x,-1.0,coarsesolonfinegrid,upscaleerror);
    
    if (args.visualize) // visualize upscaling error
        fem_parallel_visualize_gf(*pmesh, upscaleerror);

    double upscaleerrorL2 = upscaleerror.ComputeL2Error(&zero_coefficient);
    SA_RPRINTF(0,"  L2 upscaling error = %e\n",upscaleerrorL2);
    SA_RPRINTF(0,"  relative L2 upscaling error = %e\n",upscaleerrorL2/L2norm);

    /*
    ConstantCoefficient onecoefficient(1.0);
    Vector zerovector(dim);
    for (int d=0; d<dim; ++d)
        zerovector(d) = 0.0;
    VectorConstantCoefficient zerovectorcoefficient(zerovector);
    double h1seminorm = upscaleerror.ComputeH1Error(&zero_coefficient,&zerovectorcoefficient,
                                                    &onecoefficient, 1.0, 1);
    SA_PRINTF("  H1 seminorm upscaling error no coefficient = %e\n",h1seminorm);

    h1seminorm = upscaleerror.ComputeH1Error(&zero_coefficient,&zerovectorcoefficient,
                                             &conduct_coeff, 1.0, 1);
    // h1seminorm = H1SemiNorm(&upscaleerror,conduct_coeff);
    SA_PRINTF("  H1 seminorm upscaling error with coefficient = %e\n",h1seminorm);
    // double scale = H1SemiNorm(&x,conduct_coeff);
    double scale = x.ComputeH1Error(&zero_coefficient,&zerovectorcoefficient,
                                    &conduct_coeff, 1.0, 1);
    SA_PRINTF("  relative H1 seminorm upscaling error with coefficient = %e\n",h1seminorm/scale);

    double matrixenergynorm = sqrt(A.InnerProduct(upscaleerror,upscaleerror));
    SA_PRINTF("  relative matrix energy norm upscaling error = %e\n",matrixenergynorm/fineenergy);

    Coefficient* coefficientarray = &zero_coefficient;
    double upscaleerrormax = upscaleerror.ComputeMaxError(&coefficientarray);
    SA_PRINTF("  upscaling error max = %e\n",upscaleerrormax);
    double maxscale = x.ComputeMaxError(&coefficientarray);
    SA_PRINTF("  relative upscaling error max norm = %e\n",upscaleerrormax/maxscale);
    */

/// =======================================================================
/// Freeing memory.
/// =======================================================================
    tg_free_data(tg_data);
    agg_free_partitioning(agg_part_rels);
    delete conduct_coeff;
    delete zero_coefficient;
    delete pxg;
    delete sxg;
    delete hxg;
    delete bg;
    delete Ag;
    delete a;
    delete b;

    delete fec;
    delete fes;
    delete cfec;
    delete cfes;

    delete pmesh;
    delete mesh;

    MPI_Finalize();

    return 0;
}

