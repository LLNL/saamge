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
   Trying to encapsulate a bunch of SAAMGe functionality in a single object.

   Andrew T. Barker
   atb@llnl.gov
   16 August 2017
*/

#include <mfem.hpp>
#include <mpi.h>
#include <saamge.hpp>

using namespace mfem;
using namespace saamge;

/// just so MPI_Finalize() happens at the very end, after all the MPI calls
class MPIWrapper
{
public:
   MPIWrapper(int * argc, char ** argv[]);
   ~MPIWrapper();
};


MPIWrapper::MPIWrapper(int * argc, char ** argv[])
{
   MPI_Init(argc, argv);
}

MPIWrapper::~MPIWrapper()
{
   MPI_Finalize();
}

int main(int argc, char *argv[])
{
    // Initialize process related stuff.
    MPIWrapper mpiw(&argc, &argv);
    proc_init(MPI_COMM_WORLD);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    Mesh *mesh;
    ParMesh *pmesh;
    ParGridFunction x;
    ParLinearForm *b;

    OptionsParser args(argc, argv);
    const char *mesh_file = "/home/barker29/meshes/mltest.mesh";
    args.AddOption(&mesh_file, "-m", "--mesh",
                   "Mesh file to use.");
    bool visualize = true;
    args.AddOption(&visualize, "-vis", "--visualization", "-no-vis",
                   "--no-visualization",
                   "Enable or disable GLVis visualization.");
    int nu_pro = 0;
    args.AddOption(&nu_pro, "-p", "--nu-pro",
                   "Degree of the smoother for the smoothed aggregation for first coarsening.");
    int first_nu_pro = -1;
    args.AddOption(&first_nu_pro, "-fp", "--first-nu-pro",
                   "Degree of smoother for smoothed aggregation on later coarsenings.");
    int nu_relax = 3;
    args.AddOption(&nu_relax, "-n", "--nu-relax",
                   "Degree for smoother in the relaxation.");
    int order = 1;
    args.AddOption(&order, "-o", "--order",
                   "Polynomial order of finite element space.");
    int dimension = 2;
    args.AddOption(&dimension, "-dim", "--dimension",
                   "Spatial dimension of internally generated mesh.");
    double theta = 0.003;
    args.AddOption(&theta, "-t", "--theta",
                   "Tolerance for eigenvalue problems.");
    double first_theta = -1.0;
    args.AddOption(&first_theta, "-ft", "--first-theta",
                   "Tolerance for eigenvalue problems for first (finest) coarsening.");
    int num_levels = 2;
    args.AddOption(&num_levels, "-l", "--num-levels",
                   "Number of levels in multilevel algorithm.");
    int elems_per_agg = 256;
    args.AddOption(&elems_per_agg, "-e", "--elems-per-agg",
                   "Number of elements per agglomerated element.");
    int first_elems_per_agg = -1;
    args.AddOption(&first_elems_per_agg, "-fe", "--first-elems-per-agg",
                   "Number of elements per AE for first (finest) coarsening.");
    bool zero_rhs = false;
    args.AddOption(&zero_rhs, "-z", "--zero-rhs",
                   "-nz", "--no-zero-rhs",
                   "Solve CG with zero RHS and random initial guess.");
    bool minimal_coarse = false;
    args.AddOption(&minimal_coarse, "-mc", "--minimal-coarse",
                   "-nmc", "--no-minimal-coarse",
                   "Minimal coarse space, ie, vector of all ones.");
    bool linear_coarse = false;
    args.AddOption(&linear_coarse, "-lc", "--linear-coarse",
                   "-nlc", "--no-linear-coarse",
                   "Add linear functions to coarse basis (only for finest coarsening).");
    bool correct_nulspace = true;
    args.AddOption(&correct_nulspace, "-c", "--correct-nulspace",
                   "-nc", "--no-correct-nulspace",
                   "Use the corrected nulspace technique on coarsest level.");
    bool double_cycle = false;
    args.AddOption(&double_cycle, "-d", "--double-cycle",
                   "-nd", "--no-double-cycle",
                   "Use the double cycle combined preconditioner.");
    bool direct_eigensolver = false;
    args.AddOption(&direct_eigensolver, "-q", "--direct-eigensolver",
                   "-nq", "--no-direct-eigensolver",
                   "Use direct eigensolver from LAPACK instead of default ARPACK.");
    bool elasticity = false;
    args.AddOption(&elasticity, "-el", "--elasticity",
                   "-nel", "--no-elasticity",
                   "Try elasticity instead of usual scalar elliptic problem.");
    int refine = 0;
    args.AddOption(&refine, "-r", "--refine",
                   "Refine mesh in parallel.");
    args.Parse();
    if (!args.Good())
    {
        if (rank == 0)
            args.PrintUsage(cout);
        MPI_Finalize();
        return 1;
    }
    if (rank == 0)
        args.PrintOptions(cout);
    if (first_elems_per_agg < 0) first_elems_per_agg = elems_per_agg;
    if (first_theta < 0.0) first_theta = theta;
    if (first_nu_pro < 0) first_nu_pro = nu_pro;

    // Read the mesh from the given mesh file.
    mesh = fem_read_mesh(mesh_file);
/*
    const int x_elements = 32;
    const int y_elements = 32;
    const int z_elements = 32;
    if (dimension == 2)
        mesh = new Mesh(x_elements, y_elements, Element::QUADRILATERAL, 1, 1.0, 1.0);
    else
        mesh = new Mesh(x_elements, y_elements, z_elements,
                        Element::HEXAHEDRON, 1, 1.0, 1.0, 1.0);
*/

    // Parallel mesh and finite elements stuff.
    Array<int> ess_bdr(mesh->bdr_attributes.Max());
    ess_bdr = 0;
    ess_bdr = 1; // Dirichlet boundaries all around

    int nprocs;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
    fem_refine_mesh_times(refine, *pmesh);

    FiniteElementCollection* fec;
    ParFiniteElementSpace* fes;
    if (elasticity)
    {
        fec = new H1_FECollection(order,pmesh->Dimension());
        fes = new ParFiniteElementSpace(pmesh,fec,pmesh->Dimension(),Ordering::byVDIM);
    }
    else
    {
        fec = new H1_FECollection(order);
        fes = new ParFiniteElementSpace(pmesh,fec);
    }
    // int pNV = pmesh->GetNV();
    // int pNE = pmesh->GetNE();
    // int pND = fes->GetNDofs();
    // int ND = fes->GlobalTrueVSize();

    FiniteElementCollection* cfec = new L2_FECollection(0, pmesh->Dimension());
    ParFiniteElementSpace* cfes = new ParFiniteElementSpace(pmesh, cfec);

    ConstantCoefficient bdr_coeff(0.0);
    ConstantCoefficient rhs(1.0);

    Vector bdr_vec(pmesh->Dimension()); // for elasticity
    bdr_vec = 1.0;
    VectorConstantCoefficient vec_bdr_coeff(bdr_vec);

    // MatrixFunctionCoefficient * matrix_conductivity = NULL;
    Coefficient * conduct_func = NULL;
    ParGridFunction conductivity(cfes);
    GridFunctionCoefficient * conduct_coeff = NULL; 
    ParBilinearForm * aform = NULL;
    if (elasticity)
    {
        conduct_func = new ConstantCoefficient(1.0);
        conductivity.ProjectCoefficient(*conduct_func);
        conduct_coeff = new GridFunctionCoefficient(&conductivity);

        const int dim = pmesh->Dimension();
        conduct_func = new ConstantCoefficient(1.0);

        const bool bdr_cond_impose = true;
        if (bdr_cond_impose)
        {
            x.SetSpace(fes);
            x.ProjectCoefficient(vec_bdr_coeff);
        }
        else
        {
            x.SetSpace(fes);
        }

        VectorArrayCoefficient f(dim); // should live outside the if...
        for (int i = 0; i < dim; i++)
            f.Set(i, new ConstantCoefficient(0.0));

        b = new ParLinearForm(fes);
        b->AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(f));
        b->Assemble();

        aform = new ParBilinearForm(fes);
        aform->AddDomainIntegrator(new ElasticityIntegrator(*conduct_coeff,1.0,1.0));
        aform->Assemble();
        if (bdr_cond_impose)
        {
            /* Imposing Dirichlet boundary conditions. */
            SA_ASSERT(ess_bdr.Size() == fes->GetMesh()->bdr_attributes.Max());
            Array<int> ess_dofs;
            fes->GetEssentialVDofs(ess_bdr, ess_dofs);
            const bool keep_diag = true;
            aform->EliminateEssentialBCFromDofs(ess_dofs, x, *b, keep_diag);
        }
        SA_RPRINTF_L(0, 4, "%s", "Finalizing global stiffness matrix...\n");
        aform->Finalize(0);
    }
    else
    {
        conduct_func = new ConstantCoefficient(1.0);
        conductivity.ProjectCoefficient(*conduct_func);
        conduct_coeff = new GridFunctionCoefficient(&conductivity);
        fem_build_discrete_problem(fes, rhs, bdr_coeff, *conduct_coeff, true, x, b,
                                   aform, &ess_bdr);
    }

    HypreParMatrix *Ag = aform->ParallelAssemble();
    HypreParVector *bg = b->ParallelAssemble();
    HypreParVector *hxg = x.ParallelAverage();
    HypreParVector *pxg = x.ParallelAverage();

    // some actual AMGe stuff
    if (zero_rhs)
    {
        pxg->Randomize(0);
        *bg = 0.0;
    }
    // int iterations = -1;
    // int converged = -1;

    int polynomial_coarse = -1;
    if (elasticity)
        polynomial_coarse = -1;
    const bool coarse_direct = false;
    SpectralAMGSolver spectral_pc(*Ag, *aform, aform->SpMat(),
                                  ess_bdr, elems_per_agg, num_levels,
                                  nu_pro, nu_relax, theta, polynomial_coarse,
                                  coarse_direct);
    CGSolver hpcg(MPI_COMM_WORLD);
    hpcg.SetOperator(*Ag);
    hpcg.SetRelTol(1.e-11); // for some reason MFEM squares this...
    hpcg.SetMaxIter(1000);
    hpcg.SetPrintLevel(1);
    hpcg.SetPreconditioner(spectral_pc);
    hpcg.Mult(*bg,*pxg);
    // iterations = hpcg.GetNumIterations();
    // converged = hpcg.GetConverged();
    x = *pxg;

    delete pxg;
    delete hxg;
    delete bg;
    delete Ag;
    delete aform;
    delete b;

    delete cfes;
    delete cfec;
    delete fes;
    delete fec;
    delete conduct_func;
    delete conduct_coeff;

    delete pmesh;
    delete mesh;

    return 0;
}
