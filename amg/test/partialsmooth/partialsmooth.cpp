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
   try a DoubleCycle with smoother interpolation of the minimal space
   and no smoothing of the spectral space

   Andrew T. Barker
   atb@llnl.gov
   1 September 2015
*/

#include <mfem.hpp>
#include <mpi.h>
#include <saamge.hpp>

#include "InversePermeabilityFunction.hpp"

using namespace mfem;
using namespace saamge;

/**
   just takes two mfem::Solvers and applies them both, either additively or
   multiplicatively

   not to be confused with DoubleCycle, which uses two different *coarse grid*
   solvers in a similar way
*/
class DoublePreconditioner : public mfem::Solver
{
public:
    DoublePreconditioner(const mfem::Solver& inner_solver,
                         const mfem::Solver& outer_solver,
                         bool symmetric = true,
                         bool additive = false,
                         bool iterative_mode = false);
    virtual ~DoublePreconditioner();
    virtual void SetOperator(const Operator &op);
    virtual void Mult(const Vector &x, Vector &y) const;
private:
    HypreParMatrix *A;
    const mfem::Solver& inner_solver;
    const mfem::Solver& outer_solver;
    bool symmetric, additive;
};

DoublePreconditioner::DoublePreconditioner(const mfem::Solver& inner_solver,
                                           const mfem::Solver& outer_solver,
                                           bool symmetric,
                                           bool additive,
                                           bool iterative_mode)
    : mfem::Solver(inner_solver.Height(), iterative_mode),
      inner_solver(inner_solver),
      outer_solver(outer_solver),
      symmetric(symmetric),
      additive(additive)
{
}

DoublePreconditioner::~DoublePreconditioner()
{
}

void DoublePreconditioner::SetOperator(const Operator &op)
{
    A = const_cast<HypreParMatrix *>(dynamic_cast<const HypreParMatrix *>(&op));
    if (A == NULL)
        mfem_error("HypreSmoother::SetOperator : not HypreParMatrix!");
}

void DoublePreconditioner::Mult(const Vector &b, Vector &x) const
{
    MFEM_ASSERT(!additive,"Additive cycle not implemented!");

    Vector x1(x);
    Vector res1(x);
    if (iterative_mode)
    {
        A->Mult(x,res1);
        subtract(b, res1, res1);
    }
    if (!iterative_mode)
    {
        x = 0.0;
        res1 = b;
    }

    outer_solver.Mult(res1, x1);
    
    A->Mult(x1, res1);
    subtract(b, res1, res1);

    inner_solver.Mult(res1, x1);
    add(x, x1, x);
    
    if (symmetric)
    {
        A->Mult(x, res1);
        subtract(b, res1, res1);
        outer_solver.Mult(res1, x1);
        add(x, x1, x);
    }    
}

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

/**
   [0.0, 0.1] x [0.0, 0.1] has coefficent 1e6, 
   [0.1, 0.2] x [0.0, 0.1] has coefficient 1, 
   etc.
*/
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
    } else
        return 1e0;
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

/**
   do the agglomerate partitioning for mltest mesh

   this partitions elements into agglomerates

   this only makes sense for the first coarsening
*/
agg_partitioning_relations_t *
fem_create_test_partitioning(HypreParMatrix& A, ParFiniteElementSpace& fes,
                             const agg_dof_status_t *bdr_dofs, int *nparts,
                             bool do_aggregates)
{
    Table *elem_to_dof, *elem_to_elem;
    Mesh *mesh = fes.GetMesh();

    //XXX: This will stay allocated in MESH till the end.
    elem_to_elem = mbox_copy_table(&(mesh->ElementToElementTable()));

    //fes.BuildElementToDofTable(); //XXX: This remains allocated in FES till the
    //                             //     end.
    elem_to_dof = mbox_copy_table(&(fes.GetElementToDofTable()));

    int * partitioning = NULL;
    if (PROC_NUM == 1)
    {
        partitioning = new int[12];
        partitioning[0] = partitioning[1] = partitioning[4] = partitioning[5] = 0;
        partitioning[2] = partitioning[3] = 1;
        partitioning[6] = partitioning[7] = partitioning[11] = 2;
        partitioning[8] = partitioning[9] = partitioning[10] = 3;
    } 
    else if (PROC_NUM == 2 && PROC_RANK == 0)
    {
        partitioning = new int[6];
        partitioning[0] = partitioning[1] = partitioning[4] = partitioning[5] = 0;
        partitioning[2] = partitioning[3] = 1;
    }
    else if (PROC_NUM == 2 && PROC_RANK == 1)
    {
        partitioning = new int[6];
        partitioning[0] = partitioning[1] = partitioning[5] = 0;
        partitioning[2] = partitioning[3] = partitioning[4] = 1;
    }
    else if (PROC_NUM == 4)
    {   
        int num_elem = 0;
        if (PROC_RANK == 0) num_elem = 4;
        if (PROC_RANK == 1) num_elem = 2;
        if (PROC_RANK == 2) num_elem = 3;
        if (PROC_RANK == 3) num_elem = 3;
        partitioning = new int[num_elem];
        for (int j=0; j<num_elem; ++j)
            partitioning[j] = 0;
    }
    else
    {
        SA_ASSERT(false);
    }

    // in what follows, bdr_dofs is only used as info to copy onto coarser level, 
    // does not actually affect partitioning
    agg_partitioning_relations_t *agg_part_rels =
        agg_create_partitioning_fine(A, fes.GetNE(), elem_to_dof, elem_to_elem,
                                     partitioning, bdr_dofs, nparts, 
                                     fes.Dof_TrueDof_Matrix(), do_aggregates, true);

    SA_ASSERT(agg_part_rels);
    return agg_part_rels;
}

/**
   this partitions elements onto processors

   fake two processor partition for mltest.mesh
*/
int *fem_partition_test_mesh(Mesh& mesh, int *nparts)
{
    SA_ASSERT(*nparts == 2 || *nparts == 4);

    int * out = new int[mesh.GetNE()];
    if (*nparts == 2)
    {
        for (int i=0; i<6; ++i)
        {
            out[i] = 0;
            out[i+6] = 1;
        }    
    }
    else if (*nparts == 4)
    {
        out[0] = out[1] = out[4] = out[5] = 0;
        out[2] = out[3] = 1;
        out[6] = out[7] = out[11] = 2;
        out[8] = out[9] = out[10] = 3;
    }
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

    Mesh *mesh;
    ParMesh *pmesh;
    int *proc_partitioning;
    ParGridFunction x;
    ParLinearForm *b;
    ParBilinearForm *a;
    agg_partitioning_relations_t *agg_part_rels;
    ml_data_t *ml_data_minimal;
    ml_data_t *ml_data_spectral;

    const char *mesh_file = "/home/barker29/meshes/cube474.mesh3d";
    const char *perm_file = "/g/g14/barker29/spe10/spe_perm.dat";
    bool visualize = true;

    int serial_times_refine = 0;
    int times_refine = 0;
    int first_nu_pro = -1;
    int nu_pro = 0;
    int nu_relax = 3;
    int spe10_scale = 5; // default is "full" SPE10
    double theta = 0.003;
    double first_theta = -1.0;
    int num_levels = 2;
    int elems_per_agg = 256;
    int first_elems_per_agg = -1;
    bool constant_coefficient = false;

    bool spe10 = false;
    bool zero_rhs = false;
    // bool minimal_coarse = false;
    // bool correct_nulspace = true;

    OptionsParser args(argc, argv);
    args.AddOption(&mesh_file, "-m", "--mesh",
                   "Mesh file to use.");
    args.AddOption(&perm_file, "-pf", "--perm",
                   "Permeability data, only relevant with --spe10.");
    args.AddOption(&visualize, "-vis", "--visualization", "-no-vis",
                   "--no-visualization",
                   "Enable or disable GLVis visualization.");
    args.AddOption(&serial_times_refine, "-sr", "--serial-refine",
                   "How many times to refine mesh before parallel partition.");
    args.AddOption(&times_refine, "-r", "--refine", 
                   "How many times to refine the mesh (in parallel).");
    args.AddOption(&nu_pro, "-p", "--nu-pro",
                   "Degree of the smoother for the smoothed aggregation for first coarsening.");
    args.AddOption(&first_nu_pro, "-fp", "--first-nu-pro",
                   "Degree of smoother for smoothed aggregation on later coarsenings.");
    args.AddOption(&nu_relax, "-n", "--nu-relax",
                   "Degree for smoother in the relaxation.");
    args.AddOption(&spe10_scale, "-ss", "--spe10-scale",
                   "Scale of SPE10, 5 is full, smaller is smaller, larger does not make sense.");
    args.AddOption(&theta, "-t", "--theta",
                   "Tolerance for eigenvalue problems.");
    args.AddOption(&first_theta, "-ft", "--first-theta",
                   "Tolerance for eigenvalue problems for first (finest) coarsening.");
    args.AddOption(&num_levels, "-l", "--num-levels",
                   "Number of levels in multilevel algorithm.");
    args.AddOption(&elems_per_agg, "-e", "--elems-per-agg",
                   "Number of elements per agglomerated element.");
    args.AddOption(&first_elems_per_agg, "-fe", "--first-elems-per-agg",
                   "Number of elements per AE for first (finest) coarsening.");
    args.AddOption(&constant_coefficient, "-k", "--constant-coefficient",
                   "-nk", "--no-constant-coefficient",
                   "Use a constant coefficient instead of the default checkerboard.");
    args.AddOption(&spe10, "-s", "--spe10",
                   "-ns", "--no-spe10",
                   "Use the SPE10 geometry and permeability data set.");
    args.AddOption(&zero_rhs, "-z", "--zero-rhs",
                   "-nz", "--no-zero-rhs",
                   "Solve CG with zero RHS and random initial guess.");

    args.Parse();
    if (!args.Good())
    {
        if (PROC_RANK == 0)
            args.PrintUsage(cout);
        MPI_Finalize();
        return 1;
    }
    if (PROC_RANK == 0)
    {
        SA_RPRINTF(0,"%s","Running partialsmooth example.\n\n");
        args.PrintOptions(cout);
    }
    if (first_elems_per_agg < 0) first_elems_per_agg = elems_per_agg;
    if (first_theta < 0.0) first_theta = theta;
    if (first_nu_pro < 0) first_nu_pro = nu_pro;

    MPI_Barrier(PROC_COMM); // try to make MFEM's debug element orientation prints not mess up the parameters above
    bool mltest = false;
    if (spe10)
    {
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
    }
    else
    {
        // Read the mesh from the given mesh file.
        mesh = fem_read_mesh(mesh_file);
        if (mesh->GetNV() == 20 && mesh->GetNE() == 12 && 
            times_refine == 0 && serial_times_refine == 0) // not very general...
            mltest = true;
    }
    fem_refine_mesh_times(serial_times_refine, *mesh);

    // Serial mesh.
    SA_RPRINTF(0,"NV: %d, NE: %d\n", mesh->GetNV(), mesh->GetNE());

    // Parallel mesh and finite elements stuff.
    Array<int> ess_bdr(mesh->bdr_attributes.Max());
    ess_bdr = 0;
    if (mltest)
        ess_bdr[3] = 1; // marked as 4 in mltest.mesh, but MFEM subtracts 1 because it's insane
    else if (spe10)
    {
        ess_bdr = 1;
        ess_bdr[0] = 0; 
        ess_bdr[1] = 0; 
    }
    else
        ess_bdr = 1; // Dirichlet boundaries all around
        
    FunctionCoefficient bdr_coeff(bdr_cond);
    FunctionCoefficient rhs(rhs_func);

    int nprocs = PROC_NUM;
    if (nprocs > 1 && mltest)
        proc_partitioning = fem_partition_test_mesh(*mesh, &nprocs);
    else
        proc_partitioning = fem_partition_mesh(*mesh, &nprocs);
    if (0 == PROC_RANK && visualize)
        fem_serial_visualize_partitioning(*mesh, proc_partitioning);
    pmesh = new ParMesh(MPI_COMM_WORLD, *mesh, proc_partitioning);
    delete [] proc_partitioning;
    fem_refine_mesh_times(times_refine, *pmesh);

    FiniteElementCollection * fec = new LinearFECollection;
    ParFiniteElementSpace * fes = new ParFiniteElementSpace(pmesh, fec);
    int pNV = pmesh->GetNV();
    int pNE = pmesh->GetNE();
    int pND = fes->GetNDofs();
    int ND = fes->GlobalTrueVSize();
    SA_RPRINTF(0,"pNV: %d, pNE: %d, pND: %d, ND: %d\n", 
               pNV, pNE, pND, ND);

    FiniteElementCollection * cfec = new L2_FECollection(0, pmesh->Dimension());
    ParFiniteElementSpace * cfes = new ParFiniteElementSpace(pmesh, cfec);

    MatrixFunctionCoefficient * matrix_conductivity = NULL; 
    Coefficient * conduct_func = NULL;
    ParGridFunction conductivity(cfes);
    GridFunctionCoefficient * conduct_coeff = NULL; 
    if (spe10 && !constant_coefficient)
    {
        matrix_conductivity = new MatrixFunctionCoefficient(pmesh->Dimension(),
                                                            InversePermeabilityFunction::PermeabilityTensor);
        fem_build_discrete_problem(fes, rhs, bdr_coeff, *matrix_conductivity, true, x, b,
                                   a, &ess_bdr);
    }
    else
    {
        if (constant_coefficient)
            conduct_func = new ConstantCoefficient(1.0);
        else
            conduct_func = new FunctionCoefficient(checkboard_coef);
        conductivity.ProjectCoefficient(*conduct_func);
        conduct_coeff = new GridFunctionCoefficient(&conductivity);
        if (false)
            fem_parallel_visualize_gf(*pmesh, conductivity,
                                      pmesh->Dimension() == 2?"jfR":"f");
        fem_build_discrete_problem(fes, rhs, bdr_coeff, *conduct_coeff, true, x, b,
                                   a, &ess_bdr);
    }

    SparseMatrix& Al = a->SpMat();
    HypreParMatrix *Ag = a->ParallelAssemble();
    HypreParVector *bg = b->ParallelAssemble();
    HypreParVector *hxg = x.ParallelAverage();
    HypreParVector *pxg = x.ParallelAverage();
    chrono.Stop();
    SA_RPRINTF(0, "TIMING: fem setup %f seconds.\n", chrono.RealTime());

    // basic solver stuff.
    chrono.Clear();
    chrono.Start();
    SA_RPRINTF(0, "%s", "\n");
    SA_RPRINTF(0, "%s", "\t\t\tSOLVING THE ORIGINAL FINE SCALE PROBLEM USING HYPRE:\n");
    SA_RPRINTF(0, "%s", "\n");
    HypreBoomerAMG *hbamg = new HypreBoomerAMG(*Ag);
    hbamg->SetPrintLevel(0);

    CGSolver * pcg = new CGSolver(MPI_COMM_WORLD);
    pcg->SetOperator(*Ag);
    pcg->SetRelTol(1e-6); // for some reason MFEM squares this...
    pcg->SetMaxIter(1000);
    pcg->SetPrintLevel(1);
    pcg->SetPreconditioner(*hbamg);

    if (zero_rhs)
    {
        // helpers_random_vect(*agg_part_rels, *hxg);
        hxg->Randomize(0);
        *bg = 0.0;
    }
    SA_RPRINTF(0, "hxg->Norml2() = %f\n", hxg->Norml2());
    pcg->Mult(*bg, *hxg);
    delete pcg;
    delete hbamg;
    x = *hxg;
    if (false)
        fem_parallel_visualize_gf(*pmesh, x);
    SA_RPRINTF(0, "TIMING: setup and solve with Hypre BoomerAMG preconditioned CG %f seconds.\n", 
               chrono.RealTime());

    // some actual AMGe stuff
    chrono.Clear();
    chrono.Start();
    int * nparts_arr = new int[num_levels-1];
    agg_dof_status_t *bdr_dofs = fem_find_bdr_dofs(*fes, &ess_bdr);
    if (mltest)
    {
        nparts_arr[0] = 4 / PROC_NUM;
        agg_part_rels = fem_create_test_partitioning(*Ag, *fes, bdr_dofs, nparts_arr, false);
        if (num_levels > 2)
            nparts_arr[1] = 2 / PROC_NUM;
        if (num_levels > 3)
            SA_ASSERT(false);
    }
    else
    {
        nparts_arr[0] = pmesh->GetNE() / first_elems_per_agg;
        agg_part_rels = fem_create_partitioning(*Ag, *fes, bdr_dofs, nparts_arr, false);
        for(int i=1; i < num_levels-1; ++i)
        {
            nparts_arr[i] = (int) round((double) nparts_arr[i-1] / (double) elems_per_agg);
            if (nparts_arr[i] < 1) nparts_arr[i] = 1;
        }
    }
    delete [] bdr_dofs;

    if (mltest)
        fes->Dof_TrueDof_Matrix()->Print("Dof_TrueDof.mat");
    if (visualize)
    {
        fem_parallel_visualize_partitioning(*pmesh, agg_part_rels->partitioning,
                                            nparts_arr[0]);
        // fem_parallel_visualize_aggregates(fes, agg_part_rels->aggregates, nparts_arr[0]);
    }

    const bool use_arpack = true;
    const bool do_aggregates = false;
    MultilevelParameters mlp(num_levels-1, nparts_arr, 
                             first_nu_pro, nu_pro,
                             nu_relax, 0.0, 0.0, 
                             0, false, use_arpack, do_aggregates); // polynomial_coarse, correct_nullspace
    ElementMatrixProvider * emp = new ElementMatrixStandardGeometric(
        *agg_part_rels, Al, a);
    ml_data_minimal = ml_produce_data(*Ag, agg_part_rels, emp, mlp);

    MultilevelParameters mlp_spectral(num_levels-1, nparts_arr, 0, 0,
                                      nu_relax, first_theta, theta, 
                                      -1, true, use_arpack, do_aggregates); // polynomial_coarse, correct_nullspace
    ElementMatrixProvider * emp_spectral = new ElementMatrixStandardGeometric(
        *agg_part_rels, Al, a);
    ml_data_spectral = ml_produce_data(*Ag, agg_part_rels, emp_spectral, mlp_spectral);

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
    if (zero_rhs)
    {
        // helpers_random_vect(*agg_part_rels, *pxg);
        pxg->Randomize(0);
        *bg = 0.0;
    }
    int iterations = -1;
    int converged = -1;
    Solver * Bprec;

    levels_level_t * level_spectral = levels_list_get_level(ml_data_spectral->levels_list, 0);
    levels_level_t * level_minimal = levels_list_get_level(ml_data_minimal->levels_list, 0);
    VCycleSolver inner_solver(level_spectral->tg_data, false);
    VCycleSolver outer_solver(level_minimal->tg_data, false);
    inner_solver.SetOperator(*Ag);
    outer_solver.SetOperator(*Ag);
    Bprec = new DoublePreconditioner(inner_solver, outer_solver); // inner, outer is standard
    Bprec->SetOperator(*Ag);

    CGSolver hpcg(MPI_COMM_WORLD);
    hpcg.SetOperator(*Ag);
    hpcg.SetRelTol(1e-6); // for some reason MFEM squares this...
    hpcg.SetMaxIter(1000);
    hpcg.SetPrintLevel(1);
    hpcg.SetPreconditioner(*Bprec);
    hpcg.Mult(*bg,*pxg);
    iterations = hpcg.GetNumIterations();
    converged = hpcg.GetConverged();

    delete Bprec;
    if (converged)
        SA_RPRINTF(0, "Outer PCG converged in %d iterations.\n", iterations);
    else
        SA_RPRINTF(0, "Outer PCG failed to converge after %d iterations!\n", iterations);
    x = *pxg;
    if (false)
        fem_parallel_visualize_gf(*pmesh, x);
    chrono.Stop();
    SA_RPRINTF(0,"TIMING: solve with SA-AMGe preconditioned CG %f seconds.\n", chrono.RealTime());

    ml_free_data(ml_data_spectral);
    ml_free_data(ml_data_minimal);

    agg_free_partitioning(agg_part_rels);

    delete pxg;
    delete hxg;
    delete bg;
    delete Ag;
    delete a;
    delete b;

    delete cfes;
    delete cfec;
    delete fes;
    delete fec;
    if (spe10 && !constant_coefficient)
        delete matrix_conductivity;
    else
    {
        delete conduct_func;
        delete conduct_coeff;
    }

    delete [] nparts_arr;
    delete pmesh;
    delete mesh;

    MPI_Finalize();

    return 0;
}
