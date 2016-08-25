/*! \file

    SAAMGE: smoothed aggregation element based algebraic multigrid hierarchies
            and solvers.

    Copyright (c) 2015, Lawrence Livermore National Security,
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

    Example command line:

    mpirun -np 2 ./spe10 --permeability /home/user/spe_perm.dat --refine 0 --solver cg --elems-per-agg 100 --theta 0.001 --nu-pro 2 --nu-relax 2
*/

#include <mfem.hpp>
#include <mpi.h>
#include <saamge.hpp>

#include "InversePermeabilityFunction.hpp"

class Arguments
{
public:
    Arguments(int argc, char *argv[]);
    void unknown_arg(std::string & arg);

    bool visualize;
    bool big_spe10;
    bool small_spe10;
    std::string permfile; 
    std::string solver;
    int num_refine;
    int elems_per_agg; 
    int nu_pro;
    int nu_relax;
    double theta;
};

Arguments::Arguments(int argc, char *argv[])
{
    visualize = false;
    big_spe10 = false;
    small_spe10 = false;
    permfile = "/home/user/spe10/spe_perm.dat";
    solver = "cg";
    num_refine = 2;
    elems_per_agg = 200; 
    nu_pro = 3;
    nu_relax = 3;
    theta = 0.003;

    int i = 1;
    while (i < argc)
    {
	std::string arg(argv[i]);
	if (arg == "--visualize")
	{
	    visualize = true;
	    i++;
	}
	else if (arg == "--big")
	{
	    big_spe10 = true;
	    i++;
	}
	else if (arg == "--small")
	{
	    small_spe10 = true;
	    i++;
	}
	else if (arg == "--permeability")
	{
	    permfile = argv[i+1];
	    i += 2;
	}
	else if (arg == "--solver")
	{
   	    solver = argv[i+1];
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
	else
	{	  
	    unknown_arg(arg);
	}
    }
}

void Arguments::unknown_arg(std::string & arg)
{
    std::cout << "Unknown or incorrectly used argument " << arg << std::endl;
    std::cout << "Example command line: " << std::endl 
	      << "  mpirun -np 2 ./spe10 --permeability /home/user/spe_perm.dat --refine 0 --solver cg --elems-per-agg 100 --theta 0.001 --nu-pro 2 --nu-relax 2" << std::endl;
    throw 1;
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

/*!
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


/*!
  this class designed for building and rebuilding the Hierarchy with 
  different boundary conditions, which we no longer really need to do
*/
class DiscreteTwoLevelProblem
{
public:
    DiscreteTwoLevelProblem(ParMesh * pmesh, ParFiniteElementSpace * fes,
			    Arguments & args, bool neumann);
    void SolveStationary();
    void SolveCG();
    void PrintTGData();
    void PrintP(std::string filename);
    void PrintAEtoDoF(std::string filename);
    ~DiscreteTwoLevelProblem();
private:
    // the following points to stuff owned by other objects
    ParMesh * pmesh;
    ParFiniteElementSpace * fes;
    Arguments& args;

    // we actually own and (if necessary) free the following
    StopWatch chrono;
    Array<int> ess_bdr;
    tg_data_t *tg_data;
    agg_partititoning_relations_t *agg_part_rels;
    ParGridFunction x;
    ParLinearForm *b;
    ParBilinearForm *a;

    HypreParMatrix *Ag;
    HypreParVector *bg;
    HypreParVector *hxg;
    HypreParVector *sxg;
    HypreParVector *pxg;
};

DiscreteTwoLevelProblem::DiscreteTwoLevelProblem(ParMesh * pmesh, ParFiniteElementSpace * fes,
						 Arguments& args, bool neumann) :
    pmesh(pmesh),
    fes(fes),
    args(args)
{
    chrono.Clear();
    chrono.Start(); // partition, matrix_assembly
    MatrixFunctionCoefficient matrix_conductivity(pmesh->Dimension(),
						  InversePermeabilityFunction::PermeabilityTensor);
	
    // build the discrete problem.
    // in the SPE10 context it is not clear the best way to assign these boundaries
    ess_bdr.SetSize(pmesh->bdr_attributes.Max());
    if (neumann)
    // if (false)
	ess_bdr = 0;
    else
	ess_bdr = 1;
    ess_bdr[0] = 0; 
    ess_bdr[1] = 0; 

    if (neumann)
    {
	SA_RPRINTF(0,"%s","Constructing problem using pure Neumann conditions.\n");
	CONFIG_ACCESS_OPTION(CONTRIB, avoid_ess_brd_dofs) = false;
    }
    else
    {
	SA_RPRINTF(0,"%s","Constructing problem with some Dirichlet conditions.\n");
	CONFIG_ACCESS_OPTION(CONTRIB, avoid_ess_brd_dofs) = true;
    }

    FunctionCoefficient bdr_coeff(bdr_cond);
    FunctionCoefficient rhs(rhs_func);

    // build problem on fine level
    fem_build_discrete_problem(fes, rhs, bdr_coeff, matrix_conductivity, true, x, b,
			       a, &ess_bdr);
    
    SparseMatrix& Al = a->SpMat();
    Ag = a->ParallelAssemble();
    bg = b->ParallelAssemble();
    hxg = x.ParallelAverage();
    sxg = x.ParallelAverage();
    pxg = x.ParallelAverage();
    chrono.Stop();
    SA_RPRINTF(0,"time for MFEM fine grid matrix assembly: %f\n",chrono.RealTime());
    SA_RPRINTF(0,"Ag->M() = %d, Ag->N() = %d\n", Ag->M(), Ag->N());

    chrono.Clear();
    chrono.Start(); // spectral two-grid setup
    int pne = pmesh->GetNE(); // this is a local (to this processor) number of elements
    int nparts = (int) round((double) pne / (double) args.elems_per_agg);
    agg_dof_status_t *bdr_dofs = fem_find_bdr_dofs(*fes, &ess_bdr);
    agg_part_rels = fem_create_partitioning(Al, *fes, bdr_dofs, &nparts);
    int global_nparts;
    MPI_Reduce(&nparts,&global_nparts,1,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);
    SA_RPRINTF(0,"global nparts = %d\n",global_nparts);
    SA_RPRINTF(0,"elems_per_agg = %d\n",args.elems_per_agg);
    SA_RPRINTF(0,"%s","fem_create_partitioning done.\n");
    delete [] bdr_dofs;
    if (args.visualize)
    {
	fem_parallel_visualize_partitioning(*pmesh, agg_part_rels->partitioning,
					    nparts);
	fem_parallel_visualize_aggregates(fes, agg_part_rels->aggregates, nparts);
    }

    {
	CONFIG_ACCESS_OPTION(TG, coarse_solver_solve_init) = solve_spd_AMG_init;
	CONFIG_ACCESS_OPTION(TG, coarse_solver_solve_free) = solve_spd_AMG_free;
	CONFIG_ACCESS_OPTION(TG, coarse_solver_solve_copy) = solve_spd_AMG_copy;
	CONFIG_ACCESS_OPTION(TG, coarse_solver_solver) = solve_spd_AMG;
	CONFIG_ACCESS_OPTION(TG, coarse_solver_data) = NULL; 
    }

    tg_data = tg_produce_data(Al, *Ag, *agg_part_rels, args.nu_pro, args.nu_relax, 0., a,
                              args.theta);

    chrono.Stop();
    SA_RPRINTF(0,"time for spectral two-grid setup: %f\n",chrono.RealTime());
}

void DiscreteTwoLevelProblem::SolveStationary()
{
    chrono.Clear();
    chrono.Start();    
    SA_RPRINTF(0, "%s", "\n");
    SA_RPRINTF(0, "%s", "\t\t\tRUNNING STATIONARY ITERATION WITH RANDOM INITIAL"
	       " GUESS AND ZERO R.H.S:\n");
    SA_RPRINTF(0, "%s", "\n");
    tg_run(*Ag, agg_part_rels, *sxg, *bg, 1000, 1e-12, 0., 1., tg_data, true);
    x = *sxg;
    if (args.visualize)
	fem_parallel_visualize_gf(*pmesh, x);
    chrono.Stop();
    SA_RPRINTF(0,"time for stationary solve: %f\n",chrono.RealTime());
}

void DiscreteTwoLevelProblem::SolveCG()
{
    chrono.Clear();
    chrono.Start();    
    SA_RPRINTF(0, "%s", "\n");
    SA_RPRINTF(0, "%s", "\t\t\tRUNNING PCG WITH RANDOM INITIAL GUESS AND ZERO"
	       " R.H.S:\n");
    SA_RPRINTF(0, "%s", "\n");
    tg_pcg_run(*Ag, agg_part_rels, *pxg, *bg, 1000, 1e-12, 0., tg_data, true);
    x = *pxg;
    chrono.Stop();
    SA_RPRINTF(0,"time for PCG solve: %f\n",chrono.RealTime());

    // tg_print_data(*Ag, tg_data);
}

void DiscreteTwoLevelProblem::PrintTGData()
{
    tg_print_data(*Ag, tg_data);
}

void DiscreteTwoLevelProblem::PrintP(std::string filename)
{
    tg_data->interp->Print(filename.c_str());
}

void DiscreteTwoLevelProblem::PrintAEtoDoF(std::string filename)
{
    // Table *AE_to_dof; /*!< The relation table that relates AEs to DoFs. */
    std::ofstream out(filename.c_str());
    agg_part_rels->AE_to_dof->Print(out);
}

DiscreteTwoLevelProblem::~DiscreteTwoLevelProblem()
{
    delete Ag;
    delete bg;
    delete hxg;
    delete sxg;
    delete pxg;

    delete a;
    delete b;

    tg_free_data(tg_data);
    agg_free_partitioning(agg_part_rels);
}


int main(int argc, char *argv[])
{
    Mesh *mesh;
    ParMesh *pmesh;
    int *proc_partitioning;

    StopWatch chrono;
    chrono.Clear();
    chrono.Start(); // serial mesh loading, initialization

    if (argc == 1) {
	SA_RPRINTF(0,"%s","Requires some command line arguments! (at least --permeability permfile)\n");
        return 1;
    }

    // Initialize process related stuff.
    MPI_Init(&argc, &argv);
    proc_init(MPI_COMM_WORLD);
    CONFIG_ACCESS_OPTION(GLOBAL, output_level) = 0;

    SA_RPRINTF(0,"%s","========================================\n");
    SA_RPRINTF(0,"%s","spe10 parallel spectral method test\n");
    SA_RPRINTF(0,"%s","========================================\n\n");

    Arguments args(argc,argv);
        
    int Nx,Ny,Nz;
    // number of cells for the original SPE10 test case.
    // In the benchmark we have Nx = 60, Ny = 220, Nz = 85.
    // This is equivalent to more or less 4.5 Million unknowns 
    // in the lowest order case.
    if (args.big_spe10)
    {
	// full SPE10, recommend either parallel computing or patience
	Nx = 60;
	Ny = 220;
	Nz = 85;
    } 
    else if (args.small_spe10)
    {
	// very small test example (for eg. valgrind)
	Nx = 10;
	Ny = 18;
	Nz = 15;
    }
    else 
    {
	// regular run, for example on a local desktop, roughly 30K unknowns.
	Nx = 15;
	Ny = 27;
	Nz = 20;
    }

    // cell sizes for fine cartesian grid (do not change this)
    double hx = 20.;
    double hy = 10.;
    double hz = 2.;

    // Create the finite element mesh (serial, redundant on every processor)
    mesh = create_hexahedral_mesh(Nx, Ny, Nz,
				  hx, hy, hz);

    // Serial mesh.
    SA_RPRINTF(0,"NV: %d, NE: %d\n", mesh->GetNV(), mesh->GetNE());

    // Parallel mesh and finite elements stuff.
    chrono.Stop();
    SA_RPRINTF(0,"time for serial mesh loading/initialization: %f\n",chrono.RealTime());
    
    chrono.Clear();
    chrono.Start();
    int nprocs = PROC_NUM;
    proc_partitioning = fem_partition_mesh(*mesh, &nprocs);
    if (args.visualize && 0 == PROC_RANK)
        fem_serial_visualize_partitioning(*mesh, proc_partitioning);
    pmesh = new ParMesh(MPI_COMM_WORLD, *mesh, proc_partitioning);
    // delete [] proc_partitioning;
    fem_refine_mesh_times(args.num_refine, *pmesh);

    FiniteElementCollection * fec = new LinearFECollection;
    ParFiniteElementSpace * fes = new ParFiniteElementSpace(pmesh, fec);
    pmesh->PrintInfo();

    // Load the SPE10 data set
    InversePermeabilityFunction::SetNumberCells(Nx,Ny,Nz);
    InversePermeabilityFunction::SetMeshSizes(hx, hy, hz);
    InversePermeabilityFunction::ReadPermeabilityFile(args.permfile);
    chrono.Stop();
    SA_RPRINTF(0,"time for fine mesh parallel refinement and partitioning: %f\n",chrono.RealTime());

    DiscreteTwoLevelProblem * wholeballofwax;

    // build (or rebuild) hierarchy, now with Dirichlet boundary conditions
    wholeballofwax = new DiscreteTwoLevelProblem(pmesh, fes, args, false);
    // if (nprocs == 1)
    // 	wholeballofwax->PrintP("dirichletP.mat");

    SA_RPRINTF(0,"spectral tolerance theta = %f, nu_pro = %d, nu_relax = %d\n",
	       args.theta, args.nu_pro, args.nu_relax);

    if (args.solver == "stationary" || args.solver == "both")
    { 
	wholeballofwax->SolveStationary();
    }
    if (args.solver == "cg" || args.solver == "both")
    {
	wholeballofwax->SolveCG();
    }
    wholeballofwax->PrintTGData();

    InversePermeabilityFunction::ClearMemory();
    delete wholeballofwax;

    delete fec;
    delete fes;

    delete pmesh;
    delete mesh;

    MPI_Finalize();

    return 0;
}

