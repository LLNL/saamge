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
*/

#include <mfem.hpp>
#include <mpi.h>
#include <saamge.hpp>

/* Parameters */

#define TIMES_REFINE            4

#define NUM_OF_PARTITIONS       100

#define NU_PRO                  3
#define NU_RELAX                3

#define THETA                   0.003

/* Program */

Mesh *mesh;
ParMesh *pmesh;
int *proc_partitioning;
ParGridFunction x;
ParLinearForm *b;
ParBilinearForm *a;
agg_partititoning_relations_t *agg_part_rels;
tg_data_t *tg_data;

double checkbard_coef(Vector& x)
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

int main(int argc, char *argv[])
{
    if (argc == 1) {
        cout << "\nUsage: " << argv[0] << " <mesh_file>\n" << endl;
        return 1;
    }

    // Initialize process related stuff.
    MPI_Init(&argc, &argv);
    proc_init(MPI_COMM_WORLD);

    // Read the mesh from the given mesh file.
    mesh = fem_read_mesh(argv[1]);

    // Serial mesh.
    SA_PRINTF("NV: %d, NE: %d\n", mesh->GetNV(), mesh->GetNE());

    // Parallel mesh and finite elements stuff.
    Array<int> ess_bdr(mesh->bdr_attributes.Max());
    ess_bdr = 1;
    ess_bdr[0] = 0; ///< South in 2D unit square. (Dirchlet, Nuemann) = (1, 0)
    ess_bdr[2] = 0; ///< North in 2D unit square. (Dirchlet, Nuemann) = (1, 0)

    FunctionCoefficient bdr_coeff(bdr_cond);
    FunctionCoefficient rhs(rhs_func);

    int nprocs = PROC_NUM;
    int nparts = NUM_OF_PARTITIONS;
    proc_partitioning = fem_partition_mesh(*mesh, &nprocs);
    if (0 == PROC_RANK)
        fem_serial_visualize_partitioning(*mesh, proc_partitioning);
    // XXX: The parallel mesh will own and deallocate the partitioning.
    pmesh = new ParMesh(MPI_COMM_WORLD, *mesh, proc_partitioning);
    fem_refine_mesh_times(TIMES_REFINE, *pmesh);

    FEM_ALLOC_FEC_FES(LinearFECollection, pmesh, fec, fes);
    SA_PRINTF("pNV: %d, pNE: %d, pND: %d, ND: %d\n", pmesh->GetNV(),
              pmesh->GetNE(), fes->GetNDofs(), fes->GlobalTrueVSize());

    FEM_ALLOC_PWC_FEC_FES(pmesh, cfec, cfes);
    FunctionCoefficient conduct_func(checkbard_coef);
    ParGridFunction conductivity(cfes);
    conductivity.ProjectCoefficient(conduct_func);
    GridFunctionCoefficient conduct_coeff(&conductivity);
    fem_parallel_visualize_gf(*pmesh, conductivity,
                              pmesh->Dimension() == 2?"jfR":"f");

    fem_build_discrete_problem(fes, rhs, bdr_coeff, conduct_coeff, true, x, b,
                               a, &ess_bdr);
    SparseMatrix& Al = a->SpMat();
    HypreParMatrix *Ag = a->ParallelAssemble();
    HypreParVector *bg = b->ParallelAssemble();
    HypreParVector *hxg = x.ParallelAverage();
    HypreParVector *sxg = x.ParallelAverage();
    HypreParVector *pxg = x.ParallelAverage();

    // Solver stuff.
    SA_RPRINTF(0, "%s", "\n");
    SA_RPRINTF(0, "%s", "\t\t\tSOLVING THE PROBLEM USING HYPRE:\n");
    SA_RPRINTF(0, "%s", "\n");
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
    fem_parallel_visualize_gf(*pmesh, x);

    agg_dof_status_t *bdr_dofs = fem_find_bdr_dofs(*fes, &ess_bdr);
    agg_part_rels = fem_create_partitioning(Al, *fes, bdr_dofs, &nparts);
    delete [] bdr_dofs;
    fem_parallel_visualize_partitioning(*pmesh, agg_part_rels->partitioning,
                                        nparts);
    fem_parallel_visualize_aggregates(fes, agg_part_rels->aggregates, nparts);
    tg_data = tg_produce_data(Al, *Ag, *agg_part_rels, NU_PRO, NU_RELAX, 0., a,
                              THETA);

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

    SA_RPRINTF(0, "%s", "\n");
    SA_RPRINTF(0, "%s", "\t\t\tRUNNING STATIONARY ITERATION WITH RANDOM INITIAL"
                        " GUESS AND ZERO R.H.S:\n");
    SA_RPRINTF(0, "%s", "\n");
    tg_run(*Ag, agg_part_rels, *sxg, *bg, 1000, 1e-12, 0., 1., tg_data, true);
    x = *sxg;
    fem_parallel_visualize_gf(*pmesh, x);

    SA_RPRINTF(0, "%s", "\n");
    SA_RPRINTF(0, "%s", "\t\t\tSOLVING THE PROBLEM USING PCG:\n");
    SA_RPRINTF(0, "%s", "\n");
    tg_pcg_run(*Ag, agg_part_rels, *pxg, *bg, 1000, 1e-12, 0., tg_data, false);
    x = *pxg;
    fem_parallel_visualize_gf(*pmesh, x);
    *pxg -= *hxg;
    x = *pxg;
    fem_parallel_visualize_gf(*pmesh, x);

    SA_RPRINTF(0, "%s", "\n");
    SA_RPRINTF(0, "%s", "\t\t\tRUNNING PCG WITH RANDOM INITIAL GUESS AND ZERO"
                        " R.H.S:\n");
    SA_RPRINTF(0, "%s", "\n");
    tg_pcg_run(*Ag, agg_part_rels, *pxg, *bg, 1000, 1e-12, 0., tg_data, true);
    x = *pxg;
    fem_parallel_visualize_gf(*pmesh, x);

    tg_free_data(tg_data);
    agg_free_partitioning(agg_part_rels);
    delete pxg;
    delete sxg;
    delete hxg;
    delete bg;
    delete Ag;
    delete a;
    delete b;
    FEM_FREE_FEC_FES(cfec, cfes);
    FEM_FREE_FEC_FES(fec, fes);
    delete pmesh;
    delete mesh;

    MPI_Finalize();

    return 0;
}
