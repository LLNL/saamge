#include <mfem.hpp>
#include <mpi.h>
#include <saamge.hpp>
#include <phyagg.hpp>

using namespace mfem;
using namespace saamge;
using namespace phyagg;

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    proc_init(MPI_COMM_WORLD);

    ParLinearForm *b;
    ParBilinearForm *a;
    ParGridFunction x;
    int *partitioning = nullptr;

    OptionsParser args(argc, argv);

    int intervals = 50;
    args.AddOption(&intervals, "-i", "--intervals", 
                   "How many intervals in one direction for the mesh.");
    int order = 2;
    args.AddOption(&order, "-o", "--order",
                   "Polynomial order of finite element space.");
    int parts = 8;
    args.AddOption(&parts, "-p", "--parts",
                   "Number of partitions.");

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

    MPI_Barrier(PROC_COMM); // try to make MFEM's debug element orientation prints not mess up the parameters above

    Mesh mesh(intervals, intervals, Element::QUADRILATERAL, 1);
    ParMesh pmesh(MPI_COMM_WORLD, mesh);
    Array<int> ess_bdr(mesh.bdr_attributes.Max());
    ess_bdr = 0;
    H1_FECollection fec(order);
    ParFiniteElementSpace fes(&pmesh, &fec);
    ConstantCoefficient one(1.0);

    const double alpha = 0.0;
    const double coeff_eps = 1e-6;

    MatrixArrayCoefficient coeff(2);
    coeff.Set(0, 0, new ConstantCoefficient(coeff_eps +
                                             cos(alpha) * cos(alpha)));
    coeff.Set(0, 1, new ConstantCoefficient(sin(alpha) * cos(alpha)));
    coeff.Set(1, 0, new ConstantCoefficient(sin(alpha) * cos(alpha)));
    coeff.Set(1, 1, new ConstantCoefficient(coeff_eps +
                                             sin(alpha) * sin(alpha)));

    fem_build_discrete_problem(&fes, one, one, coeff, false, x, b, a, &ess_bdr);

    HypreParMatrix *A = a->ParallelAssemble();
    std::shared_ptr<const ParBilinearForm> bform(a);
    MFEMElementData eldata(bform);

    DenseMatrix nullspace(A->Width(), 1);
    nullspace = 1.0;
    {
    Vector vect;
    nullspace.GetColumnReference(0, vect);
    vect /= vect.Norml2();
    }

    HypreSmoother GS_(*A, HypreSmoother::GS, 1);
    Vector ww(A->Width());
//    Vector ff(A->Height());
//    ff = 0.0;
//    ww.Randomize();
//    SLI(*A, GS_, ff, ww, 1, 200, 0.0, 0.0);
    Smoother S_(*A, GS_, nullspace);
    S_.Smooth(ww, 200);
    {
    Vector r(A->Height());
    A->Mult(ww, r);
    std::cout << "A smoothness: " << r.Norml2() << std::endl;
    }
    x = ww;
    fem_parallel_visualize_gf(pmesh, x);

    SparseMatrix LDT;
    auto X = ConnStrengthMatrix(eldata, LDT, nullspace);
    SA_ASSERT(X->Height() == X->Width());
    std::cout << "X symmetry: " << X->IsSymmetric() << std::endl;

    Array<int> row_starts(3);
    row_starts[0] = 0;
    row_starts[1] = row_starts[2] = X->Height();
    HypreParMatrix Xp(MPI_COMM_WORLD, X->Height(), row_starts.GetData(), const_cast<SparseMatrix *>(X.get()));
    SA_ASSERT(Xp.Height() == Xp.Width());
    HypreSmoother GS(Xp, HypreSmoother::GS, 1);
    Vector w(X->Width());
//    Vector f(X->Height());
//    f = 0.0;
//    w.Randomize();
//    SLI(Xp, GS, f, w, 1, 200, 0.0, 0.0);
    Smoother S(Xp, GS, nullspace);

//    S.Smooth(w, 200);
    LDT.Mult(ww, w);
    w /= w.Norml2();

    {
    Vector r(X->Height());
    X->Mult(w, r);
    std::cout << "X smoothness: " << r.Norml2() << std::endl;
    }
//    w.Print();

    auto graph = NormalizedConnStrengthGraph(eldata, *X, w);
    SA_ASSERT(graph->Height() == graph->Width());
    std::cout << "Graph symmetry: " << graph->IsSymmetric() << std::endl;
//    graph->Print();
    Array<int> strengths;
    IntegeriseStrengths(*graph, strengths);
//    strengths.Print();

    Table graph_table;
    graph_table.SetIJ(std::const_pointer_cast<SparseMatrix>(graph)->GetI(),
                      std::const_pointer_cast<SparseMatrix>(graph)->GetJ(),
                      graph->Height());
    SA_ASSERT(graph_table.Size() == graph_table.Width());

//    const int * const I = graph->GetI();
//    const int * const J = graph->GetJ();
//    for (int i=0; i < graph->Height(); ++i)
//    {
//        Vector ic;
//        fem_get_element_center(pmesh, i, ic);
//        for (int j=I[i]; j < I[i+1]; ++j)
//        {
//            const int neigh = J[j];
//            Vector nc;
//            fem_get_element_center(pmesh, neigh, nc);
//            nc -= ic;
//            if (fabs(nc(0)) > fabs(nc(1)))
//                strengths[j] = 10;
//            else
//                strengths[j] = 1;
//        }
//    }

    partitioning = part_generate_partitioning(graph_table, NULL, &parts,
                                              strengths.GetData());
    graph_table.LoseData();

//    partitioning = fem_partition_mesh(pmesh, &parts);

    fem_parallel_visualize_partitioning(pmesh, partitioning, parts);

    delete A;
    delete [] partitioning;
    delete b;
    MPI_Finalize();
    return 0;
}
