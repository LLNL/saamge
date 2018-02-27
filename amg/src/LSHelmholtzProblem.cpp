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

#include "LSHelmholtzProblem.hpp"
#include "SecondOrderEllipticIntegrator.hpp"

namespace saamge
{

void LSHelmholtzProblem::Init(
    const std::shared_ptr<mfem::ParFiniteElementSpace> &U_space, 
    const std::shared_ptr<mfem::ParFiniteElementSpace> &W_space,
    const double k_local,
    const double beta_val)
{

    using namespace mfem;
    using namespace std;

    bool verbose = true;
    dims = U_space->GetMesh()->Dimension();

    HYPRE_Int dimU = U_space->GlobalTrueVSize();
    HYPRE_Int dimW = W_space->GlobalTrueVSize();

    int myid;
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    if (verbose && myid == 0)
    {
        std::cout << "*******************************************************\n";
        std::cout << "dim(R) = " << dimU << "\n";
        std::cout << "dim(W) = " << dimW << "\n";
        std::cout << "dim(R+W) = " << dimU + dimW << "\n";
        std::cout << "*******************************************************\n";
    }

    //Parameters
    double c_val = k_local;
    double f_val = 0.5;

    c_squared = make_shared<ConstantCoefficient>(c_val*c_val);
    c         = make_shared<ConstantCoefficient>(c_val);
    f         = make_shared<ConstantCoefficient>(f_val);
    f_nat     = make_shared<ConstantCoefficient>(0);
    f_times_c = make_shared<ConstantCoefficient>(f_val * c_val);
    beta      = make_shared<ConstantCoefficient>(beta_val);

    x = make_shared<ParGridFunction>(U_space.get());
    (*x) = 0.0;

    ///////////////////////////////////////////////////////////////
    // Create rhs blocks

    f_form = make_shared<ParLinearForm>(U_space.get());
    f_form->AddDomainIntegrator(new DomainLFIntegrator(*f_times_c));
    f_form->Assemble();

    f_dot_diff_form = make_shared<ParLinearForm>(W_space.get());
    f_dot_diff_form->AddDomainIntegrator(new DivDomainLFIntegrator(*f));
    f_dot_diff_form->Assemble();

    ///////////////////////////////////////////////////////////////
    // Create operator blocks 

    u_bf = make_shared<ParBilinearForm>(U_space.get());
    u_bf->AddDomainIntegrator(new DiffusionIntegrator());
    u_bf->AddDomainIntegrator(new MassIntegrator(*c_squared));
    u_bf->Assemble();

    gradu_bf = make_shared<ParBilinearForm>(W_space.get());
    gradu_bf->AddDomainIntegrator(new VectorDivDivIntegrator());
    gradu_bf->AddDomainIntegrator(new VectorMassIntegrator());
    gradu_bf->AddDomainIntegrator(new VectorCurlCurlIntegrator(*beta));
    gradu_bf->Assemble();

    mixed_bf = make_shared<ParMixedBilinearForm>(U_space.get(), W_space.get());
    mixed_bf->AddDomainIntegrator(new Vector2ndOrderEllipticIntegratorMixed(*c));
    mixed_bf->Assemble();

    if (U_space->GetMesh()->bdr_attributes.Size())
    {
        ess_bdr.SetSize(U_space->GetMesh()->bdr_attributes.Max());
        ess_bdr = 1;
        Array<int> ess_dof;
        U_space->GetEssentialVDofs(ess_bdr, ess_dof);
        mixed_bf->EliminateEssentialBCFromTrialDofs(ess_dof, *x, *f_dot_diff_form);
        u_bf->EliminateEssentialBC(ess_bdr);
    }

    u_bf->Finalize();
    mixed_bf->Finalize();
    gradu_bf->Finalize();

    ///////////////////////////////////////////////////////////////
    //FIXME in SAAMGE this semantical dependency might be confusing
    M_spMat = &u_bf->SpMat();
    G_spMat = &gradu_bf->SpMat();
    ///////////////////////////////////////////////////////////////

    M  = shared_ptr<HypreParMatrix>(u_bf->ParallelAssemble());
    B  = shared_ptr<HypreParMatrix>(mixed_bf->ParallelAssemble());
    BT = shared_ptr<HypreParMatrix>(B->Transpose());
    G  = shared_ptr<HypreParMatrix>(gradu_bf->ParallelAssemble());
}

void LSHelmholtzProblem::MakeMonolithic(
    std::shared_ptr<mfem::HypreParMatrix> &mat, 
    std::shared_ptr<mfem::Vector>         &rhs,
    std::shared_ptr<mfem::SparseMatrix>   &mat_l)
{
    using namespace std;
    using namespace mfem;

    CheckSerial(__FUNCTION__, __LINE__);

    bool sort_columns = false;

    HYPRE_Int dimU = M->GetNumRows();
    HYPRE_Int dimW = G->GetNumRows();
    HYPRE_Int nnz = M->NNZ() + BT->NNZ() + B->NNZ() + G->NNZ();

    vector<HYPRE_Int> I(dimU + dimW + 1, 0);
    vector<HYPRE_Int> J;
    J.reserve(nnz);

    vector<double> data;
    data.reserve(nnz);

    ///////////////////////////////////////
    //upper blocks
    SparseMatrix M_diag_block, BT_diag_block;
    M->GetDiag(M_diag_block);
    BT->GetDiag(BT_diag_block);

    if (sort_columns)
    {
        M_diag_block.SortColumnIndices();
        BT_diag_block.SortColumnIndices();
    }

    for (int i = 0; i < M_diag_block.Height(); ++i)
    {
        HYPRE_Int M_row_size  = M_diag_block.RowSize(i);
        HYPRE_Int BT_row_size = BT_diag_block.RowSize(i);

        I[i + 1] = I[i] + M_row_size + BT_row_size;  

        const int * col_index  = M_diag_block.GetRowColumns(i);
        const double * entries = M_diag_block.GetRowEntries(i);
      
        for (int j = 0; j < M_row_size; ++j) {
            J.push_back(col_index[j]);
            data.push_back(entries[j]);
        }

        col_index  = BT_diag_block.GetRowColumns(i);
        entries    = BT_diag_block.GetRowEntries(i);

        for (int j = 0; j < BT_row_size; ++j) {
            J.push_back(dimU + col_index[j]);
            data.push_back(entries[j]);
        }
    }

    ///////////////////////////////////////
    //lower blocks
    SparseMatrix B_diag_block, G_diag_block;
    B->GetDiag(B_diag_block);
    G->GetDiag(G_diag_block);

    if (sort_columns)
    {
        B_diag_block.SortColumnIndices();
        G_diag_block.SortColumnIndices();
    }

    for (int i = 0; i < G_diag_block.Height(); ++i)
    {
        HYPRE_Int B_row_size = B_diag_block.RowSize(i);
        HYPRE_Int G_row_size = G_diag_block.RowSize(i);

        I[dimU + i + 1] += I[dimU + i] + B_row_size + G_row_size;  

        const int * col_index  = B_diag_block.GetRowColumns(i);
        const double * entries = B_diag_block.GetRowEntries(i);

        for (int j = 0; j < B_row_size; ++j)
        {
            J.push_back(col_index[j]);
            data.push_back(entries[j]);
        }

        col_index  = G_diag_block.GetRowColumns(i);
        entries = G_diag_block.GetRowEntries(i);

        for (int j = 0; j < G_row_size; ++j)
        {
            J.push_back(dimU + col_index[j]);
            data.push_back(entries[j]);
        }
    }

    ///////////////////////////////////////

    assert(data.size() == (unsigned int) nnz);
    assert(J.size() == (unsigned int) nnz);

    const HYPRE_Int rows = I.size() - 1;

    HYPRE_Int offset[2] =  { 0, rows };
    vector<HYPRE_Int> offsets(I.size(), 0);

    mat = make_shared<HypreParMatrix>(
        MPI_COMM_WORLD,
        rows,
        rows,
        rows,
        &I[0],
        &J[0],
        &data[0],
        offset,
        offset 
        ); 

    ///////////////////////////////////////
    ///////////////////////////////////////

    Vector v1(dimU), v2(dimW);

    f_form->ParallelAssemble(v1);
    f_dot_diff_form->ParallelAssemble(v2);

    rhs = make_shared<Vector>(dimU + dimW);

    for (HYPRE_Int i = 0; i < dimU; ++i)
    {
        rhs->Elem(i) = v1.Elem(i);
    }

    for (HYPRE_Int i = 0; i < dimW; ++i)
    {
        rhs->Elem(dimU + i) = v2.Elem(i);
    }

    mat_l = make_shared<SparseMatrix>();
    mat->GetDiag(*mat_l);

    if (sort_columns)
    {
        mat_l->SortColumnIndices();
    }

    if (eliminate_bc_dofs)
    {
        this->EliminateBCDOFs(mat, rhs, mat_l);
    }
}

void LSHelmholtzProblem::RecoverSolution(const mfem::Vector &v_no_bc,
                                         mfem::ParGridFunction &u,
                                         mfem::ParGridFunction &gradu)
{
    using namespace mfem;

    CheckSerial(__FUNCTION__, __LINE__);

    mfem::Vector v;
    if(eliminate_bc_dofs)
    {
        RecoverBCDOFs(v_no_bc, v);
    }
    else
    {
        v = *v_no_bc;
    }

    HYPRE_Int dimU = M->GetNumRows();
    HYPRE_Int dimW = G->GetNumRows();

    Vector v1(dimU), v2(dimW);

    for (HYPRE_Int i = 0; i < dimU; ++i)
    {
        v1.Elem(i) = v.Elem(i);
    }

    for (HYPRE_Int i = 0; i < dimW; ++i)
    {
        v2.Elem(i) = v.Elem(dimU + i);
    }

    u.Distribute(v1);
    gradu.Distribute(v2);
}

/////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////

template<class It>
void Print(const It &b, const It &e, std::ostream &os) 
{
    for (It it = b; it != e; ++it)
    {
        os << *it << " ";
    }

    os << std::endl;
}

void LSHelmholtzProblem::LocalBlocksMakeMonolithic(
    std::shared_ptr<mfem::HypreParMatrix> &mat, 
    std::shared_ptr<mfem::Vector>         &rhs,
    std::shared_ptr<mfem::SparseMatrix>   &mat_l)
{
    using namespace std;
    using namespace mfem;

    CheckSerial(__FUNCTION__, __LINE__);

    bool sort_columns = false;

    HYPRE_Int dimU = M->GetNumRows();
    HYPRE_Int dimW = G->GetNumRows();
    HYPRE_Int dimTotal = dimU + dimW;
    HYPRE_Int nnz = M->NNZ() + BT->NNZ() + B->NNZ() + G->NNZ();
    assert(nnz >= 0);

    assert(dimU * dims == dimW && "Works only if they are of the same polynomial order");

    vector<HYPRE_Int> I(dimTotal + 1, 0);
    vector<HYPRE_Int> J;
    J.reserve(nnz);

    vector<double> data;
    data.reserve(nnz);

    ///////////////////////////////////////
    //upper blocks
    SparseMatrix M_diag_block, BT_diag_block;
    M->GetDiag(M_diag_block);
    BT->GetDiag(BT_diag_block);

    SparseMatrix B_diag_block, G_diag_block;
    B->GetDiag(B_diag_block);
    G->GetDiag(G_diag_block);

    if (sort_columns)
    {
        M_diag_block.SortColumnIndices();
        BT_diag_block.SortColumnIndices();
        B_diag_block.SortColumnIndices();
        G_diag_block.SortColumnIndices();
    }

    const int stride = dims + 1;
    HYPRE_Int index = 0;
    for (int i = 0; i < M_diag_block.Height(); ++i)
    {
        HYPRE_Int M_row_size  = M_diag_block.RowSize(i);
        HYPRE_Int BT_row_size = BT_diag_block.RowSize(i);

        I[index + 1] = I[index] + M_row_size + BT_row_size;  

        const int * col_index  = M_diag_block.GetRowColumns(i);
        const double * entries = M_diag_block.GetRowEntries(i);
      
        for (int j = 0; j < M_row_size; ++j)
        {
            J.push_back(stride * col_index[j]);
            assert(J.back() < dimTotal);
            data.push_back(entries[j]);
        }

        col_index  = BT_diag_block.GetRowColumns(i);
        entries    = BT_diag_block.GetRowEntries(i);

        for (int j = 0; j < BT_row_size; ++j)
        {
            const int block_j = col_index[j] / dims;
            const int entry_j = col_index[j] % dims;

            J.push_back(stride * block_j + entry_j + 1);
            data.push_back(entries[j]);
        }

        ///////////
        ++index;
        ///////////

        const HYPRE_Int ixd = i * dims;

        // index incremented here
        for (int d = 0; d < dims; ++d, ++index)
        {
            const HYPRE_Int k = ixd + d;

            HYPRE_Int B_row_size = B_diag_block.RowSize(k);
            HYPRE_Int G_row_size = G_diag_block.RowSize(k);

            I[index + 1] += I[index] + B_row_size + G_row_size;  

            const int * col_index  = B_diag_block.GetRowColumns(k);
            const double * entries = B_diag_block.GetRowEntries(k);

            for (int j = 0; j < B_row_size; ++j)
            {
                J.push_back(stride * col_index[j]);
                assert(J.back() < dimTotal);
                data.push_back(entries[j]);
            }

            col_index = G_diag_block.GetRowColumns(k);
            entries   = G_diag_block.GetRowEntries(k);

            for (int j = 0; j < G_row_size; ++j)
            {
                const int block_j = col_index[j] / dims;
                const int entry_j = col_index[j] % dims;

                J.push_back(stride * block_j + entry_j + 1);
                assert(J.back() < dimTotal);
                data.push_back(entries[j]);
            }
        }
    }

    ///////////////////////////////////////

    assert(data.size() == (unsigned int) nnz);
    assert(J.size() == (unsigned int) nnz);

    // std::cout << "dims: " << dims << "\n";
    // Print(I.begin(), I.end(), std::cout);
    // std::cout << "-------------------------\n"; 
    // Print(J.begin(), J.end(), std::cout);
    // Print(data.begin(), data.end(), std::cout);

    const HYPRE_Int rows = I.size() - 1;

    HYPRE_Int offset[2] =  { 0, rows };
    vector<HYPRE_Int> offsets(I.size(), 0);

    mat = make_shared<HypreParMatrix>(
        MPI_COMM_WORLD,
        rows,
        rows,
        rows,
        &I[0],
        &J[0],
        &data[0],
        offset,
        offset); 

    ///////////////////////////////////////
    ///////////////////////////////////////

    Vector v1(dimU), v2(dimW);

    f_form->ParallelAssemble(v1);
    f_dot_diff_form->ParallelAssemble(v2);

    rhs = make_shared<Vector>(dimU + dimW);

    index = 0;
    for (HYPRE_Int i = 0; i < dimU; ++i)
    {
        rhs->Elem(index++) = v1.Elem(i);

        const HYPRE_Int ixd = i * dims;
        for (int d = 0; d < dims; ++d)
        {
            rhs->Elem(index++) = v2.Elem(ixd + d);
        }
    }

    mat_l = make_shared<SparseMatrix>();
    mat->GetDiag(*mat_l);

    if (sort_columns)
    {
        mat_l->SortColumnIndices();
    }

    if (eliminate_bc_dofs)
    {
        this->EliminateBCDOFs(mat, rhs, mat_l);
    }
}

void LSHelmholtzProblem::LocalBlocksRecoverSolution(const mfem::Vector &v_no_bc,
                                                    mfem::ParGridFunction &u,
                                                    mfem::ParGridFunction &gradu)
{
    using namespace mfem;

    CheckSerial(__FUNCTION__, __LINE__);

    mfem::Vector v;
    if(eliminate_bc_dofs)
    {
        RecoverBCDOFs(v_no_bc, v);
    }
    else
    {
        v = v_no_bc;
    }

    HYPRE_Int dimU = M->GetNumRows();
    HYPRE_Int dimW = G->GetNumRows();

    Vector v1(dimU), v2(dimW);

    HYPRE_Int index = 0;
    for (HYPRE_Int i = 0; i < dimU; ++i)
    {
        v1.Elem(i) = v.Elem(index++);

        const HYPRE_Int ixd = i * dims;
        for(int d = 0; d < dims; ++d)
        {
            v2.Elem(ixd + d) = v.Elem(index++);
        }
    }

    u.Distribute(v1);
    gradu.Distribute(v2);
}


void LSHelmholtzProblem::CheckSerial(const std::string &function,
                                     const int line) const
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if(size > 1)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        if(rank == 0)
        {
            std::cerr << "LSHelmholtzProblem::" << function
                      << "(...) This functions does not support parallel calls yet. At:" << std::endl;

            std::cerr << __FILE__ << ": " << line << std::endl;
        }

        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
}


void LSHelmholtzProblem::RecoverBCDOFs(const mfem::Vector &x_sol,
                                       mfem::Vector &x_sol_with_bc) const
{
    x_sol_with_bc.SetSize(dof_map.Size());
    x_sol_with_bc = 0.0;

    for (HYPRE_Int i = 0; i < x->Size(); ++i)
    {
        x_sol_with_bc.Elem(i) = x->Elem(i);
    }

    for (HYPRE_Int i = 0; i < x_sol.Size(); ++i)
    {
        x_sol_with_bc.Elem(inv_dof_map[i]) = x_sol.Elem(i);
    }
}

void LSHelmholtzProblem::EliminateBCDOFs(
    std::shared_ptr<mfem::HypreParMatrix> &mat, 
    std::shared_ptr<mfem::Vector>         &rhs,
    std::shared_ptr<mfem::SparseMatrix>   &mat_l)
{
    using namespace std;
    using namespace mfem;

    CheckSerial(__FUNCTION__, __LINE__);

    dof_map.SetSize(mat->Height());
    inv_dof_map.SetSize(mat->Height());
    inv_dof_map = -1;

    is_boundary.SetSize(mat->Height());
    is_boundary = false;

    SparseMatrix diag;
    mat->GetDiag(diag);

    int n_rows = 0;
    for (int i = 0; i < diag.Height(); ++i)
    {
        HYPRE_Int row_size   = diag.RowSize(i);
        if (row_size == 0) continue;

        const int * col_index  = diag.GetRowColumns(i);

        if (row_size == 1 && col_index[0] == i)
        {
            //Diagonal matrix
            is_boundary[i] = true;
        }
        else
        {
            dof_map[i]            = n_rows;
            inv_dof_map[n_rows++] = i;
        }
    }

    const HYPRE_Int nnz = diag.Height() - n_rows + mat->NNZ();

    vector<HYPRE_Int> I(n_rows + 1, 0);
    vector<HYPRE_Int> J;
    J.reserve(nnz);

    vector<double> data;
    data.reserve(nnz);

    auto new_rhs = make_shared<mfem::Vector>(n_rows);
    (*new_rhs) = 0.0;

    for (int i = 0; i < diag.Height(); ++i)
    {
        HYPRE_Int row_size   = diag.RowSize(i);
        if(row_size == 0) continue;

        const int * col_index  = diag.GetRowColumns(i);
        const double * entries = diag.GetRowEntries(i);

        if (is_boundary[i])
        {
            continue;
        }

        const HYPRE_Int i_new = dof_map[i];

        new_rhs->Elem(i_new) += rhs->Elem(i);

        for (int k = 0; k < row_size; ++k)
        {
            const HYPRE_Int j = col_index[k];
            const double    d = entries[k];


            if (is_boundary[j])
            {
                new_rhs->Elem(i_new) += d * rhs->Elem(j);
            }
            else
            {
                ++I[i_new+1];
                J.push_back(dof_map[j]);
                data.push_back(d);
            }
        }

        I[i_new + 1] += I[i_new];
    }


    HYPRE_Int offset[2] =  { 0, n_rows };
    mat = make_shared<HypreParMatrix>(
        MPI_COMM_WORLD,
        n_rows,
        n_rows,
        n_rows,
        &I[0],
        &J[0],
        &data[0],
        offset,
        offset 
        ); 

    mat->GetDiag(*mat_l);
    rhs = new_rhs;
}

} // namespace saamge
