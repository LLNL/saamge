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

#ifndef LS_HELMHOLTZ_HPP
#define LS_HELMHOLTZ_HPP 

#include <mfem.hpp>
#include <memory>

namespace saamge
{

class LSHelmholtzProblem
{
public:
    LSHelmholtzProblem(const bool eliminate_bc_dofs = true) : dims(0), eliminate_bc_dofs(eliminate_bc_dofs) {}

    /*! @brief Sets-up the following system:
     *  | M B^T | 
     *  | B G   |
     *
     *  @todo this should just be part of the constructor
     */
    void Init(
        const std::shared_ptr<mfem::ParFiniteElementSpace> &U_space, 
        const std::shared_ptr<mfem::ParFiniteElementSpace> &W_space,
        const double k_local,
        const double beta_val);

    /// @warning it only works in serial 
    void MakeMonolithic(std::shared_ptr<mfem::HypreParMatrix> &mat, 
                        std::shared_ptr<mfem::Vector> &rhs,
                        std::shared_ptr<mfem::SparseMatrix> &mat_l);

    /// @warning it only works in serial 
    void RecoverSolution(const mfem::Vector &v, mfem::ParGridFunction &u,
                         mfem::ParGridFunction &gradu);


    /// @warning it only works in serial and if U_space and W_space have same order
    void LocalBlocksMakeMonolithic(std::shared_ptr<mfem::HypreParMatrix> &mat, 
                                   std::shared_ptr<mfem::Vector> &rhs,
                                   std::shared_ptr<mfem::SparseMatrix> &mat_l);

    /// @warning it only works in serial and if U_space and W_space have same order
    void LocalBlocksRecoverSolution(const mfem::Vector &v,
                                    mfem::ParGridFunction &u,
                                    mfem::ParGridFunction &gradu);

    /*!
     * @brief Transforms the following BVP system
     * [A, B; 0, Id] * [u; u_b] = [f; g] into A * u = f + (B * g)
     */
    void EliminateBCDOFs(std::shared_ptr<mfem::HypreParMatrix> &mat, 
                         std::shared_ptr<mfem::Vector>         &rhs,
                         std::shared_ptr<mfem::SparseMatrix>   &mat_l);
    void RecoverBCDOFs(const mfem::Vector &x_sol,
                       mfem::Vector &x_sol_with_bc) const;

    // Yes all public 
    std::shared_ptr<mfem::ConstantCoefficient> c;
    std::shared_ptr<mfem::ConstantCoefficient> c_squared;
    std::shared_ptr<mfem::ConstantCoefficient> f;
    std::shared_ptr<mfem::ConstantCoefficient> f_nat;
    std::shared_ptr<mfem::ConstantCoefficient> f_times_c;
    std::shared_ptr<mfem::ConstantCoefficient> beta;

    std::shared_ptr<mfem::ParGridFunction> x;

    std::shared_ptr<mfem::ParLinearForm> f_form;
    std::shared_ptr<mfem::ParLinearForm> f_dot_diff_form;

    std::shared_ptr<mfem::ParBilinearForm> u_bf;
    std::shared_ptr<mfem::ParBilinearForm> gradu_bf;

    std::shared_ptr<mfem::ParMixedBilinearForm> mixed_bf;

    mfem::Array<int> ess_bdr;

    mfem::SparseMatrix *M_spMat;
    mfem::SparseMatrix *G_spMat;

    std::shared_ptr<mfem::HypreParMatrix> M;
    std::shared_ptr<mfem::HypreParMatrix> B;
    std::shared_ptr<mfem::HypreParMatrix> BT;
    std::shared_ptr<mfem::HypreParMatrix> G;

private:
    int dims;
    // boundary handling
    bool eliminate_bc_dofs;
    mfem::Array<HYPRE_Int> dof_map;
    mfem::Array<HYPRE_Int> inv_dof_map;
    mfem::Array<bool> is_boundary;

    void CheckSerial(const std::string &function, const int line) const;
};

} // namespace saamge

#endif //LS_HELMHOLTZ_HPP
