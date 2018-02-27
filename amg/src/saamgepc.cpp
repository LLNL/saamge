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

#include "saamgepc.hpp"
#include <memory>

namespace saamge
{

SAAMGePC::SAAMGePC(const std::shared_ptr<mfem::ParFiniteElementSpace> &fe, 
                   mfem::Array<int> &ess_bdr)
    : fe(fe),  has_stuff_to_destroy(false)
{
    if (PROC_COMM == 0)
    {
        proc_init(MPI_COMM_WORLD);
    }
    this->ess_bdr.MakeRef(ess_bdr);
    InitDefaults();
}

SAAMGePC::SAAMGePC()
    : has_stuff_to_destroy(false)
{
    InitDefaults();
}

SAAMGePC::~SAAMGePC()
{
    Destroy();
}

void SAAMGePC::Destroy()
{
    if (has_stuff_to_destroy)
    {
        has_stuff_to_destroy = false;

        ml_free_data(ml_data);
        agg_free_partitioning(agg_part_rels);
    }
}

void SAAMGePC::InitDefaults()
{
    num_levels = 3;
    elems_per_agg = 64;
    first_elems_per_agg = 128;
    do_aggregates = false;
    first_nu_pro = 0;
    nu_pro = 1.0;
    nu_relax = 3;
    theta = 0.003;
    first_theta = theta;
    polynomial_coarse = -1;
    correct_nulspace = false;
    direct_eigensolver = false;
}

// void SAAMGePC::AutoInit(const HypreParMatrix &A)
// {
//      // fe->
//      // nnz 

//      num_levels = 3;                         //log_2(A.Height())
//      first_elems_per_agg = 128;      //larger eigs more precomputaiton (play with theta for optimal)
//      elems_per_agg = 64;                     //...

//      nu_pro = 1;                                     //2 is expensive but better coarse space
//      first_nu_pro = 0;                       //...

//      theta = 0.003;                          //0 theta for 1 eigen vector (does not make sense to use it)
//      first_theta = theta;            //...

//      do_aggregates = false; 
//      nu_relax = 2;                           //V Cycle smoothing steps
        
//      polynomial_coarse = 1;          //-1 does not use it, 1 for linear
//      correct_nulspace = false;       //leave it false
//      direct_eigensolver = false;
// }

void SAAMGePC::Print(std::ostream &os) const
{
    using std::endl;
    using std::setw;

    os << setw(10);
    os << "num_levels:\t\t"        << num_levels           << endl;
    os << "nparts_arr[0]:\t\t"     << nparts_arr[0]        << endl;
    os << "first_nu_pro:\t\t"      << first_nu_pro         << endl;
    os << "nu_pro:\t\t\t"          << nu_pro               << endl;
    os << "nu_relax:\t\t"          << nu_relax             << endl;
    os << "first_theta:\t\t"       << first_theta          << endl;
    os << "theta:\t\t\t"           << theta                << endl;
    os << "theta:\t\t\t"           << theta                << endl;
    os << "polynomial_coarse:\t"   << polynomial_coarse    << endl;
    os << "correct_nulspace:\t"    << correct_nulspace     << endl;
    os << "direct_eigensolver:\t"  << direct_eigensolver   << endl;
    os << "do_aggregates:\t\t"     << do_aggregates        << endl;
}

bool SAAMGePC::Make(const std::shared_ptr<mfem::ParBilinearForm> &a,
                    const std::shared_ptr<mfem::HypreParMatrix> &A,
                    mfem::SparseMatrix &Al)
{
    using namespace std;
    using namespace mfem;

    Destroy();

    auto pmesh = fe->GetMesh();
    agg_dof_status_t *bdr_dofs = fem_find_bdr_dofs(*fe, &ess_bdr);

    nparts_arr.resize(num_levels); 
    std::fill(nparts_arr.begin(), nparts_arr.end(), 0);             
    nparts_arr[0] = pmesh->GetNE() / first_elems_per_agg;

    const bool do_aggregates_here = do_aggregates && (num_levels == 2);
    agg_part_rels = fem_create_partitioning(
        *A, *fe, bdr_dofs, &nparts_arr[0], do_aggregates_here);
        
    for (int i=1; i < num_levels-1; ++i)
    {
        nparts_arr[i] = (int) round((double) nparts_arr[i-1] /
                                    (double) elems_per_agg);
        if (nparts_arr[i] < 1) nparts_arr[i] = 1;
    }

    MultilevelParameters mlp(
        num_levels-1, &nparts_arr[0], first_nu_pro, nu_pro, nu_relax, first_theta,
        theta, polynomial_coarse, correct_nulspace, !direct_eigensolver,
        do_aggregates);
    Print();
    emp = new ElementMatrixStandardGeometric(*agg_part_rels, Al, a.get());
    ml_data = ml_produce_data(*A, agg_part_rels, emp, mlp);

    levels_level_t * level = levels_list_get_level(ml_data->levels_list, 0);
    Bprec = make_shared<VCycleSolver>(level->tg_data, false);
    Bprec->SetOperator(*A);

    this->height = A->Height();
    this->width  = A->Width();

    has_stuff_to_destroy = true;
    return true;
}

void SAAMGePC::Mult(const mfem::Vector &x, mfem::Vector &y) const
{
    Bprec->Mult(x, y);
}

void SAAMGePC::MultTranspose(const mfem::Vector &x, mfem::Vector &y) const
{
    Bprec->MultTranspose(x, y);
}

} // namespace saamge
