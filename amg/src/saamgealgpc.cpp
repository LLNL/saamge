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

#include "saamgealgpc.hpp"
#include <memory>

namespace saamge
{

SAAMGeAlgPC::SAAMGeAlgPC()
    : has_stuff_to_destroy(false)
{
    if (PROC_COMM == 0)
    {
        proc_init(MPI_COMM_WORLD);
    }

    InitDefaults();
}

SAAMGeAlgPC::~SAAMGeAlgPC()
{
    Destroy();
}

void SAAMGeAlgPC::Destroy()
{
    if (has_stuff_to_destroy)
    {
        has_stuff_to_destroy = false;
        tg_free_data(tg_data);
        agg_free_partitioning(agg_part_rels);
    }
}

void SAAMGeAlgPC::InitDefaults()
{
    num_levels = 2;
    elems_per_agg = 256;
    first_elems_per_agg = elems_per_agg;
    first_nu_pro = 0;
    nu_pro = 0.0;
    nu_relax = 3;
    theta = 0.003;
    first_theta = theta;
        
    polynomial_coarse = -1;
    correct_nulspace = false;
    direct_eigensolver = false;

    use_arpack = true;
    window_amg = false;
    minimal_coarse = false;
}

void SAAMGeAlgPC::Print(std::ostream &os) const
{
    using std::endl;
    using std::setw;

    os << setw(10);
    os << "num_levels:\t\t"             << num_levels           << endl;
    os << "nparts_arr[0]:\t\t"          << nparts_arr[0]        << endl;
    os << "first_nu_pro:\t\t"           << first_nu_pro         << endl;
    os << "nu_pro:\t\t\t"               << nu_pro               << endl;
    os << "nu_relax:\t\t"               << nu_relax             << endl;
    os << "first_theta:\t\t"            << first_theta          << endl;
    os << "theta:\t\t\t"                << theta                << endl;
    os << "polynomial_coarse:\t"        << polynomial_coarse    << endl;
    os << "correct_nulspace:\t"         << correct_nulspace     << endl;
    os << "direct_eigensolver:\t"       << direct_eigensolver   << endl;
}

bool SAAMGeAlgPC::Make(const std::shared_ptr<mfem::HypreParMatrix> &A,
                       mfem::SparseMatrix &Al)
{
    using namespace std;
    using namespace mfem;

    SA_ASSERT(!(correct_nulspace && minimal_coarse));

    Destroy();
    nparts_arr.resize(num_levels-1);

    nparts_arr[0] = Al.Height() / first_elems_per_agg;
    shared_ptr<SparseMatrix> identity(IdentitySparseMatrix(Al.Height()));

    int drow_starts[2];
    drow_starts[0] = 0;
    drow_starts[1] = Al.Height();
        
    auto dof_truedof = make_shared<HypreParMatrix>(
        PROC_COMM, identity->Height(), drow_starts, identity.get());
        
    Array<int> isolated_cells(0);

    if (window_amg)
        TestWindowSubMatrices();

    agg_part_rels = fem_create_partitioning_from_matrix(
        Al, &nparts_arr[0], dof_truedof.get(), isolated_cells);

    for (int i = 1; i < num_levels - 1; ++i)
    {
        nparts_arr[i] = (int) round((double) nparts_arr[i-1] /
                                    (double) elems_per_agg);
        if (nparts_arr[i] < 1) nparts_arr[i] = 1;
    }
        
    const bool avoid_ess_bdr_dofs = true;
    int polynomial_coarse;
    if (minimal_coarse)
        polynomial_coarse = 0;
    else
        polynomial_coarse = -1;
        
    tg_data = tg_produce_data_algebraic(
        Al, *A, *agg_part_rels, first_nu_pro, nu_relax, first_theta,
        (nu_pro > 0), polynomial_coarse, window_amg, use_arpack,
        avoid_ess_bdr_dofs);


    tg_fillin_coarse_operator(*A, tg_data, false);
    tg_data->coarse_solver = new AMGSolver(*tg_data->Ac, false);

    Bprec = make_shared<VCycleSolver>(tg_data, false);
    Bprec->SetOperator(*A);

    this->height = A->Height();
    this->width  = A->Width();

    has_stuff_to_destroy = true;
    return false;
}

void SAAMGeAlgPC::Mult(const mfem::Vector &x, mfem::Vector &y) const
{
    Bprec->Mult(x, y);
}

void SAAMGeAlgPC::MultTranspose(const mfem::Vector &x, mfem::Vector &y) const
{
    Bprec->MultTranspose(x, y);
}

} // namespace saamge
