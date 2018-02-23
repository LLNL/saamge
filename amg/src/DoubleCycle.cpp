/*! \file

    SAAMGE: smoothed aggregation element based algebraic multigrid hierarchies
            and solvers.

    Copyright (c) 2016, Lawrence Livermore National Security,
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

#include "DoubleCycle.hpp"

using namespace mfem;

DoubleCycle::DoubleCycle(HypreParMatrix& A, ml_data_t &ml_data) :
    Solver(A.M(),false), // PCG does not give me an initial guess when this is a preconditioner?
    A(A)
{
    tg_data_t &tg_data = *ml_data.levels_list.finest->tg_data;
    Ac = tg_data.Ac;
    interp = tg_data.interp;
    restr = tg_data.restr;
    pre_smoother = tg_data.pre_smoother;
    post_smoother = tg_data.post_smoother;
    smoother_data = (void*) tg_data.poly_data;

    inner_solver = new CorrectNullspace(*Ac, tg_data.scaling_P, 2, false, true, false);
    outer_solver = new VCycleSolver(ml_data.levels_list.finest->coarser->tg_data, false);
    outer_solver->SetOperator(*Ac);
}

DoubleCycle::~DoubleCycle()
{
    delete inner_solver;
    delete outer_solver;
}

void DoubleCycle::Mult(const Vector &b, Vector &x) const
{
    // SA_RPRINTF(0,"--> DoubleCycle:Mult runs with size %d\n",A.M());

    x = 0.0; // seems I need to do this even if iterative_mode = true;

    Vector res(x.Size()), resc(mbox_rows_in_current_process(*Ac));
    Vector xc(mbox_rows_in_current_process(*Ac));
    xc = 0.0; // moved out of coarse_solver.solver for W-cycle

    pre_smoother(A, b, x, smoother_data);

    A.Mult(x, res);
    subtract(b, res, res);
    restr->Mult(res, resc);

    HypreParVector RESC(PROC_COMM, Ac->GetGlobalNumRows(), resc.GetData(),
                        Ac->GetRowStarts());
    HypreParVector XC(PROC_COMM, Ac->GetGlobalNumRows(), xc.GetData(), Ac->GetRowStarts());

    outer_solver->Mult(RESC, XC);

    HypreParVector RESC2(RESC);
    Ac->Mult(XC, RESC2);
    subtract(RESC, RESC2, RESC2);
    inner_solver->Mult(RESC2, XC);

    bool symmetric = true;
    if (symmetric)
    {
        Ac->Mult(XC, RESC);
        subtract(RESC2, RESC, RESC);
        outer_solver->Mult(RESC, XC);
    }

    interp->Mult(1.0, XC, 1.0, x);

    post_smoother(A, b, x, smoother_data);    
}
