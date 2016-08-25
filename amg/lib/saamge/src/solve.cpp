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

#include "common.hpp"
#include "solve.hpp"
#include <mfem.hpp>
#include "xpacks.hpp"
#include "mbox.hpp"
#include "tg.hpp"
#include "mfem_addons.hpp"

/* Options */

CONFIG_DEFINE_CLASS(SOLVE);

/* Functions */

void solve_empty(HypreParMatrix& A, HypreParVector& b, HypreParVector& x,
                 void *data)
{
    x = 0.;
}

void *solve_spd_AMG_init(HypreParMatrix& A)
{
    SA_ASSERT(CONFIG_ACCESS_OPTION(SOLVE, rtol) >= 0.);
    SA_ASSERT(CONFIG_ACCESS_OPTION(SOLVE, iters_coeff) >= 0.);

    solve_amg_t *data = new solve_amg_t;
    data->amg = new HypreBoomerAMG(A);
    data->pcg = new HyprePCG(A);
    data->pcg->SetTol(CONFIG_ACCESS_OPTION(SOLVE, rtol));
    data->pcg->SetMaxIter((int)(CONFIG_ACCESS_OPTION(SOLVE, iters_coeff)*
                                A.GetGlobalNumRows()));
    data->pcg->SetPrintLevel(SA_IS_OUTPUT_LEVEL(15) ? 2 : 0);
    data->pcg->SetPreconditioner(*(data->amg));
    data->ref_cntr = 1;

    return (void *)data;
}

void *solve_spd_AMG_free(HypreParMatrix& A, void *data)
{
    if (!data)
        return NULL;

    solve_amg_t *ldata = (solve_amg_t *)data;
    SA_ASSERT(ldata->ref_cntr > 0);
    --(ldata->ref_cntr);
    if (ldata->ref_cntr <= 0)
    {
        delete ldata->pcg;
        delete ldata->amg;
        delete ldata;
    }

    return NULL;
}

void *solve_spd_AMG_copy(HypreParMatrix& A, void *data)
{
    if (!data)
        return NULL;

    solve_amg_t *ldata = (solve_amg_t *)data;
    SA_ASSERT(ldata->ref_cntr > 0);
    ++(ldata->ref_cntr);

    return data;
}

void solve_spd_AMG(HypreParMatrix& A, HypreParVector& b, HypreParVector& x,
                   void *data)
{
    SA_ASSERT(data);

    x = 0.;
    ((solve_amg_t *)data)->pcg->Mult(b, x);
}
