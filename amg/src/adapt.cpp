/*! \file

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

#include "common.hpp"
#include "adapt.hpp"
#include <cfloat>
#include <cmath>
#include <mfem.hpp>
#include "tg.hpp"
#include "mbox.hpp"
#include "helpers.hpp"
using std::pow;
using std::sqrt;

namespace saamge
{
using namespace mfem;

/* Functions */

int adapt_approx_xbad(HypreParMatrix& A,
    const agg_partitioning_relations_t& agg_part_rels, int maxiter,
    tg_data_t *tg_data, HypreParVector& xbad, double *cf, double *acf,
    double *err, double *err0, int *executed_iters, double rtol, double atol,
    bool normalize, bool output/*=true*/)
{
    SA_ASSERT(A.GetGlobalNumRows() == A.GetGlobalNumCols());
    const int n = mbox_rows_in_current_process(A);
    const int iters = *executed_iters;
    bool exit_flag = 0;
    int i, reason = 0;
    double err_, err0_=0., err_prev, cf_ = HUGE_VAL, l2_err = 0.;
    double ende, acf_=0.;
    Vector lb(n);
    lb = 0.;
    HypreParVector b(PROC_COMM, A.GetGlobalNumRows(), lb.GetData(), A.GetRowStarts());

    SA_ASSERT(iters >= 0);
    SA_ASSERT(maxiter > 0);
    SA_ASSERT(xbad.Size() == n);

    if (0 == iters)
        helpers_random_vect(agg_part_rels, xbad);

    err_ = mbox_energy_norm_parallel(A, xbad);
    if (0 != iters)
    {
        err0_ = *err0;
        acf_ = pow(err_/err0_, 1./(double)iters);
    }

    if (normalize || 0 == iters)
    {
        xbad /= err_;
        err_ = 1.;
    }

    if (0 == iters)
        err0_ = err_;

    if ((ende = rtol * err_) < atol)
        ende = atol;
    SA_ASSERT(ende >= 0.);
    if (SA_IS_OUTPUT_LEVEL(2) && output)
    {
        SA_PRINTF("err0: %g\n", err0_);
        SA_PRINTF("aimed error: %g\n", ende);
    }
    for (i=1; ; ++i)
    {
        if (SA_IS_OUTPUT_LEVEL(3) && output)
            l2_err = sqrt(mbox_parallel_inner_product(xbad, xbad));
        // if (SA_IS_OUTPUT_LEVEL(2) && output && 0 == PROC_RANK)
        if (output && 0 == PROC_RANK)
        {
            PROC_STR_STREAM << "Stationary iteration: after " << i-1
                            << " iterations";
            if (0 != iters) PROC_STR_STREAM << ", total: " << i + iters - 1;
            PROC_STR_STREAM << " | err: " << err_ << ", cf: ";
            if (i>1)
                PROC_STR_STREAM << cf_;
            else
                PROC_STR_STREAM << "wait";
            PROC_STR_STREAM << ", acf: ";
            if (i>1 || 0 != iters) 
                PROC_STR_STREAM << acf_;
            else
                PROC_STR_STREAM << "wait";
            if (SA_IS_OUTPUT_LEVEL(3))
                PROC_STR_STREAM << ", l2 norm err: " << l2_err;
            PROC_STR_STREAM << "\n";
            SA_PRINTF("%s", PROC_STR_STREAM.str().c_str());
            PROC_CLEAR_STR_STREAM;
        }

        if (err_ <= ende)
        {
            SA_SET_FLAGS(reason, ADAPT_XBAD_ERR_TOL_FLAG);
            exit_flag = 1;
        }

        if (i > maxiter)
        {
            SA_SET_FLAGS(reason, ADAPT_XBAD_MAX_ITER_FLAG);
            exit_flag = 1;
        }

        if (exit_flag)
        {
            *cf = cf_;
            *acf = acf_;
            *err = err_;
            *err0 = err0_;
            *executed_iters = i-1;
            SA_ASSERT(reason);
            return reason;
        }

        err_prev = err_;

        tg_cycle_atb(A, *(tg_data->Ac), *(tg_data->interp), *(tg_data->restr), b,
                     tg_data->pre_smoother, tg_data->post_smoother, xbad,
                     *(tg_data->coarse_solver), tg_data->poly_data);

        err_ = mbox_energy_norm_parallel(A, xbad);
        cf_ = err_/err_prev;
        acf_ = pow(err_/err0_, 1./(double)(i + iters));

        if (normalize)
        {
            xbad /= err_;
            err_ = 1.;
        }

        if (err_ > err_prev)
        {
            SA_SET_FLAGS(reason, ADAPT_XBAD_ERR_INC_FLAG);
            exit_flag = 1;
        }
    }
}

void adapt_update_operators(HypreParMatrix& A, tg_data_t &tg_data, bool resmooth_interp)
{
    SA_ASSERT(tg_data.poly_data);
    SA_ASSERT(smpr_get_Dinv_neg(tg_data.poly_data));
    SA_ASSERT(tg_data.interp);
    SA_ASSERT(tg_data.restr);
    SA_ASSERT(tg_data.interp_data);

    smpr_update_Dinv_neg(A, tg_data.poly_data);

    if (resmooth_interp && tg_data.smooth_interp &&
        tg_data.interp_data->interp_smoother_degree > 0 &&
        tg_data.interp_data->times_apply_smoother > 0)
        tg_smooth_interp(A, tg_data);

    tg_free_coarse_operator(tg_data);
}

void adapt_update_operators(HypreParMatrix& A, ml_data_t &ml_data,
                            const MultilevelParameters &mlp, bool resmooth_interp)
{
    SA_ASSERT(ml_data.levels_list.num_levels > 0);
    SA_ASSERT(ml_data.levels_list.finest);
    SA_ASSERT(ml_data.levels_list.finest->tg_data);

    adapt_update_operators(A, *ml_data.levels_list.finest->tg_data, resmooth_interp);

    tg_update_coarse_operator(A, ml_data.levels_list.finest->tg_data,
                              NULL == ml_data.levels_list.finest->coarser, mlp.get_coarse_direct());
    HypreParMatrix *Af = ml_data.levels_list.finest->tg_data->Ac;
    levels_level_t *level;
    for(level = ml_data.levels_list.finest->coarser; level; level = level->coarser)
    {
        SA_ASSERT(Af);
        SA_ASSERT(level->tg_data);
        SA_ASSERT(level->tg_data->poly_data);
        smpr_update_Dinv_neg(*Af, level->tg_data->poly_data);
        if (resmooth_interp && level->tg_data->smooth_interp &&
            level->tg_data->interp_data->interp_smoother_degree > 0 &&
            level->tg_data->interp_data->times_apply_smoother > 0)
            tg_smooth_interp(*Af, *level->tg_data);
        tg_update_coarse_operator(*Af, level->tg_data, NULL == level->coarser, mlp.get_coarse_direct());
        Af = level->tg_data->Ac;
    }
    SA_ASSERT(Af);
    ml_impose_cycle(ml_data, false);
}

} // namespace saamge
