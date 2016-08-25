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
#include "tg.hpp"
#include <mfem.hpp>
#include "smpr.hpp"
#include "solve.hpp"
#include "helpers.hpp"
#include "interp.hpp"
#include "adapt.hpp"
#include "mfem_addons.hpp"

/* Options */

CONFIG_BEGIN_CLASS_DEFAULTS(TG)
    CONFIG_DEFINE_OPTION_DEFAULT(calc_res, tg_calc_res_tgprod),
    CONFIG_DEFINE_OPTION_DEFAULT(recalc_res, tg_recalc_res_tgprod),
    CONFIG_DEFINE_OPTION_DEFAULT(pre_smoother, smpr_sym_poly),
    CONFIG_DEFINE_OPTION_DEFAULT(post_smoother, smpr_sym_poly),
    CONFIG_DEFINE_OPTION_DEFAULT(coarse_solver_solve_init, solve_spd_AMG_init),
    CONFIG_DEFINE_OPTION_DEFAULT(coarse_solver_solve_free, solve_spd_AMG_free),
    CONFIG_DEFINE_OPTION_DEFAULT(coarse_solver_solve_copy, solve_spd_AMG_copy),
    CONFIG_DEFINE_OPTION_DEFAULT(coarse_solver_solver, solve_spd_AMG),
    CONFIG_DEFINE_OPTION_DEFAULT(coarse_solver_data, NULL),
    CONFIG_DEFINE_OPTION_DEFAULT(smooth_interp, true)
CONFIG_END_CLASS_DEFAULTS

CONFIG_DEFINE_CLASS(TG);

/* Functions */

void tg_cycle(HypreParMatrix& A, HypreParMatrix& Ac, HypreParMatrix& interp,
              HypreParMatrix& restr, HypreParVector& b, smpr_ft pre_smoother,
              smpr_ft post_smoother, HypreParVector& x, solve_t& coarse_solver,
              void *data)
{
    SA_ASSERT(&A);
    SA_ASSERT(&Ac);
    SA_ASSERT(A.GetGlobalNumRows() == A.GetGlobalNumCols());
    SA_ASSERT(Ac.GetGlobalNumRows() == Ac.GetGlobalNumCols());
    SA_ASSERT(interp.GetGlobalNumRows() == A.GetGlobalNumRows());
    SA_ASSERT(restr.GetGlobalNumCols() == A.GetGlobalNumCols());
    SA_ASSERT(interp.GetGlobalNumRows() >= interp.GetGlobalNumCols());
    SA_ASSERT(interp.GetGlobalNumCols() == restr.GetGlobalNumRows());
    SA_ASSERT(restr.GetGlobalNumRows() == Ac.GetGlobalNumRows());
    SA_ASSERT(mbox_parallel_vector_size(b) == A.GetGlobalNumRows());
    SA_ASSERT(mbox_parallel_vector_size(x) == A.GetGlobalNumRows());
    SA_ASSERT(pre_smoother);
    SA_ASSERT(post_smoother);
    SA_ASSERT(coarse_solver.solver);

    Vector res(b.Size()), resc(mbox_rows_in_current_process(Ac));
    Vector xc(mbox_rows_in_current_process(Ac));

    pre_smoother(A, b, x, data);

    A.Mult(x, res);
    subtract(b, res, res);
    restr.Mult(res, resc);

    HypreParVector RESC(Ac.GetGlobalNumRows(), resc.GetData(),
                        Ac.GetRowStarts());
    HypreParVector XC(Ac.GetGlobalNumRows(), xc.GetData(), Ac.GetRowStarts());

    coarse_solver.solver(Ac, RESC, XC, coarse_solver.data);

    interp.Mult(XC, x, 1., 1.);

    post_smoother(A, b, x, data);
}

double tg_calc_res_tgprod(HypreParMatrix& A, HypreParVector& b,
                          HypreParVector& x, HypreParVector *& res,
                          HypreParVector *& psres, void *data)
{
    tg_data_t *tg_data = (tg_data_t *)data;
    SA_ASSERT(A.GetGlobalNumRows() == A.GetGlobalNumCols());
    SA_ASSERT(mbox_parallel_vector_size(b) == A.GetGlobalNumRows());
    SA_ASSERT(mbox_parallel_vector_size(x) == A.GetGlobalNumRows());

    Vector lres, lpsres;
    mbox_vector_initialize_for_hypre(lres, b.Size());
    mbox_vector_initialize_for_hypre(lpsres, b.Size());
    double *p;

    A.Mult(x, lres);
    subtract(b, lres, lres);
    lpsres = 0.;

    lres.StealData(&p);
    res = new HypreParVector(A.GetGlobalNumRows(), p, A.GetRowStarts());
    lpsres.StealData(&p);
    psres = new HypreParVector(A.GetGlobalNumRows(), p, A.GetRowStarts());

    tg_cycle(A, *(tg_data->Ac), *(tg_data->interp), *(tg_data->restr), *res,
             tg_data->pre_smoother, tg_data->post_smoother, *psres,
             tg_data->coarse_solver, tg_data->poly_data);

    mbox_make_owner_data(*res);
    mbox_make_owner_partitioning(*res);
    mbox_make_owner_data(*psres);
    mbox_make_owner_partitioning(*psres);

    return mbox_parallel_inner_product(*psres, *res);
}

double tg_recalc_res_tgprod(HypreParMatrix& A, HypreParVector& b,
                            HypreParVector& x, HypreParVector& x_prev,
                            HypreParVector& res, HypreParVector& psres,
                            void *data)
{
    double rr;
    SA_ASSERT(A.GetGlobalNumRows() == A.GetGlobalNumCols());
    SA_ASSERT(mbox_parallel_vector_size(b) == A.GetGlobalNumRows());
    SA_ASSERT(mbox_parallel_vector_size(x) == A.GetGlobalNumRows());
    SA_ASSERT(mbox_parallel_vector_size(res) == A.GetGlobalNumRows());
    SA_ASSERT(mbox_parallel_vector_size(psres) == A.GetGlobalNumRows());

    psres = x;
    psres -= x_prev;
    rr = mbox_parallel_inner_product(psres, res);
    A.Mult(x, res);
    subtract(b, res, res);
    return rr;
}

double tg_fullrecalc_res_tgprod(HypreParMatrix& A, HypreParVector& b,
                                HypreParVector& x, HypreParVector& x_prev,
                                HypreParVector& res, HypreParVector& psres,
                                void *data)
{
    double rr;
    tg_data_t *tg_data = (tg_data_t *)data;
    SA_ASSERT(A.GetGlobalNumRows() == A.GetGlobalNumCols());
    SA_ASSERT(mbox_parallel_vector_size(b) == A.GetGlobalNumRows());
    SA_ASSERT(mbox_parallel_vector_size(x) == A.GetGlobalNumRows());
    SA_ASSERT(mbox_parallel_vector_size(res) == A.GetGlobalNumRows());
    SA_ASSERT(mbox_parallel_vector_size(psres) == A.GetGlobalNumRows());

    psres = 0.;
    tg_cycle(A, *(tg_data->Ac), *(tg_data->interp), *(tg_data->restr), res,
             tg_data->pre_smoother, tg_data->post_smoother, psres,
             tg_data->coarse_solver, tg_data->poly_data);
    rr = mbox_parallel_inner_product(psres, res);
    A.Mult(x, res);
    subtract(b, res, res);
    return rr;
}

int tg_solve(HypreParMatrix& A, HypreParVector& b, HypreParVector& x,
             HypreParVector *& x_prev, int maxiter, tg_calc_res_ft calc_res,
             tg_recalc_res_ft recalc_res, tg_data_t *tg_data, double rtol,
             double atol, double reducttol, bool output/*=true*/)
{
    int i;
    double rr, end, rr_prev = 1., reduction = -1.;
    HypreParVector *res = NULL, *psres = NULL;

    SA_ASSERT(A.GetGlobalNumRows() == A.GetGlobalNumCols());
    SA_ASSERT(mbox_parallel_vector_size(b) == A.GetGlobalNumRows());
    SA_ASSERT(mbox_parallel_vector_size(x) == A.GetGlobalNumRows());
    SA_ASSERT(reducttol >= 0.);

    Vector lx_prev;
    mbox_vector_initialize_for_hypre(lx_prev, x.Size());
    double *p;
    lx_prev.StealData(&p);
    x_prev = new HypreParVector(A.GetGlobalNumRows(), p, A.GetRowStarts());
    mbox_make_owner_data(*x_prev);

    rr = calc_res(A, b, x, res, psres, tg_data);
    if ((end = rtol * rr) < atol)
        end = atol;
    if (output)
        SA_PRINTF_L(2, "Stationary iteration end tolerance: %g\n", end);
    SA_ASSERT(end >= 0.);
    for (i=1; i <= maxiter && rr > end; ++i)
    {
        if (output && 0 == PROC_RANK && SA_IS_OUTPUT_LEVEL(2))
            PROC_STR_STREAM << "Stationary iteration: after " << i-1
                            << " iterations | (r,r) = " << rr
                            << ", reduction: ";
        if (i>2)
        {
            reduction = rr / rr_prev;
            if (output && 0 == PROC_RANK && SA_IS_OUTPUT_LEVEL(2))
            {
                PROC_STR_STREAM << reduction << "\n";
                SA_PRINTF("%s", PROC_STR_STREAM.str().c_str());
                PROC_CLEAR_STR_STREAM;
            }
            if (reduction > reducttol)
                break;
        } else if (output && 0 == PROC_RANK && SA_IS_OUTPUT_LEVEL(2))
        {
            PROC_STR_STREAM << "wait\n";
            SA_PRINTF("%s", PROC_STR_STREAM.str().c_str());
            PROC_CLEAR_STR_STREAM;
        }

        (*x_prev) = x;
        tg_cycle(A, *(tg_data->Ac), *(tg_data->interp), *(tg_data->restr), b,
                 tg_data->pre_smoother, tg_data->post_smoother, x,
                 tg_data->coarse_solver, tg_data->poly_data);

        rr_prev = rr;
        rr = recalc_res(A, b, x, *x_prev, *res, *psres, tg_data);
    }
    if (output)
        SA_RPRINTF_L(0, 2, "Stationary iteration: after %d iterations"
                           " | (r,r) = %g, reduction: %g\n", i-1, rr,
                           rr / rr_prev);

    if (rr > end)
    {
        if (SA_IS_OUTPUT_LEVEL(2) && output)
        {
            if (maxiter + 1 == i)
                SA_PRINTF("%s", "Stationary iteration: Maximum allowed"
                                " iterations reached without computing a"
                                " solution!\n");
            else
                SA_PRINTF("Stationary iteration: Reduction factor is worse than"
                          " %g\n", reducttol);
        }
        SA_ASSERT(maxiter + 1 == i || reduction > reducttol);
        delete res;
        delete psres;
        return -(i-1);
    }

    delete res;
    delete psres;
    return i-1;
}

int tg_pcg_solve(HypreParMatrix& A, HypreParVector& b, HypreParVector& x,
                 int maxiter, tg_data_t *tg_data, double rtol, double atol,
                 bool zero_rhs, bool output/*=true*/)
{
    tg_fillin_coarse_operator(A, tg_data, true);

    CFunctionSmoother tg_precond(A, smpr_tg, tg_data);

    return pcg(A, tg_precond, b, x, (int)(SA_IS_OUTPUT_LEVEL(2) && output),
               maxiter, rtol, atol, zero_rhs);
}

int tg_run(HypreParMatrix& A,
           const agg_partititoning_relations_t *agg_part_rels,
           HypreParVector& x, HypreParVector& b, int maxiter, double rtol,
           double atol, double reducttol, tg_data_t *tg_data, bool zero_rhs,
           bool output/*=true*/)
{
    int tgiters=0;
    tg_calc_res_ft calc_res;
    tg_recalc_res_ft recalc_res;
    HypreParVector *x_past = NULL;

    SA_ASSERT(CONFIG_ACCESS_OPTION(TG, calc_res) &&
              CONFIG_ACCESS_OPTION(TG, recalc_res));
    calc_res = CONFIG_ACCESS_OPTION(TG, calc_res);
    recalc_res = CONFIG_ACCESS_OPTION(TG, recalc_res);

    SA_ASSERT(tg_data);

    if (output)
        SA_PRINTF_L(4, "%s", "---------- tg_solve { -----------------------\n");
    tg_fillin_coarse_operator(A, tg_data, true);
    if (zero_rhs)
    {
        int reason;
        double cf, acf, eem, trash;

        SA_ASSERT(agg_part_rels);
        reason = adapt_approx_xbad(A, *agg_part_rels, maxiter, tg_data, x, &cf,
                                   &acf, &eem, &trash, &tgiters, rtol, atol,
                                   false, output);
        if (!SA_IS_SET_A_FLAG(reason, ADAPT_XBAD_ERR_TOL_FLAG))
            tgiters = -tgiters;
        if (SA_IS_OUTPUT_LEVEL(3) && output)
        {
            SA_PRINTF("Xbad reason: 0x%X\n", reason);
        }
        if (SA_IS_OUTPUT_LEVEL(2) && output)
        {
            SA_PRINTF("Energy norm final error: %g\n", eem);
            SA_PRINTF("Average convergence factor: %g\n", acf);
            SA_PRINTF("(Asymptotic) convergence factor: %g\n", cf);
        }
    } else
    {
        tgiters = tg_solve(A, b, x, x_past, maxiter, calc_res, recalc_res,
                           tg_data, rtol, atol, reducttol, output);
        if (SA_IS_OUTPUT_LEVEL(3) && output)
        {
            HypreParVector *res, *psres;
            SA_PRINTF("Last residual (r,r): %g\n",
                      calc_res(A, b, x, res, psres, tg_data));
            delete res;
            delete psres;
        }
        delete x_past;
    }
    if (output)
    {
        SA_PRINTF_L(2, "Iterations: %d\n", tgiters);
        SA_PRINTF_L(4, "%s", "---------- } tg_solve -----------------------\n");
    }

    return tgiters;
}

int tg_pcg_run(HypreParMatrix& A,
           const agg_partititoning_relations_t *agg_part_rels,
           HypreParVector& x, HypreParVector& b, int maxiter, double rtol,
           double atol, tg_data_t *tg_data, bool zero_rhs,
           bool output/*=true*/)
{
    int tgiters=0;

    SA_ASSERT(tg_data);

    if (zero_rhs)
    {
        SA_ASSERT(agg_part_rels);
        helpers_random_vect(*agg_part_rels, x);
        x /= mbox_energy_norm_parallel(A, x);
    }

    if (output)
        SA_PRINTF_L(4, "%s", "---------- tg_pcg_solve { ----------------------"
                             "-\n");
    tgiters = tg_pcg_solve(A, b, x, maxiter, tg_data, rtol, atol, zero_rhs,
                           output);
    if (output)
        SA_PRINTF_L(4, "%s", "---------- } tg_pcg_solve ----------------------"
                             "-\n");

    return tgiters;
}

tg_data_t *tg_init_data(HypreParMatrix& A,
    const agg_partititoning_relations_t& agg_part_rels, int nu_interp,
    int nu_relax, double param_relax, double theta)
{
    tg_data_t *tg_data = new tg_data_t;
    SA_ASSERT(tg_data);
    memset(tg_data, 0, sizeof(*tg_data));

    SA_ASSERT(CONFIG_ACCESS_OPTION(TG, pre_smoother) &&
              CONFIG_ACCESS_OPTION(TG, post_smoother));
    tg_data->pre_smoother = CONFIG_ACCESS_OPTION(TG, pre_smoother);
    tg_data->post_smoother = CONFIG_ACCESS_OPTION(TG, post_smoother);

    SA_ASSERT(CONFIG_ACCESS_OPTION(TG, coarse_solver_solver));
    tg_data->coarse_solver.solve_init = CONFIG_ACCESS_OPTION(TG,
                                            coarse_solver_solve_init);
    tg_data->coarse_solver.solve_free = CONFIG_ACCESS_OPTION(TG,
                                            coarse_solver_solve_free);
    tg_data->coarse_solver.solve_copy = CONFIG_ACCESS_OPTION(TG,
                                            coarse_solver_solve_copy);
    tg_data->coarse_solver.solver = CONFIG_ACCESS_OPTION(TG,
                                                         coarse_solver_solver);
    tg_data->coarse_solver.data = CONFIG_ACCESS_OPTION(TG, coarse_solver_data);

    tg_data->theta = theta;

    tg_data->interp_data = interp_init_data(agg_part_rels, nu_interp);
    tg_data->poly_data = smpr_init_poly_data(A, nu_relax, param_relax);

    tg_data->smooth_interp = CONFIG_ACCESS_OPTION(TG, smooth_interp);

    return tg_data;
}

void tg_build_hierarchy(const SparseMatrix& Al, HypreParMatrix& Ag,
                        tg_data_t& tg_data,
                        const agg_partititoning_relations_t& agg_part_rels,
                        void *elem_data_finest)
{
    double tol = 0.; // Essentially, not used.
    double theta = tg_data.theta;

    //XXX: Currently, we see no reason to actually compute all eigenpairs.
    const bool all_eigens = false;

    delete tg_data.ltent_interp;
    delete tg_data.tent_interp;

    tg_data.ltent_interp =
        interp_sparse_tent_build(Al, agg_part_rels, *tg_data.interp_data,
                                 elem_data_finest, tol, theta, NULL, NULL, NULL,
                                 false, false, all_eigens, true);
    SA_ASSERT(tg_data.ltent_interp);

    tg_data.tent_interp =
        interp_global_tent_assemble(agg_part_rels, *tg_data.interp_data,
                                    tg_data.ltent_interp);

    tg_smooth_interp(Ag, tg_data);

    if (SA_IS_OUTPUT_LEVEL(3))
    {
        SA_PRINTF("fine DoFs: %d\n", agg_part_rels.ND);
        SA_PRINTF("interp: %d x %d\n", tg_data.interp->GetGlobalNumRows(),
                  tg_data.interp->GetGlobalNumCols());
        SA_PRINTF("restr: %d x %d\n", tg_data.restr->GetGlobalNumRows(),
                  tg_data.restr->GetGlobalNumCols());
        SA_PRINTF("\t\t\t\t\t\t\tCOARSE SPACE DIMENSION: %d\n",
                  tg_data.interp->GetGlobalNumCols());
    }
    SA_ASSERT(tg_data.interp->GetGlobalNumCols() ==
              tg_data.restr->GetGlobalNumRows());
}

tg_data_t *tg_produce_data(const SparseMatrix& Al, HypreParMatrix& Ag,
    const agg_partititoning_relations_t& agg_part_rels, int nu_interp,
    int nu_relax, double param_relax, void *elem_data_finest, double theta)
{
    tg_data_t *tg_data = tg_init_data(Ag, agg_part_rels, nu_interp, nu_relax,
                                      param_relax, theta);

    tg_build_hierarchy(Al, Ag, *tg_data, agg_part_rels, elem_data_finest);

    return tg_data;
}

void tg_free_data(tg_data_t *tg_data)
{
    if (!tg_data) return;
    smpr_free_poly_data(tg_data->poly_data);
    interp_free_data(tg_data->interp_data);
    delete tg_data->ltent_interp;
    delete tg_data->tent_interp;
    delete tg_data->interp;
    delete tg_data->restr;
    tg_free_coarse_operator(*tg_data);
    delete tg_data;
}

tg_data_t *tg_copy_data(const tg_data_t *src)
{
    if (!src) return NULL;
    tg_data_t *dst = new tg_data_t;

    dst->interp_data = interp_copy_data(src->interp_data);
    dst->pre_smoother = src->pre_smoother;
    dst->post_smoother = src->post_smoother;

    dst->coarse_solver = src->coarse_solver;
    dst->Ac = mbox_clone_parallel_matrix(src->Ac);
    if (dst->Ac && src->coarse_solver.solve_copy)
    {
        SA_PRINTF_L(5, "%s", "Copying coarse solver...\n");
        dst->coarse_solver.data =
            src->coarse_solver.solve_copy(*src->Ac, src->coarse_solver.data);
    }

    dst->ltent_interp = mbox_copy_sparse_matr(src->ltent_interp);
    dst->tent_interp = mbox_clone_parallel_matrix(src->tent_interp);
    dst->interp = mbox_clone_parallel_matrix(src->interp);
    dst->restr = mbox_clone_parallel_matrix(src->restr);
    dst->poly_data = smpr_copy_poly_data(src->poly_data);
    dst->smooth_interp = src->smooth_interp;
    dst->theta = src->theta;

    return dst;
}
