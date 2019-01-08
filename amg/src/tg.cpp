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
#include "tg.hpp"
#include <mfem.hpp>
#include "smpr.hpp"
#include "solve.hpp"
#include "helpers.hpp"
#include "interp.hpp"
#include "adapt.hpp"
#include "mfem_addons.hpp"

namespace saamge
{
using namespace mfem;

/* Options */

CONFIG_BEGIN_CLASS_DEFAULTS(TG)
//    CONFIG_DEFINE_OPTION_DEFAULT(calc_res, tg_calc_res_tgprod),
//    CONFIG_DEFINE_OPTION_DEFAULT(recalc_res, tg_recalc_res_tgprod),
    CONFIG_DEFINE_OPTION_DEFAULT(pre_smoother, smpr_sym_poly),
    CONFIG_DEFINE_OPTION_DEFAULT(post_smoother, smpr_sym_poly)
//    CONFIG_DEFINE_OPTION_DEFAULT(pre_smoother, smpr_gauss_seidel),
//    CONFIG_DEFINE_OPTION_DEFAULT(post_smoother, smpr_gauss_seidel),
CONFIG_END_CLASS_DEFAULTS

CONFIG_DEFINE_CLASS(TG);

/* Class implementations */

HypreDirect::HypreDirect(mfem::HypreParMatrix& mat)
{
   hypre_ParCSRMatrix* h_mat = mat;
   hypre_CSRMatrix* diag = h_mat->diag;
   mat_ = new SparseMatrix(diag->i, diag->j, diag->data, diag->num_rows,
                           diag->num_cols, false, false, false);
   solver_.SetOperator(*mat_);
}

HypreDirect::~HypreDirect()
{
   delete mat_;
}

void HypreDirect::SetOperator(const mfem::Operator &op)
{
}

void HypreDirect::Mult(const mfem::Vector &x, mfem::Vector &y) const
{
   solver_.Mult(x, y);
}

/* Functions */

/**
   slight reworking of tg_cycle with x, b as Vector instead of HypreParVector
   we use this to precondition CG with this cycle in coarse_solve_pcg_correct_nulspace()
   (in general it works better in an object-oriented mfem::Solver setting)
*/
void tg_cycle_atb(HypreParMatrix& A, HypreParMatrix& Ac, HypreParMatrix& interp,
                  HypreParMatrix& restr, const Vector& b, smpr_ft pre_smoother,
                  smpr_ft post_smoother, Vector& x, Solver& coarse_solver,
                  void *data)
{
    SA_ASSERT(A.GetGlobalNumRows() == A.GetGlobalNumCols());
    SA_ASSERT(Ac.GetGlobalNumRows() == Ac.GetGlobalNumCols());
    SA_ASSERT(interp.GetGlobalNumRows() == A.GetGlobalNumRows());
    SA_ASSERT(restr.GetGlobalNumCols() == A.GetGlobalNumCols());
    //SA_ASSERT(interp.GetGlobalNumRows() >= interp.GetGlobalNumCols());
    SA_ASSERT(interp.GetGlobalNumCols() == restr.GetGlobalNumRows());
    // SA_ASSERT(restr.GetGlobalNumRows() == Ac.GetGlobalNumRows()); // Does not hold in the nonconforming case.
    // SA_ASSERT(mbox_parallel_vector_size(b) == A.GetGlobalNumRows());
    // SA_ASSERT(mbox_parallel_vector_size(x) == A.GetGlobalNumRows());
    SA_ASSERT(pre_smoother);
    SA_ASSERT(post_smoother);
    // SA_ASSERT(coarse_solver.solver);

    Vector res(b.Size()), resc(mbox_rows_in_current_process(restr));
    Vector xc(mbox_rows_in_current_process(restr));
    xc = 0.0;

    pre_smoother(A, b, x, data);

    A.Mult(x, res);
    subtract(b, res, res);
    restr.Mult(res, resc);

    HypreParVector RESC(PROC_COMM, restr.GetGlobalNumRows(), resc.GetData(),
                        restr.GetRowStarts());
    HypreParVector XC(PROC_COMM, restr.GetGlobalNumRows(), xc.GetData(),
                      restr.GetRowStarts());

    // could repeat this for W-cycle...
    // coarse_solver.solver(Ac, RESC, XC, coarse_solver.data);
    coarse_solver.Mult(RESC, XC);

    // interp.Mult(XC, x, 1., 1.);
    interp.Mult(1.0, XC, 1.0, x);

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
    res = new HypreParVector(PROC_COMM, A.GetGlobalNumRows(), p,
                             A.GetRowStarts());
    lpsres.StealData(&p);
    psres = new HypreParVector(PROC_COMM, A.GetGlobalNumRows(), p,
                               A.GetRowStarts());

    tg_cycle_atb(A, *(tg_data->Ac), *(tg_data->interp), *(tg_data->restr), *res,
                 tg_data->pre_smoother, tg_data->post_smoother, *psres,
                 *tg_data->coarse_solver, tg_data->poly_data);

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
    tg_cycle_atb(A, *(tg_data->Ac), *(tg_data->interp), *(tg_data->restr), res,
                 tg_data->pre_smoother, tg_data->post_smoother, psres,
                 *tg_data->coarse_solver, tg_data->poly_data);
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
    x_prev = new HypreParVector(PROC_COMM, A.GetGlobalNumRows(), p, A.GetRowStarts());
    mbox_make_owner_data(*x_prev);

    rr = tg_calc_res_tgprod(A, b, x, res, psres, tg_data);
    if ((end = rtol * rr) < atol)
        end = atol;
    if (output)
        SA_RPRINTF_L(0,2, "Stationary iteration end tolerance: %g\n", end);
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
        tg_cycle_atb(A, *(tg_data->Ac), *(tg_data->interp), *(tg_data->restr), b,
                     tg_data->pre_smoother, tg_data->post_smoother, x,
                     *tg_data->coarse_solver, tg_data->poly_data);

        rr_prev = rr;
        rr = tg_recalc_res_tgprod(A, b, x, *x_prev, *res, *psres, tg_data);
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

    VCycleSolver tg_precond(tg_data,false);
    tg_precond.SetOperator(A);

    return kalchev_pcg(A, tg_precond, b, x, (int)(output),
                       maxiter, rtol, atol, zero_rhs);
}

int tg_run(HypreParMatrix& A,
           const agg_partitioning_relations_t *agg_part_rels,
           HypreParVector& x, HypreParVector& b, int maxiter, double rtol,
           double atol, double reducttol, tg_data_t *tg_data, bool zero_rhs,
           bool output/*=true*/)
{
    int tgiters=0;
    HypreParVector *x_past = NULL;

    SA_ASSERT(tg_data);

    if (output)
        SA_RPRINTF_L(0,4, "%s", "---------- tg_solve { -----------------------\n");
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
            SA_RPRINTF(0,"Xbad reason: 0x%X\n", reason);
        }
        if (SA_IS_OUTPUT_LEVEL(2) && output)
        {
            SA_RPRINTF(0,"Energy norm final error: %g\n", eem);
            SA_RPRINTF(0,"Average convergence factor: %g\n", acf);
            SA_RPRINTF(0,"(Asymptotic) convergence factor: %g\n", cf);
        }
    } else
    {
        tgiters = tg_solve(A, b, x, x_past, maxiter, tg_calc_res_tgprod,
                           tg_recalc_res_tgprod, tg_data, rtol, atol, reducttol,
                           output);
        if (SA_IS_OUTPUT_LEVEL(3) && output)
        {
            HypreParVector *res, *psres;
            SA_PRINTF("Last residual (r,r): %g\n",
                      tg_calc_res_tgprod(A, b, x, res, psres, tg_data));
            delete res;
            delete psres;
        }
        delete x_past;
    }
    if (output)
    {
        SA_RPRINTF_L(0,2, "Iterations: %d\n", tgiters);
        SA_RPRINTF_L(0,4, "%s", "---------- } tg_solve -----------------------\n");
    }

    return tgiters;
}

int tg_pcg_run(HypreParMatrix& A,
           const agg_partitioning_relations_t *agg_part_rels,
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
        SA_RPRINTF_L(0,4, "%s", "---------- tg_pcg_solve { ----------------------"
                             "-\n");
    tgiters = tg_pcg_solve(A, b, x, maxiter, tg_data, rtol, atol, zero_rhs,
                           output);
    if (output)
        SA_RPRINTF_L(0,4, "%s", "---------- } tg_pcg_solve ----------------------"
                             "-\n");

    return tgiters;
}

tg_data_t *tg_init_data(
    HypreParMatrix& A, const agg_partitioning_relations_t& agg_part_rels, 
    int nu_pro, int nu_relax, double theta, bool smooth_interp, 
    double smooth_drop_tol, bool use_arpack)
{
    tg_data_t *tg_data = new tg_data_t;
    SA_ASSERT(tg_data);
    memset(tg_data, 0, sizeof(*tg_data));

    SA_ASSERT(CONFIG_ACCESS_OPTION(TG, pre_smoother) &&
              CONFIG_ACCESS_OPTION(TG, post_smoother));
    tg_data->pre_smoother = CONFIG_ACCESS_OPTION(TG, pre_smoother);
    tg_data->post_smoother = CONFIG_ACCESS_OPTION(TG, post_smoother);

    tg_data->theta = theta;

    tg_data->interp_data = interp_init_data(agg_part_rels, nu_pro,
                                            use_arpack, false);
    tg_data->interp_data->drop_tol = smooth_drop_tol;
    tg_data->poly_data = smpr_init_poly_data(A, nu_relax, 0.0);

    tg_data->smooth_interp = smooth_interp;

    tg_data->tag = -1;

    tg_data->doing_spectral = false;

    return tg_data;
}

void tg_assemble_and_smooth(HypreParMatrix &Ag,
                            tg_data_t& tg_data,
                            const agg_partitioning_relations_t& agg_part_rels)
{
    StopWatch chrono;
    chrono.Start();
    tg_data.tent_interp =
        interp_global_tent_assemble(agg_part_rels, *tg_data.interp_data,
                                    tg_data.ltent_interp);
    chrono.Stop();
    SA_RPRINTF_L(0, 5, "Time for global_tent_assemble: %f\n",chrono.RealTime());

    if (tg_data.interp_data->scaling_P)
    {
        // TODO: check here if we have actually filled in the
        // local_coarse_one_representation, and print an informative
        // error message and/or fall back to no corrected nullspace

        SA_RPRINTF_L(0, 5, "%s","Building scaling P.\n");
        tg_data.scaling_P = interp_scaling_P_assemble(
            agg_part_rels, *tg_data.interp_data, tg_data.ltent_interp, 
            tg_data.interp_data->local_coarse_one_representation);
    }

    chrono.Clear();
    chrono.Start();
    tg_smooth_interp(Ag, tg_data);
    chrono.Stop();
    SA_RPRINTF_L(0, 5, "Time for tg_smooth_interp: %f\n",chrono.RealTime());
    if (SA_IS_OUTPUT_LEVEL(3))
    {
        SA_RPRINTF(0,"fine DoFs: %d\n", agg_part_rels.ND);
        SA_RPRINTF(0,"interp: %d x %d\n", tg_data.interp->GetGlobalNumRows(),
                  tg_data.interp->GetGlobalNumCols());
        SA_RPRINTF(0,"restr: %d x %d\n", tg_data.restr->GetGlobalNumRows(),
                  tg_data.restr->GetGlobalNumCols());
        SA_RPRINTF(0,"\t\t\t\t\t\t\tCOARSE SPACE DIMENSION: %d\n",
                  tg_data.interp->GetGlobalNumCols());
    }
    SA_ASSERT(tg_data.interp->GetGlobalNumCols() ==
              tg_data.restr->GetGlobalNumRows());
}

void tg_build_hierarchy_with_polynomial(
    HypreParMatrix& Ag, Mesh& mesh, tg_data_t& tg_data,
    const agg_partitioning_relations_t& agg_part_rels,
    ElementMatrixProvider *elem_data, int polynomial_order, bool use_spectral,
    bool avoid_ess_bdr_dofs)
{
    delete tg_data.ltent_interp;
    delete tg_data.tent_interp;

    SA_ASSERT(tg_data.polynomial_coarse_space == 0 ||
              tg_data.polynomial_coarse_space == 1);

    // GetNV() should be GetNNodes() or something that does not exist,
    // this will not work for higher order as is...
    Vector nodes;
    mesh.GetNodes(nodes);
    tg_data.doing_spectral = use_spectral;
    tg_data.elem_data = elem_data;
    tg_data.ltent_interp = interp_build_composite(
        agg_part_rels, *tg_data.interp_data, elem_data, tg_data.theta,
        mesh.SpaceDimension(), mesh.GetNV(), nodes, polynomial_order,
        avoid_ess_bdr_dofs, use_spectral);

    SA_ASSERT(tg_data.ltent_interp);
    tg_assemble_and_smooth(Ag, tg_data, agg_part_rels);
}

void tg_build_hierarchy(HypreParMatrix& Ag, tg_data_t& tg_data,
                        const agg_partitioning_relations_t& agg_part_rels,
                        ElementMatrixProvider *elem_data,
                        bool avoid_ess_bdr_dofs)
{
    double useless_parameter = 0.; // Essentially, not used.

    //XXX: Currently, we see no reason to actually compute all eigenpairs.
    const bool all_eigens = false;

    delete tg_data.ltent_interp;
    delete tg_data.tent_interp;

    if (tg_data.polynomial_coarse_space == -1 && tg_data.theta > 0.0)
    {
        // save the pointer so tg_data can free it
        tg_data.elem_data = elem_data;
        tg_data.doing_spectral = true;
        tg_data.ltent_interp = interp_sparse_tent_build(
            agg_part_rels, *tg_data.interp_data, elem_data,
            useless_parameter, tg_data.theta, NULL, NULL, NULL, false, false,
            all_eigens, true, avoid_ess_bdr_dofs);
    }
    else if (tg_data.polynomial_coarse_space == 1)
    {
        std::cerr << "NOT IMPLEMENTED: try tg_build_hierarchy_with_polynomial()?"
                  << std::endl;
        throw 1;
    }
    else
    {
        tg_data.elem_data = elem_data; // should be NULL? but copy anyway
        tg_data.doing_spectral = false;
        tg_data.ltent_interp = 
            interp_build_minimal(agg_part_rels, *tg_data.interp_data);
    }
    SA_ASSERT(tg_data.ltent_interp);
    tg_assemble_and_smooth(Ag, tg_data, agg_part_rels);
}

void tg_augment_interp_with_identity(tg_data_t& tg_data, int k)
{
    SA_ASSERT(PROC_NUM == 1);

    hypre_ParCSRMatrix * hInterp = *tg_data.interp;
    SparseMatrix * new_interp = new SparseMatrix(hInterp->diag->num_rows + k,
                                                 hInterp->diag->num_cols + k);
    int * AI = hInterp->diag->i;
    int * AJ = hInterp->diag->j;
    double * Adata = hInterp->diag->data;
    for (int i=0; i<k; ++i)
        new_interp->Add(i,i,1.0);
    for (int i=0; i<hInterp->diag->num_rows; ++i)
        for (int j=AI[i]; j<AI[i+1]; ++j)
            new_interp->Add(i+k,AJ[j]+k,Adata[j]);
    new_interp->Finalize();

    delete tg_data.interp;
    int * row_starts = new int[2];
    row_starts[0] = 0;
    row_starts[1] = new_interp->Height();
    int * col_starts = new int[2];
    col_starts[0] = 0;
    col_starts[1] = new_interp->Width();

    tg_data.interp = new HypreParMatrix(
        PROC_COMM, new_interp->Height(), new_interp->Height(),
        new_interp->Width(), new_interp->GetI(), new_interp->GetJ(),
        new_interp->GetData(), row_starts, col_starts);
    mbox_make_owner_rowstarts_colstarts(*tg_data.interp);
    delete [] row_starts;
    delete [] col_starts;
    delete new_interp;

    delete tg_data.restr;
    tg_data.restr = tg_data.interp->Transpose();
}

void ExtractSubMatrices(const SparseMatrix& A, 
                        const agg_partitioning_relations_t& agg_part_rels,
                        Array<SparseMatrix*>& agglomerate_element_matrices)
{
    int nparts = agg_part_rels.nparts;
    agglomerate_element_matrices.SetSize(nparts);

    // SparseMatrix ** agglomerate_element_matrices = new SparseMatrix*[nparts];
    for (int part=0; part<nparts; ++part)
    {
        int localsize = agg_part_rels.AE_to_dof->RowSize(part);
        int * row = agg_part_rels.AE_to_dof->GetRow(part);
        agglomerate_element_matrices[part] = new SparseMatrix(localsize,localsize);

        // --- paste from agg_build_AE_stiffm_with_global(), removed ASSERTS, could put back
        // loop over dof belonging to this AE
        for (int i=0; i < localsize; ++i)
        {
            int glob_dof, row_start, row_size;
            int *neighbours;
            double *neigh_data;

            glob_dof = row[i];
            row_start = A.GetI()[glob_dof];
            neighbours = &(A.GetJ()[row_start]);
            neigh_data = &(A.GetData()[row_start]);
            // row_size = const_cast<SparseMatrix&>(A).RowSize(glob_dof);
            row_size = A.RowSize(glob_dof);
            // loop over fine dof neighbors of fine dof i
            // this loop basically extracts a principal submatrix from A
            // with some (important) modifications for boundary conditions etc.
            for (int j=0; j < row_size; ++j)
            {
                int glob_neigh, local_neigh;

                glob_neigh = neighbours[j];

                if (agg_elem_in_col(glob_neigh, part, *agg_part_rels.dof_to_AE) < 0)
                    continue;

                local_neigh = agg_map_id_glob_to_AE(glob_neigh, part, agg_part_rels);

                if (0. != neigh_data[j])
                    agglomerate_element_matrices[part]->Set(i, local_neigh, neigh_data[j]);
            }
        }
        agglomerate_element_matrices[part]->Finalize(); // should we finalize at the end instead of here?
        // --- end paste
    }
    // now modify these matrices to have rowsum 0, so that vector of all 1s is in coarse space
    // (this ignores some boundary condition issues, makes assumptions on basis, but for
    // algebraic I'm not sure what else to do)
    for (int part=0; part<nparts; ++part)
    {
        int * row = agg_part_rels.AE_to_dof->GetRow(part);
        int numrows = agglomerate_element_matrices[part]->Size();
        Vector rowsums(numrows);
        agglomerate_element_matrices[part]->GetRowSums(rowsums);
        if (numrows > 1)
        {
            for (int i=0; i<numrows; ++i)
            {
                if (agglomerate_element_matrices[part]->Elem(i,i) <= 0.0)
                    std::cout << "    Warning: part " << part << " row " << i
                              << " has negative diagonal before modifications " 
                              << agglomerate_element_matrices[part]->Elem(i,i)
                              << std::endl;
                if (agglomerate_element_matrices[part]->RowSize(i) > 1)
                    agglomerate_element_matrices[part]->Add(i,i,-rowsums(i));
                if (agglomerate_element_matrices[part]->Elem(i,i) <= 0.0)
                {
                    std::cout << "    Warning: part " << part << " row " << i 
                              << " has negative diagonal after modifications "
                              << agglomerate_element_matrices[part]->Elem(i,i)
                              << ", glob_dof = " << row[i] << std::endl;
                    std::cout << "      row size = "
                              << agglomerate_element_matrices[part]->RowSize(i)
                              << ", rowsum = " << rowsums(i) << std::endl;
                    agglomerate_element_matrices[part]->Set(i,i,1.0);
                }
            }
        }
        else
        {
            // aggregates with a single cell need a 1 on the diagonal (?)
            agglomerate_element_matrices[part]->Set(0,0,1.0);
        }
    }
}

void TestWindowSubMatrices()
{
    SparseMatrix A(9,9);
    A.Set(0,0,2.0); A.Set(0,1,-1.0); A.Set(0,3,-1.0);
    A.Set(1,0,-1.0); A.Set(1,1,3.0); A.Set(1,2,-1.0); A.Set(1,5,-1.0);
    A.Set(2,1,-1.0); A.Set(2,2,2.0); A.Set(2,4,-1.0);
    A.Set(3,0,-1.0); A.Set(3,3,3.0); A.Set(3,5,-1.0); A.Set(3,6,-1.0);
    A.Set(4,2,-1.0); A.Set(4,4,3.0); A.Set(4,5,-1.0); A.Set(4,8,-1.0);
    A.Set(5,1,-1.0); A.Set(5,3,-1.0); A.Set(5,4,-1.0); A.Set(5,5,4.0); A.Set(5,7,-1.0);
    A.Set(6,3,-1.0); A.Set(6,6,2.0); A.Set(6,7,-1.0);
    A.Set(7,5,-1.0); A.Set(7,6,-1.0); A.Set(7,7,3.0); A.Set(7,8,-1.0);
    A.Set(8,4,-1.0); A.Set(8,7,-1.0); A.Set(8,8,2.0);
    A.Finalize();

    HypreParMatrix * Aglobal = FakeParallelMatrix(&A);

    agg_partitioning_relations_t * agg_part_rels = new agg_partitioning_relations_t;
    memset(agg_part_rels, 0, sizeof(*agg_part_rels));
    // agg_part_rels->nparts = 2;
    int nparts = 2;
    int partitioning[A.Size()];
    for (int i=0; i<5; ++i)
        partitioning[i] = 0;
    for (int i=5; i<9; ++i)
        partitioning[i] = 1;

    SparseMatrix * identity = IdentitySparseMatrix(A.Height());
    int drow_starts[2];
    drow_starts[0] = 0;
    drow_starts[1] = A.Height();
    HypreParMatrix * dof_truedof = new HypreParMatrix(PROC_COMM, identity->Height(), drow_starts, identity);

    Table *elem_to_elem = TableFromSparseMatrix(A);
    Table *elem_to_dof = IdentityTable(A.Size());
    SA_ASSERT(elem_to_dof);
    SA_ASSERT(elem_to_elem);

    agg_part_rels->nparts = nparts;
    agg_part_rels->Dof_TrueDof = dof_truedof;
    agg_part_rels->owns_Dof_TrueDof = false;

    agg_part_rels->dof_to_elem = new Table();
    agg_part_rels->AE_to_dof = new Table();
    agg_part_rels->dof_to_AE = new Table();

    SA_ASSERT(elem_to_elem);
    agg_part_rels->elem_to_elem = elem_to_elem;
    agg_part_rels->partitioning = partitioning;

    char * bdr_dofs = new char[A.Size()];
    memset(bdr_dofs, 0, sizeof(char) * A.Size());

    const bool do_aggregates = false;
    agg_create_partitioning_tables(agg_part_rels,
                                   *Aglobal, A.Size(), elem_to_dof,
                                   elem_to_elem, partitioning,
                                   bdr_dofs, &nparts,
                                   dof_truedof,
                                   do_aggregates);

    Array<SparseMatrix*> agglomerate_element_matrices;
    WindowSubMatrices(A, *agg_part_rels, agglomerate_element_matrices);

    SA_RPRINTF(0,"%s","window matrix 0:\n");
    agglomerate_element_matrices[0]->Print();
    SA_RPRINTF(0,"%s","window matrix 1:\n");
    agglomerate_element_matrices[1]->Print();

    delete Aglobal;
}

void WindowSubMatrices(const SparseMatrix& A, 
                       const agg_partitioning_relations_t& agg_part_rels,
                       Array<SparseMatrix*>& agglomerate_element_matrices)
{
    int nparts = agg_part_rels.nparts;
    agglomerate_element_matrices.SetSize(nparts);

    for (int part=0; part<nparts; ++part)
    {
        int localsize = agg_part_rels.AE_to_dof->RowSize(part);
        if (localsize == 1)
        {
            agglomerate_element_matrices[part] = new SparseMatrix(localsize,localsize);
            agglomerate_element_matrices[part]->Set(0,0,1.0);
            agglomerate_element_matrices[part]->Finalize();
            continue;
        }
        int * AE_dof = agg_part_rels.AE_to_dof->GetRow(part);
        // agglomerate_element_matrices[part] = new SparseMatrix(localsize,localsize);

        // let's actually construct the extension mapping
        // we begin by constructing the denominators of the alpha coefficients, 
        // and figuring out the size of the extension operator
        std::map<int, double> denominators;
        std::map<int, int> globalcol_to_Xnumber;
        int num_extension_rows = 0;
        for (int i=0; i<localsize; ++i)
        {
            int glob_dof = AE_dof[i];
            int row_start = A.GetI()[glob_dof];
            int *neighbours = &(A.GetJ()[row_start]);
            int row_size = A.RowSize(glob_dof);

            // loop over fine dof neighbors of fine dof i
            for (int j=0; j<row_size; ++j)
            {
                int glob_neigh = neighbours[j];

                // if this neighbor is outside T, make an entry in the denominators std::map
                if (!denominators.count(glob_neigh)
                    &&
                    agg_elem_in_col(glob_neigh, part, *agg_part_rels.dof_to_AE) < 0)
                {
                    double value = 0.0;
                    int inner_row_start = A.GetI()[glob_neigh];
                    int * inner_neighbours = &(A.GetJ()[inner_row_start]);
                    double *inner_data = &(A.GetData()[inner_row_start]);
                    int inner_row_size = A.RowSize(glob_neigh);
                    for (int k=0; k<inner_row_size; ++k)
                    {
                        // SA_RPRINTF(0,"    k=%d, inner_neighbours[k]=%d, inner_data[k]=%f\n",
                        //         k, inner_neighbours[k], inner_data[k]);
                        if (agg_elem_in_col(inner_neighbours[k], part, *agg_part_rels.dof_to_AE) >= 0)
                            value += inner_data[k];
                    }
                    // SA_RPRINTF(0,"  i=%d, j=%d, glob_dof=%d, glob_neigh=%d, value=%f\n",
                    //         i, j, glob_dof, glob_neigh, value);
                    SA_ASSERT(fabs(value) > 0.0);
                    denominators[glob_neigh] = value;
                    globalcol_to_Xnumber[glob_neigh] = num_extension_rows++;
                }
            }
        }
        // now construct the extension mapping and A_{T,\X} (and also A_TT...)
        SparseMatrix extension(num_extension_rows, localsize);
        SparseMatrix ATX(localsize, num_extension_rows);
        SparseMatrix ATT(localsize, localsize);
        for (int i=0; i<localsize; ++i)
        {
            int glob_dof = AE_dof[i];
            int row_start = A.GetI()[glob_dof];
            int * neighbours = &(A.GetJ()[row_start]);
            double * neigh_data = &(A.GetData()[row_start]);
            int row_size = A.RowSize(glob_dof);

            // loop over neighbors of i that are not in T, but do appear in denominators
            for (int j=0; j<row_size; ++j)
            {
                int glob_neigh = neighbours[j];
                if (denominators.count(glob_neigh) // TODO: I think these two conditions are testing the same thing...
                    &&
                    agg_elem_in_col(glob_neigh, part, *agg_part_rels.dof_to_AE) < 0)
                {
                    ATX.Add(i, globalcol_to_Xnumber[glob_neigh], neigh_data[j]);
                    extension.Add(globalcol_to_Xnumber[glob_neigh], i, neigh_data[j] / denominators[glob_neigh]);
                }
                else
                {
                    SA_ASSERT(agg_elem_in_col(glob_neigh, part, *agg_part_rels.dof_to_AE) >= 0);
                    int local_neigh = agg_map_id_glob_to_AE(glob_neigh, part, agg_part_rels);
                    ATT.Add(i, local_neigh, neigh_data[j]);
                }
            }
        }
        extension.Finalize();
        ATX.Finalize();
        ATT.Finalize();

        SparseMatrix * temp;
        if (num_extension_rows == 0)
        {
            temp = new SparseMatrix(ATT.Height(),ATT.Width());
            temp->Finalize();
            // temp = 0.0;
        }
        else
            temp = Mult(ATX, extension);
        agglomerate_element_matrices[part] = Add(ATT, *temp);
        // agglomerate_element_matrices[part] = Add(*temp, ATT);
#if (SA_IS_DEBUG_LEVEL(3))
        for (int q=0; q<agglomerate_element_matrices[part]->Height(); ++q)
            SA_ASSERT(agglomerate_element_matrices[part]->Elem(q,q) > 0.0);
#endif
        delete temp;
    }
}

/**
   In the current implementation, we are assuming serial, and assuming Ag and
   Alocal are essentially the same matrix.
*/
tg_data_t *tg_produce_data_algebraic(
    const SparseMatrix& Alocal,
    HypreParMatrix& Ag, const agg_partitioning_relations_t& agg_part_rels,
    int nu_pro, int nu_relax, double spectral_tol, bool smooth_interp,
    int polynomial_coarse_arg, bool use_window, bool use_arpack,
    bool avoid_ess_bdr_dofs)
{
    SA_ASSERT(PROC_NUM == 1);

    Array<SparseMatrix*> agglomerate_element_matrices;
    // only need SubMatrices if !minimal_coarse_arg (may still need them for local correction)
    if (use_window)
        WindowSubMatrices(Alocal, agg_part_rels, agglomerate_element_matrices);
    else
        ExtractSubMatrices(Alocal, agg_part_rels, agglomerate_element_matrices);

    ElementMatrixProvider * emp = new ElementMatrixArray(
        agg_part_rels, agglomerate_element_matrices);

    tg_data_t * out = tg_produce_data(
        Ag, agg_part_rels, nu_pro, nu_relax, emp, spectral_tol,
        smooth_interp, polynomial_coarse_arg, use_arpack, avoid_ess_bdr_dofs);

    return out;
}

void tg_replace_submatrices(tg_data_t &tg_data, const SparseMatrix &Alocal, 
                            const agg_partitioning_relations_t& agg_part_rels,
                            bool use_window)
{
    SA_ASSERT(PROC_NUM == 1);

    Array<SparseMatrix*> agglomerate_element_matrices;
    if (use_window)
        WindowSubMatrices(Alocal, agg_part_rels, agglomerate_element_matrices);
    else
        ExtractSubMatrices(Alocal, agg_part_rels, agglomerate_element_matrices);

    for (int i=0; i<agg_part_rels.nparts; ++i)
    {
        delete tg_data.interp_data->AEs_stiffm[i];
        tg_data.interp_data->AEs_stiffm[i] = agglomerate_element_matrices[i];
    }    
}
                            

/**
   NOTE: only usable for actual two level method, do not use this
   routine as part of a multilevel method

   The SAAMGe steering committee is considering a proposal to mark
   this routine DEPRECATED and require users to call tg_init_data and
   tg_build_hierarchy[_linears] separately. Please send feedback on
   this proposal to atb@llnl.gov.
*/
tg_data_t *tg_produce_data(
    HypreParMatrix& Ag, const agg_partitioning_relations_t& agg_part_rels,
    int nu_pro, int nu_relax, ElementMatrixProvider *elem_data, double theta,
    bool smooth_interp, int polynomial_coarse_arg,
    bool use_arpack, bool avoid_ess_bdr_dofs)
{
    const double smooth_drop_tol = 0.0;
    tg_data_t *tg_data = tg_init_data(
        Ag, agg_part_rels, nu_pro, nu_relax, theta, smooth_interp,
        smooth_drop_tol, use_arpack);
    tg_data->polynomial_coarse_space = polynomial_coarse_arg;

    tg_build_hierarchy(Ag, *tg_data, agg_part_rels, elem_data,
                       avoid_ess_bdr_dofs);
    return tg_data;
}

void tg_free_data(tg_data_t *tg_data)
{
    if (!tg_data) return;
    smpr_free_poly_data(tg_data->poly_data);
    interp_free_data(tg_data->interp_data, tg_data->doing_spectral,
                     tg_data->theta);
    delete tg_data->ltent_interp;
    delete tg_data->ltent_restr;
    delete tg_data->tent_interp;
    delete tg_data->scaling_P;
    delete tg_data->interp;
    delete tg_data->restr;
    tg_free_coarse_operator(*tg_data);
    delete tg_data->elem_data;
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

    dst->ltent_interp = mbox_copy_sparse_matr(src->ltent_interp);
    dst->ltent_restr = mbox_copy_sparse_matr(src->ltent_restr);
    dst->tent_interp = mbox_clone_parallel_matrix(src->tent_interp);
    dst->interp = mbox_clone_parallel_matrix(src->interp);
    dst->restr = mbox_clone_parallel_matrix(src->restr);
    dst->poly_data = smpr_copy_poly_data(src->poly_data);
    dst->smooth_interp = src->smooth_interp;
    dst->theta = src->theta;

    return dst;
}

double tg_compute_OC(HypreParMatrix& A, tg_data_t& tg_data)
{
    SA_ASSERT(tg_data.Ac);
    return 1.0 + tg_data.Ac->NNZ() / ((double) A.NNZ());
}

} // namespace saamge
