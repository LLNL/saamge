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

/**
  A in this context is the coarse element agglomeration matrix, Ac is a slightly 
  coarser matrix with corrected nulspace so that Hypre will work well
*/
CorrectNullspace::CorrectNullspace(HypreParMatrix& A,
                                   HypreParMatrix * interp_arg,
                                   int smoother_steps,
                                   bool smooth_phat,
                                   bool v_cycle,
                                   bool iterative_mode) :
    Solver(A.M(), iterative_mode),
    smoother_steps(smoother_steps),
    smooth_phat(smooth_phat),
    interp(interp_arg)
{
    SA_ASSERT(interp);
    SA_RPRINTF(0,"%s\n","Initializing corrected nulspace solver.");

    cumulative_spectral_cg = 0;
    spectral_cycles = 1;

    // A is matrix at spectral level
    SA_RPRINTF(0,"  [correctnulspace] A is %d by %d\n",A.M(), A.N());
    SA_RPRINTF(0,"  [correctnulspace] interp is %d by %d\n", interp->M(), interp->N());

    // TODO 3 is like nu_pro, nu_relax etc, it's more like a DEGREE than a number of steps
    // 3 is number of smoothing steps, 0.0 is a parameter that has no effect for the smoother we choose (?)
    // default smoother is SMPR_POLY_SAS
    smoother_data = (void*) smpr_init_poly_data(A, smoother_steps, 0.0); 
    smoother = smpr_sym_poly;

    // smooth out the interpolation from nulspace level to spectral level
    StopWatch sw;
    sw.Start();
    if (smooth_phat)
    {
        int nunu = 3;
        int degree;
        double * roots = smpr_sa_poly_roots(nunu, &degree);
        SA_RPRINTF(0,"%s","  Smoothing interpolation from nulspace level to spectral level.\n");
            
        smpr_poly_data_t* sd = (smpr_poly_data_t*) smoother_data;
        SA_ASSERT(sd->Dinv_neg);
        HypreParMatrix * smoothed_interp = interp_smooth(degree, roots, 1, A,
                                                         *interp, *sd->Dinv_neg);
        delete[] roots;
        delete interp;
        interp = smoothed_interp;
    }
    sw.Stop();
    // SA_RPRINTF(0,"  Actual smoothing took %f seconds.\n",sw.RealTime());

    // construct matrix Ac at nulspace level
    SA_ASSERT(interp->M() == A.N());
    restr = interp->Transpose();
    sw.Clear();
    sw.Start();
    Ac = RAP(&A,interp);
    sw.Stop();
    // SA_RPRINTF(0,"  RAP took %f seconds.\n",sw.RealTime()); // this takes forever if you smooth_phat!
    SA_RPRINTF(0,"  [correctnulspace] Ac is %d by %d\n", Ac->M(), Ac->N());
    SA_RPRINTF(0,"  [correctnulspace] operator complexity is %.2f\n", 
               ((double) Ac->NNZ() + (double) A.NNZ()) / (double) A.NNZ());

    if (v_cycle)
    {
        HypreBoomerAMG* temp_ptr = new HypreBoomerAMG(*Ac);
        temp_ptr->SetPrintLevel(0);
        correct_nullspace_coarse = temp_ptr;
    }
    else
        correct_nullspace_coarse = new AMGSolver(*Ac,false);

    SetOperator(A);
}

CorrectNullspace::~CorrectNullspace()
{
    SA_RPRINTF(0,"%s\n","Freeing corrected nulspace solver.");

    SA_RPRINTF(0,"Cumulative CG iterations on spectral level: %d\n", cumulative_spectral_cg);

    // cn->interp is freed by tg_free_data, because that's who creates it (we just use it...)
    // delete cn->interp;
    
    delete restr;
    delete Ac;

    delete correct_nullspace_coarse;

    smpr_free_poly_data((smpr_poly_data_t*) smoother_data);
}

void CorrectNullspace::SetOperator(const Operator &op)
{
    A = const_cast<HypreParMatrix *>(dynamic_cast<const HypreParMatrix *>(&op));
    if (A == NULL)
        mfem_error("HypreSmoother::SetOperator : not HypreParMatrix!");
}

void CorrectNullspace::Mult(const Vector &b, Vector &x) const
{
    SA_ASSERT(correct_nullspace_coarse);
    SA_ASSERT(A);
    SA_ASSERT(Ac);
    SA_ASSERT(interp);
    SA_ASSERT(restr);

    if (!iterative_mode)
        x = 0.0; 

    // SA_RPRINTF(0,"---> OOP coarse_solve_correct_nulspace runs with size %d.\n", A->M());

    // second to last argument below is structure that gets passed to coarser (nulspace) solver
    // last argument below is structure that gets passed to smoother
    for (int i=0; i<spectral_cycles; ++i)
        tg_cycle_atb(*A, *Ac, *interp, *restr,
                     b, smoother, smoother,
                     x, *correct_nullspace_coarse, smoother_data); 
}


void solve_empty(HypreParMatrix& A, HypreParVector& b, HypreParVector& x,
                 void *data)
{
    x = 0.;
}


// Arguably this is more like SetOperator than a constructor...
AMGSolver::AMGSolver(HypreParMatrix& A, bool iterative_mode) :
    mfem::Solver(A.M(), iterative_mode)
{
    SA_RPRINTF(0,"solve_spd_AMG_init() runs with size %d\n", A.M());

    SA_ASSERT(CONFIG_ACCESS_OPTION(SOLVE, rtol) >= 0.);
    SA_ASSERT(CONFIG_ACCESS_OPTION(SOLVE, iters_coeff) >= 0.);

    // amg_data = new solve_amg_t;
    HypreBoomerAMG * hbamg = new HypreBoomerAMG(A);
    hbamg->SetPrintLevel(0);
    amg = hbamg;
    pcg = new HyprePCG(A);
    pcg->SetTol(CONFIG_ACCESS_OPTION(SOLVE, rtol));
    pcg->SetMaxIter((int)(CONFIG_ACCESS_OPTION(SOLVE, iters_coeff)*
                                A.GetGlobalNumRows()));
    pcg->SetPrintLevel(SA_IS_OUTPUT_LEVEL(15) ? 2 : 0);
    pcg->SetPreconditioner(*(amg));
    ref_cntr = 1;
    cumulative_iterations = 0;
}

AMGSolver::~AMGSolver()
{
    SA_RPRINTF(0,"Cumulative hypre-PCG iterations in coarsest solver: %d\n",cumulative_iterations);

    // not sure this ref_cntr business is important in OOP context
    SA_ASSERT(ref_cntr > 0);
    --(ref_cntr);
    if (ref_cntr <= 0)
    {
        delete pcg;
        delete amg;
        // delete amg_data;
    }
}

void AMGSolver::Mult(const Vector &b, Vector &x) const
{
    // SA_RPRINTF(0,"---> OOP solve_spd_AMG runs with size %d.\n", b.Size());

    if (!iterative_mode)
        x = 0.;
    pcg->Mult(b, x);
    int thisits = 0;
    // double finalnorm = 0.0;
    // GetNumIterations(int &num_iterations)
    pcg->GetNumIterations(thisits);
    cumulative_iterations += thisits;
}

VCycleSolver::VCycleSolver(tg_data_t * tg_data_in, bool iterative_mode) :
    Solver(tg_data_in->restr->N(), iterative_mode),
    tg_data(tg_data_in),
    A(NULL)
{
}

VCycleSolver::~VCycleSolver()
{
}

void VCycleSolver::SetOperator(const Operator &op)
{
    A = const_cast<HypreParMatrix *>(dynamic_cast<const HypreParMatrix *>(&op));
    // SA_ASSERT(op.Height() == height);
    if (A == NULL)
        mfem_error("HypreSmoother::SetOperator : not HypreParMatrix!");
}

void VCycleSolver::Mult(const Vector &b, Vector &x) const
{
    SA_ASSERT(tg_data);
    SA_ASSERT(A);
    SA_ASSERT(A->Width() == b.Size());
    SA_ASSERT(b.Size() == x.Size());

    if (!iterative_mode)
        x = 0.0;

    // SA_RPRINTF(0,"---> OOP VCycleSolver runs with size %d, tg_data->tag = %d.\n", A->M(), tg_data->tag);

    SA_ASSERT(tg_data->coarse_solver);
    tg_cycle_atb(*A, *(tg_data->Ac), *(tg_data->interp), *(tg_data->restr), b,
                 tg_data->pre_smoother, tg_data->post_smoother, x,
                 *tg_data->coarse_solver, tg_data->poly_data);
}

/**
   Note well that this has no corresponding init, free routines (uses tg_data, which
   is freed elsewhere, instead of amg_data)

   DEPRECATED
*/
void solve_spd_Vcycle(HypreParMatrix& A, const HypreParVector& b, HypreParVector& x,
                      void *data)
{
    tg_data_t *tg_data = (tg_data_t *)data;
    SA_ASSERT(tg_data);

    // SA_RPRINTF(0,"---> solve_spd_Vcycle runs with size %d, tg_data->tag = %d.\n", A.M(), tg_data->tag);

    // x = 0.; // whoever calls this now has to 0 x, ATB 29 May 2015
    tg_cycle_atb(A, *(tg_data->Ac), *(tg_data->interp), *(tg_data->restr), b,
                 tg_data->pre_smoother, tg_data->post_smoother, x,
                 *tg_data->coarse_solver, tg_data->poly_data);
}

void solve_spd_Wcycle(HypreParMatrix& A, const HypreParVector& b, HypreParVector& x,
                      void *data)
{
    SA_ASSERT(false);
    tg_data_t *tg_data = (tg_data_t *)data;
    SA_ASSERT(tg_data);

    // SA_RPRINTF(0,"---> solve_spd_Wcycle runs with size %d.\n", A.M());

    // x = 0.; // whoever calls this now has to 0 x, ATB 29 May 2015
    /*
    tg_cycle(A, *(tg_data->Ac), *(tg_data->interp), *(tg_data->restr), b,
             tg_data->pre_smoother, tg_data->post_smoother, x,
             tg_data->coarse_solver, tg_data->poly_data, 2);
    */
}
