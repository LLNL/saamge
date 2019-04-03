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
#include "solve.hpp"
#include <mfem.hpp>
#include "xpacks.hpp"
#include "mbox.hpp"
#include "tg.hpp"
#include "levels.hpp"
#include "fem.hpp"
#include "mfem_addons.hpp"

namespace saamge
{
using namespace mfem;

/* Functions */

/**
  A in this context is the coarse element agglomeration matrix, Ac is a slightly
  coarser matrix with corrected nulspace so that Hypre will work well
*/
CorrectNullspace::CorrectNullspace(
    HypreParMatrix& A, HypreParMatrix * interp_arg, int smoother_steps,
    bool smooth_phat, bool v_cycle, bool iterative_mode)
    :
    Solver(A.M(), iterative_mode),
    smoother_steps(smoother_steps),
    smooth_phat(smooth_phat),
    interp(interp_arg)
{
    SA_ASSERT(interp);
    SA_RPRINTF_L(0,6,"%s\n","Initializing corrected nulspace solver.");

    cumulative_spectral_cg = 0;
    spectral_cycles = 1;

    // A is matrix at spectral level
    SA_RPRINTF_L(0,8,"  [correctnulspace] A is %d by %d\n",A.M(), A.N());
    SA_RPRINTF_L(0,8,"  [correctnulspace] interp is %d by %d\n",
               interp->M(), interp->N());

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
        HypreParMatrix * smoothed_interp = interp_smooth(
            degree, roots, 0.0, 1, A, *interp, *sd->Dinv_neg);
        delete[] roots;
        delete interp;
        interp = smoothed_interp;
    }
    sw.Stop();

    // construct matrix Ac at nulspace level
    SA_ASSERT(interp->M() == A.N());
    restr = interp->Transpose();
    sw.Clear();
    sw.Start();
    Ac = RAP(&A,interp);
    sw.Stop();
    SA_RPRINTF_L(0, 9, "  [correctnulspace] Ac is %d by %d\n", Ac->M(), Ac->N());
    SA_RPRINTF_L(0, 9, "  [correctnulspace] operator complexity is %.2f\n", 
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
    SA_RPRINTF_L(0, 5, "Cumulative CG iterations on spectral level: %d\n", cumulative_spectral_cg);

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
    {
        tg_cycle_atb(*A, *Ac, *interp, *restr,
                     b, smoother, smoother,
                     x, *correct_nullspace_coarse, smoother_data);
    }
}

/// Why do we need a HypreParMatrix, a ParBilinearForm, and a SparseMatrix that all do the same thing?
SpectralAMGSolver::SpectralAMGSolver(mfem::HypreParMatrix& Ag,
                                     mfem::ParBilinearForm& aform,
                                     mfem::SparseMatrix& Alocal,
                                     mfem::Array<int>& ess_bdr,
                                     int elems_per_agg, int num_levels,
                                     int nu_pro, int nu_relax, double theta,
                                     int polynomial_coarse, bool coarse_direct)
    :
    Ag_(Ag)
{
    nparts_arr_ = new int[num_levels-1];
    ParFiniteElementSpace* fes = aform.ParFESpace();
    agg_dof_status_t* bdr_dofs = fem_find_bdr_dofs(*fes, &ess_bdr);
    int first_elems_per_agg = elems_per_agg;
    nparts_arr_[0] = fes->GetParMesh()->GetNE() / first_elems_per_agg;
    for (int i=1; i < num_levels-1; ++i)
    {
        nparts_arr_[i] = (int) round((double) nparts_arr_[i-1] / (double) elems_per_agg);
        if (nparts_arr_[i] < 1) nparts_arr_[i] = 1;
    }

    const bool do_aggregates_here = false;
    agg_part_rels_ = fem_create_partitioning(
        Ag, *fes, bdr_dofs, nparts_arr_, do_aggregates_here);
    delete [] bdr_dofs;

    // emp deleted in ml_free_data()
    ElementMatrixProvider * emp =
        new ElementMatrixStandardGeometric(*agg_part_rels_, Alocal, &aform);
    const bool correct_nulspace = (polynomial_coarse == -1);
    const bool direct_eigensolver = false;
    const bool do_aggregates = false;
    MultilevelParameters mlp(
        num_levels-1, nparts_arr_, nu_pro, nu_pro, nu_relax, theta,
        theta, polynomial_coarse, correct_nulspace, !direct_eigensolver,
        do_aggregates);
    if (coarse_direct)
        mlp.set_coarse_direct(true);
    ml_data_ = ml_produce_data(Ag_, agg_part_rels_, emp, mlp);

    levels_level_t* level = levels_list_get_level(ml_data_->levels_list, 0);
    v_cycle_ = new VCycleSolver(level->tg_data, false);
    v_cycle_->SetOperator(Ag_);
}

SpectralAMGSolver::~SpectralAMGSolver()
{
    delete [] nparts_arr_; // this one probably doesn't go here...

    ml_free_data(ml_data_);
    agg_free_partitioning(agg_part_rels_);
    delete v_cycle_;
}

void SpectralAMGSolver::SetOperator(const mfem::Operator &op)
{
    // implemented in constructor, more or less
    // (should be possible to change operator here?)
}

void SpectralAMGSolver::Mult(const Vector &x, Vector &y) const
{
    v_cycle_->Mult(x, y);
}

void solve_empty(HypreParMatrix& A, HypreParVector& b, HypreParVector& x,
                 void *data)
{
    x = 0.;
}


// Arguably this is more like SetOperator than a constructor...
AMGSolver::AMGSolver(HypreParMatrix& A, bool iterative_mode, double rel_tol, int iters_coeff) :
    mfem::Solver(A.M(), iterative_mode),
    rel_tol(rel_tol),
    iters_coeff(iters_coeff)
{
    SA_RPRINTF(0,"solve_spd_AMG_init() runs with size %d\n", A.M());

    // amg_data = new solve_amg_t;
    HypreBoomerAMG * hbamg = new HypreBoomerAMG(A);
    hbamg->SetPrintLevel(0);
    amg = hbamg;
    pcg = new HyprePCG(A);
    pcg->SetMaxIter((int)(iters_coeff*A.GetGlobalNumRows()));
    pcg->SetPrintLevel(SA_IS_OUTPUT_LEVEL(15) ? 2 : 0);
    pcg->SetPreconditioner(*(amg));
    ref_cntr = 1; // ref_cntr is a legacy of doing fake C object-oriented stuff
    cumulative_iterations = 0;
}

AMGSolver::~AMGSolver()
{
    SA_RPRINTF(0,"Cumulative hypre-PCG iterations in coarsest solver: %d\n",
               cumulative_iterations);

    // probably this ref_cntr business should be removed
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
    pcg->SetTol(rel_tol);
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
        mfem_error("VCycleSolver::SetOperator : not HypreParMatrix!");
}

void VCycleSolver::Mult(const Vector &b, Vector &x) const
{
    SA_ASSERT(tg_data);
    SA_ASSERT(A);
    SA_ASSERT(A->Width() == b.Size());
    SA_ASSERT(b.Size() == x.Size());

    if (!iterative_mode)
        x = 0.0;

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

} // namespace saamge
