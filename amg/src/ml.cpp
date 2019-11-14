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

/**
   @file Delyan Kalchev's multilevel (serial) interface,
   now made parallel.

   Andrew T. Barker
   atb@llnl.gov
   29 September 2015
*/

#include "common.hpp"
#include "ml.hpp"
#include "levels.hpp"
#include <mfem.hpp>
#include "contrib.hpp"
#include "tg.hpp"
#include "elmat.hpp"
#include "solve.hpp"

namespace saamge
{
using namespace mfem;

// not sure whether to copy nparts_arr or just use the given pointer
MultilevelParameters::MultilevelParameters(
    int coarsenings, int *nparts_arr_arg, int first_nu_pro, int nu_pro_arg, 
    int nu_relax_arg, double first_theta, double theta_arg,
    int polynomial_coarse_space_arg, bool use_correct_nullspace,
    bool use_arpack, bool do_aggregates, int fixed_num_evecs_f, int fixed_num_evecs_c)
    :
    num_coarsenings(coarsenings),
    use_correct_nullspace(use_correct_nullspace),
    use_arpack(use_arpack),
    do_aggregates(do_aggregates),
    avoid_ess_bdr_dofs(true),
    use_double_cycle(false),
    coarse_direct(false),
    smooth_drop_tol(0.0),
    svd_min_skip(0)
{
    nparts_arr = new int[num_coarsenings];
    nu_pro = new int[num_coarsenings];
    nu_relax = new int[num_coarsenings];
    theta = new double[num_coarsenings];
    fixed_num_evecs = new int[num_coarsenings];
    polynomial_coarse_space = new int[num_coarsenings];

    nparts_arr[0] = nparts_arr_arg[0];
    nu_pro[0] = first_nu_pro;
    nu_relax[0] = nu_relax_arg;
    theta[0] = first_theta;
    fixed_num_evecs[0] = fixed_num_evecs_f;
    polynomial_coarse_space[0] = polynomial_coarse_space_arg;

    for (int i=1; i<num_coarsenings; ++i)
    {
        nparts_arr[i] = nparts_arr_arg[i];
        nu_pro[i] = nu_pro_arg;
        nu_relax[i] = nu_relax_arg;
        theta[i] = theta_arg;
        fixed_num_evecs[i] = fixed_num_evecs_c;
        polynomial_coarse_space[i] = polynomial_coarse_space_arg;
    }
}

MultilevelParameters::MultilevelParameters(
    int num_coarsenings, int elems_per_agg, int nu_pro,
    int nu_relax, double theta, int polynomial_coarse_space)
    :
    num_coarsenings(num_coarsenings),
    use_correct_nullspace(true)
{
    SA_ASSERT(false);
}

MultilevelParameters::~MultilevelParameters()
{
    delete [] nparts_arr;
    delete [] nu_pro;
    delete [] nu_relax;
    delete [] theta;
    delete [] polynomial_coarse_space;
}


void ml_produce_hierarchy_from_level(
    int coarsenings, int starting_level, ml_data_t& ml_data,
    const MultilevelParameters &mlp)
{
    SA_ASSERT(coarsenings >= 0);

    agg_partitioning_relations_t *agg_part_rels;
    tg_data_t *tg_data;

    SA_ASSERT(1 <= ml_data.levels_list.num_levels);
    SA_ASSERT(ml_data.levels_list.coarsest);
    // at this point "coarsest" may actually be the finest, or next finest...
    agg_part_rels = ml_data.levels_list.coarsest->agg_part_rels;
    SA_ASSERT(agg_part_rels);
    tg_data = ml_data.levels_list.coarsest->tg_data;
    SA_ASSERT(tg_data);
    SA_ASSERT(tg_data->interp_data);

    // Build levels.
    for (int i=starting_level; i < coarsenings; ++i)
    {
        SA_ASSERT(tg_data->Ac);
        HypreParMatrix * A = tg_data->Ac; 
        const int level = ml_data.levels_list.num_levels;
        if (SA_IS_OUTPUT_LEVEL(5))
        {
            SA_RPRINTF(0,"%s","\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/"
                      "\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/"
                      "\\/\\/\\/\\/\n");
            SA_RPRINTF(0,"Coarsening: %d -> %d ...\n", level, level+1);
            SA_RPRINTF(0,"%s","\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/"
                      "\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/"
                      "\\/\\/\\/\\/\n");
        }

        int nparts = mlp.get_nparts(i);
        // could use regular (smoothed) interp, but tent_interp is default in serial SAAMGE, we focus on it for now
        bool do_aggregates = (mlp.get_do_aggregates() && (i == coarsenings-1));
        agg_part_rels = agg_create_partitioning_coarse(
            A, *agg_part_rels, tg_data->interp_data->coarse_truedof_offset,
            tg_data->interp_data->mis_numcoarsedof,
            tg_data->interp_data->mis_tent_interps, tg_data->tent_interp,
            &nparts, do_aggregates);
        SA_ASSERT(agg_part_rels);
        if (agg_part_rels->testmesh)
        {
            std::stringstream filename;
            filename << "coarse_AE_to_elem." << PROC_RANK << ".table";
            std::ofstream out(filename.str().c_str());
            agg_part_rels->AE_to_elem->Print(out);
            filename.str("");
            filename << "coarse_AE_to_dof." << PROC_RANK << ".table";
            std::ofstream out2(filename.str().c_str());
            agg_part_rels->AE_to_dof->Print(out2);
            filename.str("");
            filename << "coarse_dof_id_inAE." << PROC_RANK << ".array";
            std::ofstream out3(filename.str().c_str());
            for (int j=0; j<agg_part_rels->dof_to_AE->Size_of_connections(); ++j)
                out3 << agg_part_rels->dof_id_inAE[j] << std::endl;
        }

        tg_data = tg_init_data(
            *A, *agg_part_rels, mlp.get_nu_pro(i), mlp.get_nu_relax(i),
            mlp.get_theta(i), mlp.get_smooth_interp(i),
            mlp.get_smooth_drop_tol(), mlp.get_use_arpack());
        SA_ASSERT(tg_data);
        SA_ASSERT(tg_data->interp_data);

        tg_data->use_w_cycle = false;
        tg_data->polynomial_coarse_space = mlp.get_polynomial_coarse_space(i);

        if (mlp.get_use_correct_nullspace() &&
            i == coarsenings-1)
        {
            tg_data->interp_data->scaling_P = true;
        }
        ElementMatrixProvider * emp = new ElementMatrixParallelCoarse(
            *agg_part_rels, ml_data.levels_list.coarsest);
        tg_build_hierarchy(*A, *tg_data, *agg_part_rels,
                           emp, mlp.get_avoid_ess_bdr_dofs(), mlp.get_fixed_num_evecs(i), mlp.get_svd_min_skip());

        if (agg_part_rels->testmesh)
        {
            std::stringstream filename;
            filename << "tent_interp_l" << level << ".mat";
            tg_data->tent_interp->Print(filename.str().c_str());
            filename.str("");
            filename << "interp_l" << level << ".mat";
            tg_data->interp->Print(filename.str().c_str());
        }

        SA_ASSERT(!tg_data->Ac);
        tg_update_coarse_operator(*A, tg_data, i+1 == coarsenings, mlp.get_coarse_direct());
        if (false)
        {
            std::stringstream s;
            s << "operator" << 1 + i << ".mat";
            tg_data->Ac->Print(s.str().c_str());
        }

        levels_list_push_coarse_data(ml_data.levels_list, agg_part_rels,
                                     tg_data);
        SA_ASSERT(ml_data.levels_list.finest != ml_data.levels_list.coarsest);
        SA_ASSERT(level+1 == ml_data.levels_list.num_levels);
    }
    SA_ASSERT(levels_check_list(ml_data.levels_list));
    SA_RPRINTF_L(0,5, "LEVELS in levels_list = %d\n", ml_data.levels_list.num_levels);
    SA_RPRINTF_L(0,5, "%s",
                 "END \\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/"
                 "\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/"
                 "\\/\n");

    ml_impose_cycle(ml_data,false);
    if (mlp.get_use_correct_nullspace())
    {
        tg_data = ml_data.levels_list.coarsest->tg_data;
        SA_ASSERT(tg_data);
        SA_ASSERT(tg_data->Ac);
        if (tg_data->coarse_solver)
            delete tg_data->coarse_solver;
        tg_data->coarse_solver = new CorrectNullspace(*tg_data->Ac,
                                                      tg_data->scaling_P,
                                                      3, false, true, false);
    }
}

double ml_compute_OC_from_level(HypreParMatrix& A, const ml_data_t& ml_data,
                                levels_level_t *starting_level)
{
    double sum = 0.;

    levels_level_t *level;
    for (level = starting_level; level; level = level->coarser)
    {
        SA_ASSERT(level->tg_data->Ac);
        sum += (double)level->tg_data->Ac->NNZ() /
               (double)A.NNZ();
    }
    return (1. + sum);
}

double ml_compute_OC(HypreParMatrix& A, const ml_data_t& ml_data)
{
    double sum = 0.;

    levels_level_t *level;
    for (level = ml_data.levels_list.finest; level; level = level->coarser)
    {
        SA_ASSERT(level->tg_data->Ac);
        sum += (double)level->tg_data->Ac->NNZ() /
               (double)A.NNZ();
    }
    return (1. + sum);
}

double ml_compute_OC_for_level(HypreParMatrix& A, const ml_data_t& ml_data,
                               int level)
{
    SA_ASSERT(1 <= level && level <= ml_data.levels_list.num_levels);
    const levels_level_t *lev = levels_list_get_level(ml_data.levels_list,
                                                      level - 1);
    SA_ASSERT(lev);
    SA_ASSERT(lev->tg_data);
    SA_ASSERT(lev->tg_data->Ac);
    SA_ASSERT((level > 1 && lev->finer && lev->finer->tg_data &&
               lev->finer->tg_data->Ac) || (1 == level && !lev->finer));
    const int finennz = (level <= 1) ? A.NNZ() :
                            lev->finer->tg_data->Ac->NNZ();
    const int coarsennz = lev->tg_data->Ac->NNZ();

    return(1. + (double)coarsennz / (double)finennz);
}

void ml_get_dims(const ml_data_t& ml_data, Array<int>& dims)
{
    levels_level_t *level;
    
    dims.Append(ml_data.levels_list.finest->tg_data->interp->M());
    for (level = ml_data.levels_list.finest; level; level = level->coarser)
    {
        dims.Append(level->tg_data->interp->N());
    }
}

void ml_print_dims(HypreParMatrix& A, const ml_data_t& ml_data)
{
    levels_level_t *level;
    int i=0;

    SA_RPRINTF(0,"%s","Number of levels: ");
    SA_RPRINTF(0,"%d\n", ml_data.levels_list.num_levels + 1);
    /*
    if (!ml_data.levels_list.num_levels)
    {
        SA_RPRINTF_NOTS(0,"%s","0\n");
        return;
    }
    SA_RPRINTF_NOTS(0,"%d\n", ml_data.levels_list.num_levels + 1);
    */

    SA_ASSERT(A.GetGlobalNumRows() == A.GetGlobalNumCols());
    SA_ASSERT(ml_data.levels_list.finest->tg_data->interp->M() ==
              A.GetGlobalNumRows());
    SA_RPRINTF(0,"Level 0 dimension: %d, Operator nnz: %d\n",
               ml_data.levels_list.finest->tg_data->interp->M(),
               A.NNZ());

    for (level = ml_data.levels_list.finest; level; level = level->coarser)
    {
        SA_ASSERT(level->tg_data->Ac);
        SA_ASSERT(level->tg_data->Ac->M() == level->tg_data->Ac->N());
        SA_ASSERT(level->tg_data->Ac->M() ==
                  level->tg_data->interp->N());
        SA_RPRINTF(0,"Level %d dimension: %d, Operator nnz: %d\n", ++i,
                   level->tg_data->interp->N(),
                   level->tg_data->Ac->NNZ());
    }
}

double ml_print_data(HypreParMatrix& A, const ml_data_t& ml_data)
{
    int i;
    levels_level_t *level;
    SA_RPRINTF(0,"%s",">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n");
    SA_RPRINTF(0,"%s","\tMultilevel data:\n");
    ml_print_dims(A, ml_data);
    for (i=1; i <= ml_data.levels_list.num_levels; ++i)
    {
        SA_RPRINTF(0,"Coarsening %d -> %d operator complexity: %g\n", i-1, i,
                  ml_compute_OC_for_level(A, ml_data, i));
    }
    double overall_complexity = ml_compute_OC(A, ml_data);
    SA_RPRINTF(0,"Overall operator complexity: %g\n", overall_complexity);
    // SA_ASSERT(overall_complexity < 3.0); // let's not waste our time...

    for ((level = ml_data.levels_list.finest), (i=0); level;
         (level = level->coarser), (++i))
    {
        SA_ASSERT(level->agg_part_rels);
        agg_print_data(*level->agg_part_rels);
    }
    SA_RPRINTF(0,"%s","<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n");
    return overall_complexity;
}

/**
   sets all but the coarsest level to do V-cycle
   (coarsest level normally does solve_spd_amg())
*/
void ml_impose_cycle(ml_data_t& ml_data, bool Wcycle)
{
    SA_ASSERT(!Wcycle);

    levels_level_t *level;
    int i = 0;
    for (level = ml_data.levels_list.finest; level && level->coarser;
         level = level->coarser)
    {
        level->tg_data->tag = i;
        level->tg_data->coarse_solver = new VCycleSolver(level->coarser->tg_data,false);
        level->tg_data->coarse_solver->SetOperator(*level->tg_data->Ac);
        ++i;
    }
    level = ml_data.levels_list.coarsest;
    level->tg_data->tag = i;
}

ml_data_t * ml_produce_data(
    HypreParMatrix& Ag, agg_partitioning_relations_t *agg_part_rels, 
    ElementMatrixProvider *elem_data_finest, const MultilevelParameters &mlp)
{
    SA_ASSERT(elem_data_finest);
    ml_data_t *ml_data = new ml_data_t;
    SA_ASSERT(ml_data);
    memset(ml_data, 0, sizeof(*ml_data));

    SA_ASSERT(mlp.get_num_coarsenings() > 0);
    SA_ASSERT(agg_part_rels);

    // Coarsen the finest level.
    SA_RPRINTF_L(0,4,"%s", "---------- ml_produce_all { ---------------------\n");
    if (SA_IS_OUTPUT_LEVEL(5))
    {
        SA_RPRINTF(0,"%s","\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/"
                  "\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/"
                  "\\/\\/\\/\\/\n");
        SA_RPRINTF(0,"%s","Coarsening: 0 -> 1 ...\n");
        SA_RPRINTF(0,"%s","\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/"
                  "\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/"
                  "\\/\\/\\/\\/\n");
    }

    tg_data_t *tg_data = tg_init_data(
        Ag, *agg_part_rels, mlp.get_nu_pro(0), mlp.get_nu_relax(0),
        mlp.get_theta(0), mlp.get_smooth_interp(0), mlp.get_smooth_drop_tol(),
        mlp.get_use_arpack());
    SA_ASSERT(tg_data);
    SA_ASSERT(tg_data->interp_data);
    
    tg_data->use_w_cycle = false;
    tg_data->polynomial_coarse_space = mlp.get_polynomial_coarse_space(0);

    if (mlp.get_use_correct_nullspace() && 
        (1 == mlp.get_num_coarsenings() || mlp.get_use_double_cycle()) )
    {
        tg_data->interp_data->scaling_P = true;
    }

    if (tg_data->polynomial_coarse_space == 0 ||
        tg_data->polynomial_coarse_space == 1)
    {
        ElementMatrixStandardGeometric * elmat_geom = 
            dynamic_cast<ElementMatrixStandardGeometric*>(elem_data_finest);
        SA_ASSERT(elmat_geom);
        ParBilinearForm * pform = elmat_geom->GetParBilinearForm();
        ParFiniteElementSpace * pfes = pform->ParFESpace();
        ParMesh * pmesh = pfes->GetParMesh();
        // whew, getting that pmesh was a lot of work...
        bool use_spectral;
        if (mlp.get_theta(0) <= 0.0)
            use_spectral = false;
        else
            use_spectral = true;

        tg_build_hierarchy_with_polynomial(
            Ag, *pmesh, *tg_data, *agg_part_rels,
            elem_data_finest, tg_data->polynomial_coarse_space,
            use_spectral, mlp.get_avoid_ess_bdr_dofs());
    }
    else
    {
        tg_build_hierarchy(Ag, *tg_data, *agg_part_rels,
                           elem_data_finest, mlp.get_avoid_ess_bdr_dofs(), mlp.get_fixed_num_evecs(0), mlp.get_svd_min_skip());
    }

    if (agg_part_rels->testmesh)
    {
        tg_data->tent_interp->Print("tent_interp_l0.mat");
        tg_data->interp->Print("interp_l0.mat");
    }

    SA_ASSERT(!tg_data->Ac);
    tg_update_coarse_operator(Ag, tg_data, 1 >= mlp.get_num_coarsenings(),
                              mlp.get_coarse_direct());

    levels_list_push_coarse_data(ml_data->levels_list, agg_part_rels, tg_data);
    SA_ASSERT(ml_data->levels_list.finest == ml_data->levels_list.coarsest);
    SA_ASSERT(1 == ml_data->levels_list.num_levels);

    // Build all other levels.
    ml_produce_hierarchy_from_level(mlp.get_num_coarsenings(), 1, *ml_data, mlp);

    if (SA_IS_OUTPUT_LEVEL(3))
    {
        ml_print_data(Ag, *ml_data);
    }

    SA_RPRINTF_L(0,4, "%s", "---------- } ml_produce_all ---------------------\n");

    return ml_data;
}

void ml_free_data(ml_data_t *ml_data)
{
    if (!ml_data) return;
    if (ml_data->levels_list.num_levels > 1)
        levels_list_free_level_and_all_coarser(ml_data->levels_list, 1);

    // on finest level we let caller free agg_part_rels, because they created it
    levels_level_t *level = ml_data->levels_list.finest;
    tg_free_data(level->tg_data);
    levels_free_level_struct(level);

    delete ml_data;
}

int ml_run(HypreParMatrix& A, HypreParVector& x, HypreParVector& b, int maxiter,
           double rtol, double atol, double reducttol, ml_data_t& ml_data,
           bool zero_rhs, int from_level)
{
    SA_RPRINTF_L(0,4,"%s","---------- ml_run { -----------------------\n");

    if (!ml_data.levels_list.num_levels)
        return 0;

    SA_ASSERT(0 <= from_level && from_level < ml_data.levels_list.num_levels);
    const levels_level_t *level = levels_list_get_level(ml_data.levels_list,
                                                        from_level);
    SA_ASSERT(level);
    if (!level)
        return 0;

    SA_ASSERT(level->tg_data);
    SA_ASSERT(level->tg_data->Ac);
    SA_ASSERT(A.M() == A.N());
    SA_ASSERT(A.M() == level->tg_data->interp->M());

    int ml_iters = tg_run(A, level->agg_part_rels, x, b, maxiter, rtol, atol,
                          reducttol, level->tg_data, zero_rhs);

    SA_RPRINTF_L(0,4,"%s","---------- } ml_run -----------------------\n");

    return ml_iters;
}

/// not ever called? deprecated?
int ml_pcg_run(HypreParMatrix& A, HypreParVector& x, HypreParVector& b, int maxiter,
               double rtol, double atol, ml_data_t& ml_data, bool zero_rhs,
               int from_level)
{
    SA_RPRINTF_L(0,4,"%s","---------- ml_pcg_run { -----------------------\n");

    if (!ml_data.levels_list.num_levels)
        return 0;

    SA_ASSERT(0 <= from_level && from_level < ml_data.levels_list.num_levels);
    const levels_level_t *level = levels_list_get_level(ml_data.levels_list,
                                                        from_level);
    SA_ASSERT(level);
    if (!level)
        return 0;

    SA_ASSERT(level->tg_data);
    SA_ASSERT(level->tg_data->Ac);
    SA_ASSERT(A.M() == A.N());
    SA_ASSERT(A.M() == level->tg_data->interp->M());

    int pcg_iters = tg_pcg_run(A, level->agg_part_rels, x, b, maxiter, rtol,
                               atol, level->tg_data, zero_rhs);

    SA_RPRINTF_L(0,4,"%s","---------- } ml_pcg_run -----------------------\n");

    return pcg_iters;
}

} // namespace saamge
