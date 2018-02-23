/*
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

/**
   bare bones, piecemeal copy of Delyan Kalchev's multilevel
   (serial) interface

   Andrew T. Barker
   atb@llnl.gov
   24 April 2015
*/

#pragma once
#ifndef _ML_HPP
#define _ML_HPP

#include "common.hpp"
#include "levels.hpp"
#include <mfem.hpp>
using namespace mfem;
#include "tg.hpp"
#include "solve.hpp"

/*! \brief Multilevel parameters

  This is just a way to pass a bunch of parameters down
  the multilevel hierarchy, without having to always think about them,
  that is, sensible default constructors but also ways to get
  more fine-grained control.
*/
class MultilevelParameters
{
public:
    /**
       Constructor following old ml_produce_data calling sequence.
    */
    MultilevelParameters(int coarsenings, int *nparts_arr, int first_nu_pro, int nu_pro, 
                         int nu_relax, double first_theta, double theta, bool minimal_coarse_space,
                         bool use_correct_nullspace, bool use_arpack, bool do_aggregates);

    /**
       Constructor so every level has same parameters.

       NOT IMPLEMENTED.
    */
    MultilevelParameters(int num_coarsenings, int elems_per_agg, int nu_pro, 
                         int nu_relax, double theta, bool minimal_coarse_space);

    ~MultilevelParameters();

    int get_num_coarsenings() const {return num_coarsenings;}
    int get_nu_pro(int j) const {return nu_pro[j];}
    int get_nu_relax(int j) const {return nu_relax[j];}
    double get_theta(int j) const {return theta[j];}
    bool get_smooth_interp(int j) const {return (nu_pro[j] > 0);}
    bool get_minimal_coarse_space(int j) const {return minimal_coarse_space[j];}
    bool get_use_correct_nullspace() const {return use_correct_nullspace;}
    bool get_use_arpack() const {return use_arpack;}
    bool get_do_aggregates() const {return do_aggregates;}
    int get_nparts(int j) const {return nparts_arr[j];}

    void set_minimal_coarse_space(int j, bool val) {minimal_coarse_space[j] = val;}
    
private:
    int num_coarsenings;
    int * nparts_arr;
    int * nu_pro;
    int * nu_relax;
    double * theta;
    bool * minimal_coarse_space;
    // TODO: something reasonable with use_correct_nulspace
    bool use_correct_nullspace;
    bool use_arpack;
    bool do_aggregates;
};

/*! \brief Multilevel data.
*/
typedef struct {
    levels_list_t levels_list; /*!< List of levels. */
} ml_data_t;

/*! \brief Fully build the hierarchy including partitions and relations.

    Fully build the hierarchy including partitions and relations at the tail of
    the levels. That is, starting from the coarsest level already in the
    hierarchy. This function cannot coarsen a geometric level.

    All arrays must be of size \a coarsenings.

    \param coarsenings (IN) How many coarsening will be performed to add up to
                            the hierarchy.
    \param ml_data (IN/OUT) The ML structure.
*/
void ml_produce_hierarchy_from_level(
    int coarsenings, int starting_level, ml_data_t& ml_data, const MultilevelParameters &mlp);

/*! \brief Compute OC for a hierarchy where we regard starting_level as the finest level.

  used in startfromcoarse.cpp
*/
double ml_compute_OC_from_level(HypreParMatrix& A, const ml_data_t& ml_data,
                                levels_level_t *starting_level);

/*! \brief Computes OC over the entire hierarchy.

    \param A (IN) The finest operator.
    \param ml_data (IN) Multilevel data.

    \returns The operator complexity.
*/
double ml_compute_OC(HypreParMatrix& A, const ml_data_t& ml_data);

/*! \brief Computes OC for one coarsening \a level is the coarse level.

    Computes the OC of the coarsening \a level - 1 -> \a level.

    \param A (IN) The finest operator.
    \param ml_data (IN) Multilevel data.
    \param level (IN) The level.

    \returns The operator complexity.
*/
double ml_compute_OC_for_level(HypreParMatrix& A, const ml_data_t& ml_data,
                               int level);

/*! \brief put dimensions of the operators at each level into dims
 */
void ml_get_dims(const ml_data_t& ml_data, Array<int>& dims);

/*! \brief Prints the dimensions operator NNZ of all levels in the hierarchy.

    \param A (IN) The finest operator.
    \param ml_data (IN) Multilevel data.
*/
void ml_print_dims(HypreParMatrix& A, const ml_data_t& ml_data);

/*! \brief Prints information about the hierarchy.

    \param A (IN) The finest operator.
    \param ml_data (IN) Multilevel data.
*/
double ml_print_data(HypreParMatrix& A, const ml_data_t& ml_data);

void ml_impose_cycle(ml_data_t& ml_data, bool Wcycle);

/*!
  Fill in the ml_data data structure with all the many parameters.

  Note that elem_data_finest will be freed by some tg_data object that this calls.
  The caller should free agg_part_rels.
*/
ml_data_t * ml_produce_data(
    const SparseMatrix& Al, HypreParMatrix& Ag,
    agg_partitioning_relations_t *agg_part_rels, 
    ElementMatrixProvider *elem_data_finest,
    const MultilevelParameters &mlp);

void ml_free_data(ml_data_t *ml_data);

/*! \brief Executes a stationary ML-cycle method.

    This function runs a stationary multilevel solver to solve a given problem.

    \param A (IN) The matrix of the system being solved (usually the global
                  stiffness matrix). It must correspond to \a from_level.
    \param x (IN/OUT) Outputs the solution approximation. As input it is the
                      initial approximation. If \a zero_rhs == \em true, then
                      the initial approximation is generated randomly.
    \param b (IN) The right-hand side. Not used if \a zero_rhs == \em true.
    \param maxiter (IN) The maximal number of iteration to be done.
    \param rtol (IN) Relative tolerance. Depending on \a zero_rhs, it may be
                     used for the error or for the residual (see \b tg_solve
                     and \b adapt_approx_xbad).
    \param atol (IN) Absolute tolerance. Depending on \a zero_rhs, it may be
                     used for the error or for the residual (see \b tg_solve
                     and \b adapt_approx_xbad).
    \param reducttol (IN) See \b tg_solve (only for calls that are directed to
                          \b tg_solve this argument is relevant).
    \param ml_data (IN) This is the ML data to be used.
    \param zero_rhs (IN) If it is \em true, then \b adapt_approx_xbad is used
                         since it outputs more error-related information.
                         Otherwise, \b tg_solve is used.
    \param from_level (IN) The starting level, where \a A lives. Have in mind
                           that it might be necessary to execute, say,
                           \b ml_update_operators_from_level_down prior to this
                           call, so that the coarse operators in the hierarchy
                           would correspond to \a A. This might create a mess
                           in the operators in the hierarchy -- they may
                           come from different matrices on different levels.
                           You should keep track of this and know what you are
                           doing.

    \returns The number of iterations done. If a desired convergence criteria
             was not reached, then this number is negative.

    DEPRECATED, this is never called, probably use VCycleSolver class or something similar instead
*/
int ml_run(HypreParMatrix& A, HypreParVector& x, HypreParVector& b, int maxiter,
           double rtol, double atol, double reducttol, ml_data_t& ml_data,
           bool zero_rhs, int from_level);

/*! \brief Executes PCG preconditioned by a ML-cycle.

    This function runs a PCG solver to solve a given problem.

    \param A (IN) The matrix of the system being solved (usually the global
                  stiffness matrix). It must correspond to \a from_level.
    \param x (IN/OUT) Outputs the solution approximation. As input it is the
                      initial approximation. If \a zero_rhs == \em true, then
                      the initial approximation is generated randomly.
    \param b (IN) The right-hand side. Not used if \a zero_rhs == \em true.
    \param maxiter (IN) The maximal number of iteration to be done.
    \param rtol (IN) Relative tolerance.
    \param atol (IN) Absolute tolerance.
    \param ml_data (IN) This is the ML data to be used.
    \param zero_rhs (IN) If it is \em true, then it outputs more error-related
                         information.
    \param from_level (IN) The starting level, where \a A lives. Have in mind
                           that it might be necessary to execute, say,
                           \b ml_update_operators_from_level_down prior to this
                           call, so that the coarse operators in the hierarchy
                           would correspond to \a A. This might create a mess
                           in the operators in the hierarchy -- they may
                           come from different matrices on different levels.
                           You should keep track of this and know what you are
                           doing.

    \returns The number of iterations done. If a desired convergence criteria
             was not reached, then this number is negative.

    DEPRECATED, try mfem::CGSolver with VCycleSolver as a preconditioner
*/
int ml_pcg_run(HypreParMatrix& A, HypreParVector& x, HypreParVector& b, int maxiter,
               double rtol, double atol, ml_data_t& ml_data, bool zero_rhs,
               int from_level);


#endif // _ML_HPP
