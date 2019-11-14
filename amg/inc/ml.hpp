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
   Multilevel interface, does some of the things as tg.hpp
   but in multilevel context.

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
#include "tg.hpp"

namespace saamge
{

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
    MultilevelParameters(
        int coarsenings, int *nparts_arr, int first_nu_pro, int nu_pro,
        int nu_relax, double first_theta, double theta,
        int polynomial_coarse_space, bool use_correct_nullspace, bool use_arpack,
        bool do_aggregates, int fixed_num_evecs_f=0, int fixed_num_evecs_c=0);

    /**
       Constructor so every level has same parameters.

       @todo NOT IMPLEMENTED.
    */
    MultilevelParameters(int num_coarsenings, int elems_per_agg, int nu_pro, 
                         int nu_relax, double theta, int polynomial_coarse_space);

    ~MultilevelParameters();

    int get_num_coarsenings() const {return num_coarsenings;}
    int get_nu_pro(int j) const {return nu_pro[j];}
    int get_nu_relax(int j) const {return nu_relax[j];}
    double get_theta(int j) const {return theta[j];}
    double get_fixed_num_evecs(int j) const {return fixed_num_evecs[j];}
    bool get_smooth_interp(int j) const {return (nu_pro[j] > 0);}
    int get_polynomial_coarse_space(int j) const {return polynomial_coarse_space[j];}
    bool get_use_correct_nullspace() const {return use_correct_nullspace;}
    bool get_use_arpack() const {return use_arpack;}
    bool get_do_aggregates() const {return do_aggregates;}
    int get_nparts(int j) const {return nparts_arr[j];}
    bool get_avoid_ess_bdr_dofs() const {return avoid_ess_bdr_dofs;}
    bool get_use_double_cycle() const {return use_double_cycle;}
    double get_smooth_drop_tol() const {return smooth_drop_tol;}

    void set_polynomial_coarse_space(int j, int val) {polynomial_coarse_space[j] = val;}
    void set_use_double_cycle(bool use) {use_double_cycle = use;}
    bool get_coarse_direct() const {return coarse_direct;}
    void set_coarse_direct(bool cd) {coarse_direct = cd;}
    void set_smooth_drop_tol(double tol) {smooth_drop_tol = tol;}

    int get_svd_min_skip() const {return svd_min_skip;}
    void set_svd_min_skip(int svd_ms) {svd_min_skip = svd_ms;}
private:
    int num_coarsenings;
    int * nparts_arr;
    int * nu_pro;
    int * nu_relax;
    double * theta;
    int * fixed_num_evecs;
    int * polynomial_coarse_space;
    bool use_correct_nullspace;
    bool use_arpack;
    bool do_aggregates;
    bool avoid_ess_bdr_dofs;
    bool use_double_cycle;
    bool coarse_direct; // use direct solver on coarsest level
    double smooth_drop_tol;
    int svd_min_skip;
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
double ml_compute_OC_from_level(mfem::HypreParMatrix& A, const ml_data_t& ml_data,
                                levels_level_t *starting_level);

/*! \brief Computes OC over the entire hierarchy.

    \param A (IN) The finest operator.
    \param ml_data (IN) Multilevel data.

    \returns The operator complexity.
*/
double ml_compute_OC(mfem::HypreParMatrix& A, const ml_data_t& ml_data);

/*! \brief Computes OC for one coarsening \a level is the coarse level.

    Computes the OC of the coarsening \a level - 1 -> \a level.

    \param A (IN) The finest operator.
    \param ml_data (IN) Multilevel data.
    \param level (IN) The level.

    \returns The operator complexity.
*/
double ml_compute_OC_for_level(mfem::HypreParMatrix& A, const ml_data_t& ml_data,
                               int level);

/*! \brief put dimensions of the operators at each level into dims
 */
void ml_get_dims(const ml_data_t& ml_data, mfem::Array<int>& dims);

/*! \brief Prints the dimensions operator NNZ of all levels in the hierarchy.

    \param A (IN) The finest operator.
    \param ml_data (IN) Multilevel data.
*/
void ml_print_dims(mfem::HypreParMatrix& A, const ml_data_t& ml_data);

/*! \brief Prints information about the hierarchy.

    \param A (IN) The finest operator.
    \param ml_data (IN) Multilevel data.
*/
double ml_print_data(mfem::HypreParMatrix& A, const ml_data_t& ml_data);

void ml_impose_cycle(ml_data_t& ml_data, bool Wcycle);

/*!
  Fill in the ml_data data structure with all the many parameters.

  Note that elem_data_finest will be freed by some tg_data object that this calls.
  The caller should free agg_part_rels.
*/
ml_data_t * ml_produce_data(
    mfem::HypreParMatrix& Ag, agg_partitioning_relations_t *agg_part_rels, 
    ElementMatrixProvider *elem_data_finest, const MultilevelParameters &mlp);

void ml_free_data(ml_data_t *ml_data);

} // namespace saamge

#endif // _ML_HPP
