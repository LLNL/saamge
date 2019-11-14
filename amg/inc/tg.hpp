/*! \file
    \brief The two-grid (TG) implementation.

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

#pragma once
#ifndef _TG_HPP
#define _TG_HPP

#include "common.hpp"
#include "tg_data.hpp"
#include <mfem.hpp>
#include "smpr.hpp"
#include "mbox.hpp"
#include "interp.hpp"

namespace saamge
{

/* Types */

/*! \brief The type of functions that compute (r, r).

    Where r is the residual and (*, *) is an inner product.
    It computes the (pseudo) residual from "scratch" i.e. no past information
    about previous iterates is used. It's used in the TG solver.

    \param A (IN) The matrix of the system being solved (usually the global
                  stiffness matrix).
    \param b (IN) The right-hand side.
    \param x (IN) The current iterate (approximation of the solution).
    \param res (OUT) The residual. MUST BE FREED BY THE CALLER.
    \param psres (OUT) The "pseudo residual" i.e. \f$ B^{-1} \mathbf{r} \f$,
                       where B is some preconditioner and r is the residual.
                       MUST BE FREED BY THE CALLER.
    \param data (IN/OUT) Function-specific data.

    \returns (r, r). The kind of the inner product itself is function-specific.
*/
typedef double (*tg_calc_res_ft)(mfem::HypreParMatrix& A, mfem::HypreParVector& b,
                                 mfem::HypreParVector& x, mfem::HypreParVector *& res,
                                 mfem::HypreParVector *& psres, void *data);

/*! \brief The type of functions that recompute (r, r).

    Where r is the residual and (*, *) is an inner product.
    It computes the (pseudo) residual NOT from "scratch". It uses past
    information about previous iterates. It's used in the TG solver.

    \param A (IN) The matrix of the system being solved (usually the global
                  stiffness matrix).
    \param b (IN) The right-hand side.
    \param x (IN) The current iterate (approximation of the solution).
    \param x_prev (IN) The previous iterate.
    \param res (IN/OUT) The residual. As input it is the former residual.
    \param psres (IN/OUT) The "pseudo residual" i.e. \f$ B^{-1} \mathbf{r} \f$,
                          where B is some preconditioner and r is the residual.
                          As input it is the former pseudo residual.
    \param data (IN/OUT) Function-specific data.

    \returns (r, r). The kind of the inner product itself is function-specific.
*/
typedef double (*tg_recalc_res_ft)(mfem::HypreParMatrix& A, mfem::HypreParVector& b,
                                   mfem::HypreParVector& x, mfem::HypreParVector& x_prev,
                                   mfem::HypreParVector& res, mfem::HypreParVector& psres,
                                   void *data);

/* Options */

/*! \brief The configuration class of this module.
*/
CONFIG_BEGIN_CLASS_DECLARATION(TG)

    /*! The pre-smoother.

        \warning This parameter is used during the construction of objects
                 (structure instances) so if modified it will only have effect
                 for new objects and will NOT modify the behavior of existing
                 ones. For altering the option for existing objects look at the
                 corresponding fields in the respective structure(s). */
    CONFIG_DECLARE_OPTION(smpr_ft, pre_smoother);

    /*! The post-smoother.

        \warning This parameter is used during the construction of objects
                 (structure instances) so if modified it will only have effect
                 for new objects and will NOT modify the behavior of existing
                 ones. For altering the option for existing objects look at the
                 corresponding fields in the respective structure(s). */
    CONFIG_DECLARE_OPTION(smpr_ft, post_smoother);

CONFIG_END_CLASS_DECLARATION(TG)

/* Objects */

/**
   Use UMFPackSolver for HypreParMatrix, by secretely converting
   to SparseMatrix

   (only works for serial HypreParMatrix)

   this belongs more naturally in solve.hpp, but we end up wit a circular dependency
*/
class HypreDirect : public mfem::Solver
{
public:
   HypreDirect(mfem::HypreParMatrix& mat);
   ~HypreDirect();

   /// Set/update the solver for the given operator. (Solver interface)
   void SetOperator(const mfem::Operator &op);

   /// Operator interface
   virtual void Mult(const mfem::Vector &x, mfem::Vector &y) const;

private:
   mfem::SparseMatrix * mat_;
   mfem::UMFPackSolver solver_;
};

/* Functions */

/*! \brief The TG algorithm.

    Computes an TG iteration. It computes
    \f$ \mathbf{x} += B^{-1}(\mathbf{b} - A\mathbf{x}) \f$, where B is the TG
    preconditioner.

    \param A (IN) The matrix of the system being solved (usually the global
                  stiffness matrix).
    \param Ac (IN) The coarse-grid operator.
                   \a Ac = \a restr * \a A * \a interp.
    \param interp (IN) The interpolant.
    \param restr (IN) The restriction operator.
    \param b (IN) The right-hand side.
    \param pre_smoother (IN) The smoother for the pre-smoothing step.
    \param post_smoother (IN) The smoother for the post-smoothing step.
    \param x (IN/OUT) The current iterate as input and the next iterate as
                      output.
    \param coarse_solver (IN) The solver for the coarse-grid correction.
    \param data (IN/OUT) The data for \a pre_smoother and \a post_smoother.
    \param mu (IN) 1 for V-cycle, 2 for W-cycle
*/
void tg_cycle_atb(mfem::HypreParMatrix& A, mfem::HypreParMatrix& Ac,
                  mfem::HypreParMatrix& interp, mfem::HypreParMatrix& restr,
                  const mfem::Vector& b, smpr_ft pre_smoother,
                  smpr_ft post_smoother, mfem::Vector& x,
                  mfem::Solver& coarse_solver, void *data);

/*! \brief Computes \f$ (B^{-1}\mathbf{r}, \mathbf{r}) \f$.

    Computes \f$ (B^{-1}\mathbf{r}, \mathbf{r}) \f$, where B is the TG
    preconditioner.

    Where r is the residual and (*, *) is the usual dot product.
    It computes the (pseudo) residual from "scratch" i.e. no past information
    about previous iterates is used.

    \param A (IN) The matrix of the system being solved (usually the global
                  stiffness matrix).
    \param b (IN) The right-hand side.
    \param x (IN) The current iterate (approximation of the solution).
    \param res (OUT) The residual. MUST BE FREED BY THE CALLER.
    \param psres (OUT) The "pseudo residual" i.e. \f$ B^{-1} \mathbf{r} \f$.
                       MUST BE FREED BY THE CALLER.
    \param data (IN) Data for the TG algorithm. It is of type \b tg_data_t.

    \returns \f$ (B^{-1}\mathbf{r}, \mathbf{r}) \f$.
*/
double tg_calc_res_tgprod(mfem::HypreParMatrix& A, mfem::HypreParVector& b,
                          mfem::HypreParVector& x, mfem::HypreParVector *& res,
                          mfem::HypreParVector *& psres, void *data);

/*! \brief Recomputes \f$ (B^{-1}\mathbf{r}, \mathbf{r}) \f$.

    Recomputes \f$ (B^{-1}\mathbf{r}, \mathbf{r}) \f$, where B is the TG
    preconditioner.

    Where r is the residual and (*, *) is the usual dot product.
    It computes the residual from "scratch" but for the pseudo residual the
    previous iterate is used. It's computed like \a psres = \a x - \a x_prev.

    \param A (IN) The matrix of the system being solved (usually the global
                  stiffness matrix).
    \param b (IN) The right-hand side.
    \param x (IN) The current iterate (approximation of the solution).
    \param x_prev (IN) Not used.
    \param res (IN/OUT) The residual. As input it is the former residual.
    \param psres (IN/OUT) The "pseudo residual" i.e. \f$ B^{-1} \mathbf{r} \f$.
                          As input it is the former pseudo residual.
    \param data (IN) Data for the TG algorithm. It has as a type \b tg_data_t.

    \returns \f$ (B^{-1}\mathbf{r}, \mathbf{r}) \f$.

    \warning The returned inner product and \a psres are actually for the
             former iterate (\a x_prev) while \a res is for the current iterate
             (\a x).
*/
double tg_recalc_res_tgprod(mfem::HypreParMatrix& A, mfem::HypreParVector& b,
                            mfem::HypreParVector& x, mfem::HypreParVector& x_prev,
                            mfem::HypreParVector& res, mfem::HypreParVector& psres,
                            void *data);

/*! \brief Recomputes \f$ (B^{-1}\mathbf{r}, \mathbf{r}) \f$.

    Recomputes \f$ (B^{-1}\mathbf{r}, \mathbf{r}) \f$, where B is the TG
    preconditioner.

    Where r is the residual and (*, *) is the usual dot product.
    It still computes the (pseudo) residual from "scratch" i.e. no past
    information about previous iterates is used. This is just a special version
    for recomputing that actually computes instead of recomputing.

    \param A (IN) The matrix of the system being solved (usually the global
                  stiffness matrix).
    \param b (IN) The right-hand side.
    \param x (IN) The current iterate (approximation of the solution).
    \param x_prev (IN) The previous iterate.
    \param res (IN/OUT) The residual. As input it is the former residual.
    \param psres (IN/OUT) The "pseudo residual" i.e. \f$ B^{-1} \mathbf{r} \f$.
                          As input it is the former pseudo residual.
    \param data (IN) Data for the TG algorithm. It has as a type \b tg_data_t.

    \returns \f$ (B^{-1}\mathbf{r}, \mathbf{r}) \f$.

    \deprecated

    \warning The returned inner product and \a psres are actually for the
             former iterate (\a x_prev) while \a res is for the current iterate
             (\a x).
*/
double tg_fullrecalc_res_tgprod(mfem::HypreParMatrix& A, mfem::HypreParVector& b,
                                mfem::HypreParVector& x, mfem::HypreParVector& x_prev,
                                mfem::HypreParVector& res, mfem::HypreParVector& psres,
                                void *data);

/*! \brief Tries to solve \f$ A\mathbf{x} = \mathbf{b} \f$.

    Uses the hierarchy in \a tg_data. It stops when the maximal number of
    iterations is reach, or (r, r) becomes smaller than \a atol or \a rtol
    times (*, *) of the initial residual. Here r denotes a residual and (*, *)
    is an inner product determined by \a calc_res and \a recalc_res.

    \param A (IN) The matrix of the system being solved (usually the global
                  stiffness matrix).
    \param b (IN) The right-hand side.
    \param x (IN/OUT) Outputs the solution approximation. As input it is the
                      initial approximation.
    \param x_prev (OUT) The previous iterate. MUST BE FREED BY THE CALLER.
    \param maxiter (IN) The maximal number of iteration to be done (see the
                        description).
    \param calc_res (IN) Function for computing the initial residual.
    \param recalc_res (IN) Function for recomputing the residual on each
                           iteration.
    \param tg_data (IN) Data for the TG algorithm.
    \param rtol (IN) Relative tolerance (see the description).
    \param atol (IN) Absolute tolerance (see the description).
    \param reducttol (IN) Reduction factor tolerance. If the reduction factor
                          becomes larger than this tolerance, the function
                          exits due to bad convergence.
    \param output (IN) Whether to generate any console output. This way output
                       can be suppressed independent of the global output level.

    \returns The number of iterations performed. If a solution was not
             successfully computed (according to the stopping criteria), then
             this number is negative.

    \warning The coarse operator must be precomputed and present in \a tg_data.
*/
int tg_solve(mfem::HypreParMatrix& A, mfem::HypreParVector& b, mfem::HypreParVector& x,
             mfem::HypreParVector *& x_prev, int maxiter, tg_calc_res_ft calc_res,
             tg_recalc_res_ft recalc_res, tg_data_t *tg_data, double rtol/*=10e-12*/,
             double atol/*=10e-24*/, double reducttol/*=1.*/, bool output=true);

/*! \brief Tries to solve \f$ A\mathbf{x} = \mathbf{b} \f$.

    Uses PCG and the TG method (based on the hierarchy in \a tg_data) as a
    preconditioner.

    \param A (IN) The matrix of the system being solved (usually the global
                  stiffness matrix).
    \param b (IN) The right-hand side.
    \param x (IN/OUT) Outputs the solution approximation. As input it is the
                      initial approximation.
    \param maxiter (IN) The maximal number of iteration to be done.
    \param tg_data (IN) Data for the TG algorithm.
    \param rtol (IN) Relative tolerance.
    \param atol (IN) Absolute tolerance.
    \param zero_rhs (IN) If it is \em true, it outputs more error-related
                         information.
    \param output (IN) Whether to generate any console output. This way output
                       can be suppressed independent of the global output level.

    \returns The number of iterations performed. If a solution was not
             successfully computed (according to the stopping criteria), then
             this number is negative.
*/
int tg_pcg_solve(mfem::HypreParMatrix& A, mfem::HypreParVector& b, mfem::HypreParVector& x,
                 int maxiter, tg_data_t *tg_data, double rtol/*=10e-12*/, double atol/*=10e-24*/,
                 bool zero_rhs/*=false*/, bool output=true);

/*! \brief Executes the TG method.

    This function runs the TG solver to solve a given problem.

    \param A (IN) The matrix of the system being solved (usually the global
                  stiffness matrix).
    \param agg_part_rels (IN) The partitioning relations. It is only needed in
                              case \a zero_rhs is \em true. Otherwise, it can
                              be NULL.
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
    \param tg_data (IN) This is the TG data to be used.
    \param zero_rhs (IN) If it is \em true, then \b adapt_approx_xbad is used
                         since it outputs more error-related information.
                         Otherwise, \b tg_solve is used.
    \param output (IN) Whether to generate any console output. This way output
                       can be suppressed independent of the global output level.

    \returns The number of iterations done. If a desired convergence criteria
             was not reached, then this number is negative.

    \warning It uses the options \b calc_res and \b recalc_res to determine how
             residuals and their norms are computed and recomputed.
*/
int tg_run(mfem::HypreParMatrix& A,
           const agg_partitioning_relations_t *agg_part_rels,
           mfem::HypreParVector& x, mfem::HypreParVector& b, int maxiter,
           double rtol/*=10e-12*/, double atol/*=10e-24*/,
           double reducttol/*=1.*/, tg_data_t *tg_data, bool zero_rhs/*=0*/,
           bool output=true);

/*! \brief Executes the PCG preconditioned with the TG method.

    This function runs the MFEM PCG solver, preconditined with the TG method,
    to solve a given problem.

    \param A (IN) The matrix of the system being solved (usually the global
                  stiffness matrix).
    \param agg_part_rels (IN) The partitioning relations. It is only needed in
                              case \a zero_rhs is \em true. Otherwise, it can
                              be NULL.
    \param x (IN/OUT) Outputs the solution approximation. As input it is the
                      initial approximation.
    \param b (IN) The right-hand side.
    \param maxiter (IN) The maximal number of iteration to be done.
    \param rtol (IN) Relative tolerance.
    \param atol (IN) Absolute tolerance.
    \param tg_data (IN/OUT) This is the TG data to be used.
    \param zero_rhs (IN) If it is \em true, it outputs more error-related
                         information.
    \param output (IN) Whether to generate any console output. This way output
                       can be suppressed independent of the global output level.

    \returns The number of iterations done. If a desired convergence criteria
             was not reached, then this number is negative.

    semi-DEPRECATED
*/
int tg_pcg_run(mfem::HypreParMatrix& A,
           const agg_partitioning_relations_t *agg_part_rels,
           mfem::HypreParVector& x, mfem::HypreParVector& b, int maxiter, double rtol/*=10e-12*/,
           double atol/*=10e-24*/, tg_data_t *tg_data, bool zero_rhs/*=0*/,
           bool output=true);

/*! \brief Initializes TG data structure with default values.

    The hierarchy does NOT get built.

    \param A (IN) The matrix of the system being solved (usually the global
                  stiffness matrix).
    \param agg_part_rels (IN) The partitioning relations.
    \param nu_pro (IN) The degree of the polynomial smoother if the
                          tentative interpolant will be smoothed by a
                          polynomial.
    \param nu_relax (IN) nu of the polynomial relaxation.
    \param param_relax (IN) Extra parameter for the relaxation smoother.
    \param theta (IN) The spectral threshold.
    \param smooth_interp (IN) whether or not to smooth the tentative interpolator
    \param use_arpack (IN) whether or not to use ARPACK iterative eigensolver

    \returns The initialized TG data.

    \warning The returned structure must be freed by the caller using
             \b tg_free_data.
    \warning It uses the options \b pre_smoother, \b post_smoother,
             \b coarse_solver_solve_init, \b coarse_solver_solver, and
             \b coarse_solver_data.
*/
tg_data_t *tg_init_data(
    mfem::HypreParMatrix& A,
    const agg_partitioning_relations_t& agg_part_rels, int nu_pro,
    int nu_relax, double theta, bool smooth_interp, double smooth_drop_tol,
    bool use_arpack);

/*! \brief Shared code from tg_build_hierarchy functions */
void tg_assemble_and_smooth(mfem::HypreParMatrix &Ag,
                            tg_data_t& tg_data,
                            const agg_partitioning_relations_t& agg_part_rels);

/*! \brief Builds the coarse level with polynomial basis functions

  Optionally includes also spectral basis functions.

  \param mesh (IN) we need this in order to get coordinates, but probably
              the right thing is to ask for the coordinate vector directly
              in case you have it somehow in an algebraic setting.

  \param polynomial_order (IN) either 0 or 1 for adding constants or linears.
  \param use_spectral (IN) whether to also add spectral dofs

  See tg_build_hierarchy() for other parameters
*/
void tg_build_hierarchy_with_polynomial(
    mfem::HypreParMatrix& Ag, mfem::Mesh& mesh, tg_data_t& tg_data,
    const agg_partitioning_relations_t& agg_part_rels,
    ElementMatrixProvider *elem_data, int polynomial_order, bool use_spectral,
    bool avoid_ess_bdr_dofs);

/*! \brief Builds the coarse level.

    TG data must be initialized prior to calling this function (see
    \b tg_init_data).

    \param Ag (IN) The matrix of the system being solved (usually the global
                   (among all processes) stiffness matrix). It must have the
                   boundary conditions imposed.

    \param tg_data (IN/OUT) The initialized TG data. In the end the coarse
                            space is created there.
    \param agg_part_rels (IN) The partitioning relations.
    \param elem_data_finest (IN) Data corresponding to the finest element
                                 matrices callback
                                 \a interp_data.finest_elmat_callback. If the
                                 current level is not the finest this MUST be
                                 NULL to invoke coarse level construction.
                                 Usually this is a MFEM \em BilinearForm used
                                 for assembling \a A (on a geometric level).
*/
void tg_build_hierarchy(mfem::HypreParMatrix& Ag, tg_data_t& tg_data,
                        const agg_partitioning_relations_t& agg_part_rels,
                        ElementMatrixProvider *elem_data,
                        bool avoid_ess_bdr_dofs, int fixed_num_evecs=0, int svd_min_skip=0);

/*! \brief Extends an identity block to top-left of tg_data.interp

  This is a very dirty hack, does not work in parallel, involves way
  too much deleteion / recreation, is a general mess.

  We use this to for example when we eliminate a degree of freedom
  for a pure Neumann problem, build the hierarchy for the smaller
  matrix without the dof, and then do this so we can solve the whole
  problem with one constrained dof.
*/
void tg_augment_interp_with_identity(tg_data_t& tg_data, int k);

/*! \brief For algebraic interface, extracts submatrices from A and modifies
  diagonal so submatrices represent Neumann problems.

  In PSV's note, this is referred to as the "diagonal compensation" strategy,
  in contrast to the window AMG strategy in WindowSubMatrices()

  Probably this does not really belong in tg, maybe in part.hpp or something?

  A and agg_part_rels are input, agglomerate_element_matrices is output.
*/
void ExtractSubMatrices(
    const mfem::SparseMatrix& A, const agg_partitioning_relations_t& agg_part_rels,
    mfem::Array<mfem::SparseMatrix*>& agglomerate_element_matrices);

void TestWindowSubMatrices();

void WindowSubMatrices(
    const mfem::SparseMatrix& A, const agg_partitioning_relations_t& agg_part_rels,
    mfem::Array<mfem::SparseMatrix*>& agglomerate_element_matrices);

/*! \brief Essentially wraps ExtractSubMatrices and
  tg_produce_data, with an algebraic ElementMatrixProvider
*/
tg_data_t *tg_produce_data_algebraic(
    const mfem::SparseMatrix &Alocal,
    mfem::HypreParMatrix& Ag, const agg_partitioning_relations_t& agg_part_rels,
    int nu_pro, int nu_relax, double spectral_tol, bool smooth_interp,
    int polynomial_coarse_arg, bool use_window, bool use_arpack,
    bool avoid_ess_bdr_dofs);

/*! \brief Switch from window to diagonal compensation matrices, or vice versa.

  We may want to use window AMG for eigenvectors, and diagonal compensation
  for local corrections, so we use this.

  This is used in fv-mfem.cpp
*/
void tg_replace_submatrices(tg_data_t &tg_data, const mfem::SparseMatrix &Alocal, 
                            const agg_partitioning_relations_t& agg_part_rels,
                            bool use_window);

/*! \brief Produces the TG data.

    Almost all data is produced (from scratch) ready to be used for solving
    systems. This includes the final (smoothed) interpolator and the smoother
    in the TG cycle.

    \param Ag (IN) The matrix of the system being solved (usually the global
                   (among all processes) stiffness matrix). It must have the
                   boundary conditions imposed.
    \param agg_part_rels (IN) The partitioning relations.
    \param nu_pro (IN) The degree of the polynomial smoother if the
                          tentative interpolant will be smoothed by a
                          polynomial.
    \param nu_relax (IN) nu of the polynomial relaxation.
    \param elem_data_finest (IN) Data corresponding to the finest element
                                 matrices callback
                                 \a interp_data.finest_elmat_callback. If the
                                 current level is not the finest this MUST be
                                 NULL to invoke coarse level construction.
                                 Usually this is a MFEM \em BilinearForm used
                                 for assembling \a A (on a geometric level).

    \param theta (IN) The spectral threshold.

    \returns The produced TG data.

    \warning The returned structure must be freed by the caller using
             \b tg_free_data.
*/
tg_data_t *tg_produce_data(
    mfem::HypreParMatrix& Ag,
    const agg_partitioning_relations_t& agg_part_rels, int nu_pro,
    int nu_relax, ElementMatrixProvider *elem_data_finest, double theta,
    bool smooth_interp, int polynomial_coarse_arg, bool use_arpack,
    bool avoid_ess_bdr_dofs, int fixed_num_evecs=0, int svd_min_skip=0);

/*! \brief Frees the TG data.

    \param tg_data (IN) The TG data.
*/
void tg_free_data(tg_data_t *tg_data);

/*! \brief Makes a copy of TG data.

    \param src (IN) The TG data to be copied.

    \returns A copy of \a src.

    \warning The returned structure must be freed by the caller using
             \b tg_free_data.
    \warning Be careful here with the "coarse solver" and its data, and \b Ac.
             They should probably align depending on the "coarse solver". Have
             in mind that the "coarse solver" data is not copied in its
             entirety. Rather, only the pointer to it is copied.
*/
tg_data_t *tg_copy_data(const tg_data_t *src);

double tg_compute_OC(mfem::HypreParMatrix& A, tg_data_t& tg_data);

/*! \brief \em Ac is updated (recomputed).

    The previous one is freed and a new one is computed. Also, the "coarse
    solver" is initialized in case \a perform_solve_init is \em true and
    \a tg_data->coarse_solver.solve_init is not \em NULL.

    \param A (IN) The fine-grid operator.
    \param tg_data (IN) The TG data.
    \param perform_solve_init (IN) Whether to initialize the "coarse solver".
                                   Unless specially needed, this is usually
                                   \em true.
    \param coarse_direct (IN) Use a direct solver on the coarse level.

    \warning \em Ac must be freed using \b tg_free_coarse_operator.
*/
void tg_update_coarse_operator(mfem::HypreParMatrix& A, tg_data_t *tg_data,
                               bool perform_solve_init,
                               bool coarse_direct);

/* Inline Functions */
/*! \brief Smooths the tentative interpolant to produce the final one.

    Also the restriction operator is produced (by transposition).

    \param A (IN) The global (among all processes) stiffness matrix.
    \param tg_data (IN) The TG data.
*/
static inline
void tg_smooth_interp(mfem::HypreParMatrix& A, tg_data_t& tg_data);

/*! \brief Computes the coarse-grid operator.

    \param A (IN) The fine-grid operator.
    \param interp (IN) The interpolator.

    \returns The coarse operator.

    \warning The returned sparse matrix must be freed by the caller.
    \warning Use this function only in case you specially need it. Otherwise,
             \b tg_fillin_coarse_operator and \b tg_update_coarse_operator are
             recommended because they have the capability to also initialize
             the "coarse solver".
*/
static inline
mfem::HypreParMatrix *tg_coarse_matr(mfem::HypreParMatrix& A, mfem::HypreParMatrix& interp);

/*! \brief If \em Ac is empty, it is computed.

    If \em Ac is empty, it is computed and the "coarse solver" is initialized
    in case \a perform_solve_init is \em true and
    \a tg_data->coarse_solver.solve_init is not \em NULL.

    \param A (IN) The fine-grid operator.
    \param tg_data (IN) The TG data.
    \param perform_solve_init (IN) Whether to initialize the "coarse solver".
                                   Unless specially needed, this is usually
                                   \em true.

    \warning \em Ac must be freed using \b tg_free_coarse_operator.
*/
static inline
void tg_fillin_coarse_operator(mfem::HypreParMatrix& A, tg_data_t *tg_data,
                               bool perform_solve_init/*=true*/);

/*! \brief \em Ac is freed.

    If \a tg_data.Ac is not NULL, it is freed and the coarse solver is
    destroyed. Otherwise, nothing happens.

    \param tg_data (IN/OUT) The TG data.
*/
static inline
void tg_free_coarse_operator(tg_data_t& tg_data);

/*! \brief Prints information about the hierarchy.

    \param A (IN) The finest operator.
    \param tg_data (IN) Two-grid data.
*/
static inline
double tg_print_data(mfem::HypreParMatrix& A, const tg_data_t *tg_data);

/* Inline Functions Definitions */
static inline
void tg_smooth_interp(mfem::HypreParMatrix& A, tg_data_t& tg_data)
{
    SA_ASSERT(tg_data.tent_interp);
    delete tg_data.interp;
    delete tg_data.restr;
    tg_free_coarse_operator(tg_data);
    mfem::HypreParMatrix *interp;
    tg_data.interp = interp =
        tg_data.smooth_interp ?
            interp_smooth_interp(A, *tg_data.interp_data, *tg_data.tent_interp,
                                 *smpr_get_Dinv_neg(tg_data.poly_data))
                              :
            mbox_clone_parallel_matrix(tg_data.tent_interp);
    tg_data.restr = interp->Transpose();
}

static inline
mfem::HypreParMatrix *tg_coarse_matr(mfem::HypreParMatrix& A,
                                     mfem::HypreParMatrix& interp)
{
    SA_RPRINTF_L(0, 5, "%s", "Computing coarse operator...\n");
    mfem::HypreParMatrix *Ac = RAP(&A, &interp);
    SA_ASSERT(Ac);
    SA_ASSERT(Ac->GetGlobalNumRows() == Ac->GetGlobalNumCols());
    SA_ASSERT(Ac->GetGlobalNumRows() == interp.GetGlobalNumCols());

    SA_RPRINTF_L(0, 3, "Ac nnz: %d, A nnz: %d, OC: %g\n", Ac->NNZ(),
                A.NNZ(), ((double)Ac->NNZ()) / ((double)A.NNZ()) + 1.);

    return Ac;
}

static inline
void tg_fillin_coarse_operator(mfem::HypreParMatrix& A, tg_data_t *tg_data,
                               bool perform_solve_init)
{
    SA_ASSERT(tg_data);
    SA_ASSERT(tg_data->interp);
    SA_ASSERT(tg_data->restr);

    if (!(tg_data->Ac))
    {
        tg_data->Ac = tg_coarse_matr(A, *(tg_data->interp));
        if (perform_solve_init)
        {
            SA_RPRINTF_L(0, 5, "%s",
                         "Setting coarse solver as a single BoomerAMG v-cycle.\n");
            mfem::HypreBoomerAMG * hbamg =
                new mfem::HypreBoomerAMG(*tg_data->Ac);
            hbamg->SetPrintLevel(0);
            tg_data->coarse_solver = hbamg;
        }
    }
}

static inline
void tg_free_coarse_operator(tg_data_t& tg_data)
{
    if (!tg_data.Ac)
        return;

    delete tg_data.coarse_solver;
    tg_data.coarse_solver = NULL;

    delete tg_data.Ac;
    tg_data.Ac = NULL;
}

static inline
double tg_print_data(mfem::HypreParMatrix& A, const tg_data_t *tg_data)
{
    double OC = 0.0;
    SA_RPRINTF(0,"%s", ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
               ">>>>>>>>>>>>>>>>>>>>>>>>>>>>\n");
    SA_RPRINTF(0,"%s", "\tTwo-grid data:\n");
    SA_RPRINTF(0,"Level 0 dimension: %d, Operator nnz: %d\n", A.GetGlobalNumRows(),
              A.NNZ());
    SA_ASSERT(tg_data);
    SA_ASSERT(tg_data->interp);
    SA_ASSERT(A.GetGlobalNumRows() ==  A.GetGlobalNumCols());
    SA_ASSERT(tg_data->interp->GetGlobalNumRows() == A.GetGlobalNumRows());
    PROC_STR_STREAM << "Level 1 dimension: "
                    << (tg_data->Ac ? tg_data->Ac->GetGlobalNumCols() :
                                      tg_data->interp->GetGlobalNumCols());
    if (tg_data->Ac)
    {
        PROC_STR_STREAM << ", Operator nnz: " << tg_data->Ac->NNZ() << "\n";
        SA_RPRINTF(0,"%s", PROC_STR_STREAM.str().c_str());
        PROC_CLEAR_STR_STREAM;
        SA_RPRINTF(0,"Operator complexity: %g\n",
                   OC = 1. + tg_data->Ac->NNZ() / (double)A.NNZ());
        SA_ASSERT(tg_data->Ac->GetGlobalNumRows() ==
                  tg_data->Ac->GetGlobalNumCols());
        //SA_ASSERT(tg_data->interp->GetGlobalNumCols() ==
        //          tg_data->Ac->GetGlobalNumRows()); // Not in the nonconforming case.
    } else
    {
        PROC_STR_STREAM << "\n";
        SA_RPRINTF(0,"%s", PROC_STR_STREAM.str().c_str());
        PROC_CLEAR_STR_STREAM;
    }
    SA_RPRINTF(0,"%s", ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
               ">>>>>>>>>>>>>>>>>>>>>>>>>>>>\n");
    return OC;
}

} // namespace saamge

#endif // _TG_HPP
