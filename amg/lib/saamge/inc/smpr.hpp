/*! \file
    \brief Functions used as smoothers and preconditioners.

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

#pragma once
#ifndef _SMPR_HPP
#define _SMPR_HPP

#include "common.hpp"
#include <mfem.hpp>
#include "mbox.hpp"

/* Types */
/*! \brief A function type for a smoother/preconditioner.

    It computes \f$\mathbf{x} += M^{-1}(\mathbf{b} - A\mathbf{x})\f$, where M
    is the smoother/preconditioner.
    \a x is the initial guess as an input and the next iterate as an output.

    \param A (IN) The matrix.
    \param b (IN) The right-hand side.
    \param x (IN/OUT) \f$\mathbf{x} += M^{-1}(\mathbf{b} - A\mathbf{x})\f$.
    \param data (IN/OUT) Smoother/preconditioner-specific data.
*/
typedef void (*smpr_ft)(HypreParMatrix& A, const Vector& b, Vector& x,
                        void *data);

/*! \brief Computes roots for a polynomial smoother.

    \param nu (IN/OUT) The degree of the polynomial is \a nu. Can be modified
                       in the function.
    \param degree (OUT) The degree of the polynomial (\a nu).

    \returns An array of the roots.

    \warning The returned array must be freed by the caller. The best is to use
             \b smpr_free_poly_data for the whole structure.
*/
typedef double *(*smpr_roots_ft)(int& nu, int *degree);

/*! \brief The type of the polynomial smoother.
*/
typedef enum {
    SMPR_POLY_ONEMINUSX, /*!< The polynomial 1-x. */
    SMPR_POLY_SA, /*!< The "smoothed aggregation (SA)" polynomial. */
    SMPR_POLY_SAS, /*!< The "smoothed aggregation smoothing (SAS)"
                            polynomial. */
    SMPR_POLY_INVX, /*!< The best uniform approximation of 1/x. */

    SMPR_POLY_MAX /*!< Sentinel. */
} smpr_poly_t;

/*! \brief The data for the polynomial smoother.
*/
typedef struct {
    int nu; /*!< The used nu */
    int degree; /*!< The degree of the polynomial (first) */
    const double *roots; /*!< The roots of the polynomial (first) */
    HypreParVector *Dinv_neg; /*!< This is \f$-D^{-1}\f$ in the smoother
                                 \f$M^{-1} = \left[ I - p\left( D^{-1}A \right) \right] A^{-1}\f$,
                                 where p is the polynomial given by \a roots
                                 and \a degree.
                                 XXX: Have in mind that, as it can be seen
                                      above, D needs to be spectrally
                                      equivalent to the diagonal of A and also
                                      v' A v <= v' D v. */
    mbox_snd_vec_from_mat_par_ft build_Dinv_neg; /*!< The -D^{-1} construction
                                                      method. */

    double weightfirst; /*!< The weight of the first polynomial (if needed) */
    int degree2; /*!< The degree of the second polynomial (if needed) */
    const double *roots2; /*!< The roots of the second polynomial (if
                               needed) */
    double param; /*!< A real (double) parameter (polynomial smoother
                       specific) */
} smpr_poly_data_t;

/* Options */

/*! \brief The configuration class of this module.
*/
CONFIG_BEGIN_CLASS_DECLARATION(SMPR)

    /*! The -D^{-1} construction method.

        \warning Have in mind that usually D needs to be spectrally equivalent
                 to the diagonal of the stiffness matrix and also
                 v' A v <= v' D v.
        \warning This parameter is used during the construction of objects
                 (structure instances) so if modified it will only have effect
                 for new objects and will NOT modify the behavior of existing
                 ones. For altering the option for existing objects look at the
                 corresponding fields in the respective structure(s). */
    CONFIG_DECLARE_OPTION(mbox_snd_vec_from_mat_par_ft, build_Dinv_neg);

    /*! The polynomial smoother to be generated during initialization.

        This smoother is used for relaxation.

        \warning This parameter is used during the construction of objects
                 (structure instances) so if modified it will only have effect
                 for new objects and will NOT modify the behavior of existing
                 ones. For altering the option for existing objects look at the
                 corresponding fields in the respective structure(s). */
    CONFIG_DECLARE_OPTION(smpr_poly_t, smpr_poly);

CONFIG_END_CLASS_DECLARATION(SMPR)

CONFIG_BEGIN_INLINE_CLASS_DEFAULTS(SMPR)
    CONFIG_DEFINE_OPTION_DEFAULT(build_Dinv_neg,
                                 mbox_build_Dinv_neg_parallel_matrix),
    CONFIG_DEFINE_OPTION_DEFAULT(smpr_poly, SMPR_POLY_SAS)
CONFIG_END_CLASS_DEFAULTS

/* Functions */
/*! \brief Computes appropriate \em nu from \a a.

    The following holds
        \f$\left( \frac{1 - \sqrt{a}}{1 + \sqrt{a}} \right)^\nu < \frac{a}{1 - a}\f$.
    See \b smpr_invx_poly_init.

    \param a (IN) See the description.

    \returns nu. See the description.
*/
int smpr_nu_from_a(double a);

/*! \brief Computes the Chebyshev polynomial of the first kind at a point.

    It computes \f$T_n(x)\f$, where n >= 0.

    \param n (IN) This is the degree of the polynomial.
    \param x (IN) This is the argument of the polynomial.

    \returns \f$T_n(x)\f$.
*/
double smpr_cheb_firstkind(int n, double x);

/*! \brief Symmetric polynomial smoother.

    It computes \f$\mathbf{x} += M^{-1}(\mathbf{b} - A\mathbf{x})\f$, where M
    is the smoother.
    Here \f$M^{-1} = \left[ I - p\left( D^{-1}A \right) \right] A^{-1}\f$,
    where p is the polynomial given by \a data.
    \a x is the initial guess as an input and the next iterate as an output.

    \param A (IN) The matrix.
    \param b (IN) The right-hand side.
    \param x (IN/OUT) \f$\mathbf{x} += M^{-1}(\mathbf{b} - A\mathbf{x})\f$.
    \param data (IN) Must be of type \b smpr_poly_data_t.
*/
void smpr_sym_poly(HypreParMatrix& A, const Vector& b, Vector& x, void *data);

/*! \brief The two-grid SA-\f$\rho\f$AMGe is used as a preconditioner.

    It computes \f$\mathbf{x} += M^{-1}(\mathbf{b} - A\mathbf{x})\f$, where M
    is the preconditioner.
    \a x is the initial guess as an input and the next iterate as an output.

    \param A (IN) The matrix.
    \param b (IN) The right-hand side.
    \param x (IN/OUT) \f$\mathbf{x} += M^{-1}(\mathbf{b} - A\mathbf{x})\f$.
    \param data (IN) Must be of type \b tg_data_t.

    \warning \em Ac in the TG data (\a data) must be already computed and
             present.
*/
void smpr_tg(HypreParMatrix& A, const Vector& b, Vector& x, void *data);

/*! \brief Computes roots for the polynomial smoother.

    Computes the root of the 1-x polynomial, which is:
        \f$\tau_{0} = 1\f$.

    \param nu (OUT) Gets set to 1.
    \param degree (OUT) The degree of the polynomial (1).

    \returns An array of the roots.

    \warning The returned array must be freed by the caller. The best is to use
             \b smpr_free_poly_data for the whole structure.
*/
double *smpr_oneminusx_poly_roots(int& nu, int *degree);

/*! \brief Computes roots for the polynomial smoother.

    Computes the roots of the "smoothed aggregation (SA)" polynomial, which
    are:
        \f$\tau_{i-1} = \sin^2\left( \frac{i}{2\nu + 1} \pi \right)\f$,
    for i = 1, ...,\a nu

    \param nu (IN) The degree of the polynomial is \a nu.
    \param degree (OUT) The degree of the polynomial (\a nu).

    \returns An array of the roots.

    \warning The returned array must be freed by the caller. The best is to use
             \b smpr_free_poly_data for the whole structure.
*/
double *smpr_sa_poly_roots(int& nu, int *degree);

/*! \brief Computes roots for the polynomial smoother.

    Computes the roots of the "smoothed aggregation smoothing (SAS)"
    polynomial, which are:
        \f$\tau_i = \cos^2\left( \frac{i}{2\nu + 1} \pi \right)\f$,
    for i = 0, ...,2 * \a nu;
        \f$\tau_{j + 2\nu} = \sin^2\left( \frac{j}{2\nu + 1} \pi \right)\f$,
    for j = 1, ...,\a nu.

    \param nu (IN) The degree of the polynomial is 3 * \a nu + 1.
    \param degree (OUT) The degree of the polynomial (3 * \a nu + 1).

    \returns An array of the roots.

    \warning The returned array must be freed by the caller. The best is to use
             \b smpr_free_poly_data for the whole structure.
*/
double *smpr_sas_poly_roots(int& nu, int *degree);

/*! \brief Initialization for the 1/x smoother.

    A polynomial smoother based on the best uniform approximation of 1/x by a
    polynomial in the interval [a, 1], 0 < a < 1. The polynomial is of degree
    nu + 1.

    \param nu (IN) See the description.
    \param a (IN) See the description.
    \param poly_data (OUT) The initialized data.
*/
void smpr_invx_poly_init(int nu, double a, smpr_poly_data_t *poly_data);

/*! \brief Bulds/updates Dinv_neg.

    \param A (IN) the global (among all processes) stiffness matrix.
    \param poly_data (IN) The polynomial smoother data.

    \returns A pointer to Dinv_neg.

    \warning The returned vector must NOT be freed by the caller.
*/
HypreParVector *smpr_update_Dinv_neg(HypreParMatrix& A,
                                     smpr_poly_data_t *poly_data);

/*! \brief Creates and initializes the polynomial smoother data.

    \param A (IN) The global (among all processes) stiffness matrix.
    \param nu (IN) A parameter for the polynomial smoother (case dependant).
    \param param (IN) A parameter for the polynomial smoother (case dependant).

    \returns A pointer to the polynomial smoother data.

    \warning The returned structure must be freed by the caller using
             \b smpr_free_poly_data.
    \warning It uses the option \b smpr_poly to determine the type of the
             polynomial smoother for relaxation.
    \warning It uses the option \b build_Dinv_neg.
*/
smpr_poly_data_t *smpr_init_poly_data(HypreParMatrix& A, int nu, double param);

/*! \brief Frees the structure for the polynomial smoother.

    \param data (IN) The polynomial structure.

    \warning It tries to delete the structure itself.
*/
void smpr_free_poly_data(smpr_poly_data_t *data);

/*! \brief Copies a structure for the polynomial smoother.

    \param src (IN) Copies this.

    \returns The copy. If \a src is NULL, it returns NULL.

    \warning The returned structure must be freed by the caller using
             \b smpr_free_poly_data.
*/
smpr_poly_data_t *smpr_copy_poly_data(const smpr_poly_data_t *src);

/* Inline Functions */
/*! \brief Returns a pointer to Dinv_neg.

    \param poly_data (IN) The polynomial smoother data.

    \returns A pointer to Dinv_neg.

    \warning The returned vector must NOT be freed by the caller.
*/
static inline
HypreParVector *smpr_get_Dinv_neg(const smpr_poly_data_t *poly_data);

/*! \brief Computes a polynomial smoother.

    It computes \f$\mathbf{x} += M^{-1}(\mathbf{b} - A\mathbf{x})\f$, where M
    is the polynomial smoother.
    Here \f$M^{-1} = \left[ I - p\left( D^{-1}A \right) \right] A^{-1}\f$,
    where p is the polynomial given by \a degree and \a roots.
    \a x is the initial guess as an input and the next iterate as an output.

    param A (IN) The matrix.
    \param b (IN) The right-hand side.
    \param x (IN/OUT) \f$\mathbf{x} += M^{-1}(\mathbf{b} - A\mathbf{x})\f$.
    \param degree (IN) The degree of the polynomial.
    \param roots (IN) The roots of the polynomial.
    \param Dinv_neg (IN) \f$-D^{-1}\f$.

    \warning Works only with "diagonal" \a Dinv_neg.
*/
static inline
void smpr_compute_poly(HypreParMatrix& A, const Vector& b, Vector& x,
                       int degree, const double *roots,
                       HypreParVector *Dinv_neg);

/* Inline Functions Definitions */
static inline
HypreParVector *smpr_get_Dinv_neg(const smpr_poly_data_t *poly_data)
{
     return poly_data->Dinv_neg;
}

static inline
void smpr_compute_poly(HypreParMatrix& A, const Vector& b, Vector& x,
                       int degree, const double *roots,
                       HypreParVector *Dinv_neg)
{
    SA_ASSERT(A.GetGlobalNumRows() == A.GetGlobalNumCols());
    SA_ASSERT(A.GetGlobalNumRows() == mbox_parallel_vector_size(*Dinv_neg));
    SA_ASSERT(degree >= 0);

    Vector tmp(b.Size());
    Vector tmp1(b.Size());
    for (int i=0; i < degree; ++i)
    {
        const double mult = 1. / roots[i];
        tmp.Set(-1., b);
        A.Mult(x, tmp1);
        tmp += tmp1;
        mbox_entry_mult_vector(tmp, *Dinv_neg);
        x.Add(mult, tmp);
    }
}

#endif // _SMPR_HPP
