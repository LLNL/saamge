/*! \file
    \brief MFEM-like classes. Adapters to the MFEM abstraction.

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
#ifndef _MFEM_ADDONS_HPP
#define _MFEM_ADDONS_HPP

#include "common.hpp"
#include <mfem.hpp>
#include "smpr.hpp"
#include "aggregates.hpp"
#include "elmat.hpp"
#include "mbox.hpp"

// arguably this should be a different namespace?
namespace saamge
{

/* Types */
/*! \brief A coefficient function type.

    This is the type for any function used as a coefficient \f$k\f$ in
    \f$-\textrm{div}\left( k \nabla p \right)\f$.

    \param T (IN) An element transformation object (see MFEM).
    \param x (IN) The coordinates of the point where the values of \f$k\f$ is
                  wanted.
    \param mesh (IN) The mesh.
    \param param (IN) Function-specific parameter.

    \returns The value of \f$k\f$ in \a x.

    \warning Originally it was meant for piecewise constant \f$k\f$ on the
             elements but now it's not limited to only that case.
    \warning Of course, MFEM can take any kind of coefficients to be used as
             \f$k\f$ and they are not necessary based on this function type.

    DEPRECATED - do not think this is ever used
*/
typedef double (*mfem_ew_ft)(mfem::ElementTransformation& T,
                             mfem::Vector& x, mfem::Mesh& mesh, double param);

/*! \brief A function type for the right-hand side.

    A function type for the right-hand side of the elliptic equation and the
    (right-hand side of) the Dirichlet boundary condition.

    \param x (IN) The coordinates of the point where the values of the
                  right-hand side function is wanted.
    \param attr (IN) The attribute of the elements.
    \param param (IN) Function-specific parameter.

    \returns The value of the right-hand side function in \a x.

    DEPRECATED - do not think this is ever used
*/
typedef double (*mfem_bdr_rhs_ft)(const mfem::Vector& x, int attr, double param);

/* Classes */
/*! \brief Adapts \b mfem_ew_ft coefficients to be usable as MFEM coefficients.

DEPRECATED - do not think this is ever used
*/
class ElementWiseCoefficient : public mfem::Coefficient
{
public:
    /*! \brief The constructor.

        \param function (IN) The coefficient of type \b mfem_ew_ft that is
                             getting adapted to be usable as MFEM coefficient.
        \param mesh (IN) The mesh.
        \param given_param (IN) Function-specific parameter.
    */
    ElementWiseCoefficient(mfem_ew_ft function, mfem::Mesh &given_mesh,
                           double given_param/*=0.*/) :
        func(function), mesh(given_mesh), param(given_param)
    {
    }

    /*! \brief Coefficient evaluation. Part of the required interface.

        \param T (IN) An element transformation object (see MFEM).
        \param ip (IN) An integration point (see MFEM).

        \returns The value of the coefficient.
    */
    virtual double Eval(mfem::ElementTransformation& T,
                        const mfem::IntegrationPoint& ip);

    /*! \brief Not used.

        \param in (IN) Input stream.
    */
    virtual void Read(std::istream &in)
    {
    }

private:
    mfem_ew_ft func; /*!< The coefficient of type \b mfem_ew_ft that is getting
                          adapted to be usable as MFEM coefficient. */
    mfem::Mesh& mesh; /*!< The mesh. */
    mfem::Vector transip; /*!< For inner use. */
    double param; /*!< Function-specific parameter. */
};

/*! \brief Wraps \b mfem_bdr_rhs_ft coefficients as MFEM coefficients.

    Wraps \b mfem_bdr_rhs_ft coefficients to be usable as MFEM coefficients.

    DEPRECATED - do not think this is ever used
*/
class BdrRhsCoefficient : public mfem::Coefficient
{
public:
    /*! \brief The constructor.

        \param function (IN) The coefficient of type \b mfem_bdr_rhs_ft that is
                             getting adapted to be usable as MFEM coefficient.
        \param mesh (IN) The mesh.
        \param bdr (IN) \em true if this is a boundary condition.
        \param given_param (IN) Function-specific parameter.
    */
    BdrRhsCoefficient(mfem_bdr_rhs_ft function, mfem::Mesh &given_mesh, bool bdr,
                      double given_param/*=0.*/) :
        func(function), mesh(given_mesh), border(bdr), param(given_param)
    {
    }

    /*! \brief Coefficient evaluation. Part of the required interface.

        \param T (IN) An element transformation object (see MFEM).
        \param ip (IN) An integration point (see MFEM).

        \returns The value of the coefficient.
    */
    virtual double Eval(mfem::ElementTransformation& T,
                        const mfem::IntegrationPoint& ip);

    /*! \brief Not used.

        \param in (IN) Input stream.
    */
    virtual void Read(std::istream &in)
    {
    }

private:
    mfem_bdr_rhs_ft func; /*!< The coefficient of type \b mfem_bdr_rhs_ft that
                               is getting adapted to be usable as MFEM
                               coefficient. */
    mfem::Mesh& mesh; /*!< The mesh. */
    mfem::Vector transip; /*!< For inner use. */
    bool border; /*!< If it is a boundary condition. */
    double param; /*!< Function-specific parameter. */
};

/*! \brief Wraps \b smpr_ft preconditioners to be usable in MFEM.

    Wraps \b smpr_ft preconditioners to be usable as MFEM preconditioners.

    DEPRECATED, but want to resurrect and use it exclusively?
*/
class CFunctionSmoother : public mfem::Operator
{
public:
    /*! \brief The constructor.

        \param A (IN) The matrix that will be passed to \a smoothera.
        \param smoothera (IN) The smoother of type \b smpr_ft that is getting
                              adapted to be usable as MFEM smoother or
                              preconditioner.
        \param dataa (IN/OUT) The smoother-specific data.
    */
    CFunctionSmoother(mfem::HypreParMatrix& A, smpr_ft smoothera, void *dataa) :
        mfem::Operator(A.GetGlobalNumRows()), mat(A), smoother(smoothera),
        data(dataa)
    {
    }

    /*! \brief The destructor.
    */
    virtual ~CFunctionSmoother()
    {
    }

    /*! \brief Multiplying with the smoother. Part of the required interface.

        It computes \f$\mathbf{y} = M^{-1}\mathbf{x}\f$, where M is the
        smoother.

        \param x (IN) The vector to be smoothed.
        \param y (OUT) The smoothed vector.
    */
    virtual void Mult(const mfem::Vector& x, mfem::Vector& y) const
    {
        y = 0.;
        smoother(mat, x, y, data);
    }

private:
    mfem::HypreParMatrix& mat; /*!< The operator which we precondition. */
    smpr_ft smoother; /*!< The smoother of type \b smpr_ft that is getting
                           adapted to be usable as MFEM smoother
                           (preconditioner). */
    void *data; /*!< The smoother-specific data. */
};

/* Functions */
/*! \brief Generates the relation table that relates elements to vertices.

    \param mesh (IN) The mesh.
    \param elem_to_vert (OUT) The constructed relation.

    DEPRECATED - never used
*/
void construct_elem_to_vert(mfem::Mesh& mesh, mfem::Table& elem_to_vert);

/**
   Returns a pointer to a mfem::HypreParMatrix on one processor that
   is the same as A.

   SparseMatrix A owns the data, so if you delete it the
   returned HypreParMatrix is SOL.
*/
mfem::HypreParMatrix * FakeParallelMatrix(const mfem::SparseMatrix *A);

/*! \brief PCG solver.

    This is a slightly modified version of the PCG implementation in MFEM.

    \param A (IN) The matrix of the system being solved (usually the global
                  stiffness matrix).
    \param B (IN) Preconditioner.
    \param b (IN) The right-hand side.
    \param x (IN/OUT) Outputs the solution approximation. As input it is the
                      initial approximation.
    \param print_iter (IN) Determines the amount of output.
    \param max_num_iter (IN) The maximal number of iteration to be done.
    \param RTOLERANCE (IN) Relative tolerance.
    \param ATOLERANCE (IN) Absolute tolerance.
    \param zero_rhs (IN) If it is \em true, it outputs more error-related
                         information.

    \returns The number of iterations done. If a solution was not successfully
             computed, then this number is negative.
*/
int kalchev_pcg(const mfem::HypreParMatrix &A, const mfem::Operator &B, const mfem::HypreParVector &b,
                mfem::HypreParVector &x, int print_iter/*=0*/, int max_num_iter/*=1000*/, double RTOLERANCE/*=10e-12*/,
                double ATOLERANCE/*=10e-24*/, bool zero_rhs/*=false*/);

mfem::SparseMatrix * IdentitySparseMatrix(int n);

mfem::Table * TableFromSparseMatrix(const mfem::SparseMatrix& A);

/*! \brief Makes an mfem::Table of size n that is the identity permutation
 */
mfem::Table * IdentityTable(int n);

} // namespace saamge

#endif // _MFEM_ADDONS_HPP
