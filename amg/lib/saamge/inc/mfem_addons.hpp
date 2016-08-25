/*! \file
    \brief MFEM-like classes. Adapters to the MFEM abstraction.

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
#ifndef _MFEM_ADDONS_HPP
#define _MFEM_ADDONS_HPP

#include "common.hpp"
#include <mfem.hpp>
#include "smpr.hpp"
#include "aggregates.hpp"
#include "elmat.hpp"
#include "mbox.hpp"

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
*/
typedef double (*mfem_ew_ft)(ElementTransformation& T,
                             Vector& x, Mesh& mesh, double param);

/*! \brief A function type for the right-hand side.

    A function type for the right-hand side of the elliptic equation and the
    (right-hand side of) the Dirichlet boundary condition.

    \param x (IN) The coordinates of the point where the values of the
                  right-hand side function is wanted.
    \param attr (IN) The attribute of the elements.
    \param param (IN) Function-specific parameter.

    \returns The value of the right-hand side function in \a x.
*/
typedef double (*mfem_bdr_rhs_ft)(const Vector& x, int attr, double param);

/* Classes */
/*! \brief Adapts \b mfem_ew_ft coefficients to be usable as MFEM coefficients.
*/
class ElementWiseCoefficient : public Coefficient
{
public:
    /*! \brief The constructor.

        \param function (IN) The coefficient of type \b mfem_ew_ft that is
                             getting adapted to be usable as MFEM coefficient.
        \param mesh (IN) The mesh.
        \param given_param (IN) Function-specific parameter.
    */
    ElementWiseCoefficient(mfem_ew_ft function, Mesh &given_mesh,
                           double given_param/*=0.*/) :
        func(function), mesh(given_mesh), param(given_param)
    {
    }

    /*! \brief Coefficient evaluation. Part of the required interface.

        \param T (IN) An element transformation object (see MFEM).
        \param ip (IN) An integration point (see MFEM).

        \returns The value of the coefficient.
    */
    virtual double Eval(ElementTransformation& T,
                        const IntegrationPoint& ip);

    /*! \brief Not used.

        \param in (IN) Input stream.
    */
    virtual void Read(istream &in)
    {
    }

private:
    mfem_ew_ft func; /*!< The coefficient of type \b mfem_ew_ft that is getting
                          adapted to be usable as MFEM coefficient. */
    Mesh& mesh; /*!< The mesh. */
    Vector transip; /*!< For inner use. */
    double param; /*!< Function-specific parameter. */
};

/*! \brief Wraps \b mfem_bdr_rhs_ft coefficients as MFEM coefficients.

    Wraps \b mfem_bdr_rhs_ft coefficients to be usable as MFEM coefficients.
*/
class BdrRhsCoefficient : public Coefficient
{
public:
    /*! \brief The constructor.

        \param function (IN) The coefficient of type \b mfem_bdr_rhs_ft that is
                             getting adapted to be usable as MFEM coefficient.
        \param mesh (IN) The mesh.
        \param bdr (IN) \em true if this is a boundary condition.
        \param given_param (IN) Function-specific parameter.
    */
    BdrRhsCoefficient(mfem_bdr_rhs_ft function, Mesh &given_mesh, bool bdr,
                      double given_param/*=0.*/) :
        func(function), mesh(given_mesh), border(bdr), param(given_param)
    {
    }

    /*! \brief Coefficient evaluation. Part of the required interface.

        \param T (IN) An element transformation object (see MFEM).
        \param ip (IN) An integration point (see MFEM).

        \returns The value of the coefficient.
    */
    virtual double Eval(ElementTransformation& T,
                        const IntegrationPoint& ip);

    /*! \brief Not used.

        \param in (IN) Input stream.
    */
    virtual void Read(istream &in)
    {
    }

private:
    mfem_bdr_rhs_ft func; /*!< The coefficient of type \b mfem_bdr_rhs_ft that
                               is getting adapted to be usable as MFEM
                               coefficient. */
    Mesh& mesh; /*!< The mesh. */
    Vector transip; /*!< For inner use. */
    bool border; /*!< If it is a boundary condition. */
    double param; /*!< Function-specific parameter. */
};

/*! \brief Wraps \b smpr_ft preconditioners to be usable in MFEM.

    Wraps \b smpr_ft preconditioners to be usable as MFEM preconditioners.
*/
class CFunctionSmoother : public Operator
{
public:
    /*! \brief The constructor.

        \param A (IN) The matrix that will be passed to \a smoothera.
        \param smoothera (IN) The smoother of type \b smpr_ft that is getting
                              adapted to be usable as MFEM smoother or
                              preconditioner.
        \param dataa (IN/OUT) The smoother-specific data.
    */
    CFunctionSmoother(HypreParMatrix& A, smpr_ft smoothera, void *dataa) :
        Operator(A.GetGlobalNumRows()), mat(A), smoother(smoothera),
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
    virtual void Mult(const Vector& x, Vector& y) const
    {
        y = 0.;
        smoother(mat, x, y, data);
    }

private:
    HypreParMatrix& mat; /*!< The operator which we precondition. */
    smpr_ft smoother; /*!< The smoother of type \b smpr_ft that is getting
                           adapted to be usable as MFEM smoother
                           (preconditioner). */
    void *data; /*!< The smoother-specific data. */
};

/*! \brief Wraps \b agg_elmat_callback_ft to be usable as MFEM integrators.
*/
class WrapperIntegrator: public BilinearFormIntegrator
{
public:

    /*! \brief Constructor.

        \param elmat_callbacka (IN) Callback returning element matrices.
        \param agg_part_relsa (IN) The partitioning relations.
        \param dataa (IN/OUT) \a elmat_callbacka specific data.
     */
    WrapperIntegrator(agg_elmat_callback_ft elmat_callbacka,
                      const agg_partititoning_relations_t *agg_part_relsa,
                      void *dataa, int NEa) :
        elmat_callback(elmat_callbacka), agg_part_rels(agg_part_relsa),
        data(dataa), NE(NEa), matr_type(0)
    {
        SA_ASSERT(0 < NE);
    }

    /*! \brief Computes the element stiffness matrix.

        Given a particular Finite Element computes the element stiffness matrix
        \a elmat.

        \param el (IN) A finite element.
        \param Trans (IN) An element transformation.
        \param elmat (OUT) The computed element matrix.
     */
    virtual void AssembleElementMatrix(const FiniteElement &el,
                                       ElementTransformation &Trans,
                                       DenseMatrix &elmat)
    {
        bool free_matr;
        SA_ASSERT(0 <= Trans.ElementNo && Trans.ElementNo < NE);
        Matrix *elem_matr = elmat_callback(Trans.ElementNo, agg_part_rels,
                                           data, free_matr);
        SA_ASSERT(elem_matr);

        if (!matr_type)
        {
            if (dynamic_cast<SparseMatrix *>(elem_matr))
                matr_type = -1;
            else
            {
                SA_ASSERT(dynamic_cast<DenseMatrix *>(elem_matr));
                matr_type = 1;
            }
        }

        if (0 > matr_type) // Sparse element matrices.
        {
            SparseMatrix *spm = static_cast<SparseMatrix *>(elem_matr);
            SA_ASSERT(spm);
            mbox_convert_sparse_to_dense(*spm, elmat);
            if (free_matr)
                delete spm;
        } else // Dense element matrices.
        {
            SA_ASSERT(0 < matr_type);
            DenseMatrix *dm = static_cast<DenseMatrix *>(elem_matr);
            SA_ASSERT(dm);
            if (free_matr)
            {
                mbox_swap_data_dense(*dm, elmat);
                delete dm;
            } else
                elmat = *dm;
        }
    }

    /*! Not implemented and not used currently. */
    virtual void AssembleElementMatrix2(const FiniteElement &trial_fe,
                                        const FiniteElement &test_fe,
                                        ElementTransformation &Trans,
                                        DenseMatrix &elmat)
    {
        SA_ASSERT(false);
    }

    /*! Not implemented and not used currently. */
    virtual void AssembleFaceMatrix(const FiniteElement &el1,
                                    const FiniteElement &el2,
                                    FaceElementTransformations &Trans,
                                    DenseMatrix &elmat)
    {
        SA_ASSERT(false);
    }

    /*! Not implemented and not used currently. */
    virtual void ComputeElementFlux(const FiniteElement &el,
                                    ElementTransformation &Trans,
                                    Vector &u,
                                    const FiniteElement &fluxelem,
                                    Vector &flux, int wcoef)
    {
        SA_ASSERT(false);
    }

    /*! Not implemented and not used currently. */
    virtual double ComputeFluxEnergy(const FiniteElement &fluxelem,
                                     ElementTransformation &Trans,
                                     Vector &flux)
    {
        SA_ASSERT(false);
        return 0.;
    }

private:
    agg_elmat_callback_ft elmat_callback; /*!< Callback returning element
                                                     matrices. */
    const agg_partititoning_relations_t *agg_part_rels; /*!< The partitioning
                                                             relations. */
    void *data; /*!< Function specific data. */
    const int NE; /*!< The number of elements */
    int matr_type; /*!< The type the matrices returned by the callback:
                       == 0 -- Not known, to be determined;
                        < 0 -- Sparse;
                        > 0 -- Dense. */
};

/* Functions */
/*! \brief Generates the relation table that relates elements to vertices.

    \param mesh (IN) The mesh.
    \param elem_to_vert (OUT) The constructed relation.
*/
void construct_elem_to_vert(Mesh& mesh, Table& elem_to_vert);

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
int pcg(const HypreParMatrix &A, const Operator &B, const HypreParVector &b,
        HypreParVector &x, int print_iter/*=0*/, int max_num_iter/*=1000*/, double RTOLERANCE/*=10e-12*/,
        double ATOLERANCE/*=10e-24*/, bool zero_rhs/*=false*/);

#endif // _MFEM_ADDONS_HPP
