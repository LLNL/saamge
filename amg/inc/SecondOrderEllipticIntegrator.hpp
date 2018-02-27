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

#ifndef SECOND_ORDER_ELLIPTIC_INTEGRATOR_HPP
#define SECOND_ORDER_ELLIPTIC_INTEGRATOR_HPP

#include "mfem.hpp"
#include <assert.h>

namespace saamge
{

/**
   ATB: It's not clear to me what these integrators do that is not also
   a native ability in MFEM.
*/

class VectorDivDivIntegrator : public mfem::BilinearFormIntegrator
{
public:
    mfem::DenseMatrix JInv;
    mfem::DenseMatrix g_grad;
    mfem::DenseMatrix grad;
    mfem::Vector div;

    VectorDivDivIntegrator() {}

    void AssembleElementMatrix(
        const mfem::FiniteElement &trial_fe,
        mfem::ElementTransformation &Trans, 
        mfem::DenseMatrix &elmat) override;

    void AssembleElementMatrix2(
        const mfem::FiniteElement &trial_fe, const mfem::FiniteElement &test_fe,
        mfem::ElementTransformation &Trans, mfem::DenseMatrix &elmat) override
    {
        //TODO
        assert(false); 
    }

    void AssembleElementVector(
        const mfem::FiniteElement &el, mfem::ElementTransformation &Tr,
        const mfem::Vector &elfun, mfem::Vector &elvect) override
    {
        //TODO
        assert(false); 
    }

    void ComputeElementFlux(
        const mfem::FiniteElement &el, mfem::ElementTransformation &Trans,
        mfem::Vector &u, const mfem::FiniteElement &fluxelem, mfem::Vector &flux,
        int with_coef) override
    {
        //TODO
        assert(false);
    }

    double ComputeFluxEnergy(
        const mfem::FiniteElement &fluxelem, mfem::ElementTransformation &Trans,
        mfem::Vector &flux, mfem::Vector* d_energy) override
    {
        //TODO
        assert(false); 
        return 0;
    }
};

class Vector2ndOrderEllipticIntegratorMixed : public mfem::BilinearFormIntegrator
{
private:
    mfem::Coefficient &c;         
    mfem::DenseMatrix invdfdx, JInv;

    mfem::Vector trial_shape;
    mfem::DenseMatrix trial_grad, trial_g_grad;

    mfem::Vector test_shape;              
    mfem::Vector test_div;
    mfem::DenseMatrix test_grad, test_g_grad;
    mfem::Vector test_value;

    mfem::DenseMatrix test_vshape;

public:
    Vector2ndOrderEllipticIntegratorMixed (mfem::Coefficient &c) : c(c) { }

    void AssembleElementMatrix2(
        const mfem::FiniteElement &trial_fe, const mfem::FiniteElement &test_fe,
        mfem::ElementTransformation &Trans, mfem::DenseMatrix &elmat) override;

    void AssembleElementMatrix(
        const mfem::FiniteElement &trial_fe,
        mfem::ElementTransformation &Trans, mfem::DenseMatrix &elmat) override
    {
        //TODO
        assert(false);
    }

    void AssembleElementVector(
        const mfem::FiniteElement &el, mfem::ElementTransformation &Tr,
        const mfem::Vector &elfun, mfem::Vector &elvect) override
    {
        //TODO
        assert(false); 
    }

    void ComputeElementFlux(
        const mfem::FiniteElement &el, mfem::ElementTransformation &Trans,
        mfem::Vector &u, const mfem::FiniteElement &fluxelem, mfem::Vector &flux,
        int with_coef) override
    {
        //TODO
        assert(false);
    }

    double ComputeFluxEnergy(
        const mfem::FiniteElement &fluxelem, mfem::ElementTransformation &Trans,
        mfem::Vector &flux, mfem::Vector* d_energy) override
    {
        //TODO
        assert(false); 
        return 0;
    }
};


/// Class for domain integration L(v) := (f, div(v))
class DivDomainLFIntegrator : public mfem::LinearFormIntegrator
{
private:
    mfem::Vector shape;
    mfem::Vector divshape;
    mfem::DenseMatrix dshape, JInv, g_grad;

    mfem::Coefficient &Q;
    int oq;

public:
    DivDomainLFIntegrator(mfem::Coefficient &Q, int oq = 1)
        : Q(Q), oq(oq) {}

    DivDomainLFIntegrator(mfem::Coefficient &Q, const mfem::IntegrationRule *ir)
        : mfem::LinearFormIntegrator(ir), Q(Q), oq(1) { }

    void AssembleRHSElementVect(
        const mfem::FiniteElement &el, mfem::ElementTransformation &Tr,
        mfem::Vector &elvect) override;

    using mfem::LinearFormIntegrator::AssembleRHSElementVect;
};

} // namespace saamge

#endif //SECOND_ORDER_ELLIPTIC_INTEGRATOR_HPP
