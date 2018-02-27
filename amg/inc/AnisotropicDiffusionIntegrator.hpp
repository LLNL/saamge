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

#ifndef ANISOTROPIC_DIFFUSION_INTEGRATOR_HPP
#define ANISOTROPIC_DIFFUSION_INTEGRATOR_HPP 

#include "mfem.hpp"

namespace saamge
{

/**
   Why can't we use a standard DiffusionIntegrator with a matrix
   coefficient? That seems a lot simpler.

   @todo Replace this or figure out why it's necessary
*/
class AnisotropicDiffusionIntegrator: public mfem::BilinearFormIntegrator
{
private:
    mfem::Vector vec, pointflux, shape;
#ifndef MFEM_THREAD_SAFE
    mfem::DenseMatrix dshape, dshapedxt, invdfdx, mq;
    mfem::DenseMatrix te_dshape, te_dshapedxt;
#endif
    mfem::VectorCoefficient *b;

public:
    AnisotropicDiffusionIntegrator() { b = NULL; }
    AnisotropicDiffusionIntegrator(mfem::VectorCoefficient &b) : b(&b) { }

    void AssembleElementMatrix(const mfem::FiniteElement &el,
                               mfem::ElementTransformation &Trans,
                               mfem::DenseMatrix &elmat) override;

    void AssembleElementMatrix2(const mfem::FiniteElement &trial_fe,
                                const mfem::FiniteElement &test_fe,
                                mfem::ElementTransformation &Trans,
                                mfem::DenseMatrix &elmat) override;

    void AssembleElementVector(const mfem::FiniteElement &el,
                               mfem::ElementTransformation &Tr,
                               const mfem::Vector &elfun,
                               mfem::Vector &elvect) override;

    void ComputeElementFlux(const mfem::FiniteElement &el,
                            mfem::ElementTransformation &Trans,
                            mfem::Vector &u, const mfem::FiniteElement &fluxelem,
                            mfem::Vector &flux, int with_coef = 1) override;

    double ComputeFluxEnergy(const mfem::FiniteElement &fluxelem,
                             mfem::ElementTransformation &Trans,
                             mfem::Vector &flux,
                             mfem::Vector *d_energy = NULL) override;
};

}

#endif //ANISOTROPIC_DIFFUSION_INTEGRATOR_HPP
