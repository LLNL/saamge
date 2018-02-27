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

#include "AnisotropicDiffusionIntegrator.hpp"
#include <assert.h>

namespace saamge
{
using namespace mfem;

void AnisotropicDiffusionIntegrator::AssembleElementMatrix2(
    const FiniteElement &trial_fe, const FiniteElement &test_fe,
    ElementTransformation &Trans, DenseMatrix &elmat)
{
    //TODO 
    assert(false);
}

void AnisotropicDiffusionIntegrator::AssembleElementVector(
    const FiniteElement &el, ElementTransformation &Tr, const Vector &elfun,
    Vector &elvect)
{
    //TODO
    assert(false); 
}

void AnisotropicDiffusionIntegrator::ComputeElementFlux(
    const FiniteElement &el, ElementTransformation &Trans,
    Vector &u, const FiniteElement &fluxelem, Vector &flux, int with_coef)
{
    //TODO
    assert(false);
}

double AnisotropicDiffusionIntegrator::ComputeFluxEnergy(
    const FiniteElement &fluxelem, ElementTransformation &Trans,
    Vector &flux, Vector* d_energy)
{
    //TODO
    assert(false);
    return 0;
}

void AnisotropicDiffusionIntegrator::AssembleElementMatrix(
    const FiniteElement &el, ElementTransformation &Trans,
    DenseMatrix &elmat )
{
    int nd = el.GetDof();
    int dim = el.GetDim();
    int spaceDim = Trans.GetSpaceDim();
    bool square = (dim == spaceDim);

#ifdef MFEM_THREAD_SAFE
    DenseMatrix dshape(nd,dim), dshapedxt(nd,spaceDim), invdfdx(dim,spaceDim);
#else
    dshape.SetSize(nd,dim);
    dshapedxt.SetSize(nd,spaceDim);
    invdfdx.SetSize(dim,spaceDim);
#endif
    elmat.SetSize(nd);

    const IntegrationRule *ir = IntRule;
    if (ir == NULL)
    {
        int order;
        if (el.Space() == FunctionSpace::Pk)
        {
            order = 2*el.GetOrder() - 2;
        }
        else
            // order = 2*el.GetOrder() - 2;  // <-- this seems to work fine too
        {
            order = 2*el.GetOrder() + dim - 1;
        }
                        
        if (el.Space() == FunctionSpace::rQk)
        {
            ir = &RefinedIntRules.Get(el.GetGeomType(), order);
        }
        else
        {
            ir = &IntRules.Get(el.GetGeomType(), order);
        }
    }

    elmat = 0.0;
    for (int i = 0; i < ir->GetNPoints(); i++)
    {
        const IntegrationPoint &ip = ir->IntPoint(i);
        el.CalcDShape(ip, dshape);

        Trans.SetIntPoint(&ip);
        // Compute invdfdx = / adj(J),         if J is square
        //                   \ adj(J^t.J).J^t, otherwise
        CalcAdjugate(Trans.Jacobian(), invdfdx);
        const double detT = Trans.Weight();
        Mult(dshape, invdfdx, dshapedxt);

        if (b)
        {
            //remove scaling of the det
            dshapedxt *= 1/detT;

            Vector b_eval;
            b->Eval(b_eval, Trans, ip);

            DenseMatrix aniso_term;
            int n = b_eval.Size();
            aniso_term.SetSize(n, n);

            for (int i = 0; i < n; ++i)
            {
                for (int j = 0; j < n; ++j)
                {
                    aniso_term.Elem(i, j) = b_eval(i) * b_eval(j);
                }
            }

            for (int i = 0; i < n; ++i)
            {
                aniso_term.Elem(i, i) += 0.001;
            }

            DenseMatrix trial;
            trial.SetSize(dshapedxt.Height(), aniso_term.Height());
            trial = 0.0;

            //aniso_term is symmetric this is why you can use Mult
            Mult(dshapedxt, aniso_term, trial);
            trial *= detT * ip.weight;

            AddMultABt(dshapedxt, trial, elmat);

        }
        else
        {
            const double w = ip.weight / (square ? detT : detT * detT * detT);
            AddMult_a_AAt(w, dshapedxt, elmat);
        }
    }
}

} // namespace saamge
