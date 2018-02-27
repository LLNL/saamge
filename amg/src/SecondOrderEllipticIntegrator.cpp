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

#include "SecondOrderEllipticIntegrator.hpp"
#include <memory>

namespace saamge
{
using namespace mfem;

// Layout 2d = [ v1, 0; v2, 0; v3, 0; 0 v1; 0 v2; 0 v3 ]
void CalcVShape(const int dim, const Vector &shape, DenseMatrix &vshape)
{
    const int nd = shape.Size();
    assert(vshape.Height() == dim * nd);
    assert(vshape.Width()  == dim);
    vshape = 0.0;

    for (int d = 0; d < dim; ++d)
    {
        for (int i = 0; i < nd; ++i)
        {
            vshape.Elem(d*nd + i, d) = shape(i);
        }
    }
}

void VectorDivDivIntegrator::AssembleElementMatrix(
    const FiniteElement &trial, ElementTransformation &Trans, 
    DenseMatrix &elmat)
{
    using namespace std;

    int dof = trial.GetDof();
    int dim = trial.GetDim();
    int spaceDim = Trans.GetSpaceDim();

    JInv.SetSize(spaceDim, spaceDim);
    g_grad.SetSize (dof, dim);
    grad.SetSize(dof, dim);
    div.SetSize(dof * dim);

    elmat.SetSize(dof * dim, dof * dim);
    elmat = 0.0;

    const IntegrationRule *ir = IntRule;
                
    if (ir == NULL)
    {
        int intorder = 2 * trial.GetOrder() - 2 + Trans.OrderW();
        ir = &IntRules.Get(trial.GetGeomType(), intorder);
    }

    for (int i = 0; i < ir->GetNPoints(); i++)
    {
        const IntegrationPoint &ip = ir->IntPoint(i);

        Trans.SetIntPoint (&ip);
        CalcInverse(Trans.Jacobian(), JInv);
        double w = ip.weight * Trans.Weight();

        trial.CalcDShape(ip, grad);
        Mult(grad, JInv, g_grad);
        g_grad.GradToDiv(div);

        //(div u, div v)
        AddMult_a_VVt(w, div, elmat);
    }
}


void Vector2ndOrderEllipticIntegratorMixed::AssembleElementMatrix2(
    const FiniteElement &trial, const FiniteElement &test,
    ElementTransformation &Trans, DenseMatrix &elmat)
{
    using namespace std;

    int trial_dof = trial.GetDof();
    int test_dof  = test.GetDof();

    int trial_dim = trial.GetDim();
    int test_dim  = test.GetDim();

    int spaceDim = Trans.GetSpaceDim();

    JInv.SetSize(spaceDim, spaceDim);
    test_g_grad.SetSize (test_dof, trial_dim);
    trial_g_grad.SetSize(trial_dof, trial_dim);

    trial_shape.SetSize(trial_dof);
    test_shape.SetSize(test_dof);
    trial_grad.SetSize(trial_dof, trial_dim);

    test_grad.SetSize(test_dof, test_dim);
    test_div.SetSize(test_dof * test_dim);
    test_vshape.SetSize(test_dof * test_dim, test_dim);

    elmat.SetSize(test_dof * test_dim, trial_dof);
    elmat = 0.0;

    const IntegrationRule *ir = IntRule;
                
    if (ir == NULL)
    {
        int intorder = trial.GetOrder() + test.GetOrder();
        ir = &IntRules.Get(test.GetGeomType(), intorder);
    }

    for (int i = 0; i < ir->GetNPoints(); i++)
    {
        const IntegrationPoint &ip = ir->IntPoint(i);

        Trans.SetIntPoint (&ip);
        CalcInverse(Trans.Jacobian(), JInv);
        double w = ip.weight * Trans.Weight();

        // Parameters
        double c_eval = c.Eval(Trans, ip);

        // Trial
        trial.CalcShape(ip,  trial_shape);
        trial.CalcDShape(ip, trial_grad);
        Mult(trial_grad, JInv, trial_g_grad);

        // Test
        test.CalcDShape(ip, test_grad);
        Mult(test_grad, JInv, test_g_grad);
        test_g_grad.GradToDiv(test_div);

        test.CalcShape(ip, test_shape);
        CalcVShape(test_dim, test_shape, test_vshape);
                        
        //(c . u, div q)
        AddMult_a_VWt(w * c_eval, test_div, trial_shape, elmat);
                        
        test_vshape *= w;
        //(grad u, q)
        AddMultABt(test_vshape, trial_g_grad, elmat);
    }
}



void DivDomainLFIntegrator::AssembleRHSElementVect(const FiniteElement &el,
                                                   ElementTransformation &Trans,
                                                   Vector &elvect)
{
    int dof = el.GetDof();
    int dim = el.GetDim();

    shape.SetSize(dof);      

    elvect.SetSize(dof * dim);
    elvect = 0.0;

    dshape.SetSize(dof, dim);
    divshape.SetSize(dof * dim);
    JInv.SetSize(dim, dim);
    g_grad.SetSize(dof, dim);

    const IntegrationRule *ir = IntRule;
    if (ir == NULL)
    {
        ir = &IntRules.Get(el.GetGeomType(), el.GetOrder() + oq + Trans.OrderW());
    }

    for (int i = 0; i < ir->GetNPoints(); i++)
    {
        const IntegrationPoint &ip = ir->IntPoint(i);

        Trans.SetIntPoint(&ip);
        double val = ip.weight * Trans.Weight() * Q.Eval(Trans, ip);

        CalcInverse(Trans.Jacobian(), JInv);

        el.CalcShape(ip, shape);
        el.CalcDShape(ip, dshape);
        Mult(dshape, JInv, g_grad);
        g_grad.GradToDiv(divshape);

        for (int i = 0; i < dim; ++i)
        {
            for (int j = 0; j < dof; ++j)
            {
                divshape(i*dof + j) *= shape(j);
            }
        }
        add(elvect, val, divshape, elvect);
    }
}

} // namespace saamge
