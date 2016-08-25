/*! \file

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

#include "common.hpp"
#include "mfem_addons.hpp"
#include <cmath>
#include <mfem.hpp>
#include "mbox.hpp"
using std::pow;
using std::sqrt;

/* Methods */

double ElementWiseCoefficient::Eval(ElementTransformation& T,
                                    const IntegrationPoint& ip)
{
    T.Transform(ip, transip);
    return func(T, transip, mesh, param);
}

double BdrRhsCoefficient::Eval(ElementTransformation& T,
                               const IntegrationPoint& ip)
{
    int attr;

    T.Transform(ip, transip);
#if 0
    if (border)
        attr = mesh.GetBdrAttribute(T.ElementNo);
    else
#endif
        attr = mesh.GetAttribute(T.ElementNo);

    return func(transip, attr, param);
}

/* Functions */

void construct_elem_to_vert(Mesh& mesh, Table& elem_to_vert)
{
    const int NE = mesh.GetNE();
    int i;

    elem_to_vert.MakeI(NE);
    for (i=0; i < NE; ++i)
        elem_to_vert.AddColumnsInRow(i, (mesh.GetElement(i))->GetNVertices());
    elem_to_vert.MakeJ();
    for (i=0; i < NE; ++i)
    {
        int *verts;
        Element *el;

        verts = (el = mesh.GetElement(i))->GetVertices();
        elem_to_vert.AddConnections(i, verts, el->GetNVertices());
    }
    elem_to_vert.ShiftUpI();
}

int pcg(const HypreParMatrix &A, const Operator &B, const HypreParVector &b,
        HypreParVector &x, int print_iter, int max_num_iter, double RTOLERANCE,
        double ATOLERANCE, bool zero_rhs)
{
    int i, dim = x.Size(), iters=0;
    double r0, den, nom, nom0, betanom=0., alpha, beta;
    Vector r(dim), d(dim), z(dim);
    Vector tmp(dim);
    double norm_x_prev=0., norm_x=0., norm_x_initial=0.;

    SA_ASSERT(A.GetGlobalNumRows() == A.GetGlobalNumCols());
    HypreParVector R(A.GetGlobalNumRows(), r.GetData(), A.GetRowStarts());
    HypreParVector D(A.GetGlobalNumRows(), d.GetData(), A.GetRowStarts());
    HypreParVector Z(A.GetGlobalNumRows(), z.GetData(), A.GetRowStarts());
    HypreParVector TMP(A.GetGlobalNumRows(), tmp.GetData(), A.GetRowStarts());

    A.Mult(x, r);
    if (zero_rhs)
    {
        norm_x_initial = norm_x_prev = sqrt(mbox_parallel_inner_product(x, R));
        r *= -1.;
    } else
        subtract(b, r, r);
    B.Mult(r, z);
    d = z;
    nom0 = nom = mbox_parallel_inner_product(Z, R);

    if (print_iter == 1 && 0 == PROC_RANK)
    {
        PROC_STR_STREAM << "PCG Iteration: 0, (B r, r) = " << nom;
        if (zero_rhs)
            PROC_STR_STREAM << ", || x ||_A = " << norm_x_prev;
        PROC_STR_STREAM << "\n";
        SA_PRINTF("%s", PROC_STR_STREAM.str().c_str());
        PROC_CLEAR_STR_STREAM;
    }

    if (zero_rhs)
    {
        if ( (r0 = norm_x_initial * RTOLERANCE) < ATOLERANCE) r0 = ATOLERANCE;
        if (norm_x_initial < r0)
            return -1;
    } else
    {
        if ( (r0 = nom * RTOLERANCE) < ATOLERANCE) r0 = ATOLERANCE;
        if (nom < r0)
            return -1;
    }

    A.Mult(d, z);
    den = mbox_parallel_inner_product(Z, D);

    if (den < 0.0)
        SA_ALERT_PRINTF("Negative denominator in step 0 of PCG: %g", den);

    SA_ASSERT(0. != den);

    if (0. == den)
        return -1;

    //Start iteration
    for (i=1; i <= max_num_iter; i++)
    {
        alpha = nom/den;
        add(x, alpha, d, x);                  //  x = x + alpha d
        add(r,-alpha, z, r);                  //  r = r - alpha z

        B.Mult(r, z);                         //  z = B r
        betanom = mbox_parallel_inner_product(R, Z);

        if (zero_rhs)
        {
            A.Mult(x, tmp);
            norm_x = sqrt(mbox_parallel_inner_product(x, TMP));
        }

        if (print_iter == 1 && 0 == PROC_RANK)
        {
            PROC_STR_STREAM << "PCG Iteration: " << i << ", (B r, r) = "
                            << betanom;
            if (zero_rhs)
            {
                PROC_STR_STREAM << ", || x ||_A = " << norm_x
                                << ", || x ||_A / || x_prev ||_A = "
                                << norm_x / norm_x_prev;
                norm_x_prev = norm_x;
            }
            PROC_STR_STREAM << "\n";
            SA_PRINTF("%s", PROC_STR_STREAM.str().c_str());
            PROC_CLEAR_STR_STREAM;
        }

        if ((betanom < r0 && !zero_rhs) || (norm_x < r0 && zero_rhs))
        {
            if (print_iter == 2)
                SA_PRINTF("Number of PCG iterations: %d\n", i);
            else
                if (print_iter == 3)
                {
                    SA_PRINTF("(B r_0, r_0) = %g\n", nom0);
                    SA_PRINTF("(B r_N, r_N) = %g\n", betanom);
                    SA_PRINTF("Number of PCG iterations: %d\n", i);
                }
            iters = i;
            break;
        }

        beta = betanom/nom;
        add(z, beta, d, d);                   //  d = z + beta d
        A.Mult(d, z);
        den = mbox_parallel_inner_product(D, Z);
        nom = betanom;
    }
    if (i > max_num_iter)
    {
        SA_ALERT_PRINTF("%s", "PCG: No convergence!");
        SA_PRINTF("(B r_0, r_0) = %g\n", nom0);
        SA_PRINTF("(B r_N, r_N) = %g\n", betanom);
        SA_PRINTF("Number of PCG iterations: %d\n", i-1);
        iters = -(i-1);
    }
    if (print_iter >= 1 || i > max_num_iter)
    {
        if (i > max_num_iter)
            i--;
        PROC_STR_STREAM << "Average reduction factor = "
                        << pow(betanom/nom0, 0.5/i);
        if (zero_rhs)
            PROC_STR_STREAM << " (|| x ||_A / || x_0 ||_A)^(1/i) = "
                            << pow(norm_x / norm_x_initial, 1./(double)i);
        PROC_STR_STREAM << "\n";
        SA_PRINTF("%s", PROC_STR_STREAM.str().c_str());
        PROC_CLEAR_STR_STREAM;
    }
    return iters;
}
