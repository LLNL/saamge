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

    This file was contributed by Umberto Villa.
*/

#ifndef INVERSEPERMEABILITYFUNCTION_HPP_
#define INVERSEPERMEABILITYFUNCTION_HPP_

class InversePermeabilityFunction
{
public:

    enum SliceOrientation {NONE, XY, XZ, YZ};

    static void SetNumberCells(int Nx_, int Ny_, int Nz_);
    static void SetMeshSizes(double hx, double hy, double hz);
    static void Set2DSlice(SliceOrientation o, int npos );

    static void ReadPermeabilityFile(const std::string fileName);
    static void SetConstantInversePermeability(double ipx, double ipy, 
					       double ipz);

    static void InversePermeability(const Vector & x, Vector & val);
    static double PermeabilityXComponent(Vector &x);
    static void PermeabilityTensor(const Vector & x, DenseMatrix & val);
    static void NegativeInversePermeability(const Vector & x, Vector & val);
    static void Permeability(const Vector & x, Vector & val);

    static double Norm2InversePermeability(const Vector & x);
    static double Norm1InversePermeability(const Vector & x);
    static double NormInfInversePermeability(const Vector & x);

    static double InvNorm2(const Vector & x);
    static double InvNorm1(const Vector & x);
    static double InvNormInf(const Vector & x);

    static void ClearMemory();

private:
    static int Nx;
    static int Ny;
    static int Nz;
    static double hx;
    static double hy;
    static double hz;
    static double * inversePermeability;

    static SliceOrientation orientation;
    static int npos;
};

#endif /* INVERSEPERMEABILITYFUNCTION_HPP_ */
