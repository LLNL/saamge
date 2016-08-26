/*! \file

    SAAMGE: smoothed aggregation element based algebraic multigrid hierarchies
            and solvers.

    Copyright (c) 2016, Lawrence Livermore National Security,
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

#ifndef _DOUBLECYCLE_HPP
#define _DOUBLECYCLE_HPP

#include "common.hpp"
#include "solve.hpp"
#include <mfem.hpp>
#include "xpacks.hpp"
#include "mbox.hpp"
#include "tg.hpp"
#include "ml.hpp"
#include "mfem_addons.hpp"

/*!
  \brief use two different coarse solvers at a single level

  ATB 26 June 2015
*/
class DoubleCycle : public mfem::Solver
{
public:
    /**
       this constructor does a (two-level) CorrectNullspace and a usual ml_data V-cycle
       there could easily be other constructors
    */
    DoubleCycle(mfem::HypreParMatrix& A, ml_data_t &ml_data);

    /**
       TODO: should actually just construct from two mfem::Solver references...
    */

    virtual ~DoubleCycle();
    virtual void SetOperator(const mfem::Operator &op) {} // needed for Solver interface, but I don't actually use... 
    virtual void Mult(const mfem::Vector &x, mfem::Vector &y) const;
private:
    mfem::HypreParMatrix &A;
    mfem::HypreParMatrix *Ac;
    mfem::HypreParMatrix *interp;
    mfem::HypreParMatrix *restr;
    smpr_ft pre_smoother;
    smpr_ft post_smoother;
    void * smoother_data;

    mfem::Solver * inner_solver; // originally this has been CorrectNullspace
    mfem::Solver * outer_solver; // originally this has been a VCycleSolver
};

#endif
