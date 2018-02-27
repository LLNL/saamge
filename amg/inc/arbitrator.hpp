/*! \file
    \brief Handles arbitration of degrees of freedom, to decide what 
           aggregates they should go in.

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
#ifndef _ARBITRATOR_HPP
#define _ARBITRATOR_HPP

#include "common.hpp"
#include <mfem.hpp>
#include "aggregates.hpp"

namespace saamge
{

/**
   Arbitration is neither mandatory (you can use MISes if you want)
   nor binding (it is only a suggestion, after all).
*/
class Arbitrator
{
public:
    /**
       Constructor, need information about global matrix to
       calculate strength of connection, and all the
       partitioning information.
    */
    Arbitrator(mfem::HypreParMatrix& Aglobal, // want to be const, but won't be
               const agg_partitioning_relations_t &agg_part_rels,
               const agg_dof_status_t * const agg_flags);

    ~Arbitrator();

    /**
       Suggest a home for DoF i.

       Input i is in local, overlapped numbering, which means it is a local
       dof but not a truedof.

       Output is the aggregate to put this DoF in, in local numbering.

       This implementation copied from Delyan's agg_simple_strength_suggest(),
       and the whole class can be viewed as just a wrapper for this C-style
       routine.
    */
    int suggest(int i);

private:
    // mimicing agg_simple_strength_suggest() interface
    mfem::HypreParMatrix& Aglobal;
    mfem::HypreParMatrix& dof_truedof;
    const int * const aggregates;
    const int * const agg_size;
    const agg_dof_status_t * const agg_flags;
    const mfem::Table& dof_to_AE;
    const int nparts;
    const int num_dof; // not num_truedof

    mfem::Vector local_diag; // diagonal entries of local (diagonal...) matrix

    // utility Hypre pointers, do not free
    hypre_CSRMatrix * dof_truedof_diag;
    
    // transposed matrix, free at destructor
    hypre_CSRMatrix * truedof_dof_diag;
};

}

#endif
