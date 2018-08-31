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

/**
   @file This file and the respective .cpp file implement a nonconforming AMGe
   approach. It basically "breaks" the space with the first coarsening and reduces the problem to
   faces similar to static condensation. It is thus natural and easy to recursively extend to multiple levels and
   is suitable for high-order discretizations when a matrix-free approach is utilized.

   The main difference is in the first (finest) coarsening. On that level, the agglomerated (coarse) faces need to be obtained.
   Coarse faces are essentially certain MISes in terms of faces. Once the space is "broken" (and, thus, nonconforming in the sense
   that the coarse spaces are not subspaces of the finest), it remains "broken" on all coarse levels (which are nested). Therefore,
   after the first coarsening the rest of the coarsening procedures are very similar to the usual SAAMGE and now faces can be
   coarsened by simply considering MISes in terms of DoFs, since the faces were already separated (there are no corner DoFs that are
   shared between faces) and are entirely characterized by the DoFs on them.

   It is more convenient to keep this method separate from the rest of SAAMGE as it works in a slightly different way.

   Currently, this is aimed at solver hierarchies, although it may have some potential to become an upscaling approach for
   coarse discretizations.

   TODO: Matrix-free implementation. I will point out the exact spots where particular routines need to be replaced by matrix-free
         versions. Currently, we consider the idea of matrix-free implementations only on the finest level (first coarsening),
         since this is the most meaningful and it is less clear if there will be any benefit to have the coarse levels matrix-free.

   TODO: Consider making this easy to use for hybridized systems.
*/

#pragma once
#ifndef _NONCONF_HPP
#define _NONCONF_HPP

#include "common.hpp"
#include <mfem.hpp>
#include "tg_data.hpp"
#include "interp.hpp"
#include "aggregates.hpp"
#include "elmat.hpp"

namespace saamge
{

/**
   @brief Solves via Schur complement.

   Reduces the problem (by elimination) to a Schur complement system, then uses the given solver for that
   system and, in the end, recovers the eliminated (by backward substitution) variables.
*/
class SchurSolver : public mfem::Solver
{
public:
    SchurSolver(const interp_data_t& interp_data, const agg_partitioning_relations_t& agg_part_rels,
                const mfem::HypreParMatrix& cface_cDof_TruecDof, const mfem::HypreParMatrix& cface_TruecDof_cDof, const mfem::Solver& solver)
        : interp_data(interp_data), agg_part_rels(agg_part_rels), cface_cDof_TruecDof(cface_cDof_TruecDof),
          cface_TruecDof_cDof(cface_TruecDof_cDof), solver(solver) {}
    virtual void SetOperator(const mfem::Operator &op) {}
    virtual void Mult(const mfem::Vector &x, mfem::Vector &y) const;
private:
    const interp_data_t& interp_data;
    const agg_partitioning_relations_t& agg_part_rels;
    const mfem::HypreParMatrix& cface_cDof_TruecDof;
    const mfem::HypreParMatrix& cface_TruecDof_cDof;
    const mfem::Solver& solver;
};


/*! Builds a "coarse" interior penalty formulation and the respective space. tg_data should already
    have some basic initializations via tg_init_data().
*/
void nonconf_ip_coarsen_finest(tg_data_t& tg_data, agg_partitioning_relations_t& agg_part_rels,
                               ElementMatrixProvider *elem_data, double theta, double delta,
                               bool schur=true, bool full_space=false);

} // namespace saamge

#endif // _NONCONF_HPP
