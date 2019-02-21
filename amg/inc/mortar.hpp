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
   @file This is an addition to nonconf.cpp that concentrates on the mortar method.
*/

#pragma once
#ifndef _MORTAR_HPP
#define _MORTAR_HPP

#include "common.hpp"
#include <mfem.hpp>
#include "tg_data.hpp"
#include "interp.hpp"
#include "aggregates.hpp"
#include "elmat.hpp"

namespace saamge
{

/**
   @brief Solves via Schur complement (static condensation).

   Reduces the problem (by elimination) to a Schur complement system, then uses the given solver for that
   system and, in the end, recovers the eliminated (by backward substitution) variables.
*/
class MortarSchurSolver : public mfem::Solver
{
public:
    MortarSchurSolver(const interp_data_t& interp_data, const agg_partitioning_relations_t& agg_part_rels,
                      const mfem::Solver& solver)
        : interp_data(interp_data), agg_part_rels(agg_part_rels), solver(solver) {}
    virtual void SetOperator(const mfem::Operator &op) {}
    virtual void Mult(const mfem::Vector &x, mfem::Vector &y) const;
private:
    const interp_data_t& interp_data;
    const agg_partitioning_relations_t& agg_part_rels;
    const mfem::Solver& solver;
};

/*! Assembles the global mortar condensed rhs. Caller must free the returned vector.
*/
mfem::HypreParVector *mortar_assemble_condensed_rhs(interp_data_t& interp_data,
                                              const agg_partitioning_relations_t& agg_part_rels,
                                              ElementMatrixProvider *elem_data);

/*! Builds a "fine" condensed mortar formulation and the respective space using (or abusing) the TG structure.
    Essential BCs are removed from the space via having vanishing basis functions on that portion of the boundary.
    tg_data should already have some basic initializations via tg_init_data().
*/
void mortar_discretization(tg_data_t& tg_data, agg_partitioning_relations_t& agg_part_rels,
                           ElementMatrixProvider *elem_data, const mfem::Array<mfem::Vector *> *face_targets=NULL);

/*! Returns an H^1 vector (expressed in true dofs) from a face mortar vector (also expressed in true dofs).
    The returned vector must be freed by the caller
*/
mfem::HypreParVector *mortar_reverse_condensation(const mfem::HypreParVector& mortar_sol, const tg_data_t& tg_data,
                                            const agg_partitioning_relations_t& agg_part_rels);

/**
    Assembles the global rhs coming from eliminating the "interior" DoFs. The output vector
    is represented in terms of true cface DoFs (i.e., defined only on cface DoFs)
    and must be freed by the caller. The input vector is in terms of true DoFs that also include the actual interior DoFs.
*/
mfem::HypreParVector *mortar_assemble_schur_rhs(const interp_data_t& interp_data,
                    const agg_partitioning_relations_t& agg_part_rels, const mfem::Vector& rhs);

/**
    Performs the backward substitution from the block elimination. It takes the full (including interiors) original
    rhs in true DoFs and the face portion of the (obtained by inverting the Schur complement) solution in face true DoFs (excluding interiors).
    The returned vector (i.e., x) is in terms of all (including interiors) true DoFs.
*/
void mortar_schur_recover(const interp_data_t& interp_data,
                    const agg_partitioning_relations_t& agg_part_rels,
                    const mfem::Vector& rhs, const mfem::Vector& facev, mfem::Vector& x);

} // namespace saamge

#endif // _MORTAR_HPP
