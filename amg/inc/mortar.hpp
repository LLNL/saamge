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

   That is, reduces the problem to the cface space, solves or preconditions the reduced system, and
   recovers to the full space, including "interiors".
*/
class MortarSchurSolver : public mfem::Solver
{
public:
    MortarSchurSolver(const interp_data_t& interp_data, const agg_partitioning_relations_t& agg_part_rels,
                      const mfem::Solver& solver, bool rand_init_guess=false)
        : interp_data(interp_data), agg_part_rels(agg_part_rels), solver(solver), rand_init_guess(rand_init_guess) {}
    virtual void SetOperator(const mfem::Operator &op) {}
    virtual void Mult(const mfem::Vector &x, mfem::Vector &y) const;
private:
    const interp_data_t& interp_data;
    const agg_partitioning_relations_t& agg_part_rels;
    const mfem::Solver& solver;
    const bool rand_init_guess;
};

/*! Assembles the global mortar condensed (i.e., defined only on cface DoFs) rhs.
    Caller must free the returned vector.
*/
mfem::HypreParVector *mortar_assemble_condensed_rhs(interp_data_t& interp_data,
                                              const agg_partitioning_relations_t& agg_part_rels,
                                              ElementMatrixProvider *elem_data);

/*! Builds a "fine" condensed mortar formulation and the respective spaces using (or abusing) the TG structure.
    Essential BCs are removed from the spaces via having vanishing basis functions on that portion of the boundary.
    tg_data should already have some basic initializations via tg_init_data().

    \a face_targets provides global targets, the span of whose restriction to agglomerate faces, provides the
    coarse face space. If NOT provided, a global constant is used.

    \a diagonal gives the option to provide a vector that serves as a diagonal matrix (of size equal to
    the number of dofs on the processor), whose restrictions define inner products for the agglomerate face
    penalty terms. It is generic but the main intent is to be used to provide the entries of the global stiffness
    matrix diagonal. In parallel, some communication would be needed on the shared dofs so that all entries
    of interest become known to the processor, i.e., the processor needs to know the entries for all dofs it sees
    NOT just the dofs it owns. This involves a dof_truedof application.
*/
void mortar_discretization(tg_data_t& tg_data, agg_partitioning_relations_t& agg_part_rels,
                           ElementMatrixProvider *elem_data, const mfem::Array<mfem::Vector *> *face_targets=NULL,
                           const mfem::Vector *diagonal=NULL);

/*! Returns an H^1 vector (expressed in true dofs) from a face mortar cface vector (also expressed in true dofs).
    This involves calling the averaging interpolator from the full mortar space to H^1.
    The returned vector must be freed by the caller.
*/
mfem::HypreParVector *mortar_reverse_condensation(const mfem::HypreParVector& mortar_sol, const tg_data_t& tg_data,
                                            const agg_partitioning_relations_t& agg_part_rels);

/**
    Assembles the global rhs coming from eliminating the "interior" DoFs. The output vector
    is represented in terms of true cface DoFs (i.e., defined only on cface DoFs)
    and must be freed by the caller. The input vector (\a rhs) is in terms of true DoFs that also include the "interior" DoFs.
    This is not much of a challenge, since all "interior" dofs are
    always true dofs (no sharing) and adding and removing interior dofs is essentially working with
    \a interp_data.celements_cdofs number of dofs at the beginning of the vector. The rest of the vector
    is filled with cface dofs only. Only the "interior" DoFs are used from the input vector.

    Lagrangian multipliers DoFs are not involved in the representation of the vectors. Internally, the function
    introduces them by padding with zeros.
*/
mfem::HypreParVector *mortar_assemble_schur_rhs(const interp_data_t& interp_data,
                    const agg_partitioning_relations_t& agg_part_rels, const mfem::Vector& rhs);

/**
    Performs the backward substitution from the block elimination. It takes the full (including "interiors" and cfaces)
    \a rhs in true DoFs and the face portion of the (obtained by inverting the Schur complement) solution in face true DoFs (excluding "interiors").
    The returned vector (i.e., x) is in terms of all (including "interiors") true DoFs. Only the "interior" DoFs are used from the rhs.
*/
void mortar_schur_recover(const interp_data_t& interp_data,
                    const agg_partitioning_relations_t& agg_part_rels,
                    const mfem::Vector& rhs, const mfem::Vector& facev, mfem::Vector& x);

} // namespace saamge

#endif // _MORTAR_HPP
