/*! \file
    \brief Partitioning related functionality.

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
#ifndef _PART_HPP
#define _PART_HPP

#include "common.hpp"
#include <mfem.hpp>

namespace saamge
{

/* Functions */

/*! \brief separates partitioning into connected components.

  modifies partitioning, ensures all partitions are connected,
  and removes empty partitions

  this routine stolen from Parelag, which stole it from some old MFEM
*/
int connectedComponents(mfem::Array<int>& partitioning, const mfem::Table& conn);

/*! \brief Partitions a graph.

    Calls METIS.

    \param graph (IN) The unweighted graph as a relation table.
    \param weights (IN) the weights on the vertices of the graph (for us this
                        means the number of DOF per element).
    \param parts (IN/OUT) The desired number of partitions in the partitioning,
                          as input. As output: the number of non-empty
                          partitions, which is the number of actually generated
                          partitions.
    \param endge_weights (IN) the weights on the edges of the graph (can be used
                              for shaping the partitions in accordance to the problem).

    \returns The partitioning of the graph.

    \warning The returned array must be freed by the caller.
*/
int *part_generate_partitioning(const mfem::Table& graph, int *weights, int *parts,
                                int *endge_weights=NULL);

/*! \brief Partitions an unweighted graph.

    See part_generate_partitioning()
*/
int *part_generate_partitioning_unweighted(const mfem::Table& graph, int *parts);

/*! \brief Checks the integrity of the partitioning.

    Checks if there exist empty and non-connected partitions.

    \param graph (IN) The unweighted graph as a relation table.
    \param partitioning (IN) The partitioning.

    DEPRECATED
*/
void part_check_partitioning(const mfem::Table& graph, const int *partitioning);

}

#endif // _PART_HPP
