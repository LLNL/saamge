/*! \file
    \brief Partitioning related functionality.

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

#pragma once
#ifndef _PART_HPP
#define _PART_HPP

#include "common.hpp"
#include <mfem.hpp>

/* Options */

/*! \brief The configuration class of this module.
*/
CONFIG_BEGIN_CLASS_DECLARATION(PART)

    /*! Whether connected partitions are required. */
    CONFIG_DECLARE_OPTION(bool, connected_parts);

CONFIG_END_CLASS_DECLARATION(PART)

CONFIG_BEGIN_INLINE_CLASS_DEFAULTS(PART)
    CONFIG_DEFINE_OPTION_DEFAULT(connected_parts, true)
CONFIG_END_CLASS_DEFAULTS

/* Functions */
/*! \brief Partitions an unweighted graph.

    Calls METIS.

    \param graph (IN) The unweighted graph as a relation table.
    \param parts (IN/OUT) The desired number of partitions in the partitioning,
                          as input. As output: the number of non-empty
                          partitions, which is the number of actually generated
                          partitions.

    \returns The partitioning of the graph.

    \warning The returned array must be freed by the caller.
*/
int *part_generate_partitioning(const Table& graph, int *parts);

/*! \brief Fixes a partitioning by removing empty partitions.

    \param partitioning (IN/OUT) The partitioning to be fixed as input and the
                                 fixed one as output.
    \param empty_parts (IN) A SORTED array of the indices of the empty
                            partitions.
    \param nodes_number (IN) The number of nodes of the partitioned graph.
    \param target_parts (IN) The initially desired number of partitions.
*/
void part_remove_empty(int *partitioning, const Array<int>& empty_parts,
                       int nodes_number, int target_parts);

/*! \brief Checks the integrity of the partitioning.

    Checks if there exist empty and non-connected partitions.

    \param graph (IN) The unweighted graph as a relation table.
    \param partitioning (IN) The partitioning.
*/
void part_check_partitioning(const Table& graph, const int *partitioning);

#endif // _PART_HPP
