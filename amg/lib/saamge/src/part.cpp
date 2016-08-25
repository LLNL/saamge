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
#include "part.hpp"
#define __STDC_LIMIT_MACROS
#include <climits>
#include <mfem.hpp>
extern "C" {
#include <metis.h>
}

#if (IDX_MAX != INT_MAX || IDX_MIN != INT_MIN)
#error "idx_t does NOT match int!"
#endif

/* Options */

CONFIG_DEFINE_CLASS(PART);

/* Functions */

int *part_generate_partitioning(const Table& graph, int *parts)
{
    SA_ASSERT(graph.Size() == graph.Width());
    idx_t options[METIS_NOPTIONS];
    idx_t nodes_number = graph.Size();
    idx_t ncon = 1, objval;
    int *partitioning = new int[nodes_number];
    int stat;
    const int target_parts = *parts;
    SA_ASSERT(target_parts > 0);

    SA_PRINTF_L(4, "%s", "Partitioning graph...\n");

    if (target_parts > 1)
    {
        // Initialize options with their defaults.
        METIS_SetDefaultOptions(options);
        // Enforce, just in case, k-way partitioning.
        options[METIS_OPTION_PTYPE] = METIS_PTYPE_KWAY;
        // Enforce, just in case, C-style indices.
        options[METIS_OPTION_NUMBERING] = 0;
        // Require connected partitions.
        options[METIS_OPTION_CONTIG] = CONFIG_ACCESS_OPTION(PART,
                                       connected_parts);

        // Perform the partitioning.
        stat = METIS_PartGraphKway(&nodes_number,
                                   &ncon,
                                   const_cast<Table&>(graph).GetI(),
                                   const_cast<Table&>(graph).GetJ(),
                                   NULL,
                                   NULL,
                                   NULL,
                                   parts,
                                   NULL,
                                   NULL,
                                   options,
                                   &objval,
                                   partitioning);
        SA_ASSERT(METIS_OK == stat);
        SA_ASSERT(target_parts == *parts);
    } else
        memset(partitioning, 0, sizeof(*partitioning) * nodes_number);

    // Compute the number of non-empty partitions.
    SA_PRINTF_L(3, "Desired number of partitions: %d\n", target_parts);
    int *part_size = new int[target_parts];
    int i;
    memset(part_size, 0, sizeof(*part_size) * target_parts);
    for (i=0; i < nodes_number; ++i)
    {
        SA_ASSERT(target_parts > partitioning[i] && partitioning[i] >= 0);
        ++part_size[partitioning[i]];
    }
    int empty_partitions = 0;
    Array<int> empty_parts(target_parts);
    for (i=0; i < target_parts; ++i)
    {
        if (!part_size[i])
        {
            if (SA_IS_OUTPUT_LEVEL(4))
                SA_ALERT_PRINTF("Empty partition: %d", i);
            --(*parts);
            SA_ASSERT(empty_partitions < target_parts);
            empty_parts[empty_partitions++] = i;
        }
    }
    SA_ASSERT(target_parts == *parts); //XXX: Does not handle empty partitions
                                       //     currently.
    SA_PRINTF_L(3, "Produced partitions: %d and %d empty ones.\n", *parts,
                empty_partitions);
    SA_ASSERT(empty_partitions <= target_parts);
    SA_ASSERT(target_parts - *parts == empty_partitions);

#if (SA_IS_DEBUG_LEVEL(4))
    // Check the integrity of the partitioning.
    SA_PRINTF("%s", "--------- graph { ---------\n");
    part_check_partitioning(graph, partitioning);
    SA_PRINTF("%s", "--------- } graph ---------\n");
#endif

    // Fix partitioning if necessary.
    if (empty_partitions)
    {
        empty_parts.SetSize(empty_partitions);
        empty_parts.Sort();
        part_remove_empty(partitioning, empty_parts, nodes_number,
                          target_parts);
#if (SA_IS_DEBUG_LEVEL(4))
        // Check the integrity of the partitioning.
        SA_PRINTF("%s", "--------- graph { ---------\n");
        part_check_partitioning(graph, partitioning);
        SA_PRINTF("%s", "--------- } graph ---------\n");
#endif
    }

    delete [] part_size;
    return partitioning;
}

void part_remove_empty(int *partitioning, const Array<int>& empty_parts,
                       int nodes_number, int target_parts)
{
    SA_PRINTF_L(4, "%s", "Fixing partitioning by removing empty"
                         " partitions...\n");
    int i, idx, curr;
    int *subtr_amount = new int[target_parts];

    SA_ASSERT(partitioning);
    SA_ASSERT(empty_parts.Size() > 0);
    const int sz = empty_parts.Size();
    for ((i=0), (idx=0), (curr = empty_parts[0]); i < target_parts; ++i)
    {
        SA_ASSERT(idx <= sz);
        if (idx < sz && i > curr)
        {
            ++idx;
            curr = idx == sz ? target_parts : empty_parts[idx];
        }
        subtr_amount[i] = i - idx;
        SA_ASSERT(0 <= subtr_amount[i] && subtr_amount[i] <= target_parts - sz);

#ifdef SA_ASSERTS
        if (i == curr)
            subtr_amount[i] = -1;
#endif
    }

    for (int i=0; i < nodes_number; ++i)
    {
        const int partition = partitioning[i];
        SA_ASSERT(target_parts > partition && partition >= 0);
        SA_ASSERT(0 <= subtr_amount[partition] &&
                  subtr_amount[partition] <= target_parts - sz);
        SA_ASSERT(0 <= partition - subtr_amount[partition]);
        SA_ASSERT(partition - subtr_amount[partition] <= sz);
        if (SA_IS_OUTPUT_LEVEL(15) && subtr_amount[partition])
        {
            SA_PRINTF("Fixing partition for node %d from %d to %d by"
                      " subtracting %d.\n", i, partition,
                      subtr_amount[partition],
                      partition - subtr_amount[partition]);
        }
        partitioning[i] = subtr_amount[partition];
    }

#if (SA_IS_DEBUG_LEVEL(2))
    const int nonemptyparts = target_parts - sz;
    int *part_size = new int[nonemptyparts];
    memset(part_size, 0, sizeof(*part_size) * nonemptyparts);
    for (i=0; i < nodes_number; ++i)
    {
        SA_ASSERT(nonemptyparts > partitioning[i] && partitioning[i] >= 0);
        ++part_size[partitioning[i]];
    }
    int empty_partitions = 0;
    for (i=0; i < nonemptyparts; ++i)
    {
        if (!part_size[i])
        {
            if (SA_IS_OUTPUT_LEVEL(4))
                SA_ALERT_PRINTF("Empty partition: %d", i);
            SA_ASSERT(empty_partitions < nonemptyparts);
            ++empty_partitions;
        }
    }
    SA_PRINTF_L(3, "Produced partitions: %d and %d empty ones.\n",
                   nonemptyparts, empty_partitions);
    SA_ASSERT(empty_partitions <= nonemptyparts);
    SA_ASSERT(!empty_partitions);
    delete [] part_size;
#endif

    delete [] subtr_amount;
}

void part_check_partitioning(const Table& graph, const int *partitioning)
{
    SA_ASSERT(graph.Size() == graph.Width());
    const Array<int> partitioning_arr(const_cast<int *>(partitioning),
                                      graph.Size());
    Array<int> component, num_comp;
    int non_con_parts = 0;
    int empty_parts = 0;

    //XXX: MFEM defines this non-static function but no prototype is declared
    //     in any header file.
    void FindPartitioningComponents(Table &elem_elem,
                                    const Array<int> &partitioning,
                                    Array<int> &component,
                                    Array<int> &num_comp);

    // Find the connected components of each partition.
    FindPartitioningComponents(const_cast<Table&>(graph), partitioning_arr,
                               component, num_comp);

    SA_PRINTF("Number of components: %d\n", num_comp.Size());

    // Simply check which partitions have more than one or zero components.
    for (int i=0; i < num_comp.Size(); ++i)
    {
        if (num_comp[i] > 1)
        {
            SA_PRINTF_L(8, "Non-connected partition: %d\n", i);
            ++non_con_parts;
        } else if (!num_comp[i])
        {
            SA_ALERT_PRINTF("Empty partition: %d", i);
            ++empty_parts;
        }
    }
    if (!non_con_parts)
        SA_PRINTF("%s", "All partitions are connected.\n");
    else
        SA_PRINTF("%d NON-connected partitions!\n", non_con_parts);
    if (!empty_parts)
        SA_PRINTF("%s", "All partitions are non-empty.\n");
    else
        SA_ALERT_PRINTF("%d EMPTY partitions!", empty_parts);
}
