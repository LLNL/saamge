/*! \file

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

#include "common.hpp"
#include "part.hpp"
#define __STDC_LIMIT_MACROS
#include <climits>
#include <mfem.hpp>
#ifndef MFEM_USE_METIS_5
// METIS 4 prototypes
typedef int idxtype;
extern "C" {
   void METIS_PartGraphKway(int*, idxtype*, idxtype*, idxtype*, idxtype*,
                            int*, int*, int*, int*, int*, idxtype*);
}
#else
extern "C" {
#include "metis.h"
}
#endif

namespace saamge
{
using namespace mfem;

/* Functions */

int connectedComponents(Array<int>& partitioning, const Table& conn)
{
    MFEM_ASSERT(partitioning.Size() == conn.Size(),"Wrong sized input!");
    MFEM_ASSERT(partitioning.Size() == conn.Width(),"Wrong sized input!");
    int num_nodes = conn.Size();
    int num_part(partitioning.Max()+1);

    Array<int> component(num_nodes);
    component = -1;
    Array<int> offset_comp(num_part+1);
    offset_comp = 0;
    Array<int> num_comp(offset_comp.GetData()+1, num_part);
    int i, j, k;
    const int *i_table, *j_table;

    i_table = conn.GetI();
    j_table = conn.GetJ();

    Array<int> vertex_stack(num_nodes);
    int stack_p, stack_top_p, node;

    stack_p = 0;
    stack_top_p = 0;  // points to the first unused element in the stack
    for (node = 0; node < num_nodes; node++)
    {
        if (partitioning[node] < 0)
            continue;

        if (component[node] >= 0)
            continue;

        component[node] = num_comp[partitioning[node]]++;
        vertex_stack[stack_top_p++] = node;

        for ( ; stack_p < stack_top_p; stack_p++)
        {
            i = vertex_stack[stack_p];
            if (partitioning[i] < 0)
                continue;

            for (j = i_table[i]; j < i_table[i+1]; j++)
            {
                k = j_table[j];
                if (partitioning[k] == partitioning[i] )
                {
                    if (component[k] < 0)
                    {
                        component[k] = component[i];
                        vertex_stack[stack_top_p++] = k;
                    }
                    MFEM_ASSERT(component[k] == component[i],"Impossible topology!");
                }
            }
        }
    }
    offset_comp.PartialSum();
    for(int i(0); i < num_nodes; ++i)
        partitioning[i] = offset_comp[partitioning[i]]+component[i];

    MFEM_ASSERT(partitioning.Max()+1 == offset_comp.Last(),
                "Partitioning inconsistent with components!");
    return offset_comp.Last();
}

int *part_generate_partitioning(const Table& graph, int *weights, int *parts,
                                int *endge_weights)
{
    SA_ASSERT(graph.Size() == graph.Width() || *parts == 1);
#ifndef MFEM_USE_METIS_5
    idxtype wgtflag = 0;
    idxtype numflag = 0;
    int options[5];
#else
    idx_t ncon = 1;
    int options[METIS_NOPTIONS];
    int stat;
#endif
    int nodes_number = graph.Size();
    int objval;

    int *partitioning = new int[nodes_number];
    Array<int> p_array(partitioning, nodes_number);
    const int target_parts = *parts;
    int actual_parts;
    SA_ASSERT(target_parts > 0);

    SA_RPRINTF_L(0, 4, "%s", "Partitioning graph...\n");

    if (target_parts > 1)
    {
#ifndef MFEM_USE_METIS_5
        options[0] = 0;
        METIS_PartGraphKway(&nodes_number,
                            const_cast<Table&>(graph).GetI(),
                            const_cast<Table&>(graph).GetJ(),
                            (idxtype *) weights,
                            (idxtype *) endge_weights,
                            &wgtflag,
                            &numflag,
                            parts,
                            options,
                            &objval,
                            (idxtype *) partitioning);

#else
        METIS_SetDefaultOptions(options);
        // Enforce, just in case, k-way partitioning.
        options[METIS_OPTION_PTYPE] = METIS_PTYPE_KWAY;
        // Enforce, just in case, C-style indices.
        options[METIS_OPTION_NUMBERING] = 0;
        // Require connected partitions.
        options[METIS_OPTION_CONTIG] = true;
        options[METIS_OPTION_UFACTOR] = 30;

        // Perform the partitioning.
        stat = METIS_PartGraphKway(&nodes_number,
                                   &ncon,
                                   const_cast<Table&>(graph).GetI(),
                                   const_cast<Table&>(graph).GetJ(),
                                   weights,
                                   NULL,
                                   endge_weights,
                                   parts,
                                   NULL,
                                   NULL,
                                   options,
                                   &objval,
                                   partitioning);
        SA_ASSERT(METIS_OK == stat);
#endif
        SA_ASSERT(target_parts == *parts);
    }
    else
    {
        memset(partitioning, 0, sizeof(*partitioning) * nodes_number);
    }

    // part_check_partitioning(graph, partitioning);
    if (target_parts > 1)
        connectedComponents(p_array, graph);
    // part_check_partitioning(graph, partitioning);
    actual_parts = p_array.Max() + 1;
    *parts = actual_parts;

    // Compute the number of non-empty partitions.
    SA_RPRINTF_L(0, 3, "Desired number of partitions: %d\n", target_parts);
    SA_RPRINTF_L(0, 3, "Actual number of partitions: %d\n", actual_parts);

    return partitioning;
}

int *part_generate_partitioning_unweighted(const Table& graph, int *parts)
{
    SA_ASSERT(graph.Size() == graph.Width() || *parts == 1);
//    int * weights = new int[graph.Size()];
//    for (int i=0; i<graph.Size(); ++i)
//        weights[i] = 1.0;
    int * out = part_generate_partitioning(graph, NULL, parts);
//    delete [] weights;
    return out;
}

//XXX: MFEM defines this non-static function but no prototype is declared
//     in any header file.
// DEPRECATED, see connectedComponents()
void FindPartitioningComponents(Table &elem_elem,
                                const Array<int> &partitioning,
                                Array<int> &component,
                                Array<int> &num_comp)
{
   int i, j, k;
   int num_elem, *i_elem_elem, *j_elem_elem;

   num_elem    = elem_elem.Size();
   i_elem_elem = elem_elem.GetI();
   j_elem_elem = elem_elem.GetJ();

   component.SetSize(num_elem);

   Array<int> elem_stack(num_elem);
   int stack_p, stack_top_p, elem;
   int num_part;

   num_part = -1;
   for (i = 0; i < num_elem; i++)
   {
      if (partitioning[i] > num_part)
         num_part = partitioning[i];
      component[i] = -1;
   }
   num_part++;

   num_comp.SetSize(num_part);
   for (i = 0; i < num_part; i++)
      num_comp[i] = 0;

   stack_p = 0;
   stack_top_p = 0;  // points to the first unused element in the stack
   for (elem = 0; elem < num_elem; elem++)
   {
      if (component[elem] >= 0)
         continue;

      component[elem] = num_comp[partitioning[elem]]++;

      elem_stack[stack_top_p++] = elem;

      for ( ; stack_p < stack_top_p; stack_p++)
      {
         i = elem_stack[stack_p];
         for (j = i_elem_elem[i]; j < i_elem_elem[i+1]; j++)
         {
            k = j_elem_elem[j];
            if (partitioning[k] == partitioning[i])
            {
               if (component[k] < 0)
               {
                  component[k] = component[i];
                  elem_stack[stack_top_p++] = k;
               }
               else if (component[k] != component[i])
               {
                  mfem_error("FindPartitioningComponents");
               }
            }
         }
      }
   }
}


// DEPRECATED, see connectedComponents
void part_check_partitioning(const Table& graph, const int *partitioning)
{
    SA_ASSERT(graph.Size() == graph.Width());
    const Array<int> partitioning_arr(const_cast<int *>(partitioning),
                                      graph.Size());
    Array<int> component, num_comp;
    int non_con_parts = 0;
    int empty_parts = 0;

    // Find the connected components of each partition.
    FindPartitioningComponents(const_cast<Table&>(graph), partitioning_arr,
                               component, num_comp);

    SA_PRINTF_L(3,"Number of components: %d\n", num_comp.Size());

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
        SA_PRINTF_L(3,"%s", "All partitions are connected.\n");
    else
        SA_PRINTF_L(3,"%d NON-connected partitions!\n", non_con_parts);
    if (!empty_parts)
        SA_PRINTF_L(3,"%s", "All partitions are non-empty.\n");
    else
        SA_ALERT_PRINTF("%d EMPTY partitions!", empty_parts);
}

} // namespace saamge
