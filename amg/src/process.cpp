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
#include "process.hpp"
#include <mpi.h>
#include <sstream>
#include <mfem.hpp>
using std::stringstream;

namespace saamge
{
using namespace mfem;

/* Variables */

// global variable in the global namespace here
proc_info_t proc_info;

stringstream PROC_STR_STREAM;

/* Functions */

void proc_init(MPI_Comm comm)
{
    proc_info.comm = comm;
    MPI_Comm_size(proc_info.comm, &(proc_info.procs_num));
    MPI_Comm_rank(proc_info.comm, &(proc_info.rank));
    SA_RPRINTF_L(0, 1, "Number of processes: %d\n", proc_info.procs_num);
    PROC_CLEAR_STR_STREAM;
}


/*!
  DEPRECATED, moving to Hypre 2.10.0b which uses no global partition
  by default, which is probably what we want anyway
  see proc_determine_offsets
*/
void proc_allgather_offsets(int my_size, Array<int>& offsets)
{
    SA_ASSERT(false);
    offsets.SetSize(PROC_NUM + 1);
    MPI_Allgather(&my_size, 1, MPI_INT, &offsets[1], 1, MPI_INT, PROC_COMM);
    offsets[0] = 0;
    for (int i=1; i < PROC_NUM; ++i)
        offsets[i+1] += offsets[i];
}

/*!
  determine offsets (of length 2) for no global partition
  ATB 11 February 2015
*/
void proc_determine_offsets(int my_size, Array<int>& offsets, int& total)
{
    offsets.SetSize(2); // an Array<int> may be overkill here
    MPI_Scan(&my_size, &offsets[1], 1, MPI_INT, MPI_SUM, PROC_COMM);
    offsets[0] = offsets[1] - my_size;
    total = offsets[1];
    MPI_Bcast(&total, 1, MPI_INT, PROC_NUM-1, PROC_COMM);
}

} // namespace saamge
