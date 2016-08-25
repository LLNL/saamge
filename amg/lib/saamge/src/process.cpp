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
#include "process.hpp"
#include <mpi.h>
#include <sstream>
#include <mfem.hpp>
using std::stringstream;

/* Variables */

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

void proc_allgather_offsets(int my_size, Array<int>& offsets)
{
    offsets.SetSize(PROC_NUM + 1);
    MPI_Allgather(&my_size, 1, MPI_INT, &offsets[1], 1, MPI_INT, PROC_COMM);
    offsets[0] = 0;
    for (int i=1; i < PROC_NUM; ++i)
        offsets[i+1] += offsets[i];
}
