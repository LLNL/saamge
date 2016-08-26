/*! \file
    \brief Routines for working with the MPI process. They call MPI directly.

    SAAMGE: smoothed aggregation element based algebraic multigrid hierarchies
            and solvers.

    Copyright (c) 2016, Lawrence Livermore National Security,
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
#ifndef _PROCESS_HPP
#define _PROCESS_HPP

#include <mpi.h>
#include <sstream>
#include <string>
#include <mfem.hpp>
#include "config_mgr.hpp"
using std::stringstream;
using std::string;
using namespace mfem;

/* Defines */
/*! \brief Returns the rank of the current process.
*/
#define PROC_RANK           ((const int)proc_info.rank)

/*! \brief Returns the total number of processes.
*/
#define PROC_NUM           ((const int)proc_info.procs_num)

/*! \brief Returns the communicator.
*/
#define PROC_COMM           ((const MPI_Comm)proc_info.comm)

/*! \brief Aborts the process and the group

    \param err (IN) Error code.
*/
#define PROC_ABORT(err)     MPI_Abort(proc_info.comm, (err))

/*! \brief Clears the content of \b PROC_STR_STREAM.
*/
#define PROC_CLEAR_STR_STREAM \
    do { \
        PROC_STR_STREAM.str(string()); \
        PROC_STR_STREAM.clear(); \
    } while(0)

/* Types */
/*! \brief Structure containing process information.
*/
typedef struct {
    MPI_Comm comm; /*!< Communicator that is used by the process. */
    int rank; /*!< Rank of the current process. */
    int procs_num; /*!< Number of processes. */
} proc_info_t ;

/* Variables */
/*! Contains the process information.
*/
extern proc_info_t proc_info;

/*! Global (within the current process) string stream usually used for output.
*/
extern stringstream PROC_STR_STREAM;

/* Functions */
/*! \brief Needs to be call right after \em MPI_Init.

    \param comm (IN) The default communicator.
*/
void proc_init(MPI_Comm comm);

/*! \brief Returns global offsets for all processes. DEPRECATED.

    \param my_size (IN) How many entities the current process has.
    \param offsets (OUT) Offsets on all processes in global entities numbering.
*/
void proc_allgather_offsets(int my_size, Array<int>& offsets);

/*! \brief Returns non-global offsets for all processes.

  \param my_size (IN) How many entities the current process has.
  \param offsets (OUT) size 2, offsets of current process in global numbering
  \param total (OUT) total number of entities
*/
void proc_determine_offsets(int my_size, Array<int>& offsets, int& total);

#endif // _PROCESS_HPP
