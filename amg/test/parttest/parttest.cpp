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
   Test part.hpp routines, make sure metis weighting works.
*/

#include <mfem.hpp>
#include <mpi.h>
#include <saamge.hpp>

using namespace mfem;
using namespace saamge;

int main(int argc, char *argv[])
{
    // Initialize process related stuff.
    MPI_Init(&argc, &argv);
    proc_init(MPI_COMM_WORLD);

    // make a ring..
    const int N=10;
    Table tab;
    tab.MakeI(N);
    for (int i=0; i<N; ++i)
    {
        tab.AddAColumnInRow(i);
        tab.AddAColumnInRow(i);
    }
    tab.MakeJ();
    for (int i=0; i<N; ++i)
    {
        if (i>0)
            tab.AddConnection(i,i-1);
        else
            tab.AddConnection(i,N-1);
        if (i<N-1)
            tab.AddConnection(i,i+1);
        else
            tab.AddConnection(i,0);
    }
    tab.ShiftUpI(); // I hate the MFEM table interface, the worst thing since Pol Pot's killing fields
    tab.Finalize();

    tab.Print(std::cout);

    std::cout << "Unweighted partition:" << std::endl << "-----" << std::endl;
    int desired_parts = 2;
    int * unweighted_partition = part_generate_partitioning_unweighted(tab, &desired_parts);
    for (int i=0; i<N; ++i)
        std::cout << "  element " << i << " in part " << unweighted_partition[i] << std::endl;
    delete [] unweighted_partition;

    int * weights = new int[N];
    for (int i=0; i<N; ++i)
        weights[i] = 1;
    weights[2] = 100;
    // weights[3] = 100;
    std::cout << "Weighted partition:" << std::endl << "-----" << std::endl;
    int * weighted_partition = part_generate_partitioning(tab, weights, &desired_parts);
    for (int i=0; i<N; ++i)
        std::cout << "  element " << i << " in part " << weighted_partition[i] << std::endl;
    delete [] weighted_partition;
    delete [] weights;

    MPI_Finalize();
    return 0;
}
