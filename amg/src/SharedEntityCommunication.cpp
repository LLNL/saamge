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
   \file

   \brief A class to manage shared entity communication

   This particular file contains specific instantiations for data types.
   You need to reimplement each of these routines for each datatype
   you want to communicate.

   Andrew T. Barker
   atb@llnl.gov
   17 July 2015
*/

#include "SharedEntityCommunication.hpp"

namespace saamge
{

template <>
void SharedEntityCommunication<mfem::DenseMatrix>::SetSizeSpecifier()
{
    size_specifier = 2;
}

template <>
void SharedEntityCommunication<mfem::DenseMatrix>::PackSendSizes(
    const mfem::DenseMatrix& mat, int * sizes)
{
    sizes[0] = mat.Height();
    sizes[1] = mat.Width();
}

template <>
void SharedEntityCommunication<mfem::DenseMatrix>::CopyData(
    mfem::DenseMatrix& copyto, 
    const mfem::DenseMatrix& copyfrom)
{
    // todo: should just use operator= ?
    //       or should use std::copy ?
    copyto.SetSize(copyfrom.Height(), copyfrom.Width());
    memcpy(copyto.Data(), copyfrom.Data(), 
           copyfrom.Height() * copyfrom.Width() * sizeof(double));
}

template <>
void SharedEntityCommunication<mfem::DenseMatrix>::SendData(
    const mfem::DenseMatrix& mat,
    int recipient,
    int tag,
    MPI_Request * request)
{
    MPI_Isend(mat.Data(), mat.Height() * mat.Width(), MPI_DOUBLE, 
              recipient, tag, comm, request);
}
                                                           
template <>
void SharedEntityCommunication<mfem::DenseMatrix>::ReceiveData(
    mfem::DenseMatrix& mat,
    int * sizes,
    int sender,
    int tag,
    MPI_Request *request)
{
    int rows = sizes[0];
    int columns = sizes[1];
    mat.SetSize(rows,columns);
    MPI_Irecv(mat.Data(),
              rows * columns,
              MPI_DOUBLE,
              sender,
              tag,
              comm,
              request);
}

template class SharedEntityCommunication<mfem::DenseMatrix>;

template <>
void SharedEntityCommunication<mfem::Vector>::SetSizeSpecifier()
{
    size_specifier = 1;
}

template <>
void SharedEntityCommunication<mfem::Vector>::PackSendSizes(
    const mfem::Vector& vec, int * sizes)
{
    sizes[0] = vec.Size();
}

template <>
void SharedEntityCommunication<mfem::Vector>::CopyData(
    mfem::Vector& copyto, 
    const mfem::Vector& copyfrom)
{
    copyto.SetSize(copyfrom.Size());
    copyto = copyfrom;
}

template <>
void SharedEntityCommunication<mfem::Vector>::SendData(
    const mfem::Vector& vec,
    int recipient,
    int tag,
    MPI_Request * request)
{
    MPI_Isend(vec.GetData(), vec.Size(), MPI_DOUBLE, 
              recipient, tag, comm, request);
}

template <>
void SharedEntityCommunication<mfem::Vector>::ReceiveData(
    mfem::Vector& vec,
    int * sizes,
    int sender,
    int tag,
    MPI_Request *request)
{
    int size = sizes[0];
    vec.SetSize(size);
    MPI_Irecv(vec.GetData(),
              size,
              MPI_DOUBLE,
              sender,
              tag,
              comm,
              request);
}

template class SharedEntityCommunication<mfem::Vector>;

} // namespace saamge
