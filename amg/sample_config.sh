# SAAMGE: smoothed aggregation element based algebraic multigrid hierarchies
#         and solvers.
# 
# Copyright (c) 2018, Lawrence Livermore National Security,
# LLC. Developed under the auspices of the U.S. Department of Energy by
# Lawrence Livermore National Laboratory under Contract
# No. DE-AC52-07NA27344. Written by Delyan Kalchev, Andrew T. Barker,
# and Panayot S. Vassilevski. Released under LLNL-CODE-667453.
# 
# This file is part of SAAMGE. 
# 
# Please also read the full notice of copyright and license in the file
# LICENSE.
# 
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License (as
# published by the Free Software Foundation) version 2.1 dated February
# 1999.
# 
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the IMPLIED WARRANTY OF
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the terms and
# conditions of the GNU Lesser General Public License for more details.
# 
# You should have received a copy of the GNU Lesser General Public
# License along with this program; if not, see
# <http://www.gnu.org/licenses/>.

#!/bin/bash

SAAMGE_BASE_DIR=${HOME}/saamge/for-lido/amg
SAAMGE_BUILD_DIR=${SAAMGE_BASE_DIR}/build
SAAMGE_INSTALL_DIR=${SAAMGE_BASE_DIR}/install

mkdir -p $SAAMGE_BUILD_DIR
cd $SAAMGE_BUILD_DIR

# Force a reconfigure
rm CMakeCache.txt
rm -rf CMakeFiles

# DEBUG or OPTIMIZED

cmake \
    -DCMAKE_BUILD_TYPE=DEBUG \
    \
    -DMETIS_DIR=${HOME}/bin \
    -DHYPRE_DIR=${HOME}/hypre/debug \
    -DMFEM_DIR=${HOME}/mfem/debug-nopetsc \
    -DSuiteSparse_DIR=${HOME}/bin \
    \
    -DUSE_ARPACK=ON \
    -DARPACK_DIR=${HOME}/arpack/arpack-ng-install \
    -DARPACKPP_DIR=${HOME}/arpack/arpackpp \
    \
    -DLINK_NETCDF=OFF \
    -DNETCDF_DIR=${HOME}/packages/netcdf \
    \
    -DBLAS_LIBRARIES=/usr/lib64/libblas.so \
    -DLAPACK_LIBRARIES=/usr/lib64/liblapack.so \
    \
    -DCMAKE_INSTALL_PREFIX=${SAAMGE_INSTALL_DIR} \
    ${SAAMGE_BASE_DIR}
