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

#####
# when you invoke cmake, you should set:
#   SAAMGE_BASE_DIR
#   SAAMGE_BUILD_DIR
#
# see config.sh
# note that config.sh and config.py do very different things...
#####

###
# setup the basics
###
cmake_minimum_required(VERSION 2.8.11 FATAL_ERROR)
set(PROJECT_NAME saamge)
project(${PROJECT_NAME})
set(HOME_DIR $ENV{HOME})
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/modules)

set(MESH_DIR ${HOME_DIR}/meshes)

# TODO: no C++11 in SAAMGe
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++11")

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -fstack-protector-all")
set(CMAKE_CXX_FLAGS_OPTIMIZED "${CMAKE_CXX_FLAGS_OPTIMIZED} -O3")

# TODO: take TPLs from command line to override below locations if desired
#       also, make a config.sh that tells you where to find optimized libraries

###
# Find third-party libraries
#
# we are still essentially using Delyan's symbolic link structure,
# but just doing it with cmake now
#
# note that the order these are appended matters for linking, which means
# cmake is not doing one of the things I would hope it would do
###
set(TPL_LIBRARIES "")

# MPI
find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})
list(APPEND TPL_LIBRARIES ${MPI_LIBRARIES})
if(MPI_CXX_COMPILE_FLAGS)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MPI_CXX_COMPILE_FLAGS}")
endif()

# BLAS/LAPACK
find_package(BLAS REQUIRED)
list(APPEND TPL_LIBRARIES ${BLAS_LIBRARIES})
find_package(LAPACK REQUIRED)
list(APPEND TPL_LIBRARIES ${LAPACK_LIBRARIES})

# bulk includes (for fortran stuff...)
list(APPEND TPL_LIBRARIES "gfortran")

# MFEM
find_path(MFEM_INCLUDE_PATH mfem.hpp
          HINTS ${MFEM_DIR}/include ${MFEM_DIR} $ENV{MFEM_DIR} $ENV{MFEM_DIR}/include)
set(MFEM_LIB_NAME libmfem.a)
find_path(MFEM_LIBRARY_PATH ${MFEM_LIB_NAME}
  HINTS ${MFEM_DIR}/lib ${MFEM_DIR} $ENV{MFEM_DIR} $ENV{MFEM_DIR}/lib)
include_directories(${MFEM_INCLUDE_PATH})
add_library(MFEM_LIB STATIC IMPORTED)
set_property(TARGET MFEM_LIB PROPERTY IMPORTED_LOCATION ${MFEM_LIBRARY_PATH}/${MFEM_LIB_NAME})
list(APPEND TPL_LIBRARIES ${MFEM_LIBRARY_PATH}/${MFEM_LIB_NAME})

# Metis
find_path(METIS_INCLUDE_PATH metis.h
  HINTS ${METIS_DIR} ${METIS_DIR}/Lib ${METIS_DIR}/include $ENV{METIS_DIR}/include)
set(METIS_LIB_NAME libmetis.a)
find_path(METIS_LIBRARY_PATH ${METIS_LIB_NAME}
  HINTS ${METIS_DIR}/lib ${METIS_DIR} $ENV{METIS_DIR}/lib)
include_directories(${METIS_INCLUDE_PATH})
add_library(METIS_LIB STATIC IMPORTED)
set_property(TARGET METIS_LIB PROPERTY IMPORTED_LOCATION ${METIS_LIBRARY_PATH}/${METIS_LIB_NAME})
list(APPEND TPL_LIBRARIES ${METIS_LIBRARY_PATH}/${METIS_LIB_NAME})

# Hypre
find_path(HYPRE_INCLUDE_PATH HYPRE.h
          HINTS ${HYPRE_DIR}/include $ENV{HYPRE_DIR}/include)
include_directories(${HYPRE_INCLUDE_PATH})
set(HYPRE_LIB_NAME libHYPRE.a)
find_path(HYPRE_LIBRARY_PATH ${HYPRE_LIB_NAME}
          HINTS  ${HYPRE_DIR}/lib $ENV{HYPRE_DIR}/lib)
add_library(HYPRE_LIB STATIC IMPORTED)
set_property(TARGET HYPRE_LIB PROPERTY IMPORTED_LOCATION ${HYPRE_LIBRARY_PATH}/${HYPRE_LIB_NAME})
list(APPEND TPL_LIBRARIES ${HYPRE_LIBRARY_PATH}/${HYPRE_LIB_NAME})

# NETCDF - we do not actually use, but might link for MFEM
option(LINK_NETCDF "Should we link to NETCDF?" OFF)
if (LINK_NETCDF)
  set(NETCDF_LIB_NAME libnetcdf.so)
  find_path(NETCDF_LIBRARY_PATH ${NETCDF_LIB_NAME}
    HINTS ${NETCDF_DIR}/lib $ENV{NETCDF_DIR}/lib)
  add_library(NETCDF_LIB SHARED IMPORTED)
  set_property(TARGET NETCDF_LIB PROPERTY IMPORTED_LOCATION
    ${NETCDF_LIBRARY_PATH}/${NETCDF_LIB_NAME})
  list(APPEND TPL_LIBRARIES ${NETCDF_LIBRARY_PATH}/${NETCDF_LIB_NAME})
  set(${PROJECT_NAME}_LINK_NETCDF 1)
else()
  set(${PROJECT_NAME}_LINK_NETCDF 0)
endif()

# SuiteSparse
find_package(SuiteSparse REQUIRED UMFPACK KLU AMD BTF CHOLMOD COLAMD CAMD CCOLAMD config)
include_directories(${SuiteSparse_INCLUDE_DIRS})
list(APPEND TPL_LIBRARIES ${SuiteSparse_LIBRARIES})

# ARPACK 
option(USE_ARPACK "Should ARPACK be enabled?" ON)
if (USE_ARPACK)
  find_path(ARPACK_INCLUDE_PATH argsym.h
    HINTS ${ARPACKPP_DIR}/include $ENV{ARPACK_DIR}/include)
  set(ARPACK_LIB_NAME libarpack.a)
  find_path(ARPACK_LIBRARY_PATH ${ARPACK_LIB_NAME}
             HINTS ${ARPACK_DIR}/lib
                   ${ARPACK_DIR}/lib64
                   $ENV{ARPACK_DIR}/lib)
  include_directories(${ARPACK_INCLUDE_PATH})
  add_library(ARPACK_LIB STATIC IMPORTED)
  set_property(TARGET ARPACK_LIB PROPERTY IMPORTED_LOCATION
    ${ARPACK_LIBRARY_PATH}/${ARPACK_LIB_NAME})
  list(APPEND TPL_LIBRARIES ${ARPACK_LIBRARY_PATH}/${ARPACK_LIB_NAME})
  set(${PROJECT_NAME}_USE_ARPACK 1)
else()
  set(${PROJECT_NAME}_USE_ARPACK 0)
endif()

# PHYAGG - optional
option(USE_PHYAGG "Should PHYAGG be enabled?" OFF)
if (USE_PHYAGG)
  find_path(PHYAGG_INCLUDE_PATH phyagg.hpp
    HINTS ${PHYAGG_DIR}/src $ENV{PHYAGG_DIR}/src)
  set(PHYAGG_LIB_NAME libphyagg.a)
  find_path(PHYAGG_LIBRARY_PATH ${PHYAGG_LIB_NAME}
    HINTS ${PHYAGG_DIR}/build/src $ENV{PHYAGG_DIR}/build/src)
  include_directories(${PHYAGG_INCLUDE_PATH})
  add_library(PHYAGG_LIB STATIC IMPORTED)
  set_property(TARGET PHYAGG_LIB PROPERTY IMPORTED_LOCATION
    ${PHYAGG_LIBRARY_PATH}/${PHYAGG_LIB_NAME})
  list(APPEND TPL_LIBRARIES ${PHYAGG_LIBRARY_PATH}/${PHYAGG_LIB_NAME})
  set(${PROJECT_NAME}_USE_PHYAGG 1)
else()
  set(${PROJECT_NAME}_USE_PHYAGG 0)
endif()

list(REMOVE_DUPLICATES TPL_LIBRARIES)

###
# setup internal includes and test directories
###
enable_testing()

include_directories(${PROJECT_SOURCE_DIR}/inc)
include_directories(${PROJECT_BINARY_DIR}) # this one is for saamge_config.h

# this ends up putting the built library in
# ${BUILD_DIR}/src
# which may not be what I want
add_subdirectory(${PROJECT_SOURCE_DIR}/src)
add_subdirectory(${PROJECT_SOURCE_DIR}/test)

# Configure a header file to pass some of the CMake settings to the
# source code
configure_file(
  "${PROJECT_SOURCE_DIR}/inc/${PROJECT_NAME}_config.h.in"
  "${PROJECT_BINARY_DIR}/${PROJECT_NAME}_config.h"
  )

###
# setup install target
###
install(FILES ${PROJECT_BINARY_DIR}/${PROJECT_NAME}_config.h DESTINATION include)

###
# setup a very basic test
###

add_test(mltest
  test/mltest -m ${PROJECT_SOURCE_DIR}/test/mltest.mesh --num-levels 2 --no-visualization --no-correct-nulspace)
set_tests_properties(mltest
  PROPERTIES
  PASS_REGULAR_EXPRESSION
  "Outer PCG converged in 3 iterations.")

add_test(pmltest
  mpirun -n 2 test/mltest -m ${PROJECT_SOURCE_DIR}/test/mltest.mesh --num-levels 2 --no-visualization --no-correct-nulspace)
set_tests_properties(mltest
  PROPERTIES
  PASS_REGULAR_EXPRESSION
  "Outer PCG converged in 3 iterations.")

add_test(mltest2
  test/mltest -m ${PROJECT_SOURCE_DIR}/test/mltest.mesh --num-levels 2 --no-visualization --no-correct-nulspace --order 2)
set_tests_properties(mltest2
  PROPERTIES
  PASS_REGULAR_EXPRESSION
  "Outer PCG converged in 4 iterations.")

add_test(threelevel
  test/mltest -m ${PROJECT_SOURCE_DIR}/test/mltest.mesh --num-levels 3 --no-visualization --no-correct-nulspace)
set_tests_properties(threelevel
  PROPERTIES
  PASS_REGULAR_EXPRESSION
  "Outer PCG converged in 3 iterations.")

add_test(elasticity
  test/mltest -m ${PROJECT_SOURCE_DIR}/test/mltest.mesh --num-levels 2 --no-visualization
    --no-correct-nulspace --constant-coefficient --zero-rhs
    --elasticity)
set_tests_properties(elasticity
  PROPERTIES
  PASS_REGULAR_EXPRESSION
  "Outer PCG converged in 3 iterations.")

# test Patrick Zulian's least squares systems
add_test(leastsquarealgebraic_runs
  test/leastsquarealgebraictest -k -20 -m woiefu -r 2)
set_tests_properties(leastsquarealgebraic_runs
  PROPERTIES PASS_REGULAR_EXPRESSION
  "csv_data:2,-20,2,803,56")
add_test(leastsquarealgebraic_hypre
  test/leastsquarealgebraictest -k -20 -m woiefu -r 2 -l -ns)
set_tests_properties(leastsquarealgebraic_hypre
  PROPERTIES PASS_REGULAR_EXPRESSION
  "csv_data:2,-20,2,867,96")
add_test(leastsquarealgebraic_fail
  test/leastsquarealgebraictest -k -50 -m woiefu -r 2)
set_tests_properties(leastsquarealgebraic_fail
  PROPERTIES PASS_REGULAR_EXPRESSION
  "csv_data:2,-50,2,803,115")
# this test should also search .hpp, but I can't quite figure out how
add_test(notabs
  grep -rP --include=*.cpp \t ${PROJECT_SOURCE_DIR})
set_tests_properties(notabs PROPERTIES WILL_FAIL "TRUE")
