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

list(APPEND EXE_SRCS algebraic/algebraic.cpp basicupscale/basicupscale.cpp
  mltest/mltest.cpp partialsmooth/partialsmooth.cpp parttest/parttest.cpp startfromcoarse/startfromcoarse.cpp
  encapsulate/encapsulate.cpp nonconf/tg.cpp nonconf/interior_penalty.cpp nonconf/interior_penalty_saamge.cpp)

list(APPEND EXE_SRCS  leastsquaretest/leastsquaretest.cpp 
                      secondorderpdetest/secondorderpdetest.cpp
                      leastsquarealgebraictest/leastsquarealgebraictest.cpp
                      secondorderpdealgebraictest/secondorderpdealgebraictest.cpp
                     )


# Add one executable per cpp file
foreach(SRC_FILE IN LISTS EXE_SRCS)
  get_filename_component(SRC_FILENAME ${SRC_FILE} NAME)
  
  string(REPLACE ".cpp" "" EXE_NAME ${SRC_FILENAME})
  add_executable(${EXE_NAME} ${SRC_FILE})

  target_link_libraries(${EXE_NAME} ${PROJECT_NAME})
  target_link_libraries(${EXE_NAME} ${TPL_LIBRARIES})
  target_link_libraries(${EXE_NAME} "m")

  if(MPI_COMPILE_FLAGS)
    set_target_properties(${EXE_NAME} PROPERTIES
      COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
  endif()

  if(MPI_LINK_FLAGS)
    set_target_properties(${EXE_NAME} PROPERTIES
      LINK_FLAGS "${MPI_LINK_FLAGS}")
  endif()

endforeach(SRC_FILE)

# poached from cubetest.py (should be in test/CMakeLists.txt)
add_test(cube
  mltest -m ${MESH_DIR}/cube474.mesh3d --num-levels 3 --no-visualization
  --refine 2 --first-elems-per-agg 128 --elems-per-agg 64
  --theta 1.e-4 --nu-pro 0 --constant-coefficient)
set_tests_properties(cube
  PROPERTIES
  PASS_REGULAR_EXPRESSION
  "Outer PCG converged in 4 iterations.")

add_test(algebraic
  algebraic --elems-per-agg 128 --theta 0.01 --nu-pro 0
  --matrix ${PROJECT_SOURCE_DIR}/data/anisotropic.mat.00000 --no-correct-nulspace)
set_tests_properties(algebraic
  PROPERTIES
  PASS_REGULAR_EXPRESSION
  "Outer PCG converged in 12 iterations.")
