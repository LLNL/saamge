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

# Contents of this file stolen from Parelag's ParELAGCMakeUtilities.cmake

# Function that uses "dumb" logic to try to figure out if a library
# file is a shared or static library. This won't work on Windows; it
# will just return "unknown" for everything.
function(parelag_determine_library_type lib_name output_var)

  # Test if ends in ".a"
  string(REGEX MATCH "\\.a$" _static_match ${lib_name})
  if (_static_match)
    set(${output_var} STATIC PARENT_SCOPE)
    return()
  endif (_static_match)

  # Test if ends in ".so(.version.id.whatever)"
  string(REGEX MATCH "\\.so($|..*$)" _shared_match ${lib_name})
  if (_shared_match)
    set(${output_var} SHARED PARENT_SCOPE)
    return()
  endif (_shared_match)

  # Test if ends in ".dylib(.version.id.whatever)"
  string(REGEX MATCH "\\.dylib($|\\..*$)" _mac_shared_match ${lib_name})
  if (_mac_shared_match)
    set(${output_var} SHARED PARENT_SCOPE)
    return()
  endif (_mac_shared_match)

  set(${output_var} "UNKNOWN" PARENT_SCOPE)
endfunction(parelag_determine_library_type lib_name output)
