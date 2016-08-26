# SAAMGE: smoothed aggregation element based algebraic multigrid hierarchies
#         and solvers.
# 
# Copyright (c) 2016, Lawrence Livermore National Security,
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

LIB = m mfem metis blas lapack gfortran HYPRE umfpack cholmod amd colamd suitesparseconfig arpack
BLDLIB = saamge

##Enable OpenMP support
USE_OPENMP = NO

ifneq ($(ROOT),.)
INCSUBD = $(shell find -L $(ROOT)/inc/ -maxdepth 1 -mindepth 1 -type d)
INCS = $(addprefix -I,$(INCSUBD))

TESTSUBD = $(shell find -L $(ROOT)/test/ -maxdepth 1 -mindepth 1 -type d)

LIBSUBD = $(shell find -L $(ROOT)/lib/ -maxdepth 1 -mindepth 1 -type d)
LIBS = $(addprefix -L,$(LIBSUBD))
THELIB = $(addprefix -l,$(BLDLIB) $(LIB))

EINCSUBD = $(shell find -L $(ROOT)/extinc/ -maxdepth 1 -mindepth 1 -type d)
EINCS = $(addprefix -I,$(EINCSUBD))

ELIBSUBD = $(shell find -L $(ROOT)/extlib/ -maxdepth 1 -mindepth 1 -type d)
ELIBS = $(addprefix -L,$(ELIBSUBD))

BLDLIBINCS = $(addsuffix /inc,$(addprefix -I$(ROOT)/lib/,$(BLDLIB)))
MOREINCS = $(addprefix -I,$(MOREINC))
endif

CC = mpicxx
CFLAGS = -g -O3 \
         -Wall -Wno-unused-parameter -Wextra \
         -fdata-sections -ffunction-sections \
         -I$(ROOT)/inc -I$(ROOT)/extinc $(INCS) $(BLDLIBINCS) $(EINCS) \
         $(MOREINCS)
LDFLAGS = -Wl,--gc-sections -L$(ROOT)/lib -L$(ROOT)/extlib $(LIBS) $(ELIBS) \
          $(THELIB)

OPENMP_FLAGS = -fopenmp

ifeq ($(USE_OPENMP), YES)
CC := $(CC) $(OPENMP_FLAGS)
endif

CXX = $(CC)
CXXFLAGS = $(CFLAGS)

AR = ar
ARFLAGS = rcs

vpath %.h    $(ROOT)/inc $(ROOT)/extinc $(INCSUBD) $(EINCSUBD)
vpath %.hpp  $(ROOT)/inc $(ROOT)/extinc $(INCSUBD) $(EINCSUBD)
vpath %.c    $(ROOT)/test $(TESTSUBD)
vpath %.cpp  $(ROOT)/test $(TESTSUBD)
vpath %.o    $(ROOT)/test $(TESTSUBD)
vpath %.a    $(ROOT)/lib $(ROOT)/extlib $(LIBSUBD) $(ELIBSUBD)
vpath %.so   $(ROOT)/lib $(ROOT)/extlib $(LIBSUBD) $(ELIBSUBD)
vpath %.so.% $(ROOT)/lib $(ROOT)/extlib $(LIBSUBD) $(ELIBSUBD)
vpath %.lib  $(ROOT)/lib $(ROOT)/extlib $(LIBSUBD) $(ELIBSUBD)
