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

ROOT = .

include $(ROOT)/mk/common.mk

SRCS = test

SRCS_TRIMMED = $(subst /,___,$(SRCS))
CLEANIN = $(addprefix clean,$(SRCS_TRIMMED))
CLEANLIB = $(addprefix clean,$(BLDLIB))

.PHONY: clean all libs $(SRCS_TRIMMED) $(CLEANIN) $(BLDLIB)

libs: $(BLDLIB)

all: libs $(SRCS_TRIMMED)

$(SRCS_TRIMMED):
	($(MAKE) -SC $(ROOT)/$(subst ___,/,$@)) || exit $$?

$(BLDLIB):
	($(MAKE) -SC $(ROOT)/lib/$@) || exit $$?

$(CLEANIN):
	$(MAKE) -C $(ROOT)/$(subst ___,/,$(patsubst clean%,%,$@)) clean

$(CLEANLIB):
	$(MAKE) -C $(ROOT)/lib/$(subst ___,/,$(patsubst clean%,%,$@)) clean

clean: $(CLEANIN) $(CLEANLIB)
	rm -f *.*out*.txt
