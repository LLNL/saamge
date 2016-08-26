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

ROOT = ..

include $(ROOT)/mk/common.mk
include $(ROOT)/mk/common_srcs.mk

SUBD_TRIMMED = $(patsubst ./%,%,$(SUBD))

CLEAN = $(addprefix clean,$(SUBD_TRIMMED))

.PHONY: clean $(SUBD_TRIMMED)

all: $(SUBD_TRIMMED)

$(SUBD_TRIMMED):
	($(MAKE) -SC $@ all) || exit $$?

$(CLEAN):
	$(MAKE) -C $(patsubst clean%,%,$@) clean

clean: $(CLEAN)
