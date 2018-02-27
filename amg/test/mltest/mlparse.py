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

"""
Read output from multilevel tests ./mltest
"""
from __future__ import print_function

import sys

def linehas(p,num,val):
    if len(p) > num and p[num] == val:
        return True
    else:
        return False

def valgrindparse(lines):
    out = {}
    for line in lines:
        p = line.split()
        if len(p) > 5 and p[1] == "definitely" and p[2] == "lost:":
            out["definitely-lost"] = int(p[3].replace(',',''))
        if len(p) > 5 and p[1] == "indirectly" and p[2] == "lost:":
            out["indirectly-lost"] = int(p[3].replace(',',''))
        if len(p) > 5 and p[1] == "possibly" and p[2] == "lost:":
            out["possibly-lost"] = int(p[3].replace(',',''))
        if len(p) > 5 and p[1] == "still" and p[2] == "reachable:":
            out["still-reachable"] = int(p[3].replace(',',''))
        if len(p) > 10 and p[1] == "ERROR" and p[2] == "SUMMARY:":
            out["total-errors"] = int(p[3])
        if len(p) > 5 and p[1] == "All" and p[2] == "heap" and p[5] == "freed":
            out["definitely-lost"] = 0
            out["indirectly-lost"] = 0
            out["possibly-lost"] = 0
            out["still-reachable"] = 0
    return out

def ml_parse_fd(fd):
    """
    TODO: error/assert parsing/handling is a bit hacky, wonder
          if that should be a separate routine
    """
    out = {"clean":True}
    out["serial-refine"] = 0
    for line in fd:
        p = line.split()
        if linehas(p,2,"Number") and linehas(p,4,"processes:"):
            out["processes"] = int(p[5])
        if linehas(p,0,"elements"):
            out["elements"] = int(p[4])
        if linehas(p,2,"global") and linehas(p,3,"nparts"):
            out["nparts"] = int(p[5])
        if linehas(p,0,"--num-levels"):
            out["num-levels"] = int(p[1])
        if linehas(p,0,"--serial-refine"):
            out["serial-refine"] = int(p[1])
        if linehas(p,0,"--refine"):
            out["refine"] = int(p[1])
        if linehas(p,0,"--first-elems-per-agg"):
            out["first-elems-per-agg"] = int(p[1])
        if linehas(p,0,"--elems-per-agg"):
            out["elems-per-agg"] = int(p[1])
        if linehas(p,0,"--first-nu-pro"):
            out["first-nu-pro"] = int(p[1])
        if linehas(p,0,"--nu-pro"):
            out["nu-pro"] = int(p[1])
        if linehas(p,0,"--first-theta"):
            out["first-theta"] = float(p[1])
        if linehas(p,0,"--theta"):
            out["theta"] = float(p[1])
        if linehas(p,0,"--w-cycle"):
            out["w-cycle"] = True
        if linehas(p,0,"--no-w-cycle"):
            out["w-cycle"] = False
        if linehas(p,0,"PCG") and linehas(p,1,"Iterations"):
            out["standard-pcg-iterations"] = int(p[3])
        if linehas(p,2,"Stationary") and linehas(p,3,"iteration:"):
            out["solver"] = "stationary"
            try:
                out["iterations"] = int(p[5])
                out["cf"] = float(p[13])
            except ValueError:
                pass
        if linehas(p,2,"PCG") and linehas(p,3,"Iteration:"):
            out["solver"] = "pcg"
            out["iterations"] = int(p[4][:-1])
        if linehas(p,0,"Iteration") and linehas(p,1,":"):
            out["solver"] = "maybe-pcg"
            out["iterations"] = int(p[2])
        if linehas(p,2,"Average") and linehas(p,3,"reduction"):
            out["average-reduction"] = float(p[6])
        if linehas(p,0,"Average") and linehas(p,1,"reduction"):
            out["average-reduction"] = float(p[4])
        if linehas(p,2,"Level") and linehas(p,4,"dimension:"):
            if int(p[3]) == 0:
                out["fine-dofs"] = int(p[5][:-1])
                out["fine-nnzs"] = int(p[8])
                out["dofs"] = [out["fine-dofs"]]
                out["nnzs"] = [out["fine-nnzs"]]
            else:
                out["dofs"].append(int(p[5][:-1]))
                out["nnzs"].append(int(p[8]))
        if linehas(p,2,"TIMING:"):
            if linehas(p,3,"fem") and linehas(p,4,"setup"):
                out["time-fem-setup"] = float(p[-2])
            elif linehas(p,3,"setup") and linehas(p,8,"BoomerAMG"):
                out["time-hypre-solve"] = float(p[-2])
            elif linehas(p,3,"multilevel") and linehas(p,5,"SA-AMGe"):
                out["time-spectral-setup"] = float(p[-2])
            elif linehas(p,3,"solve") and linehas(p,7,"CG"):
                out["time-spectral-solve"] = float(p[-2])
        if linehas(p,2,"Ac") and linehas(p,8,"OC:"):
            # this one is for two-level, should get overwritten by next in multilevel setting
            out["complexity"] = float(p[9])
        if linehas(p,2,"Overall") and linehas(p,3,"operator") and linehas(p,4,"complexity:"):
            # TODO: for partialsmooth, this (and many others) happen more than once...
            out["complexity"] = float(p[5])
        if linehas(p,2,"Cumulative") and linehas(p,3,"hypre-PCG"):
            out["cumulative-coarse-pcg"] = int(p[8]) # this is on the nulspace level, in our current terminology
        if linehas(p,2,"Cumulative") and linehas(p,6,"spectral"):
            out["cumulative-spectral-pcg"] = int(p[8])
        if linehas(p,2,"Outer") and linehas(p,3,"PCG"):
            if linehas(p,4,"converged"):
                out["converged"] = True
            else:
                out["converged"] = False
                out["clean"] = False
        if linehas(p,2,"ASSERT:"):
            out["ASSERT-file"] = p[3][:-1]
            out["ASSERT-line"] = int(p[4][:-1])
            out["clean"] = False
        if linehas(p,0,"what():"):
            out["WHAT-error"] = p[1]
            out["clean"] = False
        if linehas(p,4,"CANCELLED") and linehas(p,9,"TIME"):
            out["TIME-LIMIT"] = "TIME LIMIT"
            out["clean"] = False
    return out

if __name__ == "__main__":
    fd = open(sys.argv[1])
    result = ml_parse_fd(fd)
    print(result)
