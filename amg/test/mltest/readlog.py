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
make some reports based on a logfile
these logfiles generally come from job.py
"""
from __future__ import print_function

import datetime
import getpass
import os
import platform
import subprocess
import sys
import time

import maketable
import mlparse

def get_git_revision():
    p = subprocess.Popen(['git', 'rev-parse', '--short', 'HEAD'],stdout=subprocess.PIPE)
    return p.communicate()[0].strip()

def readlog(filename):
    fd = open(filename,"r")
    results = []
    for line in fd:
        outputfilename = line.strip()
        if len(outputfilename) > 0 and not outputfilename[0] == "#":
            r = {"outputfile":outputfilename}
            try:
                fdi = open(outputfilename,"r")
                r.update(mlparse.ml_parse_fd(fdi))
                fdi.close()
            except IOError:
                pass
            results.append(r.copy())
    fd.close()
    return results

def report_header(logfile):
    out = "% this report produced on {2} is based on logfile {0} modified {1}\n".format(
        logfile, time.strftime("%d %B %Y at %H:%M",time.localtime(os.path.getmtime(logfile))),
        datetime.datetime.today().strftime("%d %B %Y at %H:%M"))
    out = out + "% on machine {0} by user {1} with git at or near revision {2}.\n".format(
        platform.node(),getpass.getuser(),get_git_revision())
    return out

def technical(logfile):
    results = readlog(logfile)
    out = report_header(logfile)
    for result in results:
        out = out + result["outputfile"].split("/")[-1]
        try:
            out = out + "  " + str(result["processes"])
            # out = out + "  " + str(result["complexity"])
            if "ASSERT-file" in result:
                out = out + "  " + result["ASSERT-file"] + " " + str(result["ASSERT-line"])
            elif "WHAT-error" in result:
                out = out + "  " + result["WHAT-error"]
            elif "TIME-LIMIT" in result:
                out = out + "  " + result["TIME-LIMIT"]
            out = out + "  " + str(result["iterations"])
        except KeyError:
            pass
        out = out + "\n"
    print(out)

def main(logfile):
    """
    just ignore missing files and files with no results
    """
    out = report_header(logfile)
    results = readlog(logfile)
    keys = ["serial-refine","refine","processes","num-levels","complexity","iterations","time-spectral-setup","time-spectral-solve"]
    formats = ["d","d","d","d",".2f","d",".2f",".2f"]
    out = out + maketable.maketable(results, keys, formats)
    print(out)

def spe10(logfile):
    out = report_header(logfile)
    results = readlog(logfile)
    keys = ["processes","num-levels",
            "first-theta","theta",
            "first-elems-per-agg","elems-per-agg",
#            "standard-pcg-iterations",
            "complexity","iterations","time-spectral-setup","time-spectral-solve"]
    formats = ["d","d",
               ".0e",".0e",
               "d", "d",
#               "d",
               ".2f","d",".2f",".2f"]
    # out = out + maketable.maketable(results, keys, formats, require={"w-cycle":True})
    # out = out + maketable.maketable(results, keys, formats, require={"w-cycle":False})
    out = out + maketable.maketable(results, keys, formats)
    print(out)

def pro(logfile):
    out = report_header(logfile)
    results = readlog(logfile)
    keys = ["refine",
            "processes","num-levels",
            "first-elems-per-agg","elems-per-agg",
            "first-nu-pro","nu-pro",
            "complexity","iterations","time-spectral-setup","time-spectral-solve"]
    formats = ["d",
               "d","d",
               "d", "d",
               "d", "d",
               ".2f","d",".2f",".2f"]
    out = out + maketable.maketable(results, keys, formats)
    print(out)
    
if __name__ == "__main__":
    technical_opt = False
    spe10_opt = False
    pro_opt = False
    for v in sys.argv:
        if v == "--technical":
            technical_opt = True
        elif v == "--spe10":
            spe10_opt = True
        elif v == "--pro":
            pro_opt = True
        else:
            logfile = v
    if technical_opt:
        technical(logfile)
    elif spe10_opt:
        spe10(logfile)
    elif pro_opt:
        pro(logfile)
    else:
        main(logfile)
