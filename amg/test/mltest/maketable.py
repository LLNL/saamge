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

from __future__ import print_function

import operator

def maketable(dicts, keys, formats, names=None, require={}):
    """
    dicts: list of dictionaries with data
    keys: list of keys, these become columns
    names: headers for top of table, if None then we just use keys
    require: everything in this table must match key:values in require
    """            
    enders = ["& "] * (len(keys) - 1) + [" \\\\ "]
    intoformatstring = "  "
    headerformatstring = "  "
    for i in range(len(keys)):
        intoformatstring = intoformatstring + "{" + str(i) + ":" + formats[i] + "}" + enders[i]
        headerformatstring = headerformatstring + "{" + str(i) + "}" + enders[i]

    filtered_dicts = []
    out = ""
    for d in dicts:
        try:
            out = out + intoformatstring.format(*[d[key] for key in keys])
            meets_requirements = True
            for k in require.keys():
                if not d[k] == require[k]: meets_requirements = False
            if meets_requirements:
                filtered_dicts.append(d)
        except KeyError:
            # for key in keys: print(d[key])
            pass            
    filtered_dicts.sort(key=operator.itemgetter(*keys))

    alignmentstring = ("l" * (len(keys) - 1)) + "r"
    out = "\\begin{center}\\begin{tabular}{" + alignmentstring + "}\n"
    out = out + "\\toprule \n"
    if names == None:
        out = out + headerformatstring.format(*keys) + "\n"
    else:
        out = out + headerformatstring.format(*names) + "\n"
    out = out + "\\midrule \n"
    for d in filtered_dicts:
        try:
            fragment = intoformatstring.format(*[d[key] for key in keys])
            if not ("clean" in d and d["clean"]):
                fragment = "*" + fragment
            fragment = fragment + "  % " + d["outputfile"].split("/")[-1] + "\n"
            out = out + fragment
        except KeyError:
            pass
    out = out + "\\bottomrule \n"
    out = out + "\\end{tabular}\\end{center}\n"
    return out

if __name__ == "__main__":
    print("maketable.py is just a module for importing.")
