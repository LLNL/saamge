/*! \file

    SAAMGE: smoothed aggregation element based algebraic multigrid hierarchies
            and solvers.

    Copyright (c) 2016, Lawrence Livermore National Security,
    LLC. Developed under the auspices of the U.S. Department of Energy by
    Lawrence Livermore National Laboratory under Contract
    No. DE-AC52-07NA27344. Written by Delyan Kalchev, Andrew T. Barker,
    and Panayot S. Vassilevski. Released under LLNL-CODE-667453.

    This file is part of SAAMGE. 

    Please also read the full notice of copyright and license in the file
    LICENSE.

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License (as
    published by the Free Software Foundation) version 2.1 dated February
    1999.

    This program is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the IMPLIED WARRANTY OF
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the terms and
    conditions of the GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with this program; if not, see
    <http://www.gnu.org/licenses/>.
*/

#include "common.hpp"
#include "helpers.hpp"
#include <fstream>
#include <mfem.hpp>
#include "tg.hpp"
using std::ofstream;
using std::ifstream;
using std::snprintf;

/* Functions */

void helpers_perturb(Vector &x, double interval_len)
{
    srand48(time(NULL));
    for (int i=0; i < x.Size(); ++i)
        x(i) = fabs(x(i) + interval_len * (drand48() - 0.5));
}

void helpers_random_gen(Vector &x, double interval_len)
{
    srand48(time(NULL));
    for (int i=0; i < x.Size(); ++i)
        x(i) = interval_len * drand48();
}

void helpers_random_vect(const agg_partitioning_relations_t& agg_part_rels,
                         Vector &x)
{
    srand48(time(NULL) + PROC_RANK);
    SA_ASSERT(x.Size());
    for (int i=0; i < x.Size(); ++i)
    {
        x(i) = agg_is_dof_on_essential_border(agg_part_rels, i) ? 0. :
                                                                  drand48();
    }
}

double *helpers_copy_dbl_arr(const double *src, int n)
{
    if (!src) return NULL;
    double *dst = new double[n];
    memcpy(dst, src, sizeof(*dst)*n);
    return dst;
}

int *helpers_copy_int_arr(const int *src, int n)
{
    if (!src) return NULL;
    int *dst = new int[n];
    memcpy(dst, src, sizeof(*dst)*n);
    return dst;
}

char *helpers_copy_char_arr(const char *src, int n)
{
    if (!src) return NULL;
    char *dst = new char[n];
    memcpy(dst, src, sizeof(*dst)*n);
    return dst;
}

void helpers_write_vector_for_gnuplot(int num, const Vector& v)
{
    const int namelen = 256;
    char name[namelen];
    snprintf(name, namelen, "eigenvalues_%d.out.txt", num);
    ofstream vect_ofs(name);
    SA_ASSERT(vect_ofs);
    vect_ofs.precision(CONFIG_ACCESS_OPTION(GLOBAL, prec));
    for (int i=0; i < v.Size(); ++i)
        vect_ofs << i << "\t\t" << v(i) << std::endl;
    vect_ofs.close();
}

double *helpers_read_dbl_arr(const char *filename, int *n)
{
    ifstream iarr(filename, ifstream::binary);
    SA_ASSERT(iarr);
    iarr.read((char *)n, sizeof(*n));
    double *arr = new double[*n];
    iarr.read((char *)arr, sizeof(*arr) * (*n));
    iarr.close();
    return arr;
}

void helpers_write_dbl_arr(const char *filename, const double *arr, int n)
{
    ofstream oarr(filename, ofstream::binary);
    SA_ASSERT(oarr);
    oarr.write((char *)&n, sizeof(n));
    oarr.write((char *)arr, sizeof(*arr)*n);
    oarr.close();
}

int *helpers_read_int_arr(const char *filename, int *n)
{
    ifstream iarr(filename, ifstream::binary);
    SA_ASSERT(iarr);
    iarr.read((char *)n, sizeof(*n));
    int *arr = new int[*n];
    iarr.read((char *)arr, sizeof(*arr) * (*n));
    iarr.close();
    return arr;
}

void helpers_write_int_arr(const char *filename, const int *arr, int n)
{
    ofstream oarr(filename, ofstream::binary);
    SA_ASSERT(oarr);
    oarr.write((char *)&n, sizeof(n));
    oarr.write((char *)arr, sizeof(*arr)*n);
    oarr.close();
}
