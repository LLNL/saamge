/*! \file
    \brief Some helping functions for more general tasks.

    SAAMGE: smoothed aggregation element based algebraic multigrid hierarchies
            and solvers.

    Copyright (c) 2015, Lawrence Livermore National Security,
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

#pragma once
#ifndef _HELPERS_HPP
#define _HELPERS_HPP

#include "common.hpp"
#include <mfem.hpp>
#include "aggregates.hpp"

/* Functions */
/*! \brief Randomly perturbs the values in \a x using uniform distribution.

    \a x(i) = | \a x(i) + \a interval_len * X |, where X is a draw from a
    uniform distributed variable in [-0.5, 0.5).

    \param x (IN/OUT) The vector to perturb.
    \param interval_len (IN) The length of the interval for the perturbation.
*/
void helpers_perturb(Vector &x, double interval_len);

/*! \brief Randomly fills in \a x using uniform distribution.

    \a x(i) = \a interval_len * X, where X is a draw from a uniform
    distributed variable in [0.0, 1.0).

    \param x (OUT) The vector to fill in.
    \param interval_len (IN) The length of the interval for the perturbation.

    \warning \a x must be constructed prior to calling this function.
*/
void helpers_random_gen(Vector &x, double interval_len);

/*! \brief Randomly fills in a vector using uniform distribution.

    The DoFs in the vector are set to 0 if they lie on the part of the boundary
    where essential boundary conditions are posed.

    \param agg_part_rels (IN) The partitioning relations.
    \param x (OUT) The vector to fill in.

    \warning \a x must be constructed prior to calling this function.
*/
void helpers_random_vect(const agg_partititoning_relations_t& agg_part_rels,
                         Vector &x);

/*! \brief Creates a copy of an array of doubles.

    \param src (IN) The array to be copied.
    \param n (IN) The number of doubles in the array.

    \returns The copy. If \a src is NULL, it returns NULL.

    \warning The returned array must be freed by the caller.
*/
double *helpers_copy_dbl_arr(const double *src, int n);

/*! \brief Creates a copy of an array of integers.

    \param src (IN) The array to be copied.
    \param n (IN) The number of integers in the array.

    \returns The copy. If \a src is NULL, it returns NULL.

    \warning The returned array must be freed by the caller.
*/
int *helpers_copy_int_arr(const int *src, int n);

/*! \brief Creates a copy of an array of chars.

    \param src (IN) The array to be copied.
    \param n (IN) The number of chars in the array.

    \returns The copy. If \a src is NULL, it returns NULL.

    \warning The returned array must be freed by the caller.
*/
char *helpers_copy_char_arr(const char *src, int n);

/*! \brief Writes a text file with data to be easily plotted using gnuplot.

    Writes a file with name "eigenvalues_<\a num>.out.txt" that has two
    columns. The first column is an index starting from 1 and the second column
    is filled with values such that on line i the value is \a v(i) / \a v(i-1).

    \param num (IN) A number for the name of the file.
    \param v (IN) The vector to be used for the data output.

    \warning The precision of the output is determined by the \b prec option.
*/
void helpers_write_vector_for_gnuplot(int num, const Vector& v);

/*! \brief Loads an array of doubles from a file.

    \a filename is a binary file with format:
    <number of elements><the array data>.

    \param filename (IN) The name of the file with the array.
    \param n (OUT) The number of doubles in the array.

    \returns The loaded array.

    \warning The returned array must be freed by the caller.
*/
double *helpers_read_dbl_arr(const char *filename, int *n);

/*! \brief Writes an array of doubles to a file.

    \a filename is a binary file with format:
    <number of elements><the array data>.

    \param filename (IN) The name of the file to write. If it exists, it will
                         be erased prior to writing.
    \param arr (IN) The array to be written.
    \param n (IN) The number of doubles in the array.
*/
void helpers_write_dbl_arr(const char *filename, const double *arr, int n);

/*! \brief Loads an array of integers from a file.

    \a filename is a binary file with format:
    <number of elements><the array data>.

    \param filename (IN) The name of the file with the array.
    \param n (OUT) The number of integers in the array.

    \returns The loaded array.

    \warning The returned array must be freed by the caller.
*/
int *helpers_read_int_arr(const char *filename, int *n);

/*! \brief Writes an array of integers to a file.

    \a filename is a binary file with format:
    <number of elements><the array data>.

    \param filename (IN) The name of the file to write. If it exists, it will
                         be erased prior to writing.
    \param arr (IN) The array to be written.
    \param n (IN) The number of integers in the array.
*/
void helpers_write_int_arr(const char *filename, const int *arr, int n);

#endif // _HELPERS_HPP
