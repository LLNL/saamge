/*! \file
    \brief Global configuration options.

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
#ifndef _CONFIG_HPP
#define _CONFIG_HPP

#include "config_mgr.hpp"
#include <cstdio>
using std::FILE;

/* Options */

/*! \brief The global configuration class.
*/
CONFIG_BEGIN_CLASS_DECLARATION(GLOBAL)

    /*! Tolerance used for defining real numbers proximity when, e.g., real
        numbers are compared. */
    CONFIG_DECLARE_OPTION(double, diff_eps);

    /*! The output stream for messages from ASSERTS and ALERTS. */
    CONFIG_DECLARE_OPTION(FILE *, asrts_output_stream);

    /*! The double precision for some output functions. */
    CONFIG_DECLARE_OPTION(unsigned int, prec);

    /*! The output level determining what messages will be displayed.

        0 -- NO output (due to the debug level output may still be generated).
        1-5 -- Normal output levels.
        6-10 -- High output levels.
        11-15 -- Very high output levels. */
    CONFIG_DECLARE_OPTION(unsigned int, output_level);

CONFIG_END_CLASS_DECLARATION(GLOBAL)

CONFIG_BEGIN_INLINE_CLASS_DEFAULTS(GLOBAL)
    CONFIG_DEFINE_OPTION_DEFAULT(diff_eps, 1e-10),
    CONFIG_DEFINE_OPTION_DEFAULT(asrts_output_stream, stdout),
    CONFIG_DEFINE_OPTION_DEFAULT(prec, 16),
    CONFIG_DEFINE_OPTION_DEFAULT(output_level, 5)
CONFIG_END_CLASS_DEFAULTS

/* Defines */
/*! \def SA_DEBUG_LEVEL
    \brief The level determining how much debug code will be compiled and run.

    0 -- NO debug.
    1-5 -- Normal debug levels.
    6-10 -- High debug levels.
    11-15 -- Very high debug levels.

    Level 1 simply turns on the asserts. Only levels greater and equal than 4
    are expected to generate explicit output.
*/
/*! \def SA_TIMESTAMPS
    \brief Turns on timestamps in the output.
*/
/*! \def SA_TIMERS
    \brief Turns timers on.
*/
#define SA_DEBUG_LEVEL 5
#undef SA_TIMESTAMPS
#undef SA_TIMERS

/*! \brief Whether the current debug level is smaller or equal to \a level.

    Use in '#if' preprocessor statements.

    \param level (IN) Desired level.

    \returns Whether the current debug level is smaller or equal to \a level.
*/
#define SA_IS_DEBUG_LEVEL(level) ((level) <= SA_DEBUG_LEVEL)

/*! \brief Whether the current output level is smaller or equal to \a level.

    Use in 'if' statements.

    \param level (IN) Desired level.

    \returns Whether the current output level is smaller or equal to \a level.

    \warning Uses the option \b output_level to determine the current output
             level.
*/
#define SA_IS_OUTPUT_LEVEL(level) \
    ((level) <= CONFIG_ACCESS_OPTION(GLOBAL, output_level))

#endif // _CONFIG_HPP
