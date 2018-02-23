/*! \file
    \brief Common stuff being used in the whole library.

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

#pragma once
#ifndef _COMMON_HPP
#define _COMMON_HPP

#include "config.hpp"
#include "process.hpp"
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <sys/time.h>
#include <iostream>
using std::cin;
using std::cout;
using std::fprintf;
using std::fflush;
using std::system;
using std::srand;
using std::rand;
using std::memset;
using std::memcpy;
using std::time;
using std::strftime;
using std::localtime;
using std::time_t;
using std::tm;
using std::size_t;

/* Defines */

#if (SA_IS_DEBUG_LEVEL(1))
#define SA_ASSERTS
#else
#define NDEBUG
#endif
#include <cassert>

/*! \brief Prints to \a outstream the process rank.

    \param outstream (IN) Output stream.
*/
#define SA_PRINT_RANK(outstream) \
    SA_FPRINTF_NOTS((outstream), "CPU %d: ", PROC_RANK)

/*! \brief Prints to \a outstream the process rank, depending on output level.

    \param level (IN) Desired level.
    \param outstream (IN) Output stream.
*/
#define SA_PRINT_RANK_L(level, outstream) \
     do { \
        if (SA_IS_OUTPUT_LEVEL(level)) \
            SA_PRINT_RANK(outstream); \
    } while(0)

/*! \brief Prints to \a outstream without time stamp.

    The variable arguments are the same as the standard \b printf.

    \param outstream (IN) Output stream.
*/
#define SA_FPRINTF_NOTS(outstream, ...) \
    do { \
        fprintf((outstream), __VA_ARGS__); \
/*        fflush(outstream); */ \
    } while(0)

/*! \brief Prints to \a outstream without time stamp.

    The variable arguments are the same as the standard \b printf.

    \param rank (IN) Outputs only in case the rank of the process equals this.
    \param outstream (IN) Output stream.
*/
#define SA_RFPRINTF_NOTS(rank, outstream, ...) \
    do { \
        if (PROC_RANK == (rank)) \
            SA_FPRINTF_NOTS((outstream), __VA_ARGS__); \
    } while(0)

/*! \brief Prints to \a outstream without time stamp, depending on output level.

    The variable arguments are the same as the standard \b printf.

    \param level (IN) Desired level.
    \param outstream (IN) Output stream.
*/
#define SA_FPRINTF_NOTS_L(level, outstream, ...) \
    do { \
        if (SA_IS_OUTPUT_LEVEL(level)) \
            SA_FPRINTF_NOTS((outstream), __VA_ARGS__); \
    } while(0)

/*! \brief Prints to \a outstream without time stamp, depending on output level.

    The variable arguments are the same as the standard \b printf.

    \param rank (IN) Outputs only in case the rank of the process equals this.
    \param level (IN) Desired level.
    \param outstream (IN) Output stream.
*/
#define SA_RFPRINTF_NOTS_L(rank, level, outstream, ...) \
    do { \
        if (PROC_RANK == (rank)) \
            SA_FPRINTF_NOTS_L((level), (outstream), __VA_ARGS__); \
    } while(0)

/*! \brief Prints to \em stdout without time stamp.

    The variable arguments are the same as the standard \b printf.
*/
#define SA_PRINTF_NOTS(...) \
    SA_FPRINTF_NOTS(stdout, __VA_ARGS__)

/*! \brief Prints to \em stdout without time stamp.

    The variable arguments are the same as the standard \b printf.

    \param rank (IN) Outputs only in case the rank of the process equals this.
*/
#define SA_RPRINTF_NOTS(rank, ...) \
    do { \
        if (PROC_RANK == (rank)) \
            SA_FPRINTF_NOTS(__VA_ARGS__); \
    } while(0)

/*! \brief Prints to \em stdout without time stamp, depending on output level.

    The variable arguments are the same as the standard \b printf.

    \param level (IN) Desired level.
*/
#define SA_PRINTF_NOTS_L(level, ...) \
    do { \
        if (SA_IS_OUTPUT_LEVEL(level)) \
            SA_PRINTF_NOTS(__VA_ARGS__); \
    } while(0)

/*! \brief Prints to \em stdout without time stamp, depending on output level.

    The variable arguments are the same as the standard \b printf.

    \param rank (IN) Outputs only in case the rank of the process equals this.
    \param level (IN) Desired level.
*/
#define SA_RPRINTF_NOTS_L(rank, level, ...) \
    do { \
        if (PROC_RANK == (rank)) \
            SA_PRINTF_NOTS_L((level), __VA_ARGS__); \
    } while(0)

/*! \brief Prints to \a outstream with time stamp.

    The \a str plus the variable arguments are the same as the standard
    \b printf.

    \param outstream (IN) Output stream.
    \param str (IN) The format string.
*/
#define SA_FPRINTF_TS(outstream, str, ...) \
    do { \
        const size_t len = 20; \
        char timestr[len]; \
        time_t curtime; \
        struct tm loctime; \
        time(&curtime); \
        localtime_r(&curtime, &loctime); \
        strftime(timestr, len, "%F,%T", &loctime); \
        SA_FPRINTF_NOTS((outstream), "|%s(%lu)|    " str, timestr, \
                        (unsigned long)curtime, __VA_ARGS__); \
    } while(0)

/*! \brief Prints to \a outstream with time stamp.

    The variable arguments are the same as the standard \b printf.

    \param rank (IN) Outputs only in case the rank of the process equals this.
    \param outstream (IN) Output stream.
*/
#define SA_RFPRINTF_TS(rank, outstream, ...) \
    do { \
        if (PROC_RANK == (rank)) \
            SA_FPRINTF_TS((outstream), __VA_ARGS__); \
    } while(0)

/*! \brief Prints to \a outstream with time stamp, depending on output level.

    The variable arguments are the same as the standard \b printf.

    \param level (IN) Desired level.
    \param outstream (IN) Output stream.
*/
#define SA_FPRINTF_TS_L(level, outstream, ...) \
    do { \
        if (SA_IS_OUTPUT_LEVEL(level)) \
            SA_FPRINTF_TS((outstream), __VA_ARGS__); \
    } while(0)

/*! \brief Prints to \a outstream with time stamp, depending on output level.

    The variable arguments are the same as the standard \b printf.

    \param rank (IN) Outputs only in case the rank of the process equals this.
    \param level (IN) Desired level.
    \param outstream (IN) Output stream.
*/
#define SA_RFPRINTF_TS_L(rank, level, outstream, ...) \
    do { \
        if (PROC_RANK == (rank)) \
            SA_FPRINTF_TS_L((level), (outstream), __VA_ARGS__); \
    } while(0)

/*! \brief Prints to \em stdout with time stamp.

    The variable arguments are the same as the standard \b printf.
*/
#define SA_PRINTF_TS(...) \
    SA_FPRINTF_TS(stdout, __VA_ARGS__)

/*! \brief Prints to \em stdout with time stamp.

    The variable arguments are the same as the standard \b printf.

    \param rank (IN) Outputs only in case the rank of the process equals this.
*/
#define SA_RPRINTF_TS(rank, ...) \
    do { \
        if (PROC_RANK == (rank)) \
            SA_PRINTF_TS(__VA_ARGS__); \
    } while(0)

/*! \brief Prints to \em stdout with time stamp, depending on output level.

    The variable arguments are the same as the standard \b printf.

    \param level (IN) Desired level.
*/
#define SA_PRINTF_TS_L(level, ...) \
    do { \
        if (SA_IS_OUTPUT_LEVEL(level)) \
            SA_PRINTF_TS(__VA_ARGS__); \
    } while(0)

/*! \brief Prints to \em stdout with time stamp, depending on output level.

    The variable arguments are the same as the standard \b printf.

    \param rank (IN) Outputs only in case the rank of the process equals this.
    \param level (IN) Desired level.
*/
#define SA_RPRINTF_TS_L(rank, level, ...) \
    do { \
        if (PROC_RANK == (rank)) \
            SA_PRINTF_TS_L((level), __VA_ARGS__); \
    } while(0)

/*! \brief Prints an alert message.

    The variable arguments are the same as the standard \b printf.

    It puts new line symbol(s) in the end.

    \warning Uses the \em GLOBAL option \b asrts_output_stream.
*/
#define SA_ALERT_PRINTF(str, ...) \
    SA_FPRINTF(CONFIG_ACCESS_OPTION(GLOBAL, asrts_output_stream), \
               "ALERT: " __FILE__ ", " SA_TOSTRING(__LINE__) ": " str "\n", \
               __VA_ARGS__)

/*! \brief Prints an alert message.

    The variable arguments are the same as the standard \b printf.

    It puts new line symbol(s) in the end.

    \param rank (IN) Outputs only in case the rank of the process equals this.

    \warning Uses the \em GLOBAL option \b asrts_output_stream.
*/
#define SA_ALERT_RPRINTF(rank, ...) \
    do { \
        if (PROC_RANK == (rank)) \
            SA_ALERT_PRINTF(__VA_ARGS__); \
    } while(0)

/*! \def SA_FPRINTF
    \brief Prints to \a outstream.

    With or without time stamp depending on \b SA_TIMESTAMPS.

    The \a str plus the variable arguments are the same as the standard
    \b printf.

    \param outstream (IN) Output stream.
    \param str (IN) The format string.
*/
/*! \def SA_FPRINTF_L
    \brief Prints to \a outstream, depending on output level.

    With or without time stamp depending on \b SA_TIMESTAMPS.

    The \a str plus the variable arguments are the same as the standard
    \b printf.

    \param level (IN) Desired level.
    \param outstream (IN) Output stream.
    \param str (IN) The format string.
*/
/*! \def SA_PRINTF
    \brief Prints to \em stdout.

    With or without time stamp depending on \b SA_TIMESTAMPS.

    The \a str plus the variable arguments are the same as the standard
    \b printf.

    \param str (IN) The format string.
*/
/*! \def SA_PRINTF_L
    \brief Prints to \em stdout, depending on output level.

    With or without time stamp depending on \b SA_TIMESTAMPS.

    The \a str plus the variable arguments are the same as the standard
    \b printf.

    \param level (IN) Desired level.
    \param str (IN) The format string.
*/
#ifdef SA_TIMESTAMPS

#define SA_FPRINTF(outstream, str, ...) \
    SA_FPRINTF_TS((outstream), "CPU %d: " str, PROC_RANK, __VA_ARGS__)

#define SA_FPRINTF_L(level, outstream, str, ...) \
    SA_FPRINTF_TS_L((level), (outstream), "CPU %d: " str, PROC_RANK, \
                    __VA_ARGS__)

#define SA_PRINTF(str, ...) \
    SA_PRINTF_TS("CPU %d: " str, PROC_RANK, __VA_ARGS__)

#define SA_PRINTF_L(level, str, ...) \
    SA_PRINTF_TS_L((level), "CPU %d: " str, PROC_RANK, __VA_ARGS__)

#else

#define SA_FPRINTF(outstream, str, ...) \
    SA_FPRINTF_NOTS((outstream), "CPU %d: " str, PROC_RANK, __VA_ARGS__)

#define SA_FPRINTF_L(level, outstream, str, ...) \
    SA_FPRINTF_NOTS_L((level), (outstream), "CPU %d: " str, PROC_RANK, \
                      __VA_ARGS__)

#define SA_PRINTF(str, ...) \
    SA_PRINTF_NOTS("CPU %d: " str, PROC_RANK, __VA_ARGS__)

#define SA_PRINTF_L(level, str, ...) \
    SA_PRINTF_NOTS_L((level), "CPU %d: " str, PROC_RANK, __VA_ARGS__)

#endif

/*! \brief Prints to \a outstream.

    With or without time stamp depending on \b SA_TIMESTAMPS.

    The variable arguments are the same as the standard \b printf.

    \param rank (IN) Outputs only in case the rank of the process equals this.
    \param outstream (IN) Output stream.
*/
#define SA_RFPRINTF(rank, outstream, ...) \
    do { \
        if (PROC_RANK == (rank)) \
            SA_FPRINTF((outstream), __VA_ARGS__); \
    } while(0)

/*! \brief Prints to \a outstream, depending on output level.

    With or without time stamp depending on \b SA_TIMESTAMPS.

    The variable arguments are the same as the standard \b printf.

    \param rank (IN) Outputs only in case the rank of the process equals this.
    \param level (IN) Desired level.
    \param outstream (IN) Output stream.
*/
#define SA_RFPRINTF_L(rank, level, outstream, ...) \
    do { \
        if (PROC_RANK == (rank)) \
            SA_FPRINTF_L((level), (outstream), __VA_ARGS__); \
    } while(0)

/*! \brief Prints to \em stdout.

    With or without time stamp depending on \b SA_TIMESTAMPS.

    \param rank (IN) Outputs only in case the rank of the process equals this.

    The variable arguments are the same as the standard \b printf.
*/
#define SA_RPRINTF(rank, ...) \
    do { \
        if (PROC_RANK == (rank)) \
            SA_PRINTF(__VA_ARGS__); \
    } while(0)

/*! \brief Prints to \em stdout, depending on output level.

    With or without time stamp depending on \b SA_TIMESTAMPS.

    The variable arguments are the same as the standard \b printf.

    \param rank (IN) Outputs only in case the rank of the process equals this.
    \param level (IN) Desired level.
*/
#define SA_RPRINTF_L(rank, level, ...) \
    do { \
        if (PROC_RANK == (rank)) \
            SA_PRINTF_L((level), __VA_ARGS__); \
    } while(0)

/*! \brief Compares \a x and \a y using tolerance \a eps.

    \param x (IN) Real number.
    \param y (IN) Real number.
    \param eps (IN) Tolerance.
*/
#define SA_IS_REAL_EQ_RAW(x, y, eps) ((x)-(y) <= (eps) && (y)-(x) <= (eps))

/*! \brief Compares \a x and \a y using global tolerance.

    \param x (IN) Real number.
    \param y (IN) Real number.

    \warning Uses the \em GLOBAL option \b diff_eps.
*/
#define SA_IS_REAL_EQ(x, y) \
    SA_IS_REAL_EQ_RAW(x, y, CONFIG_ACCESS_OPTION(GLOBAL, diff_eps))

/*! \brief Whether \a x is almost less or equal to \a y using tolerance \a eps.

    \param x (IN) Real number.
    \param y (IN) Real number.
    \param eps (IN) Tolerance.
*/
#define SA_REAL_ALMOST_LE_RAW(x, y, eps) \
    ((x) <= ((y) + (eps)))

/*! \brief Whether \a x is almost less or equal to \a y using global tolerance.

    \param x (IN) Real number.
    \param y (IN) Real number.

    \warning Uses the \em GLOBAL option \b diff_eps.
*/
#define SA_REAL_ALMOST_LE(x, y) \
    SA_REAL_ALMOST_LE_RAW(x, y, CONFIG_ACCESS_OPTION(GLOBAL, diff_eps))

/*! \brief Swaps \a x and \a y which are of type \a type.

    \param x (IN) Variable.
    \param y (IN) Variable.
    \param type (IN) The type of \a x and \a y.
*/
#define SA_SWAP(x, y, type) \
    do { \
        type z; \
        z = (x); \
        (x) = (y); \
        (y) = z; \
    } while (0)

/*! \brief Stringifies \a x.

    \param x (IN) Something.
*/
#define SA_STRINGIFY(x) #x

/*! \brief Converts \a x to string.

    \param x (IN) Something.
*/
#define SA_TOSTRING(x) SA_STRINGIFY(x)

/*! \brief Sets all \a flags in \a x.

    \param x (IN) State variable.
    \param flags (IN) Binary flags.

    \returns \a x with the set \a flags.
*/
#define SA_SET_FLAGS(x, flags)             ((x) |= (flags))

/*! \brief Unsets all \a flags in \a x.

    \param x (IN) State variable.
    \param flags (IN) Binary flags.

    \returns \a x with the unset \a flags.
*/
#define SA_UNSET_FLAGS(x, flags)           ((x) &= ~(flags))

/*! \brief Toggles all \a flags in \a x.

    \param x (IN) State variable.
    \param flags (IN) Binary flags.

    \returns \a x with the toggled \a flags.
*/
#define SA_TOGGLE_FLAGS(x, flags)          ((x) ^= (flags))

/*! \brief Checks if any of the given \a flags is set in \a x.

    \param x (IN) State variable.
    \param flags (IN) Binary flags.

    \returns Whether some of the \a flags are set in \a x.
*/
#define SA_IS_SET_A_FLAG(x, flags)         (((x) & (flags)))

/*! \brief Checks if all of the given \a flags are set in \a x.

    \param x (IN) State variable.
    \param flags (IN) Binary flags.

    \returns Whether all of the \a flags are set in \a x.
*/
#define SA_ARE_SET_ALL_FLAGS(x, flags)     (((x) & (flags)) == (flags))

/*! \def SA_ALERT
    \brief Alerts if \a expr does NOT hold.

    Or does nothing depending on \b SA_ASSERTS.

    \param expr (IN) Expression to check.

    \warning Uses the \em GLOBAL option \b asrts_output_stream.
*/
/*! \def SA_ALERT_MSG
    \brief Prints an alert message.

    Or does nothing depending on \b SA_ASSERTS.

    The variable arguments are the same as the standard \b printf.

    It puts new line symbol(s) in the end.

    \warning Uses the \em GLOBAL option \b asrts_output_stream.
*/
/*! \def SA_ALERT_COND_MSG
    \brief Prints an alert message if \a expr does NOT hold.

    Or does nothing depending on \b SA_ASSERTS.

    The variable arguments are the same as the standard \b printf.

    It puts new line symbol(s) in the end.

    \param expr (IN) Expression to check.

    \warning Uses the \em GLOBAL option \b asrts_output_stream.
*/
/*! \def SA_ASSERT
    \brief Aborts or halts the program if \a expr does NOT hold.

    Or does nothing depending on \b SA_ASSERTS.

    \param expr (IN) Expression to check.

    \warning Uses the \em GLOBAL option \b asrts_output_stream.
*/
/*! \def SA_VERIFY
    \brief A version of \b SA_ASSERT that preserves the side effects.

    If \b SA_ASSERTS is defined this is the same as \b SA_ASSERT. If
    \b SA_ASSERTS is NOT defined this, unlike \b SA_ASSERT, \b SA_ALERT, and so
    on, preserves the side effects of executing \a expr.

    \param expr (IN) Expression to check.

    \warning Uses the \em GLOBAL option \b asrts_output_stream.
*/
#ifdef SA_ASSERTS

#define SA_ALERT(expr) \
    do { \
        if (!(expr)) \
            SA_FPRINTF(CONFIG_ACCESS_OPTION(GLOBAL, asrts_output_stream), \
                       "%s", "ALERT: " __FILE__ ", " SA_TOSTRING(__LINE__) \
                       ": " #expr "\n"); \
    } while(0)

#define SA_ALERT_MSG(...) SA_ALERT_PRINTF(__VA_ARGS__)
#define SA_ALERT_COND_MSG(expr, ...) \
    do { \
        if (!(expr)) \
            SA_ALERT_MSG(#expr ": " __VA_ARGS__); \
    } while(0)

#define SA_ASSERT(expr) \
    do { \
        if (!(expr)) \
        { \
            SA_FPRINTF(CONFIG_ACCESS_OPTION(GLOBAL, asrts_output_stream), \
                       "%s", "ASSERT: " __FILE__ ", " SA_TOSTRING(__LINE__) \
                       ": " #expr "\n"); \
/*            SA_FPRINTF(CONFIG_ACCESS_OPTION(GLOBAL, asrts_output_stream), \
                         "HALTED!!!\n"); \
              for (;;) sleep((unsigned int)-1); */ \
            PROC_ABORT(99); \
        } \
    } while(0)

#define SA_VERIFY(expr) SA_ASSERT(expr)

#else

#define SA_ALERT(...) ((void)0)
#define SA_ALERT_MSG(...) ((void)0)
#define SA_ALERT_COND_MSG(...) ((void)0)
#define SA_ASSERT(...) ((void)0)
#define SA_VERIFY(expr) ((void)(expr))

#endif

/*! \def SA_START_TIMER
    \brief Sets a point where a timer starts.

    The timer \a timer is declared by this macro.

    Or does nothing depending on \b SA_TIMERS.

    \param timer (IN) Defines a timer name. The usual language rules for
                      variable names apply.
*/
/*! \def SA_STOP_TIMER
    \brief Sets a point where a timer stops.

    It prints the time elapsed between the calls to \b SA_START_TIMER and
    \b SA_STOP_TIMER.

    Or does nothing depending on \b SA_TIMERS.

    \param timer (IN) The timer to stop. The usual language rules for
                      variable names apply. This is a timer previously declared
                      and 'started' by a call to \b SA_START_TIMER.

    \warning This MUST be called in a context where the corresponding
             \b SA_START_TIMER is visible in the sense that the \a timer needs
             to be known.
*/
#ifdef SA_TIMERS

#define SA_START_TIMER(timer) \
    struct timeval __##timer##_start_; \
    SA_VERIFY(-1 != gettimeofday(&__##timer##_start_, NULL))

#define SA_STOP_TIMER(timer) \
    do { \
        struct timeval __##timer##_end_; \
        double interval; \
        SA_VERIFY(-1 != gettimeofday(&__##timer##_end_, NULL)); \
        interval = \
        (double)(__##timer##_end_.tv_sec - __##timer##_start_.tv_sec)*1000. + \
        (double)(__##timer##_end_.tv_usec - __##timer##_start_.tv_usec)/1000.; \
        SA_PRINTF("TIMER " #timer ": %.3f ms\n", interval); \
    } while (0)

#else

#define SA_START_TIMER(...) ((void)0)
#define SA_STOP_TIMER(...) ((void)0)

#endif

#endif // _COMMON_HPP
