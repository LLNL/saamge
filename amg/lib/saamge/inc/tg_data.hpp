/*! \file
    \brief The two-grid (TG) implementation.

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

#ifndef _TG_DATA_HPP
#define _TG_DATA_HPP

#include "common.hpp"
#include <mfem.hpp>
#include "smpr.hpp"
#include "mbox.hpp"
#include "interp.hpp"

using namespace mfem;

/*! \brief TG data -- interpolation, smoothers etc.
*/
typedef struct {
    interp_data_t *interp_data; /*!< Parameters and data for the
                                     interpolant. */

    HypreParMatrix *Ac; /*!< The coarse-grid operator. */
    HypreParMatrix *interp; /*!< The interpolator. */
    HypreParMatrix *restr; /*!< The restriction operator. */
    SparseMatrix *ltent_interp; /*!< The local (for the process) tentative
                                     interpolant. */
    HypreParMatrix *tent_interp; /*!< The global (among all processes)
                                      tentative interpolant. */

    HypreParMatrix * scaling_P; /*!< for corrected nullspace, replaces coarse_one_representation */

    bool smooth_interp; /*!< Whether to smooth the tentative interpolator or
                             simply copy it and use it as a final
                             prolongator. */

    double theta; /*!< Spectral tolerance. */

    smpr_ft pre_smoother; /*!< The smoother for the pre-smoothing step. */
    smpr_ft post_smoother; /*!< The smoother for the post-smoothing step. */

    mfem::Solver * coarse_solver;

    smpr_poly_data_t *poly_data; /*< The data for the polynomial smoother. */

    bool use_w_cycle; /*!< whether to use W-cycle or V-cycle DEPRECATED */
    bool minimal_coarse_space; /*!< if true, no eigenvalue problem, just vector of ones */

    int tag;

    ElementMatrixProvider * elem_data; /*!< keep a pointer so tg_data can free this */
} tg_data_t;

#endif
