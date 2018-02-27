/*! \file
    \brief Structures and methods for "levels" in a multilevel hierarchy.

    SAAMGE: smoothed aggregation element based algebraic multigrid hierarchies
            and solvers.

    Copyright (c) 2018, Lawrence Livermore National Security,
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
#ifndef _LEVELS_HPP
#define _LEVELS_HPP

#include "common.hpp"
#include "aggregates.hpp"
#include "tg.hpp"

namespace saamge
{

/* Types */
/*! \brief Structure for levels.
*/
typedef struct levels_level_struct {
    struct levels_level_struct *finer; /*!< Finer level. */

    agg_partitioning_relations_t *agg_part_rels; /*!< Partitioning data for
                                                       the current level. */
    tg_data_t *tg_data; /*!< Two-grid data connecting the current level to the
                             coarser one. */

    struct levels_level_struct *coarser; /*!< Coarser level. */
} levels_level_t;

/*! \brief Structure for lists of levels.
*/
typedef struct {
    int num_levels; /*!< The number of levels in the list. */
    levels_level_t *finest; /*!< Pointer to the finest level. */
    levels_level_t *coarsest; /*!< Pointer to the coarsest level. */
} levels_list_t;

/* Inline Functions */
/*! \brief Allocates a level object.

    \returns The allocated object.

    \warning The returned structure must be freed by the caller using functions
             in this file.
*/
static inline
levels_level_t *levels_alloc_level(void);

/*! \brief Deallocates a level object.

    \param level (IN) The level object.
*/
static inline
void levels_free_level_struct(levels_level_t *level);

/*! \brief Deallocates a level object together with the contained data.

    \param level (IN) The level object.
*/
static inline
void levels_fully_free_level(levels_level_t *level);

/*! \brief Copies a level object.

    \param src (IN) The level to be copied.

    \returns The copy of the object.

    \warning The returned structure must be freed by the caller using functions
             in this file.
*/
static inline
levels_level_t *levels_copy_level(const levels_level_t *src);

/*! \brief Allocates a levels list.

    \returns The allocated list.

    \warning The returned structure must be freed by the caller using functions
             in this file.
*/
static inline
levels_list_t *levels_alloc_list(void);

/*! \brief Deallocates a list object.

    Frees only the structure but the contained level nodes are not touched.

    \param list (IN) The list.
*/
static inline
void levels_free_list_struct(levels_list_t *list);

/*! \brief Deallocates a list object together with all level objects.

    \param list (IN) The list.
*/
static inline
void levels_free_list_and_level_structs(levels_list_t *list);

/*! \brief Deallocates a list object, all level objects, and all contained data.

    \param list (IN) The list.
*/
static inline
void levels_fully_free_list(levels_list_t *list);

/*! \brief Completely deallocates a level and all that are coarser than it.

    \param list (IN) The list.
    \param level_num (IN) Level index which increases in the coarse direction.
*/
static inline
void levels_list_free_level_and_all_coarser(levels_list_t& list, int level_num);

/*! \brief Attaches a given level object to the coarse side of the list.

    \param list (IN/OUT) The list.
    \param level (IN) The level object.
*/
static inline
void levels_list_push_coarse_level(levels_list_t& list, levels_level_t *level);

/*! \brief Adds given level data to the coarse side of the list.

    \param list (IN/OUT) The list.
    \param agg_part_rels (IN) Partitioning relations.
    \param tg_data (IN) Two-grid data.

    \returns The corresponding level structure instance.
*/
static inline
levels_level_t *levels_list_push_coarse_data(levels_list_t& list,
                    agg_partitioning_relations_t *agg_part_rels,
                    tg_data_t *tg_data);

/*! \brief Allocates a levels list and inserts one level in it.

    \param agg_part_rels (IN) Partitioning relations.
    \param tg_data (IN) Two-grid data.

    \returns The allocated list.

    \warning The returned structure must be freed by the caller using functions
             in this file.
*/
static inline
levels_list_t *levels_create_list_one_level(
                    agg_partitioning_relations_t *agg_part_rels,
                    tg_data_t *tg_data);

/*! \brief Finds a level object in a list according to its index.

    Iterates from the finest to the coarsest side.

    \param list (IN) The list.
    \param num (IN) Index which increases in the coarse direction.

    \returns The corresponding level structure instance.
*/
static inline
levels_level_t *levels_list_iterate_from_finest(const levels_list_t& list,
                                                int num);

/*! \brief Finds a level object in a list according to its index.

    Iterates from the coarsest to the finest side.

    \param list (IN) The list.
    \param num (IN) Index which increases in the fine direction.

    \returns The corresponding level structure instance.
*/
static inline
levels_level_t *levels_list_iterate_from_coarsest(const levels_list_t& list,
                                                  int num);

/*! \brief Finds a level object in a list according to its index.

    \param list (IN) The list.
    \param level_num (IN) Index which increases in the coarse direction.

    \returns The corresponding level structure instance.
*/
static inline
levels_level_t *levels_list_get_level(const levels_list_t& list, int level_num);

/*! \brief Verifies the integrity of a list.

    \param list (IN) The list.

    \returns \em true if the list id fine, \em false otherwise.
*/
static inline
bool levels_check_list(const levels_list_t& list);

/* Inline Functions Definitions */
static inline
levels_level_t *levels_alloc_level(void)
{
    levels_level_t *level = new levels_level_t;
    SA_ASSERT(level);
    memset(level, 0, sizeof(*level));
    return level;
}

static inline
void levels_free_level_struct(levels_level_t *level)
{
    delete level;
}

static inline
void levels_fully_free_level(levels_level_t *level)
{
    if (!level)
        return;
    agg_free_partitioning(level->agg_part_rels);
    tg_free_data(level->tg_data);
    levels_free_level_struct(level);
}

static inline
levels_level_t *levels_copy_level(const levels_level_t *src)
{
    if (!src) return NULL;
    levels_level_t *dst = levels_alloc_level();
    dst->agg_part_rels = agg_copy_partitioning(src->agg_part_rels);
    dst->tg_data = tg_copy_data(src->tg_data);
    return dst;
}

static inline
levels_list_t *levels_alloc_list(void)
{
    levels_list_t *list = new levels_list_t;
    SA_ASSERT(list);
    memset(list, 0, sizeof(*list));
    return list;
}

static inline
void levels_free_list_struct(levels_list_t *list)
{
    delete list;
}

static inline
void levels_free_list_and_level_structs(levels_list_t *list)
{
    if (!list)
        return;

    levels_level_t *level, *level_coarser;
    for (level = list->finest; level; level = level_coarser)
    {
        level_coarser = level->coarser;
        levels_free_level_struct(level);
    }
    levels_free_list_struct(list);
}

static inline
void levels_fully_free_list(levels_list_t *list)
{
    if (!list)
        return;

    levels_level_t *level, *level_coarser;
    for (level = list->finest; level; level = level_coarser)
    {
        level_coarser = level->coarser;
        levels_fully_free_level(level);
    }
    levels_free_list_struct(list);
}

static inline
void levels_list_free_level_and_all_coarser(levels_list_t& list, int level_num)
{
    levels_level_t *level, *level_finer;
    SA_ASSERT(0 <= level_num && level_num < list.num_levels);
    for (level = list.coarsest; level && list.num_levels > level_num;
         (--list.num_levels), (level = level_finer))
    {
        SA_ASSERT(list.num_levels);
        level_finer = level->finer;
        levels_fully_free_level(level);
        if (level_finer)
            level_finer->coarser = NULL;
        list.coarsest = level_finer;
    }
    SA_ASSERT(level_num == list.num_levels);
    if (!list.num_levels)
    {
        SA_ASSERT(!list.coarsest);
        SA_ASSERT(!level_num);
        list.finest = NULL;
    }
    SA_ASSERT(levels_check_list(list));
}

static inline
void levels_list_push_coarse_level(levels_list_t& list, levels_level_t *level)
{
    SA_ASSERT((0 < list.num_levels && list.coarsest && list.finest) ||
              (!list.num_levels && !list.coarsest && !list.finest));
    level->coarser = NULL;
    level->finer = list.coarsest;
    if (list.coarsest)
        list.coarsest->coarser = level;
    else
    {
        SA_ASSERT(!list.finest && !list.num_levels);
        list.finest = level;
    }
    list.coarsest = level;
    SA_ASSERT(list.coarsest && list.finest);
    ++list.num_levels;
    SA_ASSERT(0 < list.num_levels);
}

static inline
levels_level_t *levels_list_push_coarse_data(levels_list_t& list,
                    agg_partitioning_relations_t *agg_part_rels,
                    tg_data_t *tg_data)
{
    levels_level_t *level = levels_alloc_level();
    SA_ASSERT(level);
    level->agg_part_rels = agg_part_rels;
    level->tg_data = tg_data;
    levels_list_push_coarse_level(list, level);
    return level;
}

static inline
levels_list_t *levels_create_list_one_level(
                    agg_partitioning_relations_t *agg_part_rels,
                    tg_data_t *tg_data)
{
    levels_list_t *list = levels_alloc_list();
    levels_list_push_coarse_data(*list, agg_part_rels, tg_data);
    return list;
}

static inline
levels_level_t *levels_list_iterate_from_finest(const levels_list_t& list,
                                                int num)
{
    SA_ASSERT((0 < list.num_levels && list.coarsest && list.finest) ||
              (!list.num_levels && !list.coarsest && !list.finest));
    levels_level_t *level;
    int i;
    for ((i=0), (level = list.finest); level && i < num;
         (++i), (level = level->coarser))
    {
        SA_ASSERT(level->finer || list.finest == level);
        SA_ASSERT(level->coarser || list.coarsest == level);
    }
    SA_ASSERT(i <= list.num_levels);
    SA_ASSERT(level->finer || list.finest == level);
    SA_ASSERT(level->coarser || list.coarsest == level);
    SA_ASSERT(num || list.finest == level);
    SA_ASSERT((0 <= num && num < list.num_levels) || !level);
    SA_ASSERT(list.num_levels != num + 1 || list.coarsest == level);
    return level;
}

static inline
levels_level_t *levels_list_iterate_from_coarsest(const levels_list_t& list,
                                                  int num)
{
    SA_ASSERT((0 < list.num_levels && list.coarsest && list.finest) ||
              (!list.num_levels && !list.coarsest && !list.finest));
    levels_level_t *level;
    int i;
    for ((i=0), (level = list.coarsest); level && i < num;
         (++i), (level = level->finer))
    {
        SA_ASSERT(level->finer || list.finest == level);
        SA_ASSERT(level->coarser || list.coarsest == level);
    }
    SA_ASSERT(i <= list.num_levels);
    SA_ASSERT(level->finer || list.finest == level);
    SA_ASSERT(level->coarser || list.coarsest == level);
    SA_ASSERT(num || list.coarsest == level);
    SA_ASSERT((0 <= num && (unsigned)num < (unsigned)list.num_levels)
              || !level);
    SA_ASSERT(list.num_levels != num + 1 || list.finest == level);
    return level;
}

static inline
levels_level_t *levels_list_get_level(const levels_list_t& list, int level_num)
{
    SA_ASSERT((0 < list.num_levels && list.coarsest && list.finest) ||
              (!list.num_levels && !list.coarsest && !list.finest));
    if (0 > level_num || level_num >= list.num_levels)
        return NULL;
    if (level_num < (list.num_levels >> 1))
        return levels_list_iterate_from_finest(list, level_num);
    SA_ASSERT(list.num_levels - level_num - 1 >= 0);
    return levels_list_iterate_from_coarsest(list,
                                             list.num_levels - level_num - 1);
}

static inline
bool levels_check_list(const levels_list_t& list)
{
    if (!list.num_levels && !list.coarsest && !list.finest)
        return true;

    if (0 > list.num_levels)
        return false;

    if ((!list.num_levels || !list.coarsest || !list.finest) &&
        (list.num_levels || list.coarsest || list.finest))
        return false;

    if ((1 == list.num_levels && list.coarsest != list.finest) ||
        (1 < list.num_levels && list.coarsest == list.finest))
        return false;

    if ((list.coarsest && list.coarsest->coarser) ||
        (list.finest && list.finest->finer))
        return false;

    levels_level_t *level;
    int ctr;

    for ((ctr=0), (level = list.finest); level && ctr < list.num_levels - 1;
         (++ctr), (level = level->coarser));
    if (list.coarsest != level || list.num_levels != ctr + 1 ||
        (level && level->coarser))
        return false;

    for ((ctr=0), (level = list.coarsest); level && ctr < list.num_levels - 1;
         (++ctr), (level = level->finer));
    if (list.finest != level || list.num_levels != ctr + 1 ||
        (level && level->finer))
        return false;

    return true;
}

} // namespace saamge

#endif // _LEVELS_HPP
