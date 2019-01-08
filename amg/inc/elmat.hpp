/*! \file
    \brief Routines computing element matrices.

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

    XXX Notes:

    If OpenMP (or similar) would be used, functions of type
    \b agg_elmat_callback_ft MUST be thread-safe.
*/

#pragma once
#ifndef _ELMAT_HPP
#define _ELMAT_HPP

#include "common.hpp"
#include <mfem.hpp>
#include "aggregates.hpp"
#include "levels.hpp"

namespace saamge
{

/**
   Provides element matrices for the construction of agglomerate matrices.
*/
class ElementMatrixProvider
{
public:
    ElementMatrixProvider(
        const agg_partitioning_relations_t& agg_part_rels) 
        : agg_part_rels(agg_part_rels) {}
    virtual ~ElementMatrixProvider() {};
    
    /**
       Returns mfem::Matrix, always builds the whole matrix.
    */
    virtual mfem::Matrix * GetMatrix(int elno, bool& free_matr) const = 0;

    /**
       Returns SparseMatrix, does some magic with boundary conditions, sometimes
       copies from global SparseMatrix (if available) rather than doing its own
       assembly.
    */
    virtual mfem::SparseMatrix * BuildAEStiff(int elno) const = 0;

    bool IsGeometric() {return is_geometric;}
protected:
    const agg_partitioning_relations_t& agg_part_rels;
    bool is_geometric;
};

/**
   Uses an mfem::DomainLFIntegrator to provide a local (on AE) version of an assembled
   vector that represents the linear form on the finest (geometric) level, but
   using only a single domain linear form integrator. The vector
   is represented as a diagonal sparse matrix (to use the standard interface).
   No boundary conditions are respected currently.
*/
class ElementDomainLFVectorStandardGeometric : public ElementMatrixProvider
{
public:
    ElementDomainLFVectorStandardGeometric(
        const agg_partitioning_relations_t& agg_part_rels,
        mfem::DomainLFIntegrator *dlfi, mfem::FiniteElementSpace *fes) :
            ElementMatrixProvider(agg_part_rels), dlfi(dlfi), fes(fes) {}
    virtual mfem::Matrix *GetMatrix(int elno, bool& free_matr) const;
    virtual mfem::SparseMatrix *BuildAEStiff(int elno) const
    {
        return agg_build_AE_stiffm(elno, agg_part_rels, this, false);
    }
    virtual ~ElementDomainLFVectorStandardGeometric()
    {
        delete dlfi;
    }
private:
    mfem::DomainLFIntegrator *dlfi;
    mfem::FiniteElementSpace *fes;
};

/**
   Standard elmat for fine level, basically uses an mfem::ParBilinearForm
   to assemble everything.
*/
class ElementMatrixStandardGeometric : public ElementMatrixProvider
{
public:
    /**
       @param assembled_processor_matrix usually comes from a->GetSparseMatrix(),
       used to always go in the tg_ calling sequence but we are trying to move
       away from that.
    */
    ElementMatrixStandardGeometric(
        const agg_partitioning_relations_t& agg_part_rels,
        mfem::SparseMatrix& assembled_processor_matrix,
        mfem::ParBilinearForm* form);
    virtual mfem::Matrix * GetMatrix(int elno, bool& free_matr) const;
    /**
       Just copies (or calls?) agg_build_AE_stiffm_with_global
     */
    virtual mfem::SparseMatrix * BuildAEStiff(int elno) const;
    mfem::ParBilinearForm* GetParBilinearForm() const {return form;}
    void SetBdrCondImposed(bool val) {bdr_cond_imposed_ = val;}
private:
    mfem::ParBilinearForm* form;
    mfem::SparseMatrix& assembled_processor_matrix_;
    /**
       Used to be a CONFIG_CLASS, we will want to set this to false
       for upscaling (should be true for a solver)
    */
    bool bdr_cond_imposed_;
    /**
       In all my time with this code this has always been true
    */
    bool assemble_ess_diag_;
};

/**
   Standard elmat for coarse level.

   At its heart this is a RAP, using a saved local MIS interp matrix 
   (unsmoothed) and the fine level agglomerate matrix (also must be saved...)
*/
class ElementMatrixParallelCoarse : public ElementMatrixProvider
{
public:
    ElementMatrixParallelCoarse(
        const agg_partitioning_relations_t& agg_part_rels,
        levels_level_t * level);
    virtual mfem::Matrix * GetMatrix(int elno, bool& free_matr) const;
    /**
       Basically copies (or calls?)  agg_build_AE_stiffm()
    */
    virtual mfem::SparseMatrix * BuildAEStiff(int elno) const;
private:
    levels_level_t * level;
};

/**
   Returns element matrices from an array of generated matrices...

   Not really used yet, but might make sense for future algebraic interface.
*/
class ElementMatrixArray : public ElementMatrixProvider
{
public:
    ElementMatrixArray(const agg_partitioning_relations_t& agg_part_rels,
                       const mfem::Array<mfem::SparseMatrix *>& elem_matrs);
    virtual mfem::Matrix * GetMatrix(int elno, bool& free_matr) const;
    virtual mfem::SparseMatrix * BuildAEStiff(int elno) const;
private:
    const mfem::Array<mfem::SparseMatrix *>& elem_matrs;
};

/**
   Provider for element (GetMatrix) and agglomerate (BuildAEStiff) matrices
   Note that this class is different from ElementMatrixArray in the sense that
   both the GetMatrix and BuildAEStiff method in ElementMatrixArray returns
   agglomerate matrix

   Requires element matrices and partition relations as input
*/
class ElementMatrixDenseArray : public ElementMatrixProvider
{
public:
    ElementMatrixDenseArray(const agg_partitioning_relations_t& agg_part_rels,
                       const mfem::Array<mfem::DenseMatrix *>& elem_matrs);
    virtual mfem::Matrix * GetMatrix(int elno, bool& free_matr) const;
    virtual mfem::SparseMatrix * BuildAEStiff(int elno) const;
private:
    const mfem::Array<mfem::DenseMatrix *>& elem_matrs;
};

} // namespace saamge

#endif // _ELMAT_HPP
