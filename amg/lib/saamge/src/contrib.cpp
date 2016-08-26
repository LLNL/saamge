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
#include "contrib.hpp"
#include <mfem.hpp>
#include "aggregates.hpp"
#include "xpacks.hpp"
#include "mbox.hpp"

/* Options */

CONFIG_DEFINE_CLASS(CONTRIB);

// int abs_pair_compare(const std::pair<double, int> *a, const std::pair<double, int> *b)
int abs_pair_compare(const void *va, const void *vb)
{
    std::pair<double, int> *a, *b;
    a = (std::pair<double, int>*) va;
    b = (std::pair<double, int>*) vb;
    if (fabs(a->first) < fabs(b->first))
        return -1;
    else if (fabs(a->first) > fabs(b->first))
        return 1;
    else
        return 0;
}

/*! 
  This is a (very) naive implementation.
  Once upon a time we used this to do a few nonzeros per aggregate in 
  corrected nullspace setup, but now this is orphan code.

  DEPRECATED
*/
void n_largest(Vector &x, int n, Array<int> &locations, Array<double> &values)
{
    SA_ASSERT(n <= x.Size());
    Array<std::pair<double, int> > pairs(n);
    for (int i=0; i<n; ++i)
    {
        pairs[i].first = x(i);
        pairs[i].second = i;
    }
    std::qsort(pairs.GetData(), pairs.Size(), sizeof(std::pair<double, int>), abs_pair_compare);
    for (int i=n; i<x.Size(); ++i)
    {
        std::pair<double, int> newpair(x(i), i);
        if (abs_pair_compare((void*) &newpair, (void*) &pairs[0]) == 1)
        {
            pairs[0] = newpair;
            std::qsort(pairs.GetData(), pairs.Size(), sizeof(std::pair<double, int>), abs_pair_compare);
        }
    }
    for (int i=0; i<n; ++i)
    {
        values[i] = pairs[i].first;
        locations[i] = pairs[i].second;
    }
}

/* Static Functions */

/* Functions */

contrib_tent_struct_t *contrib_tent_init(int ND)
{
    contrib_tent_struct_t *tent_int_struct = new contrib_tent_struct_t;
    tent_int_struct->rows = ND;
    tent_int_struct->filled_cols = 0;
    tent_int_struct->tent_interp = new SparseMatrix(tent_int_struct->rows);
    // tent_int_struct->local_coarse_one_representation = new Vector();
    tent_int_struct->local_coarse_one_representation = new Array<double>();
    // tent_int_struct->coarse_ones_values_per_agg = coarse_ones_values_per_agg;

    return tent_int_struct;
}

SparseMatrix *contrib_tent_finalize(contrib_tent_struct_t *tent_int_struct)
{
    SparseMatrix *tent_interp;
    // on very coarse levels, the following assertion sometimes fails...
    // SA_ASSERT(tent_int_struct->filled_cols > 0);
    SA_ASSERT(tent_int_struct->filled_cols >= 0);
    if (tent_int_struct->filled_cols == 0)
        SA_PRINTF("%s","WARNING! no coarse degrees of freedom on this processor.\n");
    tent_int_struct->tent_interp->Finalize();
    tent_interp = new SparseMatrix(tent_int_struct->tent_interp->GetI(),
                                   tent_int_struct->tent_interp->GetJ(),
                                   tent_int_struct->tent_interp->GetData(),
                                   tent_int_struct->tent_interp->Size(),
                                   tent_int_struct->filled_cols);
    tent_int_struct->tent_interp->LoseData();
    delete (tent_int_struct->tent_interp);
    SA_ASSERT(tent_interp->GetI() && tent_interp->GetJ() &&
              tent_interp->GetData());
    SA_ASSERT(tent_interp->Size() == tent_int_struct->rows);
    SA_ASSERT(tent_interp->Width() == tent_int_struct->filled_cols);

    // delete tent_int_struct->local_coarse_one_representation;
    delete tent_int_struct;
    return tent_interp;
}

/**
   Extract some of the logic, in terms of essential boundary conditions,
   from contrib_tent_insert_from_local() so we can do SVD after
   we do this

   ATB 16 June 2015
*/
void contrib_filter_boundary(contrib_tent_struct_t *tent_int_struct,
                             const agg_partitioning_relations_t& agg_part_rels,
                             DenseMatrix& local, 
                             const int *restriction)
{
    const bool avoid_ess_brd_dofs = 
        CONFIG_ACCESS_OPTION(CONTRIB, avoid_ess_brd_dofs);
    const int vects = local.Width();
    const int dim = local.Height();
    double *data = local.Data();
    int col, i, j;

    double * newdata = new double[vects * dim];
    double * newdatap = newdata;

    SA_ASSERT(vects > 0);
    // SA_ASSERT(dim >= vects); // can we assume this will be taken care of by SVD?

    col = 0;
    double newcolumn[dim];
    for (i=0; i < vects; (++i), (data += dim))
    {
        bool atleastone = false;
        int numremoved = 0;
        for (j=0; j < dim; ++j)
        {
            SA_ASSERT(data + j < local.Data() + dim*vects);
            const int row = restriction[j];
            const double a = data[j];

            if (a == 0.0 ||
                (avoid_ess_brd_dofs &&
                 agg_is_dof_on_essential_border(agg_part_rels, row)))
            {
                if (SA_IS_OUTPUT_LEVEL(7) && 0. != a)
                    SA_ALERT_PRINTF("Non-zero DoF on essential boundary."
                                    " Ignoring entry: %g!", a); // just a single entry here...
                newcolumn[j] = 0.0;
                numremoved++;
                continue;
            }
            atleastone = true;
            newcolumn[j] = a;
        }

        if (atleastone)
        {
            for (j=0; j< dim; ++j)
            {
                newdatap[j] = newcolumn[j];
            }
            newdatap += dim;
            col++;
        }
        else
        {
            SA_ALERT_PRINTF("%s","Entire column is on essential boundary, ignoring column!");
            // wait, doesn't this imply that the entire *matrix* is on the essential boundary?
        }
    }
    local.SetSize(dim, col);
    data = local.Data();
    for (int j=0; j<dim * local.Width(); ++j)
        data[j] = newdata[j];
    delete [] newdata;
}

/**
   separated from contrib_tent_insert_from_local(),
   separating the essential boundary checking from the actual
   insertion so we can do SVD in between
*/
void contrib_tent_insert_simple(contrib_tent_struct_t *tent_int_struct,
                                const agg_partitioning_relations_t& agg_part_rels,
                                DenseMatrix& local, 
                                const int *restriction)
{
    const int vects = local.Width();
    const int dim = local.Height();
    double *data = local.Data();
    int col, i, j;

    SA_ASSERT(vects > 0);
    SA_ASSERT(dim >= vects);

    col = tent_int_struct->filled_cols;
    for (i=0; i < vects; (++i), (data += dim))
    {
        for (j=0; j< dim; ++j)
        {
            const int row = restriction[j];
            tent_int_struct->tent_interp->Set(row, col, data[j]);
        }
        ++col;
    }
    tent_int_struct->filled_cols = col;
}


/**
   this routine changed ATB 11 May 2015 to also modify local according
   to boundary conditions, not just the global interp, because I think
   this is mathematically the right thing to do and because we now store
   local as a representative of coarse DoFs for multilevel extension

   recently in multilevel setting we are moving from this to a combination 
   of contrib_filter_boundary() and contrib_tent_insert_simple(), this
   may be deprecated soon
*/
void contrib_tent_insert_from_local(contrib_tent_struct_t *tent_int_struct,
                                    const agg_partitioning_relations_t& agg_part_rels,
                                    DenseMatrix& local, 
                                    const int *restriction)
{
    const bool avoid_ess_brd_dofs = CONFIG_ACCESS_OPTION(CONTRIB,
                                                         avoid_ess_brd_dofs);
    const int vects = local.Width();
    const int dim = local.Height();
    double *data = local.Data();
    bool modified = false; // set to true if newdata != data
    int col, i, j;

    double * newdata = new double[vects * dim];
    double * newdatap = newdata;

    SA_ASSERT(vects > 0);
    SA_ASSERT(dim >= vects);

    int firstcol = tent_int_struct->filled_cols;
    col = tent_int_struct->filled_cols;
    double newcolumn[dim];
    int bestcase_nonzerodofs = 0;
    int nonzerodofs = 0;
    for (i=0; i < vects; (++i), (data += dim))
    {
        nonzerodofs = 0;
        double adhoc_column_norm = 0.0;
        for (j=0; j < dim; ++j)
        {
            SA_ASSERT(data + j < local.Data() + dim*vects);
            const int row = restriction[j];
            const double a = data[j];

            if (a == 0.0 ||
                (avoid_ess_brd_dofs &&
                 agg_is_dof_on_essential_border(agg_part_rels, row)))
            {
                if (SA_IS_OUTPUT_LEVEL(7) && 0. != a)
                    SA_ALERT_PRINTF("Non-zero DoF on essential boundary."
                                    " Ignoring entry: %g!", a); // just a single entry here...
                newdatap[j] = 0.0; // new ATB 11 May 2015
                newcolumn[j] = 0.0;
                modified = true;
                continue;
            }
            nonzerodofs++;
            adhoc_column_norm += fabs(a);
            newcolumn[j] = a;
        }
        if (nonzerodofs > bestcase_nonzerodofs) bestcase_nonzerodofs = nonzerodofs;
        if (adhoc_column_norm < 1.e-3) // not clear what the right tolerance here is, especially with varying coefficients
        {
            SA_ALERT_PRINTF("Tentative prolongator column is near zero, l1 norm %e, Ignoring column!",
                            adhoc_column_norm);
            modified = true;
        }
        else 
        {
            if (adhoc_column_norm < 1.e-1)
                SA_ALERT_PRINTF("Accepting column of small l1 norm %e!",adhoc_column_norm);
            for (j=0; j< dim; ++j)
            {
                const int row = restriction[j];
                tent_int_struct->tent_interp->Set(row, col, newcolumn[j]);
                newdatap[j] = newcolumn[j];
            }
            ++col;
            newdatap += dim;
        }
    }
    SA_ASSERT(bestcase_nonzerodofs >= col - firstcol);
    if (modified)
    {
        local.SetSize(dim, col - tent_int_struct->filled_cols);
        data = local.Data();
        for (int j=0; j<dim * local.Width(); ++j)
            data[j] = newdata[j];
    }   
    delete [] newdata;
    tent_int_struct->filled_cols = col;
}

/**
   uber-simplifed version of contrib_mises, under the assumption
   that we do one coarse DOF per MIS, in particular the (normalized)
   vector of all ones
*/
void contrib_ones(contrib_tent_struct_t *tent_int_struct,
                  const agg_partitioning_relations_t& agg_part_rels)
{
    int num_mises = agg_part_rels.num_mises;
    const bool avoid_ess_brd_dofs = CONFIG_ACCESS_OPTION(CONTRIB,
                                                         avoid_ess_brd_dofs);

    tent_int_struct->mis_tent_interps = new DenseMatrix*[num_mises]; 
    memset(tent_int_struct->mis_tent_interps,0,sizeof(DenseMatrix*) * num_mises);
    Vector svals;
    int num_coarse_dofs = 0;
    tent_int_struct->mis_numcoarsedof = new int[num_mises];
    for (int mis=0; mis<num_mises; ++mis)
    {
        int owner = agg_part_rels.mis_master[mis];
        if (owner == PROC_RANK)
        {
            int mis_size = agg_part_rels.mises_size[mis];
            tent_int_struct->mis_numcoarsedof[mis] = 1;
            tent_int_struct->mis_tent_interps[mis] = new DenseMatrix;

            // check to see if all of this MISes DOFs are on essential boundary - copied from contrib_big_aggs()
            // this only checks the dofs for one AE, but that should be sufficient
            if (avoid_ess_brd_dofs)
            {
                bool interior_dofs = false;
                for (int j=0; j < mis_size; ++j)
                {
                    const int row = agg_part_rels.mis_to_dof->GetRow(mis)[j];
                    SA_ASSERT(tent_int_struct->rows > row);
                    if (!agg_is_dof_on_essential_border(agg_part_rels, row))
                    {
                        interior_dofs = true;
                        break;
                    }
                }
                if (!interior_dofs)
                {
                    if (SA_IS_OUTPUT_LEVEL(6))
                        SA_ALERT_PRINTF("All DoFs are on essential boundary."
                                        " Ignoring the entire contribution"
                                        " introducing not more than 1 vector(s) on"
                                        " an aggregate of size %d!",
                                        mis_size);
                    tent_int_struct->mis_numcoarsedof[mis] = 0;
                    tent_int_struct->mis_tent_interps[mis]->SetSize(mis_size, 0); // this makes future assertions and communications cleaner, but is mostly unnecessary
                    continue; // TODO: remove this, refactor
                }
            }

            tent_int_struct->mis_tent_interps[mis]->SetSize(mis_size, 1);
            double scale = std::sqrt((double) mis_size);
            for (int k=0; k<mis_size; ++k)
                tent_int_struct->mis_tent_interps[mis]->Elem(k,0) = 1.0 / scale;
            contrib_filter_boundary(tent_int_struct, agg_part_rels,
                                    *tent_int_struct->mis_tent_interps[mis],
                                    agg_part_rels.mis_to_dof->GetRow(mis));

            if (agg_part_rels.testmesh)
            {
                std::stringstream filename;
                filename << "mis_tent_interp_" << mis << "." << PROC_RANK << ".densemat";
                std::ofstream out(filename.str().c_str());
                tent_int_struct->mis_tent_interps[mis]->Print(out);
            }
            int filled_cols = tent_int_struct->filled_cols;
            contrib_tent_insert_simple(tent_int_struct, agg_part_rels,
                                       *tent_int_struct->mis_tent_interps[mis],
                                       agg_part_rels.mis_to_dof->GetRow(mis));

            filled_cols = tent_int_struct->filled_cols - filled_cols;
            SA_ASSERT(filled_cols == tent_int_struct->mis_tent_interps[mis]->Width());
            tent_int_struct->mis_numcoarsedof[mis] = filled_cols;
            num_coarse_dofs += filled_cols;
        }
        else
        {
            tent_int_struct->mis_numcoarsedof[mis] = 0;
            tent_int_struct->mis_tent_interps[mis] = new DenseMatrix; // we only do this so deletion is cleaner at the end, could avoid it at the cost of more ifs
            // tent_int_struct->mis_tent_interps[mis]->SetSize(0,0); // ???
        }
    }

    tent_int_struct->coarse_truedof_offset = 0;
    MPI_Scan(&num_coarse_dofs,&tent_int_struct->coarse_truedof_offset,1,MPI_INT,MPI_SUM,PROC_COMM);
    tent_int_struct->coarse_truedof_offset -= num_coarse_dofs;
    SA_RPRINTF(PROC_NUM-1,"coarse_truedof_offset = %d\n",tent_int_struct->coarse_truedof_offset);
}

/**
   Takes solutions to spectral problems on AEs, restricts to MISes, does
   appropriate communication and SVD, and constructs tentative prolongator

   This is one of the key communication routine for the multilevel MIS extension
   to this solver

   Possibly more attention needs to be paid to boundary conditions and
   small (1-2 dof) MISes
*/
void contrib_mises(contrib_tent_struct_t *tent_int_struct,
                   const agg_partitioning_relations_t& agg_part_rels,
                   DenseMatrix * const *cut_evects_arr,
                   double eps, bool scaling_P)
{
    const bool avoid_ess_brd_dofs = CONFIG_ACCESS_OPTION(CONTRIB,
                                                         avoid_ess_brd_dofs);

    // restrict eigenvectors to MISes
    int num_mises = agg_part_rels.num_mises;
    DenseMatrix ** restricted_evects_array;
    restricted_evects_array = new DenseMatrix*[num_mises];
    for (int mis=0; mis<num_mises; ++mis)
    {
        int local_AEs_containing = agg_part_rels.mis_to_AE->RowSize(mis);
        int mis_size = agg_part_rels.mises_size[mis];
        restricted_evects_array[mis] = new DenseMatrix[local_AEs_containing];
        // restrict local AE to this MIS (copied from Delyan Kalchev's contrib_ref_aggs())
        for (int ae=0; ae<local_AEs_containing; ++ae)
        {
            // SA_PRINTF("ae = %d, restriction = %p\n", ae, restriction);
            int AE_id = agg_part_rels.mis_to_AE->GetRow(mis)[ae];
            agg_restrict_to_agg_enforce(AE_id, agg_part_rels, mis_size,
                                        agg_part_rels.mis_to_dof->GetRow(mis),
                                        *(cut_evects_arr[AE_id]),
                                        restricted_evects_array[mis][ae]);
        }
    }

    // agg_part_rels.mis_truemis->Print("mis_truemis_contrib.mat"); // TODO remove!!!

    // communication: collect MIS-restricted eigenvectors on the process that owns the MIS
    SharedEntityCommunication<DenseMatrix> sec(PROC_COMM,
                                               *agg_part_rels.mis_truemis);
    sec.ReducePrepare();

    // delete h_cd;

    for (int mis=0; mis<num_mises; ++mis)
    {
        // combine all the AEs into one DenseMatrix (this is complicated 
        // and expensive in memory but might save us latency costs...)
        int mis_size = agg_part_rels.mises_size[mis];
        int rowsize = agg_part_rels.mis_to_AE->RowSize(mis);
        int * row = agg_part_rels.mis_to_AE->GetRow(mis);
        int numvecs = 0;
        for (int j=0; j<rowsize; ++j)
        {
            int AE = row[j];
            numvecs += cut_evects_arr[AE]->Width();
        }    
        DenseMatrix send_mat(mis_size, numvecs);
        numvecs = 0;
        for (int j=0; j<rowsize; ++j)
        {
            int AE = row[j];
            memcpy(send_mat.Data() + numvecs*mis_size, 
                   restricted_evects_array[mis][j].Data(), 
                   mis_size * cut_evects_arr[AE]->Width() * sizeof(double));
            numvecs += cut_evects_arr[AE]->Width();
        }
        delete [] restricted_evects_array[mis];

        sec.ReduceSend(mis,send_mat);
    }
    delete [] restricted_evects_array;
    DenseMatrix ** received_mats = sec.Collect();

    // do SVDs on owned MISes, build tentative interpolator
    DenseMatrix lsvects;
    tent_int_struct->mis_tent_interps = new DenseMatrix*[num_mises]; // TODO: can we make this pointer to array of DenseMatrix, not DenseMatrix* ?
    memset(tent_int_struct->mis_tent_interps,0,sizeof(DenseMatrix*) * num_mises);
    Vector svals;
    int num_coarse_dofs = 0;
    tent_int_struct->mis_numcoarsedof = new int[num_mises];
    for (int mis=0; mis<num_mises; ++mis)
    {
        // SA_PRINTF("mis %d begins:\n",mis);
        int owner = agg_part_rels.mis_master[mis];
        if (owner == PROC_RANK)
        {
            // int local_AEs_containing = agg_part_rels.mis_to_AE->RowSize(mis);
            // int row_size = agg_part_rels.mis_proc->RowSize(mis);
            int row_size = sec.NumNeighbors(mis);
            tent_int_struct->mis_tent_interps[mis] = new DenseMatrix;

            // check to see if all of this MISes DOFs are on essential boundary - copied from contrib_big_aggs()
            // this only checks the dofs for one AE, but that should be sufficient
            const int dim = received_mats[mis][0].Height();
            if (avoid_ess_brd_dofs)
            {
                bool interior_dofs = false;
                for (int j=0; j < dim; ++j)
                {
                    const int row = agg_part_rels.mis_to_dof->GetRow(mis)[j];
                    SA_ASSERT(tent_int_struct->rows > row);
                    if (!agg_is_dof_on_essential_border(agg_part_rels, row))
                    {
                        interior_dofs = true;
                        break;
                    }
                }
                if (!interior_dofs)
                {
                    if (SA_IS_OUTPUT_LEVEL(6))
                        SA_ALERT_PRINTF("All DoFs are on essential boundary."
                                        " Ignoring the entire contribution"
                                        " introducing not more than %d vector(s) on"
                                        " an aggregate of size %d!",
                                        received_mats[mis][0].Width(), dim);
                    tent_int_struct->mis_numcoarsedof[mis] = 0;
                    tent_int_struct->mis_tent_interps[mis]->SetSize(dim, 0); // this makes future assertions and communications cleaner, but is mostly unnecessary
                    delete [] received_mats[mis];
                    continue; // TODO: remove this, refactor
                }
            }

            if (dim == 1) // could think about a kind of identity matrix whenever dim < total width, but I think SVD will take care of this
            {
                // see assertion in contrib_tent_insert_from_local: SA_ASSERT(dim > 1 || 1. == a);
                tent_int_struct->mis_tent_interps[mis]->SetSize(1,1);
                tent_int_struct->mis_tent_interps[mis]->Elem(0,0) = 1.0;
            }
            else
            {
                /*
                if (agg_part_rels.testmesh)
                {
                    // SA_PRINTF("  mis %d, dim %d, DenseMats in received_mats = %d\n",
                    //        mis, dim, local_AEs_containing + row_size - 1);
                    for (int q=0; q<local_AEs_containing + row_size - 1; ++q)
                    {
                        std::stringstream filename;
                        filename << "received_mats_" << mis << "_" << q << "." << PROC_RANK << ".densemat";
                        std::ofstream out(filename.str().c_str());
                        received_mats[mis][q].Print(out);
                    }
                }
                */
                int total_num_columns = 0;
                for (int q=0; q<row_size; ++q)
                {
                    contrib_filter_boundary(tent_int_struct, agg_part_rels,
                                            received_mats[mis][q],
                                            agg_part_rels.mis_to_dof->GetRow(mis));
                    total_num_columns += received_mats[mis][q].Width();
                    /*
                    if (agg_part_rels.testmesh)
                    {
                        std::stringstream filename;
                        filename << "received_mats_f_" << mis << "_" << q << "." << PROC_RANK << ".densemat";
                        std::ofstream out(filename.str().c_str());
                        received_mats[mis][q].Print(out);
                    }
                    */
                }

//                the change below is arguably correct, but something else is definitely going wrong with essential boundary conditions for elasticity
                //                                                                           comparing wtih elasticity and without, the completely zero contribution does not happen without elasticity, never, and even the zero columns are pretty rare.

                if (total_num_columns == 0)
                    svals.SetSize(0);
                else
                    xpack_svd_dense_arr(received_mats[mis], row_size, lsvects, svals);
                if (svals.Size() == 0) // we trim (near) zeros out of svals, this means all svals == 0
                {
                    SA_PRINTF("WARNING: completely zero contribution on mis %d!\n", mis);
                    SA_PRINTF("WARNING: dim = %d, row_size = %d\n", dim, row_size);
                    tent_int_struct->mis_numcoarsedof[mis] = 0;
                    tent_int_struct->mis_tent_interps[mis]->SetSize(dim, 0); // this makes future assertions and communications cleaner, but is mostly unnecessary
                    delete [] received_mats[mis];
                    continue; // TODO: remove this, refactor 
                }
                xpack_orth_set(lsvects, svals, *tent_int_struct->mis_tent_interps[mis], eps);
                // SA_PRINTF("    mis %d of size %d: after SVD, have chosen %d independent vectors out of possible %d\n", mis,dim,tent_int_struct->mis_tent_interps[mis]->Width(), svals.Size());
            }
            if (agg_part_rels.testmesh)
            {
                std::stringstream filename;
                filename << "mis_tent_interp_" << mis << "." << PROC_RANK << ".densemat";
                std::ofstream out(filename.str().c_str());
                tent_int_struct->mis_tent_interps[mis]->Print(out);
            }
            int filled_cols = tent_int_struct->filled_cols;

            contrib_tent_insert_simple(tent_int_struct, agg_part_rels,
                                       *tent_int_struct->mis_tent_interps[mis], 
                                       agg_part_rels.mis_to_dof->GetRow(mis));

            filled_cols = tent_int_struct->filled_cols - filled_cols;
            // SA_PRINTF("    MIS %d producing %d cols\n", mis, filled_cols);

            SA_ASSERT(filled_cols == tent_int_struct->mis_tent_interps[mis]->Width());
            if (scaling_P && filled_cols > 0) 
            {
                Vector x(tent_int_struct->mis_tent_interps[mis]->Width());  // size of coarse dofs for this MIS
                Vector b(tent_int_struct->mis_tent_interps[mis]->Height());
                b = 1.0;
                xpack_solve_lls(*tent_int_struct->mis_tent_interps[mis],b,x);
                double norm = 0.0;
                for (int k=0; k<x.Size(); ++k)
                    norm += x(k)*x(k);
                norm = std::sqrt(norm);
                // we can append because the coarse DOF are numbered in exactly this order, by MIS
                for (int k=0; k<x.Size(); ++k)
                    tent_int_struct->local_coarse_one_representation->Append(x(k) / norm);
            }
            tent_int_struct->mis_numcoarsedof[mis] = filled_cols;
            num_coarse_dofs += filled_cols;
            delete [] received_mats[mis];
        }
        else
        {
            tent_int_struct->mis_numcoarsedof[mis] = 0; 
            tent_int_struct->mis_tent_interps[mis] = new DenseMatrix; // we only do this so deletion is cleaner at the end, could avoid it at the cost of more ifs
            // tent_int_struct->mis_tent_interps[mis]->SetSize(0,0); // ???
        }
    }
    delete [] received_mats;

    tent_int_struct->coarse_truedof_offset = 0;
    MPI_Scan(&num_coarse_dofs,&tent_int_struct->coarse_truedof_offset,1,MPI_INT,MPI_SUM,PROC_COMM);
    tent_int_struct->coarse_truedof_offset -= num_coarse_dofs;
    SA_RPRINTF(PROC_NUM-1,"coarse_truedof_offset = %d\n",tent_int_struct->coarse_truedof_offset);
}
