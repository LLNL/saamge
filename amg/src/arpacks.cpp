/*! \file

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

#include "common.hpp"
#if SAAMGE_USE_ARPACK
#include "arpacks.hpp"
#include <mfem.hpp>

#include <argsym.h> // does not use superlu...
// #include <arlsmat.h> // uses superlu
// #include <arlgsym.h> // uses superlu

namespace saamge
{
using namespace mfem;

/* Classes */

/*! \brief Matrix multiplication class for eigenproblem with DIAGONAL r.h.s.
           matrix: Ax = lambda Dx.
*/
class arpacks_diag_rhs
{
public:
    /*! \brief The constructor.

        \param Aa (IN) The l.h.s. matrix in the generalized eigenproblem.
        \param Da (IN) The DIAGONAL r.h.s. matrix in the generalized
                       eigenproblem.
    */
    arpacks_diag_rhs(const SparseMatrix& Aa, const SparseMatrix& Da) :
        n(Aa.Size()), A(&Aa), data(Da.GetData())
    {
        SA_ASSERT(Da.Width() == Da.Size());
        SA_ASSERT(Aa.Width() == Aa.Size());
        SA_ASSERT(Da.Width() == Aa.Width());
        SA_ASSERT(Da.NumNonZeroElems() == Da.Size());
#if (SA_IS_DEBUG_LEVEL(3))
        for (int i=0; i < Da.NumNonZeroElems(); ++i)
        {
            SA_ASSERT(Da.GetI()[i] == i);
            SA_ASSERT(Da.GetJ()[i] == i);
        }
#endif

        inv_data = new double[n];
        for (int i=0; i < n; ++i)
        {
            SA_ASSERT(data[i] > 0.);
            inv_data[i] = 1./data[i];
        }
    }

    /*! \brief The destructor.
    */
    ~arpacks_diag_rhs()
    {
        delete [] inv_data;
    }

    /*! \brief Computes out = D in.

        \param in (IN) See the description.
        \param out (IN/OUT) See the description.
    */
    void MultB(double *in, double *out);

    /*! \brief Computes out = D^{-1}A in.

        \param in (IN) See the description.
        \param out (IN/OUT) See the description.
    */
    void MultOP(double *in, double *out);

private:
    const int n; /*!< Problem size. */
    const SparseMatrix * const A; /*!< Matrix A in Ax = lambda Dx. */
    const double * const data; /*!< Diagonal of matrix D in Ax = lambda Dx. */
    double *inv_data; /*!< Diagonal of matrix D^{-1} in Ax = lambda Dx. */
};

/* Static Functions */

/*! \brief Converts a symmetric CSR sparse matrix to CSC required by ARPACK++.

    This is to guarantee certain SuperLU symmetric sparse matrix class
    requirements in the ARPACK++. Note that this corresponds to the UPPER
    TRIANGULAR part of the matrix as expected by ARPACK++.

    \param A (IN) The symmetric sparse matrix to be converted.

    \returns The converted finalized sparse matrix. It must be freed by the
             caller.

    \warning The returned sparse matrix must be freed by the caller.
*/
static inline
SparseMatrix *arpacks_convert_sym_to_lucsc(const SparseMatrix& A)
{
    SA_ASSERT(const_cast<SparseMatrix&>(A).Finalized());
    SA_ASSERT(A.Size() == A.Width());
    const int n = A.Size();
    const int * const I = A.GetI();
    const int * const J = A.GetJ();
    const double * const Data = A.GetData();
    int i, j, k, nnz;

    int * const newI = new int[n+1];
    for (nnz=i=0; i < n; ++i)
    {
        newI[i] = nnz;
        for (j = I[i]; j < I[i+1]; ++j)
        {
            if (J[j] <= i)
                ++nnz;
        }
    }
    newI[n] = nnz;
    int * const newJ = new int[nnz];
    double * const newData = new double[nnz];

    Pair<int, int> *pairs = new Pair<int, int>[nnz];
    for (i=0; i < n; ++i)
    {
        k = 0;
        for (j = I[i]; j < I[i+1]; ++j)
        {
            SA_ASSERT(k < nnz);
            pairs[k].one = J[j];
            pairs[k].two = k;
            ++k;
        }
        SA_ASSERT(k >= 0);
        SA_ASSERT(I[i+1] - I[i] == k);

        if (k <= 0)
            continue;

        if (k > 1)
            SortPairs<int, int>(pairs, k);
        int newJbeg = newI[i];
        int Jbeg = I[i];
        int pos = 0;
        for (j=0; j < k; ++j)
        {
            int column = pairs[j].one;
            if (column > i)
                break;
            pos = newJbeg + j;
            SA_ASSERT(pos < newI[i+1]);
            newJ[pos] = column;
            SA_ASSERT(Jbeg + pairs[j].two < I[i+1]);
            newData[pos] = Data[Jbeg + pairs[j].two];
        }
        SA_ASSERT(pos + 1 == newI[i+1]);
    }
    delete [] pairs;

    return (new SparseMatrix(newI, newJ, newData, n, n));
}

/* Functions */

void arpacks_diag_rhs::MultB(double *in, double *out)
{
    SA_ASSERT(in);
    SA_ASSERT(out);

    for (int i=0; i < n; ++i)
    {
        out[i] = in[i] * data[i];
    }
}

void arpacks_diag_rhs::MultOP(double *in, double *out)
{
    SA_ASSERT(in);
    SA_ASSERT(out);

    Vector vin(in, n), vout(out, n);

    A->Mult(vin, vout);
    vin = vout;
    SA_ASSERT(vin.GetData() == in);
    for (int i=0; i < n; ++i)
    {
        out[i] *= inv_data[i];
    }
}

// this is the one we use in general...
int arpacks_calc_portion_eigens_sparse_diag(const SparseMatrix& Ain,
                                            Vector& evals,
                                            DenseMatrix& evects,
                                            const SparseMatrix& Bin,
                                            int num_evects, bool lower/*=true*/,
                                            int max_iters/*=0*/, int ncv/*=0*/,
                                            int tol/*=0.*/)
{
    SA_ASSERT(num_evects > 0);
    SA_ASSERT(num_evects <= Ain.Size());

    arpacks_diag_rhs matrices(Ain, Bin);
    ARSymGenEig<double, arpacks_diag_rhs, arpacks_diag_rhs>
        eigprob(Ain.Size(), num_evects, &matrices, &arpacks_diag_rhs::MultOP,
                &matrices, &arpacks_diag_rhs::MultB, (lower?"SM":"LM"), ncv,
                tol, max_iters);
    evects.SetSize(Ain.Size(), num_evects);
    evals.SetSize(num_evects);
    double *evects_data = evects.Data();
    double *evals_data = evals.GetData();
    eigprob.EigenValVectors(evects_data, evals_data, false);
    SA_ASSERT(evals.GetData() == evals_data);
    SA_ASSERT(evects.Data() == evects_data);

    const int converged = eigprob.ConvergedEigenvalues();

    SA_ALERT_COND_MSG(converged == num_evects,
                      "The number of converged eigenvalues (%d) does NOT match"
                      " the requested number (%d)!", converged, num_evects);
    SA_ALERT_COND_MSG(eigprob.GetMaxit() > eigprob.GetIter(),
                      "The number of iterations (%d) reached the maximal number"
                      " of iterations (%d)!", eigprob.GetIter(),
                                              eigprob.GetMaxit());
    if (SA_IS_OUTPUT_LEVEL(9))
    {
        SA_PRINTF("Eigenproblem size: %d\n", eigprob.GetN());
        SA_PRINTF("Requested evecs: %d, converged evecs: %d, number of Arnoldi"
                  " vectors: %d\n", num_evects, converged, eigprob.GetNcv());
        SA_PRINTF("Maximal iterations: %d, performed iterations: %d, tolerance:"
                  " %g\n", eigprob.GetMaxit(), eigprob.GetIter(),
                           eigprob.GetTol());
        SA_PRINTF("%s","calc_portion_eigens_sparse_diag: Converged Evals = [ ");
        int i;
        for (i=0; i < converged && i < num_evects; ++i)
            SA_PRINTF_NOTS("%g ", evals(i));
        SA_PRINTF_NOTS("]\n");
        if (num_evects > converged)
        {
            SA_PRINTF("%s","Remaining \"Evals\" = [ ");
            for (; i < num_evects; ++i)
                SA_PRINTF_NOTS("%g ", evals(i));
            SA_PRINTF_NOTS("%s","]\n");
        }
    }

    if (converged < num_evects)
    {
        SA_ASSERT(converged > 0);
        evals.SetSize(converged);
        evects.UseExternalData(evects.Data(), evects.Height(), converged);
    }

    return converged;
}

}

#endif // SAAMGE_USE_ARPACK
