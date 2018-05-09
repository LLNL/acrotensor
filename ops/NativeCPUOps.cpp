//Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at the Lawrence Livermore National Laboratory
//Written by Aaron Fisher (fisher47@llnl.gov). LLNL-CODE-738419.
//All rights reserved.
//This file is part of Acrotensor. For details, see https://github.com/LLNL/acrotensor.

#include "NativeCPUOps.hpp"

namespace acro
{

void NativeCPUOps::BatchMatrixInverse(Tensor &Ainv, Tensor &A)
{
    int rank = A.GetRank();
    int mdim = A.GetDim(rank-1);
    int stride = mdim*mdim;
    int num_batch = A.GetSize() / stride;
    double *A_ptr = A.GetData();
    double *Ainv_ptr = Ainv.GetData();
    if (mdim == 1)
    {
        for (int i = 0; i < num_batch; ++i)
        {
            Inv1x1(Ainv_ptr, A_ptr, Det1x1(A_ptr));
            Ainv_ptr += stride;
            A_ptr += stride;
        }
    }
    else if (mdim == 2)
    {
        for (int i = 0; i < num_batch; ++i)
        {
            Inv2x2(Ainv_ptr, A_ptr, Det2x2(A_ptr));
            Ainv_ptr += stride;
            A_ptr += stride;
        }        
    }
    else if (mdim == 3)
    {
        for (int i = 0; i < num_batch; ++i)
        {
            Inv3x3(Ainv_ptr, A_ptr, Det3x3(A_ptr));
            Ainv_ptr += stride;
            A_ptr += stride;
        }
    }
}


void NativeCPUOps::BatchMatrixDet(Tensor &Adet, Tensor &A)
{
    int rank = A.GetRank();
    int mdim = A.GetDim(rank-1);
    int stride = mdim*mdim;
    int num_batch = A.GetSize() / stride;
    double *A_ptr = A.GetData();
    double *Adet_ptr = Adet.GetData();
    if (mdim == 1)
    {
        for (int i = 0; i < num_batch; ++i)
        {
            Adet_ptr[i] = Det1x1(A_ptr);
            A_ptr += stride;
        }
    }
    else if (mdim == 2)
    {
        for (int i = 0; i < num_batch; ++i)
        {
            Adet_ptr[i] = Det2x2(A_ptr);
            A_ptr += stride;
        }        
    }
    else if (mdim == 3)
    {
        for (int i = 0; i < num_batch; ++i)
        {
            Adet_ptr[i] = Det3x3(A_ptr);
            A_ptr += stride;
        }
    }
}


void NativeCPUOps::BatchMatrixInvDet(Tensor &Ainv, Tensor &Adet, Tensor &A)
{
    int rank = A.GetRank();
    int mdim = A.GetDim(rank-1);
    int stride = mdim*mdim;
    int num_batch = A.GetSize() / stride;
    double *A_ptr = A.GetData();
    double *Ainv_ptr = Ainv.GetData();
    double *Adet_ptr = Adet.GetData();
    if (mdim == 1)
    {
        for (int i = 0; i < num_batch; ++i)
        {
            Adet_ptr[i] = Det1x1(A_ptr);
            Inv1x1(Ainv_ptr, A_ptr, Adet_ptr[i]);
            A_ptr += stride;
            Ainv_ptr += stride;
        }
    }
    else if (mdim == 2)
    {
        for (int i = 0; i < num_batch; ++i)
        {
            Adet_ptr[i] = Det2x2(A_ptr);
            Inv2x2(Ainv_ptr, A_ptr, Adet_ptr[i]);
            A_ptr += stride;
            Ainv_ptr += stride;
        }        
    }
    else if (mdim == 3)
    {
        for (int i = 0; i < num_batch; ++i)
        {
            Adet_ptr[i] = Det3x3(A_ptr);
            Inv3x3(Ainv_ptr, A_ptr, Adet_ptr[i]);
            A_ptr += stride;
            Ainv_ptr += stride;
        }
    }
}


void NativeCPUOps::FlatIndexedScatter(Tensor &Aout, Tensor &Ain, IndexMapping &M)
{
    IndexVector &I = M.GetMap();
    for (int i = 0; i < I.GetSize(); i++)
    {
        Aout[i] = Ain[I[i]];
    }
}


void NativeCPUOps::FlatIndexedSumGather(Tensor &Aout, Tensor &Ain, IndexMapping &M)
{
    IndexVector &I = M.GetMap();
    Aout.Set(0.0);
    for (int i = 0; i < I.GetSize(); i++)
    {
        Aout[I[i]] += Ain[i];
    }
}




}