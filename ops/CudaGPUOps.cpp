//Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at the Lawrence Livermore National Laboratory
//Written by Aaron Fisher (fisher47@llnl.gov). LLNL-CODE-738419.
//All rights reserved.
//This file is part of Acrotensor. For details, see https://github.com/LLNL/acrotensor.

#ifdef ACRO_HAVE_CUDA

#include "CudaGPUOps.hpp"

namespace acro
{

void CudaGPUOps::BatchMatrixInverse(Tensor &Ainv, Tensor &A)
{
    int rank = A.GetRank();
    int mdim = A.GetDim(rank-1);
    int stride = mdim*mdim;
    int num_batch = A.GetSize() / stride;
    double *A_ptr = A.GetDeviceData();
    double *Ainv_ptr = Ainv.GetDeviceData();
    if (mdim == 1)
    {
        CudaInv1x1<<<num_batch/128+1,128>>>(Ainv_ptr, A_ptr, num_batch);
    }
    else if (mdim == 2)
    {
        CudaInv2x2<<<num_batch/128+1,128>>>(Ainv_ptr, A_ptr, num_batch);
    }
    else if (mdim == 3)
    {
        CudaInv3x3<<<num_batch/128+1,128>>>(Ainv_ptr, A_ptr, num_batch);
    }
}


void CudaGPUOps::BatchMatrixDet(Tensor &Adet, Tensor &A)
{
    int rank = A.GetRank();
    int mdim = A.GetDim(rank-1);
    int stride = mdim*mdim;
    int num_batch = A.GetSize() / stride;
    double *A_ptr = A.GetDeviceData();
    double *Adet_ptr = Adet.GetDeviceData();
    if (mdim == 1)
    {
        CudaDet1x1<<<num_batch/128+1,128>>>(Adet_ptr, A_ptr, num_batch);
    }
    else if (mdim == 2)
    {
        CudaDet2x2<<<num_batch/128+1,128>>>(Adet_ptr, A_ptr, num_batch);
    }
    else if (mdim == 3)
    {
        CudaDet3x3<<<num_batch/128+1,128>>>(Adet_ptr, A_ptr, num_batch);
    }
}


void CudaGPUOps::BatchMatrixInvDet(Tensor &Ainv, Tensor &Adet, Tensor &A)
{
    int rank = A.GetRank();
    int mdim = A.GetDim(rank-1);
    int stride = mdim*mdim;
    int num_batch = A.GetSize() / stride;
    double *A_ptr = A.GetDeviceData();
    double *Ainv_ptr = Ainv.GetDeviceData();    
    double *Adet_ptr = Adet.GetDeviceData();
    if (mdim == 1)
    {
        CudaInvDet1x1<<<num_batch/128+1,128>>>(Ainv_ptr, Adet_ptr, A_ptr, num_batch);
    }
    else if (mdim == 2)
    {
        CudaInvDet2x2<<<num_batch/128+1,128>>>(Ainv_ptr, Adet_ptr, A_ptr, num_batch);
    }
    else if (mdim == 3)
    {
        CudaInvDet3x3<<<num_batch/128+1,128>>>(Ainv_ptr, Adet_ptr, A_ptr, num_batch);
    }
}


void CudaGPUOps::FlatIndexedScatter(Tensor &Aout, Tensor &Ain, IndexMapping &M)
{
    double *Aout_ptr = Aout.GetDeviceData();
    double *Ain_ptr = Ain.GetDeviceData();
    int *M_ptr = M.GetMap().GetDeviceData();
    int *InvM_ptr = M.GetInvMap().GetDeviceData();
    int *InvMOff_ptr = M.GetInvMapOffsets().GetDeviceData();
    int N = M.GetRangeSize();
    CudaScatter<<<N/128+1,128>>>(Aout_ptr, Ain_ptr, M_ptr, InvM_ptr, InvMOff_ptr, N);
}


void CudaGPUOps::FlatIndexedSumGather(Tensor &Aout, Tensor &Ain, IndexMapping &M)
{
    double *Aout_ptr = Aout.GetDeviceData();
    double *Ain_ptr = Ain.GetDeviceData();
    int *M_ptr = M.GetMap().GetDeviceData();
    int *InvM_ptr = M.GetInvMap().GetDeviceData();
    int *InvMOff_ptr = M.GetInvMapOffsets().GetDeviceData();
    int N = M.GetDomainSize();

    CudaSumGather<<<N/128+1,128>>>(Aout_ptr, Ain_ptr, M_ptr, InvM_ptr, InvMOff_ptr, N);
}


__global__ void CudaInv1x1(double *Ainv, double *A, int N)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx < N)
    {
        Ainv[idx] = 1.0 / A[idx];
    }
}


__global__ void CudaInv2x2(double *Ainv, double *A, int N)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx < N)
    {
        int b = idx*4;
        double A0 = A[b];
        double A1 = A[b+1];
        double A2 = A[b+2];
        double A3 = A[b+3];
        double invdet = 1.0 / (A0*A3 - A1*A2);
        Ainv[b+0] = invdet*A3;
        Ainv[b+1] = -invdet*A1;
        Ainv[b+2] = -invdet*A2;
        Ainv[b+3] = invdet*A0;
    }
}


__global__ void CudaInv3x3(double *Ainv, double *A, int N)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx < N)
    {
        int b = idx*9;
        double A0 = A[b];
        double A1 = A[b+1];
        double A2 = A[b+2];
        double A3 = A[b+3];
        double A4 = A[b+4];
        double A5 = A[b+5];
        double A6 = A[b+6];
        double A7 = A[b+7];
        double A8 = A[b+8];
        double invdet = 1.0 / (A0*A4*A8 + A1*A5*A6 + A2*A3*A7 
                             - A6*A4*A2 - A7*A5*A0 - A8*A3*A1);
        Ainv[b+0] = invdet*(A4*A8 - A5*A7);
        Ainv[b+1] = invdet*(A5*A6 - A3*A8);
        Ainv[b+2] = invdet*(A3*A7 - A4*A6);
        Ainv[b+3] = invdet*(A2*A7 - A1*A8);
        Ainv[b+4] = invdet*(A0*A8 - A2*A6);
        Ainv[b+5] = invdet*(A1*A6 - A0*A7);
        Ainv[b+6] = invdet*(A1*A5 - A2*A4);
        Ainv[b+7] = invdet*(A2*A3 - A0*A5);
        Ainv[b+8] = invdet*(A0*A4 - A1*A3);        
    }
}


__global__ void CudaDet1x1(double *Adet, double *A, int N)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx < N)
    {
        Adet[idx] = A[idx];
    }
}


__global__ void CudaDet2x2(double *Adet, double *A, int N)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx < N)
    {
        int b = idx*4;
        double A0 = A[b];
        double A1 = A[b+1];
        double A2 = A[b+2];
        double A3 = A[b+3];
        Adet[idx] = (A0*A3 - A1*A2);
    }
}


__global__ void CudaDet3x3(double *Adet, double *A, int N)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx < N)
    {
        int b = idx*9;
        double A0 = A[b];
        double A1 = A[b+1];
        double A2 = A[b+2];
        double A3 = A[b+3];
        double A4 = A[b+4];
        double A5 = A[b+5];
        double A6 = A[b+6];
        double A7 = A[b+7];
        double A8 = A[b+8];
        Adet[idx] = (A0*A4*A8 + A1*A5*A6 + A2*A3*A7 
                   - A6*A4*A2 - A7*A5*A0 - A8*A3*A1);       
    }
}


__global__ void CudaInvDet1x1(double *Ainv, double *Adet, double *A, int N)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx < N)
    {
        double det = A[idx];
        Adet[idx] = det;
        Ainv[idx] = 1.0 / det;
    }
}


__global__ void CudaInvDet2x2(double *Ainv, double *Adet, double *A, int N)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx < N)
    {
        int b = idx*4;
        double A0 = A[b];
        double A1 = A[b+1];
        double A2 = A[b+2];
        double A3 = A[b+3];
        double det = (A0*A3 - A1*A2);
        Adet[idx] = det;
        double invdet = 1.0 / det;
        Ainv[b+0] = invdet*A3;
        Ainv[b+1] = -invdet*A1;
        Ainv[b+2] = -invdet*A2;
        Ainv[b+3] = invdet*A0;
    }
}


__global__ void CudaInvDet3x3(double *Ainv, double *Adet, double *A, int N)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx < N)
    {
        int b = idx*9;
        double A0 = A[b];
        double A1 = A[b+1];
        double A2 = A[b+2];
        double A3 = A[b+3];
        double A4 = A[b+4];
        double A5 = A[b+5];
        double A6 = A[b+6];
        double A7 = A[b+7];
        double A8 = A[b+8];
        double det = (A0*A4*A8 + A1*A5*A6 + A2*A3*A7 
                    - A6*A4*A2 - A7*A5*A0 - A8*A3*A1);        
        Adet[idx] = det;
        double invdet = 1.0 / det;        
        Ainv[b+0] = invdet*(A4*A8 - A5*A7);
        Ainv[b+1] = invdet*(A5*A6 - A3*A8);
        Ainv[b+2] = invdet*(A3*A7 - A4*A6);
        Ainv[b+3] = invdet*(A2*A7 - A1*A8);
        Ainv[b+4] = invdet*(A0*A8 - A2*A6);
        Ainv[b+5] = invdet*(A1*A6 - A0*A7);
        Ainv[b+6] = invdet*(A1*A5 - A2*A4);
        Ainv[b+7] = invdet*(A2*A3 - A0*A5);
        Ainv[b+8] = invdet*(A0*A4 - A1*A3);
    }
}


__global__ void CudaScatter(double *Aout, double *Ain, int *M, int *invM, int *invMOff, int N)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < N)
    {
        Aout[i] = Ain[M[i]];
    }
}


__global__ void CudaSumGather(double *Aout, double *Ain, int *M, int *invM, int *invMOff, int N)
{
    int iout = blockIdx.x*blockDim.x + threadIdx.x;
    if (iout < N)
    {
        int in_beg = invMOff[iout];
        int in_end = invMOff[iout + 1];
        double sum = 0.0;
        for (int iin = in_beg; iin < in_end; ++iin)
        {
            sum += Ain[invM[iin]];
        }
        Aout[iout] = sum;
    }
}


}

#endif