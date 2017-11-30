//Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at the Lawrence Livermore National Laboratory
//Written by Aaron Fisher (fisher47@llnl.gov). LLNL-CODE-738419.
//All rights reserved.
//This file is part of Acrotensor. For details, see https://github.com/LLNL/acrotensor.

#ifdef ACRO_HAVE_CUDA

#include "CudaGPUOps.hpp"

namespace acro
{

void CudaGPUOps::BatchMatrixInverse(Tensor &out, Tensor &in)
{
    //Ensure the proper data is on the CPU
    if (!out.IsOnGPU())
        out.SwitchToGPU();
    if (!in.IsOnGPU())
        in.MoveToGPU();

    int rank = in.GetRank();
    int mdim = in.GetDim(rank-1);
    int stride = mdim*mdim;
    int num_batch = in.GetSize() / stride;
    double *in_ptr = in.GetDeviceData();
    double *out_ptr = out.GetDeviceData();
    if (mdim == 1)
    {
        CudaInv1x1<<<num_batch/128+1,128>>>(out_ptr, in_ptr, num_batch);
    }
    else if (mdim == 2)
    {
        CudaInv2x2<<<num_batch/128+1,128>>>(out_ptr, in_ptr, num_batch);
    }
    else if (mdim == 3)
    {
        CudaInv3x3<<<num_batch/128+1,128>>>(out_ptr, in_ptr, num_batch);
    }
}


__global__ void CudaInv1x1(double *out, double *in, int N)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx < N)
    {
        out[idx] = 1.0 / in[idx];
    }
}


__global__ void CudaInv2x2(double *out, double *in, int N)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx < N)
    {
        int b = idx*4;
        double in0 = in[b];
        double in1 = in[b+1];
        double in2 = in[b+2];
        double in3 = in[b+3];
        double invdet = 1.0 / (in0*in3 - in1*in2);
        out[b+0] = invdet*in3;
        out[b+1] = -invdet*in1;
        out[b+2] = -invdet*in2;
        out[b+3] = invdet*in0;
    }
}


__global__ void CudaInv3x3(double *out, double *in, int N)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx < N)
    {
        int b = idx*9;
        double in0 = in[b];
        double in1 = in[b+1];
        double in2 = in[b+2];
        double in3 = in[b+3];
        double in4 = in[b+4];
        double in5 = in[b+5];
        double in6 = in[b+6];
        double in7 = in[b+7];
        double in8 = in[b+8];
        double invdet = 1.0 / (in0*in4*in8 + in1*in5*in6 + in2*in3*in7 
                             - in6*in4*in2 - in7*in5*in0 - in8*in3*in1);
        out[b+0] = invdet*(in4*in8 - in5*in7);
        out[b+1] = invdet*(in5*in6 - in3*in8);
        out[b+2] = invdet*(in3*in7 - in4*in6);
        out[b+3] = invdet*(in2*in7 - in1*in8);
        out[b+4] = invdet*(in0*in8 - in2*in6);
        out[b+5] = invdet*(in1*in6 - in0*in7);
        out[b+6] = invdet*(in1*in5 - in2*in4);
        out[b+7] = invdet*(in2*in3 - in0*in5);
        out[b+8] = invdet*(in0*in4 - in1*in3);        
    }
}

}
#endif