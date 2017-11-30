//Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at the Lawrence Livermore National Laboratory
//Written by Aaron Fisher (fisher47@llnl.gov). LLNL-CODE-738419.
//All rights reserved.
//This file is part of Acrotensor. For details, see https://github.com/LLNL/acrotensor.

#include "NativeCPUOps.hpp"

namespace acro
{

void NativeCPUOps::BatchMatrixInverse(Tensor &out, Tensor &in)
{
    //Ensure the proper data is on the CPU
    if (out.IsOnGPU())
        out.SwitchFromGPU();
    if (in.IsOnGPU())
        in.MoveFromGPU();

    int rank = in.GetRank();
    int mdim = in.GetDim(rank-1);
    int stride = mdim*mdim;
    int num_batch = in.GetSize() / stride;
    double *in_ptr = in.GetData();
    double *out_ptr = out.GetData();
    if (mdim == 1)
    {
        for (int i = 0; i < num_batch; ++i)
        {
            Inv1x1(out_ptr, in_ptr);
            out_ptr += stride;
            in_ptr += stride;
        }
    }
    else if (mdim == 2)
    {
        for (int i = 0; i < num_batch; ++i)
        {
            Inv2x2(out_ptr, in_ptr);
            out_ptr += stride;
            in_ptr += stride;
        }        
    }
    else if (mdim == 3)
    {
        for (int i = 0; i < num_batch; ++i)
        {
            Inv3x3(out_ptr, in_ptr);
            out_ptr += stride;
            in_ptr += stride;
        }
    }
}

}