//Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at the Lawrence Livermore National Laboratory
//Written by Aaron Fisher (fisher47@llnl.gov). LLNL-CODE-738419.
//All rights reserved.
//This file is part of Acrotensor. For details, see https://github.com/LLNL/acrotensor.

#include "IndexMapping.hpp"
#include <algorithm>
#include <numeric>

namespace acro
{


IndexMapping::IndexMapping(int domain_size, int range_size) :
    DomainSize(domain_size),
    RangeSize(range_size),
    InverseComputed(false),
    M(range_size),
    InvM(range_size),
    InvMOff(domain_size+1)
{

}



void IndexMapping::ComputeInverse()
{
    ACROBATIC_ASSERT(!InverseComputed,"Can't compute the inverse mapping twice.");

    std::iota(&InvM[0], &InvM[RangeSize], 0);
    std::stable_sort(&InvM[0], &InvM[RangeSize],
       [this](size_t i1, size_t i2) {return M[i1] < M[i2];});

    int off = 0;
    for (int i = 0; i < DomainSize + 1; ++i)
    {
        InvMOff[i] = off;
        if (off < RangeSize)
        {
            int m = M[InvM[off]];
            while (off < RangeSize && M[InvM[off]] == m)
            {
                off ++;
            }
        }
        else
        {
            off = RangeSize;    //Handle the last one
        }
    }

    InverseComputed = true;

    if (OnGPU)
    {
        InvM.SwitchFromGPU();
        InvM.MoveToGPU();
        InvMOff.SwitchFromGPU();
        InvMOff.MoveToGPU();
    }
}


void IndexMapping::MapToGPU()
{
    M.MapToGPU();
    if (InverseComputed)
    {
        InvM.MapToGPU();
        InvMOff.MapToGPU();
    }
    MappedToGPU = true;
}


void IndexMapping::MoveToGPU()
{
    M.MoveToGPU();
    if (InverseComputed)
    {
        InvM.MoveToGPU();
        InvMOff.MoveToGPU();
    }
    OnGPU = true;
}


void IndexMapping::SwitchToGPU()
{
    M.SwitchToGPU();
    if (InverseComputed)
    {
        InvM.SwitchToGPU();
        InvMOff.SwitchToGPU();
    }
    OnGPU = true;
}


void IndexMapping::UnmapFromGPU()
{
    M.UnmapFromGPU();
    if (InverseComputed)
    {
        InvM.UnmapFromGPU();
        InvMOff.UnmapFromGPU();
    }
    MappedToGPU = false;
    OnGPU = false;
}


void IndexMapping::MoveFromGPU()
{
    M.MoveFromGPU();
    if (InverseComputed)
    {
        InvM.MoveFromGPU();
        InvMOff.MoveFromGPU();
    }
    OnGPU = false;
}


void IndexMapping::SwitchFromGPU()
{
    M.SwitchFromGPU();
    if (InverseComputed)
    {
        InvM.SwitchFromGPU();
        InvMOff.SwitchFromGPU();
    }
    OnGPU = false;
}




}
