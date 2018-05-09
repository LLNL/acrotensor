//Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at the Lawrence Livermore National Laboratory
//Written by Aaron Fisher (fisher47@llnl.gov). LLNL-CODE-738419.
//All rights reserved.
//This file is part of Acrotensor. For details, see https://github.com/LLNL/acrotensor.

#ifndef ACROBATIC_INDEXMAPPING_HPP
#define ACROBATIC_INDEXMAPPING_HPP

#include "IndexVector.hpp"

namespace acro
{

class IndexMapping
{
    public:
    IndexMapping(int domain_size, int range_size);

    int GetDomainSize() {return DomainSize;}
    int GetRangeSize() {return RangeSize;}
    bool IsInverseComputed() {return InverseComputed;}

    int &operator[](int raw_index);
    void ComputeInverse();

    IndexVector &GetMap();
    IndexVector &GetInvMap();
    IndexVector &GetInvMapOffsets();

    void MapToGPU();            //Allocate memory for the data on the GPU
    void MoveToGPU();           //Copy the data to the GPU and flag the data as currently on the GPU
    void SwitchToGPU();         //Flag the data as currently onGPU
    void UnmapFromGPU();        //Deallocate memory on the GPU
    void MoveFromGPU();         //Copy the data back from the GPU and flag the data as currently on the CPU
    void SwitchFromGPU();       //Flag the data as currently on the CPU
    bool IsMappedToGPU() const {return MappedToGPU;}
    bool IsOnGPU() const {return OnGPU;}

    private:
    bool InverseComputed;
    bool MappedToGPU;
    bool OnGPU;
    int DomainSize;
    int RangeSize;

    IndexVector M;
    IndexVector InvM, InvMOff;
};


inline int &IndexMapping::operator[](int raw_index) 
{
    return M[raw_index];
}


inline IndexVector &IndexMapping::GetMap()
{
    return M;
}


inline IndexVector &IndexMapping::GetInvMap()
{
    ACROBATIC_ASSERT(InverseComputed, "Trying to access inverse mapping before the inverse is computed.");
    return InvM;
}


inline IndexVector &IndexMapping::GetInvMapOffsets()
{
    ACROBATIC_ASSERT(InverseComputed, "Trying to access inverse mapping offsets before the inverse is computed.");
    return InvMOff;
}

}

#endif