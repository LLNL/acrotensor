//Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at the Lawrence Livermore National Laboratory
//Written by Aaron Fisher (fisher47@llnl.gov). LLNL-CODE-738419.
//All rights reserved.
//This file is part of Acrotensor. For details, see https://github.com/LLNL/acrotensor.

#include "IndexVector.hpp"
#include "Util.hpp"
#include "CudaUtil.hpp"
#include <iostream>

namespace acro
{


IndexVector::IndexVector()
{
    Data = nullptr;
    DeviceData = nullptr;
    OwnsData = false;
    MappedToGPU = false;
    Initialized = false;
}


IndexVector::IndexVector(int dim, int *hdata, int *ddata, bool ongpu)
{
    Initialized = false;
    Init(dim, hdata, ddata, ongpu);
}


void IndexVector::Init(int dim, int *hdata, int *ddata, bool ongpu)
{
    ACROBATIC_ASSERT(!IsInitialized(), "Can't initilize a vector a second time.")
    ACROBATIC_ASSERT(dim > 0, "Cant initilize vector with dim <= 0.");
    Size = dim;
    ByteSize = dim*sizeof(int);

    if (hdata == nullptr)
    {
        Data = new int[Size];
        OwnsData = true;
    }
    else
    {
        Data = hdata;
        OwnsData = false;
    }

    MappedToGPU = false;
    DeviceData = ddata;
    if (ddata != nullptr)
    {
        ACROBATIC_ASSERT(hdata != nullptr, 
                        "Acrotensor does not currently support GPU only tensors.");
        MappedToGPU = true;
    }

    ACROBATIC_ASSERT(ddata != nullptr || !ongpu,
                     "Acrotensor cannot mark external data as on the GPU if no GPU pointer is provided.");

    OnGPU = ongpu;
    Initialized = true;
}


IndexVector::~IndexVector()
{
    if (OwnsData)
    {
        delete [] Data;
        if (IsMappedToGPU())
        {
            UnmapFromGPU();
        }
    }
}


void IndexVector::Retarget(int *hdata, int *ddata)
{
    ACROBATIC_ASSERT(!OwnsData);
    Data = hdata;
    DeviceData = ddata;
}


void IndexVector::MapToGPU()
{
#ifdef ACRO_HAVE_CUDA
    ACROBATIC_ASSERT(!IsMappedToGPU(), "Trying to map data to the GPU a second time.");
    ensureCudaContext();
    acroCudaErrorCheck(cudaMalloc((void**)&DeviceData, ByteSize));
    MappedToGPU = true;
#endif
}

void IndexVector::MoveToGPU()
{
#ifdef ACRO_HAVE_CUDA   
    if (!IsMappedToGPU())
    {
        MapToGPU();
    }
    if (!IsOnGPU())
    {
        ensureCudaContext();
        acroCudaErrorCheck(cudaMemcpy(DeviceData, Data, ByteSize, cudaMemcpyHostToDevice));
        OnGPU = true;
    }
#endif
}

void IndexVector::SwitchToGPU()
{
#ifdef ACRO_HAVE_CUDA
    if (!IsMappedToGPU())
    {
        MapToGPU();
    }
    OnGPU = true;
#endif
}

void IndexVector::UnmapFromGPU()
{
#ifdef ACRO_HAVE_CUDA    
    ACROBATIC_ASSERT(IsMappedToGPU(), "Can't unmap data that is not mapped to the GPU.");
    ensureCudaContext();
    acroCudaErrorCheck(cudaFree(DeviceData));
    MappedToGPU = false;
    OnGPU = false;
#endif
}

void IndexVector::MoveFromGPU()
{
#ifdef ACRO_HAVE_CUDA
    if (IsOnGPU())
    {
        ensureCudaContext();
        acroCudaErrorCheck(cudaMemcpy(Data, DeviceData, ByteSize, cudaMemcpyDeviceToHost));
        OnGPU = false;
    }
#endif
}


void IndexVector::SwitchFromGPU()
{
#ifdef ACRO_HAVE_CUDA
    OnGPU = false;
#endif
}


void IndexVector::Print()
{
    for (int i = 0; i < GetSize(); ++i)
    {
        std::cout << Data[i] << std::endl;
    }
    std::cout << std::endl;
}

}
