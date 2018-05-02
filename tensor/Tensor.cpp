//Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at the Lawrence Livermore National Laboratory
//Written by Aaron Fisher (fisher47@llnl.gov). LLNL-CODE-738419.
//All rights reserved.
//This file is part of Acrotensor. For details, see https://github.com/LLNL/acrotensor.

#include "Tensor.hpp"
#include "Util.hpp"
#include "CudaUtil.hpp"
#include <iostream>

namespace acro
{


Tensor::Tensor()
{
    Data = nullptr;
    DeviceData = nullptr;
    OwnsData = false;
    MappedToGPU = false;
    Initialized = false;
}


Tensor::Tensor(std::vector<int> &dims, double *hdata, double *ddata, bool ongpu)
{
    Initialized = false;
    Init(dims, hdata, ddata, ongpu);
}


Tensor::Tensor(int d0, double *hdata, double *ddata, bool ongpu)
{
    Initialized = false;
    std::vector<int> dims = {d0};
    Init(dims, hdata, ddata, ongpu);
}


Tensor::Tensor(int d0, int d1, double *hdata, double *ddata, bool ongpu)
{
    Initialized = false;
    std::vector<int> dims = {d0, d1};
    Init(dims, hdata, ddata, ongpu);
}


Tensor::Tensor(int d0, int d1, int d2, double *hdata, double *ddata, bool ongpu)
{
    Initialized = false;
    std::vector<int> dims = {d0, d1, d2};
    Init(dims, hdata, ddata, ongpu);
}


Tensor::Tensor(int d0, int d1, int d2, int d3, double *hdata, double *ddata, bool ongpu)
{
    Initialized = false;
    std::vector<int> dims = {d0, d1, d2, d3};
    Init(dims, hdata, ddata, ongpu);
}


Tensor::Tensor(int d0, int d1, int d2, int d3, int d4, double *hdata, double *ddata, bool ongpu)
{
    Initialized = false;
    std::vector<int> dims = {d0, d1, d2, d3, d4};
    Init(dims, hdata, ddata, ongpu);
}


Tensor::Tensor(int d0, int d1, int d2, int d3, int d4, int d5, double *hdata, double *ddata, bool ongpu)
{
    Initialized = false;
    std::vector<int> dims = {d0, d1, d2, d3, d4, d5};
    Init(dims, hdata, ddata, ongpu);
}


Tensor::Tensor(int d0, int d1, int d2, int d3, int d4, int d5, int d6, double *hdata, double *ddata, bool ongpu)
{
    Initialized = false;
    std::vector<int> dims = {d0, d1, d2, d3, d4, d5, d6};
    Init(dims, hdata, ddata, ongpu);
}


Tensor::Tensor(int d0, int d1, int d2, int d3, int d4, int d5, int d6, int d7, double *hdata, double *ddata, bool ongpu)
{
    Initialized = false;
    std::vector<int> dims = {d0, d1, d2, d3, d4, d5, d6, d7};
    Init(dims, hdata, ddata, ongpu);
}


Tensor::Tensor(int d0, int d1, int d2, int d3, int d4, int d5, int d6, int d7, int d8, double *hdata, double *ddata, bool ongpu)
{
    Initialized = false;
    Init(d0, d1, d2, d3, d4, d5, d6, d7, d8, hdata, ddata, ongpu);
}


void Tensor::Init(int d0, double *hdata, double *ddata, bool ongpu)
{
    std::vector<int> dims = {d0};
    Init(dims, hdata, ddata, ongpu);
}


void Tensor::Init(int d0, int d1, double *hdata, double *ddata, bool ongpu)
{
    std::vector<int> dims = {d0, d1};
    Init(dims, hdata, ddata, ongpu);
}


void Tensor::Init(int d0, int d1, int d2, double *hdata, double *ddata, bool ongpu)
{
    std::vector<int> dims = {d0, d1, d2};
    Init(dims, hdata, ddata, ongpu);
}


void Tensor::Init(int d0, int d1, int d2, int d3, double *hdata, double *ddata, bool ongpu)
{
    std::vector<int> dims = {d0, d1, d2, d3};
    Init(dims, hdata, ddata, ongpu);
}


void Tensor::Init(int d0, int d1, int d2, int d3, int d4, double *hdata, double *ddata, bool ongpu)
{
    std::vector<int> dims = {d0, d1, d2, d3, d4};
    Init(dims, hdata, ddata, ongpu);
}


void Tensor::Init(int d0, int d1, int d2, int d3, int d4, int d5, double *hdata, double *ddata, bool ongpu)
{
    std::vector<int> dims = {d0, d1, d2, d3, d4, d5};
    Init(dims, hdata, ddata, ongpu);
}


void Tensor::Init(int d0, int d1, int d2, int d3, int d4, int d5, int d6, double *hdata, double *ddata, bool ongpu)
{
    std::vector<int> dims = {d0, d1, d2, d3, d4, d5, d6};
    Init(dims, hdata, ddata, ongpu);
}


void Tensor::Init(int d0, int d1, int d2, int d3, int d4, int d5, int d6, int d7, double *hdata, double *ddata, bool ongpu)
{
    std::vector<int> dims = {d0, d1, d2, d3, d4, d5, d6, d7};
    Init(dims, hdata, ddata, ongpu);
}


void Tensor::Init(int d0, int d1, int d2, int d3, int d4, int d5, int d6, int d7, int d8, double *hdata, double *ddata, bool ongpu)
{
    std::vector<int> dims = {d0, d1, d2, d3, d4, d5, d6, d7, d8};
    Init(dims, hdata, ddata, ongpu);
}


void Tensor::Init(std::vector<int> &dims, double *hdata, double *ddata, bool ongpu)
{
    ACROBATIC_ASSERT(!IsInitialized(), "Can't initilize a tensor a second time.")
    ACROBATIC_ASSERT(dims.size() > 0, "Cant initilize tensor without any dimensions.");
    for (int d = 0; d < dims.size(); ++d)
    {
        ACROBATIC_ASSERT(dims[d] > 0, "Can't initilize tensor with non-positive dimensions.");
    }
    Dims = dims;
    UpdateStrides();
    ComputeSize();
    if (hdata == nullptr)
    {
        Data = new double[Size];
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


Tensor::~Tensor()
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

void Tensor::Reshape(std::vector<int> &dims)
{
    ACROBATIC_ASSERT(dims.size() > 0);
    for (int d = 0; d < dims.size(); ++d)
    {
        ACROBATIC_ASSERT(dims[d] > 0);
    }

    int new_size = 1;
    for (int d = 0; d < dims.size(); ++d)
    {
        new_size *= dims[d];
    }
    ACROBATIC_ASSERT(new_size == Size);

    Dims = dims;
    UpdateStrides();
}


void Tensor::Reshape(int d0)
{
    std::vector<int> dims = {d0};
    Reshape(dims);
}


void Tensor::Reshape(int d0, int d1)
{
    std::vector<int> dims = {d0, d1};
    Reshape(dims);
}


void Tensor::Reshape(int d0, int d1, int d2)
{
    std::vector<int> dims = {d0, d1, d2};
    Reshape(dims);
}


void Tensor::Reshape(int d0, int d1, int d2, int d3)
{
    std::vector<int> dims = {d0, d1, d2, d3};
    Reshape(dims);
}


void Tensor::Reshape(int d0, int d1, int d2, int d3, int d4)
{
    std::vector<int> dims = {d0, d1, d2, d3, d4};
    Reshape(dims);
}


void Tensor::Reshape(int d0, int d1, int d2, int d3, int d4, int d5)
{
    std::vector<int> dims = {d0, d1, d2, d3, d4, d5};
    Reshape(dims);
}


void Tensor::Reshape(int d0, int d1, int d2, int d3, int d4, int d5, int d6)
{
    std::vector<int> dims = {d0, d1, d2, d3, d4, d5, d6};
    Reshape(dims);
}


void Tensor::Reshape(int d0, int d1, int d2, int d3, int d4, int d5, int d6, int d7)
{
    std::vector<int> dims = {d0, d1, d2, d3, d4, d5, d6, d7};
    Reshape(dims);
}


void Tensor::Reshape(int d0, int d1, int d2, int d3, int d4, int d5, int d6, int d7, int d8)
{
    std::vector<int> dims = {d0, d1, d2, d3, d4, d5, d6, d7, d8};
    Reshape(dims);
}


void Tensor::Retarget(double *hdata, double *ddata)
{
    ACROBATIC_ASSERT(!OwnsData);
    Data = hdata;
    DeviceData = ddata;
}


void Tensor::UpdateStrides()
{
    Strides.resize(Dims.size());
    int stride = 1;
    for (int d = Dims.size() - 1; d >= 0; --d)
    {
        Strides[d] = stride;
        stride *= Dims[d];
    }
}


void Tensor::ComputeSize()
{
    Size = 1;
    for (int d = 0; d < GetRank(); ++d)
    {
        Size *= Dims[d];
    }
    ByteSize = Size*sizeof(double);
}

void Tensor::Set(double val)
{
    if (!IsOnGPU())
    {
        for (int i = 0; i < GetSize(); ++i)
        {
            Data[i] = val;
        }
    }
    else
    {
#ifdef ACRO_HAVE_CUDA
        ensureCudaContext();
        CudaSet<<<Size/512+1,512>>>(DeviceData, val, GetSize());
        acroCudaErrorCheck(cudaPeekAtLastError());
#endif
    }
}


void Tensor::Mult(double c)
{
    if (!IsOnGPU())
    {
        for (int i = 0; i < GetSize(); ++i)
        {
            Data[i] *= c;
        }
    }
    else
    {
#ifdef ACRO_HAVE_CUDA
        ensureCudaContext();
        CudaMult<<<Size/512+1,512>>>(DeviceData, c, GetSize());
        acroCudaErrorCheck(cudaPeekAtLastError());
#endif
    }
}


void Tensor::MapToGPU()
{
#ifdef ACRO_HAVE_CUDA
    ACROBATIC_ASSERT(!IsMappedToGPU(), "Trying to map data to the GPU a second time.");
    ensureCudaContext();
    acroCudaErrorCheck(cudaMalloc((void**)&DeviceData, ByteSize));
    MappedToGPU = true;
#endif
}

void Tensor::MoveToGPU()
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

void Tensor::SwitchToGPU()
{
#ifdef ACRO_HAVE_CUDA
    if (!IsMappedToGPU())
    {
        MapToGPU();
    }
    OnGPU = true;
#endif
}

void Tensor::UnmapFromGPU()
{
#ifdef ACRO_HAVE_CUDA    
    ACROBATIC_ASSERT(IsMappedToGPU(), "Can't unmap data that is not mapped to the GPU.");
    ensureCudaContext();
    acroCudaErrorCheck(cudaFree(DeviceData));
    MappedToGPU = false;
    OnGPU = false;
#endif
}

void Tensor::MoveFromGPU()
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


void Tensor::SwitchFromGPU()
{
#ifdef ACRO_HAVE_CUDA
    OnGPU = false;
#endif
}


void Tensor::Print()
{
    std::cout << "Dims:  ";
    for (int d = 0; d < Dims.size(); ++d)
    {
        std::cout << Dims[d] << "  ";
    }
    std::cout << std::endl;

    std::cout << "Strides:  ";
    for (int d = 0; d < Dims.size(); ++d)
    {
        std::cout << Strides[d] << "  ";
    }
    std::cout << std::endl;

    for (int i = 0; i < GetSize(); ++i)
    {
        std::cout << Data[i] << std::endl;
    }
    std::cout << std::endl;
}

}
