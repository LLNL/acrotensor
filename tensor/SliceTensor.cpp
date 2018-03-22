//Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at the Lawrence Livermore National Laboratory
//Written by Aaron Fisher (fisher47@llnl.gov). LLNL-CODE-738419.
//All rights reserved.
//This file is part of Acrotensor. For details, see https://github.com/LLNL/acrotensor.

#include "SliceTensor.hpp"

namespace acro
{

SliceTensor::SliceTensor(Tensor &T, std::vector<int> &sind)
{
    SliceInit(T, sind);
}


SliceTensor::SliceTensor(Tensor &T, int d0)
{
    std::vector<int> sind = {d0};
    SliceInit(T, sind);
}


SliceTensor::SliceTensor(Tensor &T, int d0, int d1)
{
    std::vector<int> sind = {d0, d1};
    SliceInit(T, sind);
}


SliceTensor::SliceTensor(Tensor &T, int d0, int d1, int d2)
{
    std::vector<int> sind = {d0, d1, d2};
    SliceInit(T, sind);
}


SliceTensor::SliceTensor(Tensor &T, int d0, int d1, int d2, int d3)
{
    std::vector<int> sind = {d0, d1, d2, d3};
    SliceInit(T, sind);
}


SliceTensor::SliceTensor(Tensor &T, int d0, int d1, int d2, int d3, int d4)
{
    std::vector<int> sind = {d0, d1, d2, d3, d4};
    SliceInit(T, sind);
}


SliceTensor::SliceTensor(Tensor &T, int d0, int d1, int d2, int d3, int d4, int d5)
{
    std::vector<int> sind = {d0, d1, d2, d3, d4, d5};
    SliceInit(T, sind);
}


SliceTensor::SliceTensor(Tensor &T, int d0, int d1, int d2, int d3, int d4, int d5, int d6)
{
    std::vector<int> sind = {d0, d1, d2, d3, d4, d5, d6};
    SliceInit(T, sind);
}


SliceTensor::SliceTensor(Tensor &T, int d0, int d1, int d2, int d3, int d4, int d5, int d6, int d7)
{
    std::vector<int> sind = {d0, d1, d2, d3, d4, d5, d6, d7};
    SliceInit(T, sind);
}


SliceTensor::SliceTensor(Tensor &T, int d0, int d1, int d2, int d3, int d4, int d5, int d6, int d7, int d8)
{
    std::vector<int> sind = {d0, d1, d2, d3, d4, d5, d6, d7};
    SliceInit(T, sind);
}

void SliceTensor::SliceInit(Tensor &T, int d0)
{
    std::vector<int> sind = {d0};
    SliceInit(T, sind);
}


void SliceTensor::SliceInit(Tensor &T, int d0, int d1)
{
    std::vector<int> sind = {d0,d1};
    SliceInit(T, sind);
}


void SliceTensor::SliceInit(Tensor &T, int d0, int d1, int d2)
{
    std::vector<int> sind = {d0,d1,d2};
    SliceInit(T, sind);
}


void SliceTensor::SliceInit(Tensor &T, int d0, int d1, int d2, int d3)
{
    std::vector<int> sind = {d0,d1,d2,d3};
    SliceInit(T, sind);
}


void SliceTensor::SliceInit(Tensor &T, int d0, int d1, int d2, int d3, int d4)
{
    std::vector<int> sind = {d0,d1,d2,d3,d4};
    SliceInit(T, sind);
}


void SliceTensor::SliceInit(Tensor &T, int d0, int d1, int d2, int d3, int d4, int d5)
{
    std::vector<int> sind = {d0,d1,d2,d3,d4,d5};
    SliceInit(T, sind);
}


void SliceTensor::SliceInit(Tensor &T, int d0, int d1, int d2, int d3, int d4, int d5, int d6)
{
    std::vector<int> sind = {d0,d1,d2,d3,d4,d5,d6};
    SliceInit(T, sind);
}


void SliceTensor::SliceInit(Tensor &T, int d0, int d1, int d2, int d3, int d4, int d5, int d6, int d7)
{
    std::vector<int> sind = {d0,d1,d2,d3,d4,d5,d6,d7};
    SliceInit(T, sind);
}


void SliceTensor::SliceInit(Tensor &T, int d0, int d1, int d2, int d3, int d4, int d5, int d6, int d7, int d8)
{
    std::vector<int> sind = {d0,d1,d2,d3,d4,d5,d6,d7,d8};
    SliceInit(T, sind);
}


void SliceTensor::SliceInit(Tensor &T, std::vector<int> &sind)
{
    FullT = &T;
    ACROBATIC_ASSERT(T.IsInitialized(), "Can't slice an uninitilized tensor.");
    ACROBATIC_ASSERT(T.GetRank() > sind.size(), "Can't slice more dimensions than the tensor rank.");
    std::vector<int> dims(T.GetRank() - sind.size());
    for (int d = sind.size(); d < T.GetRank(); ++d)
    {
        dims[d - sind.size()] = T.GetDim(d);
    }

    Offset = T.GetRawIndex(sind);
    double *hdata = T.GetData();
    double *ddata = T.GetDeviceData();
    if (hdata)
    {
        hdata += Offset;
    }

    if (ddata)
    {
        ddata += Offset;
    }

    Initialized = false;
    Init(dims, hdata, ddata, T.IsOnGPU());
}


double* SliceTensor::GetData() const
{
    return FullT->GetData() + Offset;
}


double* SliceTensor::GetDeviceData() const
{
    return FullT->GetDeviceData() + Offset;
}


void SliceTensor::MapToGPU()
{
    FullT->MapToGPU();
    DeviceData = FullT->GetDeviceData() + Offset;
}


void SliceTensor::MoveToGPU()
{
    FullT->MoveToGPU();     //May Trigger a MapToGPU()
    DeviceData = FullT->GetDeviceData() + Offset;
}


void SliceTensor::SwitchToGPU()
{
    FullT->SwitchToGPU();   //May Trigger a MapToGPU()
    DeviceData = FullT->GetDeviceData() + Offset;
}


void SliceTensor::MoveFromGPU()   
{
    FullT->MoveFromGPU();

}


void SliceTensor::SwitchFromGPU() 
{
    FullT->SwitchFromGPU();
}     


bool SliceTensor::IsMappedToGPU() const
{
    return FullT->IsMappedToGPU();
}


bool SliceTensor::IsOnGPU() const 
{
    return FullT->IsOnGPU();
}


void SliceTensor::UnmapFromGPU()
{

}


}
