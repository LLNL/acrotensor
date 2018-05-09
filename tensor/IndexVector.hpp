//Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at the Lawrence Livermore National Laboratory
//Written by Aaron Fisher (fisher47@llnl.gov). LLNL-CODE-738419.
//All rights reserved.
//This file is part of Acrotensor. For details, see https://github.com/LLNL/acrotensor.

#ifndef ACROBATIC_INDEXVECTOR_HPP
#define ACROBATIC_INDEXVECTOR_HPP

#include <vector>
#include "Util.hpp"

namespace acro
{

class IndexVector
{
    public:
    IndexVector();
    IndexVector(int dim, int *hdata=nullptr, int *ddata=nullptr, bool ongpu=false);
    ~IndexVector();
    void Init(int dim, int *hdata=nullptr, int *ddata=nullptr, bool ongpu=false);

    int GetSize() const;
    int *GetData() const;
    int *GetDeviceData() const;
    int *GetCurrentData() const;
    int &operator[](int raw_index);

    void Retarget(int *hdata, int *ddata);

    void MapToGPU();            //Allocate memory for the data on the GPU
    void MoveToGPU();           //Copy the data to the GPU and flag the data as currently on the GPU
    void SwitchToGPU();         //Flag the data as currently onGPU
    void UnmapFromGPU();        //Deallocate memory on the GPU
    void MoveFromGPU();         //Copy the data back from the GPU and flag the data as currently on the CPU
    void SwitchFromGPU();       //Flag the data as currently on the CPU
    bool IsMappedToGPU() const {return MappedToGPU;}
    bool IsOnGPU() const {return OnGPU;}
    bool IsInitialized() const {return Initialized;}

    void Print();

    private:
    int Size;
    int ByteSize;

    bool Initialized;
    bool OwnsData;
    bool MappedToGPU;
    bool OnGPU;
    int *Data;
    int *DeviceData;
};


inline int IndexVector::GetSize() const
{
    return Size;
}


inline int *IndexVector::GetData() const
{
    return Data;
}


inline int *IndexVector::GetDeviceData() const
{
    return DeviceData;
}


inline int *IndexVector::GetCurrentData() const
{
    return (IsOnGPU()) ? DeviceData : Data;
}


inline int &IndexVector::operator[](int raw_index) 
{
#if DEBUG
    ACROBATIC_ASSERT(OnGPU, "You have accessed the CPU version of the data that is fresh on the GPU.");
#endif    
    return Data[raw_index];
}


}

#endif