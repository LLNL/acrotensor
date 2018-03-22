//Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at the Lawrence Livermore National Laboratory
//Written by Aaron Fisher (fisher47@llnl.gov). LLNL-CODE-738419.
//All rights reserved.
//This file is part of Acrotensor. For details, see https://github.com/LLNL/acrotensor.

#ifndef ACROBATIC_SLICETENSOR_HPP
#define ACROBATIC_SLICETENSOR_HPP

#include "Tensor.hpp"

namespace acro 
{

class SliceTensor : public Tensor
{
    public:
    SliceTensor() {};
    SliceTensor(Tensor &T, std::vector<int> &sind);
    SliceTensor(Tensor &T, int d0);
    SliceTensor(Tensor &T, int d0, int d1);
    SliceTensor(Tensor &T, int d0, int d1, int d2);
    SliceTensor(Tensor &T, int d0, int d1, int d2, int d3);
    SliceTensor(Tensor &T, int d0, int d1, int d2, int d3, int d4);
    SliceTensor(Tensor &T, int d0, int d1, int d2, int d3, int d4, int d5);
    SliceTensor(Tensor &T, int d0, int d1, int d2, int d3, int d4, int d5, int d6);
    SliceTensor(Tensor &T, int d0, int d1, int d2, int d3, int d4, int d5, int d6, int d7);
    SliceTensor(Tensor &T, int d0, int d1, int d2, int d3, int d4, int d5, int d6, int d7, int d8);
    void SliceInit(Tensor &T, std::vector<int> &sind);
    void SliceInit(Tensor &T, int d0);
    void SliceInit(Tensor &T, int d0, int d1);
    void SliceInit(Tensor &T, int d0, int d1, int d2);
    void SliceInit(Tensor &T, int d0, int d1, int d2, int d3);
    void SliceInit(Tensor &T, int d0, int d1, int d2, int d3, int d4);
    void SliceInit(Tensor &T, int d0, int d1, int d2, int d3, int d4, int d5);
    void SliceInit(Tensor &T, int d0, int d1, int d2, int d3, int d4, int d5, int d6);
    void SliceInit(Tensor &T, int d0, int d1, int d2, int d3, int d4, int d5, int d6, int d7);
    void SliceInit(Tensor &T, int d0, int d1, int d2, int d3, int d4, int d5, int d6, int d7, int d8);    
    ~SliceTensor() {}

    virtual void Retarget(double *hdata, double*ddata=nullptr) {ACROBATIC_ASSERT(false, "Retarget not supported on SliceTensors");}

    //Routines for Data on the GPU
    virtual double* GetData() const;
    virtual double* GetDeviceData() const;
    virtual void MapToGPU();
    virtual void MoveToGPU();
    virtual void SwitchToGPU();
    virtual void UnmapFromGPU();
    virtual void MoveFromGPU();
    virtual void SwitchFromGPU();
    virtual bool IsMappedToGPU() const;
    virtual bool IsOnGPU() const;

    private:
    Tensor *FullT;
    int Offset;
};

}

#endif //ACROBATIC_SLICETENSOR_HPP
