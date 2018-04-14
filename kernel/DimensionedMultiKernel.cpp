//Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at the Lawrence Livermore National Laboratory
//Written by Aaron Fisher (fisher47@llnl.gov). LLNL-CODE-738419.
//All rights reserved.
//This file is part of Acrotensor. For details, see https://github.com/LLNL/acrotensor.
#include "DimensionedMultiKernel.hpp"
#include <algorithm>

namespace acro
{

DimensionedMultiKernel::DimensionedMultiKernel(std::vector<DimensionedKernel*> &kernels)
{
    Kernels = kernels;
    InitMKLVars();
}

DimensionedMultiKernel::DimensionedMultiKernel(DimensionedKernel* &kernel)
{
    Kernels.push_back(kernel);
    InitMKLVars();
}


void DimensionedMultiKernel::InitMKLVars()
{
    int mvari = 0;
    for (int ki = 0; ki < Kernels.size(); ++ki)
    {
        for (int indi = 0; indi < Kernels[ki]->AllIndexNames.size(); ++indi)
        {
            auto it = std::find(AllIndexNames.begin(), AllIndexNames.end(), Kernels[ki]->AllIndexNames[indi]);
            if (it != AllIndexNames.end())
            {
                AllIndexNames.push_back(*it);
            }
        }

        for (int indi = 0; indi < Kernels[ki]->ContractionIndexNames.size(); ++indi)
        {
            auto it = std::find(ContractionIndexNames.begin(), ContractionIndexNames.end(), Kernels[ki]->ContractionIndexNames[indi]);
            if (it != ContractionIndexNames.end())
            {
                ContractionIndexNames.push_back(*it);
            }
        }

        for (int vari = 0; vari < Kernels[ki]->GetNumInputVars(); ++vari)
        {
            MVariToKi[mvari] = ki;
            MVariToVari[mvari] = vari;
            ++mvari;
        }
    }

    for (int ki = 0; ki < Kernels.size(); ++ki)
    {
        MVariToKi[-ki-1] = ki;
        MVariToVari[-ki-1] = -1;
    }
}



int DimensionedMultiKernel::GetNumVars()
{
    return GetNumInputVars()+GetNumOutputVars();
}


int DimensionedMultiKernel::GetNumInputVars()
{
    int numvars = 0;
    for (int ki = 0; ki < Kernels.size(); ++ki)
    {
        numvars += Kernels[ki]->GetNumInputVars();
    }
    return numvars;
}


int DimensionedMultiKernel::GetNumOutputVars()
{
    return Kernels.size();
}


int DimensionedMultiKernel::GetVarRank(int mvari)
{
    return Kernels[MVariToKi[mvari]]->GetVarRank(MVariToVari[mvari]);
}


void DimensionedMultiKernel::SetLoopOrder(std::vector<std::string> &idx_list)
{
    //Set the loop orders of the subkernels
    for (int ki = 0; ki < Kernels.size(); ++ki)
    {
        Kernels[ki]->SetLoopOrder(idx_list);
    }

    LoopOrder = idx_list;
}


int DimensionedMultiKernel::GetVarDimLoopNum(int mvari, int dim)
{
    return Kernels[MVariToKi[mvari]]->GetVarDimLoopNum(MVariToVari[mvari], dim);
}


int DimensionedMultiKernel::GetLoopNumVarDim(int loop_num, int mvari)
{
    return Kernels[MVariToKi[mvari]]->GetVarDimLoopNum(loop_num, MVariToVari[mvari]);
}


bool DimensionedMultiKernel::IsVarDependentOnLoop(int mvari, int loop_num)
{
    return Kernels[MVariToKi[mvari]]->IsVarDependentOnLoop(MVariToVari[mvari], loop_num);
}


bool DimensionedMultiKernel::IsContractionLoop(int loop_num)
{
    std::string idxstr = LoopOrder[loop_num];
    return std::find(ContractionIndexNames.begin(),ContractionIndexNames.end(), idxstr)
                     != ContractionIndexNames.end();
}


int DimensionedMultiKernel::GetFlatIdxSize()
{
    int flatidx_size = 1;
    for (int d = 0; d < GetNumIndices(); ++d)
    {
        flatidx_size *= LoopDims[d];
    }
    return flatidx_size;
}


int DimensionedMultiKernel::GetOutIdxSize()
{
    int outidx_size = 1;
    for (int d = 0; d < GetNumIndices(); ++d)
    {
        if (!IsContractionLoop(d))
        {
            outidx_size *= LoopDims[d];
        }
    }
    return outidx_size;
}


int DimensionedMultiKernel::GetContIdxSize()
{
    int contidx_size = 1;
    for (int d = 0; d < GetNumIndices(); ++d)
    {
        if (IsContractionLoop(d))
        {
            contidx_size *= LoopDims[d];
        }
    }
    return contidx_size;
}


int DimensionedMultiKernel::GetIdxSizeForFirstNumLoops(int num_loops)
{
    int idx_size = 1;
    for (int d = 0; d < num_loops; ++d)
    {
        idx_size *= LoopDims[d];
    }
    return idx_size;
}


int DimensionedMultiKernel::GetVarDimStride(int mvari, int dim)
{
    return Kernels[MVariToKi[mvari]]->GetVarDimStride(MVariToVari[mvari], dim);
}

int DimensionedMultiKernel::GetVarSize(int mvari)
{
    return Kernels[MVariToKi[mvari]]->GetVarSize(MVariToVari[mvari]);
}

int DimensionedMultiKernel::GetVarLoopDepth(int mvari)
{
    return Kernels[MVariToKi[mvari]]->GetVarLoopDepth(MVariToVari[mvari]);
}


int DimensionedMultiKernel::GetVarStorageReqForInnerLoops(int mvari, int num_loops)
{
    return Kernels[MVariToKi[mvari]]->GetVarStorageReqForInnerLoops(MVariToVari[mvari], num_loops);
}


int DimensionedMultiKernel::GetInputStorageReqForInnerLoops(int num_loops)
{
    int storage = 0;
    for (int ki = 0; ki < Kernels.size(); ++ki)
    {
        storage += Kernels[ki]->GetInputStorageReqForInnerLoops(num_loops);
    }
    return storage;
}


int DimensionedMultiKernel::GetOutputStorageReqForInnerLoops(int num_loops)
{
    int storage = 0;
    for (int ki = 0; ki < Kernels.size(); ++ki)
    {
        storage += Kernels[ki]->GetOutputStorageReqForInnerLoops(num_loops);
    }
    return storage;
}


int DimensionedMultiKernel::GetTotalStorageReqForInnerLoops(int num_loops)
{
    return GetInputStorageReqForInnerLoops(num_loops) + GetOutputStorageReqForInnerLoops(num_loops);
}


int DimensionedMultiKernel::GetIndexSpaceSizeForInnerLoops(int num_loops)
{
    int size = 1;
    for (int loop = GetNumIndices() - 1; loop >= GetNumIndices() - num_loops; --loop)
    {
        size *= LoopDims[loop];
    }
    return size;
}


void DimensionedMultiKernel::GetVarIndexOffsetsForInnerLoops(int mvari, int num_inner_loops, 
                                                  std::vector<int> &var_off, std::vector<int> &loop_off)
{
    Kernels[MVariToKi[mvari]]->GetVarIndexOffsetsForInnerLoops(MVariToVari[mvari], num_inner_loops, var_off, loop_off);
}

}