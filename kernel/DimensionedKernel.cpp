//Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at the Lawrence Livermore National Laboratory
//Written by Aaron Fisher (fisher47@llnl.gov). LLNL-CODE-738419.
//All rights reserved.
//This file is part of Acrotensor. For details, see https://github.com/LLNL/acrotensor.
#include "DimensionedKernel.hpp"
#include <algorithm>

namespace acro
{


DimensionedKernel::DimensionedKernel(TensorKernel *kernel, Tensor *output, std::vector<Tensor*> &inputs)
{
    //Copy these from the original kernel
    KernelStr = kernel->KernelStr;
    OutputVar = kernel->OutputVar;
    EqOperator = kernel->EqOperator;
    InputVars = kernel->InputVars;
    AllIndexNames = kernel->AllIndexNames;
    ContractionIndexNames = kernel->ContractionIndexNames;
    LoopIndices = kernel->LoopIndices;

    LoopDims = kernel->GetLoopIdxSizes(output, inputs);
    LoopStrides.resize(LoopDims.size());
    LoopStrides[LoopDims.size() - 1] = 1;
    for (int loopd = LoopDims.size() - 2; loopd >= 0; --loopd)
    {
        LoopStrides[loopd] = LoopStrides[loopd+1]*LoopDims[loopd+1];
    }
}


void DimensionedKernel::SetLoopIndices(std::vector<std::string> &idx_list)
{
    //Update the loop dims before we change all the LoopIndex info
    std::vector<int> NewLoopDims(idx_list.size(), 1);
    for (int idxi = 0; idxi < NewLoopDims.size(); ++idxi)
    {
        auto it = std::find(LoopIndices.begin(), LoopIndices.end(), idx_list[idxi]);
        if (it != LoopIndices.end())
        {
            NewLoopDims[idxi] = LoopDims[std::distance(LoopIndices.begin(), it)];
        }
        else
        {
            NewLoopDims[idxi] = 1;
        }
    }
    LoopDims = NewLoopDims;

    //Update the loop strides
    LoopStrides.resize(LoopDims.size());
    LoopStrides[LoopDims.size() - 1] = 1;
    for (int loopd = LoopDims.size() - 2; loopd >= 0; --loopd)
    {
        LoopStrides[loopd] = LoopStrides[loopd+1]*LoopDims[loopd+1];
    }

    //update all the indices and underlying variable objects
    TensorKernel::SetLoopIndices(idx_list);
}


std::string DimensionedKernel::GetLoopDimsString()
{
    std::string name = "__dim";
    for (auto idx : AllIndexNames)
    {
        name += "_" + std::to_string(GetLoopDim(idx));
    }
    
    return name;
}



int DimensionedKernel::GetFlatIdxSize()
{
    int flatidx_size = 1;
    for (int d = 0; d < GetNumIndices(); ++d)
    {
        flatidx_size *= LoopDims[d];
    }
    return flatidx_size;
}


int DimensionedKernel::GetOutIdxSize()
{
    int outidx_size = 1;
    for (int d = 0; d < GetNumIndices() - GetNumContractionIndices(); ++d)
    {
        outidx_size *= LoopDims[d];
    }
    return outidx_size;
}


int DimensionedKernel::GetContIdxSize()
{   
    int contidx_size = 1;
    for (int d = GetNumIndices() - GetNumContractionIndices(); d < GetNumIndices(); ++d)
    {
        contidx_size *= LoopDims[d];
    }
    return contidx_size;
}


int DimensionedKernel::GetLoopsIdxSize(std::vector<int> loops)
{
    int idx_size = 1;
    for (auto loopi : loops)
    {
        idx_size *= LoopDims[loopi];
    }
    return idx_size;
}


int DimensionedKernel::GetIdxSizeForFirstNumLoops(int num_loops)
{
    ACROBATIC_ASSERT(num_loops <= GetNumIndices());
    int idx_size = 1;
    for (int d = 0; d < num_loops; ++d)
    {
        idx_size *= LoopDims[d];
    }
    return idx_size;
}


int DimensionedKernel::GetVarDimStride(int vari, int dim)
{
    ACROBATIC_ASSERT(vari >= -1 && vari < GetNumInputVars());

    int trank = GetVarRank(vari);
    int stride = 1;
    for (int d = trank-2; d >= dim; --d)
    {
        stride *= LoopDims[GetVarDimLoopNum(vari, d+1)];
    }

    return stride;
}


int DimensionedKernel::GetVarSize(int vari)
{
    ACROBATIC_ASSERT(vari >= -1 && vari < GetNumInputVars());

    int rank = GetVarRank(vari);
    int size = 1;
    for (int d = 0; d < rank; ++d)
    {
        size *= LoopDims[GetVarDimLoopNum(vari, d)];
    }
    return size;
}


int DimensionedKernel::GetVarStorageReqForInnerLoops(int vari, int num_loops)
{
    ACROBATIC_ASSERT(vari >= -1 && vari < GetNumInputVars());
    ACROBATIC_ASSERT(num_loops >= 0 && num_loops <= GetNumIndices());

    int num_var_entries = 1;
    for (int loop_num = GetNumIndices() - 1; loop_num >= GetNumIndices() - num_loops; --loop_num)
    {
        if (IsVarDependentOnLoop(vari, loop_num))
        {
            num_var_entries *= LoopDims[loop_num];
        }
    }
    return num_var_entries;
}


int DimensionedKernel::GetInputStorageReqForInnerLoops(int num_loops)
{
    ACROBATIC_ASSERT(num_loops >= 0 && num_loops <= GetNumIndices());

    int num_entries = 0;
    for (int vari = 0; vari < GetNumInputVars(); ++vari) {
        num_entries += GetVarStorageReqForInnerLoops(vari, num_loops);
    }

    return num_entries;
}


int DimensionedKernel::GetOutputStorageReqForInnerLoops(int num_loops)
{
    ACROBATIC_ASSERT(num_loops >= 0 && num_loops <= GetNumIndices());

    return GetVarStorageReqForInnerLoops(-1, num_loops);
}


int DimensionedKernel::GetTotalStorageReqForInnerLoops(int num_loops)
{
    return GetInputStorageReqForInnerLoops(num_loops) + 
           GetOutputStorageReqForInnerLoops(num_loops);
}


int DimensionedKernel::GetIndexSpaceSizeForInnerLoops(int num_loops)
{
    int size = 1;
    for (int loop = GetNumIndices() - 1; loop >= GetNumIndices() - num_loops; --loop)
    {
        size *= LoopDims[loop];
    }
    return size;
}


void DimensionedKernel::GetVarIndexOffsetsForInnerLoops(int vari, int num_inner_loops, 
                                                   std::vector<int> &var_off, std::vector<int> &loop_off)
{
    int num_loops = GetNumIndices();
    int num_outer_loops = num_loops - num_inner_loops;
    int loadidx_size = 1;
    for (int loopd = num_loops - num_inner_loops; loopd < num_loops; ++loopd)
    {
        if (IsVarDependentOnLoop(vari, loopd))
        {
            loadidx_size *= LoopDims[loopd];
        }
    }

    std::vector<int> inner_loop_strides(GetVarRank(vari), 1);
    var_off.resize(loadidx_size);
    loop_off.resize(loadidx_size);
    for (int loadidx = 0; loadidx < loadidx_size; ++loadidx)
    {
        //Compute the strides for the indices in the inner_loops
        int stride = 1;
        for (int d = GetVarRank(vari) - 1; d >= 0; --d)
        {
            int loopd = GetVarDimLoopNum(vari,d);
            if (loopd >= num_outer_loops)
            {
                inner_loop_strides[d] = stride;
                stride *= GetVarDimSize(vari,d);
            }
        }

        //Compute the unflattened var indices
        int varidx = 0;
        int loopidx = 0;
        for (int d = 0; d < GetVarRank(vari); ++d)
        {
            int loopd = GetVarDimLoopNum(vari,d);
            if (loopd >= num_outer_loops)
            {
                int I = (loadidx / inner_loop_strides[d]) % GetVarDimSize(vari,d);
                varidx += I*GetVarDimStride(vari, d);
                loopidx += I*LoopStrides[loopd];
            }
        }
        var_off[loadidx] = varidx;
        loop_off[loadidx] = loopidx;
    }
}

}