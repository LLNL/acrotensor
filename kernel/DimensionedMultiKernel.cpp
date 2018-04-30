//Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at the Lawrence Livermore National Laboratory
//Written by Aaron Fisher (fisher47@llnl.gov). LLNL-CODE-738419.
//All rights reserved.
//This file is part of Acrotensor. For details, see https://github.com/LLNL/acrotensor.
#include "DimensionedMultiKernel.hpp"
#include <algorithm>
#include <set>

namespace acro
{

DimensionedMultiKernel::DimensionedMultiKernel(std::vector<DimensionedKernel*> &kernels)
{
    Kernels = kernels;
    InitMKLVars();
}

DimensionedMultiKernel::DimensionedMultiKernel(DimensionedKernel* kernel)
{
    Kernels.push_back(kernel);
    InitMKLVars();
}


void DimensionedMultiKernel::InitMKLVars()
{
    int uvari = 0;
    std::vector<std::string> added_vars;
    for (int ki = 0; ki < Kernels.size(); ++ki)
    {
        DimensionedKernel *kernel = Kernels[ki];
        for (int indi = 0; indi < kernel->AllIndexNames.size(); ++indi)
        {
            auto it = std::find(AllIndexNames.begin(), AllIndexNames.end(), kernel->AllIndexNames[indi]);
            if (it == AllIndexNames.end())
            {
                AllIndexNames.push_back(kernel->AllIndexNames[indi]);
            }
        }

        for (int indi = 0; indi < kernel->ContractionIndexNames.size(); ++indi)
        {
            auto it = std::find(ContractionIndexNames.begin(), ContractionIndexNames.end(), kernel->ContractionIndexNames[indi]);
            if (it == ContractionIndexNames.end())
            {
                ContractionIndexNames.push_back(kernel->ContractionIndexNames[indi]);
            }
        }

        for (int vari = -1; vari < kernel->GetNumInputVars(); ++vari)
        {
            auto it = std::find(added_vars.begin(), added_vars.end(), kernel->GetVarName(vari));
            if (it == added_vars.end())
            {
                added_vars.push_back(kernel->GetVarName(vari));
                UVariToFirstKiVari.push_back(std::make_pair(ki, vari));
                KiVariToUVari[std::make_pair(ki, vari)] = uvari;
                ++uvari;
            }
            else
            {
                KiVariToUVari[std::make_pair(ki, vari)] = std::distance(added_vars.begin(), it);
            }
        }
    }

    //Find all the ouder indices that are shared by all subkernels
    std::vector<std::string> remove_list;
    SharedOuterIndexNames = AllIndexNames;
    for (int ki = 0; ki < Kernels.size(); ++ki)
    {
        DimensionedKernel *kernel = Kernels[ki];
        remove_list.resize(0);
        for (int idxi = 0; idxi < SharedOuterIndexNames.size(); ++idxi)
        {
            if (!kernel->IsDependentOnIndex(SharedOuterIndexNames[idxi]) || 
                 kernel->IsContractionIndex(SharedOuterIndexNames[idxi]))
            {
                remove_list.push_back(SharedOuterIndexNames[idxi]);
            }
        }

        for (int ri = 0; ri < remove_list.size(); ++ri)
        {
            SharedOuterIndexNames.erase(std::remove(SharedOuterIndexNames.begin(), 
                                                    SharedOuterIndexNames.end(), remove_list[ri]),
                                        SharedOuterIndexNames.end());
        }
    }

    //Reorder the indices to put shared outer indices first
    std::vector<std::string> reordered_indices = SharedOuterIndexNames;
    for (int idxi = 0; idxi < AllIndexNames.size(); ++idxi)
    {
        std::string idx = AllIndexNames[idxi];
        auto it = std::find(reordered_indices.begin(), reordered_indices.end(), idx);
        if (it == reordered_indices.end())
        {
            reordered_indices.push_back(idx);
        }
    }
    SetLoopIndices(reordered_indices);

    //Finally Reorder the Shared outer indices by size (largest first)
    reordered_indices.clear();
    reordered_indices.resize(SharedOuterIndexNames.size());
    std::set<std::string> set_indices(SharedOuterIndexNames.begin(), SharedOuterIndexNames.end());
    for (int i = 0; i < reordered_indices.size(); ++i)
    {
        int biggest_loop_size = -1;
        std::string biggest_idx;
        for (auto idx : set_indices)
        {
            int loop_size = GetLoopDim(idx);
            if (loop_size > biggest_loop_size)
            {
                biggest_loop_size = loop_size;
                biggest_idx = idx;
            }
        }
        set_indices.erase(biggest_idx);
        reordered_indices[i] = biggest_idx;
    }
    SharedOuterIndexNames = reordered_indices;
    for (int idxi = 0; idxi < AllIndexNames.size(); ++idxi)
    {
        std::string idx = AllIndexNames[idxi];
        auto it = std::find(reordered_indices.begin(), reordered_indices.end(), idx);
        if (it == reordered_indices.end())
        {
            reordered_indices.push_back(idx);
        }
    }
    SetLoopIndices(reordered_indices);
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


void DimensionedMultiKernel::SetLoopIndices(std::vector<std::string> &idx_list)
{
    //Set the loop orders of the subkernels and the LoopDims
    LoopDims.clear();
    LoopDims.resize(idx_list.size(), 1);
    for (int ki = 0; ki < Kernels.size(); ++ki)
    {
        Kernels[ki]->SetLoopIndices(idx_list);
        for (int loopi = 0; loopi < idx_list.size(); ++loopi)
        {
            LoopDims[loopi] = std::max(LoopDims[loopi], Kernels[ki]->GetLoopDim(loopi));
        }
    }

    //Set the loop strides
    LoopStrides.clear();
    LoopStrides.resize(idx_list.size());
    LoopStrides[LoopDims.size() - 1] = 1;
    for (int loopd = LoopDims.size() - 2; loopd >= 0; --loopd)
    {
        LoopStrides[loopd] = LoopStrides[loopd+1]*LoopDims[loopd+1];
    }   

    LoopIndices = idx_list;
}


int DimensionedMultiKernel::GetIndexLoopNum(std::string &idx)
{
    auto it = std::find(LoopIndices.begin(), LoopIndices.end(), idx);
    if (it == LoopIndices.end())
    {
        return -1;
    }
    return std::distance(LoopIndices.begin(), it);
}


int DimensionedMultiKernel::GetVarRank(int ki, int vari)
{
    return Kernels[ki]->GetVarRank(vari);
}


int DimensionedMultiKernel::GetVarDimLoopNum(int ki, int vari, int dim)
{
    return Kernels[ki]->GetVarDimLoopNum(vari, dim);
}


int DimensionedMultiKernel::GetLoopNumVarDim(int loop_num, int ki, int vari)
{
    return Kernels[ki]->GetVarDimLoopNum(loop_num, vari);
}


std::string DimensionedMultiKernel::GetDimensionedNameString()
{
    std::string dimensioned_name;
    for (auto kernel : Kernels)
    {
        dimensioned_name += kernel->GetDimensionedNameString() + ";";
    }
    return dimensioned_name;
}

bool DimensionedMultiKernel::IsVarDependentOnLoop(int ki, int vari, int loop_num)
{
    return Kernels[ki]->IsVarDependentOnLoop(vari, loop_num);
}


bool DimensionedMultiKernel::IsContractionLoop(int loop_num)
{
    std::string idxstr = LoopIndices[loop_num];
    return std::find(ContractionIndexNames.begin(),ContractionIndexNames.end(), idxstr)
                     != ContractionIndexNames.end();
}


bool DimensionedMultiKernel::IsSharedOuterLoop(int loop_num)
{
    std::string idxstr = LoopIndices[loop_num];
    return std::find(SharedOuterIndexNames.begin(),SharedOuterIndexNames.end(), idxstr)
                     != SharedOuterIndexNames.end();
}


bool DimensionedMultiKernel::IsOutputUVar(int uvari)
{
    for (int ki = 0; ki < Kernels.size(); ++ki)
    {
        if (KiVariToUVari[std::make_pair(ki,-1)] == uvari)
        {
            return true;
        }
    }
    return false;
}


bool DimensionedMultiKernel::IsInputUVar(int uvari)
{
    for (int ki = 0; ki < Kernels.size(); ++ki)
    {
        for (int vari = 0; vari < Kernels[ki]->GetNumInputVars(); ++vari)
        {
            if (KiVariToUVari[std::make_pair(ki,vari)] == uvari)
            {
                return true;
            }
        }
    }
    return false;
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


int DimensionedMultiKernel::GetSharedOuterIdxSize()
{
    int outidx_size = 1;
    for (int d = 0; d < GetNumIndices(); ++d)
    {
        if (IsSharedOuterLoop(d))
        {
            outidx_size *= LoopDims[d];
        }
    }
    return outidx_size;
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


int DimensionedMultiKernel::GetVarDimStride(int ki, int vari, int dim)
{
    return Kernels[ki]->GetVarDimStride(vari, dim);
}


int DimensionedMultiKernel::GetVarSize(int ki, int vari)
{
    return Kernels[ki]->GetVarSize(vari);
}


int DimensionedMultiKernel::GetVarSize(int uvari)
{
    auto ki_vari = UVariToFirstKiVari[uvari];
    return GetVarSize(ki_vari.first, ki_vari.second);
}


int DimensionedMultiKernel::GetVarLoopDepth(int ki, int vari)
{
    return Kernels[ki]->GetVarLoopDepth(vari);
}


int DimensionedMultiKernel::GetVarStorageReqForInnerLoops(int ki, int vari, int num_loops)
{
    return Kernels[ki]->GetVarStorageReqForInnerLoops(vari, num_loops);
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


void DimensionedMultiKernel::GetVarIndexOffsetsForInnerLoops(int ki, int vari, int num_inner_loops, 
                                                  std::vector<int> &var_off, std::vector<int> &loop_off)
{
    Kernels[ki]->GetVarIndexOffsetsForInnerLoops(vari, num_inner_loops, var_off, loop_off);
}

}