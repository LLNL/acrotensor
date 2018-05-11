//Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at the Lawrence Livermore National Laboratory
//Written by Aaron Fisher (fisher47@llnl.gov). LLNL-CODE-738419.
//All rights reserved.
//This file is part of Acrotensor. For details, see https://github.com/LLNL/acrotensor.

#ifndef ACROBATIC_DIMENSIONED_MULTI_KERNEL_HPP
#define ACROBATIC_DIMENSIONED_MULTI_KERNEL_HPP

#include "DimensionedKernel.hpp"
#include <string>
#include <vector>

namespace acro
{


class DimensionedMultiKernel
{
    public:
    DimensionedMultiKernel(std::vector<DimensionedKernel*> &kernels);
    DimensionedMultiKernel(DimensionedKernel* kernel);

    int GetNumKernels() {return Kernels.size();}

    //The total number of loops required to execute the kernel
    int GetNumIndices() {return AllIndexNames.size();}

    //The number of outer loops in the multi kernel
    int GetNumOuterIndices() {return SharedOuterIndexNames.size();}

    //The number of inner contraction loops in the kernel
    int GetNumContractionIndices() {return ContractionIndexNames.size();}

    //The number of variables referenced in the kernel (including the output tensors)
    int GetNumVars();

    //The number of unique vars with the duplicates removed
    int GetNumUVars() {return UVariToFirstKiVari.size();}

    //The number of input variables referenced in the kernel
    int GetNumInputVars();

    //The number of input variables referenced in the kernel
    int GetNumOutputVars();

    std::string GetDimensionedNameString();

    std::string GetLoopIndex(int loopi) {return LoopIndices[loopi];}
    int GetIndexLoopNum(std::string &idx);

    //Change the order of the loops which will affect the following loop_num functions and the values of Var->LoopNums
    void SetLoopIndices(std::vector<std::string> &idx_list);

    //The rank of the given variable (mvari = -1..-n for output vars)
    int GetVarRank(int ki, int vari);

    //The loop number for the given mvariable/dimension (mvari = -1..-n for output)
    int GetVarDimLoopNum(int ki, int vari, int dim);

    //The input var dim given the loop num and the input mvari (mvari = -1..-n for output)
    //returns (-1 if input var is invariant to that loop)
    int GetLoopNumVarDim(int loop_num, int ki, int vari);

    //Does the input var have an index matching this loop num (mvari = -1..-n for outputs)
    bool IsVarDependentOnLoop(int ki, int vari, int loop_num);

    //Is this loop a contraction loop
    bool IsContractionLoop(int loop_num);

    //Is this loop bound to a shared non-contraction index for all the kernels
    bool IsSharedOuterLoop(int loop_num);

    //Is the UVar an output or input var (or both from different kernels)
    bool IsOutputUVar(int uvari);
    bool IsInputUVar(int uvari);

    //The dimensions of all the loops now that we have attached tensors
    const std::vector<int> &GetLoopDims() {return LoopDims;}
    int GetLoopDim(int i) {return LoopDims[i];}
    int GetLoopDim(std::string &idx) {return GetLoopDim(GetIndexLoopNum(idx));}
    int GetLoopStride(int i) {return LoopStrides[i];}

    //The the number of index combinations for all the loops (the product of the loop dims)
    int GetFlatIdxSize();

    //The the number of index combinations for just the outer non-contraction loops
    int GetSharedOuterIdxSize();

    //Get the number of indices in the first num_loops
    int GetIdxSizeForFirstNumLoops(int num_loops);

    //The stride in flattened index space of a given variable/dimension in the kernel 
    int GetVarDimStride(int ki, int vari, int dim);

    //The number of index combinations in a given variable in the kernel
    int GetVarSize(int ki, int vari);
    int GetVarSize(int uvari);

    //The highest loop number that the var varies by
    int GetVarLoopDepth(int ki, int vari);

    //The unique vars will be listed starting from 0..n for the unique outputs
    //followed by n+1..m for the unique inputs.  Duplicated will not be counted!
    int GetUVari(int ki, int vari) {return KiVariToUVari[std::make_pair(ki,vari)];}
    std::pair<int,int> GetFirstKiVariForUVari(int uvari) {return UVariToFirstKiVari[uvari];}

    //Information for the inner loops
    int GetVarStorageReqForInnerLoops(int ki, int vari, int num_loops);
    int GetInputStorageReqForInnerLoops(int num_loops);
    int GetOutputStorageReqForInnerLoops(int num_loops);
    int GetTotalStorageReqForInnerLoops(int num_loops);
    int GetIndexSpaceSizeForInnerLoops(int num_loops);
    void GetVarIndexOffsetsForInnerLoops(int ki, int vari, int num_inner_loops, 
                                         std::vector<int> &var_off, std::vector<int> &loop_off);  

    std::vector<DimensionedKernel*> Kernels;
    std::vector<std::string> AllIndexNames;
    std::vector<std::string> ContractionIndexNames;
    std::vector<std::string> SharedOuterIndexNames;
    std::vector<std::string> LoopIndices;

    private:
    void InitMKLVars();

    //Maps between the multikernel tensor numbering and the underlying kernel tensor numbering
    std::map<std::pair<int,int>, int> KiVariToUVari;
    std::vector<std::pair<int,int>> UVariToFirstKiVari;

    //The dimensions of the kernel loops computed to match the attached tensors
    std::vector<int> LoopDims;
    std::vector<int> LoopStrides;
};

}

#endif //ACROBATIC_DIMENSIONED_MULTI_KERNEL_HPP