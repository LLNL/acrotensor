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
    DimensionedMultiKernel(DimensionedKernel* &kernel);

    //The total number of loops required to execute the kernel
    int GetNumIndices() {return AllIndexNames.size();}

    //The number of outer loops in the kernel
    int GetNumOuterIndices() {return AllIndexNames.size() - ContractionIndexNames.size();}

    //The number of inner contraction loops in the kernel
    int GetNumContractionIndices() {return ContractionIndexNames.size();}

    //The number of variables referenced in the kernel (including the output tensor)
    int GetNumVars();

    //The number of input variables referenced in the kernel
    int GetNumInputVars();

    //The number of input variables referenced in the kernel
    int GetNumOutputVars();    

    //The rank of the given variable (mvari = -1..-n for output vars)
    int GetVarRank(int mvari);

    //Change the order of the loops which will affect the following loop_num functions and the values of Var->LoopNums
    void SetLoopOrder(std::vector<std::string> &idx_list);

    //The loop number for the given mvariable/dimension (mvari = -1..-n for output)
    int GetVarDimLoopNum(int mvari, int dim);

    //The input var dim given the loop num and the input mvari (mvari = -1..-n for output)
    //returns (-1 if input var is invariant to that loop)
    int GetLoopNumVarDim(int loop_num, int mvari);

    //Does the input var have an index matching this loop num (mvari = -1..-n for outputs)
    bool IsVarDependentOnLoop(int mvari, int loop_num);

    //Is this loop a contraction loop
    bool IsContractionLoop(int loop_num);

    //The dimensions of all the loops now that we have attached tensors
    const std::vector<int> &GetLoopDims() {return LoopDims;}
    int GetLoopDim(int i) {return LoopDims[i];}
    int GetLoopStride(int i) {return LoopStrides[i];}

    //The the number of index combinations for all the loops (the product of the loop dims)
    int GetFlatIdxSize();

    //The the number of index combinations for just the outer non-contraction loops
    int GetOutIdxSize();

    //The the number of index combinations for the inner contraction loops
    int GetContIdxSize();

    //Get the number of indices in the first num_loops
    int GetIdxSizeForFirstNumLoops(int num_loops);

    //The stride in flattened index space of a given variable/dimension in the kernel (mvari=-1 for output)
    int GetVarDimStride(int mvari, int dim);

    //The number of index combinations in a given variable in the kernel (mvari=-1 for output)
    int GetVarSize(int mvari);

    //The highes loop number that the var varies by (mvari=-1 for output)
    int GetVarLoopDepth(int mvari);

    //Information for the inner loops
    int GetVarStorageReqForInnerLoops(int mvari, int num_loops);
    int GetInputStorageReqForInnerLoops(int num_loops);
    int GetOutputStorageReqForInnerLoops(int num_loops);
    int GetTotalStorageReqForInnerLoops(int num_loops);
    int GetIndexSpaceSizeForInnerLoops(int num_loops);
    void GetVarIndexOffsetsForInnerLoops(int mvari, int num_inner_loops, 
                                         std::vector<int> &var_off, std::vector<int> &loop_off);

    std::vector<DimensionedKernel*> Kernels;
    std::vector<std::string> AllIndexNames;
    std::vector<std::string> ContractionIndexNames;

    private:
    void InitMKLVars();

    //Maps between the multikernel tensor numbering and the underlying kernel tensor numbering
    std::map<int,int> MVariToKi;
    std::map<int,int> MVariToVari;

    //The dimensions of the kernel loops computed to match the attached tensors
    std::vector<int> LoopDims;
    std::vector<int> LoopStrides;
    std::vector<std::string> LoopOrder;
};

}

#endif //ACROBATIC_DIMENSIONED_MULTI_KERNEL_HPP