//Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at the Lawrence Livermore National Laboratory
//Written by Aaron Fisher (fisher47@llnl.gov). LLNL-CODE-738419.
//All rights reserved.
//This file is part of Acrotensor. For details, see https://github.com/LLNL/acrotensor.

#ifndef ACROBATIC_DIMENSIONED_KERNEL_HPP
#define ACROBATIC_DIMENSIONED_KERNEL_HPP

#include "TensorKernel.hpp"
#include <string>
#include <vector>

namespace acro
{


class DimensionedKernel : public TensorKernel
{
    public:
    DimensionedKernel(TensorKernel *kernel, Tensor *output, std::vector<Tensor*> &inputs);

    //The dimensions of all the loops now that we have attached tensors
    const std::vector<int> &GetLoopDims() {return LoopDims;}
    const std::vector<int> &GetLoopStrides() {return LoopStrides;}
    int GetLoopDim(std::string &idx) {return LoopDims[GetLoopNum(idx)];}
    int GetLoopStride(std::string &idx) {return LoopStrides[GetLoopNum(idx)];}
    int GetLoopDim(int i) {return LoopDims[i];}
    int GetLoopStride(int i) {return LoopStrides[i];}
    virtual void SetLoopIndices(std::vector<std::string> &idx_list);

    //Get a string with all of the loop dimensions
    std::string GetLoopDimsString();
    std::string GetDimensionedNameString() {return GetNameString() + GetLoopDimsString();}
    std::string GetDimensionedNameString(Tensor *output, std::vector<Tensor*> &inputs) {return GetDimensionedNameString();}

    //The the number of index combinations for all the loops (the product of the loop dims)
    int GetFlatIdxSize();

    //The the number of index combinations for just the outer non-contraction loops
    int GetOutIdxSize();

    //The the number of index combinations for the inner contraction loops
    int GetContIdxSize();

    //Get the number of indices in the first num_loops
    int GetIdxSizeForFirstNumLoops(int num_loops);

    //Get the number of indices in the list of loops
    int GetLoopsIdxSize(std::vector<int> loops);

    //The size of the vari tensor's dim
    int GetVarDimSize(int vari, int dim) {return LoopDims[GetVarDimLoopNum(vari, dim)];}

    //The stride in flattened index space of a given variable/dimension in the kernel (vari=-1 for output)
    int GetVarDimStride(int vari, int dim);

    //The number of index combinations in a given variable in the kernel (vari=-1 for output)
    int GetVarSize(int vari);

    //Information for the inner loops
    int GetVarStorageReqForInnerLoops(int vari, int num_loops);
    int GetInputStorageReqForInnerLoops(int num_loops);
    int GetOutputStorageReqForInnerLoops(int num_loops);
    int GetTotalStorageReqForInnerLoops(int num_loops);
    int GetIndexSpaceSizeForInnerLoops(int num_loops);
    void GetVarIndexOffsetsForInnerLoops(int vari, int num_inner_loops, 
                                         std::vector<int> &var_off, std::vector<int> &loop_off);

    private:
    //The dimensions of the kernel loops computed to match the attached tensors
    std::vector<int> LoopDims;
    std::vector<int> LoopStrides;
};

}

#endif //ACROBATIC_DIMENSIONED_KERNEL_HPP