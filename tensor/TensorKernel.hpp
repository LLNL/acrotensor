//Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at the Lawrence Livermore National Laboratory
//Written by Aaron Fisher (fisher47@llnl.gov). LLNL-CODE-738419.
//All rights reserved.
//This file is part of Acrotensor. For details, see https://github.com/LLNL/acrotensor.

#ifndef ACROBATIC_TENSOR_KERNEL_HPP
#define ACROBATIC_TENSOR_KERNEL_HPP

#include "Tensor.hpp"
#include <string>

namespace acro
{

class KernelVar
{
    public:
    KernelVar() {}
    std::string Name;
    std::vector<std::string> IndexNames;
    std::vector<int> LoopNums;
};


class TensorKernel
{
    public:
    TensorKernel(const char* kernel);
    TensorKernel(std::string &kernel);

    //The total number of loops required to execute the kernel
    int GetNumLoops() {return AllIndexNames.size();}

    //The number of outer loops in the kernel
    int GetNumOuterLoops() {return AllIndexNames.size() - ContractionIndexNames.size();}

    //The number of inner contraction loops in the kernel
    int GetNumContractionLoops() {return ContractionIndexNames.size();}

    //The number of variables referenced in the kernel (including the output tensor)
    int GetNumVars() {return InputVars.size()+1;}

    //The number of input variables referenced in the kernel
    int GetNumInputVars() {return InputVars.size();}

    //The rank of the given input variable
    int GetVarRank(int vari);

    //The loop number for the given variable/dimension (vari = -1 for output)
    int GetVarDimLoopNum(int vari, int dim);

    //The input var dim given the loop num and the input vari (vari = -1 for output)
    //returns (-1 if input var is invariant to that loop)
    int GetLoopNumVarDim(int loop_num, int vari);

    //Does the input var have an index matching this loop num (vari = -1 for output)
    bool IsVarDependentOnLoop(int vari, int loop_num);

    //Is this loop a contraction loop
    bool IsContractionLoop(int loop_num) {return GetNumLoops() - loop_num - 1 < GetNumContractionLoops();}

    //This returns a modified kernel string with strict rules about variable
    //and index names.  This is useful to create a 1-1 correspondence between
    //kernel strings and the nested loop structures they describe.  
    std::string GetCanonicalNameString();

    //After attaching real tensors we can call these methods that compute 
    //various sizes based on the dimensions of the tensors
    void AttachTensors(Tensor *output, std::vector<Tensor*> &inputs);

    //Get an attached tensor (vari = -1 for output)
    Tensor *GetTensor(int vari);        
    Tensor *GetInputTensor(int vari);
    Tensor *GetOutputTensor();
    const std::vector<Tensor*> &GetInputTensors();

    //The dimensions of all the loops now that we have attached tensors
    const std::vector<int> &GetLoopDims();
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

    //The stride in flattened index space of a given variable/dimension in the kernel (vari=-1 ofr output)
    int GetVarDimStride(int vari, int dim);

    //The number of index combinations in a given variable in the kernel (vari=-1 for output)
    int GetVarSize(int vari);

    //Information for the inner loops
    int GetVarStorageReqForInnerLoops(int vari, int num_loops);
    int GetInputStorageReqForInnerLoops(int num_loops);
    int GetOutputStorageReqForInnerLoops(int num_loops);
    int GetTotalStorageReqForInnerLoops(int num_loops);
    int GetIndexSpaceSizeForInnerLoops(int num_loops);
    int GetContractionFacForInnerLoops(int num_loops);
    void GetVarIndexOffsetsForInnerLoops(int vari, int num_inner_loops, 
                                         std::vector<int> &var_off, std::vector<int> &loop_off);

    std::string KernelStr;                          //The user provided kernel string
    KernelVar OutputVar;                            //The output var extracted from the kernel string
    std::string EqOperator;                         //The assignement operator extracted from the kernel string (=, +=)
    std::vector<KernelVar*> InputVars;              //The input vars extracted from the kernel string
    std::vector<std::string> AllIndexNames;         //The names of all the indices extracted from the kernel string
    std::vector<std::string> ContractionIndexNames; //The names of the contraction indices extracted from the kernel string

    private:
    void ParseKernel();
    void ParseKernelVar(std::string::iterator &it, KernelVar &var);
    void ParseVarName(std::string::iterator &it, KernelVar &var);
    void ParseIndexVar(std::string::iterator &it, KernelVar &var);
    void ParseEqOperator(std::string::iterator &it, std::string &op);

    std::vector<int> ContractionFirstVar;           //The first input variable that has this contraction dim
    std::vector<int> ContractionFirstVarInd;        //The dimension of this variable that refrences this contraction dim

    //The attached tensors
    Tensor *OutputT;
    std::vector<Tensor*> InputT;

    //The dimensions of the kernel loops computed to match the attached tensors
    std::vector<int> LoopDims;
    std::vector<int> LoopStrides;
};

}

#endif //ACROBATIC_TENSOR_KERNEL_HPP