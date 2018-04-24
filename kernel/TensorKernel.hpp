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
    protected:
    TensorKernel() {}
    public:
    TensorKernel(const char* kernel);
    TensorKernel(std::string &kernel);

    //The total number of loops required to execute the kernel
    int GetNumIndices() {return AllIndexNames.size();}

    //The number of outer loops in the kernel
    int GetNumOuterIndices() {return AllIndexNames.size() - ContractionIndexNames.size();}

    //The number of inner contraction loops in the kernel
    int GetNumContractionIndices() {return ContractionIndexNames.size();}

    //The number of variables referenced in the kernel (including the output tensor)
    int GetNumVars() {return InputVars.size()+1;}

    //The number of input variables referenced in the kernel
    int GetNumInputVars() {return InputVars.size();}

    //The rank of the given variable (vari = -1 for output)
    int GetVarRank(int vari);

    //Change the order of the loops which will affect the following loop_num functions and the values of Var->LoopNums
    std::string GetLoopIndex(int loopi) {return LoopIndices[loopi];}
    int GetLoopNum(std::string &idx);
    virtual void SetLoopIndices(std::vector<std::string> &idx_list);

    //The loop number for the given variable/dimension (vari = -1 for output)
    int GetVarDimLoopNum(int vari, int dim);

    //The input var dim given the loop num and the input vari (vari = -1 for output)
    //returns (-1 if input var is invariant to that loop)
    int GetLoopNumVarDim(int loop_num, int vari);

    //Does the input var have an index matching this loop num (vari = -1 for output)
    bool IsVarDependentOnLoop(int vari, int loop_num);

    //Does the Kernel have dependence on this index
    bool IsDependentOnIndex(std::string &idx);

    //Does the Kernel have dependence on this index
    bool IsDependentOnLoop(int loop_num);    

    //Does this a contraction index
    bool IsContractionIndex(std::string &idx);

    //Is this loop a contraction loop
    bool IsContractionLoop(int loop_num);

    bool IsContractionVar(int vari);

    //Get the highest loop number that the entire kernel depends on
    int GetLoopDepth();

    //The highest loop number that the var varies by (vari=-1 for output)
    int GetVarLoopDepth(int vari);

    //The the name of the variable (vari=-1 for output)
    std::string &GetVarName(int vari);

    //This returns the post parsed name string
    std::string GetNameString();

    //This returns a modified kernel string with the dimensions compatible with the tensors
    virtual std::string GetDimensionedNameString() {ACROBATIC_ASSERT(false); return "";}
    virtual std::string GetDimensionedNameString(Tensor *output, std::vector<Tensor*> &inputs);

    std::vector<int> GetLoopIdxSizes(Tensor *output, std::vector<Tensor*> &inputs);

    std::string KernelStr;                          //The user provided kernel string
    KernelVar OutputVar;                            //The output var extracted from the kernel string
    std::string EqOperator;                         //The assignement operator extracted from the kernel string (=, +=)
    std::vector<KernelVar> InputVars;              //The input vars extracted from the kernel string
    std::vector<std::string> AllIndexNames;         //The names of all the indices extracted from the kernel string
    std::vector<std::string> ContractionIndexNames; //The names of the contraction indices extracted from the kernel string
    std::vector<std::string> LoopIndices;

    private:
    void ParseKernel();
    void ParseKernelVar(std::string::iterator &it, KernelVar &var);
    void ParseVarName(std::string::iterator &it, KernelVar &var);
    void ParseIndexVar(std::string::iterator &it, KernelVar &var);
    void ParseEqOperator(std::string::iterator &it, std::string &op);
    void SetVarLoopNums();
};

}

#endif //ACROBATIC_TENSOR_KERNEL_HPP