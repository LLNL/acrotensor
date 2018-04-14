//Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at the Lawrence Livermore National Laboratory
//Written by Aaron Fisher (fisher47@llnl.gov). LLNL-CODE-738419.
//All rights reserved.
//This file is part of Acrotensor. For details, see https://github.com/LLNL/acrotensor.

#include "TensorKernel.hpp"
#include "SliceTensor.hpp"
#include <algorithm>
#include <iostream>

namespace acro
{

TensorKernel::TensorKernel(const char *kernel)
{
    KernelStr = kernel;
    ParseKernel();
}

TensorKernel::TensorKernel(std::string &kernel)
{
    KernelStr = kernel;
    ParseKernel();
}

// Recursive decent parse of tensor kernel (+ means 1 or more of in succession):
// <kernel> := <kernalvar><eqop><kernalvar>+
void TensorKernel::ParseKernel()
{
    std::string ParseStr = KernelStr;
    ParseStr.erase(remove_if(ParseStr.begin(), ParseStr.end(), isspace),ParseStr.end());
    std::string::iterator it = ParseStr.begin();
    ParseKernelVar(it, OutputVar);
 
    ParseEqOperator(it, EqOperator);

    KernelVar *input_var = new KernelVar();
    ParseKernelVar(it, *input_var);
    InputVars.push_back(input_var);

    while(it != ParseStr.end()) {
        input_var = new KernelVar();
        ParseKernelVar(it, *input_var);
        InputVars.push_back(input_var);
    }

    //Gather up the IndexNames and LoopNums associated with the OutputTensor
    for (int d = 0; d < OutputVar.IndexNames.size(); ++d)
    {
        AllIndexNames.push_back(OutputVar.IndexNames[d]);
    }

    //Now gather up the IndexNames and LoopNums associated with the contraction indices
    for (int vari = 0; vari < InputVars.size(); ++vari) {
        for (int indi = 0; indi < InputVars[vari]->IndexNames.size(); ++indi) {
            auto acit = std::find(AllIndexNames.begin(), 
                                AllIndexNames.end(),
                                InputVars[vari]->IndexNames[indi]);
            if (acit == AllIndexNames.end()) {
                //The IndexName is not on the list yet so add it to everything
                ContractionIndexNames.push_back(InputVars[vari]->IndexNames[indi]);
                AllIndexNames.push_back(InputVars[vari]->IndexNames[indi]);
            }
        }
    }

    LoopOrder = AllIndexNames;
    SetVarLoopNums();
}

// <kernalvar> := <name><indexvar>+
// where <U> is an uppercase letter
void TensorKernel::ParseKernelVar(std::string::iterator &it, KernelVar &var)
{
    ParseVarName(it, var);
    ParseIndexVar(it, var);
    while (*it == '_') {
        ParseIndexVar(it, var);
    }
    var.LoopNums.resize(var.IndexNames.size());
}


// <name> := <U><U/L>*
// where <U> is an uppercase letter and <U/L/D> is any letter or digit
void TensorKernel::ParseVarName(std::string::iterator &it, KernelVar &var)
{
    //<U>
    ACROBATIC_ASSERT(isupper(*it));
    var.Name += *it;
    it ++;

    //<U/L/D>*
    while (isupper(*it) || islower(*it) || isdigit(*it)) {
        var.Name += *it;
        it ++;
    }
}


//<indexvar> :=  _<L>+
//<L> is a lowercase letter or a digit
void TensorKernel::ParseIndexVar(std::string::iterator &it, KernelVar &var)
{
    //_
    ACROBATIC_ASSERT(*it == '_');
    it ++;

    //<L>+
    ACROBATIC_ASSERT(islower(*it) || isdigit(*it));
    var.IndexNames.push_back("");
    var.IndexNames[var.IndexNames.size() - 1] += *it;
    it ++;
    while(islower(*it) || isdigit(*it))
    {
        var.IndexNames[var.IndexNames.size() - 1] += *it;
        it ++;
    }
}


// <eqop> := ("=" | "+=" | "-=")
void TensorKernel::ParseEqOperator(std::string::iterator &it, std::string &eqoper)
{
    if (*it == '=') {
        it ++;
        eqoper = "=";
    } else if (*it == '+') {
        it ++;
        ACROBATIC_ASSERT(*it == '=');
        it ++;
        eqoper = "+=";
    } else if (*it == '-') {
        it ++;
        ACROBATIC_ASSERT(*it == '=');
        it ++;
        eqoper = "-=";    
    } else {
        ACROBATIC_ASSERT(false);
    }
}


void TensorKernel::SetVarLoopNums()
{
    OutputVar.LoopNums.resize(OutputVar.IndexNames.size());
    for (int idxi = 0; idxi < OutputVar.IndexNames.size(); ++idxi)
    {
        auto loopit = std::find(LoopOrder.begin(), LoopOrder.end(), OutputVar.IndexNames[idxi]);
        OutputVar.LoopNums[idxi] = std::distance(LoopOrder.begin(), loopit);
    }

    for (int ivari = 0; ivari < InputVars.size(); ++ivari)
    {
        InputVars[ivari]->LoopNums.resize(InputVars[ivari]->IndexNames.size());
        for (int idxi = 0; idxi < InputVars[ivari]->IndexNames.size(); ++idxi)
        {
            auto loopit = std::find(LoopOrder.begin(), LoopOrder.end(), InputVars[ivari]->IndexNames[idxi]);
            InputVars[ivari]->LoopNums[idxi] = std::distance(LoopOrder.begin(), loopit);
        }
    }
}


int TensorKernel::GetVarRank(int vari)
{
    ACROBATIC_ASSERT(vari >= -1 && vari < GetNumInputVars());

    if (vari == -1)
    {
        return OutputVar.IndexNames.size();
    }

    return InputVars[vari]->IndexNames.size();
}


int TensorKernel::GetVarLoopDepth(int vari)
{
    ACROBATIC_ASSERT(vari >= -1 && vari < GetNumInputVars());

    int depth = -1; //Invariant to all loops
    for (int loopd = 0; loopd < GetNumIndices(); ++ loopd)
    {
        if (IsVarDependentOnLoop(vari, loopd))
        {
            depth = loopd;
        }
    }
    return depth;
}


void TensorKernel::SetLoopOrder(std::vector<std::string> &idx_list)
{
    LoopOrder = idx_list;

    //Update the LoopNums with the new permuted order
    SetVarLoopNums();
}


int TensorKernel::GetVarDimLoopNum(int vari, int dim)
{
    ACROBATIC_ASSERT(vari >= -1 && vari < GetNumInputVars());
    ACROBATIC_ASSERT(dim >= 0 && dim < GetVarRank(vari));

    if (vari == -1)
    {
        return OutputVar.LoopNums[dim];
    }

    return InputVars[vari]->LoopNums[dim];
}


int TensorKernel::GetLoopNumVarDim(int loop_num, int vari)
{
    ACROBATIC_ASSERT(loop_num >= 0 && loop_num < AllIndexNames.size());
    ACROBATIC_ASSERT(vari >= -1 && vari < GetNumInputVars());

    std::string loop_index_name = AllIndexNames[loop_num];

    for (int d = 0; d < GetVarRank(vari); ++d)
    {
        if (vari == -1) 
        {  
            if (OutputVar.IndexNames[d] == loop_index_name)
            {
                return d;
            }
        }
        else
        {
            if (InputVars[vari]->IndexNames[d] == loop_index_name)
            {
                return d;
            }
        }
    }
    return -1;
}


bool TensorKernel::IsVarDependentOnLoop(int vari, int loop_num)
{
    return GetLoopNumVarDim(loop_num, vari) > -1;
}


bool TensorKernel::IsContractionLoop(int loop_num)
{
    std::string idxstr = LoopOrder[loop_num];
    return std::find(ContractionIndexNames.begin(),ContractionIndexNames.end(), idxstr)
                     != ContractionIndexNames.end();
}


std::string TensorKernel::GetNameString()
{
    std::string name = OutputVar.Name;
    for (int d = 0; d < OutputVar.LoopNums.size(); ++d)
    {
        name += "_" + std::to_string(OutputVar.LoopNums[d]);
    }

    if (EqOperator == "=")
    {
        name += "eq";
    }
    else if (EqOperator == "+=")
    {
        name += "pe";
    }
    else if (EqOperator == "-=")
    {
        name += "me";
    }

    for (int ivari = 0; ivari < InputVars.size(); ++ivari)
    {
        name += InputVars[ivari]->Name;
        for (int d = 0; d < InputVars[ivari]->LoopNums.size(); ++d)
        {
            name += "_" + std::to_string(InputVars[ivari]->LoopNums[d]);
        }
    }
    return name;
}


std::string TensorKernel::GetDimensionedNameString(Tensor *output, std::vector<Tensor*> &inputs)
{
    std::string name = GetNameString();
    std::vector<int> idx_sizes = GetLoopIdxSizes(output, inputs);
    
    name += "_";
    for (int idxi = 0; idxi < idx_sizes.size(); ++idxi)
    {
        name += "_" + std::to_string(idx_sizes[idxi]);
    }

    return name;
}


std::vector<int> TensorKernel::GetLoopIdxSizes(Tensor *output, std::vector<Tensor*> &inputs)
{
    std::vector<int> idx_sizes;
    idx_sizes.resize(GetNumIndices());
    for (int idxi = 0; idxi < output->GetRank(); ++idxi)
    {
        idx_sizes[GetVarDimLoopNum(-1, idxi)] = output->GetDim(idxi);
    }

    for (int vari = 0; vari < InputVars.size(); ++vari)
    {
        for (int indi = 0; indi < InputVars[vari]->LoopNums.size(); ++indi)
        {
            idx_sizes[GetVarDimLoopNum(vari, indi)] = inputs[vari]->GetDim(indi);
        }
    }

    //Check to make sure that the dimensions of the tensors are compatible with the kernel
    for (int vari = 0; vari < InputVars.size(); ++vari)
    {
        for (int indi = 0; indi < InputVars[vari]->LoopNums.size(); ++indi)
        {
            ACROBATIC_ASSERT(idx_sizes[InputVars[vari]->LoopNums[indi]] == inputs[vari]->GetDim(indi),
                             "Incompatible tensor dimensions for kernel:  " + KernelStr);
        }
    }
    return idx_sizes;
}

}