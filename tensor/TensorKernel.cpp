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
        OutputVar.LoopNums[d] = d;  
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
                InputVars[vari]->LoopNums[indi] = AllIndexNames.size() - 1;

                ContractionFirstVar.push_back(vari);
                ContractionFirstVarInd.push_back(indi);
            }
            else
            {   
                //The on is already on the list so just put the map for it in the IndexNums
                InputVars[vari]->LoopNums[indi] = acit - AllIndexNames.begin();
            }
        }
    }

    OutputT = nullptr;
    InputT.resize(GetNumLoops(), nullptr);
    LoopDims.resize(GetNumLoops(), 0);
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


int TensorKernel::GetVarRank(int vari)
{
    ACROBATIC_ASSERT(vari >= -1 && vari < GetNumInputVars());

    if (vari == -1)
    {
        return OutputVar.IndexNames.size();
    }

    return InputVars[vari]->IndexNames.size();
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


std::string TensorKernel::GetCanonicalNameString()
{
    std::string name = "T";
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
    else
    {
        ACROBATIC_ASSERT(false);
    }

    for (int ivari = 0; ivari < InputVars.size(); ++ivari)
    {
        name += "T";
        for (int d = 0; d < InputVars[ivari]->LoopNums.size(); ++d)
        {
            name += "_" + std::to_string(InputVars[ivari]->LoopNums[d]);
        }
    }
    return name;
}


void TensorKernel::AttachTensors(Tensor *output, std::vector<Tensor*> &inputs)
{
    OutputT = output;
    InputT = inputs;

    LoopDims.resize(GetNumLoops());
    for (int loopd = 0; loopd < output->GetRank(); ++loopd)
    {
        LoopDims[loopd] = output->GetDim(loopd);
    }

    for (int i = 0; i < ContractionIndexNames.size(); ++i)
    {
       LoopDims[output->GetRank() + i] = inputs[ContractionFirstVar[i]]->GetDim(ContractionFirstVarInd[i]);
    }

    //Check to make sure that the dimensions of the tensors are compatible with the kernel
    for (int vari = 0; vari < InputVars.size(); ++vari)
    {
        for (int indi = 0; indi < InputVars[vari]->LoopNums.size(); ++indi)
        {
            ACROBATIC_ASSERT(LoopDims[InputVars[vari]->LoopNums[indi]] == inputs[vari]->GetDim(indi),
                             "Incompatible tensor dimensions for kernel:  " + KernelStr);
        }
    }

    LoopStrides.resize(GetNumLoops());
    LoopStrides[GetNumLoops() - 1] = 1;
    for (int loopd = GetNumLoops() - 2; loopd >= 0; --loopd)
    {
        LoopStrides[loopd] = LoopStrides[loopd+1]*LoopDims[loopd+1];
    }
}


Tensor *TensorKernel::GetTensor(int vari)
{
    ACROBATIC_ASSERT(vari >= -1 && vari < GetNumInputVars());
    return (vari == -1) ? OutputT : InputT[vari];
}


Tensor *TensorKernel::GetInputTensor(int vari)
{
    ACROBATIC_ASSERT(vari >= 0 && vari < GetNumInputVars());
    return InputT[vari];
}


Tensor *TensorKernel::GetOutputTensor()
{
    return OutputT;
}

const std::vector<Tensor*> &TensorKernel::GetInputTensors()
{
    return InputT;
}

const std::vector<int> &TensorKernel::GetLoopDims()
{
    return LoopDims;
}


int TensorKernel::GetFlatIdxSize()
{
    int flatidx_size = 1;
    for (int d = 0; d < GetNumLoops(); ++d)
    {
        flatidx_size *= LoopDims[d];
    }
    return flatidx_size;
}


int TensorKernel::GetOutIdxSize()
{
    int outidx_size = 1;
    for (int d = 0; d < GetNumLoops() - GetNumContractionLoops(); ++d)
    {
        outidx_size *= LoopDims[d];
    }
    return outidx_size;
}


int TensorKernel::GetContIdxSize()
{   
    int contidx_size = 1;
    for (int d = GetNumLoops() - GetNumContractionLoops(); d < GetNumLoops(); ++d)
    {
        contidx_size *= LoopDims[d];
    }
    return contidx_size;
}

int TensorKernel::GetIdxSizeForFirstNumLoops(int num_loops)
{
    ACROBATIC_ASSERT(num_loops <= GetNumLoops());
    int idx_size = 1;
    for (int d = 0; d < num_loops; ++d)
    {
        idx_size *= LoopDims[d];
    }
    return idx_size;
}


int TensorKernel::GetVarDimStride(int vari, int dim)
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


int TensorKernel::GetVarSize(int vari)
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


int TensorKernel::GetVarLoopDepth(int vari)
{
    ACROBATIC_ASSERT(vari >= -1 && vari < GetNumInputVars());

    int depth = -1; //Invariant to all loops
    for (int loopd = 0; loopd < GetNumLoops(); ++ loopd)
    {
        if (IsVarDependentOnLoop(vari, loopd))
        {
            depth = loopd;
        }
    }
    return depth;
}

int TensorKernel::GetVarStorageReqForInnerLoops(int vari, int num_loops)
{
    ACROBATIC_ASSERT(vari >= -1 && vari < GetNumInputVars());
    ACROBATIC_ASSERT(num_loops >= 0 && num_loops <= GetNumLoops());

    int num_var_entries = 1;
    for (int loop_num = GetNumLoops() - 1; loop_num >= GetNumLoops() - num_loops; --loop_num)
    {
        if (IsVarDependentOnLoop(vari, loop_num))
        {
            num_var_entries *= LoopDims[loop_num];
        }
    }
    return num_var_entries;
}


int TensorKernel::GetInputStorageReqForInnerLoops(int num_loops)
{
    ACROBATIC_ASSERT(num_loops >= 0 && num_loops <= GetNumLoops());

    int num_entries = 0;
    for (int vari = 0; vari < GetNumInputVars(); ++vari) {
        num_entries += GetVarStorageReqForInnerLoops(vari, num_loops);
    }

    return num_entries;
}


int TensorKernel::GetOutputStorageReqForInnerLoops(int num_loops)
{
    ACROBATIC_ASSERT(num_loops >= 0 && num_loops <= GetNumLoops());

    return GetVarStorageReqForInnerLoops(-1, num_loops);
}


int TensorKernel::GetTotalStorageReqForInnerLoops(int num_loops)
{
    return GetInputStorageReqForInnerLoops(num_loops) + 
           GetOutputStorageReqForInnerLoops(num_loops);
}


int TensorKernel::GetIndexSpaceSizeForInnerLoops(int num_loops)
{
    int size = 1;
    for (int loop = GetNumLoops() - 1; loop >= GetNumLoops() - num_loops; --loop)
    {
        size *= LoopDims[loop];
    }
    return size;
}


int TensorKernel::GetContractionFacForInnerLoops(int num_loops)
{
    int size = 1;
    int num_loops_wcont = std::min(num_loops, GetNumContractionLoops());
    for (int loop = GetNumLoops() - 1; loop >= GetNumLoops() - num_loops_wcont; --loop)
    {
        size *= LoopDims[loop];
    }
    return size;
}


void TensorKernel::GetVarIndexOffsetsForInnerLoops(int vari, int num_inner_loops, 
                                                   std::vector<int> &var_off, std::vector<int> &loop_off)
{
    int num_loops = GetNumLoops();
    int num_outer_loops = num_loops - num_inner_loops;
    int loadidx_size = 1;
    for (int loopd = num_loops - num_inner_loops; loopd < num_loops; ++loopd)
    {
        if (IsVarDependentOnLoop(vari, loopd))
        {
            loadidx_size *= LoopDims[loopd];
        }
    }

    Tensor *T = GetTensor(vari);
    std::vector<int> inner_loop_strides(T->GetRank(), 1);
    var_off.resize(loadidx_size);
    loop_off.resize(loadidx_size);
    for (int loadidx = 0; loadidx < loadidx_size; ++loadidx)
    {
        //Compute the strides for the indices in the inner_loops
        int stride = 1;
        for (int d = T->GetRank() - 1; d >= 0; --d)
        {
            int loopd = GetVarDimLoopNum(vari,d);
            if (loopd >= num_outer_loops)
            {
                inner_loop_strides[d] = stride;
                stride *= T->GetDim(d);
            }
        }

        //Compute the unflattened var indices
        int varidx = 0;
        int loopidx = 0;
        for (int d = 0; d < T->GetRank(); ++d)
        {
            int loopd = GetVarDimLoopNum(vari,d);
            if (loopd >= num_outer_loops)
            {
                int I = (loadidx / inner_loop_strides[d]) % T->GetDim(d);
                varidx += I*T->GetStride(d);
                loopidx += I*LoopStrides[loopd];
            }
        }
        var_off[loadidx] = varidx;
        loop_off[loadidx] = loopidx;
    }
}

}