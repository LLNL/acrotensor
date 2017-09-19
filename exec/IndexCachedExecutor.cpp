//Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at the Lawrence Livermore National Laboratory
//Written by Aaron Fisher (fisher47@llnl.gov). LLNL-CODE-738419.
//All rights reserved.
//This file is part of Acrotensor. For details, see https://github.com/LLNL/acrotensor.

#include "IndexCachedExecutor.hpp"
#include <iostream>

namespace acro
{


IndexCachedExecutor::IndexCachedExecutor(std::string &kernelstr) : KernelExecutor(kernelstr) 
{

}


IndexCachedExecutor::~IndexCachedExecutor()
{
    for (auto it = RawIndexMap.begin(); it != RawIndexMap.end(); ++it)
    {
        int *raw_indices = it->second;
        delete [] raw_indices;
    }
    RawIndexMap.clear();
}


void IndexCachedExecutor::ExecuteKernel(Tensor *output, std::vector<Tensor*> &inputs)
{
    MoveTensorsFromGPU(output, inputs);

    Kernel.AttachTensors(output, inputs);
    std::vector<int> N = Kernel.GetLoopDims();
    int outidx_size = Kernel.GetOutIdxSize();
    int contidx_size = Kernel.GetContIdxSize();

    auto it = RawIndexMap.find(N);
    if (it == RawIndexMap.end())
    {
        AddIndices();
    }

    //Since we are using += or -= into the output 
    if (Kernel.EqOperator == "=")
    {
        output->Set(0.0);
    }

    //Get Raw versions of all the variables
    int numinvars = Kernel.GetNumInputVars();
    double **din = new double*[numinvars];   //Pointers to all the input tensor data
    double *dout = output->GetData();
    for (int ivari = 0; ivari < numinvars; ++ivari)
    {
        din[ivari] = inputs[ivari]->GetData();
    }

    int *raw_indices = RawIndexMap[N];
    ExecICFlatLoop(dout, din, raw_indices, outidx_size, contidx_size, numinvars);
}


std::string IndexCachedExecutor::GetImplementation(Tensor *out, std::vector<Tensor*> &inputs)
{
    return "Interpreted\n";
}


void IndexCachedExecutor::ExecICFlatLoop(double *dout, 
                                            double **din, 
                                            int *raw_indices, 
                                            int outidx_size, 
                                            int contidx_size, 
                                            int numinvars)
{
    switch (numinvars)
    {
        case 1: ExecICFlatLoop1(dout, din, raw_indices, outidx_size, contidx_size); break;
        case 2: ExecICFlatLoop2(dout, din, raw_indices, outidx_size, contidx_size); break;
        case 3: ExecICFlatLoop3(dout, din, raw_indices, outidx_size, contidx_size); break;
        case 4: ExecICFlatLoop4(dout, din, raw_indices, outidx_size, contidx_size); break;        
        case 5: ExecICFlatLoop5(dout, din, raw_indices, outidx_size, contidx_size); break;
        case 6: ExecICFlatLoop6(dout, din, raw_indices, outidx_size, contidx_size); break;
        case 7: ExecICFlatLoop7(dout, din, raw_indices, outidx_size, contidx_size); break;
        case 8: ExecICFlatLoop8(dout, din, raw_indices, outidx_size, contidx_size); break;
        default: ExecICFlatLoopAny(dout, din, raw_indices, outidx_size, contidx_size, numinvars);
    }
}


void IndexCachedExecutor::ExecICFlatLoopAny(double *dout, 
                                               double **din, 
                                               int *raw_indices, 
                                               int outidx_size, 
                                               int contidx_size, 
                                               int numinvars)
{
    int flatidx = 0;
    int flatidx_size = outidx_size*contidx_size;
    for (int outidx = 0; outidx < outidx_size; ++outidx)
    {
        for (int contidx = 0; contidx < contidx_size; ++contidx)
        {
            double rhs_val = 1.0;
            for (int ivari = 0; ivari < numinvars; ++ivari)
            {
                int var_offset = flatidx_size*ivari;
                rhs_val *= din[ivari][raw_indices[var_offset + flatidx]];
            }
            dout[outidx] += rhs_val;
            ++flatidx;
        }
    }
}

void IndexCachedExecutor::ExecICFlatLoop1(double *dout, 
                                             double **din, 
                                             int *raw_indices, 
                                             int outidx_size, 
                                             int contidx_size)
{
    int flatidx = 0;
    for (int outidx = 0; outidx < outidx_size; ++outidx)
    {
        for (int contidx = 0; contidx < contidx_size; ++contidx)
        {
            dout[outidx] += din[0][raw_indices[flatidx]];
            ++flatidx;
        }
    }
}


void IndexCachedExecutor::ExecICFlatLoop2(double *dout, 
                                             double **din, 
                                             int *raw_indices, 
                                             int outidx_size, 
                                             int contidx_size)
{
    int flatidx = 0;
    int flatidx_size = outidx_size*contidx_size;
    for (int outidx = 0; outidx < outidx_size; ++outidx)
    {
        for (int contidx = 0; contidx < contidx_size; ++contidx)
        {
            dout[outidx] += din[0][raw_indices[flatidx]]*
                            din[1][raw_indices[flatidx_size + flatidx]];
            ++flatidx;
        }
    }
}


void IndexCachedExecutor::ExecICFlatLoop3(double *dout, 
                                             double **din, 
                                             int *raw_indices, 
                                             int outidx_size, 
                                             int contidx_size)
{
    int flatidx = 0;
    int flatidx_size = outidx_size*contidx_size;
    for (int outidx = 0; outidx < outidx_size; ++outidx)
    {
        for (int contidx = 0; contidx < contidx_size; ++contidx)
        {
            dout[outidx] += din[0][raw_indices[flatidx]]*
                            din[1][raw_indices[flatidx_size + flatidx]]*
                            din[2][raw_indices[2*flatidx_size + flatidx]];
            ++flatidx;
        }
    }
}


void IndexCachedExecutor::ExecICFlatLoop4(double *dout, 
                                             double **din, 
                                             int *raw_indices, 
                                             int outidx_size, 
                                             int contidx_size)
{
    int flatidx = 0;
    int flatidx_size = outidx_size*contidx_size;
    for (int outidx = 0; outidx < outidx_size; ++outidx)
    {
        for (int contidx = 0; contidx < contidx_size; ++contidx)
        {
            dout[outidx] += din[0][raw_indices[flatidx]]*
                            din[1][raw_indices[flatidx_size + flatidx]]*
                            din[2][raw_indices[2*flatidx_size + flatidx]]*
                            din[3][raw_indices[3*flatidx_size + flatidx]];
            ++flatidx;
        }
    }
}


void IndexCachedExecutor::ExecICFlatLoop5(double *dout, 
                                             double **din, 
                                             int *raw_indices, 
                                             int outidx_size, 
                                             int contidx_size)
{
    int flatidx = 0;
    int flatidx_size = outidx_size*contidx_size;
    for (int outidx = 0; outidx < outidx_size; ++outidx)
    {
        for (int contidx = 0; contidx < contidx_size; ++contidx)
        {
            dout[outidx] += din[0][raw_indices[flatidx]]*
                            din[1][raw_indices[flatidx_size + flatidx]]*
                            din[2][raw_indices[2*flatidx_size + flatidx]]*
                            din[3][raw_indices[3*flatidx_size + flatidx]]*
                            din[4][raw_indices[4*flatidx_size + flatidx]];
            ++flatidx;
        }
    }
}


void IndexCachedExecutor::ExecICFlatLoop6(double *dout, 
                                             double **din, 
                                             int *raw_indices, 
                                             int outidx_size, 
                                             int contidx_size)
{
    int flatidx = 0;
    int flatidx_size = outidx_size*contidx_size;
    for (int outidx = 0; outidx < outidx_size; ++outidx)
    {
        for (int contidx = 0; contidx < contidx_size; ++contidx)
        {
            dout[outidx] += din[0][raw_indices[flatidx]]*
                            din[1][raw_indices[flatidx_size + flatidx]]*
                            din[2][raw_indices[2*flatidx_size + flatidx]]*
                            din[3][raw_indices[3*flatidx_size + flatidx]]*
                            din[4][raw_indices[4*flatidx_size + flatidx]]*
                            din[5][raw_indices[5*flatidx_size + flatidx]];
            ++flatidx;
        }
    }
}


void IndexCachedExecutor::ExecICFlatLoop7(double *dout, 
                                             double **din, 
                                             int *raw_indices, 
                                             int outidx_size, 
                                             int contidx_size)
{
    int flatidx = 0;
    int flatidx_size = outidx_size*contidx_size;
    for (int outidx = 0; outidx < outidx_size; ++outidx)
    {
        for (int contidx = 0; contidx < contidx_size; ++contidx)
        {
            dout[outidx] += din[0][raw_indices[flatidx]]*
                            din[1][raw_indices[flatidx_size + flatidx]]*
                            din[2][raw_indices[2*flatidx_size + flatidx]]*
                            din[3][raw_indices[3*flatidx_size + flatidx]]*
                            din[4][raw_indices[4*flatidx_size + flatidx]]*
                            din[5][raw_indices[5*flatidx_size + flatidx]]*
                            din[6][raw_indices[6*flatidx_size + flatidx]];
            ++flatidx;
        }
    }
}


void IndexCachedExecutor::ExecICFlatLoop8(double *dout, 
                                             double **din, 
                                             int *raw_indices, 
                                             int outidx_size, 
                                             int contidx_size)
{
    int flatidx = 0;
    int flatidx_size = outidx_size*contidx_size;
    for (int outidx = 0; outidx < outidx_size; ++outidx)
    {
        for (int contidx = 0; contidx < contidx_size; ++contidx)
        {
            dout[outidx] += din[0][raw_indices[flatidx]]*
                            din[1][raw_indices[flatidx_size + flatidx]]*
                            din[2][raw_indices[2*flatidx_size + flatidx]]*
                            din[3][raw_indices[3*flatidx_size + flatidx]]*
                            din[4][raw_indices[4*flatidx_size + flatidx]]*
                            din[5][raw_indices[5*flatidx_size + flatidx]]*
                            din[6][raw_indices[6*flatidx_size + flatidx]]*
                            din[7][raw_indices[7*flatidx_size + flatidx]];
            ++flatidx;
        }
    }
}


void IndexCachedExecutor::AddIndices()
{
    int num_loops = Kernel.GetNumLoops();
    int flatidx_size = Kernel.GetFlatIdxSize();
    const std::vector<int> N = Kernel.GetLoopDims();
    int raw_indices_size = (Kernel.GetNumVars() - 1)*flatidx_size;
    int *raw_indices = new int[raw_indices_size];
    switch (num_loops)
    {
        case 1: Build1LoopInd(raw_indices); break;
        case 2: Build2LoopInd(raw_indices); break;
        case 3: Build3LoopInd(raw_indices); break;
        case 4: Build4LoopInd(raw_indices); break;
        case 5: Build5LoopInd(raw_indices); break;
        case 6: Build6LoopInd(raw_indices); break;
        case 7: Build7LoopInd(raw_indices); break;
        case 8: Build8LoopInd(raw_indices); break;
        default: BuildArbitraryLoopInd(raw_indices);
    }
    RawIndexMap[N] = raw_indices;  
}


void IndexCachedExecutor::Build1LoopInd(int *raw_indices)
{
    int I[1];
    const std::vector<int> N = Kernel.GetLoopDims();
    int flatidx = 0;
    for (int ivari = 0; ivari < Kernel.GetNumInputVars(); ++ivari)
    {
        for (I[0] = 0; I[0] < N[0]; ++I[0])
        {
            raw_indices[flatidx] = ComputeRawIdx(*Kernel.GetInputTensor(ivari), I, Kernel.InputVars[ivari]->LoopNums);
            ++flatidx;
        }
    }
}


void IndexCachedExecutor::Build2LoopInd(int *raw_indices)
{
    int I[2];
    const std::vector<int> N = Kernel.GetLoopDims();
    int flatidx = 0;
    for (int ivari = 0; ivari < Kernel.GetNumInputVars(); ++ivari)
    {
        for (I[0] = 0; I[0] < N[0]; ++I[0])
        {
            for (I[1] = 0; I[1] < N[1]; ++I[1])
            {
                raw_indices[flatidx] = ComputeRawIdx(*Kernel.GetInputTensor(ivari), I, Kernel.InputVars[ivari]->LoopNums);
                ++flatidx;
            }
        }
    }
}


void IndexCachedExecutor::Build3LoopInd(int *raw_indices)
{
    int I[3];
    const std::vector<int> N = Kernel.GetLoopDims();
    int flatidx = 0;
    for (int ivari = 0; ivari < Kernel.GetNumInputVars(); ++ivari)
    {
        for (I[0] = 0; I[0] < N[0]; ++I[0])
        {
            for (I[1] = 0; I[1] < N[1]; ++I[1])
            {
                for (I[2] = 0; I[2] < N[2]; ++I[2])
                {
                    raw_indices[flatidx] = ComputeRawIdx(*Kernel.GetInputTensor(ivari), I, Kernel.InputVars[ivari]->LoopNums);
                    ++flatidx;
                }
            }
        }
    }
}


void IndexCachedExecutor::Build4LoopInd(int *raw_indices)
{
    int I[4];
    const std::vector<int> N = Kernel.GetLoopDims();
    int flatidx = 0;
    for (int ivari = 0; ivari < Kernel.GetNumInputVars(); ++ivari)
    {
        for (I[0] = 0; I[0] < N[0]; ++I[0])
        {
            for (I[1] = 0; I[1] < N[1]; ++I[1])
            {
                for (I[2] = 0; I[2] < N[2]; ++I[2])
                {
                    for (I[3] = 0; I[3] < N[3]; ++ I[3])
                    {
                        raw_indices[flatidx] = ComputeRawIdx(*Kernel.GetInputTensor(ivari), I, Kernel.InputVars[ivari]->LoopNums);
                        ++flatidx;
                    }
                }
            }
        }
    }
}


void IndexCachedExecutor::Build5LoopInd(int *raw_indices)
{
    int I[5];
    const std::vector<int> N = Kernel.GetLoopDims();
    int flatidx = 0;
    for (int ivari = 0; ivari < Kernel.GetNumInputVars(); ++ivari)
    {
        for (I[0] = 0; I[0] < N[0]; ++I[0])
        {
            for (I[1] = 0; I[1] < N[1]; ++I[1])
            {
                for (I[2] = 0; I[2] < N[2]; ++I[2])
                {
                    for (I[3] = 0; I[3] < N[3]; ++I[3])
                    {
                        for (I[4] = 0; I[4] < N[4]; ++I[4])
                        {    
                            raw_indices[flatidx] = ComputeRawIdx(*Kernel.GetInputTensor(ivari), I, Kernel.InputVars[ivari]->LoopNums);
                            ++flatidx;
                        }
                    }
                }
            }
        }
    }
}


void IndexCachedExecutor::Build6LoopInd(int *raw_indices)
{
    int I[6];
    const std::vector<int> N = Kernel.GetLoopDims();
    int flatidx = 0;
    for (int ivari = 0; ivari < Kernel.GetNumInputVars(); ++ivari)
    {
        for (I[0] = 0; I[0] < N[0]; ++I[0])
        {
            for (I[1] = 0; I[1] < N[1]; ++I[1])
            {
                for (I[2] = 0; I[2] < N[2]; ++I[2])
                {
                    for (I[3] = 0; I[3] < N[3]; ++I[3])
                    {
                        for (I[4] = 0; I[4] < N[4]; ++I[4])
                        {
                            for (I[5] = 0; I[5] < N[5]; ++I[5])
                            {
                                raw_indices[flatidx] = ComputeRawIdx(*Kernel.GetInputTensor(ivari), I, Kernel.InputVars[ivari]->LoopNums);
                                ++flatidx;
                            }
                        }
                    }
                }
            }
        }
    }
}


void IndexCachedExecutor::Build7LoopInd(int *raw_indices)
{
    int I[7];
    const std::vector<int> N = Kernel.GetLoopDims();
    int flatidx = 0;
    for (int ivari = 0; ivari < Kernel.GetNumInputVars(); ++ivari)
    {
        for (I[0] = 0; I[0] < N[0]; ++I[0])
        {
            for (I[1] = 0; I[1] < N[1]; ++I[1])
            {
                for (I[2] = 0; I[2] < N[2]; ++I[2])
                {
                    for (I[3] = 0; I[3] < N[3]; ++I[3])
                    {
                        for (I[4] = 0; I[4] < N[4]; ++I[4])
                        {
                            for (I[5] = 0; I[5] < N[5]; ++I[5])
                            {
                                for (I[6] = 0; I[6] < N[6]; ++I[6])
                                {
                                    raw_indices[flatidx] = ComputeRawIdx(*Kernel.GetInputTensor(ivari), I, Kernel.InputVars[ivari]->LoopNums);
                                    ++flatidx;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}


void IndexCachedExecutor::Build8LoopInd(int *raw_indices)
{
    int I[8];
    const std::vector<int> N = Kernel.GetLoopDims();
    int flatidx = 0;
    for (int ivari = 0; ivari < Kernel.GetNumInputVars(); ++ivari)
    {
        for (I[0] = 0; I[0] < N[0]; ++I[0])
        {
            for (I[1] = 0; I[1] < N[1]; ++I[1])
            {
                for (I[2] = 0; I[2] < N[2]; ++I[2])
                {
                    for (I[3] = 0; I[3] < N[3]; ++I[3])
                    {
                        for (I[4] = 0; I[4] < N[4]; ++I[4])
                        {
                            for (I[5] = 0; I[5] < N[5]; ++I[5])
                            {
                                for (I[6] = 0; I[6] < N[6]; ++I[6])
                                {
                                    for (I[7] = 0; I[7] < N[7]; ++I[7])
                                    {
                                        raw_indices[flatidx] = ComputeRawIdx(*Kernel.GetInputTensor(ivari), I, Kernel.InputVars[ivari]->LoopNums);
                                        ++flatidx;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}


void IndexCachedExecutor::BuildArbitraryLoopInd(int *raw_indices)
{
    std::vector<int> I(Kernel.GetNumLoops(), 0);     //Loop indices
    std::vector<int> W(Kernel.GetNumLoops());        //Loop strides
    const std::vector<int> N = Kernel.GetLoopDims();
    W[W.size()-1] = 1;
    for (int d = W.size() - 2; d >= 0; --d)
    {
        W[d] = W[d+1]*N[d+1];
    }

    int flatidx_size = Kernel.GetFlatIdxSize();
    for (int flatidx = 0; flatidx < flatidx_size; ++flatidx)
    {
        //Compute the unflattened indices
        for (int loopd = 0; loopd < I.size(); ++loopd)
        {
            I[loopd] = (flatidx / W[loopd]) % N[loopd];
        }

        for (int ivari = 0; ivari < Kernel.GetNumInputVars(); ++ivari)
        {
            raw_indices[ivari*flatidx_size + flatidx] = ComputeRawIdx((*Kernel.GetInputTensor(ivari)), I.data(), Kernel.InputVars[ivari]->LoopNums);
        }
    }
}

}