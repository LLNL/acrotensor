//Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at the Lawrence Livermore National Laboratory
//Written by Aaron Fisher (fisher47@llnl.gov). LLNL-CODE-738419.
//All rights reserved.
//This file is part of Acrotensor. For details, see https://github.com/LLNL/acrotensor.

#include "CPUInterpretedExecutor.hpp"
#include <iostream>
#include <math.h>

namespace acro
{


CPUInterpretedExecutor::CPUInterpretedExecutor(DimensionedMultiKernel *multi_kernel) : KernelExecutor(multi_kernel) 
{

}

CPUInterpretedExecutor::~CPUInterpretedExecutor()
{

}


void CPUInterpretedExecutor::ExecuteSingle(Tensor *output, std::vector<Tensor*> &inputs)
{
    MoveTensorsFromGPU(output, inputs);

    int num_loops = FirstKernel->GetNumIndices();
    std::vector<int> N = FirstKernel->GetLoopDims();

    //Since we are using += or -= into the output 
    if (FirstKernel->EqOperator == "=")
    {
        output->Set(0.0);
    }

    switch (num_loops)
    {
        case 1: Execute1Loops(output, inputs); break;
        case 2: Execute2Loops(output, inputs); break;
        case 3: Execute3Loops(output, inputs); break;
        case 4: Execute4Loops(output, inputs); break;
        case 5: Execute5Loops(output, inputs); break;
        case 6: Execute6Loops(output, inputs); break;
        case 7: Execute7Loops(output, inputs); break;
        case 8: Execute8Loops(output, inputs); break;
        case 9: Execute9Loops(output, inputs); break;
        case 10: Execute10Loops(output, inputs); break;
        case 11: Execute11Loops(output, inputs); break;
        case 12: Execute12Loops(output, inputs); break;
        default: ExecuteArbitraryLoops(output, inputs);
    }
}

std::string CPUInterpretedExecutor::GetImplementation()
{
    return "Interpreted\n";
}


void CPUInterpretedExecutor::Execute1Loops(Tensor *output, std::vector<Tensor*> &inputs)
{
    int I[1];
    const std::vector<int> N = FirstKernel->GetLoopDims();
    int &i0 = I[0];


    for (i0 = 0; i0 < N[0]; ++i0)
    {
        (*output)[ComputeRawIdx(*output, I, FirstKernel->OutputVar.LoopNums)] += ComputeRHS(inputs, I);
    }
}


void CPUInterpretedExecutor::Execute2Loops(Tensor *output, std::vector<Tensor*> &inputs)
{
    int I[2];
    const std::vector<int> N = FirstKernel->GetLoopDims();
    for (I[0] = 0; I[0] < N[0]; ++I[0]) 
    {
        for (I[1] = 0; I[1] < N[1]; ++I[1]) 
        {
            (*output)[ComputeRawIdx(*output, I, FirstKernel->OutputVar.LoopNums)] += ComputeRHS(inputs, I);
        }
    }
}


void CPUInterpretedExecutor::Execute3Loops(Tensor *output, std::vector<Tensor*> &inputs)
{
    int I[3];
    const std::vector<int> N = FirstKernel->GetLoopDims();
    for (I[0] = 0; I[0] < N[0]; ++I[0]) 
    {
        for (I[1] = 0; I[1] < N[1]; ++I[1])
        {
            for (I[2] = 0; I[2] < N[2]; ++I[2])
            {
                (*output)[ComputeRawIdx(*output, I, FirstKernel->OutputVar.LoopNums)] += ComputeRHS(inputs, I);
            }
        }
    }    
}


void CPUInterpretedExecutor::Execute4Loops(Tensor *output, std::vector<Tensor*> &inputs)
{
    int I[4];
    const std::vector<int> N = FirstKernel->GetLoopDims();
    for (I[0] = 0; I[0] < N[0]; ++I[0]) 
    {
        for (I[1] = 0; I[1] < N[1]; ++I[1])
        {
            for (I[2] = 0; I[2] < N[2]; ++I[2])
            {
                for (I[3] = 0; I[3] < N[3]; ++I[3])
                {
                    (*output)[ComputeRawIdx(*output, I, FirstKernel->OutputVar.LoopNums)] += ComputeRHS(inputs, I);
                }
            }
        }
    }    
}


void CPUInterpretedExecutor::Execute5Loops(Tensor *output, std::vector<Tensor*> &inputs)
{
    int I[5];
    const std::vector<int> N = FirstKernel->GetLoopDims();
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
                        (*output)[ComputeRawIdx(*output, I, FirstKernel->OutputVar.LoopNums)] += ComputeRHS(inputs, I);
                    }
                }
            }
        }
    }
}


void CPUInterpretedExecutor::Execute6Loops(Tensor *output, std::vector<Tensor*> &inputs)
{
    int I[6];
    const std::vector<int> N = FirstKernel->GetLoopDims();
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
                            (*output)[ComputeRawIdx(*output, I, FirstKernel->OutputVar.LoopNums)] += ComputeRHS(inputs, I);
                        }
                    }
                }
            }
        }
    }
}


void CPUInterpretedExecutor::Execute7Loops(Tensor *output, std::vector<Tensor*> &inputs)
{
    int I[7];
    const std::vector<int> N = FirstKernel->GetLoopDims();
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
                                (*output)[ComputeRawIdx(*output, I, FirstKernel->OutputVar.LoopNums)] += ComputeRHS(inputs, I);
                            }
                        }
                    }
                }
            }
        }
    }    
}


void CPUInterpretedExecutor::Execute8Loops(Tensor *output, std::vector<Tensor*> &inputs)
{
    int I[8];
    const std::vector<int> N = FirstKernel->GetLoopDims();
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
                                    (*output)[ComputeRawIdx(*output, I, FirstKernel->OutputVar.LoopNums)] += ComputeRHS(inputs, I);
                                }
                            }
                        }
                    }
                }
            }
        }
    }    
}


void CPUInterpretedExecutor::Execute9Loops(Tensor *output, std::vector<Tensor*> &inputs)
{
    int I[9];
    const std::vector<int> N = FirstKernel->GetLoopDims();
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
                                    for (I[8] = 0; I[8] < N[8]; ++I[8])
                                    {                                    
                                        (*output)[ComputeRawIdx(*output, I, FirstKernel->OutputVar.LoopNums)] += ComputeRHS(inputs, I);
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


void CPUInterpretedExecutor::Execute10Loops(Tensor *output, std::vector<Tensor*> &inputs)
{
    int I[10];
    const std::vector<int> N = FirstKernel->GetLoopDims();
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
                                    for (I[8] = 0; I[8] < N[8]; ++I[8])
                                    {                                    
                                        for (I[9] = 0; I[9] < N[9]; ++I[9])
                                        {                                    
                                            (*output)[ComputeRawIdx(*output, I, FirstKernel->OutputVar.LoopNums)] += ComputeRHS(inputs, I);
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
}


void CPUInterpretedExecutor::Execute11Loops(Tensor *output, std::vector<Tensor*> &inputs)
{
    int I[11];
    const std::vector<int> N = FirstKernel->GetLoopDims();
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
                                    for (I[8] = 0; I[8] < N[8]; ++I[8])
                                    {                                    
                                        for (I[9] = 0; I[9] < N[9]; ++I[9])
                                        {
                                            for (I[10] = 0; I[10] < N[10]; ++I[10])
                                            {
                                                (*output)[ComputeRawIdx(*output, I, FirstKernel->OutputVar.LoopNums)] += ComputeRHS(inputs, I);
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
    }    
}


void CPUInterpretedExecutor::Execute12Loops(Tensor *output, std::vector<Tensor*> &inputs)
{
    int I[12];
    const std::vector<int> N = FirstKernel->GetLoopDims();
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
                                    for (I[8] = 0; I[8] < N[8]; ++I[8])
                                    {                                    
                                        for (I[9] = 0; I[9] < N[9]; ++I[9])
                                        {
                                            for (I[10] = 0; I[10] < N[10]; ++I[10])
                                            {
                                                for (I[11] = 0; I[11] < N[11]; ++I[11])
                                                {
                                                    (*output)[ComputeRawIdx(*output, I, FirstKernel->OutputVar.LoopNums)] += ComputeRHS(inputs, I);
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
        }
    }    
}


void CPUInterpretedExecutor::ExecuteArbitraryLoops(Tensor *output, std::vector<Tensor*> &inputs)
{
    std::vector<int> I(FirstKernel->GetNumIndices(), 0);     //Loop indices
    std::vector<int> W(FirstKernel->GetNumIndices());        //Loop strides
    const std::vector<int> N = FirstKernel->GetLoopDims();
    W[W.size()-1] = 1;
    for (int d = W.size() - 2; d >= 0; --d)
    {
        W[d] = W[d+1]*N[d+1];
    }

    int flatidx_size = 1;
    for (int d = 0; d < W.size(); ++d)
    {
        flatidx_size *= N[d];
    }

    for (int flatidx = 0; flatidx < flatidx_size; ++flatidx)
    {
        //Compute the unflattened indices
        for (int loopd = 0; loopd < I.size(); ++loopd)
        {
            I[loopd] = (flatidx / W[loopd]) % N[loopd];
        }
        (*output)[ComputeRawIdx(*output, I.data(), FirstKernel->OutputVar.LoopNums)] += ComputeRHS(inputs, I.data());
    }
}

}