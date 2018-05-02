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
    NumLoops = FirstKernel->GetNumIndices();
    NumInVars = FirstKernel->GetNumInputVars();
    N = FirstKernel->GetLoopDims();

    OutputRank = FirstKernel->GetVarRank(-1);
    OutputLoopNums = &(FirstKernel->OutputVar.LoopNums[0]);
    OutputStrides = new int[OutputRank];
    for (int di = 0; di < OutputRank; ++di)
    {
        OutputStrides[di] = FirstKernel->GetVarDimStride(-1, di);
    }

    
    InputRanks = new int[NumInVars];
    InputLoopNums = new int*[NumInVars]; 
    InputStrides = new int*[NumInVars];
    InputVars = new double*[NumInVars];
    for (int vari = 0; vari < NumInVars; ++vari)
    {
        InputRanks[vari] = FirstKernel->GetVarRank(vari);
        InputLoopNums[vari] = &(FirstKernel->InputVars[vari].LoopNums[0]);
        InputStrides[vari] = new int[InputRanks[vari]];
        for (int di = 0; di < InputRanks[vari]; ++di)
        {
            InputStrides[vari][di] = FirstKernel->GetVarDimStride(vari, di);
        }
    }
}

CPUInterpretedExecutor::~CPUInterpretedExecutor()
{
    delete [] OutputStrides;
    delete [] InputRanks;
    delete [] InputLoopNums;
    delete [] InputVars;
    for (int vari = 0; vari < NumInVars; ++vari)
    {
        delete [] InputStrides[vari];
    }
    delete [] InputStrides;
}


void CPUInterpretedExecutor::ExecuteSingle(Tensor *output, std::vector<Tensor*> &inputs)
{
    MoveTensorsFromGPU(output, inputs);

    //Since we are using += or -= into the output 
    if (FirstKernel->EqOperator == "=")
    {
        output->Set(0.0);
    }

    OutputVar = output->GetData();
    for (int vari = 0; vari < NumInVars; ++vari)
    {
        InputVars[vari] = inputs[vari]->GetData();
    }

    switch (NumLoops)
    {
        case 1: Execute1Loops(); break;
        case 2: Execute2Loops(); break;
        case 3: Execute3Loops(); break;
        case 4: Execute4Loops(); break;
        case 5: Execute5Loops(); break;
        case 6: Execute6Loops(); break;
        case 7: Execute7Loops(); break;
        case 8: Execute8Loops(); break;
        case 9: Execute9Loops(); break;
        case 10: Execute10Loops(); break;
        case 11: Execute11Loops(); break;
        case 12: Execute12Loops(); break;
        default: ExecuteArbitraryLoops();
    }
}

std::string CPUInterpretedExecutor::GetImplementation()
{
    return "Interpreted\n";
}


void CPUInterpretedExecutor::Execute1Loops()
{
    int I[1];
    int &i0 = I[0];

    for (i0 = 0; i0 < N[0]; ++i0)
    {
        OutputVar[ComputeRawIdx(I, OutputLoopNums,OutputStrides, OutputRank)] += ComputeRHS(I);
    }
}


void CPUInterpretedExecutor::Execute2Loops()
{
    int I[2];
    for (I[0] = 0; I[0] < N[0]; ++I[0]) 
    {
        for (I[1] = 0; I[1] < N[1]; ++I[1]) 
        {
            OutputVar[ComputeRawIdx(I, OutputLoopNums,OutputStrides, OutputRank)] += ComputeRHS(I);
        }
    }
}


void CPUInterpretedExecutor::Execute3Loops()
{
    int I[3];
    for (I[0] = 0; I[0] < N[0]; ++I[0]) 
    {
        for (I[1] = 0; I[1] < N[1]; ++I[1])
        {
            for (I[2] = 0; I[2] < N[2]; ++I[2])
            {
                OutputVar[ComputeRawIdx(I, OutputLoopNums,OutputStrides, OutputRank)] += ComputeRHS(I);
            }
        }
    }    
}


void CPUInterpretedExecutor::Execute4Loops()
{
    int I[4];
    for (I[0] = 0; I[0] < N[0]; ++I[0]) 
    {
        for (I[1] = 0; I[1] < N[1]; ++I[1])
        {
            for (I[2] = 0; I[2] < N[2]; ++I[2])
            {
                for (I[3] = 0; I[3] < N[3]; ++I[3])
                {
                    OutputVar[ComputeRawIdx(I, OutputLoopNums,OutputStrides, OutputRank)] += ComputeRHS(I);
                }
            }
        }
    }    
}


void CPUInterpretedExecutor::Execute5Loops()
{
    int I[5];
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
                        OutputVar[ComputeRawIdx(I, OutputLoopNums,OutputStrides, OutputRank)] += ComputeRHS(I);
                    }
                }
            }
        }
    }
}


void CPUInterpretedExecutor::Execute6Loops()
{
    int I[6];
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
                            OutputVar[ComputeRawIdx(I, OutputLoopNums,OutputStrides, OutputRank)] += ComputeRHS(I);
                        }
                    }
                }
            }
        }
    }
}


void CPUInterpretedExecutor::Execute7Loops()
{
    int I[7];
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
                                OutputVar[ComputeRawIdx(I, OutputLoopNums,OutputStrides, OutputRank)] += ComputeRHS(I);
                            }
                        }
                    }
                }
            }
        }
    }    
}


void CPUInterpretedExecutor::Execute8Loops()
{
    int I[8];
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
                                    OutputVar[ComputeRawIdx(I, OutputLoopNums,OutputStrides, OutputRank)] += ComputeRHS(I);
                                }
                            }
                        }
                    }
                }
            }
        }
    }    
}


void CPUInterpretedExecutor::Execute9Loops()
{
    int I[9];
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
                                        OutputVar[ComputeRawIdx(I, OutputLoopNums,OutputStrides, OutputRank)] += ComputeRHS(I);
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


void CPUInterpretedExecutor::Execute10Loops()
{
    int I[10];
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
                                            OutputVar[ComputeRawIdx(I, OutputLoopNums,OutputStrides, OutputRank)] += ComputeRHS(I);
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


void CPUInterpretedExecutor::Execute11Loops()
{
    int I[11];
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
                                                OutputVar[ComputeRawIdx(I, OutputLoopNums,OutputStrides, OutputRank)] += ComputeRHS(I);
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


void CPUInterpretedExecutor::Execute12Loops()
{
    int I[12];
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
                                                    OutputVar[ComputeRawIdx(I, OutputLoopNums,OutputStrides, OutputRank)] += ComputeRHS(I);
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


void CPUInterpretedExecutor::ExecuteArbitraryLoops()
{
    std::vector<int> I(FirstKernel->GetNumIndices(), 0);     //Loop indices
    std::vector<int> W(FirstKernel->GetNumIndices());        //Loop strides
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
        OutputVar[ComputeRawIdx(I.data(), OutputLoopNums,OutputStrides, OutputRank)] += ComputeRHS(I.data());
    }
}

}