//Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at the Lawrence Livermore National Laboratory
//Written by Aaron Fisher (fisher47@llnl.gov). LLNL-CODE-738419.
//All rights reserved.
//This file is part of Acrotensor. For details, see https://github.com/LLNL/acrotensor.

#ifdef ACRO_HAVE_CUDA

#include "MultiOutPerThreadExecutor.hpp"
#include <iostream>
#include <math.h>

namespace acro
{


MultiOutPerThreadExecutor::MultiOutPerThreadExecutor(std::string &kernelstr) : KernelExecutor(kernelstr) 
{
    cudaGetDeviceProperties(&CudaDeviceProp, 0);
}

MultiOutPerThreadExecutor::~MultiOutPerThreadExecutor()
{
    for (auto it = CudaKernelMap.begin(); it != CudaKernelMap.end(); ++it)
    {
        CudaKernel *cuda_kernel = it->second;
        acroCudaErrorCheck(cuModuleUnload(cuda_kernel->Module));
        delete cuda_kernel;
    }
    CudaKernelMap.clear();
}


void MultiOutPerThreadExecutor::ExecuteKernel(Tensor *output, std::vector<Tensor*> &inputs)
{
    MoveTensorsToGPU(output, inputs);
    int num_loops = Kernel.GetNumLoops();
    Kernel.AttachTensors(output, inputs);
    std::vector<int> N = Kernel.GetLoopDims();
    ExecuteLoopsCuda();
}

std::string MultiOutPerThreadExecutor::GetImplementation(Tensor *out, std::vector<Tensor*> &inputs)
{
    Kernel.AttachTensors(out, inputs);
    const std::vector<int> N = Kernel.GetLoopDims();

    auto it = CudaKernelMap.find(N);
    if (it == CudaKernelMap.end())
    {
        CudaKernelMap[N] = GenerateCudaKernel();
    }
    CudaKernel *cuda_kernel = CudaKernelMap[N];

    return cuda_kernel->Code;
}


void MultiOutPerThreadExecutor::ExecuteLoopsCuda()
{
    int numinvars = Kernel.GetNumInputVars();
    int outidx_size = Kernel.GetOutIdxSize();
    const std::vector<int> N = Kernel.GetLoopDims();
    Tensor *output = Kernel.GetOutputTensor();
    std::vector<Tensor*> inputs = Kernel.GetInputTensors();

    auto it = CudaKernelMap.find(N);
    if (it == CudaKernelMap.end())
    {
        CudaKernelMap[N] = GenerateCudaKernel();
    }
    CudaKernel *cuda_kernel = CudaKernelMap[N];

    
    double *dout = Kernel.GetOutputTensor()->GetDeviceData();
    double **din = new double*[numinvars];
    double **device_din;
    for (int ivari = 0; ivari < numinvars; ++ivari)
    {
        din[ivari] = Kernel.GetInputTensor(ivari)->GetDeviceData();
    }
    acroCudaErrorCheck(cudaMalloc(&device_din, numinvars*sizeof(double*)));
    acroCudaErrorCheck(cudaMemcpy(device_din, din, numinvars*sizeof(double*), cudaMemcpyHostToDevice));

    std::vector<void*> kernelParams;
    kernelParams.push_back(&dout);
    kernelParams.push_back(&device_din);

    cuda_kernel->Launch(kernelParams, TheCudaStream);

    acroCudaErrorCheck(cudaFree(device_din));
    delete [] din;
}


CudaKernel *MultiOutPerThreadExecutor::GenerateCudaKernel()
{
    CudaKernel *cuda_kernel = new CudaKernel;
    int num_loops = Kernel.GetNumLoops();
    int numinvars = Kernel.GetNumInputVars();
    int contidx_size = Kernel.GetContIdxSize();
    int unroll_size = contidx_size <= 16 ? contidx_size : 16;
    cuda_kernel->ThreadsPerBlock = 256;
    int num_mp = CudaDeviceProp.multiProcessorCount;
    int blocks_per_mp = CudaDeviceProp.maxThreadsPerMultiProcessor / std::max(cuda_kernel->ThreadsPerBlock, 128);

    //Decide on the number of loops of the 3 types
    int num_inner_loops = Kernel.GetNumContractionLoops();
    int num_outer_loops, num_middle_loops;
    num_outer_loops = num_loops - num_inner_loops - 0;
    // for (num_outer_loops = 0; num_outer_loops < num_loops - num_inner_loops; ++num_outer_loops)
    // {
    //     if (Kernel.GetIdxSizeForFirstNumLoops(num_outer_loops) > 100*num_mp*blocks_per_mp)
    //     {
    //         break;
    //     } 
    // }
    num_middle_loops = num_loops - num_inner_loops - num_outer_loops;
    int outeridx_size = Kernel.GetIdxSizeForFirstNumLoops(num_outer_loops);
    int mididx_size = Kernel.GetIdxSizeForFirstNumLoops(num_outer_loops + num_middle_loops) / outeridx_size;
    cuda_kernel->NumBlocks = outeridx_size / cuda_kernel->ThreadsPerBlock + 1;

    const std::vector<int> N = Kernel.GetLoopDims();
    std::vector<int> Wout(num_outer_loops); //Outer loop strides
    if (num_outer_loops > 0) 
    {
        Wout[num_outer_loops-1] = 1;
    }
    for (int d = num_outer_loops - 2; d >= 0; --d)
    {
        Wout[d] = Wout[d+1]*N[d+1];
    }

    cuda_kernel->Code = 
    "extern \"C\"  \n"
    "__global__ void <KERNEL_NAME>(double *dout, double **din)\n"
    "{\n"
    "    int I[<NUMLOOPS>];\n"
    "    double rhs_val;\n"
    "    double sum;\n"
    "    int tidx = (blockIdx.x*blockDim.x + threadIdx.x);\n"
    "    int outidx = tidx*<MIDIDX_SIZE>;\n"
    "    if (tidx < <TIDX_SIZE>) {\n"
            "<COMPUTE_IOUTER>"
            "<MIDLOOPS>"
    "            sum = 0.0;\n"
                "<CONTLOOPS>"
    "                rhs_val = 1.0;\n"
                    "<COMPUTE_RHS>"
    "                sum += rhs_val;\n"
                "<ENDCONTLOOPS>"
    "            dout[outidx] <OUT_EQ_OP> sum;\n"
    "            ++outidx;\n"
            "<ENDMIDLOOPS>"
    "    }\n"
    "}\n";

    //Generate the kernel name
    cuda_kernel->FunctionName = Kernel.GetCanonicalNameString();
    cuda_kernel->FunctionName += "dim";
    for (int loopd = 0; loopd < num_loops; ++loopd)
    {
        cuda_kernel->FunctionName += "_" + std::to_string(N[loopd]);
    }

    //Generate the indices outside the contraction loop
    std::string computeIOutStr;
    for (int loopd = 0; loopd < num_outer_loops; ++loopd)
    {
        //I[loopd] = (outidx / (Wout[loopd]) % N[loopd];
        computeIOutStr += "        ";
        computeIOutStr += "I[" + std::to_string(loopd) + "] = ";
        if (Wout[loopd] == 1)
        {
            computeIOutStr += "tidx";
        }
        else
        {
            computeIOutStr += "(tidx / " + std::to_string(Wout[loopd]) + ")";
            cuda_kernel->IntOpsPerIndex += 1;
        }
        if (loopd > 0)
        {
            computeIOutStr += " % " + std::to_string(N[loopd]);
        }
        computeIOutStr += ";\n";
    }

    //Generate the unrolled I computation
    std::string midLoops;
    std::vector<bool> hoisted(Kernel.GetNumInputVars(), false);
    for (int loopd = num_outer_loops; loopd < num_outer_loops + num_middle_loops; ++loopd)
    {
        std::string temp;
        for (int ivari = 0; ivari < Kernel.GetNumInputVars(); ++ ivari)
        {
            if (Kernel.GetVarLoopDepth(ivari) < loopd && !hoisted[ivari])
            {
                std::string ivaristr = std::to_string(ivari);
                std::string varidxstr = "I[" + std::to_string(Kernel.GetVarDimLoopNum(ivari, 0)) + "]*" +
                                        std::to_string(Kernel.GetVarDimStride(ivari, 0));
                for (int d = 1; d < Kernel.GetVarRank(ivari); ++d)
                {
                    varidxstr += " + ";
                    varidxstr += "I[" + std::to_string(Kernel.GetVarDimLoopNum(ivari, d)) + "]*" +
                                 std::to_string(Kernel.GetVarDimStride(ivari, d));
                }               

                temp += "        double din" + ivaristr + " = __ldg(&din[" + ivaristr + "][" + varidxstr + "]);\n";
                hoisted[ivari] = true;
            }
        }

        temp += "        #pragma unroll\n";
        temp += "        for (I[<LOOPD>] = 0; I[<LOOPD>] < <LOOP_SIZE>; ++I[<LOOPD>]) {\n";
        str_replace_all(temp, "<LOOPD>", loopd);
        str_replace_all(temp, "<LOOP_SIZE>", Kernel.GetLoopDim(loopd));
        midLoops += temp;
    }    
    std::string endMidLoops = "        " + std::string(num_middle_loops, '}') + "\n";

    //Generate the unrolled I computation
    std::string contLoops;
    for (int loopd = num_middle_loops + num_outer_loops; loopd < num_loops; ++loopd)
    {
        std::string temp;
        for (int ivari = 0; ivari < Kernel.GetNumInputVars(); ++ ivari)
        {
            if (Kernel.GetVarLoopDepth(ivari) < loopd && !hoisted[ivari])
            {
                std::string ivaristr = std::to_string(ivari);
                std::string varidxstr = "I[" + std::to_string(Kernel.GetVarDimLoopNum(ivari, 0)) + "]*" +
                                        std::to_string(Kernel.GetVarDimStride(ivari, 0));
                for (int d = 1; d < Kernel.GetVarRank(ivari); ++d)
                {
                    varidxstr += " + ";
                    varidxstr += "I[" + std::to_string(Kernel.GetVarDimLoopNum(ivari, d)) + "]*" +
                                 std::to_string(Kernel.GetVarDimStride(ivari, d));
                }               

                temp += "        double din" + ivaristr + " = __ldg(&din[" + ivaristr + "][" + varidxstr + "]);\n";
                hoisted[ivari] = true;
            }
        }

        temp += "        #pragma unroll\n";
        temp += "        for (I[<LOOPD>] = 0; I[<LOOPD>] < <LOOP_SIZE>; ++I[<LOOPD>]) {\n";
        str_replace_all(temp, "<LOOPD>", loopd);
        str_replace_all(temp, "<LOOP_SIZE>", Kernel.GetLoopDim(loopd));
        contLoops += temp;
    }
    std::string endContLoops = "        " + std::string(num_inner_loops, '}') + "\n";

    //Generate the RHS computation
    std::string computeRHSStr;
    for (int ivari = 0; ivari < numinvars; ++ ivari)
    {
        computeRHSStr += "                rhs_val *= __ldg(&din[" + std::to_string(ivari) + "][";
        computeRHSStr += "I[" + std::to_string(Kernel.GetVarDimLoopNum(ivari, 0)) + "]*" +
                          std::to_string(Kernel.GetVarDimStride(ivari, 0));
        for (int d = 1; d < Kernel.GetVarRank(ivari); ++d)
        {
            computeRHSStr += " + ";
            computeRHSStr += "I[" + std::to_string(Kernel.GetVarDimLoopNum(ivari, d)) + "]*" +
                              std::to_string(Kernel.GetVarDimStride(ivari, d));
        }
        computeRHSStr += "]);\n";
    }

    str_replace_all(cuda_kernel->Code, "<KERNEL_NAME>", cuda_kernel->FunctionName);
    str_replace_all(cuda_kernel->Code, "<NUMINVARS>", Kernel.GetNumInputVars());
    str_replace_all(cuda_kernel->Code, "<NUMLOOPS>", Kernel.GetNumLoops());
    str_replace_all(cuda_kernel->Code, "<TIDX_SIZE>", outeridx_size);
    str_replace_all(cuda_kernel->Code, "<MIDIDX_SIZE>", mididx_size);
    str_replace_all(cuda_kernel->Code, "<CONTIDX_SIZE>", contidx_size);
    str_replace_all(cuda_kernel->Code, "<UNROLL_SIZE>", unroll_size);
    str_replace_all(cuda_kernel->Code, "<COMPUTE_IOUTER>", computeIOutStr);
    str_replace_all(cuda_kernel->Code, "<MIDLOOPS>", midLoops);
    str_replace_all(cuda_kernel->Code, "<CONTLOOPS>", contLoops);
    str_replace_all(cuda_kernel->Code, "<ENDCONTLOOPS>", endContLoops);
    str_replace_all(cuda_kernel->Code, "<ENDMIDLOOPS>", endMidLoops);
    str_replace_all(cuda_kernel->Code, "<COMPUTE_RHS>", computeRHSStr);
    str_replace_all(cuda_kernel->Code, "<OUT_EQ_OP>", Kernel.EqOperator);

    //std::cout << Kernel.KernelStr << std::endl;
    //std::cout << cuda_kernel->Code << std::endl;
    cuda_kernel->GenerateFunction();

    return cuda_kernel;
}

}

#endif