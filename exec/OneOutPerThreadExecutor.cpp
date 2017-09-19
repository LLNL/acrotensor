//Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at the Lawrence Livermore National Laboratory
//Written by Aaron Fisher (fisher47@llnl.gov). LLNL-CODE-738419.
//All rights reserved.
//This file is part of Acrotensor. For details, see https://github.com/LLNL/acrotensor.

#ifdef ACRO_HAVE_CUDA

#include "OneOutPerThreadExecutor.hpp"
#include <iostream>
#include <math.h>

namespace acro
{


OneOutPerThreadExecutor::OneOutPerThreadExecutor(std::string &kernelstr) : KernelExecutor(kernelstr) 
{
    cudaGetDeviceProperties(&CudaDeviceProp, 0);
}

OneOutPerThreadExecutor::~OneOutPerThreadExecutor()
{
    for (auto it = CudaKernelMap.begin(); it != CudaKernelMap.end(); ++it)
    {
        CudaKernel *cuda_kernel = it->second;
        acroCudaErrorCheck(cuModuleUnload(cuda_kernel->Module));
        delete cuda_kernel;
    }
    CudaKernelMap.clear();
}


void OneOutPerThreadExecutor::ExecuteKernel(Tensor *output, std::vector<Tensor*> &inputs)
{
    MoveTensorsToGPU(output, inputs);
    int num_loops = Kernel.GetNumLoops();
    Kernel.AttachTensors(output, inputs);
    std::vector<int> N = Kernel.GetLoopDims();
    ExecuteLoopsCuda();
}

std::string OneOutPerThreadExecutor::GetImplementation(Tensor *out, std::vector<Tensor*> &inputs)
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


void OneOutPerThreadExecutor::ExecuteLoopsCuda()
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


CudaKernel *OneOutPerThreadExecutor::GenerateCudaKernel()
{
    CudaKernel *cuda_kernel = new CudaKernel;
    int numloops = Kernel.GetNumLoops();
    int numoutloops = Kernel.GetNumOuterLoops();
    int numcontloops = Kernel.GetNumContractionLoops();
    int numinvars = Kernel.GetNumInputVars();
    const std::vector<int> N = Kernel.GetLoopDims();

    std::vector<int> Wout(numoutloops); //Outer loop strides
    if (numoutloops > 0) 
    {
        Wout[numoutloops-1] = 1;
    }
    for (int d = numoutloops - 2; d >= 0; --d)
    {
        Wout[d] = Wout[d+1]*N[d+1];
    }

    std::vector<int> Wcont(numcontloops); //Contraction loop strides
    if (numcontloops > 0)
    { 
        Wcont[numcontloops-1] = 1;
    }
    for (int d = numcontloops - 2; d >= 0; --d)
    {
        Wcont[d] = Wcont[d+1]*N[d+1+numoutloops];
    }

    int flatidx_size = Kernel.GetFlatIdxSize();
    int outidx_size = Kernel.GetOutIdxSize();
    int contidx_size = Kernel.GetContIdxSize();
    int unroll_size = contidx_size <= 8 ? contidx_size : 8;
    cuda_kernel->ThreadsPerBlock = 256;
    cuda_kernel->NumBlocks = outidx_size / cuda_kernel->ThreadsPerBlock + 1;

    cuda_kernel->Code = 
    "extern \"C\"  \n"
    "__global__ void <KERNEL_NAME>(double *dout, double **din)\n"
    "{\n"
    "    int I[<NUMLOOPS>];\n"
    "    double rhs_val;\n"
    "    double sum;\n"
    "    int outidx = (blockIdx.x*blockDim.x + threadIdx.x);\n"
    "    if (outidx < <OUTIDX_SIZE>) {\n"
    "        sum = 0.0;\n"
            "<COMPUTE_IOUT>"
            "<CONTLOOPS>"
    "            rhs_val = 1.0;\n"
                "<COMPUTE_RHS>"
    "            sum += rhs_val;\n"
            "<ENDCONTLOOPS>"
    "        dout[outidx] <OUT_EQ_OP> sum;\n"
    "    }\n"
    "}\n";

    //Generate the kernel name
    cuda_kernel->FunctionName = Kernel.GetCanonicalNameString();
    cuda_kernel->FunctionName += "dim";
    for (int loopd = 0; loopd < numloops; ++loopd)
    {
        cuda_kernel->FunctionName += "_" + std::to_string(N[loopd]);
    }

    //Generate the indices outside the contraction loop
    std::string computeIOutStr;
    for (int loopd = 0; loopd < numoutloops; ++loopd)
    {
        //I[loopd] = (outidx / (Wout[loopd]) % N[loopd];
        computeIOutStr += "        ";
        computeIOutStr += "I[" + std::to_string(loopd) + "] = ";
        if (Wout[loopd] == 1)
        {
            computeIOutStr += "outidx";
        }
        else
        {
            computeIOutStr += "(outidx / " + std::to_string(Wout[loopd]) + ")";
            cuda_kernel->IntOpsPerIndex += 1;
        }
        if (loopd > 0)
        {
            computeIOutStr += " % " + std::to_string(N[loopd]);
        }
        computeIOutStr += ";\n";
    }

    //Generate the unrolled I computation
    std::string contLoops;
    for (int loopd = numoutloops; loopd < numloops; ++loopd)
    {
        std::string temp;
        temp += "        #pragma unroll\n";
        temp += "        for (I[<LOOPD>] = 0; I[<LOOPD>] < <LOOP_SIZE>; ++I[<LOOPD>]) {\n";
        str_replace_all(temp, "<LOOPD>", loopd);
        str_replace_all(temp, "<LOOP_SIZE>", Kernel.GetLoopDim(loopd));
        contLoops += temp;
    }    
    std::string endContLoops = "        " + std::string(numcontloops, '}') + "\n";

    //Generate the RHS computation
    std::string computeRHSStr;
    for (int ivari = 0; ivari < numinvars; ++ ivari)
    {
        computeRHSStr += "            rhs_val *= __ldg(&din[" + std::to_string(ivari) + "][";
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
    str_replace_all(cuda_kernel->Code, "<OUTIDX_SIZE>", outidx_size);
    str_replace_all(cuda_kernel->Code, "<CONTIDX_SIZE>", contidx_size);
    str_replace_all(cuda_kernel->Code, "<UNROLL_SIZE>", unroll_size);
    str_replace_all(cuda_kernel->Code, "<COMPUTE_IOUT>", computeIOutStr);
    str_replace_all(cuda_kernel->Code, "<CONTLOOPS>", contLoops);
    str_replace_all(cuda_kernel->Code, "<ENDCONTLOOPS>", endContLoops);
    str_replace_all(cuda_kernel->Code, "<COMPUTE_RHS>", computeRHSStr);
    str_replace_all(cuda_kernel->Code, "<OUT_EQ_OP>", Kernel.EqOperator);

    //std::cout << Kernel.KernelStr << std::endl;
    //std::cout << cuda_kernel->Code << std::endl;
    cuda_kernel->GenerateFunction();

    return cuda_kernel;
}


void OneOutPerThreadExecutor::GetSharedMemInvars(const std::vector<int> &N, std::vector<bool> &sharedmem_invars)
{
    int numinvars = Kernel.GetNumInputVars();
    sharedmem_invars.resize(numinvars);
    int shared_dbls_remaining = CudaDeviceProp.sharedMemPerBlock / 8;
    for (int ivari = 0; ivari < numinvars; ++ivari)
    {
        int ivar_size = Kernel.GetVarSize(ivari);
        if (ivar_size < shared_dbls_remaining)
        {
            sharedmem_invars[ivari] = true;
            shared_dbls_remaining -= ivar_size;
        }
        else
        {
            sharedmem_invars[ivari] = false;
        }
    }
}

}

#endif