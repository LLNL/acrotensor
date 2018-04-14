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


OneOutPerThreadExecutor::OneOutPerThreadExecutor(DimensionedMultiKernel *multi_kernel) : KernelExecutor(multi_kernel) 
{
    cudaGetDeviceProperties(&CudaDeviceProp, 0);
    GenerateCudaKernel();
}

OneOutPerThreadExecutor::~OneOutPerThreadExecutor()
{
    acroCudaErrorCheck(cuModuleUnload(TheCudaKernel->Module));
    delete TheCudaKernel;
}


void OneOutPerThreadExecutor::ExecuteSingle(Tensor *output, std::vector<Tensor*> &inputs)
{
    MoveTensorsToGPU(output, inputs);

    int numinvars = FirstKernel->GetNumInputVars();
    int outidx_size = FirstKernel->GetOutIdxSize();
    const std::vector<int> N = FirstKernel->GetLoopDims();

    double *dout = output->GetDeviceData();
    double **din = new double*[numinvars];
    double **device_din;
    for (int ivari = 0; ivari < numinvars; ++ivari)
    {
        din[ivari] = inputs[ivari]->GetDeviceData();
    }
    acroCudaErrorCheck(cudaMalloc(&device_din, numinvars*sizeof(double*)));
    acroCudaErrorCheck(cudaMemcpy(device_din, din, numinvars*sizeof(double*), cudaMemcpyHostToDevice));

    std::vector<void*> kernelParams;
    kernelParams.push_back(&dout);
    kernelParams.push_back(&device_din);

    TheCudaKernel->Launch(kernelParams, TheCudaStream);

    acroCudaErrorCheck(cudaFree(device_din));
    delete [] din;
}

std::string OneOutPerThreadExecutor::GetImplementation()
{
    return TheCudaKernel->Code;
}


void OneOutPerThreadExecutor::GenerateCudaKernel()
{
    TheCudaKernel = new CudaKernel;
    int numloops = FirstKernel->GetNumIndices();
    int numoutloops = FirstKernel->GetNumOuterIndices();
    int numcontloops = FirstKernel->GetNumContractionIndices();
    int numinvars = FirstKernel->GetNumInputVars();
    const std::vector<int> N = FirstKernel->GetLoopDims();

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

    int flatidx_size = FirstKernel->GetFlatIdxSize();
    int outidx_size = FirstKernel->GetOutIdxSize();
    int contidx_size = FirstKernel->GetContIdxSize();
    int unroll_size = contidx_size <= 8 ? contidx_size : 8;
    TheCudaKernel->ThreadsPerBlock = 256;
    TheCudaKernel->NumBlocks = outidx_size / TheCudaKernel->ThreadsPerBlock + 1;

    TheCudaKernel->Code = 
    "extern \"C\"  \n"
    "__global__ void <KERNEL_NAME>(double *dout, double **din)\n"
    "{\n"
    "    double sum;\n"
    "    int outidx = (blockIdx.x*blockDim.x + threadIdx.x);\n"
    "\n"
        "<PRELOAD_SMVARS>"
    "\n"
    "    __syncthreads();\n"
    "    if (outidx < <OUTIDX_SIZE>) {\n"
    "        sum = 0.0;\n"
            "<COMPUTE_IOUT>"
    "\n"
            "<CONTLOOPS>"
    "            sum += <COMPUTE_RHS>;\n"
            "<ENDCONTLOOPS>"
    "        dout[outidx] <OUT_EQ_OP> sum;\n"
    "    }\n"
    "}\n";

    //Generate the kernel name
    TheCudaKernel->FunctionName = FirstKernel->GetDimensionedNameString();

    //If applicable Generate the SM preload code for small tensors
    std::vector<bool> sharedmem_invars;
    GetSharedMemInvars(sharedmem_invars);
    std::string computePreloadSMVarsStr;
    for (int ivari = 0; ivari < numinvars; ++ivari)
    {
        if (sharedmem_invars[ivari])
        {
            computePreloadSMVarsStr += "    __shared__ double sdin" + std::to_string(ivari);
            computePreloadSMVarsStr += "[" + std::to_string(FirstKernel->GetVarSize(ivari)) + "];\n";
        }
    }
    for (int ivari = 0; ivari < numinvars; ++ivari)
    {
        if (sharedmem_invars[ivari])
        {
            std::string temp = 
                "    for (int idx = threadIdx.x; idx < <SMVAR_SIZE>; idx += blockDim.x)\n"
                "    {\n"
                "        sdin<VARNUM>[idx] = din[<VARNUM>][idx];\n"
                "    }\n"
                "    __syncwarp();\n\n";
            str_replace_all(temp, "<SMVAR_SIZE>", FirstKernel->GetVarSize(ivari));
            str_replace_all(temp, "<VARNUM>", ivari);
            computePreloadSMVarsStr += temp;
        }
    }


    //Generate the indices outside the contraction loop
    std::string computeIOutStr;
    for (int loopd = 0; loopd < numoutloops; ++loopd)
    {
        //I[loopd] = (outidx / (Wout[loopd]) % N[loopd];
        computeIOutStr += "        ";
        computeIOutStr += "int I" + std::to_string(loopd) + " = ";
        if (Wout[loopd] == 1)
        {
            computeIOutStr += "outidx";
        }
        else
        {
            computeIOutStr += "(outidx / " + std::to_string(Wout[loopd]) + ")";
            TheCudaKernel->IntOpsPerIndex += 1;
        }
        if (loopd > 0)
        {
            computeIOutStr += " % " + std::to_string(N[loopd]);
        }
        computeIOutStr += ";\n";
    }

    //Generate the contraction loops
    std::string contLoops;
    std::vector<bool> hoisted(FirstKernel->GetNumInputVars(), false);
    for (int loopd = numoutloops; loopd < numloops; ++loopd)
    {
        std::string temp;
        for (int ivari = 0; ivari < FirstKernel->GetNumInputVars(); ++ ivari)
        {
            if (FirstKernel->GetVarLoopDepth(ivari) < loopd && !sharedmem_invars[ivari] && !hoisted[ivari])
            {
                std::string ivaristr = std::to_string(ivari);
                std::string varidxstr = GetVarIndexString(ivari);             
                temp += "        double din" + ivaristr + " = __ldg(&din[" + ivaristr + "][" + varidxstr + "]);\n";
                hoisted[ivari] = true;
            }
        }

        temp += "        #pragma unroll\n";
        temp += "        for (int I<LOOPD> = 0; I<LOOPD> < <LOOP_SIZE>; ++I<LOOPD>) {\n";
        str_replace_all(temp, "<LOOPD>", loopd);
        str_replace_all(temp, "<LOOP_SIZE>", FirstKernel->GetLoopDim(loopd));
        contLoops += temp;
    }
    std::string endContLoops = "        " + std::string(numcontloops, '}') + "\n";

    //Generate the RHS computation inside the contraction loops
    std::string computeRHSStr;
    for (int ivari = 0; ivari < numinvars; ++ ivari)
    {
        if (sharedmem_invars[ivari])
        {
            computeRHSStr += "sdin" + std::to_string(ivari) = "[" + GetVarIndexString(ivari) + "]";
        }
        else if (hoisted[ivari])
        {
            computeRHSStr += "din" + std::to_string(ivari);
        }
        else
        {
            computeRHSStr += "__ldg(&din[" + std::to_string(ivari) + "][" + GetVarIndexString(ivari) + "])";
        }
        if (ivari < numinvars-1)
        {
            computeRHSStr += "*";
        }
    }

    str_replace_all(TheCudaKernel->Code, "<KERNEL_NAME>", TheCudaKernel->FunctionName);
    str_replace_all(TheCudaKernel->Code, "<NUMINVARS>", FirstKernel->GetNumInputVars());
    str_replace_all(TheCudaKernel->Code, "<NUMLOOPS>", FirstKernel->GetNumIndices());
    str_replace_all(TheCudaKernel->Code, "<OUTIDX_SIZE>", outidx_size);
    str_replace_all(TheCudaKernel->Code, "<CONTIDX_SIZE>", contidx_size);
    str_replace_all(TheCudaKernel->Code, "<UNROLL_SIZE>", unroll_size);
    str_replace_all(TheCudaKernel->Code, "<PRELOAD_SMVARS>", computePreloadSMVarsStr);
    str_replace_all(TheCudaKernel->Code, "<COMPUTE_IOUT>", computeIOutStr);
    str_replace_all(TheCudaKernel->Code, "<CONTLOOPS>", contLoops);
    str_replace_all(TheCudaKernel->Code, "<ENDCONTLOOPS>", endContLoops);
    str_replace_all(TheCudaKernel->Code, "<COMPUTE_RHS>", computeRHSStr);
    str_replace_all(TheCudaKernel->Code, "<OUT_EQ_OP>", FirstKernel->EqOperator);

    //std::cout << FirstKernel->KernelStr << std::endl;
    //std::cout << TheCudaKernel->Code << std::endl;
    TheCudaKernel->GenerateFunction();
}


void OneOutPerThreadExecutor::GetSharedMemInvars(std::vector<bool> &sharedmem_invars)
{
    int numinvars = FirstKernel->GetNumInputVars();
    int numoutloops = FirstKernel->GetNumOuterIndices();
    sharedmem_invars.resize(numinvars);
    int num_blocks_per_full_sm = CudaDeviceProp.maxThreadsPerMultiProcessor / 256;
    int shared_dbls_remaining = (CudaDeviceProp.sharedMemPerMultiprocessor / num_blocks_per_full_sm) / 8;
    for (int ivari = 0; ivari < numinvars; ++ivari)
    {
        sharedmem_invars[ivari] = false;
        int ivar_size = FirstKernel->GetVarSize(ivari);
        if (false && ivar_size < shared_dbls_remaining)
        {
            sharedmem_invars[ivari] = true;
            shared_dbls_remaining -= ivar_size;
        }
    }
}


std::string OneOutPerThreadExecutor::GetVarIndexString(int vari)
{
    std::string indexStr = "I" + std::to_string(FirstKernel->GetVarDimLoopNum(vari, 0)) + "*" +
                            std::to_string(FirstKernel->GetVarDimStride(vari, 0));                         
    for (int d = 1; d < FirstKernel->GetVarRank(vari); ++d)
    {
        indexStr += " + ";
        indexStr += "I" + std::to_string(FirstKernel->GetVarDimLoopNum(vari, d)) + "*" +
                          std::to_string(FirstKernel->GetVarDimStride(vari, d));                                   
    }
    return indexStr;
}

}

#endif