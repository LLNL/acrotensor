//Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at the Lawrence Livermore National Laboratory
//Written by Aaron Fisher (fisher47@llnl.gov). LLNL-CODE-738419.
//All rights reserved.
//This file is part of Acrotensor. For details, see https://github.com/LLNL/acrotensor.

#ifdef ACRO_HAVE_CUDA

#include "OneOutPerThreadExecutor.hpp"
#include <algorithm>

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

    int numuvars = MultiKernel->GetNumUVars();
    double **tptrs = new double*[numuvars];
    std::vector<void*> kernelParams;
    for (int uvari = 0; uvari < numuvars; ++uvari)
    {
        auto ki_vari = MultiKernel->GetFirstKiVariForUVari(uvari);
        int vari = ki_vari.second;
        if (vari == -1)
        {
            tptrs[uvari] = output->GetDeviceData();
        }
        else
        {
            tptrs[uvari] = inputs[vari]->GetDeviceData();
        }
        kernelParams.push_back(&(tptrs[uvari]));
    }

    // double **device_tptrs;
    // acroCudaErrorCheck(cudaMalloc(&device_tptrs, numuvars*sizeof(double*)));
    // acroCudaErrorCheck(cudaMemcpy(device_tptrs, tptrs, numuvars*sizeof(double*), cudaMemcpyHostToDevice));

    // std::vector<void*> kernelParams;
    // kernelParams.push_back(&tptrs);
    TheCudaKernel->Launch(kernelParams);
    // cudaDeviceSynchronize();

    // acroCudaErrorCheck(cudaFree(device_tptrs));
    delete [] tptrs;
}


void OneOutPerThreadExecutor::ExecuteMulti(std::vector<Tensor*> &outputs, std::vector<std::vector<Tensor*> > &inputs)
{
    for (int ki = 0; ki < MultiKernel->GetNumKernels(); ++ki)
    {
        MoveTensorsToGPU(outputs[ki], inputs[ki]);
    }

    int numuvars = MultiKernel->GetNumUVars();
    double **tptrs = new double*[numuvars];
    std::vector<void*> kernelParams;
    for (int uvari = 0; uvari < numuvars; ++uvari)
    {
        auto ki_vari = MultiKernel->GetFirstKiVariForUVari(uvari);
        int ki = ki_vari.first;
        int vari = ki_vari.second;
        if (vari == -1)
        {
            tptrs[uvari] = outputs[ki]->GetDeviceData();
        }
        else
        {
            tptrs[uvari] = inputs[ki][vari]->GetDeviceData();
        }
        kernelParams.push_back(&(tptrs[uvari]));
    }

    // double **device_tptrs;
    // acroCudaErrorCheck(cudaMalloc(&device_tptrs, numuvars*sizeof(double*)));
    // acroCudaErrorCheck(cudaMemcpy(device_tptrs, tptrs, numuvars*sizeof(double*), cudaMemcpyHostToDevice));

    
    //kernelParams.push_back(&tptrs);
    TheCudaKernel->Launch(kernelParams, TheCudaStream);

    // acroCudaErrorCheck(cudaFree(device_tptrs));
    delete [] tptrs;
}


std::string OneOutPerThreadExecutor::GetImplementation()
{
    return TheCudaKernel->Code;
}


void OneOutPerThreadExecutor::GenerateCudaKernel()
{
    TheCudaKernel = new CudaKernel;
    TheCudaKernel->Code = 
    "extern \"C\"  \n"
    "__global__ void <KERNEL_NAME>(<PARAMS>)\n"
    "{\n"
    "    double sum;\n"
    "    int outidx = (blockIdx.x*blockDim.x + threadIdx.x);\n"
    "\n"
        "<PRELOAD_SMVARS>"
    "\n"
    "    __syncthreads();\n"
    "    if (outidx < <OUTIDX_SIZE>) {\n"
            "<COMPUTE_IOUT>"
    "\n"
            "<SUBKERNEL_LOOPS>"
    "    }\n"
    "}\n";

    //Generate the kernel name
    TheCudaKernel->FunctionName = "Kernel";

    //Generate the params list
    std::string params_str;
    for (int uvari = 0; uvari < MultiKernel->GetNumUVars(); ++uvari)
    {
        params_str += "double *T" + std::to_string(uvari);
        if (uvari < MultiKernel->GetNumUVars()-1)
        {
            params_str += ", ";
        }
    }

    int numouterloops = MultiKernel->GetNumOuterIndices();
    ACROBATIC_ASSERT(numouterloops > 0, "OneOutPerThreadExecutor needs at least 1 non-contraction index.");

    int outidx_size = MultiKernel->GetSharedOuterIdxSize();
    TheCudaKernel->ThreadsPerBlock = 256;
    TheCudaKernel->NumBlocks = outidx_size / TheCudaKernel->ThreadsPerBlock + 1;

    std::vector<bool> sharedmem_uvars = GetSharedMemUvars();
    std::string preload_sm_str = GenSharedMemPreload(sharedmem_uvars);

    //Generate the indices outside the contraction loop
    std::string iout_str = GenIOutVars();

    //Generate the subkernel loops
    std::string subkernel_loops_str = GenSubKernelLoops(sharedmem_uvars);

    str_replace_all(TheCudaKernel->Code, "<KERNEL_NAME>", TheCudaKernel->FunctionName);
    str_replace_all(TheCudaKernel->Code, "<PARAMS>", params_str);
    str_replace_all(TheCudaKernel->Code, "<OUTIDX_SIZE>", outidx_size);
    str_replace_all(TheCudaKernel->Code, "<PRELOAD_SMVARS>", preload_sm_str);
    str_replace_all(TheCudaKernel->Code, "<COMPUTE_IOUT>", iout_str);
    str_replace_all(TheCudaKernel->Code, "<SUBKERNEL_LOOPS>", subkernel_loops_str);

    std::cout << TheCudaKernel->Code << std::endl;
    TheCudaKernel->GenerateFunction();
}


std::vector<bool> OneOutPerThreadExecutor::GetSharedMemUvars()
{
    std::vector<bool> sharedmem_uvars;
    int numuvars = MultiKernel->GetNumUVars();
    sharedmem_uvars.resize(numuvars);
    int num_blocks_per_full_sm = CudaDeviceProp.maxThreadsPerMultiProcessor / 256;
    int shared_dbls_remaining = (CudaDeviceProp.sharedMemPerMultiprocessor / num_blocks_per_full_sm) / 8;
    for (int uvari = 0; uvari < numuvars; ++uvari)
    {
        sharedmem_uvars[uvari] = false;
        if (!MultiKernel->IsOutputUVar(uvari))
        {
            int ivar_size = MultiKernel->GetVarSize(uvari);
            if (ivar_size < shared_dbls_remaining)
            {
                sharedmem_uvars[uvari] = true;
                shared_dbls_remaining -= ivar_size;
            }
        }
    }
    return sharedmem_uvars;
}


std::string OneOutPerThreadExecutor::GenSharedMemPreload(std::vector<bool> &sharedmem_uvars)
{
    //If applicable Generate the SM preload code for small tensors
    std::string preload_sm_str;
    for (int uvari = 0; uvari < MultiKernel->GetNumUVars(); ++uvari)
    {
        if (sharedmem_uvars[uvari])
        {
            preload_sm_str += "    __shared__ double sT" + std::to_string(uvari);
            preload_sm_str += "[" + std::to_string(MultiKernel->GetVarSize(uvari)) + "];\n";
        }
    }
    for (int uvari = 0; uvari < MultiKernel->GetNumUVars(); ++uvari)
    {
        if (sharedmem_uvars[uvari])
        {
            std::string temp = 
                "    for (int idx = threadIdx.x; idx < <SMVAR_SIZE>; idx += blockDim.x)\n"
                "    {\n"
                "        sT<VARNUM>[idx] = T<VARNUM>[idx];\n"
                "    }\n\n";
            str_replace_all(temp, "<SMVAR_SIZE>", MultiKernel->GetVarSize(uvari));
            str_replace_all(temp, "<VARNUM>", uvari);
            preload_sm_str += temp;
        }
    }
    return preload_sm_str;
}


std::string OneOutPerThreadExecutor::GenIOutVars()
{
    const std::vector<int> N = MultiKernel->GetLoopDims();
    int numoutloops = MultiKernel->GetNumOuterIndices();
    std::vector<int> Wout(numoutloops); //Outer loop strides
    if (numoutloops > 0) 
    {
        Wout[numoutloops-1] = 1;
    }
    for (int d = numoutloops - 2; d >= 0; --d)
    {
        Wout[d] = Wout[d+1]*N[d+1];
    }

    std::string iout_str;
    for (int loopd = 0; loopd < numoutloops; ++loopd)
    {
        //I[loopd] = (outidx / (Wout[loopd]) % N[loopd];
        iout_str += "        ";
        iout_str += "int I" + std::to_string(loopd) + " = ";
        if (Wout[loopd] == 1)
        {
            iout_str += "outidx";
        }
        else
        {
            iout_str += "(outidx / " + std::to_string(Wout[loopd]) + ")";
            TheCudaKernel->IntOpsPerIndex += 1;
        }
        if (loopd > 0)
        {
            iout_str += " % " + std::to_string(N[loopd]);
        }
        iout_str += ";    // " + MultiKernel->GetLoopIndex(loopd) + "\n";

    }
    return iout_str;
}



std::string OneOutPerThreadExecutor::GenSubKernelLoops(std::vector<bool> &sharedmem_uvars)
{
    std::string kernel_loops_str;

    for (int ki = 0; ki < MultiKernel->GetNumKernels(); ++ki)
    {
        std::string loop_str = 
        "        //<KERNELSTR>\n"
                "<MIDLOOPS>"
        "            sum = 0.0;\n"
                    "<CONTLOOPS>"
        "                sum += <COMPUTE_RHS>;\n"
                    "<ENDCONTLOOPS>"
        "            T<OUTUVARI>[<OUTIDX>] <OUT_EQ_OP> sum;\n"
                "<ENDMIDLOOPS>"
        "\n";


        DimensionedKernel *kernel = MultiKernel->Kernels[ki];
        int numloops = MultiKernel->GetNumIndices();
        int numoutloops = MultiKernel->GetNumOuterIndices();
        int numcontloops = kernel->GetNumContractionIndices();
        int numinvars = kernel->GetNumInputVars();
        
        //Generate the mid loops
        int nummidloops = 0;
        std::string mid_loops_str;
        std::vector<bool> hoisted(numinvars, false);
        for (int loopi = numoutloops; loopi < numloops; ++loopi)
        {
            if (kernel->IsDependentOnLoop(loopi) && !kernel->IsContractionLoop(loopi))
            {
                std::string temp;
                for (int ivari = 0; ivari < numinvars; ++ ivari)
                {
                    int uvari = MultiKernel->GetUVari(ki, ivari);
                    if (!kernel->IsContractionVar(ivari) &&
                        kernel->GetVarLoopDepth(ivari) < loopi &&
                        !sharedmem_uvars[uvari] &&
                        !hoisted[ivari])
                    {
                        std::string uvaristr = std::to_string(uvari);
                        std::string varidxstr = GenVarIndex(ki, ivari);             
                        temp += "        double hT" + uvaristr + " = __ldg(&T" + uvaristr + "[" + varidxstr + "]);\n";
                        hoisted[ivari] = true;
                    }
                }

                temp += "        #pragma unroll\n";
                temp += "        for (int I<LOOPD> = 0; I<LOOPD> < <LOOP_SIZE>; ++I<LOOPD>) {";
                temp += "    // " + MultiKernel->GetLoopIndex(loopi) + "\n";
                str_replace_all(temp, "<LOOPD>", loopi);
                str_replace_all(temp, "<LOOP_SIZE>", kernel->GetLoopDim(loopi));
                mid_loops_str += temp;
                nummidloops += 1;
            }
        }
        std::string end_mid_loops_str = "        " + std::string(nummidloops, '}') + "\n";


        //Generate the contraction loops
        std::string cont_loops_str;
        for (int loopi = numoutloops; loopi < numloops; ++loopi)
        {
            if (kernel->IsContractionLoop(loopi))
            {
                std::string temp;
                for (int ivari = 0; ivari < numinvars; ++ ivari)
                {
                    int uvari = MultiKernel->GetUVari(ki, ivari);
                    if (kernel->GetVarLoopDepth(ivari) < loopi && !sharedmem_uvars[uvari] && !hoisted[ivari])
                    {
                        std::string uvaristr = std::to_string(uvari);
                        std::string varidxstr = GenVarIndex(ki, ivari);     
                        temp += "            double hT" + uvaristr + " = __ldg(&T" + uvaristr + "[" + varidxstr + "]);\n";
                        hoisted[ivari] = true;
                    }
                }

                temp += "            #pragma unroll\n";
                temp += "            for (int I<LOOPD> = 0; I<LOOPD> < <LOOP_SIZE>; ++I<LOOPD>) {";
                temp += "    // " + MultiKernel->GetLoopIndex(loopi) + "\n";
                str_replace_all(temp, "<LOOPD>", loopi);
                str_replace_all(temp, "<LOOP_SIZE>", kernel->GetLoopDim(loopi));
                cont_loops_str += temp;
            }
        }
        std::string end_cont_loops_str = "            " + std::string(numcontloops, '}') + "\n";

        //Generate the RHS computation inside the contraction loops
        std::string rhs_str;
        for (int ivari = 0; ivari < numinvars; ++ ivari)
        {
            int uvari = MultiKernel->GetUVari(ki, ivari);
            if (sharedmem_uvars[uvari])
            {
                rhs_str += "sT" + std::to_string(uvari) + "[" + GenVarIndex(ki, ivari) + "]";
            }
            else if (hoisted[ivari])
            {
                rhs_str += "hT" + std::to_string(uvari);
            }
            else
            {
                rhs_str += "__ldg(&T" + std::to_string(uvari) + "[" + GenVarIndex(ki, ivari) + "])";
            }

            if (ivari < numinvars-1)
            {
                rhs_str += "*";
            }
        }

        str_replace_all(loop_str, "<KERNELSTR>", kernel->KernelStr);
        str_replace_all(loop_str, "<MIDLOOPS>", mid_loops_str);
        str_replace_all(loop_str, "<CONTLOOPS>", cont_loops_str);
        str_replace_all(loop_str, "<COMPUTE_RHS>", rhs_str);
        str_replace_all(loop_str, "<ENDCONTLOOPS>", end_cont_loops_str);
        str_replace_all(loop_str, "<OUTUVARI>", MultiKernel->GetUVari(ki, -1));
        str_replace_all(loop_str, "<OUT_EQ_OP>", kernel->EqOperator);
        str_replace_all(loop_str, "<OUTIDX>", GenVarIndex(ki, -1));
        str_replace_all(loop_str, "<ENDMIDLOOPS>", end_mid_loops_str);
        kernel_loops_str += loop_str + '\n';
    }
    return kernel_loops_str;
}

std::string OneOutPerThreadExecutor::GenVarIndex(int ki, int vari)
{
    DimensionedKernel *kernel = MultiKernel->Kernels[ki];
    std::string indexStr = "I" + std::to_string(kernel->GetVarDimLoopNum(vari, 0)) + "*" +
                            std::to_string(kernel->GetVarDimStride(vari, 0));                         
    for (int d = 1; d < kernel->GetVarRank(vari); ++d)
    {
        indexStr += " + ";
        indexStr += "I" + std::to_string(kernel->GetVarDimLoopNum(vari, d)) + "*" +
                          std::to_string(kernel->GetVarDimStride(vari, d));                                   
    }
    return indexStr;
}

}

#endif