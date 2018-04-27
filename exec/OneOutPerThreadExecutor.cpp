//Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at the Lawrence Livermore National Laboratory
//Written by Aaron Fisher (fisher47@llnl.gov). LLNL-CODE-738419.
//All rights reserved.
//This file is part of Acrotensor. For details, see https://github.com/LLNL/acrotensor.

#ifdef ACRO_HAVE_CUDA

#include "OneOutPerThreadExecutor.hpp"
#include <algorithm>
#include <set>

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

    TheCudaKernel->Launch(kernelParams);
    cudaDeviceSynchronize();
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

    TheCudaKernel->Launch(kernelParams);
    cudaDeviceSynchronize();
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
    "    const int outidx = blockIdx.x;//*blockDim.x+threadIdx.x;\n"
    "    if (outidx > <OUTIDX_SIZE>) return;\n"
    "\n"
        "<PRELOAD_SMVARS>"
    "\n"
    "    __syncthreads();\n"
        "<COMPUTE_IOUT>"
    "\n"
        "<SUBKERNEL_LOOPS>"

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
    //TheCudaKernel->NumBlocks = (outidx_size / TheCudaKernel->ThreadsPerBlock) + 1;
    TheCudaKernel->NumBlocks = outidx_size;

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

    //std::cout << TheCudaKernel->Code << std::endl;
    TheCudaKernel->WriteCodeToFile("kernel.cu");
    TheCudaKernel->GenerateFunction();
}


std::vector<bool> OneOutPerThreadExecutor::GetSharedMemUvars()
{
    std::vector<bool> sharedmem_uvars;
    int numuvars = MultiKernel->GetNumUVars();
    sharedmem_uvars.resize(numuvars);
    int num_blocks_per_full_sm = CudaDeviceProp.maxThreadsPerMultiProcessor / TheCudaKernel->ThreadsPerBlock;
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


std::vector<int> OneOutPerThreadExecutor::GetMidloopsOrder(int ki, std::vector<bool> &sharedmem_uvars)
{
    DimensionedKernel* kernel = MultiKernel->Kernels[ki];
    int numoutloops = MultiKernel->GetNumOuterIndices();
    int numloops = MultiKernel->GetNumIndices();
    int numinvars = kernel->GetNumInputVars();
    
    //Generate the mid loops
    std::set<int> mid_loops_set;
    for (int loopi = numoutloops; loopi < numloops; ++loopi)
    {
        if (kernel->IsDependentOnLoop(loopi) && !kernel->IsContractionLoop(loopi))
        {
            mid_loops_set.insert(loopi);
        }
    }

    int max_ivar_rank = 0;
    for (int vari = -1; vari < numinvars; ++vari)
    { 
        max_ivar_rank = std::max(max_ivar_rank, kernel->GetVarRank(vari));
    }

    //Collect of the loop dimensions from all the variables in lowest stride order
    std::vector<int> mid_loops;
    for (int rankoff = 0; rankoff < max_ivar_rank; ++rankoff)
    {
        for (int vari = numinvars-1; vari >= -1; --vari)
        {
            int uvari = MultiKernel->GetUVari(ki, vari);
            int vidxi = kernel->GetVarRank(vari) - 1 - rankoff;
            int loopi = vidxi >= 0 ? kernel->GetVarDimLoopNum(vari, vidxi) : -1;
            auto it = mid_loops_set.find(loopi);
            if (!sharedmem_uvars[uvari] && it != mid_loops_set.end())
            {
                mid_loops.push_back(loopi);
                mid_loops_set.erase(it);
            }
        }
    }

    //Tack on the rest of the indices
    for (auto it = mid_loops_set.rbegin(); it != mid_loops_set.rend(); ++it)
    {
        mid_loops.push_back(*it);
    }

    //We want the lowest strides to be in the inner most loops
    std::reverse(mid_loops.begin(), mid_loops.end());
    return mid_loops;
}


std::vector<int> OneOutPerThreadExecutor::GetMidloopsStrides(DimensionedKernel *kernel, std::vector<int> &mid_loops)
{
    //Generate the mid loops
    int nummidloops = mid_loops.size();
    std::vector<int> strides(nummidloops);
    int stride = 1;
    for (int mloopi = nummidloops - 1; mloopi >= 0; --mloopi)
    {
        int loopi = mid_loops[mloopi];
        strides[mloopi] = stride;
        stride *= kernel->GetLoopDim(loopi);
    }

    return strides;
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
        iout_str += "    int I" + std::to_string(loopd) + " = ";
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
    int numloops = MultiKernel->GetNumIndices();
    int numoutloops = MultiKernel->GetNumOuterIndices();       
    std::vector<bool> hoisted;
    std::vector<int> loop_strides(numloops);
    for (int ki = 0; ki < MultiKernel->GetNumKernels(); ++ki)
    {
        std::string loop_str = 
        "    //<KERNELSTR>\n"
            "<MIDLOOPS>"
        "        sum = 0.0;\n"
                "<CONTLOOPS>"
        "            sum += <COMPUTE_RHS>;\n"
                "<ENDCONTLOOPS>"
        "        T<OUTUVARI>[<OUTIDX>] <OUT_EQ_OP> sum;\n"
            "<ENDMIDLOOPS>"
        "    __syncthreads();\n";     


        DimensionedKernel *kernel = MultiKernel->Kernels[ki];
        int numinvars = kernel->GetNumInputVars();
        int numcontloops = kernel->GetNumContractionIndices();
        std::vector<int> mid_loops = GetMidloopsOrder(ki, sharedmem_uvars);
        std::vector<int> mid_loop_strides = GetMidloopsStrides(kernel, mid_loops);
        int mid_loops_idx_size = kernel->GetLoopsIdxSize(mid_loops);

        int outvar_midx_size = 1;
        for (int vidxi = 0; vidxi < kernel->GetVarRank(-1); ++vidxi)
        {
            int loopi = kernel->GetVarDimLoopNum(-1, vidxi);
            if (loopi >= numoutloops)
            {
                outvar_midx_size *= kernel->GetLoopDim(loopi);
            }
        }


        std::string mid_loops_str;
        std::string end_mid_loops_str;
        mid_loops_str = "    for (int midx = threadIdx.x; midx < <MIDXEND>; midx += blockDim.x) {\n";
        str_replace_all(mid_loops_str, "<MIDXEND>", mid_loops_idx_size);
        bool first_mid_index = true;
        for (int mloopi = 0; mloopi < mid_loops.size(); ++mloopi)
        {
            int loopi = mid_loops[mloopi];
            std::string temp;
            temp += "        int I<LOOPD> = (midx / <LOOP_STRIDE>)";
            if (!first_mid_index)
            {
                temp += " % <LOOP_SIZE>";                    
            }
            first_mid_index = false;
            temp += ";    // " + MultiKernel->GetLoopIndex(loopi) + "\n";
            str_replace_all(temp, "<LOOPD>", loopi);
            str_replace_all(temp, "<LOOP_STRIDE>", mid_loop_strides[mloopi]);
            str_replace_all(temp, "<LOOP_SIZE>", kernel->GetLoopDim(loopi));
            mid_loops_str += temp;
        }
        end_mid_loops_str = "    }\n";

        //Generate the contraction loops
        std::string cont_loops_str;
        std::vector<bool> hoisted(numinvars, false);
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
                        std::string ivaristr = std::to_string(ivari);
                        std::string uvaristr = std::to_string(uvari);
                        std::string varidxstr = GenVarIndex(ki, ivari);     
                        temp += "        double hIN" + ivaristr + " = __ldg(&T" + uvaristr + "[" + varidxstr + "]);\n";
                        hoisted[ivari] = true;
                    }
                }

                temp += "        #pragma unroll\n";
                temp += "        for (int I<LOOPD> = 0; I<LOOPD> < <LOOP_SIZE>; ++I<LOOPD>) {";
                temp += "    // " + MultiKernel->GetLoopIndex(loopi) + "\n";
                str_replace_all(temp, "<LOOPD>", loopi);
                str_replace_all(temp, "<LOOP_SIZE>", kernel->GetLoopDim(loopi));
                cont_loops_str += temp;
            }
        }
        std::string end_cont_loops_str = "        " + std::string(numcontloops, '}') + "\n";

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
                rhs_str += "hIN" + std::to_string(ivari);
            }
            else
            {
                rhs_str += "T" + std::to_string(uvari) + "[" + GenVarIndex(ki, ivari) + "]";
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
    std::string indexStr = "__mul24(I" + std::to_string(kernel->GetVarDimLoopNum(vari, 0)) + "," +
                            std::to_string(kernel->GetVarDimStride(vari, 0)) + ")";                         
    for (int d = 1; d < kernel->GetVarRank(vari); ++d)
    {
        indexStr += " + ";
        indexStr += "__mul24(I" + std::to_string(kernel->GetVarDimLoopNum(vari, d)) + "," +
                          std::to_string(kernel->GetVarDimStride(vari, d)) + ")";                                   
    }
    return indexStr;
}

}

#endif