//Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at the Lawrence Livermore National Laboratory
//Written by Aaron Fisher (fisher47@llnl.gov). LLNL-CODE-738419.
//All rights reserved.
//This file is part of Acrotensor. For details, see https://github.com/LLNL/acrotensor.

#ifdef ACRO_HAVE_CUDA

#include "CudaExecutor.hpp"
#include <algorithm>
#include <set>
#include <limits>

namespace acro
{


CudaExecutor::CudaExecutor(DimensionedMultiKernel *multi_kernel) : KernelExecutor(multi_kernel) 
{
    HDeviceTensors = nullptr;
    SharedMemAllocated = 0;
    cudaGetDeviceProperties(&CudaDeviceProp, 0);
    GenerateCudaKernel();
}

CudaExecutor::~CudaExecutor()
{
    if (HDeviceTensors != nullptr)
    {
        delete HDeviceTensors;
    }
    acroCudaErrorCheck(cuModuleUnload(TheCudaKernel->Module));
    delete TheCudaKernel;
}


void CudaExecutor::ExecuteSingle(Tensor *output, std::vector<Tensor*> &inputs)
{
    MoveTensorsToGPU(output, inputs);

    int numuvars = MultiKernel->GetNumUVars();
    if (KernelParams.size() == 0)
    {
        HDeviceTensors = new double*[numuvars];
        KernelParams.resize(numuvars);
    }

    for (int uvari = 0; uvari < numuvars; ++uvari)
    {
        auto ki_vari = MultiKernel->GetFirstKiVariForUVari(uvari);
        int vari = ki_vari.second;
        double *dtensor;
        if (vari == -1)
        {
            dtensor = output->GetDeviceData();
        }
        else
        {
            dtensor = inputs[vari]->GetDeviceData();
        }
        HDeviceTensors[uvari] = dtensor;
        KernelParams[uvari] = &(HDeviceTensors[uvari]);
    }

    TheCudaKernel->Launch(KernelParams);
    //cudaDeviceSynchronize();
}


void CudaExecutor::ExecuteMulti(std::vector<Tensor*> &outputs, std::vector<std::vector<Tensor*> > &inputs)
{
    for (int ki = 0; ki < MultiKernel->GetNumKernels(); ++ki)
    {
        MoveTensorsToGPU(outputs[ki], inputs[ki]);
    }

    int numuvars = MultiKernel->GetNumUVars();
    if (KernelParams.size() == 0)
    {
        HDeviceTensors = new double*[numuvars];
        KernelParams.resize(numuvars);
    }

    for (int uvari = 0; uvari < numuvars; ++uvari)
    {
        auto ki_vari = MultiKernel->GetFirstKiVariForUVari(uvari);
        int ki = ki_vari.first;
        int vari = ki_vari.second;
        double *dtensor;
        if (vari == -1)
        {
            dtensor = outputs[ki]->GetDeviceData();
        }
        else
        {
            dtensor = inputs[ki][vari]->GetDeviceData();
        }
        HDeviceTensors[uvari] = dtensor;
        KernelParams[uvari] = &(HDeviceTensors[uvari]);
    }
    TheCudaKernel->Launch(KernelParams);
    //cudaDeviceSynchronize();
}


std::string CudaExecutor::GetImplementation()
{
    return TheCudaKernel->Code;
}


void CudaExecutor::GenerateCudaKernel()
{
    TheCudaKernel = new CudaKernel;
    TheCudaKernel->Code = 
    "extern \"C\"  \n"
    "__global__\n"
    "__launch_bounds__(<BLOCK_SIZE>)\n"
    "void <KERNEL_NAME>(<PARAMS>)\n"
    "{\n"
    "    double sum;\n"
    "    const unsigned int outidx = blockIdx.x;\n"
    "\n"
        "<SMWR_BUFFER>"
    "\n"
        "<PRELOAD_SMVARS>"
    "\n"
    "    __syncthreads();\n"
        "<INIT_INDICES>"
    "\n"
        "<SUBKERNEL_LOOPS>"

    "}\n";

    ACROBATIC_ASSERT(MultiKernel->GetNumOuterIndices() > 0, "CudaExecutor needs at least 1 non-contraction index.");

    NumBlockLoops = GetNumBlockLoops();


    int outidx_size = MultiKernel->GetIdxSizeForFirstNumLoops(NumBlockLoops);
    TheCudaKernel->FunctionName = "Kernel";
    TheCudaKernel->ThreadsPerBlock = GetNumThreadsPerBlock(NumBlockLoops);
    TheCudaKernel->NumBlocks = outidx_size;    

    //Generate the params list
    std::string params_str;
    for (int uvari = 0; uvari < MultiKernel->GetNumUVars(); ++uvari)
    {
        if (MultiKernel->IsOutputUVar(uvari))
        {
            params_str += "double * const T" + std::to_string(uvari);
        }
        else
        {
            params_str += "double const * const T" + std::to_string(uvari);
        }

        if (uvari < MultiKernel->GetNumUVars()-1)
        {
            params_str += ", ";
        }
    }

    GetSharedMemUvars();
    std::string preload_sm_str = GenSharedMemPreload();
    GetSharedMemWRKernels();
    std::string alloc_smwr_buffer_str = GenSharedMemWRBuffer();

    //Generate the indices outside the contraction loop
    std::string init_indices_str = GenInitIndices();


    //Generate the subkernel loops
    std::string subkernel_loops_str = GenSubKernelLoops();

    str_replace_all(TheCudaKernel->Code, "<BLOCK_SIZE>", TheCudaKernel->ThreadsPerBlock);
    str_replace_all(TheCudaKernel->Code, "<BLOCKS_PER_SM>", 4096 / TheCudaKernel->ThreadsPerBlock);
    str_replace_all(TheCudaKernel->Code, "<KERNEL_NAME>", TheCudaKernel->FunctionName);
    str_replace_all(TheCudaKernel->Code, "<PARAMS>", params_str);
    str_replace_all(TheCudaKernel->Code, "<NUMUVARS>", MultiKernel->GetNumUVars());
    str_replace_all(TheCudaKernel->Code, "<OUTIDX_SIZE>", outidx_size);
    str_replace_all(TheCudaKernel->Code, "<SMWR_BUFFER>", alloc_smwr_buffer_str);
    str_replace_all(TheCudaKernel->Code, "<PRELOAD_SMVARS>", preload_sm_str);
    str_replace_all(TheCudaKernel->Code, "<INIT_INDICES>", init_indices_str);
    str_replace_all(TheCudaKernel->Code, "<SUBKERNEL_LOOPS>", subkernel_loops_str);

    //std::cout << TheCudaKernel->Code << std::endl;
    //std::cout << MultiKernel->GetDimensionedNameString() << std::endl;
    //TheCudaKernel->WriteCodeToFile("kernel.cu");
    TheCudaKernel->GenerateFunction();
}


int CudaExecutor::GetNumBlockLoops()
{
    int loopi;
    for (loopi = 0; loopi < MultiKernel->GetNumOuterIndices(); ++loopi)
    {
        if (MultiKernel->GetIdxSizeForFirstNumLoops(loopi) >= 4096 || GetMinMidIdxSize(loopi) < 128)
        {
            break;
        }
    }
    return loopi;
}


int CudaExecutor::GetMinMidIdxSize(int num_block_loops)
{
    int numloops = MultiKernel->GetNumIndices();
    int min_idx_size = std::numeric_limits<int>::max();
    for (int ki = 0; ki < MultiKernel->GetNumKernels(); ++ki)
    {
        DimensionedKernel *kernel = MultiKernel->Kernels[ki];
        std::vector<int> mid_loops;
        for (int loopi = num_block_loops; loopi < numloops; ++loopi)
        {
            if (kernel->IsDependentOnLoop(loopi) && !kernel->IsContractionLoop(loopi))
            {
                mid_loops.push_back(loopi);
            }
        }        
        min_idx_size = std::min(min_idx_size, kernel->GetLoopsIdxSize(mid_loops));
    }
    return min_idx_size;
}


int CudaExecutor::GetMaxMidIdxSize(int num_block_loops)
{
    int numloops = MultiKernel->GetNumIndices();
    int max_idx_size = -1;
    for (int ki = 0; ki < MultiKernel->GetNumKernels(); ++ki)
    {
        DimensionedKernel *kernel = MultiKernel->Kernels[ki];
        std::vector<int> mid_loops;
        for (int loopi = num_block_loops; loopi < numloops; ++loopi)
        {
            if (kernel->IsDependentOnLoop(loopi) && !kernel->IsContractionLoop(loopi))
            {
                mid_loops.push_back(loopi);
            }
        }        
        max_idx_size = std::max(max_idx_size, kernel->GetLoopsIdxSize(mid_loops));
    }
    return max_idx_size;
}


int CudaExecutor::GetNumThreadsPerBlock(int num_block_loops)
{
    int min = GetMinMidIdxSize(num_block_loops);
    int max = GetMaxMidIdxSize(num_block_loops);
    int block_size;
    for (block_size = 64; block_size < 512; block_size *= 2)
    {
        if (block_size > max || block_size > int(1.3*float(min)))
        {
            break;
        } 
    }
    //std::cout << block_size << std::endl;
    return block_size;
}

void CudaExecutor::GetSharedMemUvars()
{
    int numuvars = MultiKernel->GetNumUVars();
    SharedMemUvars.resize(numuvars);
    int num_blocks_per_full_sm = CudaDeviceProp.maxThreadsPerMultiProcessor / TheCudaKernel->ThreadsPerBlock;
    int shared_mem_size = (CudaDeviceProp.sharedMemPerMultiprocessor / num_blocks_per_full_sm);
    for (int uvari = 0; uvari < numuvars; ++uvari)
    {
        SharedMemUvars[uvari] = false;
        if (!MultiKernel->IsOutputUVar(uvari))
        {
            int ivar_bytesize = MultiKernel->GetVarSize(uvari)*8;
            if (ivar_bytesize + SharedMemAllocated < shared_mem_size)
            {
                SharedMemUvars[uvari] = true;
                SharedMemAllocated += ivar_bytesize;
            }
        }
    }
}

void CudaExecutor::GetSharedMemWRKernels()
{
    int num_blocks_per_full_sm = CudaDeviceProp.maxThreadsPerMultiProcessor / TheCudaKernel->ThreadsPerBlock;
    int shared_mem_size = (CudaDeviceProp.sharedMemPerMultiprocessor / num_blocks_per_full_sm);
    SharedMemWRKernels.resize(MultiKernel->GetNumKernels(), false);
    SMWRBufferSize = 0;
    for (int ki = 0; ki < MultiKernel->GetNumKernels() - 1; ++ki)
    {
        DimensionedKernel *kernel = MultiKernel->Kernels[ki];
        DimensionedKernel *next_kernel = MultiKernel->Kernels[ki + 1];
        if (ki == 0 || SharedMemWRKernels[ki-1] == false)           //Avoid collision on the buffer
        {
            int outuvari = MultiKernel->GetUVari(ki, -1);
            for (int vari = 0; vari < next_kernel->GetNumInputVars(); ++vari)
            {
                if (MultiKernel->GetUVari(ki+1, vari) == outuvari)
                {
                    int onblock_var_idxsize = 8;            //Bytes/double
                    for (int di = 0; di < MultiKernel->GetVarRank(ki, vari); ++di)
                    {
                        int loopi = MultiKernel->GetVarDimLoopNum(ki, vari, di);
                        if (loopi >= NumBlockLoops)
                        {
                            onblock_var_idxsize *= MultiKernel->GetLoopDim(loopi);
                        }
                    }

                    if (onblock_var_idxsize + SharedMemAllocated < shared_mem_size)
                    {
                        SMWRBufferSize = std::max(SMWRBufferSize, onblock_var_idxsize);
                        SharedMemWRKernels[ki] = true;
                    }
                }
            }
        }
    }
    SharedMemAllocated += SMWRBufferSize;
}


std::vector<int> CudaExecutor::GetMidloopsOrder(int ki)
{
    DimensionedKernel* kernel = MultiKernel->Kernels[ki];
    int numloops = MultiKernel->GetNumIndices();
    int numinvars = kernel->GetNumInputVars();
    
    //Generate the mid loops
    std::set<int> mid_loops_set;
    for (int loopi = NumBlockLoops; loopi < numloops; ++loopi)
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
            if (!SharedMemUvars[uvari] && it != mid_loops_set.end())
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


std::vector<int> CudaExecutor::GetMidloopsStrides(DimensionedKernel *kernel, std::vector<int> &mid_loops)
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


std::string CudaExecutor::GenSharedMemWRBuffer()
{
    std::string smwr_str;
    if (SMWRBufferSize > 0)
    {
        smwr_str += "    __shared__ double SMWR[" + std::to_string(SMWRBufferSize / 8) + "];\n";
    }
    return smwr_str;
}


std::string CudaExecutor::GenSharedMemPreload()
{
    //If applicable Generate the SM preload code for small tensors
    std::string preload_sm_str;
    for (int uvari = 0; uvari < MultiKernel->GetNumUVars(); ++uvari)
    {
        if (SharedMemUvars[uvari])
        {
            preload_sm_str += "    __shared__ double sT" + std::to_string(uvari);
            preload_sm_str += "[" + std::to_string(MultiKernel->GetVarSize(uvari)) + "];\n";
        }
    }
    for (int uvari = 0; uvari < MultiKernel->GetNumUVars(); ++uvari)
    {
        if (SharedMemUvars[uvari])
        {
            std::string temp = 
                "    for (int idx = threadIdx.x; idx < <SMVAR_SIZE>; idx += blockDim.x)\n"
                "    {\n"
                "        sT<VARNUM>[idx] = " + GenTensor(uvari) + "[idx];\n"
                "    }\n\n";
            str_replace_all(temp, "<VARNUM>", uvari);
            str_replace_all(temp, "<SMVAR_SIZE>", MultiKernel->GetVarSize(uvari));

            preload_sm_str += temp;
        }
    }
    return preload_sm_str;
}


std::string CudaExecutor::GenInitIndices()
{
    const std::vector<int> N = MultiKernel->GetLoopDims();
    int numloops = MultiKernel->GetNumIndices();
    std::vector<int> Wout(NumBlockLoops); //Outer loop strides
    if (NumBlockLoops > 0) 
    {
        Wout[NumBlockLoops-1] = 1;
    }
    for (int d = NumBlockLoops - 2; d >= 0; --d)
    {
        Wout[d] = Wout[d+1]*N[d+1];
    }

    std::string init_indices_str;
    for (int loopd = 0; loopd < NumBlockLoops; ++loopd)
    {
        //I[loopd] = (outidx / (Wout[loopd]) % N[loopd];
        init_indices_str += "    unsigned int I";
        init_indices_str += std::to_string(loopd) + " = ";
        if (Wout[loopd] == 1)
        {
            init_indices_str += "outidx";
        }
        else
        {
            init_indices_str += "(outidx / " + std::to_string(Wout[loopd]) + ")";
            TheCudaKernel->IntOpsPerIndex += 1;
        }
        if (loopd > 0)
        {
            init_indices_str += " % " + std::to_string(N[loopd]);
        }
        init_indices_str += ";    // " + MultiKernel->GetLoopIndex(loopd) + "\n";

    }
    return init_indices_str;
}



std::string CudaExecutor::GenSubKernelLoops()
{
    std::string kernel_loops_str;
    int numloops = MultiKernel->GetNumIndices();
    std::vector<bool> hoisted;
    std::vector<int> loop_strides(numloops);

    for (int ki = 0; ki < MultiKernel->GetNumKernels(); ++ki)
    {
        std::string loop_str; 

        DimensionedKernel *kernel = MultiKernel->Kernels[ki];
        int numinvars = kernel->GetNumInputVars();
        int numcontloops = kernel->GetNumContractionIndices();
        std::vector<int> mid_loops = GetMidloopsOrder(ki);
        std::vector<int> mid_loop_strides = GetMidloopsStrides(kernel, mid_loops);
        int mid_loops_idx_size = kernel->GetLoopsIdxSize(mid_loops);
        int blockdim = TheCudaKernel->ThreadsPerBlock;
        int numblocki = mid_loops_idx_size / blockdim;
        int blocki_rem = mid_loops_idx_size % blockdim;
        if (blocki_rem != 0)
        {
            numblocki ++;
        }

        loop_str += "    //" + kernel->KernelStr + "\n";
        loop_str += "    {\n";  

        for (int mloopi = 0; mloopi < mid_loops.size(); ++mloopi)
        {
            int loopi = mid_loops[mloopi]; 
            loop_str += "        ushort2 " + GenLoopIndex(ki, loopi) + ";\n";
        }
        loop_str += GenMidLoopIndices(ki, mid_loops, mid_loop_strides, 0);
        for (int blocki = 0; blocki < numblocki; ++blocki)
        {
            std::string temp;
            if (blocki == numblocki - 1 && blocki_rem != 0)
            {
                temp += "        if (threadIdx.x < <BLOCKI_REM>)\n";
            }
            temp += "        {\n";
            temp += "            sum = 0.0;\n";
            temp +=             "<CONTLOOPS>";
            temp += "                sum += <COMPUTE_RHS>;\n";
            temp +=             "<ENDCONTLOOPS>";
            if (blocki < numblocki -1)
            {
                temp += GenMidLoopIndices(ki, mid_loops, mid_loop_strides, blocki+1);
            }
            if (SharedMemWRKernels[ki])
            {
                if (kernel->EqOperator != "=")
                {
                    temp += "            SMWR[<SMOUTIDX>] = " + GenTensor(ki,-1) + "[<OUTIDX>];\n";
                }
                temp += "            SMWR[<SMOUTIDX>] <OUT_EQ_OP> sum;\n";
            }
            temp += "            " + GenTensor(ki,-1) + "[<OUTIDX>] <OUT_EQ_OP> sum;\n";
            temp += "        }\n";
            str_replace_all(temp, "<BLOCKI>", blocki);
            str_replace_all(temp, "<BLOCKI_REM>", blocki_rem);

            loop_str += temp;

            //Generate the contraction loops
            std::string cont_loops_str;
            std::vector<bool> hoisted(numinvars, false);
            for (int loopi = NumBlockLoops; loopi < numloops; ++loopi)
            {
                if (kernel->IsContractionLoop(loopi))
                {
                    std::string temp;
                    for (int ivari = 0; ivari < numinvars; ++ ivari)
                    {
                        int uvari = MultiKernel->GetUVari(ki, ivari);
                        if (kernel->GetVarLoopDepth(ivari) < loopi && !SharedMemUvars[uvari] &&
                            !(ki > 0 && uvari == MultiKernel->GetUVari(ki-1, -1)) && 
                            !hoisted[ivari])
                        {
                            std::string ivaristr = std::to_string(ivari);
                            std::string uvaristr = std::to_string(uvari);
                            std::string varidxstr = GenVarIndex(ki, ivari, blocki);     
                            temp += "        double hIN" + ivaristr + " = __ldg(&" + GenTensor(uvari) + "[" + varidxstr + "]);\n";
                            hoisted[ivari] = true;
                        }
                    }

                    temp += "            #pragma unroll\n";
                    temp += "            for (unsigned int <LOOPIDX> = 0; <LOOPIDX> < <LOOP_SIZE>; ++<LOOPIDX>) {";
                    temp += "    // " + MultiKernel->GetLoopIndex(loopi) + "\n";
                    str_replace_all(temp, "<LOOPIDX>", GenLoopIndex(ki, loopi));
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
                std::string var_str;
                if (SharedMemUvars[uvari])
                {
                    var_str = "sT" + std::to_string(uvari) + "[" + GenVarIndex(ki, ivari, blocki) + "]";
                }
                else if (hoisted[ivari])
                {
                    var_str = "hIN" + std::to_string(ivari);
                }
                else if (ki > 0 && SharedMemWRKernels[ki-1] && uvari == MultiKernel->GetUVari(ki-1, -1))
                {
                    var_str = "SMWR[" + GenVarIndex(ki, ivari, blocki, false) + "]";
                }
                else
                {
                    var_str = GenTensor(uvari) + "[" + GenVarIndex(ki, ivari, blocki) + "]";
                }

                rhs_str += var_str;
                if (ivari < numinvars-1)
                {
                    rhs_str += "*";
                }
            }

            str_replace_all(loop_str, "<CONTLOOPS>", cont_loops_str);
            str_replace_all(loop_str, "<COMPUTE_RHS>", rhs_str);
            str_replace_all(loop_str, "<ENDCONTLOOPS>", end_cont_loops_str);
            str_replace_all(loop_str, "<OUT_EQ_OP>", kernel->EqOperator);
            str_replace_all(loop_str, "<OUTIDX>", GenVarIndex(ki, -1, blocki));
            str_replace_all(loop_str, "<SMOUTIDX>", GenVarIndex(ki, -1, blocki, false));

        }
        loop_str += "    }\n";
        loop_str += "    __syncthreads();\n\n";
        kernel_loops_str += loop_str;
    }
    return kernel_loops_str;
}

std::string CudaExecutor::GenMidLoopIndices(int ki, std::vector<int> &mid_loops, std::vector<int> &mid_loop_strides, int blocki)
{
    DimensionedKernel *kernel = MultiKernel->Kernels[ki];
    std::string indices;
    for (int mloopi = 0; mloopi < mid_loops.size(); ++mloopi)
    {
        int loopi = mid_loops[mloopi];
        std::string temp = "        " + GenLoopIndex(ki, loopi, blocki);
        temp +=  " = ((threadIdx.x + <MOFF>) / <LOOP_STRIDE>)";
        if (mloopi > 0)
        {
            temp += " % <LOOP_SIZE>;    // "+ MultiKernel->GetLoopIndex(loopi) + "\n";
        }
        else
        {
            temp += ";    // " + MultiKernel->GetLoopIndex(loopi) + "\n";
        }
        str_replace_all(temp, "<MOFF>", blocki*TheCudaKernel->ThreadsPerBlock);
        str_replace_all(temp, "<LOOP_STRIDE>", mid_loop_strides[mloopi]);
        str_replace_all(temp, "<LOOP_SIZE>", kernel->GetLoopDim(loopi));
        indices += temp;
    }
    return indices;
}


std::string CudaExecutor::GenTensor(int ki, int vari)
{
    return GenTensor(MultiKernel->GetUVari(ki, vari));
}


std::string CudaExecutor::GenTensor(int uvari)
{
    std::string tensor = "T" + std::to_string(uvari);
    return tensor;
}

std::string CudaExecutor::GenVarIndex(int ki, int vari, int blocki, bool blockdims)
{
    DimensionedKernel *kernel = MultiKernel->Kernels[ki];
    std::string index_str;
    bool first = true;              
    for (int di = 0; di < kernel->GetVarRank(vari); ++di)
    {
        int loopi = kernel->GetVarDimLoopNum(vari, di);
        if (blockdims || loopi >= NumBlockLoops)
        {
            if (!first)
            {
                index_str += " + ";
            }

            std::string loopidx = GenVarSubIndex(ki, vari, di, blocki);
            std::string stride = std::to_string(kernel->GetVarDimStride(vari, di));
            //index_str += "__umul24(" + loopidx + "," + stride + ")";
            index_str += loopidx + "*" + stride;
            first = false;
        }           
    }
    return index_str;
}


std::string CudaExecutor::GenVarSubIndex(int ki, int vari, int dimi, int blocki)
{
    DimensionedKernel *kernel = MultiKernel->Kernels[ki];   
    return GenLoopIndex(ki, kernel->GetVarDimLoopNum(vari, dimi), blocki);
}


std::string CudaExecutor::GenLoopIndex(int ki, int loopi, int blocki)
{
    DimensionedKernel *kernel = MultiKernel->Kernels[ki];
    std::string loopidx = "I" + std::to_string(loopi);
    if (blocki > -1 && loopi >= NumBlockLoops && !kernel->IsContractionLoop(loopi))
    {
        loopidx += (blocki%2 == 0) ? ".x" : ".y";
    }
    
    return loopidx;
}

}

#endif