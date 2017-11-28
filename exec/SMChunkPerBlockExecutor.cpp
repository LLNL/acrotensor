//Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at the Lawrence Livermore National Laboratory
//Written by Aaron Fisher (fisher47@llnl.gov). LLNL-CODE-738419.
//All rights reserved.
//This file is part of Acrotensor. For details, see https://github.com/LLNL/acrotensor.

#ifdef ACRO_HAVE_CUDA

#include "SMChunkPerBlockExecutor.hpp"
#include <iostream>
#include <sstream>
#include <math.h>
#include <algorithm>

namespace acro
{


SMChunkPerBlockExecutor::SMChunkPerBlockExecutor(std::string &kernelstr) : KernelExecutor(kernelstr) 
{
    cudaGetDeviceProperties(&CudaDeviceProp, 0);

}

SMChunkPerBlockExecutor::~SMChunkPerBlockExecutor()
{
    for (auto it = CudaKernelMap.begin(); it != CudaKernelMap.end(); ++it)
    {
        CudaKernel *cuda_kernel = it->second;
        acroCudaErrorCheck(cuModuleUnload(cuda_kernel->Module));
        delete cuda_kernel;
    }
    CudaKernelMap.clear();
}


void SMChunkPerBlockExecutor::ExecuteKernel(Tensor *output, std::vector<Tensor*> &inputs)
{
    MoveTensorsToGPU(output, inputs);
    Kernel.AttachTensors(output, inputs);
    ExecuteLoopsCuda();
}

void SMChunkPerBlockExecutor::ExecuteLoopsCuda()
{
    int numvars = Kernel.GetNumVars();
    const std::vector<int> N = Kernel.GetLoopDims();

    auto it = CudaKernelMap.find(N);
    if (it == CudaKernelMap.end())
    {
        CudaKernelMap[N] = GenerateCudaKernel();
    }
    CudaKernel *cuda_kernel = CudaKernelMap[N]; 

    //Since we are using += or -= into the output 
    if (Kernel.EqOperator == "=" && cuda_kernel->IsMultipleBlockPerOutput)
    {
        Tensor *output = Kernel.GetTensor(-1);
        int size = output->GetSize();
        CudaSet<<<size/512+1,512,0,TheCudaStream>>>(output->GetDeviceData(), 0.0, size);
    }

    std::vector<void*> kernelParams;
    double **tdata = new double*[numvars];
    for (int vari = -1; vari < numvars-1; ++vari)
    {
        tdata[vari+1] = Kernel.GetTensor(vari)->GetDeviceData();
        kernelParams.push_back(&tdata[vari+1]);
    }
    cuda_kernel->Launch(kernelParams, TheCudaStream);
    delete[] tdata;
}


std::string SMChunkPerBlockExecutor::GetImplementation(Tensor *output, std::vector<Tensor*> &inputs)
{
    Kernel.AttachTensors(output, inputs);
    ACROBATIC_ASSERT(output->GetRank() == Kernel.OutputVar.IndexNames.size(), 
                     "Tensor rank of the output var in the kernel does not match the rank of the actual tensor.\n"
                    +"Kernel:  " + Kernel.KernelStr);
    ACROBATIC_ASSERT(inputs.size() == Kernel.InputVars.size());
    for (int i = 0; i < inputs.size(); ++i)
    {
        ACROBATIC_ASSERT(inputs[i]->GetRank() == Kernel.InputVars[i]->IndexNames.size(),
                         "Tensor rank of input var " + std::to_string(i) +
                         " does not match the rank of the actual tensor.\n" +
                        +"Kernel:  " + Kernel.KernelStr);
    }

    const std::vector<int> N = Kernel.GetLoopDims();
    auto it = CudaKernelMap.find(N);
    if (it == CudaKernelMap.end())
    {
        CudaKernelMap[N] = GenerateCudaKernel();
    }
    CudaKernel *cuda_kernel = CudaKernelMap[N];

    std::string impl = cuda_kernel->Code + "\n";
    impl += "NumBlocks:  " + std::to_string(cuda_kernel->NumBlocks) + "\n";

    return impl;
}


CudaKernel *SMChunkPerBlockExecutor::GenerateCudaKernel()
{
    int numvars = Kernel.GetNumVars();
    int num_loops = Kernel.GetNumLoops();
    int num_cont_loops = Kernel.GetNumContractionLoops();
    int num_outer_loops, num_middle_loops, num_inner_loops;
    int block_size;
    GetLoopingStructure(block_size, num_outer_loops, num_middle_loops, num_inner_loops);

    int outidx_size = 1;
    int contidx_size = 1;
    for (int loop = num_outer_loops + num_middle_loops; loop < num_loops; ++loop)
    {
        if (Kernel.IsContractionLoop(loop))
        {
            contidx_size *= Kernel.GetLoopDim(loop);
        }
        else
        {
            outidx_size *= Kernel.GetLoopDim(loop);
        }
    }
    int smidx_size = contidx_size*outidx_size;

    CudaKernel *cuda_kernel = new CudaKernel;
    cuda_kernel->ThreadsPerBlock = block_size;
    cuda_kernel->NumBlocks = Kernel.GetIdxSizeForFirstNumLoops(num_outer_loops);
    cuda_kernel->Code = GetCodeTemplate();
    cuda_kernel->FunctionName = GetFunctionName();
    cuda_kernel->IsMultipleBlockPerOutput = (num_inner_loops < num_cont_loops) ? "1" : "0";
    std::string is_eq_op_me = (Kernel.EqOperator == "-=") ? "1" : "0";

    str_replace_all(cuda_kernel->Code, "<KERNEL_NAME>", cuda_kernel->FunctionName);
    str_replace_all(cuda_kernel->Code, "<PARAMS>", GetParamsCode());
    str_replace_all(cuda_kernel->Code, "<INIT_S_TDATA>", GetInitSTdataCode());
    str_replace_all(cuda_kernel->Code, "<OUTER_IVALS>", GetOuterIvalsCode(num_outer_loops));
    str_replace_all(cuda_kernel->Code, "<COMPUTE_BASE_INDICES>", GetBaseIndicesCode(num_outer_loops));
    str_replace_all(cuda_kernel->Code, "<COMPUTE_LOAD_INDICES>", GetLoadIndicesCode(block_size, num_outer_loops, num_middle_loops, num_inner_loops));
    str_replace_all(cuda_kernel->Code, "<LOAD_INVARIANT_INPUTS>", GetLoadInvariantInputsCode(block_size, num_outer_loops, num_middle_loops, num_inner_loops));
    str_replace_all(cuda_kernel->Code, "<MIDX_FOR_LOOPS>", GetMidxLoops(num_outer_loops, num_middle_loops));
    str_replace_all(cuda_kernel->Code, "<LOAD_VARIANT_INPUTS>", GetLoadVariantInputsCode(block_size, num_outer_loops, num_middle_loops, num_inner_loops));
    str_replace_all(cuda_kernel->Code, "<MULT_VARS>", GetMultVarsCode(block_size, smidx_size));
    str_replace_all(cuda_kernel->Code, "<END_MIDX_FOR_LOOPS>", "    " + std::string(num_middle_loops, '}') + "\n");    
    str_replace_all(cuda_kernel->Code, "<NUM_VARS>", Kernel.GetNumVars());
    str_replace_all(cuda_kernel->Code, "<SMIDX_SIZE>", smidx_size);
    str_replace_all(cuda_kernel->Code, "<OUTIDX_SIZE>", outidx_size);
    str_replace_all(cuda_kernel->Code, "<CONTIDX_SIZE>", contidx_size);
    str_replace_all(cuda_kernel->Code, "<EQ_OP>", Kernel.EqOperator);
    str_replace_all(cuda_kernel->Code, "<IS_EQ_OP_MINUS_EQ>", is_eq_op_me);
    str_replace_all(cuda_kernel->Code, "<IS_OUT_ON_MANY_BLOCKS>", cuda_kernel->IsMultipleBlockPerOutput);
    str_replace_all(cuda_kernel->Code, "<MIDX_OFF0>", GetMidxOffCode(0, num_outer_loops, num_middle_loops));

    //std::cout << cuda_kernel->Code << std::endl;
    //std::cout << "SM idx size:  " << smidx_size << std::endl;
    cuda_kernel->GenerateFunction();

    return cuda_kernel;
}


void SMChunkPerBlockExecutor::GetLoopingStructure(int &block_size, int &num_outer_loops, int &num_middle_loops, int &num_inner_loops)
{
    int num_loops = Kernel.GetNumLoops();
    int num_cont_loops = Kernel.GetNumContractionLoops();
    int numvars = Kernel.GetNumVars();
    int num_mp = CudaDeviceProp.multiProcessorCount;
    double thread_waste = 1024.0;

    //for (int t_block_size = 128; t_block_size <= 1024; t_block_size += 32)
    for (int t_block_size = 256; t_block_size <= 256; t_block_size += 32)
    {
        int t_num_outer_loops, t_num_middle_loops, t_num_inner_loops;
        int blocks_per_mp = CudaDeviceProp.maxThreadsPerMultiProcessor / std::max(t_block_size, 128);
        int max_avail_sm = CudaDeviceProp.sharedMemPerMultiprocessor / blocks_per_mp;
        t_num_inner_loops = num_loops;        
        for (int nil = 1; nil <= num_loops; ++nil)
        {
            int index_size = Kernel.GetIndexSpaceSizeForInnerLoops(nil);
            int sm_required = numvars*index_size*8 + Kernel.GetInputStorageReqForInnerLoops(nil)*8;
            if (sm_required > max_avail_sm)
            {
                t_num_inner_loops = nil - 1;
                break;
            }
        }

        for (t_num_outer_loops = 0; t_num_outer_loops < num_loops - t_num_inner_loops; ++t_num_outer_loops)
        {
            if (Kernel.GetIdxSizeForFirstNumLoops(t_num_outer_loops) > num_mp*blocks_per_mp)
            {
                break;
            } 
        }
        t_num_middle_loops = num_loops - t_num_outer_loops - t_num_inner_loops;

        int index_size = Kernel.GetIndexSpaceSizeForInnerLoops(t_num_inner_loops);
        double t_thread_waste = double((index_size % t_block_size)) / double(index_size);
        if (t_thread_waste < thread_waste)
        {
            block_size = t_block_size;
            num_outer_loops = t_num_outer_loops;
            num_middle_loops = t_num_middle_loops;
            num_inner_loops = t_num_inner_loops;
            thread_waste = t_thread_waste;
        }
    }
    //std::cout << thread_waste << ", " << block_size << ", " << num_outer_loops << ", " << num_middle_loops << ", " << num_inner_loops << std::endl;
}

//Gets offsets for each indexed 
void SMChunkPerBlockExecutor::GetVarBlockIndexData(int numblockloops, std::vector<std::vector<ushort2>> &offset)
{
    std::vector<int> var_off;
    std::vector<int> loop_off;
    for (int vi = 0; vi < Kernel.GetNumVars(); ++vi)
    {
        Kernel.GetVarIndexOffsetsForInnerLoops(vi - 1, numblockloops, var_off, loop_off);
        offset[vi].resize(var_off.size());
        //std::cout << "Var:  " << vi << std::endl;
        for (int i = 0; i < var_off.size(); ++i)
        {
            offset[vi][i].x = var_off[i];
            offset[vi][i].y = loop_off[i];
            //std::cout << var_off[i] << ", " << loop_off[i] << std::endl;
        }
    }
}


std::string SMChunkPerBlockExecutor::GetCodeTemplate()
{
    std::string temp = 
    "extern \"C\"  \n"
    "__global__ void <KERNEL_NAME>(<PARAMS>)\n"
    "{\n"
    "    int varbase, tidx;\n"
        "<INIT_S_TDATA>"
        "<OUTER_IVALS>"
        "<COMPUTE_BASE_INDICES>"
    "\n"
        "<COMPUTE_LOAD_INDICES>"
    "\n"
        "<LOAD_INVARIANT_INPUTS>"
    "    __syncthreads();\n"
        "<MIDX_FOR_LOOPS>"
    "        //Now load the inputs into shared memory and multiply them into s_tdata0\n"
            "<LOAD_VARIANT_INPUTS>"
    "        __syncthreads();\n"        
    "\n"
            "<MULT_VARS>"
    "\n"
    "        //Take the products and sum contract them into the s_tdata0 array\n"
    "        #if <OUTIDX_SIZE> > <CONTIDX_SIZE>\n"
    "            for (int outidx = threadIdx.x; outidx < <OUTIDX_SIZE>; outidx += blockDim.x)\n"
    "            {\n"
    "                double sum = s_tdata0[outidx*<CONTIDX_SIZE>];\n"
    "                #pragma unroll\n"
    "                for (int contidx = 1; contidx < <CONTIDX_SIZE>; ++contidx)\n"
    "                    sum += s_tdata0[outidx*<CONTIDX_SIZE> + contidx];\n"
    "                    s_tdata0[outidx*<CONTIDX_SIZE>] = sum;\n"
    "            }\n"
    "            __syncthreads();\n"
    "        #else\n"
    "            #pragma unroll\n"
    "            for (int outidx = 0; outidx < <OUTIDX_SIZE>; ++outidx)\n"
    "            {\n"
    "                double sum = 0.0;\n"
    "                for (int contidx = threadIdx.x; contidx < <CONTIDX_SIZE>; contidx += blockDim.x)\n"
    "                    sum += s_tdata0[outidx*<CONTIDX_SIZE> + contidx];\n"
    "                __syncthreads();\n"
    "\n"
    "                //Every thread has part of the sum so we will sum reduce\n"
    "                //First reduce the warp into lane 0\n"
    "                sum += __shfl_xor(sum, 16);\n"
    "                sum += __shfl_xor(sum, 8);\n"
    "                sum += __shfl_xor(sum, 4);\n"
    "                sum += __shfl_xor(sum, 2);\n"
    "                sum += __shfl_xor(sum, 1);\n"
    "\n"
    "                //Now collect all the lane 0 values and reduce them in shared memory\n"
    "                if (threadIdx.x == 0)\n"
    "                    s_tdata0[outidx*<CONTIDX_SIZE>] = 0.0;\n"
    "                __syncthreads();\n"
    "                if (threadIdx.x % 32 == 0)\n"
    "                {\n"
    "                    atomicAdd(&(s_tdata0[outidx*<CONTIDX_SIZE>]), sum);\n"
    "                }\n"
    "                __syncthreads();\n"
    "            }\n"
    "        #endif\n"
    "\n"
    "        //Finally store the results back in global memory\n"
    "        varbase = base0 + <MIDX_OFF0>;\n"
    "        for (int outidx = threadIdx.x; outidx < <OUTIDX_SIZE>; outidx += blockDim.x)\n"
    "        {\n"
    "            int loopidx = outidx*<CONTIDX_SIZE>;\n"
    "            int final_ind = varbase + outidx;\n"
    "            #if <IS_OUT_ON_MANY_BLOCKS> //Is output on multiple blocks\n"
    "                #if <IS_EQ_OP_MINUS_EQ> //Is EQ_OP -=\n"
    "                    atomicAdd(&(tdata0[final_ind]), -s_tdata0[loopidx]);\n"
    "                #else\n"
    "                    atomicAdd(&(tdata0[final_ind]), s_tdata0[loopidx]);\n"
    "                #endif\n"
    "            #else\n"
    "                tdata0[final_ind] <EQ_OP> s_tdata0[loopidx];\n"
    "            #endif\n"
    "        }\n"
        "<END_MIDX_FOR_LOOPS>"
    "}\n";
    return temp;
}


std::string SMChunkPerBlockExecutor::GetFunctionName()
{
    int num_loops = Kernel.GetNumLoops();
    std::string name = Kernel.GetCanonicalNameString();
    name += "dim";
    for (int loopd = 0; loopd < num_loops; ++loopd)
    {
        name += "_" + std::to_string(Kernel.GetLoopDim(loopd));
    }
    name += "_SMC";
    return name;
}


std::string SMChunkPerBlockExecutor::GetParamsCode()
{
    std::string code;
    for (int vi = 0; vi < Kernel.GetNumVars(); ++vi)
    {
        code += "double *tdata" + std::to_string(vi) + ", ";
    }
    code.pop_back();
    code.pop_back();
    return code;
}


std::string SMChunkPerBlockExecutor::GetInitSTdataCode()
{
    std::string code;
    for (int vi = 0; vi < Kernel.GetNumVars(); ++vi)
    {
        code += "    __shared__ double s_tdata" + std::to_string(vi) + "[<SMIDX_SIZE>];\n";
    }
    return code;
}


std::string SMChunkPerBlockExecutor::GetOuterIvalsCode(int num_outer_loops)
{
    const std::vector<int> &N = Kernel.GetLoopDims();
    std::vector<int> W(num_outer_loops); //Outer loop strides
    if (num_outer_loops > 0) 
    {
        W[num_outer_loops-1] = 1;
    }    
    for (int d = num_outer_loops - 2; d >= 0; --d)
    {
        W[d] = W[d+1]*N[d+1];
    }

    std::string code;  
    for (int loopd = 0; loopd < num_outer_loops; ++loopd)
    {
        std::string temp = "    int I<LOOP_NUM> = (blockIdx.x / <LOOP_BSTRIDE>) % <LOOP_DIM>;\n";   
        str_replace_all(temp, "<LOOP_NUM>", std::to_string(loopd));
        str_replace_all(temp, "<LOOP_BSTRIDE>", std::to_string(W[loopd]));
        str_replace_all(temp, "<LOOP_DIM>", std::to_string(N[loopd]));
        code += temp;
    }
    code += "\n";
    return code;
}


//Generates code to compute the where in dout and the din variables we start given the
// I0, I1, ... computed using the block number.  It should looks something like:
// int base0 = I1*4;                //dout
// int base1 = I0*16 + I1*4 +I2*1;  //din[0]
// ...
std::string SMChunkPerBlockExecutor::GetBaseIndicesCode(int num_outer_loops)
{
    std::string code = "    //using the block id compute where to start in all the variables\n";
    code += "    //base0 (dout), base1 (din[0]), ...\n";
    for (int vi = 0; vi < Kernel.GetNumVars(); ++vi)
    {
        std::string temp = "    int base<VAR_NUM> = <RHS>;\n";
        std::string rhs = "0";

        for (int d = 0; d < Kernel.GetVarRank(vi-1); ++d)
        {
            int loop_num = Kernel.GetVarDimLoopNum(vi-1, d);
            if (loop_num < num_outer_loops)
            {
                rhs += " + I" + std::to_string(loop_num) + "*"
                              + std::to_string(Kernel.GetVarDimStride(vi-1, d));
            }
        }

        str_replace_all(temp, "<VAR_NUM>", std::to_string(vi));
        str_replace_all(temp, "<RHS>", rhs);
        code += temp;
    }
    return code;
}


std::string SMChunkPerBlockExecutor::GetLoadIndicesCode(int block_size, int num_outer_loops, int num_middle_loops, int num_inner_loops)
{
    std::string code;
    int curr_thread = 0;
    for (int vi = 1; vi < Kernel.GetNumVars(); ++vi)
    {
        std::string temp = 
        "    __shared__ int2 s_inidx_off<VAR_NUM>[<LOADIDX_SIZE>];\n"
        "    for (int threadidx = threadIdx.x; threadidx >= <LOADIDX_BOT> && threadidx <= <LOADIDX_TOP>; threadidx += blockDim.x)\n"
        "    {\n"
        "        int loadidx = threadidx - <LOADIDX_BOT>;\n"
                "<VARLOOP_INDICES>"
        "        int2 inidx_off;\n"
        "        inidx_off.x = <VAROFF>;\n"
        "        inidx_off.y = <LOOPOFF>;\n"
        "        s_inidx_off<VAR_NUM>[loadidx] = inidx_off;\n"
        "    }\n\n";

        int loadidx_size = 1;
        std::vector<std::pair<int,int>> varloops;
        for (int vard = Kernel.GetVarRank(vi-1) - 1; vard >= 0; --vard)
        {
            int loopd = Kernel.GetVarDimLoopNum(vi-1, vard);
            if (loopd >= num_outer_loops + num_middle_loops)
            {
                loadidx_size *= Kernel.GetLoopDim(loopd);
                varloops.push_back(std::make_pair(loopd, loadidx_size));
            }
        }

        bool trivial_varoff = true;
        int trivial_varoff_mult = 1;
        bool trivial_loopoff = true;
        int trivial_loopoff_mult = 1;
        for (int vli = 1; vli < varloops.size(); ++vli)
        {
            int loop_vli = varloops[vli].first;
            int loop_vlim1 = varloops[vli-1].first;
            if (loop_vlim1 - loop_vli != 1)
            {
                trivial_loopoff = false;
            }
            int vardim_vli = Kernel.GetLoopNumVarDim(loop_vli, vi-1);
            int vardim_vlim1 = Kernel.GetLoopNumVarDim(loop_vlim1, vi-1);
            if (vardim_vlim1 - vardim_vli != 1)
            {
                trivial_varoff = false;
            }            
        }
        if (varloops.size() > 0)
        {
            trivial_loopoff_mult = Kernel.GetLoopStride(varloops[0].first);
            int vardim = Kernel.GetLoopNumVarDim(varloops[0].first, vi-1);
            trivial_varoff_mult = Kernel.GetVarDimStride(vi-1, vardim);
        }

        std::string varloop_indices;
        if (!trivial_varoff || !trivial_loopoff)
        {
            if (varloops.size() == 1)
            {
                varloop_indices = "        int I" + std::to_string(varloops[0].first) + " = loadidx;\n";
            }
            else if (varloops.size() > 1)
            {
                int loopd = varloops[0].first;
                varloop_indices = "        int I" + std::to_string(varloops[0].first) + " = loadidx % ";
                varloop_indices += std::to_string(Kernel.GetLoopDim(loopd)) + ";\n";
                for (int vli = 1; vli < varloops.size(); ++vli)
                {
                    int loopd = varloops[vli].first;
                    varloop_indices += "        int I" + std::to_string(loopd) + " = (loadidx / ";
                    varloop_indices += std::to_string(varloops[vli-1].second) + ")";
                    varloop_indices += " % " + std::to_string(Kernel.GetLoopDim(loopd));
                    varloop_indices += ";\n";
                }
            }
        }

        std::string varoff;
        if (trivial_varoff)
        {
            varoff = "__mul24(loadidx," + std::to_string(trivial_varoff_mult) + ")";
        }
        else
        {
            varoff = "0";
            for (int vli = 0; vli < varloops.size(); ++ vli)
            {
                int loopd = varloops[vli].first;
                int loop_stride = Kernel.GetLoopStride(loopd);
                int var_dim = Kernel.GetLoopNumVarDim(loopd, vi-1);
                int var_stride = Kernel.GetVarDimStride(vi-1, var_dim);
                varoff += " + __mul24(I" + std::to_string(loopd) + ", " + std::to_string(var_stride) + ") ";
            }           
        }

        std::string loopoff;
        if (trivial_loopoff)
        {
            loopoff = "__mul24(loadidx," + std::to_string(trivial_loopoff_mult) + ")";
        }
        else
        {
            loopoff = "0";
            for (int vli = 0; vli < varloops.size(); ++ vli)
            {
                int loopd = varloops[vli].first;
                int loop_stride = Kernel.GetLoopStride(loopd);
                loopoff += " + __mul24(I" + std::to_string(loopd) + ", " + std::to_string(loop_stride) + ") ";
            }
        }

        //Assign the threads to the load loop indices
        int loadidx_bot = (curr_thread + loadidx_size - 1 < block_size) ? curr_thread : 0;
        int loadidx_top = loadidx_bot + loadidx_size - 1;
        curr_thread = loadidx_top + 1;

        str_replace_all(temp, "<VAR_NUM>", vi);
        str_replace_all(temp, "<LOADIDX_SIZE>", loadidx_size);
        str_replace_all(temp, "<LOADIDX_BOT>", loadidx_bot);
        str_replace_all(temp, "<LOADIDX_TOP>", loadidx_top);
        str_replace_all(temp, "<VARLOOP_INDICES>", varloop_indices);
        str_replace_all(temp, "<VAROFF>", varoff);
        str_replace_all(temp, "<LOOPOFF>", loopoff);
        code += temp;
    }
    return code;
}


std::string SMChunkPerBlockExecutor::GetMidxLoops(int num_outer_loops, int num_middle_loops)
{
    std::string code;
    for (int loopd = num_outer_loops; loopd < num_outer_loops + num_middle_loops; ++loopd)
    {
        std::string temp =  "    #pragma unroll\n"
                            "    for (int I<LOOPD> = 0; I<LOOPD> < <LOOPDIM>; ++I<LOOPD>) {\n";
        str_replace_all(temp, "<LOOPD>", loopd);
        str_replace_all(temp, "<LOOPDIM>", Kernel.GetLoopDim(loopd));
        code += temp;
    }
    return code;
}


std::string SMChunkPerBlockExecutor::GetMidxOffCode(int vi, int num_outer_loops, int num_middle_loops)
{
    std::string code = "0";
    for (int loopd = num_outer_loops; loopd < num_outer_loops + num_middle_loops; ++loopd)
    {
        if (Kernel.IsVarDependentOnLoop(vi-1, loopd))
        {
            int vard = Kernel.GetLoopNumVarDim(loopd, vi-1);
            int stride = Kernel.GetVarDimStride(vi-1, vard);
            code += " + I" + std::to_string(loopd) + "*" + std::to_string(stride);
        }
    }
    return code;
}

std::string SMChunkPerBlockExecutor::GetLoadVariantInputsCode(int block_size, int num_outer_loops, int num_middle_loops, int num_inner_loops)
{
    std::string code;
    int curr_thread = 0;
    for (int vi = 1; vi < Kernel.GetNumVars(); ++vi)
    {
        bool middle_invariant = true;
        for (int loopd = num_outer_loops; loopd < num_outer_loops + num_middle_loops; ++loopd)
        {
            if (Kernel.IsVarDependentOnLoop(vi-1, loopd))
            {
                middle_invariant = false;
            }
        }

        bool outer_invariant = true;
        for (int loopd = 0; loopd < num_outer_loops; ++loopd)
        {
            if (Kernel.IsVarDependentOnLoop(vi-1, loopd))
            {
                outer_invariant = false;
            }
        } 

        if (!middle_invariant)
        {
            std::string temp = 
            "        varbase = base<VAR_NUM> + <MIDX_OFF>;\n"
            "        for (int threadidx = threadIdx.x; threadidx >= <LOADIDX_BOT> && threadidx <= <LOADIDX_TOP>; threadidx += blockDim.x)\n"
            "        {\n"
            "            int loadidx = threadidx - <LOADIDX_BOT>;\n"
            "            int2 inidx_off = s_inidx_off<VAR_NUM>[loadidx];\n"
            "            int varoff = varbase + inidx_off.x;\n"
            "            int loopoff = inidx_off.y;\n" 
            "            double tval = <LOAD>(&tdata<VAR_NUM>[varoff]);\n"
                         "<BROADCAST_TVAL>"
            "        }\n\n";

            std::string loadstr = outer_invariant ? "__ldg" : "__ldcs";

            int loadidx_size = Kernel.GetVarStorageReqForInnerLoops(vi-1, num_inner_loops);
            std::string broadcast_tval = "";
            std::string sidx;
            int bcast_loop_count = 0;
            for (int loopd = num_outer_loops + num_middle_loops; loopd < Kernel.GetNumLoops(); ++loopd)
            {
                if (!Kernel.IsVarDependentOnLoop(vi-1, loopd))
                {
                    broadcast_tval += std::string(4*(3 + bcast_loop_count), ' ');
                    broadcast_tval += "#pragma unroll\n";
                    broadcast_tval += std::string(4*(3 + bcast_loop_count), ' ');
                    broadcast_tval += "for (int I<LOOPD> = 0; I<LOOPD> < <LOOPDIM>; ++I<LOOPD>) {\n";
                    sidx += " + I" + std::to_string(loopd) + "*" + std::to_string(Kernel.GetLoopStride(loopd));
                    str_replace_all(broadcast_tval, "<LOOPD>", loopd);
                    str_replace_all(broadcast_tval, "<LOOPDIM>", Kernel.GetLoopDim(loopd));
                    bcast_loop_count ++;
                }
            }

            if (bcast_loop_count == 0)
            {
                broadcast_tval = "            s_tdata<VAR_NUM>[loopoff] = tval;\n";
            }
            else
            {
                broadcast_tval += std::string(4*(3 + bcast_loop_count), ' ');
                broadcast_tval += "s_tdata<VAR_NUM>[loopoff" + sidx + "] = tval;\n";
                for (int i = bcast_loop_count-1; i >= 0; --i)
                {
                    broadcast_tval += std::string(4*(3 + i), ' ');
                    broadcast_tval += "}\n";
                }
            }

            //Assign the threads to the load loop indices
            int loadidx_bot = (curr_thread + loadidx_size - 1 < block_size) ? curr_thread : 0;
            int loadidx_top = loadidx_bot + loadidx_size - 1;
            curr_thread = loadidx_top + 1;

            str_replace_all(temp, "<MIDX_OFF>", GetMidxOffCode(vi, num_outer_loops, num_middle_loops));
            str_replace_all(temp, "<BROADCAST_TVAL>", broadcast_tval);
            str_replace_all(temp, "<LOADIDX_BOT>", loadidx_bot);
            str_replace_all(temp, "<LOADIDX_TOP>", loadidx_top);
            str_replace_all(temp, "<LOAD>", loadstr);
            str_replace_all(temp, "<VAR_NUM>", vi);
            code += temp;
        }
    }
    code += '\n';
    return code;
}


std::string SMChunkPerBlockExecutor::GetLoadInvariantInputsCode(int block_size, int num_outer_loops, int num_middle_loops, int num_inner_loops)
{
    std::string code;
    int curr_thread = 0;
    for (int vi = 1; vi < Kernel.GetNumVars(); ++vi)
    {
        bool middle_invariant = true;
        for (int loopd = num_outer_loops; loopd < num_outer_loops + num_middle_loops; ++loopd)
        {
            if (Kernel.IsVarDependentOnLoop(vi-1, loopd))
            {
                middle_invariant = false;
            }
        }

        bool outer_invariant = true;
        for (int loopd = 0; loopd < num_outer_loops; ++loopd)
        {
            if (Kernel.IsVarDependentOnLoop(vi-1, loopd))
            {
                outer_invariant = false;
            }
        }   

        if (middle_invariant)
        {
            std::string temp = 
            "        varbase = base<VAR_NUM> + <MIDX_OFF>;\n"
            "        for (int threadidx = threadIdx.x; threadidx >= <LOADIDX_BOT> && threadidx <= <LOADIDX_TOP>; threadidx += blockDim.x)\n"
            "        {\n"
            "            int loadidx = threadidx - <LOADIDX_BOT>;\n"
            "            int2 inidx_off = s_inidx_off<VAR_NUM>[loadidx];\n"
            "            int varoff = varbase + inidx_off.x;\n"
            "            int loopoff = inidx_off.y;\n" 
            "            double tval = <LOAD>(&tdata<VAR_NUM>[varoff]);\n"
                         "<BROADCAST_TVAL>"
            "        }\n\n";

            std::string loadstr = outer_invariant ? "__ldg" : "__ldcs";

            int loadidx_size = Kernel.GetVarStorageReqForInnerLoops(vi-1, num_inner_loops);
            std::string broadcast_tval = "";
            std::string sidx;
            int bcast_loop_count = 0;
            for (int loopd = num_outer_loops + num_middle_loops; loopd < Kernel.GetNumLoops(); ++loopd)
            {
                if (!Kernel.IsVarDependentOnLoop(vi-1, loopd))
                {
                    broadcast_tval += std::string(4*(3 + bcast_loop_count), ' ');
                    broadcast_tval += "#pragma unroll\n";
                    broadcast_tval += std::string(4*(3 + bcast_loop_count), ' ');
                    broadcast_tval += "for (int I<LOOPD> = 0; I<LOOPD> < <LOOPDIM>; ++I<LOOPD>) {\n";
                    sidx += " + I" + std::to_string(loopd) + "*" + std::to_string(Kernel.GetLoopStride(loopd));
                    str_replace_all(broadcast_tval, "<LOOPD>", loopd);
                    str_replace_all(broadcast_tval, "<LOOPDIM>", Kernel.GetLoopDim(loopd));
                    bcast_loop_count ++;
                }
            }

            if (bcast_loop_count == 0)
            {
                broadcast_tval = "            s_tdata<VAR_NUM>[loopoff] = tval;\n";
            }
            else
            {
                broadcast_tval += std::string(4*(3 + bcast_loop_count), ' ');
                broadcast_tval += "s_tdata<VAR_NUM>[loopoff" + sidx + "] = tval;\n";
                for (int i = bcast_loop_count-1; i >= 0; --i)
                {
                    broadcast_tval += std::string(4*(3 + i), ' ');
                    broadcast_tval += "}\n";
                }
            }

            //Assign the threads to the load loop indices
            int loadidx_bot = (curr_thread + loadidx_size - 1 < block_size) ? curr_thread : 0;
            int loadidx_top = loadidx_bot + loadidx_size - 1;
            curr_thread = loadidx_top + 1;

            str_replace_all(temp, "<MIDX_OFF>", GetMidxOffCode(vi, num_outer_loops, num_middle_loops));
            str_replace_all(temp, "<BROADCAST_TVAL>", broadcast_tval);
            str_replace_all(temp, "<LOADIDX_BOT>", loadidx_bot);
            str_replace_all(temp, "<LOADIDX_TOP>", loadidx_top);
            str_replace_all(temp, "<LOAD>", loadstr);
            str_replace_all(temp, "<VAR_NUM>", vi);
            code += temp;
        }
    }
    code += '\n';
    return code;
}

std::string SMChunkPerBlockExecutor::GetMultVarsCode(int block_size, int smidx_size)
{
    std::string code =
            "<LOOPHEAD>"
    "        {\n"
    //"            int loopidx = threadIdx.x;\n"
    "            double tval = s_tdata1[tidx];\n"
                 "<MULTIN>"
    "            s_tdata0[tidx] = tval;\n"
    "        }\n"
    "        __syncthreads();\n";

    std::string loophead;
    if (smidx_size < block_size)
    {
        loophead = "        tidx = threadIdx.x;\n"
                   "        if (tidx < <SMIDX_SIZE>)\n";
    }
    else
    {
        loophead = "        for (int tidx = threadIdx.x; tidx < <SMIDX_SIZE>; tidx += blockDim.x)\n";
    }

    std::string multin;
    for (int vi = 2; vi <= Kernel.GetNumInputVars(); ++vi)
    {
        multin += "            tval *= s_tdata" + std::to_string(vi) + "[tidx];\n";
    }

    str_replace_all(code, "<LOOPHEAD>", loophead);
    str_replace_all(code, "<MULTIN>", multin);
    return code;
}


}

#endif