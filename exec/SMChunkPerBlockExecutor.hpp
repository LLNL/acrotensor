//Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at the Lawrence Livermore National Laboratory
//Written by Aaron Fisher (fisher47@llnl.gov). LLNL-CODE-738419.
//All rights reserved.
//This file is part of Acrotensor. For details, see https://github.com/LLNL/acrotensor.

#ifndef ACROBATIC_SMCHUNKPERBLOCK_EXECUTOR_HPP
#define ACROBATIC_SMCHUNKPERBLOCK_EXECUTOR_HPP

#ifdef ACRO_HAVE_CUDA

#include "KernelExecutor.hpp"
#include <map>

#include <cuda.h>
#include <nvrtc.h>

namespace acro
{

class SMChunkPerBlockExecutor : public KernelExecutor
{
    public:
    SMChunkPerBlockExecutor(std::string &kernelstr);
    ~SMChunkPerBlockExecutor();
    virtual std::string GetImplementation(Tensor *output, std::vector<Tensor*> &inputs);
    virtual void ExecuteKernel(Tensor *output, std::vector<Tensor*> &inputs);

    private:
    void ExecuteLoopsCuda();
    CudaKernel *GenerateCudaKernel();
    void GetLoopingStructure(int &block_size, int &num_outer_loops, int &num_middle_loops, int &num_inner_loops);

    std::string GetCodeTemplate();
    std::string GetFunctionName();
    std::string GetLoopDimsCode();
    std::string GetParamsCode();
    std::string GetInitSTdataCode();
    std::string GetOuterIvalsCode(int num_outer_loops);
    std::string GetBaseIndicesCode(int num_outer_loops);
    std::string GetLoadIndicesCode(int block_size, int num_outer_loops, int num_middle_loops, int num_inner_loops);
    std::string GetMidxLoops(int num_outer_loops, int num_middle_loops);
    std::string GetMidxOffCode(int vi, int num_outer_loops, int num_middle_loops);
    std::string GetLoadInvariantInputsCode(int block_size, int num_outer_loops, int num_middle_loops, int num_inner_loops);
    std::string GetLoadVariantInputsCode(int block_size, int num_outer_loops, int num_middle_loops, int num_inner_loops);
    std::string GetMultVarsCode(int block_size, int smidx_size);

    void GetVarBlockIndexData(int numblockloops, std::vector<std::vector<ushort2>> &offset);
    std::string GetLoadInputsToSMCode(std::vector<std::vector<int>> &offset,
                                      std::vector<std::vector<int>> &sortmap);

    cudaDeviceProp CudaDeviceProp;
    std::map<std::vector<int>, CudaKernel* > CudaKernelMap;
};

}

#endif

#endif //ACROBATIC_SMCHUNKPERBLOCK_EXECUTOR_HPP