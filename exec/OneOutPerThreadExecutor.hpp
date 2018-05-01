//Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at the Lawrence Livermore National Laboratory
//Written by Aaron Fisher (fisher47@llnl.gov). LLNL-CODE-738419.
//All rights reserved.
//This file is part of Acrotensor. For details, see https://github.com/LLNL/acrotensor.

#ifndef ACROBATIC_ONEOUTPERTHREAD_EXECUTOR_HPP
#define ACROBATIC_ONEOUTPERTHREAD_EXECUTOR_HPP

#ifdef ACRO_HAVE_CUDA
#include "KernelExecutor.hpp"
#include <string>
#include <vector>
#include <nvrtc.h>

namespace acro
{

class OneOutPerThreadExecutor : public KernelExecutor
{
    public:
    OneOutPerThreadExecutor(DimensionedMultiKernel *multi_kernel);
    ~OneOutPerThreadExecutor();
    virtual void ExecuteSingle(Tensor *output, std::vector<Tensor*> &inputs);
    virtual void ExecuteMulti(std::vector<Tensor*> &output, std::vector<std::vector<Tensor*> > &inputs);
    virtual std::string GetImplementation();
    virtual std::string GetExecType() {return "OneOutPerThread";}

    private:
    void GenerateCudaKernel();
    void ReorderIndices(std::vector<std::string> &mk_outer_indices);
    int GetNumBlockLoops();
    int GetMinMidIdxSize(int num_block_loops);
    int GetMaxMidIdxSize(int num_block_loops);
    int GetNumThreadsPerBlock(int num_block_loops);
    std::vector<bool> GetSharedMemUvars();
    std::vector<int> GetMidloopsOrder(int ki, std::vector<bool> &sharedmem_uvars);
    std::vector<int> GetMidloopsStrides(DimensionedKernel *kernel, std::vector<int> &mid_loops);

    std::string GenSharedMemPreload(std::vector<bool> &sharedmem_uvars);
    std::string GenInitIndices();
    std::string GenSubKernelLoops(std::vector<bool> &sharedmem_uvars);
    std::string GenTensor(int ki, int vari);
    std::string GenTensor(int uvari);
    std::string GenMidLoopIndices(int ki, std::vector<int> &mid_loops, std::vector<int> &mid_loop_strides, int blocki = -1);
    std::string GenVarIndex(int ki, int vari, int blocki = -1);
    std::string GenVarSubIndex(int ki, int vari, int dimi, int blocki = -1);
    std::string GenLoopIndex(int ki, int loopi, int blocki = -1);
    
    cudaDeviceProp CudaDeviceProp;
    CudaKernel *TheCudaKernel;

    int NumBlockLoops;
    double **HDeviceTensors;
    std::vector<void*> KernelParams;
};

}

#endif

#endif //ACROBATIC_ONEOUTPERTHREAD_EXECUTOR_HPP