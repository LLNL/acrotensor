//Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at the Lawrence Livermore National Laboratory
//Written by Aaron Fisher (fisher47@llnl.gov). LLNL-CODE-738419.
//All rights reserved.
//This file is part of Acrotensor. For details, see https://github.com/LLNL/acrotensor.

#ifndef ACROBATIC_CUDA_EXECUTOR_HPP
#define ACROBATIC_CUDA_EXECUTOR_HPP

#ifdef ACRO_HAVE_CUDA
#include "KernelExecutor.hpp"
#include <string>
#include <vector>
#include <nvrtc.h>

namespace acro
{

class CudaExecutor : public KernelExecutor
{
    public:
    CudaExecutor(DimensionedMultiKernel *multi_kernel);
    ~CudaExecutor();
    virtual void ExecuteSingle(Tensor *output, std::vector<Tensor*> &inputs);
    virtual void ExecuteMulti(std::vector<Tensor*> &output, std::vector<std::vector<Tensor*> > &inputs);
    virtual std::string GetImplementation();
    virtual std::string GetExecType() {return "Cuda";}

    private:
    void GenerateCudaKernel();
    void ReorderIndices(std::vector<std::string> &mk_outer_indices);
    int GetNumBlockLoops();
    int GetMinMidIdxSize(int num_block_loops);
    int GetMaxMidIdxSize(int num_block_loops);
    int GetNumThreadsPerBlock(int num_block_loops);
    void GetSharedMemUvars();
    void GetSharedMemWRKernels();
    std::vector<int> GetMidloopsOrder(int ki);
    std::vector<int> GetMidloopsStrides(DimensionedKernel *kernel, std::vector<int> &mid_loops);

    std::string GenSharedMemPreload();
    std::string GenSharedMemWRBuffer();
    std::string GenInitIndices();
    std::string GenSubKernelLoops();
    std::string GenTensor(int ki, int vari);
    std::string GenTensor(int uvari);
    std::string GenMidLoopIndices(int ki, std::vector<int> &mid_loops, std::vector<int> &mid_loop_strides, int blocki = -1);
    std::string GenVarIndex(int ki, int vari, int blocki = -1, bool blockdims=true);
    std::string GenVarSubIndex(int ki, int vari, int dimi, int blocki = -1);
    std::string GenLoopIndex(int ki, int loopi, int blocki = -1);
    
    cudaDeviceProp CudaDeviceProp;
    CudaKernel *TheCudaKernel;

    int NumBlockLoops;
    double **HDeviceTensors;

    int SharedMemAllocated;
    int SMWRBufferSize;
    std::vector<bool> SharedMemUvars;
    std::vector<bool> SharedMemWRKernels;

    std::vector<void*> KernelParams;
};

}

#endif

#endif //ACROBATIC_ONEOUTPERTHREAD_EXECUTOR_HPP