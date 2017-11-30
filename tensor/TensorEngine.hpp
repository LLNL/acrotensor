//Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at the Lawrence Livermore National Laboratory
//Written by Aaron Fisher (fisher47@llnl.gov). LLNL-CODE-738419.
//All rights reserved.
//This file is part of Acrotensor. For details, see https://github.com/LLNL/acrotensor.

#ifndef ACROBATIC_TENSOR_ENGINE_HPP
#define ACROBATIC_TENSOR_ENGINE_HPP

#include <map>
#include <string>
#include "TensorKernel.hpp"
#include "Executor.hpp"

#ifdef ACRO_HAVE_CUDA
#include <cuda.h>
#endif 


namespace acro
{

class TensorKernel;
class NonContractionOps;

class TensorEngine
{
    public:
    TensorEngine();
    TensorEngine(const char *bare_exec_type);
    TensorEngine(std::string &exec_type);
    ~TensorEngine();
    void SetExecutorType(const char *bare_exec_type);
    void SetExecutorType(std::string &exec_type);

    KernelExecutor &operator[](const char* bare_kernel_str);
    KernelExecutor &operator[](std::string &kernel_str);
    void BatchMatrixInverse(Tensor &out, Tensor &in);

    void Clear();
    bool IsGPUAvailable() {return isCudaReady();}
    void SetAsyncLaunch();
    void ReSyncLaunch();

    private:
    bool IsAsyncLaunch;

    KernelExecutor *GetNewExecutor(std::string &kernel_str);
    void MoveTensorToComputeLocation(Tensor &T);
    void SwitchTensorToComputeLocation(Tensor &T);

    std::string ExecutorType;
    std::map<std::string, KernelExecutor*> ExecutorMap;
    std::string ComputeLocation;
    NonContractionOps *Ops;

#ifdef ACRO_HAVE_CUDA
    cudaStream_t TheCudaStream;
#endif
};


}

#endif //ACROBATIC_TENSOR_ENGINE_HPP