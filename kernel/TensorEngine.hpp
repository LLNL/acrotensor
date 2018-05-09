//Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at the Lawrence Livermore National Laboratory
//Written by Aaron Fisher (fisher47@llnl.gov). LLNL-CODE-738419.
//All rights reserved.
//This file is part of Acrotensor. For details, see https://github.com/LLNL/acrotensor.

#ifndef ACROBATIC_TENSOR_ENGINE_HPP
#define ACROBATIC_TENSOR_ENGINE_HPP

#include <unordered_map>
#include <string>
#include "DimensionedMultiKernel.hpp"
#include "Executor.hpp"
#include "IndexMapping.hpp"

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
    std::string GetExecType() {return ExecutorType;}

    void operator()(const char *bare_kernel_str, Tensor &out, Tensor &in1);
    void operator()(const char *bare_kernel_str, Tensor &out, Tensor &in1, Tensor &in2);
    void operator()(const char *bare_kernel_str, Tensor &out, Tensor &in1, Tensor &in2, Tensor &in3);
    void operator()(const char *bare_kernel_str, Tensor &out, Tensor &in1, Tensor &in2, Tensor &in3, Tensor &in4);
    void operator()(const char *bare_kernel_str, Tensor &out, Tensor &in1, Tensor &in2, Tensor &in3, Tensor &in4, Tensor &in5);
    void operator()(const char *bare_kernel_str, Tensor &out, Tensor &in1, Tensor &in2, Tensor &in3, Tensor &in4, Tensor &in5, Tensor &in6);
    void operator()(const char *bare_kernel_str, Tensor &out, Tensor &in1, Tensor &in2, Tensor &in3, Tensor &in4, Tensor &in5, Tensor &in6, Tensor &in7);
    void operator()(const char *bare_kernel_str, Tensor &out, Tensor &in1, Tensor &in2, Tensor &in3, Tensor &in4, Tensor &in5, Tensor &in6, Tensor &in7, Tensor &in8);
    void operator()(const char *bare_kernel_str, Tensor *out, std::vector<Tensor*> &inputs);

    void operator()(std::string &kernel_str, Tensor &out, Tensor &in1);
    void operator()(std::string &kernel_str, Tensor &out, Tensor &in1, Tensor &in2);
    void operator()(std::string &kernel_str, Tensor &out, Tensor &in1, Tensor &in2, Tensor &in3);
    void operator()(std::string &kernel_str, Tensor &out, Tensor &in1, Tensor &in2, Tensor &in3, Tensor &in4);
    void operator()(std::string &kernel_str, Tensor &out, Tensor &in1, Tensor &in2, Tensor &in3, Tensor &in4, Tensor &in5);
    void operator()(std::string &kernel_str, Tensor &out, Tensor &in1, Tensor &in2, Tensor &in3, Tensor &in4, Tensor &in5, Tensor &in6);
    void operator()(std::string &kernel_str, Tensor &out, Tensor &in1, Tensor &in2, Tensor &in3, Tensor &in4, Tensor &in5, Tensor &in6, Tensor &in7);
    void operator()(std::string &kernel_str, Tensor &out, Tensor &in1, Tensor &in2, Tensor &in3, Tensor &in4, Tensor &in5, Tensor &in6, Tensor &in7, Tensor &in8);
    void operator()(std::string &kernel_str, Tensor *out, std::vector<Tensor*> &inputs);

    std::string GetImplementation(const char *bare_kernel_str, Tensor &out, Tensor &in1);
    std::string GetImplementation(const char *bare_kernel_str, Tensor &out, Tensor &in1, Tensor &in2);
    std::string GetImplementation(const char *bare_kernel_str, Tensor &out, Tensor &in1, Tensor &in2, Tensor &in3);
    std::string GetImplementation(const char *bare_kernel_str, Tensor &out, Tensor &in1, Tensor &in2, Tensor &in3, Tensor &in4);
    std::string GetImplementation(const char *bare_kernel_str, Tensor &out, Tensor &in1, Tensor &in2, Tensor &in3, Tensor &in4, Tensor &in5);
    std::string GetImplementation(const char *bare_kernel_str, Tensor &out, Tensor &in1, Tensor &in2, Tensor &in3, Tensor &in4, Tensor &in5, Tensor &in6);
    std::string GetImplementation(const char *bare_kernel_str, Tensor &out, Tensor &in1, Tensor &in2, Tensor &in3, Tensor &in4, Tensor &in5, Tensor &in6, Tensor &in7);
    std::string GetImplementation(const char *bare_kernel_str, Tensor &out, Tensor &in1, Tensor &in2, Tensor &in3, Tensor &in4, Tensor &in5, Tensor &in6, Tensor &in7, Tensor &in8);
    std::string GetImplementation(const char *bare_kernel_str, Tensor *out, std::vector<Tensor*> &inputs);

    std::string GetImplementation(std::string &kernel_str, Tensor &out, Tensor &in1);
    std::string GetImplementation(std::string &kernel_str, Tensor &out, Tensor &in1, Tensor &in2);
    std::string GetImplementation(std::string &kernel_str, Tensor &out, Tensor &in1, Tensor &in2, Tensor &in3);
    std::string GetImplementation(std::string &kernel_str, Tensor &out, Tensor &in1, Tensor &in2, Tensor &in3, Tensor &in4);
    std::string GetImplementation(std::string &kernel_str, Tensor &out, Tensor &in1, Tensor &in2, Tensor &in3, Tensor &in4, Tensor &in5);
    std::string GetImplementation(std::string &kernel_str, Tensor &out, Tensor &in1, Tensor &in2, Tensor &in3, Tensor &in4, Tensor &in5, Tensor &in6);
    std::string GetImplementation(std::string &kernel_str, Tensor &out, Tensor &in1, Tensor &in2, Tensor &in3, Tensor &in4, Tensor &in5, Tensor &in6, Tensor &in7);
    std::string GetImplementation(std::string &kernel_str, Tensor &out, Tensor &in1, Tensor &in2, Tensor &in3, Tensor &in4, Tensor &in5, Tensor &in6, Tensor &in7, Tensor &in8);
    std::string GetImplementation(std::string &kernel_str, Tensor *out, std::vector<Tensor*> &inputs);

    void BatchMatrixInverse(Tensor &Ainv, Tensor &A);
    void BatchMatrixDet(Tensor &Adet, Tensor &A);
    void BatchMatrixInvDet(Tensor &Ainv, Tensor &Adet, Tensor &A);
    void FlatIndexedScatter(Tensor &Aout, Tensor &Ait, IndexMapping &M);
    void FlatIndexedSumGather(Tensor &Aout, Tensor &Ait, IndexMapping &M);

    void Clear();
    bool IsGPUAvailable() {return isCudaReady();}
    void BeginMultiKernelLaunch();
    void EndMultiKernelLaunch();

    private:
    TensorKernel *GetAddTensorKernel(std::string &kernel_str);
    DimensionedKernel *GetAddDimensionedKernel(TensorKernel *kernel, Tensor *output, std::vector<Tensor*> &inputs);
    KernelExecutor *GetAddKernelExecutor();
    void MoveToComputeLocation(Tensor &T);
    void SwitchToComputeLocation(Tensor &T);
    void MoveToComputeLocation(IndexMapping &M);
    void SwitchToComputeLocation(IndexMapping &M);

    std::string ExecutorType;
    std::unordered_map<std::string, TensorKernel*> KernelMap;
    std::unordered_map<std::string, DimensionedKernel*> DimensionedKernelMap;
    std::unordered_map<std::string, KernelExecutor*> ExecutorMap;
    std::string ComputeLocation;
    NonContractionOps *Ops;

    bool IsMultiKernelLaunch;
    std::vector<DimensionedKernel*> MKLKernels;
    std::vector<Tensor*> MKLOutputT;
    std::vector<std::vector<Tensor*> > MKLInputT;

#ifdef ACRO_HAVE_CUDA
    cudaStream_t TheCudaStream;
#endif
};


}

#endif //ACROBATIC_TENSOR_ENGINE_HPP