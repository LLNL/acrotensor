//Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at the Lawrence Livermore National Laboratory
//Written by Aaron Fisher (fisher47@llnl.gov). LLNL-CODE-738419.
//All rights reserved.
//This file is part of Acrotensor. For details, see https://github.com/LLNL/acrotensor.

#include "TensorEngine.hpp"
#include "TensorKernel.hpp"
#include "Ops.hpp"
#include <iostream>

namespace acro
{


TensorEngine::TensorEngine()
{
    std::string exec_type("CPUInterpreted");
    SetExecutorType(exec_type);
    IsAsyncLaunch = false;
    TheCudaStream = NULL;    
}


TensorEngine::TensorEngine(const char *bare_exec_type)
{
    std::string exec_type(bare_exec_type);
    SetExecutorType(exec_type);
    IsAsyncLaunch = false;
    TheCudaStream = NULL;
}


TensorEngine::TensorEngine(std::string &exec_type)
{
    SetExecutorType(exec_type);
    IsAsyncLaunch = false;
    TheCudaStream = NULL;  
}



TensorEngine::~TensorEngine()
{
    Clear();
#ifdef ACRO_HAVE_CUDA
    if (TheCudaStream != NULL)
    {
        acroCudaErrorCheck(cuStreamDestroy(TheCudaStream));
    }
#endif
}


void TensorEngine::SetExecutorType(const char *bare_exec_type)
{
    std::string exec_type(bare_exec_type);
    SetExecutorType(exec_type);
}


void TensorEngine::SetExecutorType(std::string &exec_type)
{
    ACROBATIC_ASSERT(exec_type == "OneOutPerThread" ||
                     exec_type == "MultiOutPerThread" ||
                     exec_type == "SMChunkPerBlock" ||
                     exec_type == "CPUInterpreted" ||
                     exec_type == "IndexCached",
                     "Unknown Executor type:  " + exec_type);
#ifndef ACRO_HAVE_CUDA
    ACROBATIC_ASSERT(exec_type != "OneOutPerThread" &&
                     exec_type != "MultiOutPerThread" &&
                     exec_type != "SMChunkPerBlock",
                     "CUDA required for executor:  " + exec_type);
#endif

    Clear();
    ExecutorType = exec_type;
}


KernelExecutor *TensorEngine::GetNewExecutor(std::string &kernel_str)
{
    if (ExecutorType == "CPUInterpreted")
    {
        ComputeLocation = "CPU";
        Ops = new NativeCPUOps();
        return new CPUInterpretedExecutor(kernel_str);    
    }
    if (ExecutorType == "IndexCached")
    {
        ComputeLocation = "CPU";
        Ops = new NativeCPUOps();
        return new IndexCachedExecutor(kernel_str);
    }
#ifdef ACRO_HAVE_CUDA    
    if (ExecutorType == "OneOutPerThread")
    {
        ComputeLocation = "GPU";
        Ops = new CudaGPUOps();
        return new OneOutPerThreadExecutor(kernel_str);
    }
    if (ExecutorType == "MultiOutPerThread")
    {
        ComputeLocation = "GPU";
        Ops = new CudaGPUOps();
        return new MultiOutPerThreadExecutor(kernel_str);    
    }
    if (ExecutorType == "SMChunkPerBlock")
    {
        ComputeLocation = "GPU";
        Ops = new CudaGPUOps();
        return new SMChunkPerBlockExecutor(kernel_str);
    }
#endif

    ACROBATIC_ASSERT(false, "Executor type does not exist:  " + ExecutorType);
    return nullptr;
}


KernelExecutor &TensorEngine::operator[](const char* bare_kernel_str)
{
    std::string kernel_str(bare_kernel_str);
    return (*this)[kernel_str];
}


KernelExecutor &TensorEngine::operator[](std::string &kernel_str)
{
    auto it = ExecutorMap.find(kernel_str);
    KernelExecutor *executor;
    if (it != ExecutorMap.end())
    {
        executor = it->second;
    }
    else
    {
        executor = GetNewExecutor(kernel_str);
        ExecutorMap[kernel_str] = executor;
    }

#ifdef ACRO_HAVE_CUDA
    if (IsAsyncLaunch)
    {
        executor->SetCudaStream(TheCudaStream);
    }
    else
    {
        executor->SetCudaStream(NULL);
    }
#endif    

    return *executor;
}
 

void TensorEngine::BatchMatrixInverse(Tensor &out, Tensor &in)
{
    int rank = out.GetRank();
    ACROBATIC_ASSERT(rank >= 2, "Can't BatchMatrixInverse a matrix of rank < 2.");
    ACROBATIC_ASSERT(rank == in.GetRank(), "Can't BatchMatrixInverse with mismatched ranks for out and in.");
    for (int d = 0; d < rank; ++d)
    {
        ACROBATIC_ASSERT(out.GetDim(d) == in.GetDim(d), "Can't BatchMatrixInverse with mismatched dims for out and in.");
    }
    ACROBATIC_ASSERT(out.GetDim(rank-1) == out.GetDim(rank-2), "Can't BatchMatrixInverse with mismatched dims for m,n.")
    ACROBATIC_ASSERT(out.GetDim(rank-1) <= 3, "Can't BatchMatrixInverse with matrix dims > 3.")

    SwitchTensorToComputeLocation(out);
    MoveTensorToComputeLocation(in);

    Ops->BatchMatrixInverse(out, in);
}

void TensorEngine::Clear()
{
    for (auto it = ExecutorMap.begin(); it != ExecutorMap.end(); ++it)
    {
        delete it->second;
    }
    ExecutorMap.clear();
}

void TensorEngine::SetAsyncLaunch()
{
    ACROBATIC_ASSERT(!IsAsyncLaunch, "Trying to SetAsyncLaunch twice.");
    IsAsyncLaunch = true;
#ifdef ACRO_HAVE_CUDA
    if (TheCudaStream == NULL)
    {
        acroCudaErrorCheck(cuCtxSynchronize());
        acroCudaErrorCheck(cuStreamCreate(&TheCudaStream,CU_STREAM_DEFAULT));
    }
#endif 
}


void TensorEngine::ReSyncLaunch()
{
#ifdef ACRO_HAVE_CUDA
    if (TheCudaStream != NULL)
    {
        acroCudaErrorCheck(cuCtxSynchronize());
        acroCudaErrorCheck(cuStreamSynchronize(TheCudaStream));
    }
#endif 
    IsAsyncLaunch = false;
}


void TensorEngine::MoveTensorToComputeLocation(Tensor &T)
{
    if (ComputeLocation == "CPU")
    {
        if (T.IsOnGPU())
        {
            T.MoveFromGPU();
        }
    }
    else if (ComputeLocation == "GPU")
    {
        if (!T.IsOnGPU())
        {
            T.MoveToGPU();
        }
    }
}

void TensorEngine::SwitchTensorToComputeLocation(Tensor &T)
{
    if (ComputeLocation == "CPU")
    {
        if (T.IsOnGPU())
        {
            T.SwitchFromGPU();
        }
    }
    else if (ComputeLocation == "GPU")
    {
        if (!T.IsOnGPU())
        {
            T.SwitchToGPU();
        }
    }
}


}