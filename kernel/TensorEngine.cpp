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
    Ops = NULL;
    SetExecutorType(exec_type);
#ifdef ACRO_HAVE_CUDA
    TheCudaStream = NULL;
#endif
    IsMultiKernelLaunch = false;
}


TensorEngine::TensorEngine(const char *bare_exec_type)
{
    std::string exec_type(bare_exec_type);
    Ops = NULL;
    SetExecutorType(exec_type);
#ifdef ACRO_HAVE_CUDA    
    TheCudaStream = NULL;
#endif
    IsMultiKernelLaunch = false;
}


TensorEngine::TensorEngine(std::string &exec_type)
{
    Ops = NULL;
    SetExecutorType(exec_type);
#ifdef ACRO_HAVE_CUDA    
    TheCudaStream = NULL;
#endif
    IsMultiKernelLaunch = false;
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
    ACROBATIC_ASSERT(exec_type == "Cuda" ||
                     exec_type == "CPUInterpreted",
                     "Unknown Executor type:  " + exec_type);
#ifndef ACRO_HAVE_CUDA
    ACROBATIC_ASSERT(exec_type != "Cuda", 
                     "CUDA required for executor:  " + exec_type);
#endif

    Clear();
    ExecutorType = exec_type;

    if (ExecutorType == "CPUInterpreted")
    {
        ComputeLocation = "CPU";
        Ops = new NativeCPUOps();
    }
#ifdef ACRO_HAVE_CUDA    
    if (ExecutorType == "Cuda")
    {
        ComputeLocation = "GPU";
        Ops = new CudaGPUOps();
    }
#endif
}


void TensorEngine::operator()(const char *bare_kernel_str, Tensor &out, Tensor &in1)
{
    std::string kernel_str(bare_kernel_str);
    std::vector<Tensor*> input_tensors;
    input_tensors.push_back(&in1);
    (*this)(kernel_str, &out, input_tensors);
}


void TensorEngine::operator()(const char *bare_kernel_str, Tensor &out, Tensor &in1, Tensor &in2)
{
    std::string kernel_str(bare_kernel_str);
    std::vector<Tensor*> input_tensors;
    input_tensors.push_back(&in1);
    input_tensors.push_back(&in2);
    (*this)(kernel_str, &out, input_tensors);
}


void TensorEngine::operator()(const char *bare_kernel_str, Tensor &out, Tensor &in1, Tensor &in2, Tensor &in3)
{
    std::string kernel_str(bare_kernel_str);
    std::vector<Tensor*> input_tensors;
    input_tensors.push_back(&in1);
    input_tensors.push_back(&in2);
    input_tensors.push_back(&in3);
    (*this)(kernel_str, &out, input_tensors);
}


void TensorEngine::operator()(const char *bare_kernel_str, Tensor &out, Tensor &in1, Tensor &in2, Tensor &in3, Tensor &in4)
{
    std::string kernel_str(bare_kernel_str);
    std::vector<Tensor*> input_tensors;
    input_tensors.push_back(&in1);
    input_tensors.push_back(&in2);
    input_tensors.push_back(&in3);
    input_tensors.push_back(&in4);
    (*this)(kernel_str, &out, input_tensors);
}


void TensorEngine::operator()(const char *bare_kernel_str, Tensor &out, Tensor &in1, Tensor &in2, Tensor &in3, Tensor &in4, Tensor &in5)
{
    std::string kernel_str(bare_kernel_str);
    std::vector<Tensor*> input_tensors;
    input_tensors.push_back(&in1);
    input_tensors.push_back(&in2);
    input_tensors.push_back(&in3);
    input_tensors.push_back(&in4);
    input_tensors.push_back(&in5);
    (*this)(kernel_str, &out, input_tensors);
}


void TensorEngine::operator()(const char *bare_kernel_str, Tensor &out, Tensor &in1, Tensor &in2, Tensor &in3, Tensor &in4, Tensor &in5, Tensor &in6)
{
    std::string kernel_str(bare_kernel_str);
    std::vector<Tensor*> input_tensors;
    input_tensors.push_back(&in1);
    input_tensors.push_back(&in2);
    input_tensors.push_back(&in3);
    input_tensors.push_back(&in4);
    input_tensors.push_back(&in5);
    input_tensors.push_back(&in6);
    (*this)(kernel_str, &out, input_tensors);
}


void TensorEngine::operator()(const char *bare_kernel_str, Tensor &out, Tensor &in1, Tensor &in2, Tensor &in3, Tensor &in4, Tensor &in5, Tensor &in6, Tensor &in7)
{
    std::string kernel_str(bare_kernel_str);
    std::vector<Tensor*> input_tensors;
    input_tensors.push_back(&in1);
    input_tensors.push_back(&in2);
    input_tensors.push_back(&in3);
    input_tensors.push_back(&in4);
    input_tensors.push_back(&in5);
    input_tensors.push_back(&in6);
    input_tensors.push_back(&in7);
    (*this)(kernel_str, &out, input_tensors);
}


void TensorEngine::operator()(const char *bare_kernel_str, Tensor &out, Tensor &in1, Tensor &in2, Tensor &in3, Tensor &in4, Tensor &in5, Tensor &in6, Tensor &in7, Tensor &in8)
{
    std::string kernel_str(bare_kernel_str);
    std::vector<Tensor*> input_tensors;
    input_tensors.push_back(&in1);
    input_tensors.push_back(&in2);
    input_tensors.push_back(&in3);
    input_tensors.push_back(&in4);
    input_tensors.push_back(&in5);
    input_tensors.push_back(&in6);
    input_tensors.push_back(&in7);
    input_tensors.push_back(&in8);
    (*this)(kernel_str, &out, input_tensors);
}


void TensorEngine::operator()(std::string &kernel_str, Tensor &out, Tensor &in1)
{
    std::vector<Tensor*> input_tensors;
    input_tensors.push_back(&in1);
    (*this)(kernel_str, &out, input_tensors);
}


void TensorEngine::operator()(std::string &kernel_str, Tensor &out, Tensor &in1, Tensor &in2)
{
    std::vector<Tensor*> input_tensors;
    input_tensors.push_back(&in1);
    input_tensors.push_back(&in2);
    (*this)(kernel_str, &out, input_tensors);
}


void TensorEngine::operator()(std::string &kernel_str, Tensor &out, Tensor &in1, Tensor &in2, Tensor &in3)
{
    std::vector<Tensor*> input_tensors;
    input_tensors.push_back(&in1);
    input_tensors.push_back(&in2);
    input_tensors.push_back(&in3);
    (*this)(kernel_str, &out, input_tensors);
}


void TensorEngine::operator()(std::string &kernel_str, Tensor &out, Tensor &in1, Tensor &in2, Tensor &in3, Tensor &in4)
{
    std::vector<Tensor*> input_tensors;
    input_tensors.push_back(&in1);
    input_tensors.push_back(&in2);
    input_tensors.push_back(&in3);
    input_tensors.push_back(&in4);
    (*this)(kernel_str, &out, input_tensors);
}


void TensorEngine::operator()(std::string &kernel_str, Tensor &out, Tensor &in1, Tensor &in2, Tensor &in3, Tensor &in4, Tensor &in5)
{
    std::vector<Tensor*> input_tensors;
    input_tensors.push_back(&in1);
    input_tensors.push_back(&in2);
    input_tensors.push_back(&in3);
    input_tensors.push_back(&in4);
    input_tensors.push_back(&in5);
    (*this)(kernel_str, &out, input_tensors);
}


void TensorEngine::operator()(std::string &kernel_str, Tensor &out, Tensor &in1, Tensor &in2, Tensor &in3, Tensor &in4, Tensor &in5, Tensor &in6)
{
    std::vector<Tensor*> input_tensors;
    input_tensors.push_back(&in1);
    input_tensors.push_back(&in2);
    input_tensors.push_back(&in3);
    input_tensors.push_back(&in4);
    input_tensors.push_back(&in5);
    input_tensors.push_back(&in6);
    (*this)(kernel_str, &out, input_tensors);
}


void TensorEngine::operator()(std::string &kernel_str, Tensor &out, Tensor &in1, Tensor &in2, Tensor &in3, Tensor &in4, Tensor &in5, Tensor &in6, Tensor &in7)
{
    std::vector<Tensor*> input_tensors;
    input_tensors.push_back(&in1);
    input_tensors.push_back(&in2);
    input_tensors.push_back(&in3);
    input_tensors.push_back(&in4);
    input_tensors.push_back(&in5);
    input_tensors.push_back(&in6);
    input_tensors.push_back(&in7);
    (*this)(kernel_str, &out, input_tensors);
}


void TensorEngine::operator()(std::string &kernel_str, Tensor &out, Tensor &in1, Tensor &in2, Tensor &in3, Tensor &in4, Tensor &in5, Tensor &in6, Tensor &in7, Tensor &in8)
{
    std::vector<Tensor*> input_tensors;
    input_tensors.push_back(&in1);
    input_tensors.push_back(&in2);
    input_tensors.push_back(&in3);
    input_tensors.push_back(&in4);
    input_tensors.push_back(&in5);
    input_tensors.push_back(&in6);
    input_tensors.push_back(&in7);
    input_tensors.push_back(&in8);
    (*this)(kernel_str, &out, input_tensors);
}


void TensorEngine::operator()(std::string &kernel_str, Tensor *output, std::vector<Tensor*> &inputs)
{
    ACROBATIC_ASSERT(output->IsInitialized(), "Output variable not initilized in:  " + kernel_str);
    for (int ivari = 0; ivari < inputs.size(); ++ivari)
    {
        ACROBATIC_ASSERT(inputs[ivari]->IsInitialized(), "Input variable not initilized in:  " + kernel_str);
    }

    if (!IsMultiKernelLaunch)
    {
        MKLKernels.resize(0);
        MKLOutputT.resize(0);
        MKLInputT.resize(0);
    }
    TensorKernel *kernel = GetAddTensorKernel(kernel_str);
    DimensionedKernel *dimensioned_kernel = GetAddDimensionedKernel(kernel, output, inputs);
    MKLKernels.push_back(dimensioned_kernel);
    MKLOutputT.push_back(output);
    MKLInputT.push_back(inputs);

    if (!IsMultiKernelLaunch)
    {
        KernelExecutor *executor = GetAddKernelExecutor();
        executor->ExecuteSingle(output, inputs);
    }
}


void TensorEngine::BeginMultiKernelLaunch()
{
    ACROBATIC_ASSERT(!IsMultiKernelLaunch, "Trying to BeginMultiKernelLaunch twice.");
    IsMultiKernelLaunch = true;
    MKLKernels.resize(0);
    MKLOutputT.resize(0);
    MKLInputT.resize(0);
}


void TensorEngine::EndMultiKernelLaunch()
{
    ACROBATIC_ASSERT(IsMultiKernelLaunch, "Found and EndMultiKernelLaunch before a BeginMultiKernelLaunch!");
    IsMultiKernelLaunch = false;

    KernelExecutor *executor = GetAddKernelExecutor();
    executor->ExecuteMulti(MKLOutputT, MKLInputT);

}


std::string TensorEngine::GetImplementation(const char *bare_kernel_str, Tensor *output, std::vector<Tensor*> &inputs)
{
    std::string kernel_str(bare_kernel_str);
    return GetImplementation(kernel_str, output, inputs);
}


std::string TensorEngine::GetImplementation(const char *bare_kernel_str, Tensor &out, Tensor &in1)
{
    std::string kernel_str(bare_kernel_str);
    std::vector<Tensor*> input_tensors;
    input_tensors.push_back(&in1);
    return GetImplementation(kernel_str, &out, input_tensors);
}


std::string TensorEngine::GetImplementation(const char *bare_kernel_str, Tensor &out, Tensor &in1, Tensor &in2)
{
    std::string kernel_str(bare_kernel_str);
    std::vector<Tensor*> input_tensors;
    input_tensors.push_back(&in1);
    input_tensors.push_back(&in2);
    return GetImplementation(kernel_str, &out, input_tensors);
}


std::string TensorEngine::GetImplementation(const char *bare_kernel_str, Tensor &out, Tensor &in1, Tensor &in2, Tensor &in3)
{
    std::string kernel_str(bare_kernel_str);
    std::vector<Tensor*> input_tensors;
    input_tensors.push_back(&in1);
    input_tensors.push_back(&in2);
    input_tensors.push_back(&in3);
    return GetImplementation(kernel_str, &out, input_tensors);
}


std::string TensorEngine::GetImplementation(const char *bare_kernel_str, Tensor &out, Tensor &in1, Tensor &in2, Tensor &in3, Tensor &in4)
{
    std::string kernel_str(bare_kernel_str);
    std::vector<Tensor*> input_tensors;
    input_tensors.push_back(&in1);
    input_tensors.push_back(&in2);
    input_tensors.push_back(&in3);
    input_tensors.push_back(&in4);
    return GetImplementation(kernel_str, &out, input_tensors);
}


std::string TensorEngine::GetImplementation(const char *bare_kernel_str, Tensor &out, Tensor &in1, Tensor &in2, Tensor &in3, Tensor &in4, Tensor &in5)
{
    std::string kernel_str(bare_kernel_str);
    std::vector<Tensor*> input_tensors;
    input_tensors.push_back(&in1);
    input_tensors.push_back(&in2);
    input_tensors.push_back(&in3);
    input_tensors.push_back(&in4);
    input_tensors.push_back(&in5);
    return GetImplementation(kernel_str, &out, input_tensors);
}


std::string TensorEngine::GetImplementation(const char *bare_kernel_str, Tensor &out, Tensor &in1, Tensor &in2, Tensor &in3, Tensor &in4, Tensor &in5, Tensor &in6)
{
    std::string kernel_str(bare_kernel_str);
    std::vector<Tensor*> input_tensors;
    input_tensors.push_back(&in1);
    input_tensors.push_back(&in2);
    input_tensors.push_back(&in3);
    input_tensors.push_back(&in4);
    input_tensors.push_back(&in5);
    input_tensors.push_back(&in6);
    return GetImplementation(kernel_str, &out, input_tensors);
}


std::string TensorEngine::GetImplementation(const char *bare_kernel_str, Tensor &out, Tensor &in1, Tensor &in2, Tensor &in3, Tensor &in4, Tensor &in5, Tensor &in6, Tensor &in7)
{
    std::string kernel_str(bare_kernel_str);
    std::vector<Tensor*> input_tensors;
    input_tensors.push_back(&in1);
    input_tensors.push_back(&in2);
    input_tensors.push_back(&in3);
    input_tensors.push_back(&in4);
    input_tensors.push_back(&in5);
    input_tensors.push_back(&in6);
    input_tensors.push_back(&in7);
    return GetImplementation(kernel_str, &out, input_tensors);
}


std::string TensorEngine::GetImplementation(const char *bare_kernel_str, Tensor &out, Tensor &in1, Tensor &in2, Tensor &in3, Tensor &in4, Tensor &in5, Tensor &in6, Tensor &in7, Tensor &in8)
{
    std::string kernel_str(bare_kernel_str);
    std::vector<Tensor*> input_tensors;
    input_tensors.push_back(&in1);
    input_tensors.push_back(&in2);
    input_tensors.push_back(&in3);
    input_tensors.push_back(&in4);
    input_tensors.push_back(&in5);
    input_tensors.push_back(&in6);
    input_tensors.push_back(&in7);
    input_tensors.push_back(&in8);
    return GetImplementation(kernel_str, &out, input_tensors);
}


std::string TensorEngine::GetImplementation(std::string &kernel_str, Tensor &out, Tensor &in1)
{
    std::vector<Tensor*> input_tensors;
    input_tensors.push_back(&in1);
    return GetImplementation(kernel_str, &out, input_tensors);
}


std::string TensorEngine::GetImplementation(std::string &kernel_str, Tensor &out, Tensor &in1, Tensor &in2)
{
    std::vector<Tensor*> input_tensors;
    input_tensors.push_back(&in1);
    input_tensors.push_back(&in2);
    return GetImplementation(kernel_str, &out, input_tensors);
}


std::string TensorEngine::GetImplementation(std::string &kernel_str, Tensor &out, Tensor &in1, Tensor &in2, Tensor &in3)
{
    std::vector<Tensor*> input_tensors;
    input_tensors.push_back(&in1);
    input_tensors.push_back(&in2);
    input_tensors.push_back(&in3);
    return GetImplementation(kernel_str, &out, input_tensors);
}


std::string TensorEngine::GetImplementation(std::string &kernel_str, Tensor &out, Tensor &in1, Tensor &in2, Tensor &in3, Tensor &in4)
{
    std::vector<Tensor*> input_tensors;
    input_tensors.push_back(&in1);
    input_tensors.push_back(&in2);
    input_tensors.push_back(&in3);
    input_tensors.push_back(&in4);
    return GetImplementation(kernel_str, &out, input_tensors);
}


std::string TensorEngine::GetImplementation(std::string &kernel_str, Tensor &out, Tensor &in1, Tensor &in2, Tensor &in3, Tensor &in4, Tensor &in5)
{
    std::vector<Tensor*> input_tensors;
    input_tensors.push_back(&in1);
    input_tensors.push_back(&in2);
    input_tensors.push_back(&in3);
    input_tensors.push_back(&in4);
    input_tensors.push_back(&in5);
    return GetImplementation(kernel_str, &out, input_tensors);
}


std::string TensorEngine::GetImplementation(std::string &kernel_str, Tensor &out, Tensor &in1, Tensor &in2, Tensor &in3, Tensor &in4, Tensor &in5, Tensor &in6)
{
    std::vector<Tensor*> input_tensors;
    input_tensors.push_back(&in1);
    input_tensors.push_back(&in2);
    input_tensors.push_back(&in3);
    input_tensors.push_back(&in4);
    input_tensors.push_back(&in5);
    input_tensors.push_back(&in6);
    return GetImplementation(kernel_str, &out, input_tensors);
}


std::string TensorEngine::GetImplementation(std::string &kernel_str, Tensor &out, Tensor &in1, Tensor &in2, Tensor &in3, Tensor &in4, Tensor &in5, Tensor &in6, Tensor &in7)
{
    std::vector<Tensor*> input_tensors;
    input_tensors.push_back(&in1);
    input_tensors.push_back(&in2);
    input_tensors.push_back(&in3);
    input_tensors.push_back(&in4);
    input_tensors.push_back(&in5);
    input_tensors.push_back(&in6);
    input_tensors.push_back(&in7);
    return GetImplementation(kernel_str, &out, input_tensors);
}


std::string TensorEngine::GetImplementation(std::string &kernel_str, Tensor &out, Tensor &in1, Tensor &in2, Tensor &in3, Tensor &in4, Tensor &in5, Tensor &in6, Tensor &in7, Tensor &in8)
{
    std::vector<Tensor*> input_tensors;
    input_tensors.push_back(&in1);
    input_tensors.push_back(&in2);
    input_tensors.push_back(&in3);
    input_tensors.push_back(&in4);
    input_tensors.push_back(&in5);
    input_tensors.push_back(&in6);
    input_tensors.push_back(&in7);
    input_tensors.push_back(&in8);
    return GetImplementation(kernel_str, &out, input_tensors);
}


std::string TensorEngine::GetImplementation(std::string &kernel_str, Tensor *output, std::vector<Tensor*> &inputs)
{
    ACROBATIC_ASSERT(!IsMultiKernelLaunch, "GetImplementation not currently supported for multi kernel launch.");

    MKLKernels.resize(0);
    MKLOutputT.resize(0);
    MKLInputT.resize(0);
    TensorKernel *kernel = GetAddTensorKernel(kernel_str);
    DimensionedKernel *dimensioned_kernel = GetAddDimensionedKernel(kernel, output, inputs);
    MKLKernels.push_back(dimensioned_kernel);
    MKLOutputT.push_back(output);
    MKLInputT.push_back(inputs);

    KernelExecutor *executor = GetAddKernelExecutor();
    return executor->GetImplementation();
}



TensorKernel *TensorEngine::GetAddTensorKernel(std::string &kernel_str)
{
    auto it = KernelMap.find(kernel_str);
    if (it != KernelMap.end())
    {
        return it->second;
    }
    else
    {
        TensorKernel *new_kernel = new TensorKernel(kernel_str);
        KernelMap[kernel_str] = new_kernel;
        return new_kernel;
    }
}


DimensionedKernel *TensorEngine::GetAddDimensionedKernel(TensorKernel *kernel, Tensor *output, std::vector<Tensor*> &inputs)
{
    DimensionedKernel *new_kernel;
    std::string dimensioned_kernel_str = kernel->GetDimensionedNameString(output, inputs);
    auto it = DimensionedKernelMap.find(dimensioned_kernel_str);
    if (it != DimensionedKernelMap.end())
    {
        return it->second;
    }
    else
    {
        new_kernel = new DimensionedKernel(kernel, output, inputs);
        DimensionedKernelMap[dimensioned_kernel_str] = new_kernel;
    
        ACROBATIC_ASSERT(output->GetRank() == kernel->OutputVar.IndexNames.size(), 
                         "Tensor rank of the output var in the kernel does not match the rank of the actual tensor.\n"
                        +"Kernel:  " + kernel->KernelStr);
        ACROBATIC_ASSERT(inputs.size() == kernel->InputVars.size());
        for (int i = 0; i < inputs.size(); ++i)
        {
            ACROBATIC_ASSERT(inputs[i]->GetRank() == kernel->InputVars[i].IndexNames.size(),
                             "Tensor rank of input var " + std::to_string(i) +
                             " does not match the rank of the actual tensor.\n" +
                            +"Kernel:  " + kernel->KernelStr);
        }
        return new_kernel;
    }
}


KernelExecutor *TensorEngine::GetAddKernelExecutor()
{
    std::string mkl_string = "";
    for (int ki = 0; ki < MKLKernels.size(); ++ki)
    {
        mkl_string += MKLKernels[ki]->GetDimensionedNameString() + ";";
    }

    auto it = ExecutorMap.find(mkl_string);
    if (it != ExecutorMap.end())
    {
        return it->second;
    }
    else
    {
        DimensionedMultiKernel *dimensioned_multi_kernel = new DimensionedMultiKernel(MKLKernels);
        KernelExecutor *new_executor = KernelExecutor::Create(ExecutorType, dimensioned_multi_kernel);
        ExecutorMap[mkl_string] = new_executor;
        return new_executor;
    }
}

 

void TensorEngine::BatchMatrixInverse(Tensor &Ainv, Tensor &A)
{
    int rank = A.GetRank();
    ACROBATIC_ASSERT(rank >= 2, "Can't BatchMatrixInverse a tensor of rank < 2.");
    ACROBATIC_ASSERT(rank == Ainv.GetRank(), "Can't BatchMatrixInverse with mismatched ranks for Ainv and in.");
    for (int d = 0; d < rank; ++d)
    {
        ACROBATIC_ASSERT(Ainv.GetDim(d) == A.GetDim(d), "Can't BatchMatrixInverse with mismatched dims for Ainv and in.");
    }
    ACROBATIC_ASSERT(Ainv.GetDim(rank-1) == Ainv.GetDim(rank-2), "Can't BatchMatrixInverse with mismatched dims for m,n.")
    ACROBATIC_ASSERT(Ainv.GetDim(rank-1) <= 3, "Can't BatchMatrixInverse with matrix dims > 3.")

    SwitchToComputeLocation(Ainv);
    MoveToComputeLocation(A);

    Ops->BatchMatrixInverse(Ainv, A);
}


void TensorEngine::BatchMatrixDet(Tensor &Adet, Tensor &A)
{
    int rank = A.GetRank();
    ACROBATIC_ASSERT(rank >= 3, "Can't BatchMatrixDet a tensor of rank < 2.");
    ACROBATIC_ASSERT(Adet.GetRank() == rank - 2, "Can't BatchMatrixDet with mismatched ranks for Adet and A.");
    for (int d = 0; d < Adet.GetRank(); ++d)
    {
        ACROBATIC_ASSERT(Adet.GetDim(d) == A.GetDim(d), "Can't BatchMatrixDet with mismatched dims for Adet and A.");
    }
    ACROBATIC_ASSERT(A.GetDim(rank-1) == A.GetDim(rank-2), "Can't BatchMatrixDet with mismatched dims for m,n in the matrix.")
    ACROBATIC_ASSERT(A.GetDim(rank-1) <= 3, "Can't BatchMatrixInverse with matrix dims > 3.")

    SwitchToComputeLocation(Adet);
    MoveToComputeLocation(A);

    Ops->BatchMatrixDet(Adet, A);
}


void TensorEngine::BatchMatrixInvDet(Tensor &Ainv, Tensor &Adet, Tensor &A)
{
    int rank = A.GetRank();
    ACROBATIC_ASSERT(rank >= 2, "Can't BatchMatrixInverse a tensor of rank < 2.");
    ACROBATIC_ASSERT(rank == Ainv.GetRank(), "Can't BatchMatrixInverse with mismatched ranks for Ainv and in.");
    for (int d = 0; d < rank; ++d)
    {
        ACROBATIC_ASSERT(Ainv.GetDim(d) == A.GetDim(d), "Can't BatchMatrixInverse with mismatched dims for Ainv and A.");
    }
    ACROBATIC_ASSERT(Ainv.GetDim(rank-1) == Ainv.GetDim(rank-2), "Can't BatchMatrixInverse with mismatched dims for m,n.")
    ACROBATIC_ASSERT(Ainv.GetDim(rank-1) <= 3, "Can't BatchMatrixInverse with matrix dims > 3.")

    ACROBATIC_ASSERT(rank >= 3, "Can't BatchMatrixDet a tensor of rank < 2.");
    ACROBATIC_ASSERT(Adet.GetRank() == rank - 2, "Can't BatchMatrixDet with mismatched ranks for Adet and A.");
    for (int d = 0; d < Adet.GetRank(); ++d)
    {
        ACROBATIC_ASSERT(Adet.GetDim(d) == A.GetDim(d), "Can't BatchMatrixDet with mismatched dims for Adet and A.");
    }
    ACROBATIC_ASSERT(A.GetDim(rank-1) == A.GetDim(rank-2), "Can't BatchMatrixDet with mismatched dims for m,n in the matrix.")

    SwitchToComputeLocation(Ainv);
    SwitchToComputeLocation(Adet);
    MoveToComputeLocation(A);

    Ops->BatchMatrixInvDet(Ainv, Adet, A);
}


void TensorEngine::FlatIndexedScatter(Tensor &Aout, Tensor &Ain, IndexMapping &M)
{
    ACROBATIC_ASSERT(Aout.GetSize() == M.GetRangeSize(), "IndexMapping RangeSize does not match the output Tensor size.");
    SwitchToComputeLocation(Aout);
    MoveToComputeLocation(Ain);
    MoveToComputeLocation(M);
    Ops->FlatIndexedScatter(Aout, Ain, M);
}


void TensorEngine::FlatIndexedSumGather(Tensor &Aout, Tensor &Ain, IndexMapping &M)
{
    ACROBATIC_ASSERT(Aout.GetSize() == M.GetDomainSize(), "IndexMapping DomainSize does not match the output Tensor size.");
    if(!M.IsInverseComputed())
    {
        M.ComputeInverse();
    }
    SwitchToComputeLocation(Aout);
    MoveToComputeLocation(Ain);
    MoveToComputeLocation(M);
    Ops->FlatIndexedSumGather(Aout, Ain, M);
}




void TensorEngine::Clear()
{
    for (auto kernel : KernelMap)
    {
        delete kernel.second;
    }
    KernelMap.clear();
    for (auto dimensioned_kernel : DimensionedKernelMap)
    {
        delete dimensioned_kernel.second;
    }
    DimensionedKernelMap.clear();
    for (auto executor : ExecutorMap)
    {
        delete executor.second;
    }
    ExecutorMap.clear();

    if (Ops != NULL)
    {
        delete Ops;
    }
    Ops = NULL;
}


void TensorEngine::MoveToComputeLocation(Tensor &T)
{
    if (ComputeLocation == "CPU")
    {
        T.MoveFromGPU();
    }
    else if (ComputeLocation == "GPU")
    {
        T.MoveToGPU();
    }
}

void TensorEngine::SwitchToComputeLocation(Tensor &T)
{
    if (ComputeLocation == "CPU")
    {
        T.SwitchFromGPU();
    }
    else if (ComputeLocation == "GPU")
    {
        T.SwitchToGPU();
    }
}


void TensorEngine::MoveToComputeLocation(IndexMapping &M)
{
    if (ComputeLocation == "CPU")
    {
        M.MoveFromGPU();
    }
    else if (ComputeLocation == "GPU")
    {
        M.MoveToGPU();
    }
}

void TensorEngine::SwitchToComputeLocation(IndexMapping &M)
{
    if (ComputeLocation == "CPU")
    {
        M.SwitchFromGPU();
    }
    else if (ComputeLocation == "GPU")
    {
        M.SwitchToGPU();
    }
}


}