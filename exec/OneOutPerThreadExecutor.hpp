//Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at the Lawrence Livermore National Laboratory
//Written by Aaron Fisher (fisher47@llnl.gov). LLNL-CODE-738419.
//All rights reserved.
//This file is part of Acrotensor. For details, see https://github.com/LLNL/acrotensor.

#ifndef ACROBATIC_ONEOUTPERTHREAD_EXECUTOR_HPP
#define ACROBATIC_ONEOUTPERTHREAD_EXECUTOR_HPP

#ifdef ACRO_HAVE_CUDA
#include "KernelExecutor.hpp"
#include <map>
#include <nvrtc.h>

namespace acro
{

class OneOutPerThreadExecutor : public KernelExecutor
{
    public:
    OneOutPerThreadExecutor(std::string &kernelstr);
    ~OneOutPerThreadExecutor();
    virtual void ExecuteKernel(Tensor *output, std::vector<Tensor*> &inputs);
    virtual std::string GetImplementation(Tensor *out, std::vector<Tensor*> &inputs);

    private:
    void ExecuteLoopsCuda();

    CudaKernel *GenerateCudaKernel();
    void GetSharedMemInvars(const std::vector<int> &N, std::vector<bool> &sharedmem_invars);
    cudaDeviceProp CudaDeviceProp;
    std::map<std::vector<int>, CudaKernel* > CudaKernelMap;
};

}

#endif

#endif //ACROBATIC_ONEOUTPERTHREAD_EXECUTOR_HPP