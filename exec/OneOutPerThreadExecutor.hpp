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
    std::vector<bool> GetSharedMemUvars();

    std::string GenSharedMemPreload(std::vector<bool> &sharedmem_uvars);
    std::string GenIOutVars();
    std::string GenSubKernelLoops(std::vector<bool> &sharedmem_uvars);
    std::string GenVarIndex(int ki, int vari);
    cudaDeviceProp CudaDeviceProp;
    CudaKernel *TheCudaKernel;
};

}

#endif

#endif //ACROBATIC_ONEOUTPERTHREAD_EXECUTOR_HPP