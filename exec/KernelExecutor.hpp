//Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at the Lawrence Livermore National Laboratory
//Written by Aaron Fisher (fisher47@llnl.gov). LLNL-CODE-738419.
//All rights reserved.
//This file is part of Acrotensor. For details, see https://github.com/LLNL/acrotensor.

#ifndef ACROBATIC_KERNEL_EXECUTOR_HPP
#define ACROBATIC_KERNEL_EXECUTOR_HPP

#include <string>
#include "Tensor.hpp"
#include "DimensionedMultiKernel.hpp"

namespace acro
{

class KernelExecutor
{
    public:
    KernelExecutor(DimensionedMultiKernel *multi_kernel);
    static KernelExecutor *Create(std::string exec_type, DimensionedMultiKernel *multi_kernel);
    virtual ~KernelExecutor();

    virtual std::string GetImplementation() = 0;
    virtual std::string GetExecType() = 0;
    virtual void ExecuteSingle(Tensor *output, std::vector<Tensor*> &inputs) = 0;
    virtual void ExecuteMulti(std::vector<Tensor*> &output, std::vector<std::vector<Tensor*> > &inputs);

#ifdef ACRO_HAVE_CUDA    
    inline void SetCudaStream(cudaStream_t cuda_stream) {TheCudaStream = cuda_stream;}
#endif

    protected:

    void MoveTensorsFromGPU(Tensor *output, std::vector<Tensor*> &inputs);
    void MoveTensorsToGPU(Tensor *output, std::vector<Tensor*> &inputs);
    void MoveTensorsToOutputLocation(Tensor *output, std::vector<Tensor*> &inputs);
    DimensionedMultiKernel *MultiKernel;
    DimensionedKernel *FirstKernel;
    std::vector<DimensionedMultiKernel*> SubKernels;
    std::vector<KernelExecutor*> SubExecutors;

#ifdef ACRO_HAVE_CUDA
    cudaStream_t TheCudaStream;
#endif    
};

}


#endif //ACROBATIC_KERNEL_EXECUTOR_HPP
