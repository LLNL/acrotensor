//Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at the Lawrence Livermore National Laboratory
//Written by Aaron Fisher (fisher47@llnl.gov). LLNL-CODE-738419.
//All rights reserved.
//This file is part of Acrotensor. For details, see https://github.com/LLNL/acrotensor.

#include "Executor.hpp"
#include "TensorKernel.hpp"

namespace acro
{


KernelExecutor::KernelExecutor(DimensionedMultiKernel *multi_kernel)
{
    MultiKernel = multi_kernel;
    if (MultiKernel->Kernels.size() > 0)
    {
        FirstKernel = MultiKernel->Kernels[0];
    }
    else
    {
        FirstKernel = NULL;
    }

#ifdef ACRO_HAVE_CUDA
    TheCudaStream = NULL;
#endif
}


void KernelExecutor::MoveTensorsFromGPU(Tensor *output, std::vector<Tensor*> &inputs)
{
    if (output->IsOnGPU())
    {
        output->MoveFromGPU();
    }

    for (int i = 0; i < inputs.size(); ++i)
    {
        if (inputs[i]->IsOnGPU())
        {
            inputs[i]->MoveFromGPU();
        }
    }
}


void KernelExecutor::MoveTensorsToGPU(Tensor *output, std::vector<Tensor*> &inputs)
{
    if (!output->IsOnGPU())
    {
        if (!output->IsMappedToGPU())
        {
            output->MapToGPU();
        }
        output->MoveToGPU();
    }

    for (int i = 0; i < inputs.size(); ++i)
    {
        if (!inputs[i]->IsOnGPU())
        {
            if (!inputs[i]->IsMappedToGPU())
            {
                inputs[i]->MapToGPU();
            }
            inputs[i]->MoveToGPU();
        }
    }
}


void KernelExecutor::MoveTensorsToOutputLocation(Tensor *output, std::vector<Tensor*> &inputs)
{
    if (output->IsOnGPU())
    {
        MoveTensorsToGPU(output, inputs);
    }
    else
    {
        MoveTensorsFromGPU(output, inputs);
    }
}


void KernelExecutor::ExecuteMulti(std::vector<Tensor*> &output, std::vector<std::vector<Tensor*> > &inputs)
{
    if (SubExecutors.size() != MultiKernel->Kernels.size())
    {
        SubKernels.resize(MultiKernel->Kernels.size());
        SubExecutors.resize(MultiKernel->Kernels.size());
        for (int ki = 0; ki < MultiKernel->Kernels.size(); ++ki)
        {
            SubKernels[ki] = new DimensionedMultiKernel(MultiKernel->Kernels[ki]);
            SubExecutors[ki] = KernelExecutor::Create(GetExecType(), SubKernels[ki]);
        }
    }

    for (int ki = 0; ki < MultiKernel->Kernels.size(); ++ki)
    {
        SubExecutors[ki]->ExecuteSingle(output[ki], inputs[ki]);
    }
}


KernelExecutor *KernelExecutor::Create(std::string exec_type, DimensionedMultiKernel *multi_kernel)
{
    if (exec_type == "CPUInterpreted")
    {
        return new CPUInterpretedExecutor(multi_kernel);    
    }
#ifdef ACRO_HAVE_CUDA    
    if (exec_type == "Cuda")
    {
        return new CudaExecutor(multi_kernel);
    }
#endif

    ACROBATIC_ASSERT(false, "Executor type does not exist:  " + exec_type);
    return NULL;
}


KernelExecutor::~KernelExecutor()
{
    for (int ki = 0; ki < SubExecutors.size(); ++ki)
    {
        delete SubKernels[ki];
        delete SubExecutors[ki];
    }
}

}