//Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at the Lawrence Livermore National Laboratory
//Written by Aaron Fisher (fisher47@llnl.gov). LLNL-CODE-738419.
//All rights reserved.
//This file is part of Acrotensor. For details, see https://github.com/LLNL/acrotensor.

#include "KernelExecutor.hpp"
#include "TensorKernel.hpp"
#include <cctype>
#include <algorithm>
#include <iostream>

namespace acro
{

void KernelExecutor::operator()(Tensor &out, Tensor &in1)
{
    std::vector<Tensor*> input_tensors;
    input_tensors.push_back(&in1);
    (*this)(&out, input_tensors);
}


void KernelExecutor::operator()(Tensor &out, Tensor &in1, Tensor &in2)
{
    std::vector<Tensor*> input_tensors;
    input_tensors.push_back(&in1);
    input_tensors.push_back(&in2);
    (*this)(&out, input_tensors);
}


void KernelExecutor::operator()(Tensor &out, Tensor &in1, Tensor &in2, Tensor &in3)
{
    std::vector<Tensor*> input_tensors;
    input_tensors.push_back(&in1);
    input_tensors.push_back(&in2);
    input_tensors.push_back(&in3);
    (*this)(&out, input_tensors);
}


void KernelExecutor::operator()(Tensor &out, Tensor &in1, Tensor &in2, Tensor &in3, Tensor &in4)
{
    std::vector<Tensor*> input_tensors;
    input_tensors.push_back(&in1);
    input_tensors.push_back(&in2);
    input_tensors.push_back(&in3);
    input_tensors.push_back(&in4);
    (*this)(&out, input_tensors);
}


void KernelExecutor::operator()(Tensor &out, Tensor &in1, Tensor &in2, Tensor &in3, Tensor &in4, Tensor &in5)
{
    std::vector<Tensor*> input_tensors;
    input_tensors.push_back(&in1);
    input_tensors.push_back(&in2);
    input_tensors.push_back(&in3);
    input_tensors.push_back(&in4);
    input_tensors.push_back(&in5);
    (*this)(&out, input_tensors);
}


void KernelExecutor::operator()(Tensor &out, Tensor &in1, Tensor &in2, Tensor &in3, Tensor &in4, Tensor &in5, Tensor &in6)
{
    std::vector<Tensor*> input_tensors;
    input_tensors.push_back(&in1);
    input_tensors.push_back(&in2);
    input_tensors.push_back(&in3);
    input_tensors.push_back(&in4);
    input_tensors.push_back(&in5);
    input_tensors.push_back(&in6);
    (*this)(&out, input_tensors);
}


void KernelExecutor::operator()(Tensor &out, Tensor &in1, Tensor &in2, Tensor &in3, Tensor &in4, Tensor &in5, Tensor &in6, Tensor &in7)
{
    std::vector<Tensor*> input_tensors;
    input_tensors.push_back(&in1);
    input_tensors.push_back(&in2);
    input_tensors.push_back(&in3);
    input_tensors.push_back(&in4);
    input_tensors.push_back(&in5);
    input_tensors.push_back(&in6);
    input_tensors.push_back(&in7);
    (*this)(&out, input_tensors);
}


void KernelExecutor::operator()(Tensor &out, Tensor &in1, Tensor &in2, Tensor &in3, Tensor &in4, Tensor &in5, Tensor &in6, Tensor &in7, Tensor &in8)
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
    (*this)(&out, input_tensors);
}


void KernelExecutor::operator()(Tensor *out, std::vector<Tensor*> &inputs)
{
    ACROBATIC_ASSERT(out->GetRank() == Kernel.OutputVar.IndexNames.size(), 
                     "Tensor rank of the output var in the kernel does not match the rank of the actual tensor.\n"
                    +"Kernel:  " + Kernel.KernelStr);
    ACROBATIC_ASSERT(inputs.size() == Kernel.InputVars.size());
    for (int i = 0; i < inputs.size(); ++i)
    {
        ACROBATIC_ASSERT(inputs[i]->GetRank() == Kernel.InputVars[i]->IndexNames.size(),
                         "Tensor rank of input var " + std::to_string(i) +
                         " does not match the rank of the actual tensor.\n" +
                        +"Kernel:  " + Kernel.KernelStr);
    }

    ExecuteKernel(out, inputs);
}


std::string KernelExecutor::GetImplementation(Tensor &out, Tensor &in1)
{
    std::vector<Tensor*> input_tensors;
    input_tensors.push_back(&in1);
    return GetImplementation(&out, input_tensors);
}


std::string KernelExecutor::GetImplementation(Tensor &out, Tensor &in1, Tensor &in2)
{
    std::vector<Tensor*> input_tensors;
    input_tensors.push_back(&in1);
    input_tensors.push_back(&in2);
    return GetImplementation(&out, input_tensors);
}


std::string KernelExecutor::GetImplementation(Tensor &out, Tensor &in1, Tensor &in2, Tensor &in3)
{
    std::vector<Tensor*> input_tensors;
    input_tensors.push_back(&in1);
    input_tensors.push_back(&in2);
    input_tensors.push_back(&in3);
    return GetImplementation(&out, input_tensors);
}


std::string KernelExecutor::GetImplementation(Tensor &out, Tensor &in1, Tensor &in2, Tensor &in3, Tensor &in4)
{
    std::vector<Tensor*> input_tensors;
    input_tensors.push_back(&in1);
    input_tensors.push_back(&in2);
    input_tensors.push_back(&in3);
    input_tensors.push_back(&in4);
    return GetImplementation(&out, input_tensors);
}


std::string KernelExecutor::GetImplementation(Tensor &out, Tensor &in1, Tensor &in2, Tensor &in3, Tensor &in4, Tensor &in5)
{
    std::vector<Tensor*> input_tensors;
    input_tensors.push_back(&in1);
    input_tensors.push_back(&in2);
    input_tensors.push_back(&in3);
    input_tensors.push_back(&in4);
    input_tensors.push_back(&in5);
    return GetImplementation(&out, input_tensors);
}


std::string KernelExecutor::GetImplementation(Tensor &out, Tensor &in1, Tensor &in2, Tensor &in3, Tensor &in4, Tensor &in5, Tensor &in6)
{
    std::vector<Tensor*> input_tensors;
    input_tensors.push_back(&in1);
    input_tensors.push_back(&in2);
    input_tensors.push_back(&in3);
    input_tensors.push_back(&in4);
    input_tensors.push_back(&in5);
    input_tensors.push_back(&in6);
    return GetImplementation(&out, input_tensors);
}


std::string KernelExecutor::GetImplementation(Tensor &out, Tensor &in1, Tensor &in2, Tensor &in3, Tensor &in4, Tensor &in5, Tensor &in6, Tensor &in7)
{
    std::vector<Tensor*> input_tensors;
    input_tensors.push_back(&in1);
    input_tensors.push_back(&in2);
    input_tensors.push_back(&in3);
    input_tensors.push_back(&in4);
    input_tensors.push_back(&in5);
    input_tensors.push_back(&in6);
    input_tensors.push_back(&in7);
    return GetImplementation(&out, input_tensors);
}


std::string KernelExecutor::GetImplementation(Tensor &out, Tensor &in1, Tensor &in2, Tensor &in3, Tensor &in4, Tensor &in5, Tensor &in6, Tensor &in7, Tensor &in8)
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
    return GetImplementation(&out, input_tensors);
}


void KernelExecutor::MoveTensorsFromGPU(Tensor *out, std::vector<Tensor*> &inputs)
{
    if (out->IsOnGPU())
    {
        out->MoveFromGPU();
    }

    for (int i = 0; i < inputs.size(); ++i)
    {
        if (inputs[i]->IsOnGPU())
        {
            inputs[i]->MoveFromGPU();
        }
    }
}


void KernelExecutor::MoveTensorsToGPU(Tensor *out, std::vector<Tensor*> &inputs)
{
    if (!out->IsOnGPU())
    {
        if (!out->IsMappedToGPU())
        {
            out->MapToGPU();
        }
        out->MoveToGPU();
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


void KernelExecutor::MoveTensorsToOutputLocation(Tensor *out, std::vector<Tensor*> &inputs)
{
    if (out->IsOnGPU())
    {
        MoveTensorsToGPU(out, inputs);
    }
    else
    {
        MoveTensorsFromGPU(out, inputs);
    }
}


}