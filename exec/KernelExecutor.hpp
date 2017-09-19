//Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at the Lawrence Livermore National Laboratory
//Written by Aaron Fisher (fisher47@llnl.gov). LLNL-CODE-738419.
//All rights reserved.
//This file is part of Acrotensor. For details, see https://github.com/LLNL/acrotensor.

#ifndef ACROBATIC_KERNEL_EXECUTOR_HPP
#define ACROBATIC_KERNEL_EXECUTOR_HPP

#include <string>
#include "Tensor.hpp"
#include "TensorKernel.hpp"

namespace acro
{

class TensorKernel;

class KernelExecutor
{
    public:
    KernelExecutor(std::string &kernelstr) : Kernel(kernelstr), TheCudaStream(NULL) {};
    void operator()(Tensor &out, Tensor &in1);
    void operator()(Tensor &out, Tensor &in1, Tensor &in2);
    void operator()(Tensor &out, Tensor &in1, Tensor &in2, Tensor &in3);
    void operator()(Tensor &out, Tensor &in1, Tensor &in2, Tensor &in3, Tensor &in4);
    void operator()(Tensor &out, Tensor &in1, Tensor &in2, Tensor &in3, Tensor &in4, Tensor &in5);
    void operator()(Tensor &out, Tensor &in1, Tensor &in2, Tensor &in3, Tensor &in4, Tensor &in5, Tensor &in6);
    void operator()(Tensor &out, Tensor &in1, Tensor &in2, Tensor &in3, Tensor &in4, Tensor &in5, Tensor &in6, Tensor &in7);
    void operator()(Tensor &out, Tensor &in1, Tensor &in2, Tensor &in3, Tensor &in4, Tensor &in5, Tensor &in6, Tensor &in7, Tensor &in8);
    void operator()(Tensor *out, std::vector<Tensor*> &inputs);

    std::string GetImplementation(Tensor &out, Tensor &in1);
    std::string GetImplementation(Tensor &out, Tensor &in1, Tensor &in2);
    std::string GetImplementation(Tensor &out, Tensor &in1, Tensor &in2, Tensor &in3);
    std::string GetImplementation(Tensor &out, Tensor &in1, Tensor &in2, Tensor &in3, Tensor &in4);
    std::string GetImplementation(Tensor &out, Tensor &in1, Tensor &in2, Tensor &in3, Tensor &in4, Tensor &in5);
    std::string GetImplementation(Tensor &out, Tensor &in1, Tensor &in2, Tensor &in3, Tensor &in4, Tensor &in5, Tensor &in6);
    std::string GetImplementation(Tensor &out, Tensor &in1, Tensor &in2, Tensor &in3, Tensor &in4, Tensor &in5, Tensor &in6, Tensor &in7);
    std::string GetImplementation(Tensor &out, Tensor &in1, Tensor &in2, Tensor &in3, Tensor &in4, Tensor &in5, Tensor &in6, Tensor &in7, Tensor &in8);
    virtual std::string GetImplementation(Tensor *out, std::vector<Tensor*> &inputs) = 0;
    
    virtual void ExecuteKernel(Tensor *out, std::vector<Tensor*> &inputs) = 0;
    inline int ComputeRawIdx(const Tensor &T, const int *RESTRICT I, const std::vector<int> &loop_nums);

#ifdef ACRO_HAVE_CUDA    
    inline void SetCudaStream(cudaStream_t cuda_stream) {TheCudaStream = cuda_stream;}
#endif

    protected:
    void MoveTensorsFromGPU(Tensor *out, std::vector<Tensor*> &inputs);
    void MoveTensorsToGPU(Tensor *out, std::vector<Tensor*> &inputs);
    void MoveTensorsToOutputLocation(Tensor *out, std::vector<Tensor*> &inputs);
    TensorKernel Kernel;

#ifdef ACRO_HAVE_CUDA
    cudaStream_t TheCudaStream;
#endif    
};


inline int KernelExecutor::ComputeRawIdx(const Tensor &T, const int *RESTRICT I, const std::vector<int> &loop_nums)
{   
    int raw_idx = I[loop_nums[0]]*T.GetStride(0);
    for (int d = 1; d < T.GetRank(); ++d)
    {
        raw_idx += I[loop_nums[d]]*T.GetStride(d);
    }
    return raw_idx;
}

}


#endif //ACROBATIC_KERNEL_EXECUTOR_HPP
