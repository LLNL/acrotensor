//Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at the Lawrence Livermore National Laboratory
//Written by Aaron Fisher (fisher47@llnl.gov). LLNL-CODE-738419.
//All rights reserved.
//This file is part of Acrotensor. For details, see https://github.com/LLNL/acrotensor.

#ifndef ACROBATIC_CPUINTERPRETED_EXECUTOR_HPP
#define ACROBATIC_CPUINTERPRETED_EXECUTOR_HPP

#include "KernelExecutor.hpp"
#include <map>

namespace acro
{

class CPUInterpretedExecutor : public KernelExecutor
{
    public:
    CPUInterpretedExecutor(std::string &kernelstr);
    ~CPUInterpretedExecutor();
    virtual void ExecuteKernel(Tensor *output, std::vector<Tensor*> &inputs);
    virtual std::string GetImplementation(Tensor *out, std::vector<Tensor*> &inputs);

    private:
    void Execute1Loops();
    void Execute2Loops();
    void Execute3Loops();
    void Execute4Loops();
    void Execute5Loops();
    void Execute6Loops();
    void Execute7Loops();
    void Execute8Loops();
    void Execute9Loops();
    void Execute10Loops();
    void Execute11Loops();
    void Execute12Loops();
    void ExecuteArbitraryLoops();

    inline double ComputeRHS(const std::vector<Tensor*> &inputs, const int *RESTRICT I);
};


inline double CPUInterpretedExecutor::ComputeRHS(const std::vector<Tensor*> &inputs, const int *RESTRICT I)
{
    double rhs_val = (*inputs[0])[ComputeRawIdx(*inputs[0], I, Kernel.InputVars[0]->LoopNums)];
    for  (unsigned int vari = 1; vari < inputs.size(); ++vari)
    {
        rhs_val *= (*inputs[vari])[ComputeRawIdx(*inputs[vari], I, Kernel.InputVars[vari]->LoopNums)];
    }
    return rhs_val;
}

}

#endif //ACROBATIC_CPUINTERPRETED_EXECUTOR_HPP