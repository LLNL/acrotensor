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
    CPUInterpretedExecutor(DimensionedMultiKernel *multi_kernel);
    ~CPUInterpretedExecutor();
    virtual void ExecuteSingle(Tensor *output, std::vector<Tensor*> &inputs);
    virtual std::string GetImplementation();
    virtual std::string GetExecType() {return "CPUInterpreted";}

    private:
    void Execute1Loops(Tensor *output, std::vector<Tensor*> &inputs);
    void Execute2Loops(Tensor *output, std::vector<Tensor*> &inputs);
    void Execute3Loops(Tensor *output, std::vector<Tensor*> &inputs);
    void Execute4Loops(Tensor *output, std::vector<Tensor*> &inputs);
    void Execute5Loops(Tensor *output, std::vector<Tensor*> &inputs);
    void Execute6Loops(Tensor *output, std::vector<Tensor*> &inputs);
    void Execute7Loops(Tensor *output, std::vector<Tensor*> &inputs);
    void Execute8Loops(Tensor *output, std::vector<Tensor*> &inputs);
    void Execute9Loops(Tensor *output, std::vector<Tensor*> &inputs);
    void Execute10Loops(Tensor *output, std::vector<Tensor*> &inputs);
    void Execute11Loops(Tensor *output, std::vector<Tensor*> &inputs);
    void Execute12Loops(Tensor *output, std::vector<Tensor*> &inputs);
    void ExecuteArbitraryLoops(Tensor *output, std::vector<Tensor*> &inputs);

    inline double ComputeRHS(const std::vector<Tensor*> &inputs, const int *RESTRICT I);

    int *OutputLoopNums;
    int NumInVars;
    int **InputLoopNums;
};


inline double CPUInterpretedExecutor::ComputeRHS(const std::vector<Tensor*> &inputs, const int *RESTRICT I)
{
    double rhs_val = (*inputs[0])[ComputeRawIdx(*inputs[0], I, InputLoopNums[0])];
    for (unsigned int vari = 1; vari < NumInVars; ++vari)
    {
        rhs_val *= (*inputs[vari])[ComputeRawIdx(*inputs[vari], I, InputLoopNums[vari])];
    }
    return rhs_val;
}

}

#endif //ACROBATIC_CPUINTERPRETED_EXECUTOR_HPP