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

    inline double ComputeRHS(const int *RESTRICT I);
    inline int ComputeRawIdx(const int *RESTRICT I, const int *loop_nums, const int *var_stride, int rank);

    int NumInVars;
    int NumLoops;
    std::vector<int> N;

    int OutputRank;
    double *OutputVar;
    int *OutputLoopNums;
    int *OutputStrides;

    int *InputRanks;
    double **InputVars;
    int **InputLoopNums;
    int **InputStrides;
};


inline double CPUInterpretedExecutor::ComputeRHS(const int *RESTRICT I)
{
    double rhs_val = InputVars[0][ComputeRawIdx(I, InputLoopNums[0], InputStrides[0], InputRanks[0])];
    for (int vari = 1; vari < NumInVars; ++vari)
    {
        rhs_val *= InputVars[vari][ComputeRawIdx(I, InputLoopNums[vari], InputStrides[vari], InputRanks[vari])];
    }
    return rhs_val;
}


inline int CPUInterpretedExecutor::ComputeRawIdx(const int *RESTRICT I, const int *loop_nums, const int *var_stride, int rank)
{   
    int raw_idx = I[loop_nums[0]]*var_stride[0];
    for (int d = 1; d < rank; ++d)
    {
        raw_idx += I[loop_nums[d]]*var_stride[d];
    }
    return raw_idx;
}

}

#endif //ACROBATIC_CPUINTERPRETED_EXECUTOR_HPP