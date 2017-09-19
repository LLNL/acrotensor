//Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at the Lawrence Livermore National Laboratory
//Written by Aaron Fisher (fisher47@llnl.gov). LLNL-CODE-738419.
//All rights reserved.
//This file is part of Acrotensor. For details, see https://github.com/LLNL/acrotensor.

#ifndef ACROBATIC_INDEX_CACHED_EXECUTOR_HPP
#define ACROBATIC_INDEX_CACHED_EXECUTOR_HPP

#include "KernelExecutor.hpp"
#include <map>


namespace acro
{

class IndexCachedExecutor : public KernelExecutor
{
    public:
    IndexCachedExecutor(std::string &kernelstr);
    ~IndexCachedExecutor();

    virtual void ExecuteKernel(Tensor *output, std::vector<Tensor*> &inputs);
    virtual std::string GetImplementation(Tensor *out, std::vector<Tensor*> &inputs);

    private:
    void AddIndices();
    void Build1LoopInd(int *raw_indices);
    void Build2LoopInd(int *raw_indices);
    void Build3LoopInd(int *raw_indices);
    void Build4LoopInd(int *raw_indices);
    void Build5LoopInd(int *raw_indices);
    void Build6LoopInd(int *raw_indices);
    void Build7LoopInd(int *raw_indices);
    void Build8LoopInd(int *raw_indices);
    void BuildArbitraryLoopInd(int *raw_indices);

    void ExecICFlatLoop(double *dout, double **din, int *raw_indices, int outidx_size, int contidx_size, int numinvars);
    void ExecICFlatLoop1(double *dout, double **din, int *raw_indices, int outidx_size, int contidx_size);
    void ExecICFlatLoop2(double *dout, double **din, int *raw_indices, int outidx_size, int contidx_size);
    void ExecICFlatLoop3(double *dout, double **din, int *raw_indices, int outidx_size, int contidx_size);
    void ExecICFlatLoop4(double *dout, double **din, int *raw_indices, int outidx_size, int contidx_size);
    void ExecICFlatLoop5(double *dout, double **din, int *raw_indices, int outidx_size, int contidx_size);
    void ExecICFlatLoop6(double *dout, double **din, int *raw_indices, int outidx_size, int contidx_size);
    void ExecICFlatLoop7(double *dout, double **din, int *raw_indices, int outidx_size, int contidx_size);
    void ExecICFlatLoop8(double *dout, double **din, int *raw_indices, int outidx_size, int contidx_size);
    void ExecICFlatLoopAny(double *dout, double **din, int *raw_indices, int outidx_size, int contidx_size, int numinvars);

    std::map<std::vector<int>, int* > RawIndexMap;
};

}

#endif //ACROBATIC_INDEX_CACHED_EXECUTOR_HPP