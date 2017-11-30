//Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at the Lawrence Livermore National Laboratory
//Written by Aaron Fisher (fisher47@llnl.gov). LLNL-CODE-738419.
//All rights reserved.
//This file is part of Acrotensor. For details, see https://github.com/LLNL/acrotensor.

#ifndef ACROBATIC_NATIVE_CPU_OPS_HPP
#define ACROBATIC_NATIVE_CPU_OPS_HPP

#include "NonContractionOps.hpp"
#include "Tensor.hpp"

namespace acro
{


//Internal CPU operations on tensors that are exposed properly by the kernel executors.
//Use of this class directly is not recommended.
class NativeCPUOps : public NonContractionOps
{
    public:
    void BatchMatrixInverse(Tensor &out, Tensor &in);

    private:
    inline void Inv1x1(double *out, double *in);
    inline void Inv2x2(double *out, double *in);
    inline void Inv3x3(double *out, double *in);
};


inline void NativeCPUOps::Inv1x1(double *out, double *in)
{
    out[0] = 1.0 / in[0];
}


inline void NativeCPUOps::Inv2x2(double *out, double *in)
{
    double invdet = 1.0 / (in[0]*in[3] - in[1]*in[2]);
    out[0] = invdet*in[3];
    out[1] = -invdet*in[1];
    out[2] = -invdet*in[2];
    out[3] = invdet*in[0];

}


inline void NativeCPUOps::Inv3x3(double *out, double *in)
{
    double invdet = 1.0 / (in[0]*in[4]*in[8] + in[1]*in[5]*in[6] + in[2]*in[3]*in[7] 
                         - in[6]*in[4]*in[2] - in[7]*in[5]*in[0] - in[8]*in[3]*in[1]);
    out[0] = invdet*(in[4]*in[8] - in[5]*in[7]);
    out[1] = invdet*(in[5]*in[6] - in[3]*in[8]);
    out[2] = invdet*(in[3]*in[7] - in[4]*in[6]);
    out[3] = invdet*(in[2]*in[7] - in[1]*in[8]);
    out[4] = invdet*(in[0]*in[8] - in[2]*in[6]);
    out[5] = invdet*(in[1]*in[6] - in[0]*in[7]);
    out[6] = invdet*(in[1]*in[5] - in[2]*in[4]);
    out[7] = invdet*(in[2]*in[3] - in[0]*in[5]);
    out[8] = invdet*(in[0]*in[4] - in[1]*in[3]);
}

}


#endif //ACROBATIC_NATIVE_CPU_OPS_HPP