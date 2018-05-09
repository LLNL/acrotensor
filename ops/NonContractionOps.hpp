//Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at the Lawrence Livermore National Laboratory
//Written by Aaron Fisher (fisher47@llnl.gov). LLNL-CODE-738419.
//All rights reserved.
//This file is part of Acrotensor. For details, see https://github.com/LLNL/acrotensor.

#ifndef ACROBATIC_NON_CONTRACTION_OPS_HPP
#define ACROBATIC_NON_CONTRACTION_OPS_HPP

#include <string>
#include "Tensor.hpp"
#include "IndexMapping.hpp"

namespace acro
{


class NonContractionOps
{
    public:
    //Batched 1x1, 2x2, and 3x3 matrix inverses and determinents
    //The last 2 indices are for the matrices and the rests are batched over
    virtual void BatchMatrixInverse(Tensor &Ainv, Tensor &A) = 0;
    virtual void BatchMatrixDet(Tensor &Adet, Tensor &A) = 0;
    virtual void BatchMatrixInvDet(Tensor &Ainv, Tensor &Adet, Tensor &A) = 0;

    //Aout[i] = Ain[I[i]]
    virtual void FlatIndexedScatter(Tensor &Aout, Tensor &Ain, IndexMapping &M) = 0;

    //Aout[:] = 0.0
    //Aout[I[i]] += Ain[i]
    virtual void FlatIndexedSumGather(Tensor &Aout, Tensor &Ain, IndexMapping &M) = 0;
};

}


#endif //ACROBATIC_NON_CONTRACTION_OPS_HPP