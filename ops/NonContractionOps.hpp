//Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at the Lawrence Livermore National Laboratory
//Written by Aaron Fisher (fisher47@llnl.gov). LLNL-CODE-738419.
//All rights reserved.
//This file is part of Acrotensor. For details, see https://github.com/LLNL/acrotensor.

#ifndef ACROBATIC_NON_CONTRACTION_OPS_HPP
#define ACROBATIC_NON_CONTRACTION_OPS_HPP

#include <string>
#include "Tensor.hpp"

namespace acro
{


class NonContractionOps
{
    public:
    virtual void BatchMatrixInverse(Tensor &Ainv, Tensor &A) = 0;
    virtual void BatchMatrixDet(Tensor &Adet, Tensor &A) = 0;
    virtual void BatchMatrixInvDet(Tensor &Ainv, Tensor &Adet, Tensor &A) = 0;
};

}


#endif //ACROBATIC_NON_CONTRACTION_OPS_HPP