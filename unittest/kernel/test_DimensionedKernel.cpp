//Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at the Lawrence Livermore National Laboratory
//Written by Aaron Fisher (fisher47@llnl.gov). LLNL-CODE-738419.
//All rights reserved.
//This file is part of Acrotensor. For details, see https://github.com/LLNL/acrotensor.

#include "catch.hpp"
#include "DimensionedKernel.hpp"

using namespace acro;


TEST_CASE("DimensionedKernel operations", "[DimensionedKernel]")
{
   Tensor T1out_3(3), T2out_3_3(3, 3), T1_3(3), T1_2(2), T2_3_3(3,3);


   SECTION("A_i=B_iC_i")
   {
      TensorKernel Kernel("A_i=B_iC_i");
      std::vector<Tensor*> inputs;
      inputs.push_back(&T1_3);
      inputs.push_back(&T1_3);
      DimensionedKernel DKernel(&Kernel, &T1out_3, inputs);
      REQUIRE(DKernel.GetFlatIdxSize() == 3);
      REQUIRE(DKernel.GetOutIdxSize() == 3);
      REQUIRE(DKernel.GetContIdxSize() == 1);
      REQUIRE(DKernel.GetOutputStorageReqForInnerLoops(1) == 3);
      REQUIRE(DKernel.GetInputStorageReqForInnerLoops(1) == 6);
   }

   SECTION("A_i=B_s_iC_i_s")
   {
      TensorKernel Kernel("A_i=B_s_iC_i_s");
      std::vector<Tensor*> inputs;
      inputs.push_back(&T2_3_3);
      inputs.push_back(&T2_3_3);
      DimensionedKernel DKernel(&Kernel, &T1out_3, inputs);
      REQUIRE(DKernel.GetFlatIdxSize() == 9);
      REQUIRE(DKernel.GetOutIdxSize() == 3);
      REQUIRE(DKernel.GetContIdxSize() == 3);
      REQUIRE(DKernel.GetOutputStorageReqForInnerLoops(1) == 1);
      REQUIRE(DKernel.GetOutputStorageReqForInnerLoops(2) == 3);
      REQUIRE(DKernel.GetInputStorageReqForInnerLoops(1) == 6);
      REQUIRE(DKernel.GetInputStorageReqForInnerLoops(2) == 18);
   }

   SECTION("A_i=B_s_iC_i")
   {
      TensorKernel Kernel("A_i=B_s_iC_i");
      std::vector<Tensor*> inputs;
      inputs.push_back(&T2_3_3);
      inputs.push_back(&T1_3);
      DimensionedKernel DKernel(&Kernel, &T1out_3, inputs);
      REQUIRE(DKernel.GetFlatIdxSize() == 9);
      REQUIRE(DKernel.GetOutIdxSize() == 3);
      REQUIRE(DKernel.GetContIdxSize() == 3);
      REQUIRE(DKernel.GetOutputStorageReqForInnerLoops(1) == 1);
      REQUIRE(DKernel.GetOutputStorageReqForInnerLoops(2) == 3);
      REQUIRE(DKernel.GetInputStorageReqForInnerLoops(1) == 4);
      REQUIRE(DKernel.GetInputStorageReqForInnerLoops(2) == 12);            
   }

   SECTION("S_e_i1_i2_i3_j1_j2_j3=B_i1_j1_k1_m_nB_i2_j2_k2_m_nB_i3_j3_k3_m_nD_e_k1_k2_k3_m_n")
   {
      std::string kernel_str = "S_e_i1_i2_i3_j1_j2_j3 =B_i1_j1_k1_m_nB_i2_j2_k2_m_nB_i3_j3_k3_m_n D_e_k1_k2_k3_m_n";
      TensorKernel Kernel(kernel_str);
      Tensor S(10, 5, 5, 5, 5, 5, 5);
      Tensor Btilde1(5, 5, 5, 3, 3);
      Tensor Btilde2(5, 5, 5, 3, 3);
      Tensor Btilde3(5, 5, 5, 3, 3);
      Tensor D(10, 5, 5, 5, 3, 3);
      std::vector<Tensor*> inputs = {&Btilde1, &Btilde2, &Btilde3, &D};
      DimensionedKernel DKernel(&Kernel, &S, inputs);
      REQUIRE(DKernel.GetFlatIdxSize() == 175781250);
      REQUIRE(DKernel.GetOutIdxSize() == 156250);
      REQUIRE(DKernel.GetContIdxSize() == 1125);        
   }
}