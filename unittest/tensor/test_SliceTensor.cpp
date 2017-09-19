//Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at the Lawrence Livermore National Laboratory
//Written by Aaron Fisher (fisher47@llnl.gov). LLNL-CODE-738419.
//All rights reserved.
//This file is part of Acrotensor. For details, see https://github.com/LLNL/acrotensor.

#include "catch.hpp"
#include "AcroTensor.hpp"
#include <iostream>
using namespace acro;


TEST_CASE("Basic SliceTensor unit tests", "[SliceTensor]")
{
   SECTION("Prefixed sliced indexing works")
   {
      Tensor T(2, 3, 4);
      SliceTensor S0(T, 0);
      SliceTensor S1(T, 1);
      SliceTensor S00(T, 0, 0);
      SliceTensor S01(T, 0, 1);
      SliceTensor S02(T, 0, 2);
      SliceTensor S10(T, 1, 0);
      SliceTensor S11(T, 1, 1);
      SliceTensor S12(T, 1, 2);
      SliceTensor S20(T, 2, 0);
      SliceTensor S21(T, 2, 1);
      SliceTensor S22(T, 2, 2);

      REQUIRE(S0.GetRank() == 2);
      REQUIRE(S1.GetRank() == 2);
      REQUIRE(S00.GetRank() == 1);
      REQUIRE(S01.GetRank() == 1);
      REQUIRE(S10.GetRank() == 1);
      REQUIRE(S11.GetRank() == 1);

      REQUIRE(S0.GetSize() == 12);
      REQUIRE(S1.GetSize() == 12);
      REQUIRE(S00.GetSize() == 4);
      REQUIRE(S01.GetSize() == 4);
      REQUIRE(S10.GetSize() == 4);
      REQUIRE(S11.GetSize() == 4);

      REQUIRE(S0.GetDim(0) == 3);
      REQUIRE(S1.GetDim(0) == 3);
      REQUIRE(S0.GetDim(1) == 4);
      REQUIRE(S1.GetDim(1) == 4);
      REQUIRE(S00.GetDim(0) == 4);
      REQUIRE(S01.GetDim(0) == 4);
      REQUIRE(S10.GetDim(0) == 4);
      REQUIRE(S11.GetDim(0) == 4);

      REQUIRE(S0.GetStride(0) == 4);
      REQUIRE(S1.GetStride(0) == 4);
      REQUIRE(S0.GetStride(1) == 1);
      REQUIRE(S1.GetStride(1) == 1);
      REQUIRE(S00.GetStride(0) == 1);
      REQUIRE(S01.GetStride(0) == 1);
      REQUIRE(S10.GetStride(0) == 1);
      REQUIRE(S11.GetStride(0) == 1);    

      for (int idx = 0; idx < 24; ++idx)
      {
         T[idx] = idx;
      }

      for (int k = 0; k < 4; ++k)
      {
         for (int j = 0; j < 3; ++j)
         {
            REQUIRE(S0(j, k) == T(0, j, k));
            REQUIRE(S1(j, k) == T(1, j, k));
         }
         REQUIRE(S00(k) == T(0,0,k));
         REQUIRE(S01(k) == T(0,1,k));
         REQUIRE(S02(k) == T(0,2,k));
         REQUIRE(S10(k) == T(1,0,k));
         REQUIRE(S11(k) == T(1,1,k));
         REQUIRE(S12(k) == T(1,2,k));
         REQUIRE(S20(k) == T(2,0,k));
         REQUIRE(S21(k) == T(2,1,k));
         REQUIRE(S22(k) == T(2,2,k));
      }
   }

   SECTION("Prefixed sliced Set Method")
   {
      Tensor T(2, 3, 4);
      SliceTensor S0(T, 0);
      SliceTensor S1(T, 1);

      S0.Set(1.0);
      S1.Set(2.0);
      for (int j = 0; j < 3; ++ j)
      {
         for (int k = 0; k < 4; ++ k)
         {
            REQUIRE(T(0,j,k) == Approx(1.0));
            REQUIRE(T(1,j,k) == Approx(2.0));
            REQUIRE(S0(j,k) == Approx(1.0));
            REQUIRE(S1(j,k) == Approx(2.0));            
         }
      }
   }

   SECTION("Prefixed sliced tensor Set Method on GPU")
   {
      if (isCudaReady())
      {
         Tensor T(2, 10, 4, 4);
         T.SwitchToGPU();
         SliceTensor S0(T, 0);
         SliceTensor S1(T, 1);

         S0.Set(1.0);
         S1.Set(2.0);
         T.MoveFromGPU();
         for (int i = 0; i < 10; ++ i)
         {
            for (int j = 0; j < 4; ++ j)
            {
               for (int k = 0; k < 4; ++ k)
               {
                  REQUIRE(T(0,i,j,k) == Approx(1.0));
                  REQUIRE(T(1,i,j,k) == Approx(2.0));
                  REQUIRE(S0(i,j,k) == Approx(1.0));
                  REQUIRE(S1(i,j,k) == Approx(2.0));
               }
            }
         }
      }
   }

   SECTION("GPU Move Semantics")
   {
      if (isCudaReady())
      {
         Tensor T(2, 3);
         T.MapToGPU();
         double *t_cpu = T.GetData();
         double *t_gpu = T.GetDeviceData();
         CHECK(t_cpu != t_gpu);
         CHECK(T.GetCurrentData() == t_cpu);

         SliceTensor S(T, 0);
         double *s_cpu = S.GetData();
         double *s_gpu = S.GetDeviceData();
         CHECK(s_cpu != s_gpu);
         CHECK(S.GetCurrentData() == s_cpu);
         CHECK(S.IsMappedToGPU());

         S.MoveToGPU();
         CHECK(T.IsOnGPU());
         CHECK(S.IsOnGPU());
         CHECK(T.GetCurrentData() == t_gpu);
         CHECK(S.GetCurrentData() == s_gpu);

         S.Set(2.0);
         T.MoveFromGPU();
         CHECK(!T.IsOnGPU());
         CHECK(!S.IsOnGPU());
         CHECK(T.GetCurrentData() == t_cpu);
         CHECK(S.GetCurrentData() == s_cpu);
         for (int i = 0; i < S.GetSize(); ++i)
         {
            CHECK(S[i] == Approx(2.0));
         }
      }
   }
}