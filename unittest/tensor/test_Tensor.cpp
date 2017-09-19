//Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at the Lawrence Livermore National Laboratory
//Written by Aaron Fisher (fisher47@llnl.gov). LLNL-CODE-738419.
//All rights reserved.
//This file is part of Acrotensor. For details, see https://github.com/LLNL/acrotensor.

#include "catch.hpp"
#include "AcroTensor.hpp"
#include <iostream>
using namespace acro;


TEST_CASE("Basic Tensor unit tests", "[Tensor]")
{
   Tensor T1a(10), T2a(5, 6), T3a(4, 3, 1), T4a(5, 1, 3, 2), T5a(1, 1, 1, 2, 3);
   std::vector<int> dims = {1, 4, 3, 2};
   Tensor Tdims(dims);

   SECTION("Assert dims > 0")
   {
      REQUIRE_NOTHROW(new Tensor(1));
      REQUIRE_THROWS(new Tensor(-1));
      REQUIRE_THROWS(new Tensor(10, 0));
      REQUIRE_THROWS(new Tensor(-1, 10));
      REQUIRE_THROWS(new Tensor(10, -1, 10));

      std::vector<int> empty_dims;
      REQUIRE_THROWS(new Tensor(empty_dims));

      std::vector<int> bogus_dims = {1, 2, 3, 4, 5, 6, 7, -100};
      REQUIRE_THROWS(new Tensor(bogus_dims));

      REQUIRE_NOTHROW(new Tensor(dims));
   }

   SECTION("Dimensions set properly")
   {
      SECTION("Ranks")
      {
         REQUIRE(T1a.GetRank() == 1);
         REQUIRE(T2a.GetRank() == 2);
         REQUIRE(T3a.GetRank() == 3);
         REQUIRE(T4a.GetRank() == 4);
         REQUIRE(T5a.GetRank() == 5);
         REQUIRE(Tdims.GetRank() == 4);
      }
      
      SECTION("Dims")
      {
         REQUIRE(T1a.GetDim(0) == 10);
         REQUIRE(T2a.GetDim(0) == 5);  
         REQUIRE(T2a.GetDim(1) == 6);
         REQUIRE(T3a.GetDim(0) == 4); 
         REQUIRE(T3a.GetDim(1) == 3); 
         REQUIRE(T3a.GetDim(2) == 1);
         REQUIRE(T4a.GetDim(0) == 5); 
         REQUIRE(T4a.GetDim(1) == 1); 
         REQUIRE(T4a.GetDim(2) == 3); 
         REQUIRE(T4a.GetDim(3) == 2);
         REQUIRE(T5a.GetDim(0) == 1); 
         REQUIRE(T5a.GetDim(1) == 1); 
         REQUIRE(T5a.GetDim(2) == 1); 
         REQUIRE(T5a.GetDim(3) == 2); 
         REQUIRE(T5a.GetDim(4) == 3);
         REQUIRE(Tdims.GetDim(0) == 1); 
         REQUIRE(Tdims.GetDim(1) == 4); 
         REQUIRE(Tdims.GetDim(2) == 3); 
         REQUIRE(Tdims.GetDim(3) == 2);
      }

      SECTION("Sizes")
      {
         REQUIRE(T1a.GetSize() == 10);
         REQUIRE(T2a.GetSize() == 30);
         REQUIRE(T3a.GetSize() == 12);
         REQUIRE(T4a.GetSize() == 30);
         REQUIRE(T5a.GetSize() == 6);
         REQUIRE(Tdims.GetSize() == 24);
      }
   }

   SECTION("Index Space Covered")
   {
      std::vector<bool> covered(T4a.GetSize(), false);
      for (int i = 0; i < T4a.GetDim(0); ++i)
      {
         for (int j = 0; j < T4a.GetDim(1); ++j)
         {
            for (int k = 0; k < T4a.GetDim(2); ++k)
            {
               for (int l = 0; l < T4a.GetDim(3); ++l)
               {
                  int raw_index = T4a.GetRawIndex(i,j,k,l);
                  REQUIRE(raw_index >= 0);
                  REQUIRE(raw_index < T4a.GetSize());
                  covered[raw_index] = true;
               }
            }
         }
      }

      for (int raw_index = 0; raw_index < T4a.GetSize(); ++raw_index)
      {
         REQUIRE(covered[raw_index]);
      }
   }

   SECTION("Accessing the Data")
   {
      T1a.Set(0.0);
      T2a.Set(0.0);
      T3a.Set(0.0);
      T4a.Set(0.0);
      T5a.Set(0.0);

      T1a(3) = 4.0;
      REQUIRE(T1a(3) == Approx(4.0));
      REQUIRE(T1a[3] == Approx(4.0));

      T2a(2,1) = 3.0;
      REQUIRE(T2a(2,1) == Approx(3.0));
      REQUIRE(T2a[T2a.GetRawIndex(2,1)] == Approx(3.0));
   }

   SECTION("Reshaping")
   {
      Tensor T(6);
      for (int flatidx = 0; flatidx < T.GetSize(); ++flatidx)
      {
         T[flatidx] = double(flatidx);
      }

      T.Reshape(3, 2);
      REQUIRE_NOTHROW(T(1,0));
      REQUIRE(T(1,0) == Approx(2.0));
      REQUIRE_THROWS(T.Reshape(3,4));
   }

   SECTION("Tensor on existing data")
   {
      double data[6] = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0};
      Tensor T(2, 3, data);
      REQUIRE(T(1,1) == Approx(4.0));
   }

   SECTION("Defered initialization")
   {
      Tensor T;
      REQUIRE(!T.IsInitialized());
      REQUIRE_NOTHROW(T.Init(2, 2));
      REQUIRE(T.IsInitialized());
      REQUIRE_NOTHROW(T(0,0) = 2.0);
      REQUIRE(T(0,0) == 2.0);
   }

   SECTION("Basic CUDA tests")
   {
      if (isCudaReady())
      {         
         Tensor T(2);
         T.Set(3.0);
         REQUIRE(T(0) == Approx(3.0));
         REQUIRE(T(1) == Approx(3.0));

         T.MapToGPU();
         REQUIRE(T.IsMappedToGPU());
         REQUIRE(!T.IsOnGPU());

         T.SwitchToGPU();
         REQUIRE(T.IsOnGPU());

         T.Set(9.0);
         REQUIRE(T(0) == Approx(3.0));    //Not moved back from GPU yet
         REQUIRE(T(1) == Approx(3.0));

         T.MoveFromGPU();
         REQUIRE(!T.IsOnGPU());
         REQUIRE(T(0) == Approx(9.0));
         REQUIRE(T(1) == Approx(9.0));
      }
      else
      {
         std::cout << "No GPU found.  Ignoring CUDA tests." << std::endl;
      }            
   }
}