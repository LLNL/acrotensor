//Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at the Lawrence Livermore National Laboratory
//Written by Aaron Fisher (fisher47@llnl.gov). LLNL-CODE-738419.
//All rights reserved.
//This file is part of Acrotensor. For details, see https://github.com/LLNL/acrotensor.

#include "catch.hpp"
#include "TensorKernel.hpp"

using namespace acro;

std::string reconstruct_kernel_str(TensorKernel &Kernel);

TEST_CASE("TensorKernel operations", "[TensorKernel]")
{
   SECTION("Assert Parsable String")
   {
      REQUIRE_THROWS(new TensorKernel("Blah"));
      REQUIRE_THROWS(new TensorKernel("Blah="));
      REQUIRE_THROWS(new TensorKernel("Blah=Alah"));
      REQUIRE_THROWS(new TensorKernel("BLA1h=Alah"));
      REQUIRE_NOTHROW(new TensorKernel("B_lah=A_lah"));
      REQUIRE_THROWS(new TensorKernel("B_lah=A_lah_"));
      REQUIRE_THROWS(new TensorKernel("B_lah_=A_lah"));
      REQUIRE_THROWS(new TensorKernel("a_lah_=A_lah"));
   }

   SECTION("Can Reconstruct Various Kernels")
   {
      REQUIRE(reconstruct_kernel_str(*(new TensorKernel("BA_i=A_j"))) == "BA_i=A_j");
      REQUIRE(reconstruct_kernel_str(*(new TensorKernel("B1_i=CB_jBr_j"))) == "B1_i=CB_jBr_j");
   }

   SECTION("A_i=B_iC_i")
   {
      TensorKernel Kernel("A_i=B_iC_i");

      SECTION("Basic Parsing")
      {
         REQUIRE(reconstruct_kernel_str(Kernel) == "A_i=B_iC_i");
         REQUIRE(Kernel.AllIndexNames.size() == 1);
         REQUIRE(Kernel.AllIndexNames[0] == "i");
         REQUIRE(Kernel.ContractionIndexNames.size() == 0);

         REQUIRE(Kernel.GetNumIndices() == 1);
         REQUIRE(Kernel.GetNumContractionIndices() == 0);
         REQUIRE(Kernel.GetNumVars() == 3);
         REQUIRE(Kernel.GetNumInputVars() == 2);
         REQUIRE(Kernel.GetVarDimLoopNum(0, 0) == 0);
         REQUIRE(Kernel.GetVarDimLoopNum(1, 0) == 0);
         REQUIRE(Kernel.IsVarDependentOnLoop(-1, 0));
         REQUIRE(Kernel.IsVarDependentOnLoop(0, 0));
         REQUIRE(Kernel.IsVarDependentOnLoop(1, 0));
      }
   }

   SECTION("A_i=B_s_iC_i_s")
   {
      TensorKernel Kernel("A_i=B_s_iC_i_s");
      SECTION("Basic Parsing")
      {
         REQUIRE(reconstruct_kernel_str(Kernel) == "A_i=B_s_iC_i_s");
         REQUIRE(Kernel.AllIndexNames.size() == 2);
         REQUIRE(Kernel.AllIndexNames[0] == "i");
         REQUIRE(Kernel.AllIndexNames[1] == "s");
         REQUIRE(Kernel.ContractionIndexNames.size() == 1);
         REQUIRE(Kernel.ContractionIndexNames[0] == "s");

         REQUIRE(Kernel.GetNumIndices() == 2);
         REQUIRE(Kernel.GetNumContractionIndices() == 1);
         REQUIRE(Kernel.GetNumVars() == 3);
         REQUIRE(Kernel.GetNumInputVars() == 2);
         REQUIRE(Kernel.GetVarDimLoopNum(0, 0) == 1);
         REQUIRE(Kernel.GetVarDimLoopNum(0, 1) == 0);
         REQUIRE(Kernel.GetVarDimLoopNum(1, 0) == 0);
         REQUIRE(Kernel.GetVarDimLoopNum(1, 1) == 1);
         REQUIRE(Kernel.IsVarDependentOnLoop(-1, 0));
         REQUIRE(!Kernel.IsVarDependentOnLoop(-1, 1));
         REQUIRE(Kernel.IsVarDependentOnLoop(0, 0));
         REQUIRE(Kernel.IsVarDependentOnLoop(0, 1));
         REQUIRE(Kernel.IsVarDependentOnLoop(1, 0));
         REQUIRE(Kernel.IsVarDependentOnLoop(1, 1));
      }
   }

   SECTION("A_i=B_s_iC_i")
   {
      TensorKernel Kernel("A_i=B_s_iC_i");
      SECTION("Basic Parsing")
      {
         REQUIRE(reconstruct_kernel_str(Kernel) == "A_i=B_s_iC_i");
         REQUIRE(Kernel.AllIndexNames.size() == 2);
         REQUIRE(Kernel.AllIndexNames[0] == "i");
         REQUIRE(Kernel.AllIndexNames[1] == "s");
         REQUIRE(Kernel.ContractionIndexNames.size() == 1);
         REQUIRE(Kernel.ContractionIndexNames[0] == "s");

         REQUIRE(Kernel.GetNumIndices() == 2);
         REQUIRE(Kernel.GetNumContractionIndices() == 1);
         REQUIRE(Kernel.GetNumVars() == 3);
         REQUIRE(Kernel.GetNumInputVars() == 2);
         REQUIRE(Kernel.GetVarDimLoopNum(0, 0) == 1);
         REQUIRE(Kernel.GetVarDimLoopNum(0, 1) == 0);
         REQUIRE(Kernel.GetVarDimLoopNum(1, 0) == 0);
         REQUIRE(Kernel.IsVarDependentOnLoop(-1, 0));
         REQUIRE(!Kernel.IsVarDependentOnLoop(-1, 1));
         REQUIRE(Kernel.IsVarDependentOnLoop(0, 0));
         REQUIRE(Kernel.IsVarDependentOnLoop(0, 1));
         REQUIRE(Kernel.IsVarDependentOnLoop(1, 0));
         REQUIRE(!Kernel.IsVarDependentOnLoop(1, 1));   
      }      
   }

   SECTION("A_i1_i2=B_i1_i2_sum1")
   {
      TensorKernel Kernel("A_i1_i2=B_i1_i2_sum1");
      SECTION("Basic Parsing")
      {
         REQUIRE(reconstruct_kernel_str(Kernel) == "A_i1_i2=B_i1_i2_sum1");
         REQUIRE(Kernel.AllIndexNames.size() == 3);
         REQUIRE(Kernel.AllIndexNames[0] == "i1");
         REQUIRE(Kernel.AllIndexNames[1] == "i2");
         REQUIRE(Kernel.AllIndexNames[2] == "sum1");
         REQUIRE(Kernel.ContractionIndexNames.size() == 1);
         REQUIRE(Kernel.ContractionIndexNames[0] == "sum1");
      }
   }

   SECTION("S_e_i1_i2_i3_j1_j2_j3=B_i1_j1_k1_m_nB_i2_j2_k2_m_nB_i3_j3_k3_m_nD_e_k1_k2_k3_m_n")
   {
      std::string kernel_str = "S_e_i1_i2_i3_j1_j2_j3 =B_i1_j1_k1_m_nB_i2_j2_k2_m_nB_i3_j3_k3_m_n D_e_k1_k2_k3_m_n";
      TensorKernel Kernel(kernel_str);
      SECTION("Basic Parsing")
      {
         REQUIRE(Kernel.AllIndexNames.size() == 12);
         REQUIRE(Kernel.AllIndexNames[0] == "e");
         REQUIRE(Kernel.AllIndexNames[1] == "i1");
         REQUIRE(Kernel.AllIndexNames[2] == "i2");
         REQUIRE(Kernel.AllIndexNames[3] == "i3");
         REQUIRE(Kernel.AllIndexNames[4] == "j1");
         REQUIRE(Kernel.AllIndexNames[5] == "j2");
         REQUIRE(Kernel.AllIndexNames[6] == "j3");
         REQUIRE(Kernel.AllIndexNames[7] == "k1");
         REQUIRE(Kernel.AllIndexNames[8] == "m");
         REQUIRE(Kernel.AllIndexNames[9] == "n");
         REQUIRE(Kernel.AllIndexNames[10] == "k2");
         REQUIRE(Kernel.AllIndexNames[11] == "k3");

         REQUIRE(Kernel.ContractionIndexNames.size() == 5);
         REQUIRE(Kernel.ContractionIndexNames[0] == "k1");
         REQUIRE(Kernel.ContractionIndexNames[1] == "m");
         REQUIRE(Kernel.ContractionIndexNames[2] == "n");
         REQUIRE(Kernel.ContractionIndexNames[3] == "k2");
         REQUIRE(Kernel.ContractionIndexNames[4] == "k3");

         REQUIRE(Kernel.GetNumIndices() == 12);
         REQUIRE(Kernel.GetNumContractionIndices() == 5);
         REQUIRE(Kernel.GetNumVars() == 5);
         REQUIRE(Kernel.GetNumInputVars() == 4);     
      }            
   }
}

std::string reconstruct_kernel_str(TensorKernel &Kernel)
{
   std::string str;
   str += Kernel.OutputVar.Name;
   for (int d = 0; d < Kernel.OutputVar.IndexNames.size(); ++d)
   {
      str += "_";
      str += Kernel.OutputVar.IndexNames[d];
   }

   str += Kernel.EqOperator;

   for (int vari = 0; vari < Kernel.InputVars.size(); ++vari)
   {
      str += Kernel.InputVars[vari].Name;
      for (int d = 0; d < Kernel.InputVars[vari].IndexNames.size(); ++d)
      {
         str += "_";
         str += Kernel.InputVars[vari].IndexNames[d];
      }
   }
   return str;
}
