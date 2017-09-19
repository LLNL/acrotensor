//Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at the Lawrence Livermore National Laboratory
//Written by Aaron Fisher (fisher47@llnl.gov). LLNL-CODE-738419.
//All rights reserved.
//This file is part of Acrotensor. For details, see https://github.com/LLNL/acrotensor.

#include "catch.hpp"
#include "AcroTensor.hpp"
#include <iostream>

using namespace acro;

void test_suite_on_cpu_engine(TensorEngine &TE);
void test_suite_on_gpu_engine(TensorEngine &TE);

TEST_CASE("TensorEngine", "[TensorEngine]")
{
   SECTION("CPUInterpretedExecutor")
   {
      TensorEngine TE("CPUInterpreted");
      test_suite_on_cpu_engine(TE);
   }


   SECTION("IndexCachedExecutor")
   {
      TensorEngine TE("IndexCached");
      test_suite_on_cpu_engine(TE);
   }

   if (isCudaReady()) 
   {
      SECTION("OneOutPerThreadExecutor")
      {
         TensorEngine TE("OneOutPerThread");
         test_suite_on_gpu_engine(TE);
      }


      SECTION("SMChunkPerBlockExecutor")
      {
         TensorEngine TE("SMChunkPerBlock");
         test_suite_on_gpu_engine(TE);
      }
   } 
}


void test_suite_on_cpu_engine(TensorEngine &TE)
{
   Tensor T1out_3(3), T2out_3_3(3, 3), T1_3(3), T1_2(2), T2_3_3(3,3);

   SECTION("Assert Compatible Dimensions")
   {
      REQUIRE_NOTHROW(TE["A_i=B_iC_i"](T1out_3, T1_3, T1_3));
      REQUIRE_THROWS(TE["A_i=B_iC_i"](T1out_3, T1_3, T1_2));
      REQUIRE_NOTHROW(TE["A_i=B_i_jC_i"](T1out_3, T2_3_3, T1_3));
      REQUIRE_THROWS(TE["A_i=B_i_jC_i"](T1out_3, T2_3_3, T1_2));
   }

   SECTION("Basic computations")
   {
      T2_3_3.Set(0.0);
      T1_3.Set(0.0);

      //Simple as it gets
      T1_3(0) = 1.0; T1_3(1) = 2.0; T1_3(2) = 3.0;
      TE["A_i=B_i"](T1out_3, T1_3);
      for (int d = 0; d < 3; ++d) {
         REQUIRE(T1out_3(d) == Approx(T1_3(d)));
      }
  
      //Get the diagonal
      T2_3_3(0,0) = 4.0; T2_3_3(1,1) = 5.0; T2_3_3(2,2) = 6.0;
      TE["A_i=B_i_i"](T1out_3, T2_3_3);
      for (int d = 0; d < 3; ++d) {
         REQUIRE(T1out_3(d) == Approx(T2_3_3(d,d)));
      }

      //Contract the diagonal and broadcast it to all of A
      TE["A_i=B_j_j"](T1out_3, T2_3_3);
      for (int d = 0; d < 3; ++d) {
         REQUIRE(T1out_3(d) == Approx(T2_3_3(0,0) + T2_3_3(1,1) + T2_3_3(2,2)));
      }

      //Matvec
      T2_3_3.Set(1.0);
      T1_3.Set(1.0);
      TE["A_i = C_i_j B_j"](T1out_3, T2_3_3, T1_3);
      for (int d = 0; d < 3; ++d) {
         REQUIRE(T1out_3(d) == Approx(3.0));
      }

      //Try out a transpose
      for (int flatidx = 0; flatidx < T2_3_3.GetSize(); ++flatidx)
      {
         T2_3_3[flatidx] = double(flatidx);
      }
      TE["A_i_j=B_j_i"](T2out_3_3, T2_3_3);
      for (int i = 0; i < 3; ++i)
      {
         for (int j = 0; j < 3; ++j)
         {
            REQUIRE(T2out_3_3(i,j) == Approx(T2_3_3(j,i)));
         }
      }

      //Try a 5D contraction with singletons
      Tensor T5(1, 2, 3, 1, 5);
      T5.Set(1.0);
      TE["A_i=B_j_k_l_m_n"](T1out_3, T5);
      REQUIRE(T1out_3(0) == Approx(30.0));
   }

   SECTION("Tensor Engine Clear")
   {
      REQUIRE_NOTHROW(TE.Clear());
   }
}


void test_suite_on_gpu_engine(TensorEngine &TE)
{
   SECTION("GPU Computations")
   {
      SECTION("Autotransfer to GPU")
      {
         Tensor A(3);
         Tensor B(3);

         B(0) = 1.0; B(1) = 2.0; B(2) = 3.0;
         TE["A_i=B_i"](A, B);
         A.MoveFromGPU();
         if (A(0) != Approx(B(0)))
            std::cout << TE["A_i=B_i"].GetImplementation(A, B) << std::endl;
         REQUIRE(A(0) == Approx(B(0)));
         REQUIRE(A(1) == Approx(B(1)));
         REQUIRE(A(2) == Approx(B(2)));
      }

      SECTION("Set External Cuda Context")
      {
         CUcontext ctx;
         cuCtxCreate(&ctx, 0, 0);
         setCudaContext(ctx);

         Tensor A(3);
         Tensor B(3);

         B(0) = 1.0; B(1) = 2.0; B(2) = 3.0;
         TE["A_i=B_i"](A, B);
         A.MoveFromGPU();
         CHECK(A(0) == Approx(B(0)));
         CHECK(A(1) == Approx(B(1)));
         CHECK(A(2) == Approx(B(2)));
      }      

      SECTION("Computation with externally defined Data")
      {
         double a_data[125];
         for (int i = 0; i < 125; ++i)
         {
            a_data[i] = double(i);
         }
         double *a_device_data;
         acroCudaErrorCheck(cudaMalloc((void**)&a_device_data, 125*sizeof(double)));
         Tensor A(125, a_data, a_device_data, false);
         Tensor B(125);
         TE["B_i=A_i"](B, A);
         A.MoveFromGPU();
         B.MoveFromGPU();
         for (int i = 0; i < 125; ++i)
         {
            REQUIRE(A(i) == B(i));
         }
         acroCudaErrorCheck(cudaFree(a_device_data));
      }

      SECTION("Two Step Computation")
      {
         Tensor A(3);
         Tensor B(3);
         Tensor C(3,3);

         A.MapToGPU();
         B.MapToGPU();
         C.MapToGPU();

         B(0) = 1.0; B(1) = 2.0; B(2) = 3.0;
         A.SwitchToGPU();
         B.MoveToGPU();
         TE["A_i=B_i"](A, B);
         A.MoveFromGPU();
         CHECK(A(0) == Approx(B(0)));
         CHECK(A(1) == Approx(B(1)));
         CHECK(A(2) == Approx(B(2)));

         A.SwitchToGPU();
         B.Set(1.0);    //B is still on the GPU
         C.Set(1.0);
         C.MoveToGPU();
         TE["A_i = C_i_j B_j"](A, C, B);
         A.MoveFromGPU();
         if (A(0) != Approx(3.0))
            std::cout << TE["A_i = C_i_j B_j"].GetImplementation(A, C, B) << std::endl;
         CHECK(A(0) == Approx(3.0));
         CHECK(A(1) == Approx(3.0));
         CHECK(A(2) == Approx(3.0));
      }

      SECTION("Transpose")
      {
         Tensor A(4,4);
         Tensor B(4,4);

         for (int flatidx = 0; flatidx < B.GetSize(); ++flatidx)
         {
            B[flatidx] = double(flatidx);
         }

         B.MapToGPU();
         B.MoveToGPU();
         A.MapToGPU();
         A.SwitchToGPU();
         TE["A_i_j=B_j_i"](A, B);
         A.MoveFromGPU();
         B.SwitchFromGPU();
         for (int i = 0; i < 4; ++i)
         {
            for (int j = 0; j < 4; ++j)
            {
               CHECK(A(i,j) == Approx(B(j,i)));
            }
         }
      }

      SECTION("FE Type Contractions")
      {
         SECTION("2D Mass Matrix")
         {
            Tensor M(10,4,4,4,4);
            Tensor D(10,4,4);
            Tensor T(10,4,4);
            Tensor W(4,4);
            Tensor B(4,4);

            M.MapToGPU();
            D.MapToGPU();
            T.MapToGPU();
            W.MapToGPU();
            B.MapToGPU();            

            for (int flatidx = 0; flatidx < T.GetSize(); ++flatidx)
            {
               T[flatidx] = double(flatidx);
            }

            for (int flatidx = 0; flatidx < W.GetSize(); ++flatidx)
            {
               W[flatidx] = double(flatidx);
            }

            for (int flatidx = 0; flatidx < B.GetSize(); ++flatidx)
            {
               B[flatidx] = double(flatidx);
            }

            M.SwitchToGPU();
            D.SwitchToGPU();
            T.MoveToGPU();
            W.MoveToGPU();
            B.MoveToGPU();
            TE["D_e_k1_k2 = W_k1_k2 T_e_k1_k2"](D,W,T);
            TE["M_e_i1_i2_j1_j2=B_k1_i1 B_k1_j1 B_k2_i2 B_k2_j2 D_e_k1_k2"](M,B,B,B,B,D);

            M.MoveFromGPU();
            D.MoveFromGPU();
            T.SwitchFromGPU();
            W.SwitchFromGPU();
            B.SwitchFromGPU();

            //Check D computation
            for (int e = 0; e < 10; ++e)
            {
               for (int k1 = 0; k1 < 4; ++k1)
               {
                  for (int k2 = 0; k2 < 4; ++k2)
                  {
                     CHECK(D(e,k1,k2) == Approx(W(k1,k2)*T(e,k1,k2)));
                  }
               }
            }

            //Check M computation
            int massBadCount = 0;
            for (int e = 0; e < 10; ++e)
            {
               for (int i1 = 0; i1 < 4; ++i1)
               {
                  for (int j1 = 0; j1 < 4; ++j1)
                  {
                     for (int i2 = 0; i2 < 4; ++i2)
                     {
                        for (int j2 = 0; j2 < 4; ++j2)
                        {
                           double Msum = 0.0;
                           for (int k1 = 0; k1 < 4; ++k1)
                           {
                              for (int k2 = 0; k2 < 4; ++k2)
                              {
                                 Msum += B(k1,i1)*B(k1,j1)*B(k2,i2)*B(k2,j2)*D(e,k1,k2);
                              }
                           }
                           massBadCount += (M(e,i1,i2,j1,j2) == Approx(Msum)) ? 0 : 1;
                        }
                     }
                  }
               }
            }
            
            if (massBadCount > 0)
            {
               std::cout << "Number bad indices:  " << massBadCount << std::endl;
               std::cout << TE["M_e_i1_i2_j1_j2=B_k1_i1 B_k1_j1 B_k2_i2 B_k2_j2 D_e_k1_k2"].GetImplementation(M,B,B,B,B,D);
            }
            REQUIRE(massBadCount == 0);  
         }

         SECTION("2D Stiffness Partial Assembly")
         {
            Tensor B(4, 4);
            Tensor G(4, 4);
            Tensor T(5, 4, 4);
            Tensor X(5, 4, 4);
            Tensor D(5, 2, 2, 4, 4);
            Tensor U(2, 5, 4, 4);
            Tensor Z(2, 5, 4, 4);
            Tensor Y(5, 4, 4);
            for (int i = 0; i < B.GetSize(); ++ i)
               B[i] = double(i);
            for (int i = 0; i < G.GetSize(); ++ i)
               G[i] = double(i)/2.0;
            for (int i = 0; i < X.GetSize(); ++ i)
               X[i] = double(i);            
            for (int i = 0; i < D.GetSize(); ++ i)
               D[i] = double(i);  

            SliceTensor U1(U, 0), U2(U, 1);
            SliceTensor Z1(Z, 0), Z2(Z, 1);

            //U1_e_k1_k2 = G_k1_i1 B_k2_i2 X_e_i1_i2
            TE["BX_e_i1_k2 = B_k2_i2 X_e_i1_i2"](T, B, X);
            TE["U1_e_k1_k2 = G_k1_i1 BX_e_i1_k2"](U1, G, T);
            U1.MoveFromGPU();
            for (int e = 0; e < 5; ++ e)
            {
               for (int k1 = 0; k1 < 4; ++ k1)
               {
                  for (int k2 = 0; k2 < 4; ++ k2)
                  {
                     double U1sum = 0.0;
                     for (int i1 = 0; i1 < 4; ++i1)
                        for (int i2 = 0; i2 < 4; ++i2)
                           U1sum += G(k1,i1)*B(k2,i2)*X(e,i1,i2);
                     REQUIRE(U1(e,k1,k2) == Approx(U1sum));
                  }
               }
            }            

            //U2_e_k1_k2 = B_k1_i1 G_k2_i2 X_e_i1_i2
            TE["GX_e_i1_k2 = G_k2_i2 X_e_i1_i2"](T, G, X);
            TE["U2_e_k1_k2 = B_k1_i1 GX_e_i1_k2"](U2, B, T);
            U2.MoveFromGPU();
            for (int e = 0; e < 5; ++ e)
            {
               for (int k1 = 0; k1 < 4; ++ k1)
               {
                  for (int k2 = 0; k2 < 4; ++ k2)
                  {
                     double U2sum = 0.0;
                     for (int i1 = 0; i1 < 4; ++i1)
                        for (int i2 = 0; i2 < 4; ++i2)
                           U2sum += B(k1,i1)*G(k2,i2)*X(e,i1,i2);
                     REQUIRE(U2(e,k1,k2) == Approx(U2sum));
                  }
               }
            } 


            TE["Z_m_e_k1_k2 = D_e_m_n_k1_k2 U_n_e_k1_k2"](Z, D, U);
            Z.MoveFromGPU();
            for (int m = 0; m < 2; ++m)
            {
               for (int e = 0; e < 5; ++ e)
               {
                  for (int k1 = 0; k1 < 4; ++ k1)
                  {
                     for (int k2 = 0; k2 < 4; ++ k2)
                     {
                        double Zsum = 0.0;
                        for (int n = 0; n < 2; ++n)
                           Zsum += D(e,m,n,k1,k2)*U(n,e,k1,k2);

                        if (Z(m,e,k1,k2) != Approx(Zsum))
                           std::cout << TE["Z_m_e_k1_k2 = D_e_m_n_k1_k2 U_n_e_k1_k2"].GetImplementation(Z, D, U);
                        REQUIRE(Z(m,e,k1,k2) == Approx(Zsum));
                     }
                  }
               }
            }

            //Y_e_i1_i2 += G_k1_i1 B_k2_i2 Z1_e_k1_k2
            TE["BZ1_e_i2_k1 = B_k2_i2 Z1_e_k1_k2"](T, B, Z1);
            TE["Y_e_i1_i2 = G_k1_i1 BZ1_e_i2_k1"](Y, G, T);
            Y.MoveFromGPU();
            {
               for (int e = 0; e < 5; ++ e)
               {
                  for (int i1 = 0; i1 < 4; ++ i1)
                  {
                     for (int i2 = 0; i2 < 4; ++ i2)
                     {
                        double Ysum = 0.0;
                        for (int k1 = 0; k1 < 4; ++k1)
                           for (int k2 = 0; k2 < 4; ++k2)
                              Ysum += G(k1,i1)*B(k2,i2)*Z1(e,k1,k2);
                        REQUIRE(Y(e,i1,i2) == Approx(Ysum));
                     }
                  }
               }
            }

            //Y_e_i1_i2 += B_k1_i1 G_k2_i2 Z2_e_k1_k2
            TE["GZ2_e_i2_k1 = G_k2_i2 Z2_e_k1_k2"](T, G, Z2);
            TE["Y_e_i1_i2 = B_k1_i1 GZ2_e_i2_k1"](Y, B, T);
            Y.MoveFromGPU();
            {
               for (int e = 0; e < 5; ++ e)
               {
                  for (int i1 = 0; i1 < 4; ++ i1)
                  {
                     for (int i2 = 0; i2 < 4; ++ i2)
                     {
                        double Ysum = 0.0;
                        for (int k1 = 0; k1 < 4; ++k1)
                           for (int k2 = 0; k2 < 4; ++k2)
                              Ysum += B(k1,i1)*G(k2,i2)*Z2(e,k1,k2);
                        REQUIRE(Y(e,i1,i2) == Approx(Ysum));
                     }
                  }
               }
            }
         }

         SECTION("3D Stiffness")
         {
            std::string kernel_str = "S_e_i1_i2_i3_j1_j2_j3 = B_i1_j1_k1_m_n B_i2_j2_k2_m_n B_i3_j3_k3_m_n D_e_k1_k2_k3_m_n";
            Tensor S(10, 5, 5, 5, 5, 5, 5);
            Tensor Btilde1(5, 5, 5, 3, 3);
            Tensor Btilde2(5, 5, 5, 3, 3);
            Tensor Btilde3(5, 5, 5, 3, 3);
            Tensor D(10, 5, 5, 5, 3, 3);

            S.MapToGPU();
            Btilde1.MapToGPU();
            Btilde2.MapToGPU();
            Btilde3.MapToGPU();
            D.MapToGPU();

            for (int flatidx = 0; flatidx < Btilde1.GetSize(); ++flatidx)
            {
               Btilde1[flatidx] = double(flatidx % 11);
               Btilde2[flatidx] = 2.0*double(flatidx % 7);
               Btilde3[flatidx] = 3.0*double(flatidx % 13);
            }

            for (int flatidx = 0; flatidx < D.GetSize(); ++flatidx)
            {         
               D[flatidx] = 3.5*double(flatidx % 5);
            }

            S.SwitchToGPU();
            Btilde1.MoveToGPU();
            Btilde2.MoveToGPU();
            Btilde3.MoveToGPU();
            D.MoveToGPU();

            TE[kernel_str](S, Btilde1, Btilde2, Btilde3, D);

            S.MoveFromGPU();
            Btilde1.MoveFromGPU();
            Btilde2.MoveFromGPU();
            Btilde3.MoveFromGPU();
            D.MoveFromGPU();               

            //Intentionally flatttened whitespace
            int stiff3DBadCount = 0;
            for (int e = 0; e < 10; ++e)
            for (int i1 = 0; i1 < 5; ++i1)
            for (int i2 = 0; i2 < 5; ++i2)
            for (int i3 = 0; i3 < 5; ++i3)
            for (int j1 = 0; j1 < 5; ++j1)
            for (int j2 = 0; j2 < 5; ++j2)
            for (int j3 = 0; j3 < 5; ++j3)
            {
               double Ssum = 0.0;
               for (int k1 = 0; k1 < 5; ++k1)
               for (int m = 0; m < 3; ++m)
               for (int n = 0; n < 3; ++n)                     
               for (int k2 = 0; k2 < 5; ++k2)
               for (int k3 = 0; k3 < 5; ++k3)
               {
                  Ssum += Btilde1(i1,j1,k1,m,n) * Btilde2(i2,j2,k2,m,n) * Btilde3(i3,j3,k3,m,n) *
                          D(e,k1,k2,k3,m,n);
               }
               if (!(S(e,i1,i2,i3,j1,j2,j3) == Approx(Ssum)))
                  stiff3DBadCount++;
            }
            CHECK(stiff3DBadCount == 0);
            if (stiff3DBadCount > 0)
            {
               std::vector<Tensor*> inputs = {&Btilde1, &Btilde2, &Btilde3, &D};
               std::cout << "Number bad indices:  " << stiff3DBadCount << std::endl;
               std::cout << TE[kernel_str].GetImplementation(&S, inputs);
            }              
         }

         /*SECTION("Async launching")
         {
            Tensor A1(256, 256);
            Tensor A2(256, 256);
            Tensor B(256, 256);
            Tensor C(256, 256);
            Tensor D(256);

            for (int i = 0; i < B.GetSize(); ++i)
               B[i] = double(i);

            for (int i = 0; i < C.GetSize(); ++i)
               C[i] = 2*double(i) - 256.0;

            for (int i = 0; i < D.GetSize(); ++i)
               D[i] = 256.0 - double(i);

            TE.SetAsyncLaunch();
            TE["A_i_j = B_i_j D_j"](A1, B, D);
            TE["A_i_j += C_i_j D_i"](A1, C, D);
            TE.ReSyncLaunch();

            TE["A_i_j = B_i_j D_j"](A2, B, D);
            TE["A_i_j += C_i_j D_i"](A2, C, D);

            A1.MoveFromGPU();
            A2.MoveFromGPU();
            for (int i = 0; i < A1.GetSize(); ++i)
               REQUIRE(A1[i] == Approx(A2[i]));

         }*/
      }
   }
}