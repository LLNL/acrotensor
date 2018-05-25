//Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at the Lawrence Livermore National Laboratory
//Written by Aaron Fisher (fisher47@llnl.gov). LLNL-CODE-738419.
//All rights reserved.
//This file is part of Acrotensor. For details, see https://github.com/LLNL/acrotensor.

#include "catch.hpp"
#include "AcroTensor.hpp"
#include <iostream>
#include <random>

using namespace acro;

void test_suite_on_engine(TensorEngine &TE);


TEST_CASE("TensorEngine", "[TensorEngine]")
{
   TensorEngine TE1("CPUInterpreted");
   test_suite_on_engine(TE1);

   if (isCudaReady()) 
   {
      TensorEngine TE2("Cuda");
      test_suite_on_engine(TE2);
   } 
}


void test_suite_on_engine(TensorEngine &TE)
{
   std::random_device rd;
   std::mt19937 twister(rd());
   std::uniform_real_distribution<double> random(0.0, 1.0);

   SECTION(TE.GetExecType() + " Computations")
   {
      SECTION("Assert Compatible Dimensions")
      {
         Tensor T1out_3(3), T2out_3_3(3, 3), T1_3(3), T1_2(2), T2_3_3(3,3);
         Tensor Tnoinit;
         REQUIRE_NOTHROW(TE("A_i=B_iC_i", T1out_3, T1_3, T1_3));
         REQUIRE_THROWS(TE("A_i=B_iC_i", T1out_3, T1_3, T1_2));
         REQUIRE_NOTHROW(TE("A_i=B_i_jC_i", T1out_3, T2_3_3, T1_3));
         REQUIRE_THROWS(TE("A_i=B_i_jC_i", T1out_3, T2_3_3, T1_2));
         REQUIRE_THROWS(TE("A_i=B_i_jC_i", Tnoinit, T2_3_3, T1_2));
         REQUIRE_THROWS(TE("A_i=B_i_jC_i", T1out_3, T2_3_3, Tnoinit));
      }

      SECTION("Basic computations")
      {
         Tensor T1out_3(3), T2out_3_3(3, 3), T1_3(3), T1_2(2), T2_3_3(3,3);
         T2_3_3.Set(0.0);
         T1_3.Set(0.0);

         SECTION("Simple copy")
         {
            T1_3(0) = 1.0; T1_3(1) = 2.0; T1_3(2) = 3.0;
            TE("A_i=B_i", T1out_3, T1_3);
            T1out_3.MoveFromGPU();
            for (int d = 0; d < 3; ++d) {
               REQUIRE(T1out_3(d) == Approx(T1_3(d)));
            }
         }
     
         SECTION("Extract diagonal")
         {
            T2_3_3(0,0) = 4.0; T2_3_3(1,1) = 5.0; T2_3_3(2,2) = 6.0;
            TE("A_i=B_i_i", T1out_3, T2_3_3);
            T1out_3.MoveFromGPU();
            for (int d = 0; d < 3; ++d) {
               REQUIRE(T1out_3(d) == Approx(T2_3_3(d,d)));
            }
         }

         SECTION("Contract and broadcast")
         {
            TE("A_i=B_j_j", T1out_3, T2_3_3);
            T1out_3.MoveFromGPU();
            for (int d = 0; d < 3; ++d) {
               REQUIRE(T1out_3(d) == Approx(T2_3_3(0,0) + T2_3_3(1,1) + T2_3_3(2,2)));
            }
         }

         //Matvec
         SECTION("Matvec")
         {
            T2_3_3.Set(1.0);
            T1_3.Set(1.0);
            TE("A_i = C_i_j B_j", T1out_3, T2_3_3, T1_3);
            T1out_3.MoveFromGPU();
            for (int d = 0; d < 3; ++d) {
               CHECK(T1out_3(d) == Approx(3.0));
            }
         }

         SECTION("Transpose")
         {
            for (int flatidx = 0; flatidx < T2_3_3.GetSize(); ++flatidx)
            {
               T2_3_3[flatidx] = double(flatidx);
            }
            TE("A_i_j=B_j_i", T2out_3_3, T2_3_3);
            T2out_3_3.MoveFromGPU();
            for (int i = 0; i < 3; ++i)
            {
               for (int j = 0; j < 3; ++j)
               {
                  REQUIRE(T2out_3_3(i,j) == Approx(T2_3_3(j,i)));
               }
            }
         }

         SECTION("Matrix times Matrix transpose")
         {
            for (int mj = 0; mj < 9; ++ mj) 
            {
               T2_3_3[mj] = float(mj+1);
            }
            TE("A_m_n=B_m_j B_n_j", T2out_3_3, T2_3_3, T2_3_3);
            T2out_3_3.MoveFromGPU();
            for (int m = 0; m < 3; ++m)
            {
               for (int n = 0; n < 3; ++n)
               {
                  double sum = 0.0;
                  for (int j = 0; j < 3; ++j)
                  {
                     sum += T2_3_3(m,j)*T2_3_3(n,j);
                  }
                  REQUIRE(T2out_3_3(m,n) == Approx(sum));
               }
            }
         }

         SECTION("Singleton dimensions")
         {
            Tensor T5(1, 2, 3, 1, 5);
            T5.Set(1.0);
            TE("A_i=B_j_k_l_m_n", T1out_3, T5);
            T1out_3.MoveFromGPU();
            REQUIRE(T1out_3(0) == Approx(30.0));
         }
      }


#ifdef ACRO_HAVE_CUDA
      if (isCudaReady()) 
      {
         SECTION("Set External Cuda Context")
         {
            CUcontext ctx;
            cuCtxCreate(&ctx, 0, 0);
            setCudaContext(ctx);

            Tensor A(3);
            Tensor B(3);

            B(0) = 1.0; B(1) = 2.0; B(2) = 3.0;
            TE("A_i=B_i", A, B);
            A.MoveFromGPU();
            CHECK(A(0) == Approx(B(0)));
            CHECK(A(1) == Approx(B(1)));
            CHECK(A(2) == Approx(B(2)));
         }

         SECTION("Computation with externally defined cuda Data")
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
            TE("B_i=A_i", B, A);
            A.MoveFromGPU();
            B.MoveFromGPU();
            for (int i = 0; i < 125; ++i)
            {
               REQUIRE(A(i) == B(i));
            }
            acroCudaErrorCheck(cudaFree(a_device_data));
         }
      }
#endif

      SECTION("Transpose")
      {
         Tensor A(4,4);
         Tensor B(4,4);

         for (int flatidx = 0; flatidx < B.GetSize(); ++flatidx)
         {
            B[flatidx] = double(flatidx);
         }

         TE("A_i_j=B_j_i", A, B);
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

      SECTION("MultiKernel Launch")
      {
         Tensor A(3);
         Tensor B1(3), B2(3);
         Tensor C(3,3);

         B1(0) = 1.0; B1(1) = 2.0; B1(2) = 3.0;
         B2.Set(1.0);
         C.Set(1.0);
         TE.BeginMultiKernelLaunch();
         TE("A_i = B1_i", A, B1);
         TE("A_i = C_i_j B2_j", A, C, B2);
         TE.EndMultiKernelLaunch();
         A.MoveFromGPU();
         CHECK(A(0) == Approx(3.0));
         CHECK(A(1) == Approx(3.0));
         CHECK(A(2) == Approx(3.0));
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

            TE("D_e_k1_k2 = W_k1_k2 T_e_k1_k2", D,W,T);
            TE("M_e_i1_i2_j1_j2=B_k1_i1 B_k1_j1 B_k2_i2 B_k2_j2 D_e_k1_k2", M,B,B,B,B,D);

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
               std::cout << TE.GetImplementation("M_e_i1_i2_j1_j2=B_k1_i1 B_k1_j1 B_k2_i2 B_k2_j2 D_e_k1_k2",M,B,B,B,B,D);
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
            TE("BX_e_i1_k2 = B_k2_i2 X_e_i1_i2", T, B, X);
            TE("U1_e_k1_k2 = G_k1_i1 BX_e_i1_k2", U1, G, T);
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
            TE("GX_e_i1_k2 = G_k2_i2 X_e_i1_i2", T, G, X);
            TE("U2_e_k1_k2 = B_k1_i1 GX_e_i1_k2", U2, B, T);
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


            TE("Z_m_e_k1_k2 = D_e_m_n_k1_k2 U_n_e_k1_k2", Z, D, U);
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
                           std::cout << TE.GetImplementation("Z_m_e_k1_k2 = D_e_m_n_k1_k2 U_n_e_k1_k2",Z, D, U);
                        REQUIRE(Z(m,e,k1,k2) == Approx(Zsum));
                     }
                  }
               }
            }

            //Y_e_i1_i2 += G_k1_i1 B_k2_i2 Z1_e_k1_k2
            TE("BZ1_e_i2_k1 = B_k2_i2 Z1_e_k1_k2", T, B, Z1);
            TE("Y_e_i1_i2 = G_k1_i1 BZ1_e_i2_k1", Y, G, T);
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
            TE("GZ2_e_i2_k1 = G_k2_i2 Z2_e_k1_k2", T, G, Z2);
            TE("Y_e_i1_i2 = B_k1_i1 GZ2_e_i2_k1", Y, B, T);
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
            std::string kernel_str = "S_e_i1_i2_i3_j1_j2_j3 = B1_i1_j1_k1_m_n B2_i2_j2_k2_m_n B3_i3_j3_k3_m_n D_e_k1_k2_k3_m_n";
            Tensor S(10, 5, 5, 5, 5, 5, 5);
            Tensor Btilde1(5, 5, 5, 3, 3);
            Tensor Btilde2(5, 5, 5, 3, 3);
            Tensor Btilde3(5, 5, 5, 3, 3);
            Tensor D(10, 5, 5, 5, 3, 3);

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

            TE(kernel_str, S, Btilde1, Btilde2, Btilde3, D);

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
               std::cout << TE.GetImplementation(kernel_str, &S, inputs);
            }              
         }

         SECTION("2D Stiffness Create D for partial Assembly (Order 1)")
         {
            Tensor D(10, 2, 2, 2, 2);
            Tensor J(10, 2, 2, 2, 2);
            Tensor Jinv(10, 2, 2, 2, 2);
            Tensor Jdet(10, 2, 2);
            Tensor C(10, 2, 2);
            Tensor W(10, 2, 2);

            for (int flatidx = 0; flatidx < D.GetSize(); ++flatidx)
            {
               J[flatidx] = double(flatidx);
               Jinv[flatidx] = double(flatidx)+1.0;
            }

            for (int flatidx = 0; flatidx < C.GetSize(); ++flatidx)
            {
               C[flatidx] = double(flatidx)+2;
               Jdet[flatidx] = double(flatidx)+3;
               W[flatidx] = double(flatidx) / 10.0;
            }

            TE("D_e_m_n_k1_k2 = W_e_k1_k2 C_e_k1_k2 Jdet_e_k1_k2 Jinv_e_k1_k2_m_j Jinv_e_k1_k2_n_j",
                  D, W, C, Jdet, Jinv, Jinv);

            //Intentionally flatttened whitespace
            D.MoveFromGPU();
            int DBadCount = 0;
            for (int e = 0; e < 10; ++e)
            for (int m = 0; m < 2; ++m)
            for (int n = 0; n < 2; ++n)
            for (int k1 = 0; k1 < 2; ++k1)
            for (int k2 = 0; k2 < 2; ++k2)
            {
               double sum = 0.0;
               for (int j = 0; j < 2; ++j)
               {
                  sum += W(e,k1,k2)*C(e,k1,k2)*Jdet(e,k1,k2)*Jinv(e,k1,k2,m,j)*Jinv(e,k1,k2,n,j);
               }
               if (!(D(e,m,n,k1,k2) == Approx(sum)))
                  DBadCount++;
            }
            CHECK(DBadCount == 0);
         }

         SECTION("3D Stiffness Create D for partial Assembly (Order 4)")
         {
            Tensor D(10, 3, 3, 5, 5, 5);
            Tensor Jinv(10, 5, 5, 5, 3, 3);
            Tensor Jdet(10, 5, 5, 5);
            Tensor C(10, 5, 5, 5);
            Tensor W(5, 5, 5);

            for (int flatidx = 0; flatidx < D.GetSize(); ++flatidx)
            {
               Jinv[flatidx] = double(flatidx)+1.0;
            }

            for (int flatidx = 0; flatidx < C.GetSize(); ++flatidx)
            {
               C[flatidx] = double(flatidx)+2;
               Jdet[flatidx] = double(flatidx)+3;
               
            }

            for (int flatidx = 0; flatidx < W.GetSize(); ++flatidx)
            {
               W[flatidx] = double(flatidx) / 10.0;
            }

            TE("D_e_m_n_k1_k2_k3 = W_k1_k2_k3 C_e_k1_k2_k3 Jdet_e_k1_k2_k3 Jinv_e_k1_k2_k3_n_j Jinv_e_k1_k2_k3_m_j",
                  D, W, C, Jdet, Jinv, Jinv);

            //Intentionally flatttened whitespace
            D.MoveFromGPU();
            int DBadCount = 0;
            for (int e = 0; e < 10; ++e)
            for (int m = 0; m < 2; ++m)
            for (int n = 0; n < 2; ++n)
            for (int k1 = 0; k1 < 2; ++k1)
            for (int k2 = 0; k2 < 2; ++k2)
            for (int k3 = 0; k3 < 2; ++k3)
            {
               double sum = 0.0;
               for (int j = 0; j < 3; ++j)
               {
                  sum += W(k1,k2,k3)*C(e,k1,k2,k3)*Jdet(e,k1,k2,k3)*Jinv(e,k1,k2,k3,m,j)*Jinv(e,k1,k2,k3,n,j);
               }
               if (!(D(e,m,n,k1,k2,k3) == Approx(sum)))
                  DBadCount++;
            }
            CHECK(DBadCount == 0);
         }
      }   
   }

   SECTION("Batched Inverses/Determinents")
   {
      Tensor T1(1,1), T2(2,2), T3(3,3);
      Tensor TB1(10,1,1), TB2(10,2,2), TB3(10,3,3);
      Tensor DB1(10), DB2(10), DB3(10);
      Tensor TBOUT1(10,1,1), TBOUT2(10,2,2), TBOUT3(10,3,3);
      Tensor IB1(10,1,1), IB2(10,2,2), IB3(10,3,3);
      Tensor TBB1(4,2,1,1), TBB2(4,2,2,2), TBB3(4,2,3,3);
      Tensor X1(10), X2(3), X3(3,2), X4(4,4);

      SECTION("Assert compatible dimensions")
      {
         REQUIRE_THROWS(TE.BatchMatrixInverse(X1, X1));
         REQUIRE_THROWS(TE.BatchMatrixInverse(X2, X2));
         REQUIRE_THROWS(TE.BatchMatrixInverse(X3, X3));
         REQUIRE_THROWS(TE.BatchMatrixInverse(X4, X4));
         REQUIRE_THROWS(TE.BatchMatrixInverse(T3, T2));
         REQUIRE_NOTHROW(TE.BatchMatrixInverse(T1, T1));
         REQUIRE_NOTHROW(TE.BatchMatrixInverse(T2, T2));
         REQUIRE_NOTHROW(TE.BatchMatrixInverse(T3, T3));
         REQUIRE_NOTHROW(TE.BatchMatrixInverse(TB1, TB1));
         REQUIRE_NOTHROW(TE.BatchMatrixInverse(TB2, TB2));
         REQUIRE_NOTHROW(TE.BatchMatrixInverse(TB3, TB3));
         REQUIRE_NOTHROW(TE.BatchMatrixInverse(TBB1, TBB1));
         REQUIRE_NOTHROW(TE.BatchMatrixInverse(TBB2, TBB2));
         REQUIRE_NOTHROW(TE.BatchMatrixInverse(TBB3, TBB3));         
      }

      SECTION("Random Symm Diag Dominant A*Ainv = I and det(A)*det(Ainv) = 1")
      {
         SECTION("1x1")
         {
            for (int b = 0; b < TB1.GetDim(0); ++b)
            {
               TB1(b,0,0) = random(twister) + 1e-20;
            }
            TE.BatchMatrixInvDet(TBOUT1, DB1, TB1);
            TE.BatchMatrixDet(DB2, TBOUT1);
            TE("I_b_i_k = TI_b_i_j T_b_j_k", IB1, TBOUT1, TB1);
            TE("I_b = DI_b D_b", DB3, DB2, DB1);

            IB1.MoveFromGPU();
            DB3.MoveFromGPU();
            for (int b = 0; b < IB1.GetDim(0); ++b)
            {
               REQUIRE(IB1(b,0,0) == Approx(1.0));
               REQUIRE(DB3(b) == Approx(1.0));
            }
         }

         SECTION("2x2")
         {
            for (int b = 0; b < TB2.GetDim(0); ++b)
            {
               TB2(b, 0, 0) = random(twister) + 4.0;
               TB2(b, 0, 1) = random(twister);
               TB2(b, 1, 0) = TB2(b, 0, 1);
               TB2(b, 1, 1) = random(twister) + 4.0;
            }
            TE.BatchMatrixInvDet(TBOUT2, DB1, TB2);
            TE.BatchMatrixDet(DB2, TBOUT2);
            TE("I_b_i_k = TI_b_i_j T_b_j_k", IB2, TBOUT2, TB2);
            TE("I_b = DI_b D_b", DB3, DB2, DB1);

            IB2.MoveFromGPU();
            DB3.MoveFromGPU();
            for (int b = 0; b < IB2.GetDim(0); ++b)
            {
               REQUIRE(IB2(b,0,0) == Approx(1.0));
               REQUIRE(IB2(b,0,1) == Approx(0.0));
               REQUIRE(IB2(b,1,0) == Approx(0.0));
               REQUIRE(IB2(b,1,1) == Approx(1.0));
               REQUIRE(DB3(b) == Approx(1.0));
            }
         }

         SECTION("3x3")
         {
            for (int b = 0; b < TB3.GetDim(0); ++b)
            {
               TB3(b, 0, 0) = random(twister) + 4.0;
               TB3(b, 0, 1) = random(twister);
               TB3(b, 0, 2) = random(twister);
               TB3(b, 1, 0) = TB3(b, 0, 1);
               TB3(b, 1, 1) = random(twister) + 4.0;
               TB3(b, 1, 2) = random(twister);
               TB3(b, 2, 0) = TB3(b, 0, 2);
               TB3(b, 2, 1) = TB3(b, 1, 2);
               TB3(b, 2, 2) = random(twister) + 4.0;
            }
            TE.BatchMatrixInvDet(TBOUT3, DB1, TB3);
            TE.BatchMatrixDet(DB2, TBOUT3);
            TE("I_b_i_k = TI_b_i_j T_b_j_k", IB3, TBOUT3, TB3);
            TE("I_b = DI_b D_b", DB3, DB2, DB1);

            IB3.MoveFromGPU();
            DB3.MoveFromGPU();
            for (int b = 0; b < IB3.GetDim(0); ++b)
            {
               REQUIRE(IB3(b,0,0) == Approx(1.0));
               REQUIRE(IB3(b,0,1) == Approx(0.0));
               REQUIRE(IB3(b,0,2) == Approx(0.0));
               REQUIRE(IB3(b,1,0) == Approx(0.0));
               REQUIRE(IB3(b,1,1) == Approx(1.0));
               REQUIRE(IB3(b,1,2) == Approx(0.0));
               REQUIRE(IB3(b,2,0) == Approx(0.0));
               REQUIRE(IB3(b,2,1) == Approx(0.0));
               REQUIRE(IB3(b,2,2) == Approx(1.0));
               REQUIRE(DB3(b) == Approx(1.0));
            }
         }         
      }
   }

   SECTION("Scatters and Gathers")
   {
      Tensor TBIG(8);
      Tensor T(6);
      IndexMapping M(6,8);
      int Marr[8] = {0,1,3,4,1,2,4,5};
      for (int i = 0; i < 6; ++i)
      {
         T[i] = 2*i + 37;
      }
      for (int i = 0; i < 8; ++i)
      {
         M[i] = Marr[i];
         TBIG[i] = 3*i - 5;
      }

      SECTION("FlatIndexedScatter")
      {
         TE.FlatIndexedScatter(TBIG, T, M);
         TBIG.MoveFromGPU();
         for (int i = 0; i < 8; ++i)
         {
            REQUIRE(TBIG[i] == Approx(T[M[i]]));
         }
      }

      SECTION("FlatIndexedSumGather")
      {
         TE.FlatIndexedSumGather(T, TBIG, M);
         REQUIRE(T[0] == Approx(TBIG[0]));
         REQUIRE(T[1] == Approx(TBIG[1] + TBIG[4]));
         REQUIRE(T[2] == Approx(TBIG[5]));
         REQUIRE(T[3] == Approx(TBIG[2]));
         REQUIRE(T[4] == Approx(TBIG[3] + TBIG[6]));
         REQUIRE(T[5] == Approx(TBIG[7]));
      }

   }

}