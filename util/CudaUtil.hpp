//Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at the Lawrence Livermore National Laboratory
//Written by Aaron Fisher (fisher47@llnl.gov). LLNL-CODE-738419.
//All rights reserved.
//This file is part of Acrotensor. For details, see https://github.com/LLNL/acrotensor.

#ifndef ACROBATIC_CUDA_UTIL_HPP
#define ACROBATIC_CUDA_UTIL_HPP

#include <vector>
#include <string>
#include <cstring>
#include <map>
#include <iostream>
#include "Error.hpp"
#include <type_traits>

#ifdef ACRO_HAVE_CUDA
#include "cuda.h"
#include "nvrtc.h"
#include "cuda_runtime.h"
#endif

namespace acro
{

#define RESTRICT __restrict__

#ifdef ACRO_HAVE_CUDA
class CudaKernel
{
   public:
   CudaKernel();
   ~CudaKernel();
   void GenerateFunction();
   void SetGlobalArray(std::string &ame, std::vector<int> &arr);
   void WriteCodeToFile(const char *fname);
   void WriteCodeToFile(std::string &fname);
   template<typename T>
   inline void AddTextureData(int id, std::vector<T> &data);
   cudaTextureObject_t GetTextureObject(int id);
   void Launch(std::vector<void*> &kernel_params, cudaStream_t cuda_stream = NULL);

   std::string FunctionName;
   std::string Code;
   CUmodule Module;
   CUfunction Function;
   int IntOpsPerIndex;
   int FloatOpsPerIndex;
   int MemReadsPerIndex;
   int NumBlocks;
   int ThreadsPerBlock;
   int MaxRegCount;
   bool IsMultipleBlockPerOutput;

   private:
   std::map<int, cudaTextureObject_t> Textures;
};

#define acroCudaErrorCheck(ans) acroCudaAssert((ans), __FILE__, __LINE__);
inline void acroCudaAssert(cudaError_t code, const char *file, int line)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"CUDA Error: %s\n", cudaGetErrorString(code));
      throw_error(std::string("Encountered at:  ") + std::string(file) + ":  " + std::to_string(line));
   }
}


inline void acroCudaAssert(nvrtcResult code, const char *file, int line)
{
   if (code != NVRTC_SUCCESS) 
   {
      fprintf(stderr,"NVRTC Error: %s\n", nvrtcGetErrorString(code));
      throw_error(std::string("Encountered at:  ") + std::string(file) + ":  " + std::to_string(line));
   }
}


inline void acroCudaAssert(CUresult code, const char *file, int line)
{
   if (code != CUDA_SUCCESS) 
   {
      const char *msg;
      cuGetErrorName(code, &msg);
      fprintf(stderr,"CUDA Error: %s\n", msg);
      throw_error(std::string("Encountered at:  ") + std::string(file) + ":  " + std::to_string(line));
   }
}


extern CUcontext theCudaContext;
inline void setCudaContext(void *ctx)
{

   theCudaContext = (CUcontext) ctx;

}


inline void ensureCudaContext()
{
    if (!theCudaContext)
    {
        acroCudaErrorCheck(cuCtxCreate(&theCudaContext, 0, 0));
    }
    acroCudaErrorCheck(cuCtxSetCurrent(theCudaContext));
}


template<typename T>
inline void CudaKernel::AddTextureData(int id, std::vector<T> &data)
{
    int Tsize = sizeof(T);
    int bitT = Tsize * 8;
    int bitTo2 = bitT / 2;
    int bitTo4 = bitT / 4;
    int arr_bytesize = Tsize*data.size();
    T *buffer;
    acroCudaErrorCheck(cudaMalloc(&buffer, arr_bytesize));
    acroCudaErrorCheck(cudaMemcpy((void*)buffer, &data[0], arr_bytesize, cudaMemcpyHostToDevice));

    // create texture object
    cudaResourceDesc resDesc;
    std::memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr = buffer;
    resDesc.res.linear.sizeInBytes = arr_bytesize;
    if (std::is_same<T,uchar2>::value || std::is_same<T,ushort2>::value || std::is_same<T,uint2>::value)
    {
        resDesc.res.linear.desc = cudaCreateChannelDesc( bitTo2, bitTo2, 0, 0, cudaChannelFormatKindUnsigned);
    }
    else if (std::is_same<T,char2>::value || std::is_same<T,short2>::value || std::is_same<T,int2>::value)
    {
        resDesc.res.linear.desc = cudaCreateChannelDesc( bitTo2, bitTo2, 0, 0, cudaChannelFormatKindSigned);
    }
    else if (std::is_same<T,float2>::value)
    {
        resDesc.res.linear.desc = cudaCreateChannelDesc( bitTo2, bitTo2, 0, 0, cudaChannelFormatKindFloat);
    }    
    else if (std::is_same<T,uchar4>::value || std::is_same<T,ushort4>::value || std::is_same<T,uint4>::value)
    {
        resDesc.res.linear.desc = cudaCreateChannelDesc( bitTo4, bitTo4, bitTo4, bitTo4, cudaChannelFormatKindUnsigned);
    }
    else if (std::is_same<T,char4>::value || std::is_same<T,short4>::value || std::is_same<T,int4>::value)
    {
        resDesc.res.linear.desc = cudaCreateChannelDesc( bitTo4, bitTo4, bitTo4, bitTo4, cudaChannelFormatKindSigned);
    }    
    else if (std::is_same<T,float4>::value)
    {
        resDesc.res.linear.desc = cudaCreateChannelDesc( bitTo4, bitTo4, bitTo4, bitTo4, cudaChannelFormatKindFloat);
    }
    else
    {
        resDesc.res.linear.desc = cudaCreateChannelDesc( bitTo2, bitTo2, 0, 0, cudaChannelFormatKindUnsigned);
    }


    cudaTextureDesc texDesc;
    std::memset(&texDesc, 0, sizeof(texDesc));
    texDesc.readMode = cudaReadModeElementType;

    Textures[id] = 0;
    cudaCreateTextureObject(&Textures[id], &resDesc, &texDesc, NULL);
}

__global__ void CudaSet(double *d, double val, int N);
__global__ void CudaMult(double *d, double c, int N);

__device__ int2 CudaWarpSort(int2 val);
__device__ int2 shfl_sync_int2(unsigned mask, int2 var, int srcLane, int width=32);

#endif

inline bool isCudaReady()
{
#ifndef ACRO_HAVE_CUDA
   return false;
#else
   int cuda_device_count = -1;
   cudaGetDeviceCount(&cuda_device_count);
   return (cuda_device_count > 0);
#endif
}



}

#endif //ACROBATIC_CUDA_UTIL_HPP