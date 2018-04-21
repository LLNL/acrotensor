//Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at the Lawrence Livermore National Laboratory
//Written by Aaron Fisher (fisher47@llnl.gov). LLNL-CODE-738419.
//All rights reserved.
//This file is part of Acrotensor. For details, see https://github.com/LLNL/acrotensor.

#include "CudaUtil.hpp"
#include <iostream>
#include <fstream>

#ifdef ACRO_HAVE_CUDA
namespace acro
{
CUcontext theCudaContext = NULL;

CudaKernel::CudaKernel() : 
    IntOpsPerIndex(0),
    FloatOpsPerIndex(0),
    MemReadsPerIndex(0),
    NumBlocks(0),
    ThreadsPerBlock(0),
    IsMultipleBlockPerOutput(true)
{

}


CudaKernel::~CudaKernel()
{
    for (auto it = Textures.begin(); it != Textures.end(); ++it)
    {
        cudaDestroyTextureObject(it->second);
    }
}


cudaTextureObject_t CudaKernel::GetTextureObject(int id)
{
    return Textures[id];
}


void CudaKernel::GenerateFunction()
{
    ensureCudaContext();
    nvrtcProgram prog;
    acroCudaErrorCheck(nvrtcCreateProgram(&prog,         // prog
                                          Code.c_str(),         // buffer
                                          NULL,    // name
                                          0,             // numHeaders
                                          NULL,          // headers
                                          NULL));        // includeNames
    const char *opts[] = {"--restrict", "--use_fast_math", "--gpu-architecture=compute_60", "-lineinfo"};
    //const char *opts[] = {"--restrict", "--use_fast_math", "-lineinfo"};
    nvrtcResult rcode = nvrtcCompileProgram(prog,  // prog
                                            4,     // numOptions
                                            opts); // options
    if (rcode != NVRTC_SUCCESS)
    {
        std::cout << "NVRTC Compilation error found in:" << std::endl;
        std::cout << Code << std::endl;
        size_t log_size;
        nvrtcGetProgramLogSize(prog, &log_size);
        char *compile_log = new char[log_size];
        nvrtcGetProgramLog(prog, compile_log);
        std::cout << compile_log << std::endl;
        delete[] compile_log;
        throw_error("Encountered in CudaKernel::GenerateFunction()");
    }


    size_t ptxSize;
    acroCudaErrorCheck(nvrtcGetPTXSize(prog, &ptxSize));
    char *ptx = new char[ptxSize];
    acroCudaErrorCheck(nvrtcGetPTX(prog, ptx));
    // Load the generated PTX and get a handle to the kernel.
    acroCudaErrorCheck(cuModuleLoadDataEx(&Module, ptx, 0, 0, 0));
    acroCudaErrorCheck(cuModuleGetFunction(&Function, Module, FunctionName.c_str()));
    acroCudaErrorCheck(nvrtcDestroyProgram(&prog));

    delete [] ptx;
}


void CudaKernel::SetGlobalArray(std::string &name, std::vector<int> &arr)
{
    CUdeviceptr device_arr;
    int bytesize = sizeof(int)*arr.size();
    acroCudaErrorCheck(cuModuleGetGlobal(&device_arr, NULL, Module, name.c_str()));
    acroCudaErrorCheck(cudaMemcpy((void*)device_arr, &arr[0], bytesize, cudaMemcpyHostToDevice));
}

void CudaKernel::Launch(std::vector<void*> &kernel_params, cudaStream_t cuda_stream)
{
    ensureCudaContext();
    acroCudaErrorCheck(cuLaunchKernel(Function,
                                      NumBlocks, 1, 1,            // grid dim
                                      ThreadsPerBlock, 1, 1,      // threads per block
                                      0, cuda_stream,             // shared mem and stream
                                      &kernel_params[0], 0));     // arguments
}


void CudaKernel::WriteCodeToFile(const char *fname)
{
    std::string fname_str(fname);
    WriteCodeToFile(fname_str);
}


void CudaKernel::WriteCodeToFile(std::string &fname)
{
    std::ofstream file;
    file.open(fname);
    file << Code;
    file.close();
}



__global__ void CudaSet(double *d, double val, int N)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx < N)
    {
        d[idx] = val;
    }
}


__global__ void CudaMult(double *d, double c, int N)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx < N)
    {
        d[idx] *= c;
    }
}


}

#endif