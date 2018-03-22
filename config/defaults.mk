#Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at the Lawrence Livermore National Laboratory
#Written by Aaron Fisher (fisher47@llnl.gov). LLNL-CODE-738419.
#All rights reserved.
#This file is part of Acrotensor. For details, see https://github.com/LLNL/acrotensor.

#Default values for utilizing nvcc+gcc on a P100 system
DEBUG = NO
CUDADIR = /usr/local/cuda
CXX = $(CUDADIR)/bin/nvcc
UTILCXX = $(CXX)
CXX_OPT = -O3 -g -arch compute_60 -x cu --std=c++11 -DACRO_HAVE_CUDA --compiler-options="-fPIC"
CXX_DEBUG = -G -g -arch compute_60 -x cu --std=c++11 -DACRO_HAVE_CUDA --compiler-options="-fPIC"
UNITTEST_LDFLAGS = -O0 -G -arch compute_60 --std=c++11 -lnvrtc -lcuda -L$(CUDADIR)/lib64