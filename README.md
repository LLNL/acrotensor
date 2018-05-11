# Acrotensor

Acrotensor is a C++/CUDA library for computing arbitrary tensor contractions both on CPUs and GPUs.  The tensors are dynamically sized allowing for a high degree of flexibility, and the tensor contractions are defined with a natural mathematical notation for maximum usability.  In order to maintain good performance contraction code is dynamical generated with fixed sizes and unrolled loops and then Just In Time (JIT) compiled to produce better optimized execution.

## Getting started

Acrotensor depends on a C++11 compiler and requires the nvcc CUDA wrapper on the compiler in order to handle the mix of C++ and CUDA.  To get the build started you will want to enter the acrotensor directory and run:
```
make config
```

This will generate a `config/config.mk` file with a set of defaults that you may need to change for your environment.  Once you have edited your `config.mk` simply enter the acrotensor directory and run:
```
make
```

This will build both static and dynamic libraries that you can link against in the `lib` folder and generate an `inc` directory with all of the header files that you will need.

If you would like to perform some sanity checks on Acrotensor before moving forward you can build and run the unit test suite by entering the acrotensor directory and running:
```
make unittest
```

## Usage

To gain access to the Acrotensor objects be sure to include `AcroTensor.hpp` and link against either the static or dynamic library.  The two user facing objects needed to utilize Acrotensor are `acrobatic::Tensor` and `acrobatic::TensorEngine`.  The `Tensor` objects can be constructed on the CPU with dimensions provided by a list of numbers or an `std::vector<int>`:
```
//Start of an example contraction that will add 1000 random matrices together on the GPU
std::vector<int> dims {1000, 3, 3};
acro::Tensor A(dims);    //1000x3x3 entry tensor
acro::Tensor B(1000);    //1000 entry tensor
acro::Tensor S(3,3);     //3x3 tensor
```

Once the tensors are created they can be accessed on the CPU with Tensor indexing using the `()` operator and linear indexing using the `[]` operator.  The data in the tensors are layed out linearly with the most significant index on the left.  There are also utility methods such as `Set()` and `Print()`:
```
for (int linear = 0; linear < 1000*3*3; ++linear)
   A[linear] = (double)rand() / RAND_MAX;

B.Set(1.0);
for (int i = 0; i < 3; ++i)
   for (int j = 0; j < 3; ++j)
      S(i, j) = 0.0;
```

Memory motion between the CPU and GPU can be accomplished by using the following `Tensor` methods:
```
A.MapToGPU();     //Allocate memory on the GPU
B.MapToGPU();
S.MapToGPU();

A.MoveToGPU();    //Copy the data to the GPU and indicate the the GPU has the fresh copy
B.MoveToGPU();

S.SwitchToGPU();  //Indicate that the GPU has the fresh copy without copying the data (good for outputs)
```

Tensor contractions can now be handled through a `TensorEngine` object.  Thesor engines can be initilized with different execution policies that can handle contractions on the CPU or GPU with different approaches.  The contraction string in the `[]` operator defines how the tensors will be indexed, multiplied and added.  The dimensions of the contraction operation are set by the dimensions of the tensors that are passed in via the `()` operator.  Any index that does not appear on the left hand side is sum across and contracted away in the ouput tensor.
```
acro::TensorEngine TE("Cuda");  //Initilize the engine with the Cuda exec policy
TE("S_i_j = A_n_i_j B_n", S, A, B);             //Contract on n and sum the 1000 matrices into 1
TE("S_i_j = A_n_i_j", S, A);                    //Same result as before since n is still contracted

S.MoveFromGPU();     //Get the results back from the GPU
S.Print();           //Display the results of the contraction
```
