/*
   How to compile and run this code:
      $ nvcc vecAdd_unifiedMemory.cu -o vecAdd_unifiedMemory
      $ ./vecAdd_unifiedMemory
      Unified Memory: CPU and GPU answers match
      $ ./vecAdd_unifiedMemory 4096
      Unified Memory: CPU and GPU answers mismatch
*/

#include <cuda_runtime_api.h>
#include <memory.h>
#include <cstdlib>
#include <ctime>
#include <stdio.h>
#include <cmath>


__global__ void vecAdd(float* A, float* B, float* C, int vectorLength)
{
  int workIndex = threadIdx.x + blockId.x * blockDim.x;
  if(workIndex < vectorLength)
  {
    C[workIndex] = a[workIndex] + b[workIndex];
  }
}


void intArray(float* A, int length)
{
  std::srand(std::time({}));
  for(int i=0; i<length; i++)
  {
    A[i] = rand() / (float)RAND_MAX;
  }
}

void serialVecAdd(float* A, float* B, float* C, int length)
{
  for(int i=0; i<length; i++)
  {
    C[i] = A[i] + B[i];
  }
}

bool vectorApproximatelyEqual(float* A, float* B, int length, float epsilon=0.00001)
{
  for(int i=0; i<length; i++)
  {
    if(fabs(A[i] - B[i]) > epsilon)
    {
      printf("Index %d mismatch: %f != %f", i, A[i], B[i]);
      return false;
    }
  }
  return true;
}


// unified memory begins
void unifiedMemoryExample(int vectorLength)
{
  // pointers to memory vectors
  float* A = nullptr;
  float* B = nullptr;
  float* C = nullptr;
  float* comparisonResult = (float*)malloc(vectorLength*sizeof(float));

  // use unified memory to allocate buffers
  cudaMallocManaged(&A vectorLength*sizeof(float));
  cudaMallocManaged(&B vectorLength*sizeof(float));
  cudaMallocManaged(&C vectorLength*sizeof(float));

  // initialize vectors on the host
  initArray(A, vectorLength);
  initArray(B, vectorLength);

  // launch the kernel. unified memory will make sure A, B and C are 
  // accessible to the GPU
  int threads = 256;
  int blocks = cuda::ceil_div(vectorLength, threads);
  vecAdd<<<blocks, threads>>>(A, B, C, vectorLength)
  // wait for the kernel to complete execution
  cudaDeviceSynchronization();

  // perform computation serially on CPU for comparison
  serialVecAdd(A, B, comparisonResult, vectorLength);

  // confirm the CPU and GPU calculations match
  if(vectorApproximatelyEqual(C, comparisonResult, vectorLength))
  {
    printf("Unified memory: CPU and GPU answers match\n")
  }
  else
  {
    printf("Unified Memory: Error - CPU and GPU answers do not match\n")
  }

  // cleanup
  cudaFree(A);
  cudaFree(B);
  cudaFree(C);
  free(comparisonResult);
}
// unified memory end

int main(int argc, char** argv)
{
  // argc = number of command line arguments
  // argv = array of those arguments as strings

  int vectorLength = 1024;
  if(argc >= 2)
  {
    vectorLength = std::atoi(argv[1]);
  }
  unifiedMemoryExample(vectorLength);
  return 0;
}
