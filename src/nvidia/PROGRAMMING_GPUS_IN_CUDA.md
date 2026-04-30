# Intro to CUDA C++
    - CUDA runtime API is most commonly used way of using CUDA in C++ (built on top of CUDA driver)

    - Checklist to run:
        [ ] CUDA Toolkit
        [ ] CUDA Driver API 
        [ ] NVIDIA Supported GPU

1. Compilation with NVCC
    - GPU code is written in C++ then compiled using *nvcc* (NVIDIA CUDA Compiler)
2. Kernels
    - kernel: functions that execute on the GPU which can be invoked from the host
        - Written to run by many parallel threads simultaneously

    a. Specifying Kernels   
        - Code for kernel is specified by `__global__` declaration
        - A kernel launch is an operation in which starts a kernel, usually from CPU
        - Kernels are functions with a *void* type

        ```
            // Kernel definition
            __global__ void vecAdd(float* A, float* B, float* C)
            {

            }
        ```

        ```
            // Kernel definition with chevron on launch
            // Where 4 represents blocks and 256 is threads (1024 total)
            // all running vecAdd simultaneously
            vecAdd<<<4, 256>>>(A, B, C);
        ```
    b. Launching Kernels
        - Number of threads that execute the kernel in parallel is specified as part of kernel launch - this is called the **execution function**
        - Two ways of launching kernels from CPU code:
            1. Triple chevron `<<< >>>` - the most common
            2. cudaLaunchKernelEx
        - Triple Chevron
            ```
            __global__ void vecAdd(float* A, float* B, float* C)
            {
            }

            int main() 
            {
                ...
                // Kernel invocation
                vecAdd<<<1, 256>>>(A, B, C)
                ...
            }
            ```
            - The above launches a single thread block with 256 threads. Each thread will execute the exact same kernel code
            - Limit to number of threads per block as all threads reside in the same SM and must share resources
            - On current GPUs, a thread block may contain up to 1024 threads

        - When using 2 or 3d grids or thread blocks, the CUDA type `dim3` is used as the grid and thread block dimension parameters
        - Example of kernel launch of MatAdd on a 16x16 grid of thread blocks where each block is 8x8
        ```
        int main()
        {
            ...
            dim3 grid(16,16);
            dim3 block(8,8);
            MatAdd<<<grid, block>>>(A, B, C);
            ...
        }
        ```

    c. Thread and Grid Index Intrinsics 
        - In kernel code, CUDA provides intrinics to access params of the execution config and index of a thread or block
            - **threadIdx** gives index of a thread within its thread block (each thread has a different index)
            - **blockDim** gives the dimensions of the thread block, which is specified in execution configuration of the kernel launch
            - **blockIdx** gives index of a thread block in the grid (each thread block has a different index)
            - **gridDim** gives dimensions of the grid, which is specified in the execution configruation at kernel launch
        - **All of the above intrinsics are a 3-component vector with a `.x`, `.y` and `.z` member
            - *threadIdx* and *blockIdx* are indexed to 0
                - threadIdx.x will take values from 0 up to and including *blockDim.x-1* (.y and .z operate the same in their own dims)
        - Returngn to the vecAdd function:
            ```
            __global__ void vecAdd(float* A, float* B, float* C)
            {
                // calculate which element this thread is responsable for computing
                // threadIdx == index of thread in blockIdx
                // blockDim  == dimensions of thread block
                // blockIdx  == index of block in grid
                int workIndex = threadIdx.x + blockDim.x * blockIdx.x
                
                // perform computation
                C[workIndex] = A[workIndex] + B[workIndex];
            }

            int main()
            {
                ...
                // A, B, C are vectors of 1024 elements
                // 4 == number of thread blocks
                // 256 == number of threads
                vecAdd<<<4, 256>>>(A, B, C)
                ...
            }
            ```

            - In the first block in this example, blockIdx.x will be 0 so each threads workIndex will just be its threadIdx.x
            - **This workIndex computation is very common for 1-dimensional parallelizations**
    
    d. Bounds Checking
        - Prior example assumes the length of the vector is some multiple of the block size, 256 threads
        - To make the kernel handle a vector of any len we can add checks that memory access is not exceeding the bounds of the arrays
            ```
                __global__ void vecAdd(float* A, float* B, float* C, int vectorLength)
                {
                    // calculate which element this thread is responsible for computing
                    int workIndex = threadIdx.x + blockDim.x * blockIdx.x

                    if(workIndex < vectorLength)
                    {
                        // perform computation
                        C[workIndex] = A[workIndex] + B[workIndex];
                    }
                }
            ```

            - In the above more threads than needed can be launched without causing out of bounds access
            - When workIndex > vectorLength threads exit and do not do any work (**avoid this but it does not damage performance**)
            - *Key Takeaway* - this kernel can now handle vector lengths that are nto a multiple of block size

            - Number of thread blocks which are needed is calculated as the ceiling of the number of threads needed

            ```
                // vectorLength is an integer storing number of elements in the vector
                int threads = 256;
                int blocks = (vectorLength + threads - 1) / threads;
                vecAdd<<<blocks, threads>>>(devA, devB, devC, vectorLength);
            ```

            - The cuda compute library provides a utility for calculating number of blocks needed at kernel launch
                - cuda::ceil_div which is accessed through <cuda/cmath>

                ```
                    // vectorLength is an integer storing number of elements in a vector
                    int threads = 256;
                    int blocks = cuda::ceil_div(vectorLength, threads);
                    vecAdd<<<blocks, threads>>>(devA, devB, devC, vectorLength);
                ```

3. Memory in GPU computing
    - To use vecAdd above, the vectors A, B, C must be in memory accessible to the GPU. Two ways to do this...
    a. Unified Memory
        - Feature of CUDA runtime that lets NVIDIA driver manageme movement of data between host and device(s)
        - Memory is allocated using *cudaMallocManaged API* or declaring a variable with *__managed__*
        - Complete function to launch vecAdd kernel using unified memory for input/output vectors. *cudaMallocManaged* allocates buffers which can be accessed through GPU or CPU and are released using *cudaFree*

        ```
            void unifiedMemExample(int vectorLength)
            {
                // Pointers to memory vectors
                float* A = nullptr;
                float* B = nullptr;
                float* C = nullptr;
                float* comparisonResult = (float*)malloc(vectorLength*sizeof(float));

                // Use unified memory to allocate buffers
                cudaMallocManaged(&A vectorLength*sizeof(float));
                cudaMallocManaged(&B vectorLength*sizeof(float));
                cudaMallocManaged(&C vectorLength*sizeof(float));

                // Initialize vectors on the host
                initArray(A, vectorLength);
                initArray(B, vectorLength);

                // Launch the kernel. Unified memory will make sure A, B, and C are accessible to the GPU

                int threads = 256;
                int blocks = cuda::ceil_div(vectorLength, threads);
                vecAdd<<<blocks, threads>>>(A,B,C,vectorLength);

                // Wait for the kernel to complete execution
                cudaDeviceSynchronize();

                // Perform computation serially for CPU comparision
                serialVecAdd(A,B,comparisonResult,vectorLength);

                // Confirm CPU and GPU got same answer
                if(vectorApproximatelyEqual(C, comparisonResult, vectorLength)) 
                {
                printf("Unified memory: CPU and GPU answers match\n")
                }
                else
                {
                printf("Unified memory: Error - CPU and GPU answers do not match\n")
                }

                // Clean up
                cudaFree(A);
                cudaFree(B);
                cudaFree(C);
                free(comparisonResult);

            }
        ```


    b. Explicit Memory Management
        - Can help but leads to more verbose code
            - Referenced here: https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html#explicit-memory-management
            - Reference full C++ file code for unified / explicit memory management 
        - CUDA API `cudaMemcpy` is used to copy data from a buffer residing on the CPU to buffer on the GPU
            - `cudaMemcpy` is async - does not return until copy is completed
            - **cudaMemcpyHostToDevice** to copy from CPU to GPU
            - **cudaMemcpyDeviceToHost** to copy from GPU to CPU

    c. Memory Management and Application Performance
        - Explicit memory management provides more control of when data is copied btw host/device, where memory resides and what is allocated where. 
        - When using unified memory there are CUDA APIs that hint to the NVIDIA driver which can help



4. Synchronizing CPU and GPU
    - Kernel launches are async with respect to the CPU thread which called them
        - The control flow of the CPU thread will continue executing before the kernel has completed, possibly before it has launched
        - To guarantee kernel has completed execution before proceding host code, you must sync
    - Best way to sync GPU and host thread is with `cudaDeviceSynchronize` which blocks the host thread until work on GPU is completed


