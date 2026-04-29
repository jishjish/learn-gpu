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
