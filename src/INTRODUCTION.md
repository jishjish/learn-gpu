# Introduction
1. The Graphics Processing Unit
    - Originated as fixed-function hardware to accelerate parallel computation for 3d rendering
    - NVIDIA created **Compute Unified Device Architecture** (CUDA) to use GPUs independent of graphics APIs
2. Benefits of GPUs
    - Provides much higher throughput and memory bandwidth than a comparable CPU 
    - GPUs and CPUs are designed with different goals in mind:
        - **CPUs are designed** to execute a serial sequence of operations (thread), can operate *tens* of threads in parallel
        - **GPUs are designed** to execute thousands of threads in parallel; trade off is lower single thread performance 
    - GPUs devote more transistors to data processing units; CPUs devote more to data caching and flow control
3. Getting Started 
    - Libraries like cuBLAS, cuFFT, cuDNN, and CUTLASS are examples of libraries that help avoid reimplementing established algos
    - Many ML frameworks are also optimized for GPU acceleration
    - DSLs (Domain Specific Languages) like NVIDIA's Warp compile directly to run on the CUDA platform
        - This provides an even higher level of programming than say, C++

# The Programming Model
1. Heterogenous Systems
    - CUDA assumes a heterogenous computing system, meaning there are CPUs *and* GPUs
        - CPU and the memory associated with it are the **host** and **host memory**
        - GPU and the memory associated with it are the **device** and **device memory**
    - CUDA applications run some of their code on GPU but always start on the CPU
        - CPU and GPU computations can run simultaneously - best performance often found by maximizing both
    - The code an application runs on the GPU is called **device code**
        - A function invoked for execution on a GPU is called a **kernel**
        - The act of starting a kernel running is *launching the kernel*
            - Can be thought of as launching many threads executing the kernel code in parallel on the GPU

2. GPU Hardware Model
    - Like any programmng system CUDA relies on a conceptual model of the hardware
    - In CUDA, a GPU is considered a collection of **Streaming Multiprocessors (SMs)** which are organized into a group of **Graphics Processing Clusters (GPCs)**
    - Contents of a Streaming Multiprocessors   
        - Local register file
        - Unified data cache - provides physical resources for shared memory / L1 cache
        - Functional units to perform computations
    - CPU and GPU is connected through *PCIe* or *NVLINK*

3. Thread Blocks and Grids
    **GPU Hierarchy**
    1. threads
    2. warps
    3. blocks
    4. grids
    5. SM (streaming multiprocessor)

    - When an app launches, it does so with millions of threads. These threads are organized into *blocks* (aka thread blocks).
    - Thread blocks are organized onto a grid and have same size/dimensions. 
        - A kernel runs with specified *execution configuration* which sets the grid and thread block dimensions.
        - With built-in variables a thread running the kernel can:
            - Determine its location within the containing block and the containing block within the *containing grid*
            - Determine the dimensions of the thread block and grid
            - ==This gives each thread a unique identity; often used to determine what data/operations it is responsible for==
        - No scheduling guarantee btw thread blocks so they cannot rely on results from other blocks
        - No guarantee about the order which thread blocks from a grid are assigned to SMs
    - Under CUDA, there are no data dependencies btw threads in different blocks; must be able to execute blocks in any order

    1. Thread Block Clusters    
        - GPUs with compute capability >=9.0 have optional level of grouping called *clusters*
        - This can allow for communication at the cluster level; handled through **Cooperative Groups**
        - Threads in clusters can access shared memory - **distributed shared memory**
        - Thread blocks in a cluster are always adjacent to each other

4. Warps and SIMT
    - Within a thread block, threads are organized into groups of 32 called *warps*
    - A warp executes the kernel code in a **Single-Instruction Multiple-Threads (SIMT)** paradigm
        - Every thread is executing same kernel code, but may follow different branches through the code
    - When threads are executed by a warp they are assigned a *warp lane* and ordered 0 to 31
    - Non-executing threads in the warp (say through conditional logic), are masked 
    - Utilization of the GPU is maximized when threads in a warp follow the same control flow path

    ### Threads of even index execute, others are masked
        ```
            if(thread.Idx.x%2 == 0)
                {a = r(t);}
            else
                {a = q(t);}
            y = f(a);
        ```
    - Thread blocks are best specified to have total number of threads divisible by 32 (warp count)

4. GPU Memory
    a. DRAM Memory in Heterogenous Systems
        - Both CPU and GPU have directly attached DRAM chips
        - DRAM attached to GPU == *global memory* (accessible to all SMs in the GPU)... CPU DRAM == *host memory*
        - Both CPU/GPU use virtual memory addressing
        - Unified memory allows the placement of memory to be handled automatically by CUDA runtime
    b. On-Chip Memory in GPUs
        - In addition to global memory each GPU has some on-chip memory 
        - Each SM has its own register file (stores thread local variables)
        - To schedule thread block to SM:
            - total register * threads in block <= available registers
        - Shared memory allocations are done at the block level
    c. Caches
        - In addition to programmable memories, GPUs have L1 and L2 caches
        - Each SM has an L1 which is part of the unified data cache. A larger L2 is shared by all SMs in a GPU
    d. Unified Memory
        - When apps allocate memory explicitly on CPU/GPU, that memory is only accessible to code on that device
            - CPU memory can only be accessed from CPU and GPU from the kernels on the GPU
        - CUDA feature, **unified memory** allows for apps to make memory allocations across CPU and GPU


