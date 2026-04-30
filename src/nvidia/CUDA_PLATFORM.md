# The CUDA Platform

1. Compute Capability and Streaming Multiprocessor Versions
    - Every NVIDIA GPU has *Compute Capability (CC)* number identifying which features are supported and hardware needs
        - Denoted in major/minor version number in format of X.Y; ex: 12.0

2. CUDA Toolkit and NVIDIA Driver
    - **NVIDIA Driver** - the operating system of a GPU
        - Must be installed on host system's operating system and is necessary for all GPU uses
        - Version numbers like *r580*
    - **CUDA Toolkit** - set of libraries, headers and tools for writing/building/analyzing software that uses GPU computing
    - **CUDA Runtime** - special case of CUDA toolkit; for API and language ext. for allocating memory, copying data btw CPU/GPU and launchign kernels

3. CUDA Runtime API and CUDA Driver API
    - CUDA runtime API implemented on top of a lower level API called CUDA driver (API exposed by Driver)

4. Parallel Thread Execution (PTX)
    - High level assembly language for NVIDIA GPUs
    - Provides abstraction layer over ISA (instruction set architecture) 
    - Domain Specific Languages and compilers can generate PTX code as an intermediate representation (IR) and use NVIDIAs JIT compilation to produce executable binary GPU code
    - Bersioned similar to CC (reference 1 above)

5. Cubins and Fatbins
    - CUDA apps typically written in language like C++
    - HIgher language compiled to PTX then PTX to real binary for GPU called a CUDA binary or **cubin** for short
        - Cubin has specific binary format for specific SM versions
    - Executibles and library binaries that use GPU have both CPU and GPU code. GPU code is stored in a container called a **fatbin**. Fatbins can contain cubins and PTX for multiple targets. 
        - Ex: app could be built with binaries for multiple GPU architectures (meaning different SM versions)

6. Binary Compatibility
    - Cubin is not backward compatible, only forward compatible within major versions

7. PTX Compatibility
    - GPU code can be stored in executibles binary or PTX form
        - When an app stores the PTX version, that PTX can be JIT compiled at app runtime
        - Forward compatible for say `compute_80` --> `sm_120`

8. Just-In-Time Compilation
    - PTX code loaded by app at runtime is compiled to binary by device driver. This is called **just in time compilation**. 
    - JIT increases load time but allows for benefits from any new compiler improvements
    - Device caches a copy when JIT is called

