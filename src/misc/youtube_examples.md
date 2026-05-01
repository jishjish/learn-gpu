
[30 Min CUDA](https://www.youtube.com/watch?v=xewKxorikwE)

- measure perf in gflops: giga floating point operations per second (billions of calcs/sec). standard for performance computing
- kitchen analogy
    - cpu is like a master chef, world class at *sequential* things
    - gpu does one specific task well, like 10_000 junior carrot chopping chefs
- The CUDA hierarchy
    1. **Thread** - single junior chef running the code
    2. **Block**  - team of chefs at a "cooking station" (they can operate / share info quickly)
    3. **Grid***  - entire collection of blocks (the whole "kitchen")
- CUDA is about Memory Access Patterns (kitchen analogy)

    | Memory Type    | Kitchen Analogy  | Who Can Access?           | Size          | Latency (the 'walk') 
    | -------------  | ---------------- | ------------------------  | ------------  | ---------------------------
    | Global Memory  | Main Pantry      | All threads in Grid       | 80GB          | VERY SLOW (~400-800 cycles)
    | Shared Memory  | Shared Workbench | Threads within ONE Block  | 164KB per SM  | ULTRA FAST (~20-30 cycles)
    | Registers      | Chef's Hands     | A single Thread           | 256 KB per SM | INSTANT (~1 cycle)

- Optimization strategy
    1. Minimize global memory Access
    2. Load data into shared memory
    3. Reuse cached data for computations

- CUDA's *hello world* equivalent
    - Given two arrays (A, B), compute a third array
        - C[i] = A[i] + B[i]
    - Plan: assign one GPU thread to cauculate one element of C. A million elements would mean we launch a million threads

- How it all works:
    - **Device Code** 
        - code that runs on the GPU
        - executed by thousands of threads
        - the "recipe" for the junior chefs
    - **Host Code**
        - code that runs on the cpu
        - Manages the whole operation
        - the "project manager"
    - How each of the thousands of threads figures out its unique id:
        - int i = blockIdx.x * blockDim.x + threadIdx.x
            - **blockIdx.x**   - "Which team (block) am i in?"
            - **blockDim.x**   - "How many threads are in each team?"
            - **threadIdx.x**  - "Whats my personal ID within my team?"
        - For *Thread #12* in *Team 5*, with 256 threads per team... the unique global index..
            - i = 5 * 256 + 12 = 1_292
            - This thread now knows its one job to compute - c[1292]
    - **Every CUDA kernel follows this 5 step Plan**
        1. Allocate memory on the GPU 
            a. CPU RAM and GPU VRAM are separate, you must explicitly allocate GPU memory
            ```
            int *dev_a, *dev_b, dev_c;
            cudaMalloc(&dev_a, bytes);
            cudaMalloc(&dev_b, bytes);
            cudaMalloc(&dev_c, bytes);
            ```
        2. Copy data from host to device
            a. Copy input arrays from CPU RAM to GPU memory we just allocated
            b. Be cognizant - cudaMemcpy can KILL PERFORMANCE 
            ```
            cudaMemcpy(dev_a, h_a, bytes, cudaMemcpyHostToDevice)
            cudaMemcpy(dev_b, h_b, bytes, cudaMemcpyHostToDevice)
            ```

            | Conn. Type      | Kitchen Analogy     | Speed(Approx.)  |
            | --------------  | ------------------- | --------------- |
            | PCIe 4.0 Bus    | Highway to kitchen  | ~32 GB/s        | 
            | GPU Global Mem  | Fast walk to pantry | ~2,000 GB/s     | 
            | GPU Shared Mem  | Grab from workbench | ~19,000 GB/s    |
            

