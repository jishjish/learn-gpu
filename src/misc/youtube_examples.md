
[30 Min CUDA](https://www.youtube.com/watch?v=xewKxorikwE)

- measure perf in gflops: giga floating point operations per second (billions of calcs/sec). standard for performance computing
- kitchen analogy
    - cpu is like a master chef, world class at *sequential* things
    - gpu does one specific task well, like 10_000 junior carrot chopping chefs
- The CUDA hierarchy
    1. **Thread** - single junior chef running the code
    2. **Block**  - team of chefs at a "cooking station" (they can operate / share info quickly)
    3. **Grid***  - entire collection of blocks (the whole "kitchen")
- CUDA is about Memory Access Patterns
    - global memory (performance villain) - huge, main pantry - huge but far away, long slow walk
    - shared memory (performance hero) - tiny but next to chefs

    | Memory Type    | Kitchen Analogy  | Who Can Access?           | Size          | Latency (the 'walk') 
    | -------------  | ---------------- | ------------------------  | ------------  | ---------------------------
    | Global Memory  | Main Pantry      | All threads in Grid       | 80GB          | VERY SLOW (~400-800 cycles)
    | Shared Memory  | Shared Workbench | Threads within ONE Block  | 164KB per SM  | ULTRA FAST (~20-30 cycles)
    | Registers      | Chef's Hands     | A single Thread           | 256 KB per SM | INSTANT (~1 cycle)



