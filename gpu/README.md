                        GPU
        ┌───────────────────────────────────┐
        │                                   │
        │   ┌──────── SM 0 ────────┐        │
        │   │                      │        │
        │   │  THREAD EXECUTION    │        │
        │   │                      │        │
        │   │  up to 64 warps      │        │
        │   │       ×              │        │
        │   │  32 threads/warp     │        │
        │   │       =              │        │
        │   │  2,048 resident      │        │
        │   │  threads             │        │
        │   │                      │        │
        │   │  ┌────────────────┐  │        │
        │   │  │ Warp 0         │  │        │
        │   │  │ 32 threads     │  │        │
        │   │  ├────────────────┤  │        │
        │   │  │ Warp 1         │  │        │
        │   │  │ 32 threads     │  │        │
        │   │  ├────────────────┤  │        │
        │   │  │ ...            │  │        │
        │   │  ├────────────────┤  │        │
        │   │  │ Warp 63        │  │        │
        │   │  │ 32 threads     │  │        │
        │   │  └────────────────┘  │        │
        │   │                      │        │
        │   │  SM RESOURCES        │        │
        │   │                      │        │
        │   │  Registers           │        │
        │   │  64K × 32-bit        │        │
        │   │  = 256 KB            │        │
        │   │                      │        │
        │   │  L1 / Shared SRAM    │        │
        │   │  256 KB combined     │        │
        │   │                      │        │
        │   │  Shared memory can   │        │
        │   │  use up to ~228 KB   │        │
        │   │  (~227 KB usable)    │        │
        │   │                      │        │
        │   └──────────────────────┘        │
        │                                   │
        │   ┌──────── SM 1 ────────┐        │
        │   │    same resources    │        │
        │   └──────────────────────┘        │
        │                                   │
        │              ...                  │
        │                                   │
        │   ┌──────── SM N ────────┐        │
        │   │    same resources    │        │
        │   └──────────────────────┘        │
        └───────────────────────────────────┘

The most important idea is that those numbers are per SM, not for the whole GPU.  
Think of an SM as a little throughput-processing factory:  

                    ONE SM
                      │
          ┌───────────┴────────────┐
          │                        │
      COMPUTE STATE            FAST MEMORY
          │                        │
     64 warps max          ┌───────┴────────┐
     2048 threads          │                │
                           │                │
                       Registers       L1 / Shared
                        256 KB           256 KB
Each warp is a 32-threads group.

````
Thread Blocks
     │
     ▼
assigned to an SM
     │
     ├── consume THREAD slots
     ├── consume WARP slots
     ├── consume REGISTERS
     └── consume SHARED MEMORY
                │
                ▼
        How many blocks can
        live on the SM at once?
                │
                ▼
            OCCUPANCY
````
For example, suppose our kernel launches a block of 256 threads:  
````
256 threads/block
      ÷
32 threads/warp
      =
8 warps/block
````
Ignoring other constraints, a 64-warp SM could theoretically hold:  
````
64 warps / 8 warps per block
        =
8 resident blocks
````
But now suppose each block asks for a huge amount of shared memory:  
````
~227 KB shared memory available per SM

Block requires 110 KB
        ↓
Only ~2 such blocks fit
        ↓
2 blocks × 8 warps
        =
16 resident warps
````
So although the hardware could track far more warps, shared-memory usage became the limiting resource.  

**Registers + shared memory + large warp/thread capacity allow an SM to keep many threads resident simultaneously, which is what enables GPU thread-level parallelism.**  

Resident warps are the pool of work, and the four warp schedulers decide which ready warps feed the execution pipelines on a given cycle.  NVIDIA’s current Blackwell profiling documentation explicitly refers to the four warp schedulers.  

**Resident work**  
````
up to 64 warps
waiting / ready / stalled
````
**Warp schedulers**  
````
4
independent schedulers
````
**Conceptual max**  
````
4 × up to 2
warp instructions / cycle*
````

Cycle N: schedulers select ready warps  
````
Pool of resident warps
Warp 0
Warp 1
Warp 2
...
Warp 63
Some are ready; others may be waiting on data or dependencies.
````
Each scheduler chooses a ready warp from its scheduling domain  
````
Scheduler 0
selects Warp 7
ADD + LOAD
independent → dual issue
````
````
Scheduler 1
selects Warp 18
FMA
single issue
````
````
Scheduler 2
selects Warp 31
INT + STORE
independent → dual issue
````
````
Scheduler 3
selects Warp 52
LOAD
single issue
````
Instructions flow into different execution pipelines  
````
FP / INT
arithmetic
````
````
Load / Store
memory
````
````
Tensor
matrix operations
````
````
Specialized
other pipelines
````
````
64 resident warps
lots of available thread-level parallelism
|
4 warp schedulers
choose ready warps each cycle
|
Instruction issue
possibly two independent instructions from a selected warp
|
Execution pipelines
arithmetic · memory · tensor · specialized
````
 