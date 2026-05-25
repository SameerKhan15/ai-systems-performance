# Lab Setup Instructions
## GPU Model (rented from RunPod)
A100 PCIe, 80GB VRAM, 117GB RAM, 12 vCPU, 1 GPU

## Commands
>which nsys

NSYS already installed if the cmd outputs one of the following outputs
>/usr/local/cuda/bin/nsys
>/opt/nvidia/nsight-systems/nsys

NSYS is not installed if the cmd outputs the following
>nsys not found

### Install Nsight Systems
Install required dependencies
>apt update  
>apt install -y wget gnupg2

>wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb  
>dpkg -i cuda-keyring_1.1-1_all.deb  

>apt update  
>apt install -y nsight-systems-2025.5.2

If the above cmd fails, execute the following
>apt install -y nsight-systems-2025.3.2

Verify nsys exists
>which nsys || true
>find /opt -name nsys -type f 2>/dev/null | head
>find /usr/local -name nsys -type f 2>/dev/null | head

Should see a path like this
>/usr/local/bin/nsys

Test it
>/opt/nvidia/nsight-systems/*/bin/nsys --version

Make nsys available on PATH 
>ln -sf /opt/nvidia/nsight-systems/*/bin/nsys /usr/local/bin/nsys
>nsys --version

>nsys --version
>which nsys

Install the minimal Python stack  
>apt update  
>apt install -y python3-pip  
>pip install --upgrade pip  
>pip install torch torchvision torchaudio  
>pip install transformers datasets accelerate  

Sanity check GPU visibility in PyTorch  
>>python - << 'EOF'  
>>import torch  
>>print("cuda_available:", torch.cuda.is_available())  
>>print("gpu:", torch.cuda.get_device_name(0))  
>>EOF  

Install Text Editor  
>apt update  
>apt install nano  

## Script Execution Command  
nsys profile \  
   --trace=cuda,nvtx,osrt \  
   --output=<output_file_name> \  
   python <script_name>.py  

Command to download the profiles     
scp -P <port> root@<ip_address>:/<file_location> .

# Nsight Profile Vocabulary  
## ampere_sgemm_32x128_tn
-) It is a highly optimized NVIDIA GEMM (matrix multiply) kernel for Ampere GPUs (A100).  
-) In plain English: This kernel is doing the math in your transformer model.  

Break the name apart:  
**ampere**  
-) Optimized specifically for Ampere architecture  
-) Uses Ampere scheduling, memory hierarchy, and Tensor Cores where possible  

**sgemm**  
-) Single-precision General Matrix Multiplication  
-) Mathematically: C=A×B  

**32x128**  
This describes the tile shape used by the kernel:  
-) Each kernel instance operates on a 32 × 128 block of the matrix  
-) Carefully chosen to:  
	-)  Match Ampere SM resources  
	-) Maximize data reuse  
	-) Minimize memory traffic  
-) Mental model: How the GPU slices the big matrix multiply into chunks.  

**_tn**  
This describes matrix layout:  
-) t = matrix A is transposed  
-) n = matrix B is not transposed  

So the operation is:  
C=AT×B  

This is very common in transformers, especially in:  
-) Attention score computation  
-) Linear projections (Q, K, V)	  

# Experiments  
The SEQ_MULT was kept constant at 50. This variable defines the length of the (input) text generator from which the prompt is constructed. It multiplies the text sentence.  
With the token size of the text sentence ~equal to 10, SEQ_MULT value of 50 ensures there is enough text that complete prompts of varying length.  

MAX_LENGTH is the number of tokens per input sequence. 
BATCH is the number of sequences 

Increasing batch size mostly makes kernels wider; increasing sequence length makes kernels both wider and deeper, and introduces new quadratic work (attention).  

The study has two dimensions:  
**- Sequence-scaling study**  
BATCH = 30  
MAX_LENGTH = 64 -> 128 -> 256 -> 512

**Goal**  	  
Small, readable pipeline trace | max_length = 64  
Typical inference | max_length = 128  
Stress attention cost | max_length = 256 / 512  

Varying max_length affects:  
-) Attention complexity -> grows O(seq^2)  
-) Activation sizes → grow linearly  
-) Intermediate tensors → larger and more numerous  
-) Kernel shapes → change significantly  

What we expect to see in Nsight Systems when increasing max_length:  
-) New ampere_sgemm_* variants  
-) Different tile sizes (32x32 → 64x128, etc.)  
-) Different “sliced” strategies  
-) Attention kernels grow superlinearly and attention-related GEMMs start dominating more  

**- Batch-scaling study**  
MAX_LENGTH = 128  
BATCH = 3 → 30 → 300  

What we expect to see in Nsight Systems when increasing max_length:  
-) Matrix sizes grow mostly in the M dimension (rows)  
-) More parallel work per kernel launch  
-) Same number of layers, same attention structure  
-) Longer GEMM kernels, but Kernel types often stay the same and Kernel count per iteration stays similar  
-) CUDA API calls: similar count  
-) GPU utilization increases smoothly  
-) Memory traffic increases linearly  
-) Mental model: Batch size = “more examples in parallel”  
-) This mostly affects throughput, not algorithmic structure.  

Script name:  inference_load_1.py  

# Observations
## Sequence-scaling study  
### HtoD memcpy  
Begins / Ends (+X ms)  
This time progressively increasing from max_length = 64 -> 512  
+7.349 -> +13.905 ms  

**Observation**  
-) Same bytes copied in ALL cases  
-) The inputs["input_ids"].shape output confirms element count is increasing from 64 -> 512  
-) Its not understood why 93.8MB stays the same   
-) ~93.8MB is substantial for “just token IDs + attention mask” unless your batch is big   

I exported HtoD memcpy raw event into a spreadsheet and computed Total Duration and Bytes, for each of these runs. 

**Findings:**  
1. Same transfer pattern:   
   a) One large ~89.42 MB HtoD copy  
   b) Many ~9 MB HtoD copies  
   c) Many ~2.25 MB HtoD copies 
    
   These HtoD copies are framework-internal transfers, not user input tensors.  
   
2. Duration changed (increased) with increasing max_length  
   max_length: 64, bytes: 361 MB, duration: 29.8 msecs  
   max_length: 128, bytes: 361 MB, duration: 28 msecs  
   max_length: 256, bytes: 361 MB, duration: 52.7 msecs  
   max_length: 512, bytes: 361 MB, duration: 53.3 msecs  
   
   Increasing max_length increases kernel execution time and stream contention, which inflates the apparent duration of pageable HtoD memcpy operations even though the transferred bytes remain unchanged.  
   
### GPU Utilization 
**ampere_sgemm_XxY_tn**  
NVDIA matrix multiplication kernel  optimized for A100  
In math terms, its doing C = A^T * B  
 -) ampere: tuned for A100/Ampere architecture  
 -) sgemm: matrix multiply using FP32  
 -) X*Y: a tile/block shape the kernel uses internally (how the GPU chunks the GEMM)  
 -) _tn: A is transposed (t), B is not (n)  
 
A kernel like this is used for linear algebra ops, which is exactly what transformers do:  
-) Q/K/V projections  
-) Attention matmuls  
-) MLP matmuls  
This kernel is one of those “heavy lifters.”  
   
**Notes about different tile shapes in the profiles**  
ampere_sgemm_32x128_tn  
-) Tile shape: 32 × 128  
-) Favors Taller matrices in the N dimension. Often used when one dimension is relatively wide  
-) Common in Attention projections and Some MLP layers when batch × seq is large  

ampere_sgemm_128x64_tn  
-) Tile shape: 128 × 64  
-) Favors larger M dimension (more rows / more parallel work)  
-) Common in larger batched GEMMs and MLP up-projection or down-projection  

Transformers don’t do one matrix multiply — they do many, and their shapes differ:  
-) Q, K, V projections  
-) Attention score computation  
-) Attention output projection  
-) MLP expand (hidden → 4×hidden)  
-) MLP contract (4×hidden → hidden)  

Each of those has different:  
-) M (rows)  
-) N (columns)  
-) K (reduction dimension)  

**Results and Observation** 
Insight#1: GPU total time vs max_length: near-linear scaling  
-) max_length = 64, Total GPU time (s) = 2.64  
-) max_length = 128, Total GPU time (s) = 5.0  
-) max_length = 256, Total GPU time (s) = 9.0  
-) max_length = 512, Total GPU time (s) = 18.75  

Conclusion: Inference workload is compute bound and well-scaled  

Insight#2: Dominant kernels shift as sequence length grows  
-) max_length = 64  
   32x128_tn  ~0.71s  
   128x64_tn  ~0.29s  
   




 
 
