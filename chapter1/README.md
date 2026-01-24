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
>/opt/nvidia/nsight-systems/.../bin/nsys

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
