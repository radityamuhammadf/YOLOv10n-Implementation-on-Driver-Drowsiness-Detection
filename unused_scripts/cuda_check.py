import torch
import subprocess
import re

def get_tegrastats_output():
    try:
        # Run tegrastats for a short time (e.g., 1 second) to gather output
        result = subprocess.run(['tegrastats', '--interval', '10'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=2)
        return result.stdout
    except subprocess.TimeoutExpired:
        return "Timeout expired"
    except Exception as e:
        return str(e)

def parse_tegrastats_output(output):
    # Example tegrastats output format:
    # RAM 1942/3963MB (lfb 214x4MB) SWAP 2/1982MB (cached 0MB) CPU [0%@102,off,off,off] EMC_FREQ 0% GR3D_FREQ 0%
    match = re.search(r'GR3D_FREQ (\d+)%', output)
    if match:
        return int(match.group(1))
    return None

print("Is CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
    print("Number of GPUs:", torch.cuda.device_count())

# Basic CUDA tensor operations
x = torch.randn(3, 3)
if torch.cuda.is_available():
    x = x.cuda()
    y = x * x
    print(y)
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Memory Allocated: {torch.cuda.memory_allocated(i)}")
        print(f"  Memory Cached: {torch.cuda.memory_reserved(i)}")
        print(f"  CUDA Utilized: {torch.cuda.utilization(i)}")

    
    # Get and display GPU utilization from tegrastats
    tegrastats_output = get_tegrastats_output()
    print("Tegrastats output:", tegrastats_output)  # For debugging
    gpu_utilization = parse_tegrastats_output(tegrastats_output)
    if gpu_utilization is not None:
        print(f"GPU Utilization: {gpu_utilization}%")
    else:
        print("Failed to parse GPU utilization from tegrastats output.")
