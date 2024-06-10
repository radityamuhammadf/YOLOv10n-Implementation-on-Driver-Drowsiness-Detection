import torch

print("Is CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
    print("Number of GPUs:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Memory Allocated: {torch.cuda.memory_allocated(i)}")
        print(f"  Memory Cached: {torch.cuda.memory_reserved(i)}")

# Basic CUDA tensor operations
x = torch.randn(3, 3)
if torch.cuda.is_available():
    x = x.cuda()
    y = x * x
    print(y)
