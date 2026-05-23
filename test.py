import torch

print(f"torch {torch.__version__}")
print(f"cuda available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
