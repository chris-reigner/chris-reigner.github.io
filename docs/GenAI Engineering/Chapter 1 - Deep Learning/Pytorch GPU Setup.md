----

description: Useful tips for optimizing Pytorch runs
----

# When to move from CPU to GPU

With the rise of Large Models (Language, Vision, Speech...) it becomes absolutely necessary to swtich from CPU to GPU.
While some "small" embedding models (say 1 billion parameters) can run on CPU, some models require GPU even without training and fine-tuning.
We will see later some techniques to optimize (i.e. reduce) the use of GPU.

However, in order to run a lot of components in this repository, it is better to setup GPU usage as early as possible.
It will be just faster to run some code :).

# How to test your current setup

For nvidia installation and pytorch setup you will need to install

```python
$pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

To test your current setup

```python
import torch
import os

def display_cuda_info():
    """Display comprehensive CUDA and GPU information"""
    print(f"CUDA is available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Current GPU ID: {torch.cuda.current_device()}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        
        # Display information for each GPU
        for i in range(torch.cuda.device_count()):
            print(f"\nGPU {i} info:")
            print(f"Name: {torch.cuda.get_device_name(i)}")
            print(f"Memory allocated: {torch.cuda.memory_allocated(i) / 1e9:.2f} GB")
            print(f"Memory cached: {torch.cuda.memory_reserved(i) / 1e9:.2f} GB")

def switch_gpu(gpu_id):
    """Switch to specified GPU"""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    
    if gpu_id >= torch.cuda.device_count():
        raise ValueError(f"GPU {gpu_id} not found. Available GPUs: 0-{torch.cuda.device_count()-1}")
    
    # Set environment variable
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # Set PyTorch default device
    torch.cuda.set_device(gpu_id)
    
    print(f"Switched to GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")

def move_model_to_gpu(model, gpu_id):
    """Move a PyTorch model to specified GPU"""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    
    switch_gpu(gpu_id)
    return model.cuda()
```

You can test with a small pytorch model:

```python
# Test with a dumb model
model = torch.nn.Linear(10, 10)  # Simple example model
try:
    model = move_model_to_gpu(model, 0)  # Move to GPU 0
    print("Model successfully moved to GPU")
except Exception as e:
    print(f"Error moving model to GPU: {e}")
```

Alternatively, you can display nvidia information using the command:

```bash
!nvidia-smi
```

## Get your hands dirty

<https://elanapearl.github.io/blog/2025/the-bug-that-taught-me-pytorch/>
