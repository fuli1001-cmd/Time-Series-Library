import os
import torch

def cpu():
    """Get the CPU device.

    Defined in :numref:`sec_use_gpu`"""
    return torch.device('cpu')

def gpu(i=0):
    """Get a GPU device.

    Defined in :numref:`sec_use_gpu`"""
    return torch.device(f'cuda:{i}')

def num_gpus():
    """Get the number of available GPUs.

    Defined in :numref:`sec_use_gpu`"""
    return torch.cuda.device_count()

def try_gpu(i=0):
    """Return gpu(i) if exists, otherwise return cpu().

    Defined in :numref:`sec_use_gpu`"""
    if num_gpus() >= i + 1:
        return gpu(i)
    return cpu()

def try_all_gpus():
    """Return all available GPUs, or [cpu(),] if no GPU exists.

    Defined in :numref:`sec_use_gpu`"""
    return [gpu(i) for i in range(num_gpus())]

def can_use_cudf():
    '''
    to use cuDF, A GPU with NVIDIA Volta™ (Compute Capability 7.0) or newer architecture is required
    '''
    try:
        # Windows can't use cuDF
        if os.name == "nt":
            return False
    
        if not torch.cuda.is_available():
            return False
        
        if torch.cuda.device_count() == 0:
            return False
        
        gpu_capability = float(f"{torch.cuda.get_device_properties(0).major}.{torch.cuda.get_device_properties(0).minor}")
        return gpu_capability >= 7.0
    except Exception as e:
        print(f"Error checking GPU information: {e}")
        return False
    
def is_compile_supported():
    # 只支持 CUDA 7.0 及以上
    if not torch.cuda.is_available():
        return False
    major = torch.cuda.get_device_capability()[0]
    return major >= 7