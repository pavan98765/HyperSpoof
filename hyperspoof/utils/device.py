"""
Device and seed management utilities.
"""

import torch
import random
import numpy as np
from typing import Union, Optional


def get_device(device: Union[str, torch.device] = "auto") -> torch.device:
    """
    Get the appropriate device for computation.
    
    Args:
        device: Device specification ('auto', 'cpu', 'cuda', etc.)
        
    Returns:
        PyTorch device
    """
    if device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device)
    
    return device


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device_info() -> dict:
    """
    Get information about available devices.
    
    Returns:
        Dictionary containing device information
    """
    info = {
        'cuda_available': torch.cuda.is_available(),
        'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'current_device': str(torch.cuda.current_device()) if torch.cuda.is_available() else None,
    }
    
    if torch.cuda.is_available():
        info['cuda_devices'] = []
        for i in range(torch.cuda.device_count()):
            device_info = {
                'device_id': i,
                'name': torch.cuda.get_device_name(i),
                'memory_total': torch.cuda.get_device_properties(i).total_memory,
                'memory_allocated': torch.cuda.memory_allocated(i),
                'memory_cached': torch.cuda.memory_reserved(i),
            }
            info['cuda_devices'].append(device_info)
    
    return info


def move_to_device(tensor_or_module: Union[torch.Tensor, torch.nn.Module], device: torch.device) -> Union[torch.Tensor, torch.nn.Module]:
    """
    Move tensor or module to specified device.
    
    Args:
        tensor_or_module: Tensor or module to move
        device: Target device
        
    Returns:
        Moved tensor or module
    """
    return tensor_or_module.to(device)


def get_memory_usage(device: Optional[torch.device] = None) -> dict:
    """
    Get memory usage information.
    
    Args:
        device: Device to check (uses current device if None)
        
    Returns:
        Dictionary containing memory usage information
    """
    if device is None:
        device = torch.cuda.current_device() if torch.cuda.is_available() else None
    
    if device is None or not torch.cuda.is_available():
        return {
            'device': 'cpu',
            'memory_allocated': 0,
            'memory_cached': 0,
            'memory_total': 0,
        }
    
    return {
        'device': str(device),
        'memory_allocated': torch.cuda.memory_allocated(device),
        'memory_cached': torch.cuda.memory_reserved(device),
        'memory_total': torch.cuda.get_device_properties(device).total_memory,
    }


def clear_cache(device: Optional[torch.device] = None) -> None:
    """
    Clear GPU cache.
    
    Args:
        device: Device to clear (uses current device if None)
    """
    if torch.cuda.is_available():
        if device is None:
            torch.cuda.empty_cache()
        else:
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
