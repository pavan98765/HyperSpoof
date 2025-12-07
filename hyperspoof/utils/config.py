"""
Configuration management utilities.
"""

import os
import yaml
import json
from typing import Dict, Any, Optional
from pathlib import Path


def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration for HyperSpoof.
    
    Returns:
        Dictionary containing default configuration
    """
    return {
        'model': {
            'name': 'hyperspoof',
            'input_channels': 3,
            'hsi_channels': 31,
            'attention_channels': 3,
            'num_classes': 2,
            'classifier': 'efficientnet_b0',
            'pretrained': True,
            'dropout_rate': 0.2,
        },
        'data': {
            'image_size': [256, 256],
            'batch_size': 32,
            'num_workers': 4,
            'pin_memory': True,
            'augmentation': True,
            'augmentation_strength': 'medium',
            'normalize': True,
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
        },
        'training': {
            'epochs': 100,
            'learning_rate': 0.001,
            'weight_decay': 1e-4,
            'scheduler': 'cosine',
            'warmup_epochs': 5,
            'gradient_clip': 1.0,
            'early_stopping_patience': 10,
            'save_best_only': True,
        },
        'optimizer': {
            'name': 'adamw',
            'betas': [0.9, 0.999],
            'eps': 1e-8,
        },
        'loss': {
            'name': 'cross_entropy',
            'label_smoothing': 0.1,
            'class_weights': None,
        },
        'logging': {
            'log_dir': './logs',
            'save_dir': './checkpoints',
            'log_interval': 10,
            'save_interval': 5,
            'use_wandb': False,
            'wandb_project': 'hyperspoof',
        },
        'evaluation': {
            'metrics': ['accuracy', 'precision', 'recall', 'f1', 'auc', 'acer', 'hter', 'eer'],
            'tta': False,
            'tta_transforms': 4,
        },
        'device': 'auto',  # 'auto', 'cpu', 'cuda', 'cuda:0', etc.
        'seed': 42,
    }


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Dictionary containing configuration
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    # Load configuration based on file extension
    if config_path.suffix.lower() == '.yaml' or config_path.suffix.lower() == '.yml':
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    elif config_path.suffix.lower() == '.json':
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")
    
    # Merge with default configuration
    default_config = get_default_config()
    config = _merge_configs(default_config, config)
    
    return config


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """
    Save configuration to file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save configuration based on file extension
    if config_path.suffix.lower() == '.yaml' or config_path.suffix.lower() == '.yml':
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
    elif config_path.suffix.lower() == '.json':
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
    else:
        raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")


def _merge_configs(default: Dict[str, Any], user: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge user configuration with default configuration.
    
    Args:
        default: Default configuration
        user: User configuration
        
    Returns:
        Merged configuration
    """
    result = default.copy()
    
    for key, value in user.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge_configs(result[key], value)
        else:
            result[key] = value
    
    return result


def create_experiment_config(
    experiment_name: str,
    base_config: Optional[Dict[str, Any]] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Create configuration for a specific experiment.
    
    Args:
        experiment_name: Name of the experiment
        base_config: Base configuration (uses default if None)
        overrides: Configuration overrides
        
    Returns:
        Experiment configuration
    """
    if base_config is None:
        config = get_default_config()
    else:
        config = base_config.copy()
    
    # Add experiment name
    config['experiment'] = {
        'name': experiment_name,
        'timestamp': None,  # Will be set when saving
    }
    
    # Apply overrides
    if overrides:
        config = _merge_configs(config, overrides)
    
    return config


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration dictionary.
    
    Args:
        config: Configuration to validate
        
    Returns:
        True if configuration is valid
        
    Raises:
        ValueError: If configuration is invalid
    """
    required_keys = ['model', 'data', 'training', 'optimizer', 'loss']
    
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required configuration key: {key}")
    
    # Validate model configuration
    model_config = config['model']
    required_model_keys = ['name', 'input_channels', 'num_classes']
    for key in required_model_keys:
        if key not in model_config:
            raise ValueError(f"Missing required model configuration key: {key}")
    
    # Validate data configuration
    data_config = config['data']
    required_data_keys = ['batch_size', 'num_workers']
    for key in required_data_keys:
        if key not in data_config:
            raise ValueError(f"Missing required data configuration key: {key}")
    
    # Validate training configuration
    training_config = config['training']
    required_training_keys = ['epochs', 'learning_rate']
    for key in required_training_keys:
        if key not in training_config:
            raise ValueError(f"Missing required training configuration key: {key}")
    
    return True
