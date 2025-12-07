"""
Training script for HyperSpoof.
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from hyperspoof.models import HyperSpoofModel
from hyperspoof.data import create_cross_dataset_dataloaders, create_single_dataset_dataloaders
from hyperspoof.utils import load_config, get_device, set_seed, setup_logging, save_checkpoint
from hyperspoof.metrics import calculate_metrics
from hyperspoof.training import Trainer


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train HyperSpoof model')
    
    # Data arguments
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--data_root', type=str, required=True, help='Root directory of datasets')
    parser.add_argument('--train_datasets', nargs='+', required=True, help='Training dataset names')
    parser.add_argument('--test_datasets', nargs='+', required=True, help='Test dataset names')
    parser.add_argument('--val_datasets', nargs='+', default=None, help='Validation dataset names')
    
    # Model arguments
    parser.add_argument('--model_name', type=str, default='hyperspoof', help='Model name')
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained weights')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--scheduler', type=str, default='cosine', help='Learning rate scheduler')
    
    # Logging arguments
    parser.add_argument('--log_dir', type=str, default='./logs', help='Log directory')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Save directory')
    parser.add_argument('--experiment_name', type=str, default=None, help='Experiment name')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases')
    
    # Other arguments
    parser.add_argument('--device', type=str, default='auto', help='Device to use')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')
    
    return parser.parse_args()


def create_model(config: Dict[str, Any], device: torch.device) -> HyperSpoofModel:
    """Create HyperSpoof model."""
    model_config = config['model']
    
    model = HyperSpoofModel(
        input_channels=model_config['input_channels'],
        hsi_channels=model_config['hsi_channels'],
        attention_channels=model_config['attention_channels'],
        num_classes=model_config['num_classes'],
        model_name=model_config['classifier'],
        pretrained=model_config['pretrained'],
        dropout_rate=model_config['dropout_rate'],
    )
    
    model = model.to(device)
    return model


def create_optimizer(model: nn.Module, config: Dict[str, Any]) -> optim.Optimizer:
    """Create optimizer."""
    training_config = config['training']
    optimizer_config = config['optimizer']
    
    if optimizer_config['name'] == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=training_config['learning_rate'],
            weight_decay=training_config['weight_decay'],
        )
    elif optimizer_config['name'] == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=training_config['learning_rate'],
            weight_decay=training_config['weight_decay'],
            betas=optimizer_config.get('betas', [0.9, 0.999]),
            eps=optimizer_config.get('eps', 1e-8),
        )
    elif optimizer_config['name'] == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=training_config['learning_rate'],
            weight_decay=training_config['weight_decay'],
            momentum=optimizer_config.get('momentum', 0.9),
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_config['name']}")
    
    return optimizer


def create_scheduler(optimizer: optim.Optimizer, config: Dict[str, Any]) -> Optional[optim.lr_scheduler._LRScheduler]:
    """Create learning rate scheduler."""
    training_config = config['training']
    scheduler_name = training_config.get('scheduler', 'cosine')
    
    if scheduler_name == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=training_config['epochs'],
            eta_min=training_config['learning_rate'] * 0.01
        )
    elif scheduler_name == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=training_config['epochs'] // 3,
            gamma=0.1
        )
    elif scheduler_name == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
    else:
        scheduler = None
    
    return scheduler


def create_loss_function(config: Dict[str, Any]) -> nn.Module:
    """Create loss function."""
    loss_config = config['loss']
    
    if loss_config['name'] == 'cross_entropy':
        loss_fn = nn.CrossEntropyLoss(
            label_smoothing=loss_config.get('label_smoothing', 0.0),
            weight=loss_config.get('class_weights', None)
        )
    elif loss_config['name'] == 'focal':
        # Focal loss implementation would go here
        loss_fn = nn.CrossEntropyLoss()
    else:
        loss_fn = nn.CrossEntropyLoss()
    
    return loss_fn


def main():
    """Main training function."""
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Setup logging
    logger = setup_logging(
        log_dir=args.log_dir,
        experiment_name=args.experiment_name
    )
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    config['training']['epochs'] = args.epochs
    config['data']['batch_size'] = args.batch_size
    config['training']['learning_rate'] = args.learning_rate
    config['training']['weight_decay'] = args.weight_decay
    config['training']['scheduler'] = args.scheduler
    config['data']['num_workers'] = args.num_workers
    
    # Get device
    device = get_device(args.device)
    logger.info(f"Using device: {device}")
    
    # Create model
    model = create_model(config, device)
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create optimizer
    optimizer = create_optimizer(model, config)
    
    # Create scheduler
    scheduler = create_scheduler(optimizer, config)
    
    # Create loss function
    loss_fn = create_loss_function(config)
    
    # Create data loaders
    # This is a simplified version - you would need to implement proper dataset configuration
    logger.info("Creating data loaders...")
    # train_loader, val_loader, test_loader = create_data_loaders(args, config)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        device=device,
        config=config,
        logger=logger
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Train model
    logger.info("Starting training...")
    trainer.train()
    
    logger.info("Training completed!")


if __name__ == "__main__":
    main()
