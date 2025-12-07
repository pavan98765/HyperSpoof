"""
Example script for training HyperSpoof model.
"""

import os
import sys
import torch
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from hyperspoof.models import HyperSpoofModel
from hyperspoof.data import create_cross_dataset_dataloaders, create_single_dataset_dataloaders
from hyperspoof.utils import load_config, get_device, set_seed, setup_logging
from hyperspoof.training import Trainer
from hyperspoof.metrics import calculate_metrics


def main():
    """Main training example."""
    
    # Set random seed
    set_seed(42)
    
    # Setup logging
    logger = setup_logging(experiment_name="hyperspoof_example")
    
    # Load configuration
    config_path = "configs/default.yaml"
    config = load_config(config_path)
    
    # Get device
    device = get_device(config['device'])
    logger.info(f"Using device: {device}")
    
    # Create model
    model = HyperSpoofModel(
        input_channels=config['model']['input_channels'],
        hsi_channels=config['model']['hsi_channels'],
        attention_channels=config['model']['attention_channels'],
        num_classes=config['model']['num_classes'],
        model_name=config['model']['classifier'],
        pretrained=config['model']['pretrained'],
        dropout_rate=config['model']['dropout_rate'],
    )
    
    model = model.to(device)
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create optimizer
    import torch.optim as optim
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Create scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['epochs'],
        eta_min=config['training']['learning_rate'] * 0.01
    )
    
    # Create loss function
    loss_fn = torch.nn.CrossEntropyLoss(
        label_smoothing=config['loss']['label_smoothing']
    )
    
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
    
    # Example dataset configuration (you would need to adapt this to your data)
    train_configs = [
        {
            'root_dir': '/path/to/dataset',
            'category': 'real',
            'label': 0,
            'name': 'real_train'
        },
        {
            'root_dir': '/path/to/dataset',
            'category': 'attack_print1',
            'label': 1,
            'name': 'attack_train'
        }
    ]
    
    test_configs = [
        {
            'root_dir': '/path/to/dataset',
            'category': 'real',
            'label': 0,
            'name': 'real_test'
        },
        {
            'root_dir': '/path/to/dataset',
            'category': 'attack_print2',
            'label': 1,
            'name': 'attack_test'
        }
    ]
    
    # Create data loaders
    logger.info("Creating data loaders...")
    # train_loader, val_loader, test_loader, dataset_info = create_cross_dataset_dataloaders(
    #     train_configs=train_configs,
    #     test_configs=test_configs,
    #     batch_size=config['data']['batch_size'],
    #     num_workers=config['data']['num_workers'],
    #     augmentation=config['data']['augmentation'],
    #     augmentation_strength=config['data']['augmentation_strength']
    # )
    
    # Train model
    logger.info("Starting training...")
    # history = trainer.train(
    #     train_loader=train_loader,
    #     val_loader=val_loader,
    #     num_epochs=config['training']['epochs']
    # )
    
    # Evaluate model
    logger.info("Evaluating model...")
    # test_metrics = trainer.evaluate(test_loader)
    
    logger.info("Training example completed!")


if __name__ == "__main__":
    main()
