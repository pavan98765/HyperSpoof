"""
Prediction script for HyperSpoof.
"""

import argparse
import torch
import numpy as np
from PIL import Image
import os
import sys
from typing import List, Dict, Any

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from hyperspoof.models import HyperSpoofModel
from hyperspoof.utils import load_config, get_device, load_checkpoint
from hyperspoof.data.transforms import get_test_transforms


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Predict with HyperSpoof model')
    
    # Model arguments
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    
    # Input arguments
    parser.add_argument('--input', type=str, required=True, help='Path to input image or directory')
    parser.add_argument('--output', type=str, default=None, help='Path to save predictions')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for prediction')
    
    # Other arguments
    parser.add_argument('--device', type=str, default='auto', help='Device to use')
    parser.add_argument('--confidence_threshold', type=float, default=0.5, help='Confidence threshold for predictions')
    
    return parser.parse_args()


def load_image(image_path: str) -> Image.Image:
    """
    Load and preprocess image.
    
    Args:
        image_path: Path to image file
        
    Returns:
        PIL Image
    """
    try:
        image = Image.open(image_path).convert('RGB')
        return image
    except Exception as e:
        raise ValueError(f"Error loading image {image_path}: {e}")


def preprocess_image(image: Image.Image, transform) -> torch.Tensor:
    """
    Preprocess image for model input.
    
    Args:
        image: PIL Image
        transform: Image transform
        
    Returns:
        Preprocessed tensor
    """
    if transform:
        return transform(image)
    else:
        # Default preprocessing
        import torchvision.transforms as transforms
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transform(image)


def predict_single_image(
    model: HyperSpoofModel,
    image_path: str,
    transform,
    device: torch.device,
    confidence_threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Predict on a single image.
    
    Args:
        model: Trained model
        image_path: Path to image
        transform: Image transform
        device: Device to use
        confidence_threshold: Confidence threshold
        
    Returns:
        Dictionary containing prediction results
    """
    # Load and preprocess image
    image = load_image(image_path)
    image_tensor = preprocess_image(image, transform)
    image_tensor = image_tensor.unsqueeze(0).to(device)  # Add batch dimension
    
    # Make prediction
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = outputs['probabilities']
        prediction = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, prediction].item()
    
    # Determine if prediction meets confidence threshold
    is_confident = confidence >= confidence_threshold
    
    # Get class name
    class_names = ['Real', 'Spoof']
    class_name = class_names[prediction]
    
    return {
        'image_path': image_path,
        'prediction': prediction,
        'class_name': class_name,
        'confidence': confidence,
        'is_confident': is_confident,
        'probabilities': {
            'real': probabilities[0, 0].item(),
            'spoof': probabilities[0, 1].item()
        }
    }


def predict_batch(
    model: HyperSpoofModel,
    image_paths: List[str],
    transform,
    device: torch.device,
    batch_size: int = 1,
    confidence_threshold: float = 0.5
) -> List[Dict[str, Any]]:
    """
    Predict on a batch of images.
    
    Args:
        model: Trained model
        image_paths: List of image paths
        transform: Image transform
        device: Device to use
        batch_size: Batch size
        confidence_threshold: Confidence threshold
        
    Returns:
        List of prediction results
    """
    results = []
    
    # Process images in batches
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch_images = []
        
        # Load and preprocess batch
        for path in batch_paths:
            image = load_image(path)
            image_tensor = preprocess_image(image, transform)
            batch_images.append(image_tensor)
        
        # Stack into batch tensor
        batch_tensor = torch.stack(batch_images).to(device)
        
        # Make predictions
        model.eval()
        with torch.no_grad():
            outputs = model(batch_tensor)
            probabilities = outputs['probabilities']
            predictions = torch.argmax(probabilities, dim=1)
            confidences = torch.max(probabilities, dim=1)[0]
        
        # Process results
        for j, (path, pred, conf) in enumerate(zip(batch_paths, predictions, confidences)):
            prediction = pred.item()
            confidence = conf.item()
            is_confident = confidence >= confidence_threshold
            
            class_names = ['Real', 'Spoof']
            class_name = class_names[prediction]
            
            results.append({
                'image_path': path,
                'prediction': prediction,
                'class_name': class_name,
                'confidence': confidence,
                'is_confident': is_confident,
                'probabilities': {
                    'real': probabilities[j, 0].item(),
                    'spoof': probabilities[j, 1].item()
                }
            })
    
    return results


def get_image_paths(input_path: str) -> List[str]:
    """
    Get list of image paths from input.
    
    Args:
        input_path: Path to image file or directory
        
    Returns:
        List of image paths
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    if os.path.isfile(input_path):
        return [input_path]
    elif os.path.isdir(input_path):
        image_paths = []
        for root, dirs, files in os.walk(input_path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    image_paths.append(os.path.join(root, file))
        return sorted(image_paths)
    else:
        raise ValueError(f"Input path does not exist: {input_path}")


def save_predictions(results: List[Dict[str, Any]], output_path: str) -> None:
    """
    Save predictions to file.
    
    Args:
        results: List of prediction results
        output_path: Path to save results
    """
    import json
    
    # Save as JSON
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Predictions saved to: {output_path}")


def main():
    """Main prediction function."""
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Get device
    device = get_device(args.device)
    print(f"Using device: {device}")
    
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
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = load_checkpoint(args.checkpoint, model=model, device=device)
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    
    # Get image paths
    image_paths = get_image_paths(args.input)
    print(f"Found {len(image_paths)} images")
    
    if len(image_paths) == 0:
        print("No images found!")
        return
    
    # Create transform
    transform = get_test_transforms()
    
    # Make predictions
    print("Making predictions...")
    if len(image_paths) == 1:
        results = [predict_single_image(
            model=model,
            image_path=image_paths[0],
            transform=transform,
            device=device,
            confidence_threshold=args.confidence_threshold
        )]
    else:
        results = predict_batch(
            model=model,
            image_paths=image_paths,
            transform=transform,
            device=device,
            batch_size=args.batch_size,
            confidence_threshold=args.confidence_threshold
        )
    
    # Print results
    print("\nPrediction Results:")
    print("-" * 50)
    for result in results:
        print(f"Image: {result['image_path']}")
        print(f"Prediction: {result['class_name']} (confidence: {result['confidence']:.3f})")
        print(f"Confident: {result['is_confident']}")
        print(f"Probabilities - Real: {result['probabilities']['real']:.3f}, Spoof: {result['probabilities']['spoof']:.3f}")
        print("-" * 50)
    
    # Save results if requested
    if args.output:
        save_predictions(results, args.output)
    
    print("Prediction completed!")


if __name__ == "__main__":
    main()
