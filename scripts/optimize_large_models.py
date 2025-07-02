#!/usr/bin/env python3
"""
Script to demonstrate memory optimization techniques for large models
"""
import os
import sys
import torch
import argparse
import logging

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Import the backbone registry
try:
    from terratorch.registry import BACKBONE_REGISTRY
    logging.info("Successfully imported TerraTorch backbone registry")
except ImportError as e:
    logging.error(f"Failed to import TerraTorch: {e}")
    logging.error("Make sure TerraTorch is installed and in your Python path")
    sys.exit(1)

# Import the backbone modules to ensure they're registered
try:
    # Import from the package
    import src
    logging.info("Successfully imported backbone modules")
except ImportError as e:
    logging.error(f"Failed to import backbone modules: {e}")
    logging.error("Make sure the src/__init__.py file exists")
    sys.exit(1)

# Import memory optimization utilities
try:
    from src.utils import (
        get_4bit_quantization_config,
        get_8bit_quantization_config,
        apply_lora,
        enable_gradient_checkpointing,
        optimize_memory_for_inference,
        optimize_memory_for_training,
        print_model_memory_usage
    )
    logging.info("Successfully imported memory optimization utilities")
except ImportError as e:
    logging.error(f"Failed to import memory optimization utilities: {e}")
    logging.error("Make sure the src/utils/__init__.py file exists")
    sys.exit(1)

def test_baseline_memory(backbone_name):
    """Test baseline memory usage without optimizations"""
    logging.info(f"\n=== Testing Baseline Memory Usage for {backbone_name} ===")
    
    try:
        # Build the backbone
        backbone = BACKBONE_REGISTRY.get(backbone_name)()
        
        # Print memory usage
        print_model_memory_usage(backbone)
        
        return backbone
    except Exception as e:
        logging.error(f"Error testing baseline memory: {e}")
        return None

def test_quantization(backbone_name, bit_width=8):
    """Test memory usage with quantization"""
    logging.info(f"\n=== Testing {bit_width}-bit Quantization for {backbone_name} ===")
    
    try:
        # Get quantization config
        if bit_width == 4:
            quant_config = get_4bit_quantization_config()
            logging.info("Using 4-bit quantization")
        else:
            quant_config = get_8bit_quantization_config()
            logging.info("Using 8-bit quantization")
        
        # Load model with quantization
        from transformers import AutoModel
        
        if backbone_name == "earthmind_v1":
            model = AutoModel.from_pretrained(
                "shuyansy/EarthMind-1.0-base",
                quantization_config=quant_config
            )
            backbone = model.vision_model
        elif backbone_name == "internvl2_4b":
            model = AutoModel.from_pretrained(
                "OpenGVLab/InternVL2.5-4B",
                quantization_config=quant_config
            )
            backbone = model.vision_model
        elif backbone_name == "granite_4_tiny":
            from transformers import AutoBackbone
            backbone = AutoBackbone.from_pretrained(
                "ibm-granite/granite-4.0-tiny",
                quantization_config=quant_config
            )
        else:
            logging.error(f"Unknown backbone: {backbone_name}")
            return None
        
        # Print memory usage
        print_model_memory_usage(backbone)
        
        return backbone
    except Exception as e:
        logging.error(f"Error testing quantization: {e}")
        return None

def test_lora(backbone, rank=16):
    """Test memory usage with LoRA"""
    logging.info(f"\n=== Testing LoRA (rank={rank}) ===")
    
    try:
        # Apply LoRA
        lora_backbone = apply_lora(backbone, rank=rank)
        
        # Print memory usage
        print_model_memory_usage(lora_backbone)
        
        return lora_backbone
    except Exception as e:
        logging.error(f"Error testing LoRA: {e}")
        return None

def test_gradient_checkpointing(backbone):
    """Test memory usage with gradient checkpointing"""
    logging.info("\n=== Testing Gradient Checkpointing ===")
    
    try:
        # Enable gradient checkpointing
        backbone_with_gc = enable_gradient_checkpointing(backbone)
        
        # Print memory usage
        print_model_memory_usage(backbone_with_gc)
        
        return backbone_with_gc
    except Exception as e:
        logging.error(f"Error testing gradient checkpointing: {e}")
        return None

def test_combined_optimizations(backbone_name):
    """Test memory usage with combined optimizations"""
    logging.info(f"\n=== Testing Combined Optimizations for {backbone_name} ===")
    
    try:
        # Get quantization config
        quant_config = get_4bit_quantization_config()
        
        # Load model with quantization
        from transformers import AutoModel
        
        if backbone_name == "earthmind_v1":
            model = AutoModel.from_pretrained(
                "shuyansy/EarthMind-1.0-base",
                quantization_config=quant_config
            )
            backbone = model.vision_model
        elif backbone_name == "internvl2_4b":
            model = AutoModel.from_pretrained(
                "OpenGVLab/InternVL2.5-4B",
                quantization_config=quant_config
            )
            backbone = model.vision_model
        elif backbone_name == "granite_4_tiny":
            from transformers import AutoBackbone
            backbone = AutoBackbone.from_pretrained(
                "ibm-granite/granite-4.0-tiny",
                quantization_config=quant_config
            )
        else:
            logging.error(f"Unknown backbone: {backbone_name}")
            return None
        
        # Apply LoRA
        backbone = apply_lora(backbone, rank=8)
        
        # Enable gradient checkpointing
        backbone = enable_gradient_checkpointing(backbone)
        
        # Print memory usage
        print_model_memory_usage(backbone)
        
        return backbone
    except Exception as e:
        logging.error(f"Error testing combined optimizations: {e}")
        return None

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Test memory optimization techniques for large models')
    parser.add_argument('--backbone', type=str, default='earthmind_v1',
                        choices=['earthmind_v1', 'internvl2_4b', 'granite_4_tiny'],
                        help='Backbone to test')
    parser.add_argument('--test', type=str, default='all',
                        choices=['baseline', 'quantization', 'lora', 'gradient_checkpointing', 'combined', 'all'],
                        help='Optimization technique to test')
    parser.add_argument('--bit-width', type=int, default=8, choices=[4, 8],
                        help='Bit width for quantization (4 or 8)')
    parser.add_argument('--lora-rank', type=int, default=16,
                        help='Rank for LoRA')
    
    args = parser.parse_args()
    
    logging.info("=== Memory Optimization Test ===")
    logging.info(f"PyTorch version: {torch.__version__}")
    logging.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logging.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
        logging.info(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    
    # Run tests
    if args.test == 'baseline' or args.test == 'all':
        backbone = test_baseline_memory(args.backbone)
    
    if args.test == 'quantization' or args.test == 'all':
        backbone = test_quantization(args.backbone, args.bit_width)
    
    if args.test == 'lora' or args.test == 'all':
        if 'backbone' not in locals():
            backbone = test_baseline_memory(args.backbone)
        if backbone is not None:
            test_lora(backbone, args.lora_rank)
    
    if args.test == 'gradient_checkpointing' or args.test == 'all':
        if 'backbone' not in locals():
            backbone = test_baseline_memory(args.backbone)
        if backbone is not None:
            test_gradient_checkpointing(backbone)
    
    if args.test == 'combined' or args.test == 'all':
        test_combined_optimizations(args.backbone)
    
    logging.info("\n=== Memory Optimization Test Complete ===")

if __name__ == "__main__":
    main()
