#!/usr/bin/env python3
"""
Test script for EarthMind backbone integration with TerraTorch
"""
import os
import sys
import torch
import yaml
import time
import logging

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the backbone registry
try:
    from terratorch.registry import BACKBONE_REGISTRY
    print("Successfully imported TerraTorch backbone registry")
except ImportError as e:
    print(f"Failed to import TerraTorch: {e}")
    print("Make sure TerraTorch is installed and in your Python path")
    sys.exit(1)

# Import the backbone module to ensure it's registered
try:
    # Import from the package
    import src
    print("Successfully imported EarthMind backbone")
except ImportError as e:
    print(f"Failed to import EarthMind backbone: {e}")
    print("Make sure the src/__init__.py file exists")
    sys.exit(1)

def test_backbone_registration():
    """Test if the EarthMind backbone is properly registered"""
    print("\n=== Testing Backbone Registration ===")
    
    try:
        # Check if the backbone is in the registry
        if "earthmind_v1" in BACKBONE_REGISTRY:
            print("✓ EarthMind backbone is registered in TerraTorch")
        else:
            print("✗ EarthMind backbone is NOT registered in TerraTorch")
            return False
        
        # Try to build the backbone
        backbone = BACKBONE_REGISTRY.get("earthmind_v1")()
        print(f"✓ Successfully built backbone: {type(backbone).__name__}")
        return True
    except Exception as e:
        print(f"✗ Failed to build backbone: {e}")
        return False

def test_backbone_forward_pass():
    """Test a forward pass through the backbone"""
    print("\n=== Testing Backbone Forward Pass ===")
    
    try:
        # Build the backbone
        backbone = BACKBONE_REGISTRY.get("earthmind_v1")()
        backbone.eval()
        
        # Create a dummy input tensor (3-channel RGB image)
        dummy_input = torch.randn(1, 3, 224, 224)
        print(f"Input shape: {dummy_input.shape}")
        
        # Run a forward pass
        with torch.no_grad():
            output = backbone(dummy_input)
        
        print(f"Output shape: {output.shape}")
        print(f"Output dtype: {output.dtype}")
        print(f"Output min/max: {output.min().item():.4f}/{output.max().item():.4f}")
        
        return True
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        return False

def benchmark_performance():
    """Benchmark the backbone's performance"""
    print("\n=== Benchmarking Performance ===")
    
    try:
        # Build the backbone
        backbone = BACKBONE_REGISTRY.get("earthmind_v1")()
        backbone.eval()
        
        # Create a dummy input tensor
        dummy_input = torch.randn(1, 3, 224, 224)
        
        # Warmup
        print("Warming up...")
        with torch.no_grad():
            for _ in range(3):
                _ = backbone(dummy_input)
        
        # Benchmark
        print("Running benchmark...")
        num_runs = 5
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_runs):
                _ = backbone(dummy_input)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_runs
        print(f"Average inference time: {avg_time:.4f} seconds")
        print(f"Throughput: {1/avg_time:.2f} images/second")
        
        # Memory usage
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / (1024 ** 2)
            memory_reserved = torch.cuda.memory_reserved() / (1024 ** 2)
            print(f"GPU memory allocated: {memory_allocated:.2f} MB")
            print(f"GPU memory reserved: {memory_reserved:.2f} MB")
        
        return True
    except Exception as e:
        print(f"✗ Benchmarking failed: {e}")
        return False

def main():
    """Main test function"""
    print("=== EarthMind-TerraTorch Integration Test ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    # Run tests
    registration_success = test_backbone_registration()
    
    if registration_success:
        forward_success = test_backbone_forward_pass()
        if forward_success:
            benchmark_performance()
    
    print("\n=== Test Summary ===")
    print(f"Backbone Registration: {'✓' if registration_success else '✗'}")
    if registration_success:
        print(f"Forward Pass: {'✓' if forward_success else '✗'}")
        if forward_success:
            print("Performance Benchmark: ✓")
    
    if registration_success and forward_success:
        print("\n✅ All tests passed! EarthMind is successfully integrated with TerraTorch.")
    else:
        print("\n❌ Some tests failed. Please check the error messages above.")

if __name__ == "__main__":
    main()
