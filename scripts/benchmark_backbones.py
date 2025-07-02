#!/usr/bin/env python3
"""
Benchmark script for comparing different backbone options in TerraTorch
"""
import os
import sys
import torch
import time
import argparse
import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt

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

# Import the backbone modules to ensure they're registered
try:
    # Import from the package
    import src
    print("Successfully imported backbone modules")
except ImportError as e:
    print(f"Failed to import backbone modules: {e}")
    print("Make sure the src/__init__.py file exists")
    sys.exit(1)

def benchmark_backbone(backbone_name, batch_sizes=[1, 2, 4, 8], input_shape=(3, 224, 224), num_runs=10):
    """Benchmark a backbone with different batch sizes"""
    print(f"\n=== Benchmarking {backbone_name} ===")
    
    try:
        # Build the backbone
        backbone = BACKBONE_REGISTRY.get(backbone_name)()
        backbone.eval()
        
        if torch.cuda.is_available():
            backbone = backbone.cuda()
            print(f"Running on GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("Running on CPU")
        
        results = []
        
        for batch_size in batch_sizes:
            print(f"Testing batch size: {batch_size}")
            
            # Create a dummy input tensor
            if torch.cuda.is_available():
                dummy_input = torch.randn(batch_size, *input_shape).cuda()
            else:
                dummy_input = torch.randn(batch_size, *input_shape)
            
            # Warmup
            print("  Warming up...")
            with torch.no_grad():
                for _ in range(3):
                    _ = backbone(dummy_input)
            
            # Benchmark
            print("  Running benchmark...")
            torch.cuda.reset_peak_memory_stats()
            start_time = time.time()
            
            with torch.no_grad():
                for _ in range(num_runs):
                    output = backbone(dummy_input)
            
            end_time = time.time()
            
            # Calculate metrics
            avg_time = (end_time - start_time) / num_runs
            throughput = batch_size / avg_time
            
            # Memory usage
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)
            else:
                memory_allocated = 0
            
            # Get output shape
            output_shape = tuple(output.shape)
            
            result = {
                'backbone': backbone_name,
                'batch_size': batch_size,
                'inference_time': avg_time,
                'throughput': throughput,
                'memory_mb': memory_allocated,
                'output_shape': output_shape
            }
            
            results.append(result)
            print(f"  Inference time: {avg_time:.4f} s")
            print(f"  Throughput: {throughput:.2f} images/s")
            print(f"  Memory: {memory_allocated:.2f} MB")
            print(f"  Output shape: {output_shape}")
        
        return results
    
    except Exception as e:
        print(f"✗ Benchmarking failed: {e}")
        return []

def plot_results(results_df, output_dir='../results'):
    """Plot benchmark results"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Throughput comparison
    plt.figure(figsize=(10, 6))
    for backbone in results_df['backbone'].unique():
        subset = results_df[results_df['backbone'] == backbone]
        plt.plot(subset['batch_size'], subset['throughput'], marker='o', label=backbone)
    
    plt.xlabel('Batch Size')
    plt.ylabel('Throughput (images/s)')
    plt.title('Backbone Throughput Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'throughput_comparison.png'))
    
    # Memory usage comparison
    plt.figure(figsize=(10, 6))
    for backbone in results_df['backbone'].unique():
        subset = results_df[results_df['backbone'] == backbone]
        plt.plot(subset['batch_size'], subset['memory_mb'], marker='o', label=backbone)
    
    plt.xlabel('Batch Size')
    plt.ylabel('GPU Memory Usage (MB)')
    plt.title('Backbone Memory Usage Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'memory_comparison.png'))

def main():
    """Main benchmark function"""
    parser = argparse.ArgumentParser(description='Benchmark TerraTorch backbones')
    parser.add_argument('--backbones', nargs='+', default=['earthmind_v1', 'internvl2_4b', 'granite_4_tiny'],
                        help='List of backbones to benchmark')
    parser.add_argument('--batch-sizes', nargs='+', type=int, default=[1, 2, 4, 8],
                        help='List of batch sizes to test')
    parser.add_argument('--runs', type=int, default=10,
                        help='Number of runs for each benchmark')
    parser.add_argument('--output', type=str, default='../results',
                        help='Output directory for results')
    
    args = parser.parse_args()
    
    print("=== TerraTorch Backbone Benchmark ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    all_results = []
    
    for backbone_name in args.backbones:
        try:
            results = benchmark_backbone(
                backbone_name, 
                batch_sizes=args.batch_sizes,
                num_runs=args.runs
            )
            all_results.extend(results)
        except Exception as e:
            print(f"Error benchmarking {backbone_name}: {e}")
    
    # Create results directory
    os.makedirs(args.output, exist_ok=True)
    
    # Save results to CSV
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(os.path.join(args.output, 'backbone_benchmark_results.csv'), index=False)
        
        # Print results table
        print("\n=== Benchmark Results ===")
        print(tabulate(results_df, headers='keys', tablefmt='pretty'))
        
        # Plot results
        try:
            plot_results(results_df, args.output)
            print(f"Plots saved to {args.output}")
        except Exception as e:
            print(f"Error creating plots: {e}")
    else:
        print("No benchmark results to report")

if __name__ == "__main__":
    main()
