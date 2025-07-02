# EarthMind-TerraTorch Integration

This project integrates the EarthMind multimodal geospatial model with the TerraTorch framework, enabling systematic backbone experimentation and comparison for geospatial tasks.

## Project Overview

EarthMind is a multimodal geospatial model that processes both SAR (Sentinel-1) and optical (Sentinel-2) imagery. This project adapts EarthMind to work with TerraTorch, a modular framework for geospatial deep learning that provides a backbone registry system, YAML-based configuration, and built-in support for various geospatial foundation models.

## Backbone Options

### Tier 1: Direct Fine-tuning (2-8B parameters)
- **EarthMind-v1**: Original EarthMind vision encoder
- **InternVL2.5-4B**: Multimodal vision-language model
- **Granite 4.0 Tiny**: IBM's efficient geospatial backbone

### Tier 2: TerraTorch Native Geospatial
- **Prithvi**: Temporal ViT pretrained on Landsat + Sentinel-2
- **TerraMind**: IBM ESA's geospatial encoder
- **Clay**: Vision Transformer MAE for multi-temporal geospatial data

### Tier 3: Large Models with Optimization (26-40B)
- **InternVL2-26B**: Requires LoRA/quantization
- **InternVL2-40B**: Maximum size, needs aggressive optimization

## Project Structure

```
earthmind/
├── configs/
│   ├── backbones/         # Backbone-specific configs
│   ├── experiments/       # Full experiment configs
│   └── tasks/             # Task-specific configs
├── scripts/
│   ├── test_earthmind.py          # Basic integration testing
│   ├── benchmark_backbones.py     # Performance benchmarking
│   └── optimize_large_models.py   # Memory optimization techniques
├── src/
│   ├── earthmind_backbone.py      # EarthMind backbone wrapper
│   ├── additional_backbones.py    # Additional backbone implementations
│   └── utils/
│       ├── memory_optimization.py # Memory optimization utilities
│       └── __init__.py
├── terratorch/                    # TerraTorch framework (installed)
├── terratorch-env/                # Virtual environment
├── requirements.txt               # Project dependencies
└── terratorch_testing_instructions.txt  # Usage instructions
```

## Getting Started

### Installation

1. Set up the TerraTorch environment:
```bash
python -m venv terratorch-env
source terratorch-env/bin/activate
pip install -r requirements.txt
```

2. Install TerraTorch:
```bash
cd terratorch
pip install -e .
cd ..
```

### Testing the Integration

1. Test the EarthMind backbone integration:
```bash
python scripts/test_earthmind.py
```

2. Benchmark different backbone options:
```bash
python scripts/benchmark_backbones.py --backbones earthmind_v1 internvl2_4b granite_4_tiny
```

3. Test memory optimization techniques for large models:
```bash
python scripts/optimize_large_models.py --backbone internvl2_4b --test all
```

## Memory Optimization Techniques

For large models (26-40B parameters), this project implements several memory optimization techniques:

1. **Quantization**: 4-bit and 8-bit quantization using bitsandbytes
2. **LoRA**: Low-Rank Adaptation for parameter-efficient fine-tuning
3. **Gradient Checkpointing**: Trading computation for memory
4. **Combined Approaches**: QLoRA (Quantization + LoRA) for maximum efficiency

## Configuration System

TerraTorch uses a YAML-based configuration system for easy backbone swapping:

```yaml
# Example: configs/experiments/earthmind_test.yaml
model:
  backbone:
    type: earthmind_v1
    pretrained: true
  decoder:
    type: unet_decoder
    in_channels: 768
    num_classes: 2

training:
  batch_size: 2
  learning_rate: 1e-4
  max_epochs: 5
```

## Evaluation Metrics

- **Memory Efficiency**: GPU memory usage for different backbones and batch sizes
- **Inference Speed**: Images per second (throughput)
- **Model Size**: Number of parameters and model size on GPU
- **Output Characteristics**: Shape, dtype, and value range

## Next Steps

After testing the integration, you can:

1. Create more sophisticated configuration files for specific tasks
2. Implement task-specific heads and necks
3. Test with real geospatial data
4. Benchmark different backbone options
5. Implement memory optimization techniques for larger models

## References

- [EarthMind Paper](https://arxiv.org/abs/2401.09647)
- [TerraTorch GitHub](https://github.com/IBM/terratorch)
- [InternVL2 GitHub](https://github.com/OpenGVLab/InternVL)
- [Granite Models](https://github.com/ibm-granite/models)
