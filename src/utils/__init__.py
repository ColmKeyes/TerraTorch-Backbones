"""
Utility functions for the EarthMind project
"""

from .memory_optimization import (
    get_4bit_quantization_config,
    get_8bit_quantization_config,
    apply_lora,
    enable_gradient_checkpointing,
    optimize_memory_for_inference,
    optimize_memory_for_training,
    print_model_memory_usage
)

__all__ = [
    'get_4bit_quantization_config',
    'get_8bit_quantization_config',
    'apply_lora',
    'enable_gradient_checkpointing',
    'optimize_memory_for_inference',
    'optimize_memory_for_training',
    'print_model_memory_usage'
]
