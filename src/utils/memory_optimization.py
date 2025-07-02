"""
Memory optimization utilities for large models
"""
import torch
from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
import logging

def get_4bit_quantization_config():
    """
    Get a 4-bit quantization configuration for loading large models
    
    Returns:
        BitsAndBytesConfig: Configuration for 4-bit quantization
    """
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

def get_8bit_quantization_config():
    """
    Get an 8-bit quantization configuration for loading large models
    
    Returns:
        BitsAndBytesConfig: Configuration for 8-bit quantization
    """
    return BitsAndBytesConfig(
        load_in_8bit=True,
    )

def apply_lora(model, target_modules=None, rank=16, alpha=32, dropout=0.1):
    """
    Apply LoRA (Low-Rank Adaptation) to a model for parameter-efficient fine-tuning
    
    Args:
        model: The model to apply LoRA to
        target_modules: List of module names to apply LoRA to (e.g., ["q_proj", "v_proj"])
                        If None, will try to infer based on model architecture
        rank: LoRA rank
        alpha: LoRA alpha parameter
        dropout: LoRA dropout probability
    
    Returns:
        The model with LoRA applied
    """
    if target_modules is None:
        # Try to infer target modules based on model architecture
        if hasattr(model, "vision_model"):
            # For vision models like EarthMind
            target_modules = ["query", "key", "value", "output.dense"]
        else:
            # Default for transformer models
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    
    logging.info(f"Applying LoRA with rank={rank}, alpha={alpha} to modules: {target_modules}")
    
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias="none",
        task_type="FEATURE_EXTRACTION"
    )
    
    return get_peft_model(model, lora_config)

def enable_gradient_checkpointing(model):
    """
    Enable gradient checkpointing for memory efficiency during training
    
    Args:
        model: The model to enable gradient checkpointing for
    
    Returns:
        The model with gradient checkpointing enabled
    """
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        logging.info("Gradient checkpointing enabled")
    elif hasattr(model, "enable_gradient_checkpointing"):
        model.enable_gradient_checkpointing()
        logging.info("Gradient checkpointing enabled")
    else:
        logging.warning("Could not enable gradient checkpointing - method not found")
    
    return model

def optimize_memory_for_inference(model, device="cuda", dtype=torch.float16):
    """
    Apply memory optimizations for inference
    
    Args:
        model: The model to optimize
        device: The device to move the model to
        dtype: The data type to use for inference
    
    Returns:
        The optimized model
    """
    # Move to device with appropriate dtype
    model = model.to(device=device, dtype=dtype)
    
    # Enable eval mode
    model.eval()
    
    # Disable gradient computation
    for param in model.parameters():
        param.requires_grad = False
    
    logging.info(f"Model optimized for inference on {device} with dtype {dtype}")
    return model

def optimize_memory_for_training(model, device="cuda", apply_lora_config=None, enable_checkpointing=True):
    """
    Apply memory optimizations for training
    
    Args:
        model: The model to optimize
        device: The device to move the model to
        apply_lora_config: LoRA configuration dictionary (or None to skip LoRA)
        enable_checkpointing: Whether to enable gradient checkpointing
    
    Returns:
        The optimized model
    """
    # Move to device
    model = model.to(device)
    
    # Apply LoRA if requested
    if apply_lora_config is not None:
        model = apply_lora(
            model, 
            target_modules=apply_lora_config.get("target_modules"),
            rank=apply_lora_config.get("rank", 16),
            alpha=apply_lora_config.get("alpha", 32),
            dropout=apply_lora_config.get("dropout", 0.1)
        )
    
    # Enable gradient checkpointing if requested
    if enable_checkpointing:
        model = enable_gradient_checkpointing(model)
    
    logging.info(f"Model optimized for training on {device}")
    return model

def print_model_memory_usage(model, input_size=(1, 3, 224, 224)):
    """
    Print memory usage statistics for a model
    
    Args:
        model: The model to analyze
        input_size: Input tensor size for a forward pass
    """
    if not torch.cuda.is_available():
        logging.warning("CUDA not available, memory statistics will not be accurate")
        return
    
    # Clear cache
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # Get initial memory usage
    initial_mem = torch.cuda.memory_allocated() / (1024 ** 2)
    
    # Create input and move to GPU
    dummy_input = torch.randn(*input_size, device="cuda")
    
    # Get memory after input creation
    input_mem = torch.cuda.memory_allocated() / (1024 ** 2)
    
    # Move model to GPU if not already there
    model = model.to("cuda")
    
    # Get memory after model loaded
    model_mem = torch.cuda.memory_allocated() / (1024 ** 2)
    
    # Forward pass
    with torch.no_grad():
        _ = model(dummy_input)
    
    # Get memory after forward pass
    forward_mem = torch.cuda.memory_allocated() / (1024 ** 2)
    peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
    
    # Print statistics
    print("\n=== Model Memory Usage ===")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print(f"Model size on GPU: {model_mem - input_mem:.2f} MB")
    print(f"Memory for forward pass: {forward_mem - model_mem:.2f} MB")
    print(f"Peak memory usage: {peak_mem:.2f} MB")
    print(f"Total memory allocated: {forward_mem:.2f} MB")
    
    # Memory breakdown by parameter groups if possible
    try:
        print("\n=== Memory by Module ===")
        for name, module in model.named_children():
            params = sum(p.numel() for p in module.parameters()) / 1e6
            print(f"{name}: {params:.2f}M parameters")
    except Exception as e:
        print(f"Could not print memory by module: {e}")
