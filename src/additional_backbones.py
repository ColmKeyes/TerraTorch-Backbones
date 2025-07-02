"""
Additional backbone implementations for TerraTorch
"""
from terratorch.registry import BACKBONE_REGISTRY
from transformers import AutoModel, AutoBackbone
import torch.nn as nn

@BACKBONE_REGISTRY.register("internvl2_4b")
def build_internvl2_4b_backbone(pretrained=True, **kwargs):
    """Register InternVL2.5-4B with TerraTorch"""
    try:
        model = AutoModel.from_pretrained("OpenGVLab/InternVL2.5-4B")
        return model.vision_model
    except Exception as e:
        print(f"Failed to load InternVL2.5-4B: {e}")
        raise

@BACKBONE_REGISTRY.register("granite_4_tiny")
def build_granite_backbone(pretrained=True, **kwargs):
    """Register Granite 4.0 Tiny with TerraTorch"""
    try:
        backbone = AutoBackbone.from_pretrained("ibm-granite/granite-4.0-tiny")
        return backbone
    except Exception as e:
        print(f"Failed to load Granite 4.0 Tiny: {e}")
        raise
