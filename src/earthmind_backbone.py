"""
EarthMind backbone registration for TerraTorch
"""

from terratorch.registry import BACKBONE_REGISTRY
from transformers import AutoModel
import logging

@BACKBONE_REGISTRY.register("earthmind_v1")
def build_earthmind_backbone(pretrained=True, **kwargs):
    """Register EarthMind with TerraTorch"""
    try:
        model = AutoModel.from_pretrained("shuyansy/EarthMind-1.0-base")
        return model.vision_model
    except Exception as e:
        logging.warning(f"Failed to load EarthMind from HuggingFace: {e}")
        raise RuntimeError(f"EarthMind backbone loading failed: {e}")

if __name__ == "__main__":
    # Test the backbone registration
    print("Testing EarthMind backbone registration...")
    backbone = build_earthmind_backbone()
    print(f"Backbone loaded successfully: {type(backbone)}")
