import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small

class RTDetrMobileNetBackbone(nn.Module):
    """
    A lightweight, standalone MobileNetV3-Small backbone for Micro-RT-DETR.
    Extracts multi-scale feature maps without Hugging Face dependencies.
    """
    def __init__(self, pretrained=False):
        super().__init__()
        
        # 1. Load MobileNetV3-Small (pretrained=False ensures safe initialization for custom training)
        base_model = mobilenet_v3_small(weights=None)
        features = base_model.features
        
        # 2. Extract specific layers for the RT-DETR feature pyramid
        # PyTorch's MobileNetV3-Small natively outputs 24, 48, and 96 channels at these strides.
        
        # Stride 8 output (24 channels)
        self.stage1 = nn.Sequential(*features[:4])
        
        # Stride 16 output (48 channels)
        self.stage2 = nn.Sequential(*features[4:9])
        
        # Stride 32 output (96 channels) 
        # We explicitly stop before features[12] to avoid the massive 576-channel expansion layer.
        self.stage3 = nn.Sequential(*features[9:12]) 
        
        # Optional: Freeze the early stage to reduce memory overhead during training
        for param in self.stage1.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor):
        """
        Passes the image tensor through the network and collects intermediate features.
        """
        f8 = self.stage1(x)
        f16 = self.stage2(f8)
        f32 = self.stage3(f16)
        
        # Return the multi-scale feature maps required by the RT-DETR encoder
        return [f8, f16, f32]