import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class RTDetrResNetBackbone(nn.Module):
    """
    A lightweight, standalone ResNet backbone for RT-DETR.
    Extracts multi-scale feature maps without Hugging Face dependencies.
    """
    def __init__(self, pretrained=True):
        super().__init__()
        
        # 1. Load the standard torchvision ResNet-18
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        base_model = resnet18(weights=weights)
        
        # 2. Extract the specific layers we need for the feature pyramid
        # RT-DETR requires multi-scale features, typically from layers 2, 3, and 4
        self.stem = nn.Sequential(
            base_model.conv1,
            base_model.bn1,
            base_model.relu,
            base_model.maxpool,
            base_model.layer1
        )
        self.layer2 = base_model.layer2  # Stride 8 output (128 channels)
        self.layer3 = base_model.layer3  # Stride 16 output (256 channels)
        self.layer4 = base_model.layer4  # Stride 32 output (512 channels)
        
        # Optional: Freeze the stem to save memory during any future fine-tuning
        for param in self.stem.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor):
        """
        Passes the image tensor through the network and collects intermediate features.
        """
        x = self.stem(x)
        
        f2 = self.layer2(x)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        
        # Return the multi-scale feature maps required by the RT-DETR encoder
        return [f2, f3, f4]