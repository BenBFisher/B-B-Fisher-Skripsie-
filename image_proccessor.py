import cv2
import torch

def preprocess_image(image_path: str, target_size=(640, 640)) -> torch.Tensor:
    """
    A standalone image processor for RT-DETR, replacing Hugging Face dependencies.
    """
    # 1. Load the image using OpenCV (loads as BGR)
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Error: Could not load image at {image_path}")

    # 2. Resize to the model's expected input dimensions
    img_resized = cv2.resize(img, target_size)

    # 3. Convert from OpenCV's BGR format to standard RGB
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

    # 4. Convert to a PyTorch tensor and reshape from (H, W, C) to (C, H, W)
    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float()

    # 5. Scale pixel values from [0, 255] down to [0.0, 1.0]
    img_scaled = img_tensor / 255.0

    # 6. Apply standard ImageNet normalization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_normalized = (img_scaled - mean) / std

    # 7. Add a batch dimension to match the model's expected shape: (1, C, H, W)
    pixel_values = img_normalized.unsqueeze(0)
    
    return pixel_values