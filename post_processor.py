import torch

def post_process_rt_detr_outputs(outputs: dict, target_size: tuple, threshold: float = 0.5):
    """
    Converts raw RT-DETR decoder outputs into scaled bounding boxes and class scores,
    replacing the heavy Hugging Face RTDetrImageProcessor post-processing logic.
    """
    logits = outputs["logits"]
    pred_boxes = outputs["pred_boxes"]

    # 1. Apply sigmoid to get probabilities (RT-DETR uses sigmoid for classification)
    probabilities = torch.sigmoid(logits[0])
    
    # 2. Get the highest score and corresponding class for each query
    scores, labels = probabilities.max(dim=-1)
    
    # 3. Filter out low-confidence predictions
    keep_mask = scores > threshold
    scores = scores[keep_mask]
    labels = labels[keep_mask]
    boxes = pred_boxes[0][keep_mask]
    
    # 4. Convert boxes from [cx, cy, w, h] to [x_min, y_min, x_max, y_max]
    cx, cy, w, h = boxes.unbind(-1)
    x_min = cx - 0.5 * w
    y_min = cy - 0.5 * h
    x_max = cx + 0.5 * w
    y_max = cy + 0.5 * h
    
    converted_boxes = torch.stack((x_min, y_min, x_max, y_max), dim=-1)
    
    # 5. Scale the normalized coordinates up to the original image dimensions
    img_width, img_height = target_size
    scale_tensor = torch.tensor([img_width, img_height, img_width, img_height], device=converted_boxes.device)
    scaled_boxes = converted_boxes * scale_tensor
    
    return {
        "scores": scores,
        "labels": labels,
        "boxes": scaled_boxes
    }