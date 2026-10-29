import os

# 1. Your Custom coco5k Dictionary
COCO5K_CLASSES = {
    0: 'person',
    1: 'bicycle',
    2: 'car',
    3: 'motorcycle',
    4: 'airplane',
    5: 'bus',
    6: 'train',
    7: 'truck',
    8: 'boat',
    9: 'traffic light',
    10: 'fire hydrant',
    11: 'stop sign',
    12: 'parking meter',
    13: 'bench',
    14: 'bird',
    15: 'cat',
    16: 'dog',
    17: 'horse',
    18: 'sheep',
    19: 'cow',
    20: 'elephant',
    21: 'bear',
    22: 'zebra',
    23: 'giraffe',
    24: 'backpack',
    25: 'umbrella',
    26: 'handbag',
    27: 'tie',
    28: 'suitcase',
    29: 'frisbee',
    30: 'skis',
    31: 'snowboard',
    32: 'sports ball',
    33: 'kite',
    34: 'baseball bat',
    35: 'baseball glove',
    36: 'skateboard',
    37: 'surfboard',
    38: 'tennis racket',
    39: 'bottle',
    40: 'wine glass',
    41: 'cup',
    42: 'fork',
    43: 'knife',
    44: 'spoon',
    45: 'bowl',
    46: 'banana',
    47: 'apple',
    48: 'sandwich',
    49: 'orange',
    50: 'broccoli',
    51: 'carrot',
    52: 'hot dog',
    53: 'pizza',
    54: 'donut',
    55: 'cake',
    56: 'chair',
    57: 'couch',
    58: 'potted plant',
    59: 'bed',
    60: 'dining table',
    61: 'toilet',
    62: 'tv',
    63: 'laptop',
    64: 'mouse',
    65: 'remote',
    66: 'keyboard',
    67: 'cell phone',
    68: 'microwave',
    69: 'oven',
    70: 'toaster',
    71: 'sink',
    72: 'refrigerator',
    73: 'book',
    74: 'clock',
    75: 'vase',
    76: 'scissors',
    77: 'teddy bear',
    78: 'hair drier', 79: 'toothbrush',
}

# 2. Classes you want to drop
CLASSES_TO_REMOVE = [
    'potted plant', 'tv', 'microwave', 
    'refrigerator','clock', 'vase', 'bear', 'bed', 'stop sign', 'truck', 'car', 
    'teddy bear', 'skis', 'oven', 'sports ball', 'baseball glove', 
    'tennis racket', 'handbag', 'backpack', 'bird', 'boat', 
    'train', 'sandwich', 'surfboard', 
    'mouse', 'bus', 'cat', 
    'airplane', 'zebra', 'tie', 'traffic light', 'apple', 
    'baseball bat', 'knife', 'cake', 'wine glass', 
    'spoon', 'banana', 'donut', 'sink', 
    'toilet', 'broccoli', 'skateboard', 'fork', 'carrot', 
    'couch', 'remote', 'scissors', 'bicycle', 'sheep', 
    'bench', 'orange', 'elephant', 'frisbee', 'umbrella', 
    'horse', 'dog', 'motorcycle', 'kite', 'pizza', 
    'fire hydrant', 'suitcase', 'hot dog', 'cow', 'giraffe', 
    'snowboard', 'parking meter', 'toothbrush', 'toaster', 
    'hair drier', "book", "dining table", "cell phone", "bowl", 
    "laptop", "keyboard", "cup", "bottle"
]

def main():
    # Directories
    input_labels_dir = "./coco_train2017/labels/train"       # <-- CHANGE THIS to your actual label folder
    output_labels_dir = "./coco_train2017_small1409_2/labels/train"   # <-- New folder for the filtered labels

    os.makedirs(output_labels_dir, exist_ok=True)

    # Figure out the integer IDs of the classes being removed
    ids_to_remove = []
    for cid, name in COCO5K_CLASSES.items():
        if name in CLASSES_TO_REMOVE:
            ids_to_remove.append(cid)
            
    print(f"Removing these IDs: {ids_to_remove}")

    # Build a remapping dictionary so the new IDs are contiguous (0 to 67)
    old_id_to_new_id = {}
    new_class_dict = {}
    current_new_id = 0

    for old_id in range(80):
        if old_id not in ids_to_remove:
            old_id_to_new_id[old_id] = current_new_id
            new_class_dict[current_new_id] = COCO5K_CLASSES[old_id]
            current_new_id += 1

    # Process the text files
    txt_files = [f for f in os.listdir(input_labels_dir) if f.endswith('.txt')]
    removed_boxes_count = 0
    kept_boxes_count = 0

    print(f"Filtering {len(txt_files)} label files...")

    for filename in txt_files:
        input_path = os.path.join(input_labels_dir, filename)
        output_path = os.path.join(output_labels_dir, filename)

        with open(input_path, 'r') as f_in:
            lines = f_in.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 5:
                class_id = int(parts[0])
                if class_id in ids_to_remove:
                    removed_boxes_count += 1
                else:
                    # Remap the ID to its new contiguous number
                    new_id = old_id_to_new_id[class_id]
                    parts[0] = str(new_id)
                    new_lines.append(" ".join(parts) + "\n")
                    kept_boxes_count += 1

        # If the image still has objects in it, save the new file.
        # (If an image *only* had zebras, it will now have an empty text file, which YOLO/RT-DETR treats as a background image).
        with open(output_path, 'w') as f_out:
            f_out.writelines(new_lines)

    print("\n=== FILTERING COMPLETE ===")
    print(f"Boxes removed: {removed_boxes_count}")
    print(f"Boxes kept:    {kept_boxes_count}")
    
    print("\n=== YOUR NEW 68-CLASS CONFIG DICTIONARY ===")
    print("Replace your old dictionary in your scripts with this one:")
    print("NEW_CLASSES = {")
    for k, v in new_class_dict.items():
        print(f"    {k}: '{v}',")
    print("}")

if __name__ == "__main__":
    main()