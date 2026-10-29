import cv2
import torch
from PIL import Image
from transformers import RTDetrImageProcessor
from modeling_rt_detr_v2 import RTDetrV2ForObjectDetection

print("Loading local model for live stream...")
model = RTDetrV2ForObjectDetection.from_pretrained("./my_local_model")
processor = RTDetrImageProcessor.from_pretrained("./my_local_model")
model.eval()

# Open your laptop webcam (0 is usually the default built-in camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Webcam started! Press 'q' in the video window to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # 1. OpenCV reads frames in BGR format; convert them to RGB PIL Images
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)

    # 2. Preprocess frame for the model
    inputs = processor(images=pil_image, return_tensors="pt")

    # 3. Run inference
    with torch.no_grad():
        outputs = model(**inputs)

    # 4. Post-process predictions (using a 0.5 confidence threshold)
    target_sizes = torch.tensor([pil_image.size[::-1]])
    results = processor.post_process_object_detection(
        outputs, threshold=0.5, target_sizes=target_sizes
    )[0]

    # 5. Draw bounding boxes and labels onto the live frame
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        xmin, ymin, xmax, ymax = map(int, box.tolist())
        class_name = model.config.id2label[label.item()]
        confidence = round(score.item(), 2)

        # Draw rectangle box (Green)
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        
        # Put label text above the box
        label_text = f"{class_name}: {confidence}"
        cv2.putText(
            frame, 
            label_text, 
            (xmin, max(ymin - 10, 10)), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            (0, 255, 0), 
            2
        )

    # 6. Display the resulting frame in a window
    cv2.imshow("RT-DETR Live Feed", frame)

    # Press 'q' on your keyboard to exit the live stream loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up and close windows
cap.release()
cv2.destroyAllWindows()