import os
import glob
import json
from ultralytics import YOLO

# 1. Load the trained YOLO model
model_yolo = YOLO('yolo_weights.pt')

# 2. Get all images in the YOLO validation folder
val_images = glob.glob('data/yolo_dataset/images/val/*.JPG')

# 3. Load the true JSON to verify!
with open('data/train/train_dataset/gcp_marks.json', 'r') as f:
    annotations = json.load(f)

total = len(val_images)
found = 0
correct_boxes = 0

print(f"Running Strict YOLO Validation on {total} images...")

for img_path in val_images:
    # We have to reverse-engineer the original path from the safe YOLO filename
    safe_filename = os.path.basename(img_path)
    # This is a bit hacky, but we search the JSON for the matching filename
    original_rel_path = next(p for p in annotations.keys() if p.replace('/', '_').replace(' ', '_') == safe_filename)
    
    true_x = float(annotations[original_rel_path]['mark']['x'])
    true_y = float(annotations[original_rel_path]['mark']['y'])

    results = model_yolo(img_path, verbose=False)[0]
    
    if len(results.boxes) > 0:
        found += 1
        # Grab the highest confidence box [x1, y1, x2, y2]
        box = results.boxes[0].xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = box
        
        # STRICT CHECK: Is the true point mathematically inside the box?
        if (x1 <= true_x <= x2) and (y1 <= true_y <= y2):
            correct_boxes += 1

print(f"\n=====================================")
print(f" STRICT YOLO Validation Results ")
print(f"=====================================")
print(f"Total Images: {total}")
print(f"Boxes Predicted: {found}")
print(f"Boxes that TRULY contain the GCP: {correct_boxes}")
print(f"Strict Accuracy: {(correct_boxes/total)*100:.2f}%")
print(f"=====================================")
