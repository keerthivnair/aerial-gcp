import os
import glob
import random
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import math
from ultralytics import YOLO
from src.models.model import GCPHeatmapModel
from sklearn.metrics import f1_score

def run_end_to_end():
    base_res_dir = "submission"
    os.makedirs(base_res_dir, exist_ok=True)
    
    log_file_path = os.path.join(base_res_dir, "validation_report.txt")
    json_output_path = os.path.join(base_res_dir, "validation_predictions.json")
    
    with open(log_file_path, "w") as f:
        f.write("")
        
    def log_print(msg):
        print(msg)
        with open(log_file_path, "a") as f:
            f.write(msg + "\n")

    log_print(f"--- STARTING FULL VALIDATION PIPELINE ---")
    
    # 1. Load Both Models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    log_print("Loading Stage 1: Heavy-Duty YOLOv8...")
    yolo_model = YOLO('yolo_weights.pt')
    
    log_print("Loading Stage 2: Multi-Task U-Net...")
    unet_model = GCPHeatmapModel().to(device)
    unet_model.load_state_dict(torch.load('gcp_unet.pth', map_location=device))
    unet_model.eval()
    
    val_images = glob.glob('data/yolo_dataset/images/val/*.JPG')
    sample_images = val_images
    total_imgs = len(sample_images)
    
    with open('data/train/train_dataset/gcp_marks.json', 'r') as f:
        annotations = json.load(f)
        
    class_names = ["Cross", "Square", "L-Shape"]
    
    total_rmse = 0.0
    successful_crops = 0
    errors = []
    y_true_shapes = []
    y_pred_shapes = []
    
    validation_predictions = {}
    
    for i, img_path in enumerate(sample_images):
        log_print(f"Processing Image {i+1}/{total_imgs}: {os.path.basename(img_path)}")
        
        # --- STAGE 1: YOLO DETECTION ---
        results = yolo_model.predict(img_path, verbose=False, conf=0.5)[0]
        if len(results.boxes) == 0:
            log_print("  -> YOLO failed to find GCP. Skipping.")
            continue
            
        # Get highest confidence YOLO box
        x1, y1, x2, y2 = map(int, results.boxes[0].xyxy[0].cpu().numpy())
        
        # Find the center of the YOLO box to make our 512x512 U-Net crop
        yolo_center_x = (x1 + x2) // 2
        yolo_center_y = (y1 + y2) // 2
        
        # --- STAGE 2: U-NET CROP & HEATMAP ---
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        H, W, _ = img.shape
        
        half = 256
        cy1 = max(0, yolo_center_y - half)
        cy2 = min(H, yolo_center_y + half)
        cx1 = max(0, yolo_center_x - half)
        cx2 = min(W, yolo_center_x + half)
        
        crop = img[cy1:cy2, cx1:cx2]
        
        # Pad if hit edges
        pad_y = 512 - crop.shape[0]
        pad_x = 512 - crop.shape[1]
        if pad_y > 0 or pad_x > 0:
            crop = np.pad(crop, ((0, pad_y), (0, pad_x), (0, 0)), mode='constant')
            
        img_tensor = torch.from_numpy(crop).float() / 255.0
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
        
        with torch.no_grad():
            heatmap_pred, label_logits = unet_model(img_tensor)
            
        heatmap = heatmap_pred[0, 0].cpu().numpy()
        pred_y, pred_x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        
        class_idx = torch.argmax(label_logits, dim=1).item()
        predicted_shape = class_names[class_idx]
        
        # --- CALCULATE RMSE ERROR ---
        safe_filename = os.path.basename(img_path)
        original_rel_path = next(p for p in annotations.keys() if p.replace('/', '_').replace(' ', '_') == safe_filename)
        true_x_global = float(annotations[original_rel_path]['mark']['x'])
        true_y_global = float(annotations[original_rel_path]['mark']['y'])
        true_shape = annotations[original_rel_path].get('verified_shape', 'Unknown')
        
        y_true_shapes.append(true_shape)
        y_pred_shapes.append(predicted_shape)
        
        final_global_x = cx1 + pred_x
        final_global_y = cy1 + pred_y
        
        error = math.sqrt((final_global_x - true_x_global)**2 + (final_global_y - true_y_global)**2)
        
        validation_predictions[os.path.basename(img_path)] = {
            "mark": {"x": float(final_global_x), "y": float(final_global_y)},
            "predicted_shape": predicted_shape,
            "true_shape": true_shape,
            "pixel_error": error
        }
        
        errors.append(error)
        total_rmse += error
        successful_crops += 1
        
        log_print(f"  -> Error: {error:.2f} px.")
        
    log_print("\n=============================================")
    log_print(f" END-TO-END PIPELINE SUMMARY (submission/validation_report.txt) ")
    log_print("=============================================")
    log_print(f"YOLO Success Rate (Pipeline Recall): {(successful_crops/total_imgs)*100:.1f}%")
    if successful_crops > 0:
        pck_10 = sum(1 for e in errors if e <= 10.0) / successful_crops * 100
        pck_25 = sum(1 for e in errors if e <= 25.0) / successful_crops * 100
        pck_50 = sum(1 for e in errors if e <= 50.0) / successful_crops * 100
        
        # Calculate Macro F1 (ignoring Unknowns if they exist)
        valid_indices = [idx for idx, t in enumerate(y_true_shapes) if t in class_names]
        y_true_valid = [y_true_shapes[i] for i in valid_indices]
        y_pred_valid = [y_pred_shapes[i] for i in valid_indices]
        
        macro_f1 = f1_score(y_true_valid, y_pred_valid, labels=class_names, average='macro', zero_division=0) * 100
        
        log_print(f"Localization - Sub-Pixel RMSE: {total_rmse/successful_crops:.2f} px")
        log_print(f"Localization - PCK @ 10px: {pck_10:.1f}%")
        log_print(f"Localization - PCK @ 25px: {pck_25:.1f}%")
        log_print(f"Localization - PCK @ 50px: {pck_50:.1f}%")
        log_print(f"Classification - Macro F1-Score: {macro_f1:.1f}%")
        
    # Save the JSON predictions for validation data
    with open(json_output_path, "w") as f:
        json.dump(validation_predictions, f, indent=4)
        
    log_print(f"\nSaved validation predictions to: {json_output_path}")
    log_print("=============================================")

if __name__ == "__main__":
    run_end_to_end()
