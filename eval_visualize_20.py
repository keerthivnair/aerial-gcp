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
    base_res_dir = "results_valid_images"
    os.makedirs(base_res_dir, exist_ok=True)
    
    existing_runs = [d for d in os.listdir(base_res_dir) if d.startswith('run_')]
    run_nums = [int(d.split('_')[1]) for d in existing_runs if d.split('_')[1].isdigit()]
    next_run = max(run_nums) + 1 if run_nums else 1
    run_dir = os.path.join(base_res_dir, f"run_{next_run}")
    os.makedirs(run_dir, exist_ok=True)
    
    log_file_path = os.path.join(run_dir, "accuracy_report.txt")
    
    # Helper to print to console AND save to text file
    def log_print(msg):
        print(msg)
        with open(log_file_path, "a") as f:
            f.write(msg + "\n")

    log_print(f"--- STARTING PIPELINE: {run_dir} ---")
    
    # 1. Load Both Models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    log_print("Loading Stage 1: Heavy-Duty YOLOv8...")
    yolo_model = YOLO('yolo_weights.pt')
    
    log_print("Loading Stage 2: Multi-Task U-Net...")
    unet_model = GCPHeatmapModel().to(device)
    unet_model.load_state_dict(torch.load('gcp_unet.pth', map_location=device))
    unet_model.eval()
    
    val_images = glob.glob('data/yolo_dataset/images/val/*.JPG')
    
    sample_images = random.sample(val_images, min(20, len(val_images)))
    total_imgs = len(sample_images)
    
    with open('data/train/train_dataset/gcp_marks.json', 'r') as f:
        annotations = json.load(f)
        
    class_names = ["Cross", "Square", "L-Shape"]
    
    total_rmse = 0.0
    successful_crops = 0
    errors = []
    y_true_shapes = []
    y_pred_shapes = []
    
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
        
        # Convert local U-Net prediction back to Global Massive Image Coordinates
        final_global_x = cx1 + pred_x
        final_global_y = cy1 + pred_y
        
        error = math.sqrt((final_global_x - true_x_global)**2 + (final_global_y - true_y_global)**2)
        errors.append(error)
        total_rmse += error
        successful_crops += 1
        
        # --- VISUALIZATION ---
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        
        bx1 = max(0, x1 - cx1)
        by1 = max(0, y1 - cy1)
        bx2 = min(512, x2 - cx1)
        by2 = min(512, y2 - cy1)
        cv2.rectangle(crop, (bx1, by1), (bx2, by2), (0, 100, 255), 3) # Blue YOLO Box
        
        ax[0].imshow(crop)
        ax[0].scatter(pred_x, pred_y, color='red', s=100, marker='X', label='U-Net GCP Center')
        
        # Also plot Ground Truth
        true_local_x = true_x_global - cx1
        true_local_y = true_y_global - cy1
        ax[0].scatter(true_local_x, true_local_y, color='green', s=100, marker='o', label='Ground Truth')
        
        ax[0].set_title(f'Pred Shape: {predicted_shape} | True: {true_shape} | Error: {error:.2f}px')
        ax[0].legend()
        ax[0].axis('off')
        
        ax[1].imshow(heatmap, cmap='jet')
        ax[1].set_title('U-Net Heatmap Output')
        ax[1].axis('off')
        
        plt.tight_layout()
        save_path = os.path.join(run_dir, f'end2end_result_{i}.png')
        plt.savefig(save_path)
        plt.close()
        log_print(f"  -> Error: {error:.2f} px. Saved {save_path}")
        
    log_print("\n=============================================")
    log_print(f" END-TO-END PIPELINE SUMMARY ({run_dir}) ")
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
    log_print("=============================================")

if __name__ == "__main__":
    run_end_to_end()
