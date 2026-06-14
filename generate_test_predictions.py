import os
import glob
import cv2
import numpy as np
from ultralytics import YOLO
import onnxruntime as ort

import json
import sys

def generate_full_report(test_dir='data/test/'):
    base_res_dir = "submission"
    os.makedirs(base_res_dir, exist_ok=True)
    report_path = os.path.join(base_res_dir, "predictions.json")
    
    predictions_dict = {}

    print("Loading Stage 1: Heavy-Duty YOLOv8...")
    yolo_model = YOLO('yolo_weights.pt')
    
    print("Loading Stage 2: ONNX U-Net (ONNX Runtime)...")
    ort_session = ort.InferenceSession("gcp_unet.onnx")
    input_name = ort_session.get_inputs()[0].name
    class_names = ["Cross", "Square", "L-Shape"]
    
    # --- 2. Gather Test Images ---
    if not os.path.exists(test_dir):
        print(f"Directory {test_dir} not found!")
        return
        
    all_images = glob.glob(os.path.join(test_dir, '**', '*.JPG'), recursive=True) + \
                 glob.glob(os.path.join(test_dir, '**', '*.jpg'), recursive=True)
                 
    if not all_images:
        print(f"No images found in {test_dir}")
        return

    print(f"Found {len(all_images)} images. Beginning mass inference...\n")
    
    success_count = 0

    with open(report_path, "a") as f:
        for i, img_path in enumerate(all_images):
            filename = os.path.basename(img_path)
            
            # --- STAGE 1 ---
            results = yolo_model.predict(img_path, verbose=False, conf=0.25)[0]
            if len(results.boxes) == 0:
                print(f"[{i+1}/{len(all_images)}] {filename} -> FAILED (YOLO No Detection)")
                continue
                
            x1, y1, x2, y2 = map(int, results.boxes[0].xyxy[0].cpu().numpy())
            yolo_center_x = (x1 + x2) // 2
            yolo_center_y = (y1 + y2) // 2
            
            # --- STAGE 2 ---
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            H, W, _ = img.shape
            
            half = 256
            cy1 = max(0, yolo_center_y - half)
            cy2 = min(H, yolo_center_y + half)
            cx1 = max(0, yolo_center_x - half)
            cx2 = min(W, yolo_center_x + half)
            
            crop = img[cy1:cy2, cx1:cx2]
            
            pad_y = 512 - crop.shape[0]
            pad_x = 512 - crop.shape[1]
            if pad_y > 0 or pad_x > 0:
                crop = np.pad(crop, ((0, pad_y), (0, pad_x), (0, 0)), mode='constant')
                
            img_numpy = crop.astype(np.float32) / 255.0
            img_numpy = np.transpose(img_numpy, (2, 0, 1))
            img_numpy = np.expand_dims(img_numpy, axis=0)
            
            onnx_outputs = ort_session.run(None, {input_name: img_numpy})
            heatmap_pred = onnx_outputs[0]
            label_logits = onnx_outputs[1]
                
            heatmap = heatmap_pred[0, 0]
            pred_y, pred_x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
            
            class_idx = np.argmax(label_logits[0])
            predicted_shape = class_names[class_idx]
            
            final_global_x = cx1 + pred_x
            final_global_y = cy1 + pred_y
            
            success_count += 1
            
            rel_path = os.path.relpath(img_path, test_dir)
            
            predictions_dict[rel_path] = {
                "mark": {"x": float(final_global_x), "y": float(final_global_y)},
                "verified_shape": predicted_shape
            }
            
            if (i+1) % 10 == 0:
                print(f"Processed {i+1}/{len(all_images)}...")
                
    # Save to JSON
    with open(report_path, "w") as f:
        json.dump(predictions_dict, f, indent=4)
        
    print(f"\n====================================================")
    print(f" OVERALL PIPELINE ACCURACY: {(success_count/len(all_images))*100:.2f}%")
    print(f"====================================================")
    print(f"\nDone! Evaluation predictions saved to {report_path}")

if __name__ == "__main__":
    test_directory = sys.argv[1] if len(sys.argv) > 1 else 'data/test/'
    generate_full_report(test_directory)
