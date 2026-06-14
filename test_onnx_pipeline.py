import os
import glob
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import onnxruntime as ort

def run_test_onnx_pipeline(test_dir='data/test/'):
    # --- SETUP RUN DIRECTORY ---
    base_res_dir = "results_test_images"
    os.makedirs(base_res_dir, exist_ok=True)
    
    # Find next run number
    existing_runs = [d for d in os.listdir(base_res_dir) if d.startswith('run_')]
    run_nums = [int(d.split('_')[1]) for d in existing_runs if d.split('_')[1].isdigit()]
    next_run = max(run_nums) + 1 if run_nums else 1
    run_dir = os.path.join(base_res_dir, f"run_{next_run}")
    os.makedirs(run_dir, exist_ok=True)
    
    log_file_path = os.path.join(run_dir, "test_report.txt")
    
    def log_print(msg):
        print(msg)
        with open(log_file_path, "a") as f:
            f.write(msg + "\n")

    log_print(f"--- STARTING ONNX TEST PIPELINE: {run_dir} ---")
    
    # --- 1. Load Both Models ---
    log_print("Loading Stage 1: Heavy-Duty YOLOv8...")
    yolo_model = YOLO('yolo_weights.pt')
    
    log_print("Loading Stage 2: ONNX U-Net (ONNX Runtime)...")
    ort_session = ort.InferenceSession("gcp_unet.onnx")
    input_name = ort_session.get_inputs()[0].name
    
    class_names = ["Cross", "Square", "L-Shape"]
    
    # --- 2. Gather Test Images ---
    if not os.path.exists(test_dir):
        log_print(f"Directory {test_dir} not found! Please provide a valid test data path.")
        return
        
    all_images = glob.glob(os.path.join(test_dir, '**', '*.JPG'), recursive=True) + \
                 glob.glob(os.path.join(test_dir, '**', '*.jpg'), recursive=True)
                 
    if not all_images:
        log_print("No images found in data/test/")
        return
        
    log_print(f"Found {len(all_images)} total test images in the dataset.")
    
    # Randomly sample 20 images
    sample_images = random.sample(all_images, min(20, len(all_images)))
    
    successful_crops = 0
    
    # --- 3. Run Inference Loop ---
    for i, img_path in enumerate(sample_images):
        log_print(f"\nProcessing Test Image {i+1}/20: {os.path.basename(img_path)}")
        
        # --- STAGE 1: YOLO DETECTION ---
        results = yolo_model.predict(img_path, verbose=False, conf=0.25)[0]
        if len(results.boxes) == 0:
            log_print("  -> YOLO failed to find GCP. Saving failure image.")
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.figure(figsize=(8, 8))
            plt.imshow(img)
            plt.title("YOLO Stage 1 FAILED (No GCP Detected)", color='red')
            plt.axis('off')
            save_path = os.path.join(run_dir, f'test_result_{i}_FAILED.png')
            plt.savefig(save_path)
            plt.close()
            continue
            
        x1, y1, x2, y2 = map(int, results.boxes[0].xyxy[0].cpu().numpy())
        yolo_center_x = (x1 + x2) // 2
        yolo_center_y = (y1 + y2) // 2
        
        # --- STAGE 2: U-NET CROP & ONNX HEATMAP ---
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
            
        # Prepare input for ONNX (numpy array, shape [1, 3, 512, 512], type float32)
        img_numpy = crop.astype(np.float32) / 255.0
        img_numpy = np.transpose(img_numpy, (2, 0, 1)) # HWC to CHW
        img_numpy = np.expand_dims(img_numpy, axis=0)  # Add batch dimension
        
        # Run ONNX Inference!
        onnx_outputs = ort_session.run(None, {input_name: img_numpy})
        
        # Unpack ONNX outputs (based on our export_onnx.py output names)
        heatmap_pred = onnx_outputs[0] # Shape: [1, 1, 512, 512]
        label_logits = onnx_outputs[1] # Shape: [1, 3]
            
        heatmap = heatmap_pred[0, 0]
        pred_y, pred_x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        
        class_idx = np.argmax(label_logits[0])
        predicted_shape = class_names[class_idx]
        
        final_global_x = cx1 + pred_x
        final_global_y = cy1 + pred_y
        
        successful_crops += 1
        
        # --- VISUALIZATION ---
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        
        bx1 = max(0, x1 - cx1)
        by1 = max(0, y1 - cy1)
        bx2 = min(512, x2 - cx1)
        by2 = min(512, y2 - cy1)
        cv2.rectangle(crop, (bx1, by1), (bx2, by2), (0, 100, 255), 3) # Blue YOLO Box
        
        ax[0].imshow(crop)
        ax[0].scatter(pred_x, pred_y, color='red', s=100, marker='X', label='ONNX Prediction')
        ax[0].set_title(f'Shape: {predicted_shape} | Glob_X: {final_global_x}, Glob_Y: {final_global_y}')
        ax[0].legend()
        ax[0].axis('off')
        
        ax[1].imshow(heatmap, cmap='jet')
        ax[1].set_title('ONNX Heatmap Output')
        ax[1].axis('off')
        
        plt.tight_layout()
        save_path = os.path.join(run_dir, f'test_result_{i}.png')
        plt.savefig(save_path)
        plt.close()
        
        log_print(f"  -> SUCCESS! Found {predicted_shape} at Global (X: {final_global_x}, Y: {final_global_y}). Saved {save_path}")
        
    log_print("\n=============================================")
    log_print(f" ONNX TEST PIPELINE SUMMARY ({run_dir}) ")
    log_print("=============================================")
    log_print(f"Total Test Images Sampled: 20")
    log_print(f"YOLO Test Success Rate:   {(successful_crops/20)*100:.1f}%")
    log_print("Note: Ground truth JSON is not available for wild test images, so RMSE cannot be calculated.")
    log_print("=============================================")

if __name__ == "__main__":
    import sys
    test_directory = sys.argv[1] if len(sys.argv) > 1 else 'data/test/'
    run_test_onnx_pipeline(test_directory)
