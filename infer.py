import torch
import cv2
import json
import numpy as np 
import matplotlib.pyplot as plt
from src.models.model import GCPHeatmapModel
from src.data.dataset import get_dataloaders

def infer(image_path, annotations_file='data/train/train_dataset/gcp_marks.json', model_path='gcp_unet.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCPHeatmapModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print(f"Running Inference on: {image_path}")
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 1. "Cheat" by loading the ground truth to simulate YOLO
    with open(annotations_file, 'r') as f:
        annotations = json.load(f)
    
    # We have to extract the relative path from the absolute path
    rel_path = image_path.split('train_dataset/')[-1]
    true_x = int(annotations[rel_path]['mark']['x'])
    true_y = int(annotations[rel_path]['mark']['y'])

    # 2. Crop exactly around the true coordinate
    H, W, _ = img.shape
    crop_size = 512
    half = crop_size // 2
    
    # Calculate crop boundaries (safely clamping to edges)
    y1 = max(0, true_y - half)
    y2 = min(H, true_y + half)
    x1 = max(0, true_x - half)
    x2 = min(W, true_x + half)
    
    img_crop = img[y1:y2, x1:x2]
    
    # Pad if we hit an edge
    pad_y = crop_size - img_crop.shape[0]
    pad_x = crop_size - img_crop.shape[1]
    if pad_y > 0 or pad_x > 0:
        img_crop = np.pad(img_crop, ((0, pad_y), (0, pad_x), (0, 0)), mode='constant')

    # 3. Model Inference
    img_tensor = torch.from_numpy(img_crop).float() / 255.0
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        # Unpack both the heatmap and the classification logits
        heatmap_pred, label_logits = model(img_tensor)

    heatmap = heatmap_pred[0, 0].cpu().numpy()
    y_pred, x_pred = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    
    # 3.5 Predict the Shape
    class_idx = torch.argmax(label_logits, dim=1).item()
    class_names = ["Cross", "Square", "L-Shape"]
    predicted_shape = class_names[class_idx]
    
    print(f"SUCCESS: Predicted GCP at [X={x_pred}, Y={y_pred}] with shape: {predicted_shape}")

    # 4. Visualize
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    ax[0].imshow(img_crop)
    ax[0].scatter(x_pred, y_pred, color='red', s=100, marker='X', label='Predicted GCP')
    
    # Let's also plot the Ground Truth to see how close we are!
    # If the crop wasn't padded, the true point is exactly in the center (256, 256)
    ax[0].scatter(half, half, color='green', s=100, marker='o', label='Ground Truth')
    
    ax[0].set_title(f'Predicted Shape: {predicted_shape}')
    ax[0].legend()  

    ax[1].imshow(heatmap, cmap='jet')
    ax[1].set_title('Raw Heatmap Output')   

    plt.savefig('inference_result.png')
    print("Saved visualization to inference_result.png!")



import math
from src.data.dataset import get_dataloaders
import json

def evaluate_validation_set(annotations_file='data/train/train_dataset/gcp_marks.json', model_path='gcp_unet.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Loading Model for Validation...")
    model = GCPHeatmapModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    with open(annotations_file, 'r') as f:
        annotations = json.load(f)
        
    _, val_loader = get_dataloaders(annotations, 'data/train/train_dataset', batch_size=4, crop_size=512)
    
    total_error = 0.0
    num_samples = 0
    
    print(f"Evaluating on {len(val_loader.dataset)} validation images...")
    
    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(device)
            # We bring ground truth back to CPU for numpy math
            heatmaps_gt = batch['heatmap'].cpu().numpy() 
            
            # Predict (Model now returns heatmap AND classification logits)
            heatmap_out, class_logits = model(images)
            heatmaps_pred = heatmap_out.cpu().numpy()
            
            # Loop through each image in the batch
            for i in range(images.size(0)):
                pred_map = heatmaps_pred[i, 0]
                gt_map = heatmaps_gt[i, 0]
                
                # Check if the ground truth heatmap is entirely empty (missing GCP)
                if np.amax(gt_map) == 0:
                    continue
                
                # Argmax gives the 1D index, unravel_index converts to 2D (Y, X)
                pred_y, pred_x = np.unravel_index(np.argmax(pred_map), pred_map.shape)
                gt_y, gt_x = np.unravel_index(np.argmax(gt_map), gt_map.shape)
                
                # The Pythagorean Theorem (Euclidean Distance)
                pixel_error = math.sqrt((pred_x - gt_x)**2 + (pred_y - gt_y)**2)
                
                total_error += pixel_error
                num_samples += 1
                
    if num_samples == 0:
        print("No valid samples found!")
        return
        
    rmse = total_error / num_samples
    print(f"==================================================")
    print(f"MEAN PIXEL ERROR: {rmse:.2f} pixels")
    print(f"==================================================")
    return rmse

if __name__ == "__main__":
    # 1. Visualize a single prediction (Heatmap + Shape)
    test_image = "data/train/train_dataset/231129_CTD/231129_CTD_GDA94/30/DJI_20231129135330_0056.JPG"
    infer(test_image)

    # 2. Test the model mathematically across the entire Validation Set
    evaluate_validation_set()
