import torch
import torch.nn as nn
import json
from src.data.dataset import get_dataloaders
from src.models.model import GCPHeatmapModel

def train():
    # 1. Setup Data
    print("Loading data...")
    with open('data/train/train_dataset/gcp_marks.json', 'r') as f:
        annotations = json.load(f)
    
    # batch size of 4 
    train_loader, val_loader = get_dataloaders(
        annotations, 
        'data/train/train_dataset', 
        batch_size=4, 
        crop_size=512
    )
    
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    model = GCPHeatmapModel().to(device)
    
    # 3. Weighted Loss and Optimizer
    def weighted_mse_loss(pred, target):
        
        weight = torch.where(target > 0.05, 100.0, 1.0)
        return torch.mean(weight * (pred - target) ** 2)

    criterion = weighted_mse_loss
    # Ignore -1 because some GCPs in the JSON might have UNKNOWN shapes
    # Dataset frequencies: L-Shape(491), Square(328), Cross(177). 
    # Softened weights: L-Shape is common, but we don't want to over-penalize it.
    class_weights = torch.tensor([2.0, 1.2, 1.0]).to(device)
    classification_criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1) 
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # 4. The Training Loop
    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            # Move data to GPU
            images = batch['image'].to(device)
            heatmaps = batch['heatmap'].to(device)
            labels = batch['label'].to(device).long() # CrossEntropy requires LongTensors
            
            optimizer.zero_grad()
            
            # The model now returns two things!
            heatmap_preds, label_logits = model(images)
            
            # Calculate both losses
            heatmap_loss = criterion(heatmap_preds, heatmaps)
            class_loss = classification_criterion(label_logits, labels)
            
            # Multi-Task Learning: Sum the losses together (Give 2x weight to class loss to focus on shape!)
            loss = heatmap_loss + 2.0 * class_loss
            
            loss.backward()
            
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")
                
                
        print(f"--- Epoch {epoch} Average Loss: {train_loss / len(train_loader):.4f} ---")
    
    print("Training Complete! Saving Model...")
    torch.save(model.state_dict(),'gcp_unet.pth')    

if __name__ == '__main__':
    train()
    
