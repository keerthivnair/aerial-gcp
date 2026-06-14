import torch.nn as nn
import segmentation_models_pytorch as smp


class GCPHeatmapModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        # This one line gives us a massive, pre-trained neural network!
        self.model = smp.Unet(
            encoder_name="resnet34",        # Use a ResNet34 as the feature extractor
            encoder_weights="imagenet",     # Pre-trained on 14 million images
            in_channels=3,                  # RGB image input
            classes=1,

            aux_params=dict(
                pooling='avg',    # Global Average Pooling on the deepest features
                dropout=0.5,      # Drop 50% of neurons to prevent overfitting
                classes=3         # 3 possible shapes (Cross, Square, L-Shape)
            )                       # We want 1 heatmap channel output
        )
        
    def forward(self, x):
        heatmap, label_logits = self.model(x)
        return heatmap, label_logits