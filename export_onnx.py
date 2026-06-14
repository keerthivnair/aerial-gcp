import torch
from src.models.model import GCPHeatmapModel
import os

def export_unet_to_onnx(model_path='gcp_unet.pth', output_path='gcp_unet.onnx'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCPHeatmapModel().to(device)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded weights from {model_path}")
    else:
        print(f"Warning: {model_path} not found. Exporting untrained model.")
        
    model.eval()

    dummy_input = torch.randn(1, 3, 512, 512, device=device)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['heatmap', 'class_logits'],
        dynamic_axes={'input': {0: 'batch_size'},
                      'heatmap': {0: 'batch_size'},
                      'class_logits': {0: 'batch_size'}}
    )
    print(f"Model successfully exported to {output_path}")

if __name__ == '__main__':
    export_unet_to_onnx()
