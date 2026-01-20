import os
import sys
import cv2
import logging
import numpy as np
import torch
import timm
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from dataset import get_dataloaders 

# ==========================================
# 1. CONFIGURATION
# ==========================================
# List of all models we want to explain
MODELS_TO_EXPLAIN = ['resnet50', 'densenet121', 'vit_base_patch16_224']

DATA_PATH = '/home/projects/cgm-prj10354/ViT-vs-CNN-medical-imaging/data'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# How many examples to generate per model?
NUM_IMAGES_TO_SAVE = 3

# Log file name
LOG_FILE = 'explanation_log_all_models.txt'

# ==========================================
# 2. LOGGER SETUP
# ==========================================
def setup_logger():
    logger = logging.getLogger('Explainability')
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()

    file_handler = logging.FileHandler(LOG_FILE, mode='w')
    file_handler.setFormatter(logging.Formatter('%(message)s'))
    
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(logging.Formatter('%(message)s'))

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger

logger = setup_logger()

# ==========================================
# 3. HELPER FUNCTIONS FOR VIT
# ==========================================
def reshape_transform_vit(tensor):
    """
    SPECIAL FUNCTION FOR VIT:
    Vision Transformers process images as a 1D sequence of patches.
    Grad-CAM expects a 2D spatial grid (Height x Width).
    
    This function:
    1. Removes the 'Class Token' (the first item in the sequence).
    2. Reshapes the remaining patches back into a 14x14 grid.
    3. Transposes axes to match what Grad-CAM expects (Batch, Channels, Height, Width).
    """
    # Remove class token (index 0), keep the rest (196 patches)
    result = tensor[:, 1:, :] 
    
    # Reshape 196 -> 14x14 spatial grid
    result = result.reshape(tensor.size(0), 14, 14, tensor.size(2))
    
    # Swap axes to bring Channels to the correct position: (B, H, W, C) -> (B, C, H, W)
    result = result.transpose(2, 3).transpose(1, 2)
    return result

def get_target_layer(model, model_name):
    """
    Determines the target layer for Grad-CAM based on architecture.
    """
    if 'resnet' in model_name:
        # ResNet: Target the last bottleneck layer
        return [model.layer4[-1]]
        
    elif 'densenet' in model_name:
        # DenseNet: Target the last dense block
        return [model.features[-1]]
        
    elif 'vit' in model_name:
        # ViT: Target the LayerNorm of the very last transformer block.
        # This is where the spatial information is most rich before classification.
        return [model.blocks[-1].norm1]
        
    else:
        raise ValueError(f"Unknown model architecture: {model_name}")

# ==========================================
# 4. MAIN EXPLAIN LOOP
# ==========================================
def run_grad_cam_batch():
    logger.info(f"=== EXPLAINABILITY ANALYSIS START ===")
    
    # Iterate over every model in our list
    for model_name in MODELS_TO_EXPLAIN:
        logger.info(f"\n{'='*40}")
        logger.info(f"PROCESSING MODEL: {model_name}")
        logger.info(f"{'='*40}")
        
        model_path = f'best_model_{model_name}.pth'
        
        # 1. Load Model
        if not os.path.exists(model_path):
            logger.error(f"Skipping {model_name}: File {model_path} not found.")
            continue
            
        logger.info("Loading weights...")
        try:
            model = timm.create_model(model_name, pretrained=False, num_classes=2)
            model.load_state_dict(torch.load(model_path))
            model.to(DEVICE)
            model.eval()
        except Exception as e:
            logger.error(f"Error loading {model_name}: {e}")
            continue

        # 2. Setup Grad-CAM
        try:
            target_layers = get_target_layer(model, model_name)
            
            # If it is ViT, we MUST pass the reshape_transform function
            if 'vit' in model_name:
                cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform_vit)
            else:
                cam = GradCAM(model=model, target_layers=target_layers) # Standard for CNNs
                
        except Exception as e:
            logger.error(f"Error setting up Grad-CAM for {model_name}: {e}")
            continue

        # 3. Get Test Loader
        _, _, test_loader = get_dataloaders(DATA_PATH, batch_size=1)
        
        saved_count = 0
        
        # 4. Loop through images
        for i, (image_tensor, label) in enumerate(test_loader):
            if label.item() != 1: continue # Skip Normal cases
                
            image_tensor = image_tensor.to(DEVICE)
            
            output = model(image_tensor)
            _, prediction = torch.max(output, 1)
            
            # Only explain if the model correctly predicted PNEUMONIA (TP)
            if prediction.item() == 1:
                
                # --- Generate Heatmap ---
                targets = [ClassifierOutputTarget(1)]
                
                # Compute CAM
                grayscale_cam = cam(input_tensor=image_tensor, targets=targets)
                grayscale_cam = grayscale_cam[0, :]
                
                # --- Visualization ---
                img_to_show = image_tensor.squeeze().cpu().permute(1, 2, 0).numpy()
                img_to_show = img_to_show * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                img_to_show = np.clip(img_to_show, 0, 1) 
                
                visualization = show_cam_on_image(img_to_show, grayscale_cam, use_rgb=True)

                # Save filename with model name to distinguish them
                # Example: heatmap_resnet50_sample_1.jpg
                filename = f"heatmap_{model_name}_sample_{saved_count+1}.jpg"
                cv2.imwrite(filename, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
                
                logger.info(f"✔ Saved: {filename}")
                saved_count += 1
                
                if saved_count >= NUM_IMAGES_TO_SAVE:
                    break
    
    logger.info(f"\n=== ALL EXPLANATIONS COMPLETE ===")
    logger.info(f"Log saved to: {LOG_FILE}")

if __name__ == "__main__":
    try:
        run_grad_cam_batch()
    except Exception as e:
        logger.error(f"Critical Error: {e}")