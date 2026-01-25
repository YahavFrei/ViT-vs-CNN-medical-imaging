import os
import cv2
import torch
import numpy as np
import timm
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from PIL import Image
import glob

# ==========================================
# 1. CONFIGURATION
# ==========================================
CONFIG = {
    'data_path': 'data/test/PNEUMONIA',  # Path to test images (Pneumonia class)
    'output_dir': 'results/heatmaps/24.01',    # Where to save the results
    'model_name': 'densenet121' ,#'resnet50', #'vit_base_patch16_224',      # The model name 
    # 'densenet121',         # The winning model
    'num_classes': 2,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

# --- NEW FUNCTION: Essential for ViT to work with GradCAM ---
def reshape_transform_vit(tensor, height=14, width=14):
    """
    Reshapes the ViT output (Sequence of tokens) back into a 2D Spatial Grid (Image-like)
    so GradCAM can process it.
    """
    # Remove the class token (first token)
    result = tensor[:, 1:, :] 
    
    # Reshape: [Batch, Tokens, Channels] -> [Batch, Channels, Height, Width]
    result = result.reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result
# ------------------------------------------------------------

def find_latest_model(model_name_prefix):
    """
    Searches for the most recent .pth file for the specified model
    in the 'models/' directory (or current directory).
    """
    # Check in 'models' folder first
    search_pattern = f"models/best_model_{model_name_prefix}*.pth"
    files = glob.glob(search_pattern)
    
    # If not found, check current directory
    if not files:
        search_pattern = f"best_model_{model_name_prefix}*.pth"
        files = glob.glob(search_pattern)

    if not files:
        raise FileNotFoundError(f"No model found matching pattern: {search_pattern}")

    # Sort by modification time (newest first)
    latest_file = max(files, key=os.path.getctime)
    print(f"Found latest model: {latest_file}")
    return latest_file

def load_model(model_path, model_name):
    """
    Loads the trained model architecture and weights.
    """
    print(f"Loading model: {model_name}...")
    model = timm.create_model(model_name, pretrained=False, num_classes=CONFIG['num_classes'])
    
    # Load weights (handle potential CPU/GPU mapping)
    state_dict = torch.load(model_path, map_location=CONFIG['device'])
    model.load_state_dict(state_dict)
    
    model.to(CONFIG['device'])
    model.eval()
    return model

def get_target_layer(model, model_name):
    """
    Identifies the last convolutional layer for Grad-CAM based on architecture.
    """
    if 'densenet' in model_name:
        return [model.features[-1]]
    elif 'resnet' in model_name:
        return [model.layer4[-1]]
    elif 'vit' in model_name:
        # in ViT looking on last block normalization
        return [model.blocks[-1].norm1]
    else:
        raise ValueError(f"Target layer not defined for {model_name}")

def process_and_visualize(image_path, model, target_layers, save_name, reshape_func=None):
    """
    Generates Grad-CAM heatmap for a single image and saves it.
    """
    # 1. Load and Preprocess Image
    rgb_img = cv2.imread(image_path, 1)[:, :, ::-1] # BGR to RGB
    rgb_img = cv2.resize(rgb_img, (224, 224))
    rgb_img = np.float32(rgb_img) / 255
    
    # Standard ImageNet normalization
    input_tensor = preprocess_image(rgb_img,
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]).to(CONFIG['device'])

    # 2. Initialize Grad-CAM (Pass the reshape function if it exists)
    cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_func)

    # 3. Generate Heatmap (Target class 1 = Pneumonia)
    targets = [ClassifierOutputTarget(1)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]

    # 4. Overlay Heatmap on Image
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    # 5. Save Result
    save_path = os.path.join(CONFIG['output_dir'], save_name)
    cv2.imwrite(save_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
    print(f"Saved explanation to: {save_path}")

# ==========================================
# 3. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # Setup directories
    os.makedirs(CONFIG['output_dir'], exist_ok=True)

    try:
        # A. Locate and Load Model
        # Using the manual path as you requested for the "No Augmentation" run
        model_path = find_latest_model(CONFIG['model_name']) #"models/best_model_resnet50_20260120_174702.pth" #"models/best_model_vit_base_patch16_224_20260120_175213.pth" #find_latest_model(CONFIG['model_name'])
        model = load_model(model_path, CONFIG['model_name'])
        target_layers = get_target_layer(model, CONFIG['model_name'])

        # --- NEW LOGIC: Check if we need the reshape function (Only for ViT) ---
        reshape_func = reshape_transform_vit if 'vit' in CONFIG['model_name'] else None
        # -----------------------------------------------------------------------

        # B. Select sample images (First 3 images from the test folder)
        image_files = glob.glob(os.path.join(CONFIG['data_path'], "*.jpeg")) + \
                      glob.glob(os.path.join(CONFIG['data_path'], "*.jpg"))
        
        if len(image_files) == 0:
            print("Error: No images found in data directory. Check path.")
        else:
            # Process first 3 images
            #for i, img_path in enumerate(image_files[:3]):
            # doing for picture 10 to 20 - for densenet best 
            for i, img_path in enumerate(image_files[10:20]):
                #save_name = f"heatmap_{CONFIG['model_name']}_sample_{i+1}.jpg"
                save_name = f"heatmap_{CONFIG['model_name']}_sample_{i+10}.jpg"
                # Pass the reshape_func to the processing function
                process_and_visualize(img_path, model, target_layers, save_name, reshape_func)

            print("\nDone! Check the 'results/heatmaps/24.01' folder.")

    except Exception as e:
        print(f"An error occurred: {e}")
        # Print full trace to help debug if needed
        import traceback
        traceback.print_exc()