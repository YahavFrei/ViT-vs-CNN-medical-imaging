import os
import sys
import glob
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import torch
import timm
from datetime import datetime
from sklearn.metrics import confusion_matrix
from dataset import get_dataloaders 

# ==========================================
# 1. CONFIGURATION
# ==========================================
CONFIG = {
    'data_path': 'data',             # Path to the dataset root directory
    'logs_dir': 'logs',              # Directory where training logs are stored
    'models_dir': 'models',          # Directory where .pth model files are stored
    'output_dir': 'results/plots/24.01',   # Directory to save the generated plots
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_classes': 2
}

# Ensure output directories exist
os.makedirs(CONFIG['output_dir'], exist_ok=True)
os.makedirs(CONFIG['logs_dir'], exist_ok=True)

# ==========================================
# 2. LOGGER SETUP (For this script's execution)
# ==========================================
def setup_visualization_logger():
    """
    Sets up a unique logger for this visualization run.
    This ensures we have a record of which files were processed.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(CONFIG['logs_dir'], f"visualization_log_{timestamp}.txt")
    
    logger = logging.getLogger(f"Viz_{timestamp}")
    logger.setLevel(logging.INFO)
    
    # File Handler
    file_handler = logging.FileHandler(log_filename, mode='w')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    # Stream Handler (Console)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(logging.Formatter('%(message)s'))
    
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    
    return logger

logger = setup_visualization_logger()

# ==========================================
# 3. PARSING & PLOTTING FUNCTIONS
# ==========================================

def parse_filename_metadata(filename):
    """
    Extracts model name and timestamp from filenames like:
    'training_log_densenet121_20260120_174515.txt'
    Returns: (model_name, timestamp)
    """
    try:
        # Remove extension
        base = os.path.splitext(os.path.basename(filename))[0]
        # Split by underscore
        parts = base.split('_')
        
        # Expected format: training_log_{model}_{date}_{time}
        # Example parts: ['training', 'log', 'densenet121', '20260120', '174515']
        
        # Determine where the timestamp starts (last 2 parts)
        timestamp = f"{parts[-2]}_{parts[-1]}"
        
        # The model name is everything between 'log' and the date
        # This handles model names with underscores (e.g., vit_base_patch16...)
        model_name_parts = parts[2:-2]
        model_name = "_".join(model_name_parts)
        
        return model_name, timestamp
    except Exception as e:
        logger.warning(f"Skipping file {filename}: Could not parse metadata. Error: {e}")
        return None, None

def plot_training_curves(log_file_path, model_name, run_timestamp):
    """
    Reads the text log file and generates Loss and Accuracy curves.
    """
    logger.info(f"Generating training curves for: {model_name} ({run_timestamp})...")
    
    epochs = []
    train_loss, val_loss = [], []
    train_acc, val_acc = [], []

    try:
        with open(log_file_path, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            # Parse Epoch
            if "Epoch" in line and "/" in line:
                parts = line.split('/')
                # Example: "Epoch 1/5" -> 1
                try:
                    current_epoch = int(parts[0].split()[-1])
                    epochs.append(current_epoch)
                except ValueError:
                    continue
            
            # Parse Loss
            if "Loss" in line and "|" in line:
                parts = line.split('|')
                if len(parts) == 3:
                    train_loss.append(float(parts[1].strip()))
                    val_loss.append(float(parts[2].strip()))

            # Parse Accuracy
            if "Accuracy" in line and "|" in line:
                parts = line.split('|')
                if len(parts) == 3:
                    train_acc.append(float(parts[1].strip()))
                    val_acc.append(float(parts[2].strip()))
        
        if not epochs:
            logger.warning(f"No metric data found in {log_file_path}. Skipping plot.")
            return

        # Plotting
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss Subplot
        axes[0].plot(epochs, train_loss, label='Train Loss', marker='o')
        axes[0].plot(epochs, val_loss, label='Val Loss', marker='o')
        axes[0].set_title(f'{model_name}: Loss Curve')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)

        # Accuracy Subplot
        axes[1].plot(epochs, train_acc, label='Train Acc', marker='o')
        axes[1].plot(epochs, val_acc, label='Val Acc', marker='o')
        axes[1].set_title(f'{model_name}: Accuracy Curve')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True)

        # Save with unique name
        save_name = f'curves_{model_name}_{run_timestamp}.png'
        save_path = os.path.join(CONFIG['output_dir'], save_name)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close() # Close figure to free memory
        logger.info(f"Saved curves to: {save_path}")

    except Exception as e:
        logger.error(f"Failed to plot curves for {log_file_path}: {e}")

def plot_confusion_matrix(model_path, model_name, run_timestamp):
    """
    Loads the trained model, evaluates on Test set, and plots Confusion Matrix.
    """
    logger.info(f"Generating confusion matrix for: {model_name} ({run_timestamp})...")
    
    try:
        # 1. Load Model Architecture
        # Note: We assume num_classes=2 as per project config
        model = timm.create_model(model_name, pretrained=False, num_classes=CONFIG['num_classes'])
        
        # 2. Load Weights
        state_dict = torch.load(model_path, map_location=CONFIG['device'])
        model.load_state_dict(state_dict)
        model.to(CONFIG['device'])
        model.eval()

        # 3. Load Test Data (No Augmentation for testing)
        # We assume the 'test' folder exists inside 'data_path'
        _, _, test_loader = get_dataloaders(CONFIG['data_path'], batch_size=32, use_augmentation=False)

        all_preds = []
        all_labels = []

        # 4. Run Inference
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(CONFIG['device'])
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())

        # 5. Generate Matrix
        cm = confusion_matrix(all_labels, all_preds)
        
        # 6. Plot Heatmap
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Normal', 'Pneumonia'],
                    yticklabels=['Normal', 'Pneumonia'])
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title(f'Confusion Matrix: {model_name}')
        
        save_name = f'confusion_matrix_{model_name}_{run_timestamp}.png'
        save_path = os.path.join(CONFIG['output_dir'], save_name)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        logger.info(f"Saved confusion matrix to: {save_path}")

    except Exception as e:
        logger.error(f"Failed to generate confusion matrix for {model_path}: {e}")

# ==========================================
# 4. MAIN EXECUTION LOOP
# ==========================================
if __name__ == "__main__":
    logger.info("="*50)
    logger.info("STARTING AUTOMATED VISUALIZATION PIPELINE")
    logger.info("="*50)

    # 1. Find all log files in the logs directory
    # Pattern: logs/training_log_*.txt
    log_search_pattern = os.path.join(CONFIG['logs_dir'], "training_log_*.txt")
    log_files = glob.glob(log_search_pattern)
    
    if not log_files:
        logger.warning(f"No log files found in {CONFIG['logs_dir']}!")
    else:
        logger.info(f"Found {len(log_files)} log files. Processing...")

    # 2. Iterate over each log file
    for log_file in log_files:
        logger.info("-" * 30)
        
        # Extract metadata (model name and unique timestamp)
        model_name, timestamp = parse_filename_metadata(log_file)
        
        if not model_name or not timestamp:
            continue
            
        # A. Plot Training Curves (Always possible if log exists)
        plot_training_curves(log_file, model_name, timestamp)
        
        # B. Plot Confusion Matrix (Only if matching model file exists)
        # Look for: models/best_model_{model_name}_{timestamp}.pth
        # Or check in root directory if not in models/
        
        expected_model_name = f"best_model_{model_name}_{timestamp}.pth"
        model_path_in_dir = os.path.join(CONFIG['models_dir'], expected_model_name)
        model_path_in_root = expected_model_name
        
        final_model_path = None
        if os.path.exists(model_path_in_dir):
            final_model_path = model_path_in_dir
        elif os.path.exists(model_path_in_root):
            final_model_path = model_path_in_root
            
        if final_model_path:
            logger.info(f"Found matching model: {final_model_path}")
            plot_confusion_matrix(final_model_path, model_name, timestamp)
        else:
            logger.warning(f"No matching model file found for {log_file}. Skipping Confusion Matrix.")
            logger.info(f"(Expected: {expected_model_name})")

    logger.info("="*50)
    logger.info("VISUALIZATION COMPLETE. Check 'results/plots' folder.")
    logger.info(f"Log of this run saved to: {CONFIG['logs_dir']}")