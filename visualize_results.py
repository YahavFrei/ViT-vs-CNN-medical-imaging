import os
import re
import sys
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import timm
from dataset import get_dataloaders

# ==========================================
# 1. CONFIGURATION
# ==========================================
# CHANGE THIS to 'resnet50', 'densenet121', or 'vit_base_patch16_224'
MODEL_NAME = 'vit_base_patch16_224' 

DATA_PATH = '/home/projects/cgm-prj10354/ViT-vs-CNN-medical-imaging/data'
LOG_FILE_SOURCE = f'training_log_{MODEL_NAME}.txt' # The log file created during training
MODEL_PATH = f'best_model_{MODEL_NAME}.pth'       # The best weights saved
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Output log for this analysis script
ANALYSIS_LOG = f'analysis_results_{MODEL_NAME}.txt'

# ==========================================
# 2. LOGGER SETUP
# ==========================================
def setup_logger():
    """Sets up a logger that writes to both console and file."""
    logger = logging.getLogger('Analysis')
    logger.setLevel(logging.INFO)
    
    if logger.hasHandlers():
        logger.handlers.clear()

    # File Handler
    file_handler = logging.FileHandler(ANALYSIS_LOG, mode='w')
    file_handler.setFormatter(logging.Formatter('%(message)s'))
    
    # Stream Handler (Console)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(logging.Formatter('%(message)s'))

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger

logger = setup_logger()

# ==========================================
# 3. FUNCTIONS
# ==========================================

def plot_training_curves():
    """Reads the training log and plots Loss/Accuracy graphs."""
    logger.info(f"--- 1. Generating Training Curves for {MODEL_NAME} ---")
    
    if not os.path.exists(LOG_FILE_SOURCE):
        logger.error(f"Error: Could not find log file: {LOG_FILE_SOURCE}")
        return

    with open(LOG_FILE_SOURCE, 'r') as f:
        content = f.read()

    # Regex to extract metrics from the log table
    loss_pattern = re.compile(r"Loss\s+\|\s+([\d\.]+)\s+\|\s+([\d\.]+)")
    acc_pattern = re.compile(r"Accuracy\s+\|\s+([\d\.]+)\s+\|\s+([\d\.]+)")

    losses = loss_pattern.findall(content)
    accs = acc_pattern.findall(content)

    if not losses:
        logger.error("No data found in log file. Check format.")
        return

    train_loss = [float(t) for t, v in losses]
    val_loss = [float(v) for t, v in losses]
    train_acc = [float(t) for t, v in accs]
    val_acc = [float(v) for t, v in accs]
    epochs = range(1, len(train_loss) + 1)

    # Create Plot
    plt.figure(figsize=(12, 5))

    # Loss Subplot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label='Train Loss', marker='o')
    plt.plot(epochs, val_loss, label='Val Loss', marker='o')
    plt.title(f'Loss Curve: {MODEL_NAME}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Accuracy Subplot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, label='Train Accuracy', marker='o')
    plt.plot(epochs, val_acc, label='Val Accuracy', marker='o')
    plt.title(f'Accuracy Curve: {MODEL_NAME}')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    save_path = f"plot_{MODEL_NAME}.png"
    plt.savefig(save_path)
    logger.info(f"✔ Graphs saved to: {save_path}")
    # plt.show() # Uncomment if running locally with a screen

def generate_confusion_matrix():
    """Runs the model on Test data and generates Confusion Matrix."""
    logger.info(f"\n--- 2. Generating Confusion Matrix for {MODEL_NAME} ---")
    logger.info(f"Model Path: {MODEL_PATH}")
    logger.info(f"Data Path: {DATA_PATH}")

    # Load Data
    _, _, test_loader = get_dataloaders(DATA_PATH, batch_size=32)

    # Load Model
    try:
        model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=2)
        model.load_state_dict(torch.load(MODEL_PATH))
        model.to(DEVICE)
        model.eval()
        logger.info("✔ Model loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return

    all_preds = []
    all_labels = []

    # Inference Loop
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Create Matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Save textual report
    logger.info("\nClassification Report:")
    logger.info(classification_report(all_labels, all_preds, target_names=['Normal', 'Pneumonia']))
    
    logger.info("Confusion Matrix Raw Data:")
    logger.info(f"True Negatives (Normal correctly identified): {cm[0][0]}")
    logger.info(f"False Positives (Normal predicted as Pneumonia): {cm[0][1]}")
    logger.info(f"False Negatives (Pneumonia predicted as Normal): {cm[1][0]}")
    logger.info(f"True Positives (Pneumonia correctly identified): {cm[1][1]}")

    # Plot Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Pneumonia'], 
                yticklabels=['Normal', 'Pneumonia'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix - {MODEL_NAME}')
    
    save_path = f"confusion_matrix_{MODEL_NAME}.png"
    plt.savefig(save_path)
    logger.info(f"✔ Matrix Image saved to: {save_path}")

if __name__ == "__main__":
    logger.info(f"=== ANALYSIS START: {MODEL_NAME} ===")
    plot_training_curves()
    generate_confusion_matrix()
    logger.info("\n=== ANALYSIS COMPLETE ===")
    logger.info(f"Full report saved to: {ANALYSIS_LOG}")