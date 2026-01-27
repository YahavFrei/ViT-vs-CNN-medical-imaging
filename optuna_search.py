# optuna_search_v2 19/01

import os
import sys
import logging
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import timm
from datetime import datetime  # Added for unique timestamps
from dataset import get_dataloaders  # Importing from the new 80/20 split dataset

# ==========================================
# 1. CONFIGURATION
# ==========================================

# --- USER CONTROL PANEL ---
# Options: 'resnet50', 'densenet121', 'vit_base_patch16_224'
MODEL_TO_OPTIMIZE = 'densenet121' 

# Path to the data directory (root folder containing 'train' and 'test')
DATA_PATH = '/home/projects/cgm-prj10354/ViT-vs-CNN-medical-imaging/data'

# Device configuration (Use GPU if available for faster training)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Logging configuration: dynamic filename with TIMESTAMP to prevent overwriting
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = f'optuna_log_{MODEL_TO_OPTIMIZE}_{timestamp}.txt' 

# --- OPTIMIZATION SETTINGS ---
# Number of distinct hyperparameter combinations to try.
# 30 is a good balance between thoroughness and runtime.
N_TRIALS = 30          

# Number of training epochs per trial.
# Since we use Transfer Learning, the model converges quickly. 
# 4 epochs are sufficient to judge if a set of parameters is promising.
EPOCHS_PER_TRIAL = 4   

# ==========================================
# 2. LOGGER SETUP
# ==========================================
def setup_logger():
    """
    Sets up a logger to write results to both a text file and the console.
    This ensures we have a permanent record of the experiment (log file)
    and real-time feedback (console).
    """
    # Use specific logger name to avoid conflicts
    logger = logging.getLogger(f'Optuna_{MODEL_TO_OPTIMIZE}_{timestamp}')
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers to avoid duplicate logs if run multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    # File Handler: Saves logs to disk
    file_handler = logging.FileHandler(LOG_FILE, mode='w')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    
    # Stream Handler: Prints logs to the terminal
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(logging.Formatter('%(message)s'))

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger

logger = setup_logger()

# ==========================================
# 3. OPTUNA OBJECTIVE FUNCTION
# ==========================================
def objective(trial):
    """
    The main objective function for Optuna. 
    This function represents one single 'experiment' or 'trial'.
    
    Optuna will:
    1. Call this function.
    2. Provide a 'trial' object to suggest hyperparameters.
    3. Run a short training loop.
    4. Return the validation accuracy.
    
    Optuna's goal is to find parameters that maximize this return value.
    """
    
    # --- A. Suggest Hyperparameters ---
    # We ask Optuna to suggest a Learning Rate (logarithmic scale)
    # Range: 0.00001 to 0.001
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    
    # We ask Optuna to choose a Batch Size from the defined list
    batch_size = trial.suggest_categorical("batch_size", [16, 32])
    
    # Log the current trial parameters
    logger.info(f"\n--- Trial {trial.number} for {MODEL_TO_OPTIMIZE} ---")
    logger.info(f"Hyperparameters: LR={lr:.6f} | Batch_Size={batch_size}")

    # --- B. Setup Data ---
    # Load data using the new split logic (90% Train / 10% Val) from dataset.py.
    # We enable augmentation to simulate real training conditions.
    train_loader, val_loader, _ = get_dataloaders(DATA_PATH, batch_size=batch_size, use_augmentation=True)

    # --- C. Setup Model ---
    # Initialize the specific model architecture (Pre-trained on ImageNet)
    model = timm.create_model(MODEL_TO_OPTIMIZE, pretrained=True, num_classes=2)
    model = model.to(DEVICE)

    # --- D. Loss Function & Optimizer ---
    # We use Weighted Cross Entropy to handle the class imbalance.
    # 'Normal' has fewer images, so we give it a weight of 3.0 to penalize errors more.
    class_weights = torch.tensor([3.0, 1.0]).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # AdamW is generally the best optimizer for Transformers and modern CNNs
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    # --- E. Mini-Training Loop ---
    for epoch in range(EPOCHS_PER_TRIAL):
        # 1. Training Phase
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # 2. Validation Phase
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        # Calculate Validation Accuracy
        val_acc = correct / total
        
        # 3. Reporting to Optuna
        # We report the accuracy of the current epoch so Optuna can track progress.
        trial.report(val_acc, epoch)

        # 4. Pruning (Early Stopping)
        # If the current trial is performing significantly worse than previous best trials,
        # Optuna will throw a TrialPruned exception to stop it immediately and save time.
        if trial.should_prune():
            logger.info(f"Trial {trial.number} Pruned (Stopped early due to poor performance).")
            raise optuna.exceptions.TrialPruned()

    # Log the final result of this trial
    logger.info(f"Trial {trial.number} Finished. Final Validation Accuracy: {val_acc:.4f}")
    
    # Return the value we want to maximize
    return val_acc

# ==========================================
# 4. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    logger.info(f"Starting Optuna Hyperparameter Search for Model: {MODEL_TO_OPTIMIZE}")
    logger.info(f"Configuration: {N_TRIALS} Trials, {EPOCHS_PER_TRIAL} Epochs per Trial")
    
    # Create the study. Direction is 'maximize' because we want high Accuracy.
    study = optuna.create_study(direction="maximize")
    
    # Start the optimization process
    study.optimize(objective, n_trials=N_TRIALS)

    # --- Summary ---
    logger.info("="*40)
    logger.info(f"OPTIMIZATION COMPLETE FOR {MODEL_TO_OPTIMIZE}")
    logger.info("="*40)
    logger.info(f"Best Accuracy Achieved: {study.best_value:.4f}")
    logger.info("Best Hyperparameters Found:")
    for key, value in study.best_params.items():
        logger.info(f"  - {key}: {value}")
    
    logger.info(f"Full log has been saved to: {LOG_FILE}")
    logger.info("="*40)


