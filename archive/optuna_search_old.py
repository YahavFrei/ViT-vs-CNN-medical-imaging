import os
import sys
import logging
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import timm
from dataset import get_dataloaders

# ==========================================
# 1. CONFIGURATION
# ==========================================
DATA_PATH = '/home/projects/cgm-prj10354/ViT-vs-CNN-medical-imaging/data'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LOG_FILE = 'optuna_log.txt'

# --- CONTROL PANEL ---
# Set this to True to check if code works (runs in ~2 mins)
# Set this to False to run the full optimization overnight
QUICK_TEST_MODE = False 

if QUICK_TEST_MODE:
    N_TRIALS = 2         # Only try 2 combinations
    EPOCHS_PER_TRIAL = 1 # Only 1 epoch per try
else:
    N_TRIALS = 20        # Try 20 different combinations (Good for overnight)
    EPOCHS_PER_TRIAL = 4 # 4 Epochs is enough to know if parameters are good

# ==========================================
# 2. LOGGER SETUP
# ==========================================
def setup_logger():
    """Sets up a logger that writes to both Console and File."""
    logger = logging.getLogger('OptunaSearch')
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()

    # File Handler (Saves to txt)
    file_handler = logging.FileHandler(LOG_FILE, mode='w')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    
    # Stream Handler (Prints to screen)
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
    This function represents one "experiment".
    Optuna will call this multiple times with different hyperparameter guesses.
    """
    
    # --- A. Suggest Hyperparameters ---
    # 1. Learning Rate: Suggest a float value between 0.00001 and 0.001 (log scale)
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    
    # 2. Batch Size: Pick either 16 or 32
    batch_size = trial.suggest_categorical("batch_size", [16, 32])
    
    # 3. Model Architecture: Pick one of the models
    model_name = trial.suggest_categorical("model_name", ["resnet50", "densenet121"])

    # Log what we are testing in this trial
    logger.info(f"\n--- Trial {trial.number} ---")
    logger.info(f"Testing: Model={model_name} | LR={lr:.6f} | Batch={batch_size}")

    # --- B. Setup Data & Model ---
    # We use our existing data loader
    train_loader, val_loader, _ = get_dataloaders(DATA_PATH, batch_size=batch_size)

    # Load the model structure (pretrained on ImageNet)
    model = timm.create_model(model_name, pretrained=True, num_classes=2)
    model = model.to(DEVICE)

    # --- C. Setup Optimizer ---
    # Handle Class Imbalance
    class_weights = torch.tensor([3.0, 1.0]).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    # --- D. Mini-Training Loop ---
    # We don't need to save the model or calculate F1 score here.
    # We just need "Accuracy" to know if this configuration is good.
    
    for epoch in range(EPOCHS_PER_TRIAL):
        # 1. Train
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # 2. Validation (Calculate Accuracy)
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
        
        val_acc = correct / total
        
        # Report result to Optuna
        trial.report(val_acc, epoch)

        # --- E. Pruning (Early Stopping) ---
        # If this trial is performing very poorly compared to previous ones, stop it now.
        if trial.should_prune():
            logger.info(f"Trial {trial.number} Pruned (Stopped early due to bad performance).")
            raise optuna.exceptions.TrialPruned()

    logger.info(f"Trial {trial.number} Finished. Validation Accuracy: {val_acc:.4f}")
    return val_acc

# ==========================================
# 4. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    logger.info("Starting Optuna Hyperparameter Search...")
    logger.info(f"Mode: {'QUICK TEST' if QUICK_TEST_MODE else 'FULL OPTIMIZATION'}")
    logger.info(f"Trials: {N_TRIALS} | Epochs per trial: {EPOCHS_PER_TRIAL}")

    # Create a study object which manages the optimization
    study = optuna.create_study(direction="maximize") # We want to MAXIMIZE accuracy
    
    # Start the search
    study.optimize(objective, n_trials=N_TRIALS)

    # --- Summary ---
    logger.info("\n" + "="*40)
    logger.info("OPTIMIZATION COMPLETE")
    logger.info("="*40)
    logger.info(f"Best Trial: {study.best_trial.number}")
    logger.info(f"Best Accuracy: {study.best_value:.4f}")
    logger.info("Best Hyperparameters:")
    for key, value in study.best_params.items():
        logger.info(f"  - {key}: {value}")
    
    logger.info(f"Full log saved to: {LOG_FILE}")