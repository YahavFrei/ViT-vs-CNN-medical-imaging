#  train_v2

import os
import sys
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import timm  # Library for State-of-the-Art models
from tqdm import tqdm  # Progress bar library
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from datetime import datetime  # For unique timestamps

# IMPORTANT: Importing from the NEW dataset file with 80/20 split
from dataset import get_dataloaders

# ==========================================
# 1. CONFIGURATION & HYPERPARAMETERS
# ==========================================
CONFIG = {
    # Batch Size: 32 is standard for 224x224 images on modern GPUs.
    'batch_size': 32,
    
    # Learning Rate: Default is 1e-4. 
    # NOTE: You should update this manually based on Optuna results for each model!
    'learning_rate': 1e-4, 
    
    # Epochs: Number of times to iterate over the entire dataset.
    'epochs': 5,
    
    # Classes: 0 = Normal, 1 = Pneumonia
    'num_classes': 2,
    
    # Device: Automatically detects if NVIDIA GPU is available
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    
    # Path to your data directory
    'data_path': '/home/projects/cgm-prj10354/ViT-vs-CNN-medical-imaging/data'
}

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def setup_logger(model_name, timestamp):
    """
    Sets up a logger that writes to both console and a text file.
    Uses a timestamp to ensure unique filenames for every run.
    """
    # Create a unique filename: e.g., training_log_resnet50_20260119_103000.txt
    log_filename = f"training_log_{model_name}_{timestamp}.txt"
    
    logger = logging.getLogger(f"{model_name}_{timestamp}")
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers to prevent duplicate lines
    if logger.hasHandlers():
        logger.handlers.clear()

    # File Handler: Saves logs to disk
    file_handler = logging.FileHandler(log_filename, mode='w')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    
    # Stream Handler: Prints to terminal
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(logging.Formatter('%(message)s'))

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    
    return logger

def get_model(model_name, num_classes, logger):
    """
    Loads a pre-trained model architecture using 'timm'.
    """
    logger.info(f"Downloading/Loading model architecture: {model_name}...")
    
    # pretrained=True: Downloads weights trained on ImageNet.
    # This is 'Transfer Learning' - giving the model a head start.
    model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
    
    return model

def calculate_metrics(y_true, y_pred):
    """
    Calculates key medical metrics using Scikit-Learn.
    """
    # Precision: Accuracy of positive predictions (How many diagnosed are actually sick?)
    precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
    
    # Recall: Sensitivity (How many sick people did we correctly identify?)
    # CRITICAL for medical screening.
    recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
    
    # F1 Score: Balance between Precision and Recall.
    f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
    
    return precision, recall, f1

# ==========================================
# 3. TRAINING & VALIDATION LOOPS
# ==========================================

def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Training Loop: Updates model weights.
    """
    model.train()  # Enable BatchNorm and Dropout
    
    running_loss = 0.0
    all_preds = []
    all_labels = []

    loop = tqdm(loader, desc="Training", leave=False)
    
    for images, labels in loop:
        images = images.to(device)
        labels = labels.to(device)

        # 1. Forward Pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 2. Backward Pass (Gradient Descent)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 3. Stats
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)

        all_preds.extend(predicted.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

        loop.set_postfix(loss=loss.item())

    # Calculate metrics for the epoch
    epoch_loss = running_loss / len(all_labels)
    epoch_acc = np.mean(np.array(all_preds) == np.array(all_labels))
    precision, recall, f1 = calculate_metrics(all_labels, all_preds)

    return epoch_loss, epoch_acc, precision, recall, f1

def validate(model, loader, criterion, device):
    """
    Validation Loop: Evaluates performance without updating weights.
    """
    model.eval()  # Disable BatchNorm and Dropout
    
    running_loss = 0.0
    all_preds = []
    all_labels = []

    # 'torch.no_grad' saves memory and computation
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validating", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(all_labels)
    epoch_acc = np.mean(np.array(all_preds) == np.array(all_labels))
    precision, recall, f1 = calculate_metrics(all_labels, all_preds)

    return epoch_loss, epoch_acc, precision, recall, f1

# ==========================================
# 4. MAIN EXECUTION FLOW
# ==========================================

#def main(model_name='resnet50', learning_rate=1e-4, use_aug=True, optimizer_type='adamw'):
def main(model_name='resnet50', learning_rate=1e-4, batch_size=32, use_aug=True, optimizer_type='adamw'):
    """
    Main function to run the full training pipeline.
    Args:
        model_name: Name of model (resnet50, densenet121, vit...)
        learning_rate: LR to use (can be from Optuna)
        use_aug: Boolean, whether to use augmentation (for ablation study)
        optimizer_type: 'adamw' or 'sgd' (for ablation study)
    """
    
    # Generate Timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Setup Logging
    logger = setup_logger(model_name, timestamp)
    
    logger.info(f"\n{'='*50}")
    logger.info(f"STARTING PIPELINE: {model_name}")
    logger.info(f"Timestamp: {timestamp}")
    logger.info(f"{'='*50}")
    
    # Print Config
    logger.info("CONFIGURATION:")
    logger.info(f"  - Model: {model_name}")
    logger.info(f"  - LR: {learning_rate}")
    logger.info(f"  - Augmentation: {use_aug}")
    logger.info(f"  - Optimizer: {optimizer_type}")
    logger.info(f"  - Device: {CONFIG['device']}")

    # A. Load Data (Using dataset_v2)
    logger.info("\nLoading DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=CONFIG['data_path'], 
        batch_size= batch_size, # CONFIG['batch_size'],
        use_augmentation=use_aug  # Pass the flag to the dataset
    )

    # B. Load Model
    model = get_model(model_name, CONFIG['num_classes'], logger)
    model = model.to(CONFIG['device'])

    # C. Loss Function (Weighted for Imbalance)
    # Weights: [Normal=3.0, Pneumonia=1.0] to prioritize the minority class
    class_weights = torch.tensor([3.0, 1.0]).to(CONFIG['device'])
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # D. Optimizer Selection
    if optimizer_type == 'sgd':
        logger.info("Using SGD Optimizer")
        # SGD usually needs higher LR and momentum
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    else:
        logger.info("Using AdamW Optimizer")
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # E. Training Loop
    best_val_f1 = 0.0
    # Unique filename for saving weights
    save_path = f"best_model_{model_name}_{timestamp}.pth"
    
    for epoch in range(CONFIG['epochs']):
        logger.info(f"\nEpoch {epoch+1}/{CONFIG['epochs']}")
        
        # Train
        t_loss, t_acc, t_prec, t_recall, t_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, CONFIG['device']
        )
        
        # Validate
        v_loss, v_acc, v_prec, v_recall, v_f1 = validate(
            model, val_loader, criterion, CONFIG['device']
        )
        
        # Log Table
        logger.info(f"{'-'*65}")
        logger.info(f"{'METRIC':<10} | {'TRAIN':<10} | {'VALIDATION':<10}")
        logger.info(f"{'-'*65}")
        logger.info(f"{'Loss':<10} | {t_loss:.4f}     | {v_loss:.4f}")
        logger.info(f"{'Accuracy':<10} | {t_acc:.4f}     | {v_acc:.4f}")
        logger.info(f"{'F1 Score':<10} | {t_f1:.4f}     | {v_f1:.4f}")
        logger.info(f"{'Recall':<10} | {t_recall:.4f}     | {v_recall:.4f}")
        logger.info(f"{'Precision':<10} | {t_prec:.4f}     | {v_prec:.4f}")
        logger.info(f"{'-'*65}")

        # Save Best Model (Based on F1 Score)
        if v_f1 > best_val_f1:
            best_val_f1 = v_f1
            torch.save(model.state_dict(), save_path)
            logger.info(f"--> 🏆 New Best Model saved to {save_path}")

    logger.info("\n--- Training Loop Complete ---\n")

    # F. Final Test Evaluation
    logger.info(f"{'='*50}")
    logger.info(f"FINAL TEST: EVALUATING BEST MODEL ON UNSEEN DATA")
    logger.info(f"{'='*50}")

    logger.info(f"Loading best weights from: {save_path}")
    model.load_state_dict(torch.load(save_path))
    model.eval()

    test_loss, test_acc, test_prec, test_recall, test_f1 = validate(
        model, test_loader, criterion, CONFIG['device']
    )

    logger.info(f"\n📢 FINAL RESULTS FOR {model_name.upper()}:")
    logger.info(f"Accuracy:  {test_acc:.4f}")
    logger.info(f"Recall:    {test_recall:.4f}")
    logger.info(f"Precision: {test_prec:.4f}")
    logger.info(f"F1 Score:  {test_f1:.4f}")
    logger.info(f"{'='*50}\n")


if __name__ == "__main__":
    # # ==========================================
    # # 1. FINAL OPTIMIZED MODELS ("The Winners")
    # # ==========================================
    
    # # Run 1: DenseNet121 (Best Acc: 97.89%)
    # # Params: LR=0.000615, BS=32
    # print("--- Starting Final DenseNet Run ---")
    # main(model_name='densenet121', learning_rate=0.000615, batch_size=32)

    # # Run 2: ResNet50 (Best Acc: 96.93%)
    # # Params: LR=0.000260, BS=16
    # print("--- Starting Final ResNet Run ---")
    # main(model_name='resnet50', learning_rate=0.000260, batch_size=16)

    # # Run 3: ViT Base (Best Acc: 95.97%)
    # # Params: LR=0.000011, BS=16
    # print("--- Starting Final ViT Run ---")
    # main(model_name='vit_base_patch16_224', learning_rate=0.000011, batch_size=16)

    # # ==========================================
    # # 2. ABLATION STUDIES (For Project Proposal)
    # # ==========================================
    
    # # Study A: ViT and 2 other without Augmentation (To show overfitting)
    # # We use the optimized LR to be fair, but turn off augmentation.
    # print("--- Starting ViT No-Augmentation Study ---")
    # main(model_name='vit_base_patch16_224', learning_rate=0.000011, batch_size=16, use_aug=False)

    # print("--- Starting densenet121 No-Augmentation Study ---")
    # main(model_name='densenet121', learning_rate=0.000615, batch_size=32,use_aug=False)

    # print("--- Starting ResNet No-Augmentation Study ---")
    # main(model_name='resnet50', learning_rate=0.000260, batch_size=16,use_aug=False)

    # # Study B: ResNet with SGD Optimizer (To compare vs AdamW)
    # # SGD usually needs a higher LR (e.g. 0.01 or 0.001) to work well.
    # print("--- Starting ResNet SGD Study ---")
    # main(model_name='resnet50', learning_rate=0.001, batch_size=16, optimizer_type='sgd')
    # print("--- Starting ViT SGD Study ---")
    main(model_name='vit_base_patch16_224', learning_rate=0.001, batch_size=16, optimizer_type='sgd')

# if __name__ == "__main__":
#     # ==========================================
#     # USER CONTROLS: UNCOMMENT THE RUN YOU NEED
#     # ==========================================

#     # --- RUN 1: BASELINE (Before Optuna) ---
#     # main(model_name='resnet50', learning_rate=1e-4)
#     # main(model_name='densenet121', learning_rate=1e-4)
#     main(model_name='vit_base_patch16_224', learning_rate=1e-4)

#     # --- RUN 2: OPTIMIZED (After Optuna results) ---
#     # Update the learning_rate below based on what Optuna finds!
#     # main(model_name='resnet50', learning_rate=1e-4) 

#     # --- RUN 3: ABLATION STUDIES (To satisfy Project Proposal) ---
    
#     # A. ViT without Augmentation
#     # main(model_name='vit_base_patch16_224', learning_rate=1e-4, use_aug=False)
    
#     # B. SGD Optimizer
#     # main(model_name='resnet50', learning_rate=0.01, optimizer_type='sgd')




#####################################################################




















# import os
# import sys
# import logging
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import timm  # Library containing State-of-the-Art models (ResNet, ViT, etc.)
# from tqdm import tqdm  # Library for the progress bar in the terminal
# import numpy as np
# from sklearn.metrics import precision_score, recall_score, f1_score

# # Import the data loading function from our 'dataset.py' file
# from dataset import get_dataloaders

# # ==========================================
# # 1. CONFIGURATION & HYPERPARAMETERS
# # ==========================================
# CONFIG = {
#     # Batch Size: 32 is a safe balance for memory usage and training stability.
#     # Your RTX 3090 handles this easily.
#     'batch_size': 32,
    
#     # Learning Rate: 1e-4 (0.0001) is standard for Fine-Tuning.
#     # We don't want to destroy the pre-trained weights with large updates.
#     'learning_rate': 1e-4,
    
#     # Epochs: 5 is usually enough for Transfer Learning to converge.
#     'epochs': 5,
    
#     # Classes: 0 = Normal, 1 = Pneumonia
#     'num_classes': 2,
    
#     # Device: Automatically use GPU (CUDA) if available
#     'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    
#     # Path: Absolute path to ensure the code always finds the data folder
#     'data_path': '/home/projects/cgm-prj10354/ViT-vs-CNN-medical-imaging/data'
# }

# # ==========================================
# # 2. HELPER FUNCTIONS
# # ==========================================

# def setup_logger(model_name):
#     """
#     Sets up a logger to write training results to both:
#     1. The Terminal (Console)
#     2. A Text File (e.g., training_log_resnet50.txt)
#     """
#     log_filename = f"training_log_{model_name}.txt"
    
#     # Initialize logger
#     logger = logging.getLogger(model_name)
#     logger.setLevel(logging.INFO)
    
#     # Clear any existing handlers to prevent duplicate printing
#     if logger.hasHandlers():
#         logger.handlers.clear()

#     # Handler for writing to File
#     file_handler = logging.FileHandler(log_filename, mode='w')
#     file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    
#     # Handler for writing to Terminal
#     stream_handler = logging.StreamHandler(sys.stdout)
#     stream_handler.setFormatter(logging.Formatter('%(message)s'))

#     # Add both handlers to the logger
#     logger.addHandler(file_handler)
#     logger.addHandler(stream_handler)
    
#     return logger

# def get_model(model_name, num_classes, logger):
#     """
#     Loads a pre-trained model using the 'timm' library.
#     """
#     logger.info(f"Downloading/Loading model architecture: {model_name}...")
    
#     # pretrained=True: Downloads weights learned on ImageNet (Transfer Learning).
#     # This gives the model a huge advantage compared to training from scratch.
#     model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
    
#     return model

# def calculate_metrics(y_true, y_pred):
#     """
#     Calculates Medical Metrics using Scikit-Learn.
#     We focus on the 'Positive' class (Pneumonia).
#     """
#     # Precision: Out of all predicted Pneumonia, how many were actually Pneumonia?
#     precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
    
#     # Recall (Sensitivity): Out of all REAL Pneumonia cases, how many did we catch?
#     # This is critical in medicine - we don't want to miss sick patients.
#     recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
    
#     # F1 Score: Harmonic mean of Precision and Recall. Good for comparing models.
#     f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
    
#     return precision, recall, f1

# # ==========================================
# # 3. TRAINING & VALIDATION LOOPS
# # ==========================================

# def train_one_epoch(model, loader, criterion, optimizer, device):
#     """
#     Runs one full pass over the training dataset.
#     Updates the model weights based on the loss.
#     """
#     model.train()  # Switch to training mode (Enable Dropout/BatchNorm)
    
#     running_loss = 0.0
#     all_preds = []
#     all_labels = []

#     # tqdm wrapper for the visual progress bar
#     loop = tqdm(loader, desc="Training", leave=False)
    
#     for images, labels in loop:
#         # Move data to GPU
#         images = images.to(device)
#         labels = labels.to(device)

#         # A. Forward Pass (Model Prediction)
#         outputs = model(images)
#         loss = criterion(outputs, labels)

#         # B. Backward Pass (Optimization)
#         optimizer.zero_grad() # Clear gradients from previous step
#         loss.backward()       # Calculate new gradients
#         optimizer.step()      # Update model weights

#         # C. Collect Statistics
#         running_loss += loss.item() * images.size(0)
#         _, predicted = torch.max(outputs, 1)

#         # Move predictions to CPU to save GPU memory and store in list
#         all_preds.extend(predicted.detach().cpu().numpy())
#         all_labels.extend(labels.detach().cpu().numpy())

#         # Update progress bar
#         loop.set_postfix(loss=loss.item())

#     # Calculate metrics for the entire epoch
#     epoch_loss = running_loss / len(all_labels)
#     epoch_acc = np.mean(np.array(all_preds) == np.array(all_labels))
#     precision, recall, f1 = calculate_metrics(all_labels, all_preds)

#     return epoch_loss, epoch_acc, precision, recall, f1

# def validate(model, loader, criterion, device):
#     """
#     Evaluates the model on Validation or Test data.
#     Does NOT update weights.
#     """
#     model.eval()  # Switch to evaluation mode (Disable Dropout)
    
#     running_loss = 0.0
#     all_preds = []
#     all_labels = []

#     # torch.no_grad(): Disables gradient calculation to save memory and speed up
#     with torch.no_grad():
#         for images, labels in tqdm(loader, desc="Validating", leave=False):
#             images = images.to(device)
#             labels = labels.to(device)

#             outputs = model(images)
#             loss = criterion(outputs, labels)

#             running_loss += loss.item() * images.size(0)
#             _, predicted = torch.max(outputs, 1)

#             all_preds.extend(predicted.cpu().numpy())
#             all_labels.extend(labels.cpu().numpy())

#     # Calculate metrics
#     epoch_loss = running_loss / len(all_labels)
#     epoch_acc = np.mean(np.array(all_preds) == np.array(all_labels))
#     precision, recall, f1 = calculate_metrics(all_labels, all_preds)

#     return epoch_loss, epoch_acc, precision, recall, f1

# # ==========================================
# # 4. MAIN EXECUTION FLOW
# # ==========================================

# def main(model_name='resnet50'):
#     # A. Setup Logger
#     logger = setup_logger(model_name)
    
#     logger.info(f"\n{'='*50}")
#     logger.info(f"STARTING TRAINING PIPELINE: {model_name}")
#     logger.info(f"{'='*50}")
    
#     # B. Print Configuration for transparency
#     logger.info("CONFIGURATION:")
#     for key, value in CONFIG.items():
#         logger.info(f"  - {key}: {value}")
#     logger.info(f"{'='*50}\n")
    
#     # C. Load Data
#     logger.info("Loading DataLoaders...")
#     train_loader, val_loader, test_loader = get_dataloaders(
#         root_dir=CONFIG['data_path'], 
#         batch_size=CONFIG['batch_size'], 
#         img_size=224 # Standard input size for ResNet/ViT
#     )

#     # D. Initialize Model
#     model = get_model(model_name, CONFIG['num_classes'], logger)
#     model = model.to(CONFIG['device'])

#     # E. Define Loss Function & Optimizer
#     # Class Weights: We have ~3x more Pneumonia (1) than Normal (0).
#     # We assign weight 3.0 to Normal to penalize the model more if it misses them.
#     # This solves the Class Imbalance problem.
#     class_weights = torch.tensor([3.0, 1.0]).to(CONFIG['device'])
#     criterion = nn.CrossEntropyLoss(weight=class_weights)
    
#     # AdamW is the standard optimizer for ViT and modern CNNs
#     optimizer = optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'])

#     # F. Training Loop
#     best_val_f1 = 0.0 # Metric to track the best model
#     save_path = f"best_model_{model_name}.pth"
    
#     for epoch in range(CONFIG['epochs']):
#         logger.info(f"\nEpoch {epoch+1}/{CONFIG['epochs']}")
        
#         # 1. Train Step
#         t_loss, t_acc, t_prec, t_recall, t_f1 = train_one_epoch(
#             model, train_loader, criterion, optimizer, CONFIG['device']
#         )
        
#         # 2. Validation Step
#         v_loss, v_acc, v_prec, v_recall, v_f1 = validate(
#             model, val_loader, criterion, CONFIG['device']
#         )
        
#         # 3. Log Results (Table Format)
#         logger.info(f"{'-'*65}")
#         logger.info(f"{'METRIC':<10} | {'TRAIN':<10} | {'VALIDATION':<10}")
#         logger.info(f"{'-'*65}")
#         logger.info(f"{'Loss':<10} | {t_loss:.4f}     | {v_loss:.4f}")
#         logger.info(f"{'Accuracy':<10} | {t_acc:.4f}     | {v_acc:.4f}")
#         logger.info(f"{'F1 Score':<10} | {t_f1:.4f}     | {v_f1:.4f}")
#         logger.info(f"{'Recall':<10} | {t_recall:.4f}     | {v_recall:.4f}")
#         logger.info(f"{'Precision':<10} | {t_prec:.4f}     | {v_prec:.4f}")
#         logger.info(f"{'-'*65}")

#         # 4. Save Best Model
#         # We check if current F1 score is better than the best so far
#         if v_f1 > best_val_f1:
#             best_val_f1 = v_f1
#             torch.save(model.state_dict(), save_path)
#             logger.info(f"--> 🏆 New Best Model saved to {save_path}")

#     logger.info("\n--- Training Loop Complete ---\n")

#     # G. Final Test Evaluation
#     # We load the BEST saved model and run it on the TEST set (Unseen data)
#     logger.info(f"{'='*50}")
#     logger.info(f"FINAL TEST: EVALUATING BEST MODEL ON UNSEEN DATA")
#     logger.info(f"{'='*50}")

#     logger.info(f"Loading best weights from: {save_path}")
#     model.load_state_dict(torch.load(save_path))
#     model.eval()

#     test_loss, test_acc, test_prec, test_recall, test_f1 = validate(
#         model, test_loader, criterion, CONFIG['device']
#     )

#     logger.info(f"\n📢 FINAL RESULTS FOR {model_name.upper()}:")
#     logger.info(f"Accuracy:  {test_acc:.4f}")
#     logger.info(f"Recall:    {test_recall:.4f}")
#     logger.info(f"Precision: {test_prec:.4f}")
#     logger.info(f"F1 Score:  {test_f1:.4f}")
#     logger.info(f"{'='*50}\n")

# if __name__ == "__main__":
#     # ==========================================
#     # CHOOSE YOUR FIGHTER (MODEL)
#     # ==========================================
    
#     # 1. First Run: The Baseline (CNN)
#     #main(model_name='resnet50')
    
#     # 2. Second Run: The Challenger (ViT)
#     # After the first run finishes, comment out the line above
#     # and uncomment the line below:
    
#     #main(model_name='vit_base_patch16_224')

#     main(model_name='densenet121')