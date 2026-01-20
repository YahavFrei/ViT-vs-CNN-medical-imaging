#dataset_v2

import os
import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

def get_dataloaders(data_dir, batch_size=32, use_augmentation=True, val_split=0.1):
    """
    Creates DataLoaders with a dynamic Train/Val split.
    
    Args:
        data_dir (str): Path to dataset.
        batch_size (int): Batch size.
        use_augmentation (bool): If True, applies random transforms to Training set. 
                                 If False, only resizes (used for ablation studies).
        val_split (float): Percentage of data to use for Validation (default 0.1 = 10%).
    
    Returns:
        train_loader, val_loader, test_loader
    """
    
    # ---------------------------------------------------------
    # 1. Define Transforms
    # ---------------------------------------------------------
    
    # Standard normalization for ImageNet pre-trained models
    normalize = transforms.Normalize([0.485, 0.456, 0.406], 
                                     [0.229, 0.224, 0.225])

    # Base transform (Always applied: Resize + Tensor + Norm)
    base_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])

    # Augmentation transform (Applied only if requested)
    if use_augmentation:
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomRotation(10),           # Rotate +/- 10 degrees
            transforms.RandomHorizontalFlip(p=0.5),  # Flip left-right
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)), # Slight zoom/crop
            transforms.ToTensor(),
            normalize
        ])
    else:
        # If augmentation is OFF, Training data looks just like Validation data
        train_transform = base_transform

    # Validation and Test always use the base transform (No augmentation)
    val_test_transform = base_transform

    # ---------------------------------------------------------
    # 2. Load Datasets & Split
    # ---------------------------------------------------------
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')
    
    # We load the dataset twice:
    # 1. Once with training transforms (for the Train split)
    # 2. Once with validation transforms (for the Val split)
    # This is a trick to ensure Validation images NEVER get augmented.
    full_dataset_for_train = datasets.ImageFolder(train_dir, transform=train_transform)
    full_dataset_for_val   = datasets.ImageFolder(train_dir, transform=val_test_transform)
    
    dataset_size = len(full_dataset_for_train)
    indices = list(range(dataset_size))
    
    # Shuffle indices to get a random split
    np.random.seed(42) # Fixed seed for reproducibility
    np.random.shuffle(indices)
    
    # Calculate split point
    split = int(np.floor(val_split * dataset_size))
    val_indices, train_indices = indices[:split], indices[split:]
    
    # Create Subsets using the correct transforms
    train_dataset = Subset(full_dataset_for_train, train_indices)
    val_dataset   = Subset(full_dataset_for_val, val_indices)
    test_dataset  = datasets.ImageFolder(test_dir, transform=val_test_transform)

    # ---------------------------------------------------------
    # 3. Create Loaders
    # ---------------------------------------------------------
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    print(f"\n[DATA SUMMARY]")
    print(f"   > Train Images: {len(train_dataset)} (Augmentation: {use_augmentation})")
    print(f"   > Val Images:   {len(val_dataset)}")
    print(f"   > Test Images:  {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader