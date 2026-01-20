import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# --- CONFIGURATION & CONSTANTS ---
# ImageNet Statistics:
# We use these specific values for normalization because we plan to use 
# Transfer Learning (ResNet/ViT). These models were pre-trained on ImageNet 
# and expect inputs with this specific statistical distribution.
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

class ChestXRayDataset(Dataset):
    """
    Custom Dataset class to handle the Chest X-Ray directory structure.
    It loads images dynamically to save memory (On-the-fly loading).
    """
    def __init__(self, root_dir, split='train', transform=None):
        """
        Args:
            root_dir (str): Path to the main data folder (e.g., 'data').
            split (str): The dataset split to load ('train', 'val', or 'test').
            transform (callable, optional): The augmentation pipeline to apply.
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        # Prepare lists to hold file paths and labels
        self.images = []
        self.labels = []
        
        # Define the path to the specific split (e.g., "data/train")
        self.dataset_path = os.path.join(root_dir, split)
        
        # Mapping directory names to integer labels
        # 0 = Normal, 1 = Pneumonia
        self.class_map = {'NORMAL': 0, 'PNEUMONIA': 1}
        
        # Populate the lists
        self._load_dataset()

        # Print summary to console for verification
        print(f"[{self.split.upper()}] Loaded {len(self.images)} images.")

    def _load_dataset(self):
        """
        Internal helper: Iterates through folders and collects valid image paths.
        """
        for class_name, class_idx in self.class_map.items():
            class_dir = os.path.join(self.dataset_path, class_name)
            
            # Skip if folder doesn't exist (safety check)
            if not os.path.exists(class_dir):
                continue
            
            # Iterate over files
            for img_name in os.listdir(class_dir):
                # Filter out system files like .DS_Store or hidden files
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.images.append(os.path.join(class_dir, img_name))
                    self.labels.append(class_idx)

    def __len__(self):
        """Returns the total number of samples in this dataset split."""
        return len(self.images)

    def __getitem__(self, idx):
        """
        Fetches a single sample. This is called by the DataLoader during training.
        """
        # 1. Load Image Path and Label
        img_path = self.images[idx]
        label = self.labels[idx]

        try:
            # 2. Open Image and Convert to RGB
            # Critical: Even if X-rays are grayscale, models like ResNet/ViT expect 
            # 3 input channels (Red, Green, Blue).
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            return None

        # 3. Apply Transformations (Augmentation or Normalization)
        if self.transform:
            image = self.transform(image)

        return image, label

def get_transforms(img_size=224, augment=True):
    """
    Constructs the transformation pipeline.
    
    Args:
        img_size (int): Target size (224 is standard for ViT/ResNet).
        augment (bool): If True, applies random geometric changes (Data Augmentation).
                        If False, applies only resizing and normalization.
    """
    if augment:
        # --- TRAINING PIPELINE (With Augmentation) ---
        # "On-the-fly" augmentation: Every time an image is loaded, it is randomly
        # modified. This helps the model generalize and prevents overfitting.
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),        # Random mirror flip
            transforms.RandomRotation(10),                 # Slight rotation (+/- 10 degrees)
            transforms.RandomAffine(0, scale=(0.9, 1.1)),  # Zoom in/out (90%-110%)
            transforms.ToTensor(),                         # Convert [0, 255] -> [0, 1] Tensor
            transforms.Normalize(MEAN, STD)                # Standardization
        ])
    else:
        # --- VALIDATION/TEST PIPELINE (Clean) ---
        # No random changes. We want to evaluate the model on the real images.
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])

def get_dataloaders(root_dir, batch_size=32, img_size=224):
    """
    Main entry point: Creates DataLoaders for all three splits.
    """
    # 1. Define Transforms
    # Train set gets augmentation, Val/Test sets do not.
    train_tf = get_transforms(img_size, augment=True)
    val_tf = get_transforms(img_size, augment=False)

    # 2. Create Datasets
    train_ds = ChestXRayDataset(root_dir, 'train', train_tf)
    val_ds = ChestXRayDataset(root_dir, 'val', val_tf)
    test_ds = ChestXRayDataset(root_dir, 'test', val_tf)

    # 3. Create DataLoaders
    # num_workers=2 enables parallel loading to speed up training.
    # shuffle=True is crucial for training to break correlation between batches.
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader

# --- SANITY CHECK ---
# This block runs only when executing 'python dataset.py' directly.
if __name__ == "__main__":
    print("--- Verifying Dataset Implementation ---")
    try:
        # Attempt to load a small batch
        train, val, test = get_dataloaders('/home/projects/cgm-prj10354/ViT-vs-CNN-medical-imaging/data', batch_size=4)
        
        # Fetch one batch to ensure shapes are correct
        images, labels = next(iter(train))
        
        print("\nSUCCESS: DataLoaders created successfully.")
        print(f"Batch Shape: {images.shape} (Batch, Channels, Height, Width)")
        print(f"Labels: {labels}")
        print("Ready for training.")
    except Exception as e:
        print(f"\nERROR: {e}")
        print("Please check your 'data' folder structure.")


    