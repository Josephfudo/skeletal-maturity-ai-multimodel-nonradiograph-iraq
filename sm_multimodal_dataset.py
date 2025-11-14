import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import numpy as np
import os

# --- Import your augmentation file ---
# This script assumes 'data_augmentation.py' is in the same folder
from data_augmentation import get_train_transforms, get_val_test_transforms

class SMMDataset(Dataset):
    """
    This is the definitive, corrected Dataset class.
    It uses the 'train' flag to select augmentations for the training set
    and simple resizing for the validation/test set.
    """
    def __init__(self, metadata_csv, image_root,
                 tabular_features=None,
                 image_columns=None,
                 label_column='Growth_Phase_enc_eng',
                 img_size=(224,224),
                 # This 'train' argument fixes the TypeError
                 train=True): 
        
        try:
            self.df = pd.read_csv(metadata_csv)
        except FileNotFoundError:
            print(f"CRITICAL ERROR: Metadata file not found at {metadata_csv}")
            # This check helps if 'kfold_split.py' or 'stratified_split.py' hasn't been run
            if 'fold' in metadata_csv or 'test_metadata' in metadata_csv:
                print("Please run 'kfold_split.py' to generate your training/validation/test files.")
            raise
            
        self.image_root = image_root
        self.tabular_features = tabular_features
        self.image_columns = image_columns
        self.label_column = label_column
        self.image_size = img_size

        # Use the 'train' flag to select the correct transforms
        if train:
            self.img_transform = get_train_transforms(img_size)
            print(f"Loaded {metadata_csv} in TRAIN mode (with augmentations).")
        else:
            self.img_transform = get_val_test_transforms(img_size)
            print(f"Loaded {metadata_csv} in EVAL mode (no augmentations).")

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # Get the row of metadata for this sample
        sample = self.df.iloc[idx]
        
        # 1. Tabular data
        # Use .values to get a numpy array and avoid the FutureWarning
        tabular_data = sample[self.tabular_features].values.astype(np.float32)
        tabular = torch.tensor(tabular_data, dtype=torch.float32)
        
        # 2. Image data
        images = []
        for img_col in self.image_columns:
            # Handle potential path separators (Windows \ vs. Linux /)
            path_fragment = str(sample[img_col]).replace("\\", os.sep).replace("/", os.sep)
            img_path = os.path.join(self.image_root, path_fragment)
            
            try:
                img = Image.open(img_path).convert('RGB')
            except Exception as e:
                # If an image is broken or missing, create a black placeholder
                # This prevents training from crashing.
                # print(f"Warning: Could not open {img_path}. Using a black image.")
                img = Image.new('RGB', self.image_size, (0, 0, 0))
            
            # Apply the selected transform (train augmentations or val resizing)
            img = self.img_transform(img)
            images.append(img)
            
        # Stack the 5 images into a single tensor
        images = torch.stack(images, dim=0)  # Shape: [5, 3, 224, 224]
        
        # 3. Label
        label = int(sample[self.label_column])

        # Return a dictionary (which your training scripts expect)
        return {'tabular': tabular, 'images': images, 'label': label}

# --- Example usage (for self-testing) ---
if __name__ == '__main__':
    print("--- Testing SMMDataset ---")
    
    # Check your master config here
    TABULAR_FEATURES = [
        'Age', 'Body_Weight_kg', 'Body_Height_cm', 'BMI',
        'weight_height_ratio', 'log_BMI', 'sqrt_age',
        'Age_pre', 'Body_Weight_kg_pre', 'BMI_pre'
    ]
    IMAGE_COLUMNS = ['u_photo_eng', 'l_photo_eng', 'hp_photo_eng', 'hd_photo_eng', 'hdf_photo_eng']
    LABEL_COLUMN = 'Growth_Phase_enc_eng'
    IMAGE_ROOT = "./" # Make sure this is your correct path
    
    # You must run kfold_split.py before this test will pass
    if not os.path.exists('train_fold1.csv'):
        print("Warning: 'train_fold1.csv' not found. Run 'kfold_split.py' first.")
    else:
        print("Loading TRAIN dataset...")
        train_ds = SMMDataset(
            metadata_csv='train_fold1.csv',
            image_root=IMAGE_ROOT,
            tabular_features=TABULAR_FEATURES,
            image_columns=IMAGE_COLUMNS,
            label_column=LABEL_COLUMN,
            train=True # Test train mode
        )
        
        print("\nGetting one sample from train_ds...")
        sample = train_ds[0]
        print('Tabular shape:', sample['tabular'].shape)
        print('Images shape:', sample['images'].shape)
        print('Label:', sample['label'])
        
        print("\nGetting one batch from DataLoader...")
        loader = DataLoader(train_ds, batch_size=4)
        batch = next(iter(loader))
        print('Tabular batch shape:', batch['tabular'].shape)
        print('Images batch shape:', batch['images'].shape)
        print("--- Test complete ---")