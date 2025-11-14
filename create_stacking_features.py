import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
import pandas as pd
import numpy as np
import os
import glob
import re
import json
from tqdm import tqdm

# --- Import your project files ---
# Make sure these .py files are in the same directory
try:
    from sm_multimodal_dataset import SMMDataset 
    from sm_multimodal_model import SMMultiModalNet
except ImportError as e:
    print(f"FATAL ERROR: Could not import project files (e.g., SMMDataset). {e}")
    print("Please ensure 'sm_multimodal_dataset.py' and 'sm_multimodal_model.py' are in the same folder.")
    exit()

"""
This script creates the "Level-2" features for a stacking meta-model.

It does two things:
1.  Generates Out-of-Fold (OOF) predictions for the *validation* set.
    -   It loads the 5 validation folds (val_fold1.csv, val_fold2.csv, etc.).
    -   It loads the 25 models you trained (e.g., `best_model_fold1_resnet18.pth`).
    -   Crucially, it uses the `fold1` models to predict *only* on `val_fold1.csv`.
    -   It uses the `fold2` models to predict *only* on `val_fold2.csv`, etc.
    -   This process ensures the meta-model is trained on predictions from models
        that have *never* seen that data, preventing information leaks.
    -   Saves: `oof_train_features.npy`, `oof_train_labels.npy`

2.  Generates average predictions for the *test* set.
    -   It loads `test_metadata.csv`.
    -   It runs all 25 models on the test set and averages their predictions.
    -   Saves: `oof_test_features.npy`, `oof_test_labels.npy`
"""

# --- Load Model Architecture Config ---
PARAM_FILE = "best_params.json"
if os.path.exists(PARAM_FILE):
    print(f"Loading tuned parameters from {PARAM_FILE}...")
    with open(PARAM_FILE, 'r') as f:
        TUNED_PARAMS = json.load(f)
    TUNED_TABULAR_HIDDEN = TUNED_PARAMS.get('tabular_hidden', [32, 16])
    TUNED_DROPOUT = TUNED_PARAMS.get('dropout_rate', 0.3)
    print(f"Using Tabular MLP config: {TUNED_TABULAR_HIDDEN} (Dropout: {TUNED_DROPOUT})")
else:
    print(f"FATAL ERROR: '{PARAM_FILE}' not found. Cannot determine model architecture.")
    print("This file is required to prevent 'size mismatch' errors.")
    exit()

# --- CONFIGURATION ---
MODELS_DIR = '.' 
NUM_FOLDS = 5
VALID_BACKBONES = ["resnet18", "resnet50", "densenet121", "efficientnet_b0", "vgg16"]
N_CLASSES = 3
BATCH_SIZE = 16 # Can be larger for inference
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS_FIX = 16 

# --- Feature/Column Defs (MUST MATCH YOUR OTHER FILES) ---
IMAGE_ROOT = "./" 
TABULAR_FEATURES = [
    'Age', 'Body_Weight_kg', 'Body_Height_cm', 'BMI',
    'weight_height_ratio', 'log_BMI', 'sqrt_age',
    'Age_pre', 'Body_Weight_kg_pre', 'BMI_pre'
]
IMAGE_COLUMNS = ['u_photo_eng', 'l_photo_eng', 'hp_photo_eng', 'hd_photo_eng', 'hdf_photo_eng']
LABEL_COLUMN = 'Growth_Phase_enc_eng'
TABULAR_DIM = len(TABULAR_FEATURES)

def get_model(backbone_name, model_path):
    """Helper to build and load one model."""
    model = SMMultiModalNet(
        tabular_dim=TABULAR_DIM,
        n_classes=N_CLASSES,
        backbone=backbone_name,
        pretrained=False,
        tabular_hidden=TUNED_TABULAR_HIDDEN,
        dropout_rate=TUNED_DROPOUT
    ).to(DEVICE)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()
        return model
    except Exception as e:
        print(f"  > WARNING: Could not load model {model_path}. Skipping.")
        print(f"    Error: {e}")
        return None

def get_predictions(loader, models):
    """
    Gets average predictions for a dataloader from a list of models.
    Returns:
        avg_preds: (num_samples, num_classes)
        all_labels: (num_samples,)
    """
    all_model_preds = []
    all_labels = []
    
    # Get predictions for each model
    for model in models:
        model_preds = []
        labels_this_run = []
        with torch.no_grad():
            for batch in loader:
                x_tab = batch['tabular'].to(DEVICE)
                x_img = batch['images'].to(DEVICE)
                y = batch['label']

                logits = model(x_img, x_tab)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                model_preds.append(probs)
                
                # Only store labels on the first model's pass
                if len(all_labels) == 0:
                    labels_this_run.append(y.numpy())

        all_model_preds.append(np.concatenate(model_preds))
        if len(all_labels) == 0:
            all_labels = np.concatenate(labels_this_run)
            
    # Average predictions across all models
    # Shape: (num_models, num_samples, num_classes) -> (num_samples, num_classes)
    avg_preds = np.mean(all_model_preds, axis=0)
    return avg_preds, all_labels

def create_oof_features():
    """Generates Out-of-Fold (OOF) features for training the meta-model."""
    print("\n--- Starting OOF Feature Generation for VALIDATION set ---")
    
    all_oof_preds = []
    all_oof_labels = []
    
    for fold in range(1, NUM_FOLDS + 1):
        print(f"\nProcessing Fold {fold}/{NUM_FOLDS}")
        val_csv = f'val_fold{fold}.csv'
        if not os.path.exists(val_csv):
            print(f"  > WARNING: {val_csv} not found. Skipping fold {fold}.")
            continue
            
        # 1. Load this fold's validation data
        val_ds = SMMDataset(val_csv, IMAGE_ROOT, TABULAR_FEATURES, IMAGE_COLUMNS, LABEL_COLUMN, train=False)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS_FIX, pin_memory=True)
        
        # 2. Load all 5 models trained *without* this fold
        # (i.e., the models from `train_fold{fold}.csv`)
        models_for_this_fold = []
        for backbone in VALID_BACKBONES:
            model_path = os.path.join(MODELS_DIR, f'best_model_fold{fold}_{backbone}.pth')
            if os.path.exists(model_path):
                print(f"  > Loading model: {model_path}")
                model = get_model(backbone, model_path)
                if model:
                    models_for_this_fold.append(model)
            else:
                print(f"  > WARNING: Model not found, skipping: {model_path}")

        if not models_for_this_fold:
            print(f"  > ERROR: No models found for fold {fold}. Cannot generate OOF features.")
            continue
            
        # 3. Get predictions
        # `fold_preds` = (num_samples_in_fold, num_classes)
        # `fold_labels` = (num_samples_in_fold,)
        # These are the *averaged* predictions from the 5 backbones for this fold
        fold_preds, fold_labels = get_predictions(val_loader, models_for_this_fold)
        
        all_oof_preds.append(fold_preds)
        all_oof_labels.append(fold_labels)
        print(f"  > Generated OOF predictions for {len(fold_labels)} samples.")

    # Concatenate all fold data together
    oof_train_features = np.concatenate(all_oof_preds)
    oof_train_labels = np.concatenate(all_oof_labels)
    
    print(f"\nTotal OOF training features created. Shape: {oof_train_features.shape}")
    np.save('oof_train_features.npy', oof_train_features)
    np.save('oof_train_labels.npy', oof_train_labels)
    print("Saved 'oof_train_features.npy' and 'oof_train_labels.npy'")

def create_test_features():
    """Generates features for the TEST set for final prediction."""
    print("\n--- Starting Feature Generation for TEST set ---")
    
    test_csv = 'test_metadata.csv'
    if not os.path.exists(test_csv):
        print(f"  > ERROR: {test_csv} not found. Cannot generate test features.")
        return

    # 1. Load test data
    test_ds = SMMDataset(test_csv, IMAGE_ROOT, TABULAR_FEATURES, IMAGE_COLUMNS, LABEL_COLUMN, train=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS_FIX, pin_memory=True)
    
    # 2. Load ALL 25 trained models
    all_models = []
    for fold in range(1, NUM_FOLDS + 1):
        for backbone in VALID_BACKBONES:
            model_path = os.path.join(MODELS_DIR, f'best_model_fold{fold}_{backbone}.pth')
            if os.path.exists(model_path):
                print(f"  > Loading model: {model_path}")
                model = get_model(backbone, model_path)
                if model:
                    all_models.append(model)
            else:
                print(f"  > WARNING: Model not found, skipping: {model_path}")

    if not all_models:
        print("  > ERROR: No models found at all. Exiting.")
        return
        
    print(f"\nLoaded {len(all_models)} models for test prediction.")

    # 3. Get predictions
    # `test_preds` = (num_test_samples, num_classes)
    # `test_labels` = (num_test_samples,)
    # These are the *averaged* predictions from ALL 25 models
    test_preds, test_labels = get_predictions(test_loader, all_models)
    
    print(f"\nTotal TEST features created. Shape: {test_preds.shape}")
    np.save('oof_test_features.npy', test_preds)
    np.save('oof_test_labels.npy', test_labels)
    print("Saved 'oof_test_features.npy' and 'oof_test_labels.npy'")

if __name__ == "__main__":
    create_oof_features()
    create_test_features()
    print("\n--- Stacking feature creation complete. ---")
    print("You can now run 'train_stacking_meta_model.py'")