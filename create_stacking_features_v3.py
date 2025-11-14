import torch
import torch.nn as nn
from torch.utils.data import DataLoader
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
This is v3 of the feature creation script.

This version creates the richest dataset for the meta-model.
It combines:
1.  The 15 prediction features from the v2 script (5 models * 3 classes).
2.  The 10 original tabular features (Age, BMI, etc.).

This results in a final feature set of (samples, 25), giving the
meta-model (XGBoost) all the information possible.
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
BATCH_SIZE = 16 
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

def get_predictions_and_feats(loader, models):
    """
    Gets *concatenated* predictions AND original tabular features.
    Returns:
        concat_preds: (num_samples, num_models * num_classes) -> e.g. (samples, 15)
        tab_feats: (num_samples, num_tabular_features) -> e.g. (samples, 10)
        all_labels: (num_samples,)
    """
    all_model_preds = []
    all_tab_feats = []
    all_labels = []
    
    # Get predictions for each model
    for model_idx, model in enumerate(models):
        model_preds = []
        
        # Only get tabular data and labels on the first model's pass
        is_first_model = (model_idx == 0)
        if is_first_model:
            labels_this_run = []
            tab_feats_this_run = []

        with torch.no_grad():
            for batch in loader:
                x_tab = batch['tabular'].to(DEVICE)
                x_img = batch['images'].to(DEVICE)
                y = batch['label']

                logits = model(x_img, x_tab)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                model_preds.append(probs)
                
                if is_first_model:
                    labels_this_run.append(y.numpy())
                    # Store the raw tabular features as well
                    tab_feats_this_run.append(batch['tabular'].numpy())

        all_model_preds.append(np.concatenate(model_preds))
        
        if is_first_model:
            all_labels = np.concatenate(labels_this_run)
            all_tab_feats = np.concatenate(tab_feats_this_run)
            
    # Concatenate predictions: [5x (samples, 3)] -> (samples, 15)
    concat_preds = np.concatenate(all_model_preds, axis=1)
    
    return concat_preds, all_tab_feats, all_labels

def create_oof_features():
    """Generates Out-of-Fold (OOF) features for training the meta-model."""
    print("\n--- Starting OOF Feature Generation for VALIDATION set (v3 - 25 features) ---")
    
    all_oof_combined_feats = []
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
        
        # 2. Load all 5 models trained for this fold
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
            
        # 3. Get concatenated predictions (shape 15) and tabular features (shape 10)
        # `fold_preds` = (num_samples_in_fold, 15)
        # `fold_tab_feats` = (num_samples_in_fold, 10)
        # `fold_labels` = (num_samples_in_fold,)
        fold_preds, fold_tab_feats, fold_labels = get_predictions_and_feats(val_loader, models_for_this_fold)
        
        # --- !!! KEY CHANGE HERE !!! ---
        # Combine prediction features (15) and tabular features (10)
        combined_feats = np.concatenate([fold_preds, fold_tab_feats], axis=1)
        # --- !!! END OF CHANGE !!! ---
        
        all_oof_combined_feats.append(combined_feats)
        all_oof_labels.append(fold_labels)
        print(f"  > Generated OOF predictions for {len(fold_labels)} samples. Feature shape: {combined_feats.shape}")

    # Concatenate all fold data together
    oof_train_features = np.concatenate(all_oof_combined_feats)
    oof_train_labels = np.concatenate(all_oof_labels)
    
    print(f"\nTotal OOF training features created. Shape: {oof_train_features.shape}")
    np.save('oof_train_features_v3.npy', oof_train_features)
    np.save('oof_train_labels_v3.npy', oof_train_labels)
    print("Saved 'oof_train_features_v3.npy' and 'oof_train_labels_v3.npy'")

def create_test_features():
    """Generates features for the TEST set for final prediction."""
    print("\n--- Starting Feature Generation for TEST set (v3 - 25 features) ---")
    
    test_csv = 'test_metadata.csv'
    if not os.path.exists(test_csv):
        print(f"  > ERROR: {test_csv} not found. Cannot generate test features.")
        return

    # 1. Load test data
    test_ds = SMMDataset(test_csv, IMAGE_ROOT, TABULAR_FEATURES, IMAGE_COLUMNS, LABEL_COLUMN, train=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS_FIX, pin_memory=True)
    
    # 2. Load ALL 25 trained models
    models_by_fold = {f:[] for f in range(1, NUM_FOLDS + 1)}
    for fold in range(1, NUM_FOLDS + 1):
        for backbone in VALID_BACKBONES:
            model_path = os.path.join(MODELS_DIR, f'best_model_fold{fold}_{backbone}.pth')
            if os.path.exists(model_path):
                print(f"  > Loading model: {model_path}")
                model = get_model(backbone, model_path)
                if model:
                    models_by_fold[fold].append(model)
            else:
                print(f"  > WARNING: Model not found, skipping: {model_path}")

    print(f"\nLoaded {sum(len(m) for m in models_by_fold.values())} total models.")
    
    # 3. Get predictions
    fold_ensembled_preds = []
    all_test_labels = []
    all_test_tab_feats = []

    for fold, models in models_by_fold.items():
        if not models:
            print(f"Skipping fold {fold}, no models loaded.")
            continue
        
        print(f"Getting test predictions for fold {fold} ensemble...")
        # `fold_preds` = (num_test_samples, 15)
        # `fold_tab_feats` = (num_test_samples, 10)
        # `fold_labels` = (num_test_samples,)
        fold_preds, fold_tab_feats, fold_labels = get_predictions_and_feats(test_loader, models)
        
        fold_ensembled_preds.append(fold_preds)
        
        if len(all_test_labels) == 0:
            all_test_labels = fold_labels
            all_test_tab_feats = fold_tab_feats
    
    # 4. Average the 5 sets of (samples, 15) *prediction* features
    # Shape: (num_folds, num_test_samples, 15) -> (num_test_samples, 15)
    final_test_pred_features = np.mean(fold_ensembled_preds, axis=0)

    # --- !!! KEY CHANGE HERE !!! ---
    # Combine averaged prediction features (15) and tabular features (10)
    final_test_features = np.concatenate([final_test_pred_features, all_test_tab_feats], axis=1)
    # --- !!! END OF CHANGE !!! ---

    print(f"\nTotal TEST features created. Shape: {final_test_features.shape}")
    np.save('oof_test_features_v3.npy', final_test_features)
    np.save('oof_test_labels_v3.npy', all_test_labels)
    print("Saved 'oof_test_features_v3.npy' and 'oof_test_labels_v3.npy'")

if __name__ == "__main__":
    create_oof_features()
    create_test_features()
    print("\n--- Stacking feature creation (v3) complete. ---")
    print("You can now run 'train_stacking_meta_model_v3.py'")