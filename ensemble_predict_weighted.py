import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os
from tqdm import tqdm
import glob
import re
import json

# --- SCRIPT CONFIGURATION ---
# These are the 5-fold average validation accuracies from your 'full_training_log.txt'
# (Step 4: Main Ensemble Training)
# resnet18: 73.60%
# resnet50: 72.00%
# densenet121: 71.00%
# efficientnet_b0: 72.20%
# vgg16: 68.80%
#
# We will convert these to weights for our ensemble.

# 1. Raw validation scores
raw_scores = {
    "resnet18": 0.7360,
    "resnet50": 0.7200,
    "densenet121": 0.7100,
    "efficientnet_b0": 0.7220,
    "vgg16": 0.6880
}

# 2. Normalize scores to create weights that sum to 1
total_score = sum(raw_scores.values())
BACKBONE_WEIGHTS = {name: score / total_score for name, score in raw_scores.items()}

print("--- Using Weighted Averaging ---")
for name, weight in BACKBONE_WEIGHTS.items():
    print(f"  > {name}: {weight*100:.2f}% weight")


# --- Load Model Architecture Config ---
PARAM_FILE = "best_params.json"
if os.path.exists(PARAM_FILE):
    print(f"\nLoading tuned parameters from {PARAM_FILE}...")
    with open(PARAM_FILE, 'r') as f:
        TUNED_PARAMS = json.load(f)
    # Load the specific architecture from the tuning step
    TUNED_TABULAR_HIDDEN = TUNED_PARAMS.get('tabular_hidden', [32, 16]) 
    TUNED_DROPOUT = TUNED_PARAMS.get('dropout_rate', 0.3)
    print(f"Using Tabular MLP config: {TUNED_TABULAR_HIDDEN} (Dropout: {TUNED_DROPOUT})")
else:
    print(f"FATAL ERROR: '{PARAM_FILE}' not found. Cannot determine model architecture.")
    print("This file is required to prevent 'size mismatch' errors.")
    exit()

# --- Load Project Files ---
# Make sure these .py files are in the same directory
try:
    from sm_multimodal_dataset import SMMDataset 
    from sm_multimodal_model import SMMultiModalNet
    from data_augmentation import get_train_transforms, get_val_test_transforms
except ImportError as e:
    print(f"FATAL ERROR: Could not import project files (e.g., SMMDataset). {e}")
    print("Please ensure 'sm_multimodal_dataset.py' and 'sm_multimodal_model.py' are in the same folder.")
    exit()

# --- CONFIGURATION ---
TEST_CSV = 'test_metadata.csv'
MODELS_DIR = '.' # Directory where your .pth files are
N_TTA_ROUNDS = 10 # Number of TTA passes (1 = no TTA, >1 = TTA)
BATCH_SIZE = 16 # Increase if you have more VRAM
VALID_BACKBONES = ["resnet18", "resnet50", "densenet121", "efficientnet_b0", "vgg16"]

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
N_IMAGES = len(IMAGE_COLUMNS)
N_CLASSES = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS_FIX = 16 # Use max workers for fast data loading

def run_ensemble_predictions():
    """
    Loads all 25 models, runs TTA, and combines their predictions using
    the pre-calculated backbone weights.
    """
    print(f"\n--- Starting Ensemble Prediction on Test Set ---")
    print(f"Using Device: {DEVICE}")
    print(f"Test-Time Augmentation (TTA) Rounds: {N_TTA_ROUNDS}")

    # 1. Find all trained models
    # This glob pattern correctly finds models like 'best_model_fold1_resnet18.pth'
    model_glob_pattern = os.path.join(MODELS_DIR, 'best_model_fold*_*[0-9].pth')
    model_paths = glob.glob(model_glob_pattern)
    
    if not model_paths:
        print(f"Error: No model .pth files found in {MODELS_DIR} matching '{model_glob_pattern}'.")
        print("Please run 'train_ensemble.py' first.")
        return
        
    print(f"Found {len(model_paths)} models to ensemble.")

    # 2. Load Test Data
    try:
        # We create ONE dataset for evaluation (no augmentations)
        test_ds_no_aug = SMMDataset(
            TEST_CSV, IMAGE_ROOT, TABULAR_FEATURES, IMAGE_COLUMNS, 
            LABEL_COLUMN, train=False # train=False uses eval transforms
        )
        # We create a *second* dataset for TTA (with augmentations)
        test_ds_tta = SMMDataset(
            TEST_CSV, IMAGE_ROOT, TABULAR_FEATURES, IMAGE_COLUMNS, 
            LABEL_COLUMN, train=True # train=True uses augmentation transforms
        )
    except FileNotFoundError:
        print(f"FATAL ERROR: Cannot find test data '{TEST_CSV}'.")
        return
    
    test_loader_no_aug = DataLoader(test_ds_no_aug, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS_FIX, pin_memory=True)
    test_loader_tta = DataLoader(test_ds_tta, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS_FIX, pin_memory=True)
    
    # Store predictions PER BACKBONE so we can weight them
    # e.g., backbone_predictions['resnet18'] = [pred_fold1, pred_fold2, ...]
    backbone_predictions = {name: [] for name in VALID_BACKBONES}
    ground_truth_labels = []

    # 3. Loop over each model
    for model_path in model_paths:
        print(f"\nLoading model: {model_path}")
        
        # Parse backbone from filename
        try:
            # Updated regex to be more robust
            match = re.search(r'fold\d_(.*?).pth', os.path.basename(model_path))
            backbone = match.group(1)
            if backbone not in VALID_BACKBONES:
                 print(f"  > WARNING: Parsed backbone '{backbone}' is not valid. Skipping.")
                 continue
            print(f"  > Inferred backbone: {backbone}")
        except Exception as e:
            print(f"  > ERROR: Error parsing backbone for {model_path}. Skipping. Error: {e}")
            continue

        # CRITICAL FIX: Initialize model with the *exact* architecture from tuning
        model = SMMultiModalNet(
            tabular_dim=TABULAR_DIM,
            n_classes=N_CLASSES,
            backbone=backbone,
            pretrained=False, # We are loading weights, no need to re-download
            tabular_hidden=TUNED_TABULAR_HIDDEN, # <-- This is the fix
            dropout_rate=TUNED_DROPOUT         # <-- This is the fix
        ).to(DEVICE)
        
        try:
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        except Exception as e:
            print(f"  > WARNING: Could not load model {model_path}. Skipping.")
            print(f"    Error: {e}")
            continue
            
        model.eval()

        # 4. Run TTA for this model
        tta_predictions_for_model = []
        for tta_round in range(N_TTA_ROUNDS):
            is_first_round = (tta_round == 0)
            pbar_desc = f"  > TTA Round {tta_round + 1}/{N_TTA_ROUNDS}"
            pbar_desc += " (eval)" if is_first_round else " (aug)"
            
            # Use the correct loader (no aug for round 1, aug for others)
            current_loader = test_loader_no_aug if is_first_round else test_loader_tta
            
            round_preds = []
            # Only collect labels on the *very first pass* of the *very first model*
            if is_first_round and len(ground_truth_labels) == 0: 
                ground_truth_labels_for_model = [] 
            
            with torch.no_grad():
                for batch in tqdm(current_loader, desc=pbar_desc, leave=False):
                    x_tab = batch['tabular'].to(DEVICE)
                    x_img = batch['images'].to(DEVICE)
                    y = batch['label'].to(DEVICE)
                    
                    logits = model(x_img, x_tab)
                    # Convert to probabilities using softmax
                    probs = torch.softmax(logits, dim=1).cpu().numpy()
                    round_preds.append(probs)
                    
                    if is_first_round and len(ground_truth_labels) == 0:
                        ground_truth_labels_for_model.append(y.cpu().numpy())
            
            tta_predictions_for_model.append(np.concatenate(round_preds))
        
        # Average the TTA predictions for this *single model*
        avg_tta_pred = np.mean(tta_predictions_for_model, axis=0)
        
        # Store this model's prediction in the list for its backbone
        backbone_predictions[backbone].append(avg_tta_pred)
        
        # CRITICAL FIX: Check if list is empty, not the array itself
        if len(ground_truth_labels) == 0:
             ground_truth_labels = np.concatenate(ground_truth_labels_for_model)

    # --- 5. Average all model predictions (Weighted Ensemble) ---
    print("\n--- Averaging all model predictions (Weighted Ensembling) ---")
    
    final_weighted_predictions = []
    num_models_loaded = 0
    
    # Average the predictions for each backbone first (e.g., avg of 5 resnet18s)
    for backbone, preds_list in backbone_predictions.items():
        if len(preds_list) == 0:
            print(f"  > No models loaded for backbone: {backbone}. Skipping.")
            continue
            
        # Average the N folds for this backbone
        # Shape: (num_samples, num_classes)
        avg_pred_for_backbone = np.mean(preds_list, axis=0)
        
        # Apply the weight for this backbone
        weighted_pred = avg_pred_for_backbone * BACKBONE_WEIGHTS[backbone]
        final_weighted_predictions.append(weighted_pred)
        num_models_loaded += len(preds_list)
        print(f"  > Applied weight {BACKBONE_WEIGHTS[backbone]:.3f} to {backbone} (avg of {len(preds_list)} models)")

    if not final_weighted_predictions:
        print("No models were successfully loaded. Exiting.")
        return

    # Sum the weighted predictions
    # Shape: (num_samples, num_classes)
    final_avg_probs = np.sum(final_weighted_predictions, axis=0)
    
    # Get the final class prediction
    final_preds = np.argmax(final_avg_probs, axis=1)

    # --- 6. Final Report ---
    print("\n--- FINAL WEIGHTED ENSEMBLE RESULTS (with TTA) ---")
    accuracy = accuracy_score(ground_truth_labels, final_preds)
    report = classification_report(
        ground_truth_labels, 
        final_preds, 
        target_names=['Pre-peak', 'Peak', 'Post-peak'], # Adjust if your labels are 0,1,2
        digits=4
    )
    cm = confusion_matrix(ground_truth_labels, final_preds)
    
    print(f"Final Test Accuracy: {accuracy * 100:.4f}%")
    print("\nClassification Report:")
    print(report)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Save results to a file
    report_filename = "final_weighted_ensemble_report.txt"
    with open(report_filename, "w") as f:
        f.write(f"Final Weighted Ensembled Test Accuracy: {accuracy * 100:.4f}%\n")
        f.write(f"Models Ensembled: {num_models_loaded}\n")
        f.write(f"TTA Rounds: {N_TTA_ROUNDS}\n")
        f.write("\nWeights Used:\n")
        for name, weight in BACKBONE_WEIGHTS.items():
            f.write(f"  > {name}: {weight*100:.2f}% (Raw Score: {raw_scores[name]})\n")
        f.write("\nClassification Report:\n")
        f.write(report)
        f.write("\nConfusion Matrix:\n")
        f.write(np.array_str(cm))
    print(f"\nReport saved to '{report_filename}'")


if __name__ == "__main__":
    run_ensemble_predictions()