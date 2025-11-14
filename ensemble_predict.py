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
import json # <-- FIX 1: Import json

# --- Import your project files ---
from sm_multimodal_dataset import SMMDataset 
from sm_multimodal_model import SMMultiModalNet
from data_augmentation import get_train_transforms, get_val_test_transforms

# --- FIX 2: Load the 'best_params.json' file to get model architecture ---
PARAM_FILE = "best_params.json"
if os.path.exists(PARAM_FILE):
    print(f"Loading tuned parameters from {PARAM_FILE}...")
    with open(PARAM_FILE, 'r') as f:
        TUNED_PARAMS = json.load(f)
    # Get the architecture params found during tuning
    TUNED_TABULAR_HIDDEN = TUNED_PARAMS.get('tabular_hidden', [32, 16]) # Default as fallback
    TUNED_DROPOUT = TUNED_PARAMS.get('dropout_rate', 0.3) # Default as fallback
else:
    print("FATAL ERROR: 'best_params.json' not found. Cannot determine model architecture.")
    exit()
# --- END OF FIX 2 ---


# --- CONFIGURATION ---
TEST_CSV = 'test_metadata.csv'
MODELS_DIR = '.' # Directory where your .pth files are saved
N_TTA_ROUNDS = 10 # Number of TTA passes (1 = no TTA, >1 = TTA)
BATCH_SIZE = 16
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
NUM_WORKERS_FIX = 16 

def run_ensemble_predictions():
    print(f"--- Starting Ensemble Prediction on Test Set ---")
    print(f"Using Device: {DEVICE}")
    print(f"Test-Time Augmentation (TTA) Rounds: {N_TTA_ROUNDS}")
    print(f"Using Tabular MLP config: {TUNED_TABULAR_HIDDEN} (Dropout: {TUNED_DROPOUT})")


    # --- 1. Find all trained models ---
    model_glob_pattern = os.path.join(MODELS_DIR, 'best_model_fold*_*[0-9].pth')
    model_paths = glob.glob(model_glob_pattern)
    
    if not model_paths:
        print(f"Error: No model .pth files found in {MODELS_DIR} matching '{model_glob_pattern}'.")
        print("Please run `train_ensemble.py` first.")
        return
        
    print(f"Found {len(model_paths)} models to ensemble.")

    # --- 2. Load Test Data ---
    test_ds_no_aug = SMMDataset(
        TEST_CSV, IMAGE_ROOT, TABULAR_FEATURES, IMAGE_COLUMNS, 
        LABEL_COLUMN, train=False # train=False uses eval transforms
    )
    test_ds_tta = SMMDataset(
        TEST_CSV, IMAGE_ROOT, TABULAR_FEATURES, IMAGE_COLUMNS, 
        LABEL_COLUMN, train=True # train=True uses augmentation transforms
    )
    
    test_loader_no_aug = DataLoader(test_ds_no_aug, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS_FIX)
    test_loader_tta = DataLoader(test_ds_tta, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS_FIX)
    
    all_model_predictions = []
    ground_truth_labels = []

    # --- 3. Loop over each model ---
    for model_path in model_paths:
        print(f"\nLoading model: {model_path}")
        
        try:
            match = re.search(r'best_model_fold\d_(.*?).pth', os.path.basename(model_path))
            if not match:
                print(f"  > WARNING: Could not parse backbone from filename: {model_path}. Skipping.")
                continue
            
            backbone = match.group(1)
            if backbone not in VALID_BACKBONES:
                 print(f"  > WARNING: Parsed backbone '{backbone}' is not valid. Skipping.")
                 continue
                 
            print(f"  > Inferred backbone: {backbone}")

        except Exception as e:
            print(f"  > ERROR: Error parsing backbone for {model_path}. Skipping. Error: {e}")
            continue

        # --- FIX 3: Pass the loaded architecture params to the model constructor ---
        model = SMMultiModalNet(
            tabular_dim=TABULAR_DIM,
            n_classes=N_CLASSES,
            backbone=backbone,
            pretrained=False, # We are loading weights, no need to re-download
            tabular_hidden=TUNED_TABULAR_HIDDEN, # <-- ADDED THIS
            dropout_rate=TUNED_DROPOUT           # <-- ADDED THIS
        ).to(DEVICE)
        # --- END OF FIX 3 ---
        
        try:
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        except Exception as e:
            print(f"  > WARNING: Could not load model {model_path}. Skipping.")
            print(f"    Error: {e}")
            continue
            
        model.eval()

        # --- 4. Run TTA for this model ---
        tta_predictions_for_model = []
        for tta_round in range(N_TTA_ROUNDS):
            is_first_round = (tta_round == 0)
            pbar_desc = f"  > TTA Round {tta_round + 1}/{N_TTA_ROUNDS}"
            pbar_desc += " (eval)" if is_first_round else " (aug)"
            
            current_loader = test_loader_no_aug if is_first_round else test_loader_tta
            
            round_preds = []
            if is_first_round: 
                ground_truth_labels_for_model = [] # Only collect labels on first pass
            
            with torch.no_grad():
                for batch in tqdm(current_loader, desc=pbar_desc, leave=False):
                    x_tab = batch['tabular'].to(DEVICE)
                    x_img = batch['images'].to(DEVICE)
                    y = batch['label'].to(DEVICE)
                    
                    logits = model(x_img, x_tab)
                    probs = torch.softmax(logits, dim=1).cpu().numpy()
                    round_preds.append(probs)
                    
                    if is_first_round:
                        ground_truth_labels_for_model.append(y.cpu().numpy())
            
            tta_predictions_for_model.append(np.concatenate(round_preds))
        
        avg_tta_pred = np.mean(tta_predictions_for_model, axis=0)
        all_model_predictions.append(avg_tta_pred)
        
        # --- FIX 4: Correctly check if the list is empty before filling it ---
        if len(ground_truth_labels) == 0:
             ground_truth_labels = np.concatenate(ground_truth_labels_for_model)
        # --- END OF FIX 4 ---

    if not all_model_predictions:
        print("\n---!!! FAILED TO LOAD ANY MODELS !!!---")
        print("No models were successfully loaded. Exiting.")
        print("This is likely due to an architecture mismatch from a previous run.")
        print("Please delete 'best_params.json' and all '.pth' files, then re-run the full pipeline.")
        return

    # --- 5. Average all model predictions (Ensemble) ---
    print(f"\n--- Averaging {len(all_model_predictions)} model predictions (Ensembling) ---")
    final_avg_probs = np.mean(all_model_predictions, axis=0)
    final_preds = np.argmax(final_avg_probs, axis=1)

    # --- 6. Final Report ---
    print("\n--- FINAL ENSEMBLED RESULTS (with TTA) ---")
    accuracy = accuracy_score(ground_truth_labels, final_preds)
    report = classification_report(
        ground_truth_labels, 
        final_preds, 
        target_names=['Pre-peak', 'Peak', 'Post-peak'], 
        digits=4
    )
    cm = confusion_matrix(ground_truth_labels, final_preds)
    
    print(f"Final Test Accuracy: {accuracy * 100:.4f}%")
    print("\nClassification Report:")
    print(report)
    print("\nConfusion Matrix:")
    print(cm)
    
    with open("final_ensemble_report.txt", "w") as f:
        f.write(f"Final Ensembled Test Accuracy: {accuracy * 100:.4f}%\n")
        f.write(f"Models Ensembled: {len(all_model_predictions)}\n")
        f.write(f"TTA Rounds: {N_TTA_ROUNDS}\n")
        f.write("\nClassification Report:\n")
        f.write(report)
        f.write("\nConfusion Matrix:\n")
        f.write(np.array_str(cm))
    print("\nReport saved to 'final_ensemble_report.txt'")


if __name__ == "__main__":
    run_ensemble_predictions()