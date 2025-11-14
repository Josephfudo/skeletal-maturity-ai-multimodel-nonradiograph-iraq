import os
import sys
import importlib
import torch
import pandas as pd
from torch.utils.data import DataLoader
import traceback # To print detailed errors

# --- Helper function to check for packages ---
def check_package(package_name):
    try:
        importlib.import_module(package_name)
        print(f"[ OK ] Package '{package_name}' is installed.")
        return True
    except ImportError:
        print(f"[FAIL] Package '{package_name}' is NOT installed. Please run: pip install {package_name}")
        return False

# --- Helper function to check files ---
def check_file(filepath):
    if os.path.exists(filepath):
        print(f"[ OK ] File found: {filepath}")
        return True
    else:
        print(f"[FAIL] File NOT found: {filepath}")
        return False

# --- Helper function to check CSV columns ---
def check_columns(csv_path, required_cols):
    try:
        df = pd.read_csv(csv_path, nrows=0) # Only read header
        missing_cols = [col for col in required_cols if col not in df.columns]
        if not missing_cols:
            print(f"[ OK ] All {len(required_cols)} required columns are present in {csv_path}.")
            return True
        else:
            print(f"[FAIL] {csv_path} is missing columns: {missing_cols}")
            return False
    except Exception as e:
        print(f"[FAIL] Could not read {csv_path}. Error: {e}")
        return False

def run_checks():
    print("Starting comprehensive pre-flight check for SM_Project...")
    all_errors = []
    
    # --- 1. Define Master Configuration ---
    # These MUST match the settings in all your training scripts
    TABULAR_FEATURES = [
        'Age', 'Body_Weight_kg', 'Body_Height_cm', 'BMI',
        'weight_height_ratio', 'log_BMI', 'sqrt_age',
        'Age_pre', 'Body_Weight_kg_pre', 'BMI_pre'
    ]
    IMAGE_COLUMNS = ['u_photo_eng', 'l_photo_eng', 'hp_photo_eng', 'hd_photo_eng', 'hdf_photo_eng']
    LABEL_COLUMN = 'Growth_Phase_enc_eng'
    IMAGE_ROOT = "./" # Path you confirmed in previous steps
    BACKBONES = ["resnet18", "resnet50", "densenet121", "efficientnet_b0", "vgg16"]
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    # --- Check 1: Environment & Packages ---
    print("\n--- 1. Checking Environment & Packages ---")
    packages_to_check = ['torch', 'pandas', 'sklearn', 'optuna', 'xgboost', 'captum', 'shap', 'tqdm', 'PIL']
    for pkg in packages_to_check:
        if not check_package(pkg): all_errors.append(f"Missing package: {pkg}")
            
    if torch.cuda.is_available():
        print(f"[ OK ] GPU (CUDA) is detected. Device: {torch.cuda.get_device_name(0)}")
    else:
        print(f"[WARN] CUDA (GPU) not detected. Training will be VERY slow (on CPU).")

    # --- Check 2: Data & Split Files ---
    print("\n--- 2. Checking Data & Split Files ---")
    all_cols = TABULAR_FEATURES + IMAGE_COLUMNS + [LABEL_COLUMN]
    
    if not check_file('final_metadata.csv'): all_errors.append("Missing final_metadata.csv")
    elif not check_columns('final_metadata.csv', all_cols): all_errors.append("final_metadata.csv has missing columns.")
    
    fold_files = [f'train_fold{i}.csv' for i in range(1, 6)] + \
                 [f'val_fold{i}.csv' for i in range(1, 6)] + \
                 ['test_metadata.csv']
                 
    missing_files = False
    for f in fold_files:
        if not check_file(f): 
            all_errors.append(f"Missing split file: {f}")
            missing_files = True
    
    # Check column integrity only if files exist
    if not missing_files:
        if not check_columns('train_fold1.csv', all_cols): all_errors.append("train_fold1.csv has missing columns.")
        if not check_columns('val_fold1.csv', all_cols): all_errors.append("val_fold1.csv has missing columns.")

    # --- Check 3: Data Pipeline (Dataset & DataLoader) ---
    print("\n--- 3. Checking Data Pipeline (Dataset & DataLoader) ---")
    batch_loaded = False
    if missing_files:
        print("[WARN] Skipping Data Pipeline check because split files are missing.")
        all_errors.append("Skipped data pipeline check.")
    else:
        try:
            # This import will now work
            from sm_multimodal_dataset import SMMDataset
            print("[ OK ] Imported SMMDataset from sm_multimodal_dataset.py")
            
            print("Testing dataset __getitem__ (loads one sample)...")
            # This 'train=False' argument will now work
            ds = SMMDataset(
                'train_fold1.csv', IMAGE_ROOT, TABULAR_FEATURES, IMAGE_COLUMNS, 
                LABEL_COLUMN, train=False 
            )
            sample = ds[0] # Get first item
            
            if sample['images'].shape != torch.Size([5, 3, 224, 224]):
                all_errors.append(f"Image shape mismatch. Expected [5, 3, 224, 224], got {sample['images'].shape}")
            if sample['tabular'].shape != torch.Size([len(TABULAR_FEATURES)]):
                all_errors.append(f"Tabular shape mismatch. Expected [{len(TABULAR_FEATURES)}], got {sample['tabular'].shape}")
            
            print("[ OK ] __getitem__ test passed. Shapes are correct.")
            
            print("Testing DataLoader (loads one batch)...")
            dl = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0)
            batch = next(iter(dl))
            batch_loaded = True # Success!
            print(f"[ OK ] DataLoader batch loaded.")
            print(f"    Image batch shape: {batch['images'].shape}")
            print(f"    Tabular batch shape: {batch['tabular'].shape}")
            
        except FileNotFoundError as e:
            print(f"[FAIL] Data loading failed. Could not find file.")
            print(f"       Make sure IMAGE_ROOT is '{IMAGE_ROOT}'")
            print(f"       And that image paths in CSV are correct (e.g., '{e.filename}')")
            all_errors.append(f"Image FileNotFoundError: {e.filename}")
        except Exception as e:
            print(f"[FAIL] Data pipeline check failed!")
            print(traceback.format_exc())
            all_errors.append(f"Data pipeline error: {e}")


    # --- Check 4: Model Architectures & Forward Pass ---
    print("\n--- 4. Checking All Model Architectures ---")
    try:
        from sm_multimodal_model import SMMultiModalNet, ImageEncoder
        print("[ OK ] Imported SMMultiModalNet & ImageEncoder from sm_multimodal_model.py")
        
        if not batch_loaded:
             print("[WARN] Skipping model check because data batch failed to load.")
             all_errors.append("Skipped model check.")
        else:
            img_batch = batch['images'].to(DEVICE)
            tab_batch = batch['tabular'].to(DEVICE)
            
            for backbone in BACKBONES:
                print(f"Testing backbone: {backbone}...")
                try:
                    model = SMMultiModalNet(
                        tabular_dim=len(TABULAR_FEATURES),
                        n_classes=3,
                        backbone=backbone,
                        tabular_hidden=[32, 16],
                        dropout_rate=0.3
                    ).to(DEVICE)
                    model.eval()
                    
                    with torch.no_grad():
                        output = model(img_batch, tab_batch)
                    
                    if output.shape == torch.Size([4, 3]):
                         print(f"[ OK ] {backbone} initialized and forward pass successful.")
                    else:
                        print(f"[FAIL] {backbone} forward pass shape mismatch. Expected [4, 3], got {output.shape}")
                        all_errors.append(f"{backbone} forward pass failed.")
                
                except Exception as e:
                    print(f"[FAIL] {backbone} failed to initialize or run forward pass.")
                    print(traceback.format_exc())
                    all_errors.append(f"{backbone} failed: {e}")
    
    except ImportError as e:
        print(f"[FAIL] Could not import model: {e}")
        all_errors.append(f"Model import error: {e}")
    except Exception as e:
        print(f"[FAIL] Model check failed unexpectedly!")
        print(traceback.format_exc())
        all_errors.append(f"Model check error: {e}")


    # --- Check 5: Advanced Script Imports ---
    print("\n--- 5. Checking Advanced Script Imports ---")
    try:
        from pretrain_simclr import SimCLRModel
        print("[ OK ] Imported `pretrain_simclr.py`")
        from tune_hyperparameters import objective
        print("[ OK ] Imported `tune_hyperparameters.py`")
        from train_stacked_xgboost import create_feature_datasets
        print("[ OK ] Imported `train_stacked_xgboost.py`")
        
        # --- THIS IS THE FIX ---
        # We now import the main function, not the old class
        from ensemble_predict import run_ensemble_predictions
        print("[ OK ] Imported `ensemble_predict.py`")
        # --- END FIX ---
        
        from training_utils import EarlyStopping, get_scheduler
        print("[ OK ] Imported `training_utils.py`")
    except ImportError as e:
        print(f"[FAIL] An advanced script failed to import. This is often a syntax error.")
        print(f"       Error: {e}")
        all_errors.append(f"Advanced script import error: {e}")
    except Exception as e:
        print(f"[FAIL] Advanced script check failed!")
        print(traceback.format_exc())
        all_errors.append(f"Advanced script check error: {e}")


    # --- 6. Final Summary ---
    print("\n--- 6. Final Summary ---")
    if not all_errors:
        print("\n[  SUCCESS  ]")
        print("All checks passed! Your project is correctly configured.")
        print("You are ready to start your 6-step training campaign.")
    else:
        print(f"\n[  FAILURE  ]")
        print(f"Found {len(all_errors)} critical error(s). Please fix them before training.")
        for i, err in enumerate(all_errors):
            print(f"  {i+1}. {err}")

if __name__ == "__main__":
    run_checks()