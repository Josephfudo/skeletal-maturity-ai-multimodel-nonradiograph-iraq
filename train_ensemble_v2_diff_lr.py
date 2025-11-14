import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import pickle
import time
import os
import json
from tqdm import tqdm

# --- Import your project files ---
# Make sure these .py files are in the same directory
try:
    from sm_multimodal_dataset import SMMDataset
    from sm_multimodal_model import SMMultiModalNet
    from training_utils import EarlyStopping, get_scheduler, class_weights_from_labels
except ImportError as e:
    print(f"FATAL ERROR: Could not import project files (e.g., SMMDataset). {e}")
    print("Please ensure 'sm_multimodal_dataset.py' and 'sm_multimodal_model.py' are in the same folder.")
    exit()

# --- Load Tuned Hyperparameters ---
PARAM_FILE = "best_params.json"
if os.path.exists(PARAM_FILE):
    print(f"Loading tuned parameters from {PARAM_FILE}...")
    with open(PARAM_FILE, 'r') as f:
        TUNED_PARAMS = json.load(f)
    
    # Pop out the non-optimizer args
    TUNED_TABULAR_HIDDEN = TUNED_PARAMS.pop('tabular_hidden')
    TUNED_PARAMS.pop('n_layers', None)
    for k in list(TUNED_PARAMS.keys()):
        if 'n_units' in k:
            TUNED_PARAMS.pop(k)
else:
    print("FATAL ERROR: 'best_params.json' not found.")
    print("Please run 'tune_hyperparameters_v2.py' first to generate this file.")
    exit()

# --- CONFIGURATION ---
num_folds = 5
epochs = 100 
patience = 10 # Patience for EarlyStopping
batch_size = 8
BACKBONES = ["resnet18", "resnet50", "densenet121", "efficientnet_b0", "vgg16"]
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

print(f"--- Starting Main Ensemble Training (v2 w/ Differential LR) ---")
print(f"Using Device: {DEVICE}")
print(f"Using Tuned Params: {TUNED_PARAMS}")
print(f"Using Tuned Tabular MLP: {TUNED_TABULAR_HIDDEN}")

results = []

for backbone in BACKBONES:
    print(f"\n--- Training Backbone: {backbone} ---")
    fold_metrics = []

    # Check for SimCLR weights
    simclr_weight_path = f"simclr_{backbone}_backbone.pth"
    use_simclr = os.path.exists(simclr_weight_path)
    if use_simclr:
        print(f"Found SimCLR weights for {backbone}")
    else:
        print(f"SimCLR weights not found. Using default ImageNet pretrained=True.")


    for fold in range(1, num_folds+1):
        print(f"--- Fold {fold}/{num_folds} ---")
        start_time = time.time()

        train_csv_path = f'train_fold{fold}.csv'
        val_csv_path = f'val_fold{fold}.csv'
        if not os.path.exists(train_csv_path) or not os.path.exists(val_csv_path):
            print(f"ERROR: Cannot find {train_csv_path} or {val_csv_path}. Skipping fold.")
            continue
            
        full_train_df = pd.read_csv(train_csv_path)
        train_ds = SMMDataset(train_csv_path, IMAGE_ROOT, TABULAR_FEATURES, IMAGE_COLUMNS, LABEL_COLUMN, train=True)
        val_ds   = SMMDataset(val_csv_path, IMAGE_ROOT, TABULAR_FEATURES, IMAGE_COLUMNS, LABEL_COLUMN, train=False)
        
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS_FIX, pin_memory=True)
        val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS_FIX, pin_memory=True)

        model = SMMultiModalNet(
            tabular_dim=TABULAR_DIM,
            n_classes=N_CLASSES,
            backbone=backbone,
            pretrained=not use_simclr, # Use ImageNet if SimCLR weights not found
            tabular_hidden=TUNED_TABULAR_HIDDEN,
            dropout_rate=TUNED_PARAMS['dropout_rate']
        ).to(DEVICE)

        if use_simclr:
            model.load_backbone_weights(simclr_weight_path)

        # --- !!! KEY CHANGE: DIFFERENTIAL LEARNING RATE SETUP !!! ---
        # FIX 1: Corrected attribute name from 'image_encoder' to 'img_encoder'
        
        base_lr = TUNED_PARAMS['lr']
        
        # 1. Get parameters for the backbone (img_encoder)
        backbone_params = model.img_encoder.parameters()
        
        # 2. Get all other parameters (tabular_net, classifier)
        other_params = [
            p for name, p in model.named_parameters() 
            if "img_encoder" not in name and p.requires_grad # <-- Use 'img_encoder' here
        ]

        optimizer = getattr(optim, TUNED_PARAMS['optimizer'])(
            [
                {'params': backbone_params, 'lr': base_lr * 0.1}, # 10x smaller LR for backbone
                {'params': other_params, 'lr': base_lr}           # Full LR for other layers
            ],
            lr=base_lr, # Base LR (some optimizers might need this)
            weight_decay=TUNED_PARAMS['weight_decay']
        )
        # --- !!! END OF KEY CHANGE & FIX !!! ---
        
        scheduler = get_scheduler(optimizer, patience=patience//2) # Halve scheduler patience

        # Get class weights for imbalanced dataset
        class_weights = class_weights_from_labels(full_train_df[LABEL_COLUMN].values).to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

        # --- FIX 2: Removed the unsupported 'prefix' argument ---
        es = EarlyStopping(patience=patience, verbose=True)
        # --- END OF FIX 2 ---
        
        best_val_acc = 0

        for epoch in range(1, epochs+1):
            model.train()
            train_loss, train_correct, train_total = 0., 0, 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch:2d}/{epochs} [F{fold} {backbone}]", leave=False)
            for batch in pbar:
                x_tab, x_img, y = batch['tabular'].to(DEVICE), batch['images'].to(DEVICE), batch['label'].to(DEVICE)

                optimizer.zero_grad()
                
                # Arguments must match model definition: (images, tabular)
                logits = model(x_img, x_tab)
                
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * x_tab.size(0)
                train_correct += (logits.argmax(1) == y).sum().item()
                train_total += x_tab.size(0)
                pbar.set_postfix({"train_loss": loss.item()})

            train_loss /= train_total
            train_acc = train_correct / train_total

            # --- Validation ---
            model.eval()
            val_loss, val_correct, val_total = 0., 0, 0
            with torch.no_grad():
                for batch in val_loader:
                    x_tab, x_img, y = batch['tabular'].to(DEVICE), batch['images'].to(DEVICE), batch['label'].to(DEVICE)
                    
                    logits = model(x_img, x_tab)
                    
                    loss = criterion(logits, y)
                    val_loss += loss.item() * x_tab.size(0)
                    val_correct += (logits.argmax(1) == y).sum().item()
                    val_total += x_tab.size(0)
                    
            val_loss /= val_total
            val_acc = val_correct / val_total

            print(f"  Epoch {epoch:2d}: Train loss {train_loss:.4f}, acc {train_acc:.3f} | Val loss {val_loss:.4f}, acc {val_acc:.3f}")
            
            scheduler.step(val_loss)
            
            # Save to a new, non-conflicting model name
            es(val_loss, model, f"best_model_diff_lr_fold{fold}_{backbone}.pth")

            if es.early_stop:
                print("  Early stopping triggered.")
                break
            if val_acc > best_val_acc:
                best_val_acc = val_acc

        fold_time = time.time() - start_time
        print(f"Fold {fold} finished in {fold_time / 60:.2f} minutes. Best Val Acc: {best_val_acc:.4f}")
        fold_metrics.append({'fold': fold, 'best_val_acc': best_val_acc})

    accs = [m['best_val_acc'] for m in fold_metrics]
    if len(accs) > 0:
        print(f"--- Backbone: {backbone} K-Fold Validation Acc Mean: {np.mean(accs)*100:.2f}% | Std: {np.std(accs)*100:.2f}% ---")
        results.append({'backbone': backbone, 'fold_metrics': fold_metrics, 'mean_acc': np.mean(accs)})
    else:
        print(f"--- Backbone: {backbone} SKIPPED (no folds trained) ---")

# --- Save Final Results ---
results_filename = 'ensemble_training_results_v2_diff_lr.pkl'
with open(results_filename, 'wb') as f:
    pickle.dump(results, f)
print(f"\n--- Main Training (v2) complete. All fold models and histories saved to {results_filename} ---")