
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

from sm_multimodal_dataset import SMMDataset
from sm_multimodal_model import SMMultiModalNet
from training_utils import EarlyStopping, get_scheduler, class_weights_from_labels

PARAM_FILE = "best_params.json"
if os.path.exists(PARAM_FILE):
    print(f"Loading tuned parameters from {PARAM_FILE}...")
    with open(PARAM_FILE, 'r') as f:
        TUNED_PARAMS = json.load(f)
    TUNED_TABULAR_HIDDEN = TUNED_PARAMS.pop('tabular_hidden')
    TUNED_PARAMS.pop('n_layers', None)
    for k in list(TUNED_PARAMS.keys()):
        if 'n_units' in k:
            TUNED_PARAMS.pop(k)
else:
    print("WARNING: 'best_params.json' not found. Using default parameters.")
    TUNED_PARAMS = {
        'lr': 1e-4, 'weight_decay': 1e-5,
        'optimizer': 'AdamW', 'dropout_rate': 0.3,
    }
    TUNED_TABULAR_HIDDEN = [64, 32]

# --- CONFIGURATION ---
num_folds = 5
epochs = 100 
patience = 10 
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

print(f"--- Starting Main Ensemble Training ---")
print(f"Using Device: {DEVICE}")
print(f"Using Tuned Params: {TUNED_PARAMS}")
print(f"Using Tuned Tabular MLP: {TUNED_TABULAR_HIDDEN}")

results = []

for backbone in BACKBONES:
    print(f"\n--- Training Backbone: {backbone} ---")
    fold_metrics = []

    simclr_weight_path = f"simclr_{backbone}_backbone.pth"
    use_simclr = os.path.exists(simclr_weight_path)
    if use_simclr:
        print(f"Found SimCLR weights for {backbone}")
    else:
        print(f"SimCLR weights not found. Using default ImageNet pretrained=True.")


    for fold in range(1, num_folds+1):
        print(f"--- Fold {fold}/{num_folds} ---")
        start_time = time.time()

        full_train_df = pd.read_csv(f'train_fold{fold}.csv')
        train_ds = SMMDataset(f'train_fold{fold}.csv', IMAGE_ROOT, TABULAR_FEATURES, IMAGE_COLUMNS, LABEL_COLUMN, train=True)
        val_ds   = SMMDataset(f'val_fold{fold}.csv', IMAGE_ROOT, TABULAR_FEATURES, IMAGE_COLUMNS, LABEL_COLUMN, train=False)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS_FIX, pin_memory=True)
        val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS_FIX, pin_memory=True)

        model = SMMultiModalNet(
            tabular_dim=TABULAR_DIM,
            n_classes=N_CLASSES,
            backbone=backbone,
            pretrained=not use_simclr, 
            tabular_hidden=TUNED_TABULAR_HIDDEN,
            dropout_rate=TUNED_PARAMS['dropout_rate']
        ).to(DEVICE)

        if use_simclr:
            model.load_backbone_weights(simclr_weight_path)

        backbone_lr = TUNED_PARAMS['lr'] * 0.1
        
        param_groups = [
            {
                'params': model.image_encoder.parameters(), 
                'lr': backbone_lr
            },
            {
                'params': model.tabular_net.parameters(), 
                'lr': TUNED_PARAMS['lr']
            },
            {
                'params': model.classifier.parameters(), 
                'lr': TUNED_PARAMS['lr']
            }
        ]
        
        print(f"  > Using Differential LRs: Backbone LR = {backbone_lr:.2e}, Head LR = {TUNED_PARAMS['lr']:.2e}")
        
        optimizer = getattr(optim, TUNED_PARAMS['optimizer'])(
            param_groups, 
            lr=TUNED_PARAMS['lr'], # Default LR (for any non-grouped params, though all are grouped)
            weight_decay=TUNED_PARAMS['weight_decay']
        )
        scheduler = get_scheduler(optimizer, patience=patience//2)

        class_weights = class_weights_from_labels(full_train_df[LABEL_COLUMN].values).to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

        es = EarlyStopping(patience=patience, verbose=True)
        best_val_acc = 0

        for epoch in range(1, epochs+1):
            model.train()
            train_loss, train_correct, train_total = 0., 0, 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch:2d}/{epochs} [F{fold} {backbone}]", leave=False)
            for batch in pbar:
                x_tab, x_img, y = batch['tabular'].to(DEVICE), batch['images'].to(DEVICE), batch['label'].to(DEVICE)

                optimizer.zero_grad()

                # --- THIS IS THE FIX ---
                logits = model(x_img, x_tab)
                # --- END OF FIX ---

                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * x_tab.size(0)
                train_correct += (logits.argmax(1) == y).sum().item()
                train_total += x_tab.size(0)
                pbar.set_postfix({"train_loss": loss.item()})

            train_loss /= train_total
            train_acc = train_correct / train_total

            model.eval()
            val_loss, val_correct, val_total = 0., 0, 0
            with torch.no_grad():
                for batch in val_loader:
                    x_tab, x_img, y = batch['tabular'].to(DEVICE), batch['images'].to(DEVICE), batch['label'].to(DEVICE)

                    # --- THIS IS THE FIX ---
                    logits = model(x_img, x_tab)
                    # --- END OF FIX ---

                    loss = criterion(logits, y)
                    val_loss += loss.item() * x_tab.size(0)
                    val_correct += (logits.argmax(1) == y).sum().item()
                    val_total += x_tab.size(0)
            val_loss /= val_total
            val_acc = val_correct / val_total

            print(f"  Epoch {epoch:2d}: Train loss {train_loss:.4f}, acc {train_acc:.3f} | Val loss {val_loss:.4f}, acc {val_acc:.3f}")
            scheduler.step(val_loss)
            es(val_loss, model, f"best_model_fold{fold}_{backbone}.pth")

            if es.early_stop:
                print("  Early stopping triggered.")
                break
            if val_acc > best_val_acc:
                best_val_acc = val_acc

        fold_time = time.time() - start_time
        print(f"Fold {fold} finished in {fold_time / 60:.2f} minutes. Best Val Acc: {best_val_acc:.4f}")
        fold_metrics.append({'fold': fold, 'best_val_acc': best_val_acc})

    accs = [m['best_val_acc'] for m in fold_metrics]
    print(f"--- Backbone: {backbone} K-Fold Validation Acc Mean: {np.mean(accs)*100:.2f}% | Std: {np.std(accs)*100:.2f}% ---")
    results.append({'backbone': backbone, 'fold_metrics': fold_metrics})

with open('ensemble_training_results.pkl', 'wb') as f:
    pickle.dump(results, f)
print("\n--- Main Training complete. All fold models and histories saved. ---")

print("Overwrote train_ensemble.py with arg fix and num_workers=16")
