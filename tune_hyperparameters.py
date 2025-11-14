import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader
import optuna
import time
import json

# Assuming these files are in the same directory
from sm_multimodal_dataset import SMMDataset
from sm_multimodal_model import SMMultiModalNet
from training_utils import EarlyStopping

# --- CONFIGURATION ---
TRAIN_CSV = 'train_fold1.csv'
VAL_CSV = 'val_fold1.csv'
N_TRIALS = 30
EPOCHS = 50 
PATIENCE = 5
BACKBONE_TO_TUNE = 'resnet18' 
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

print("Loading tuning data...")
# Note: Make sure the .py files (SMMDataset, SMMultiModalNet, training_utils) are in the same folder
train_ds = SMMDataset(TRAIN_CSV, IMAGE_ROOT, TABULAR_FEATURES, IMAGE_COLUMNS, LABEL_COLUMN, train=True)
val_ds = SMMDataset(VAL_CSV, IMAGE_ROOT, TABULAR_FEATURES, IMAGE_COLUMNS, LABEL_COLUMN, train=False)
train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=NUM_WORKERS_FIX, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=NUM_WORKERS_FIX, pin_memory=True)


def objective(trial):
    # --- THIS IS THE RECOMMENDED CHANGE ---
    # The previous upper bound of 1e-3 (0.001) found a learning rate (0.00073) that was too high
    # and caused loss explosions in your main training (Fold 3).
    # We are lowering the upper bound to 5e-4 (0.0005) to search for a more stable rate.
    lr = trial.suggest_float("lr", 1e-5, 5e-4, log=True) # Changed from 1e-3 to 5e-4
    # --- END OF CHANGE ---

    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "AdamW"])
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)

    n_layers = trial.suggest_int("n_layers", 1, 3)
    tabular_hidden = []
    for i in range(n_layers):
        dim = trial.suggest_int(f"n_units_l{i}", 16, 128, log=True)
        tabular_hidden.append(dim)

    print(f"\n--- TRIAL {trial.number} ---")
    print(f"Params: LR={lr:.6f}, WD={weight_decay:.6f}, Opt={optimizer_name}, Dropout={dropout_rate:.2f}, TabularMLP={tabular_hidden}")

    model = SMMultiModalNet(
        tabular_dim=TABULAR_DIM,
        n_classes=N_CLASSES,
        backbone=BACKBONE_TO_TUNE,
        tabular_hidden=tabular_hidden,
        dropout_rate=dropout_rate
    ).to(DEVICE)

    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    es = EarlyStopping(patience=PATIENCE, verbose=False)
    best_val_acc = 0

    for epoch in range(EPOCHS):
        model.train()
        for batch in train_loader:
            x_tab, x_img, y = batch['tabular'].to(DEVICE), batch['images'].to(DEVICE), batch['label'].to(DEVICE)
            optimizer.zero_grad()

            # --- THIS IS THE FIX ---
            # Arguments must match the model definition: model(images, tabular)
            logits = model(x_img, x_tab) 
            # --- END OF FIX ---

            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        model.eval()
        val_correct, val_total = 0, 0
        val_loss_epoch = 0
        with torch.no_grad():
            for batch in val_loader:
                x_tab, x_img, y = batch['tabular'].to(DEVICE), batch['images'].to(DEVICE), batch['label'].to(DEVICE)

                # --- THIS IS THE FIX ---
                logits = model(x_img, x_tab)
                # --- END OF FIX ---

                loss = criterion(logits, y)
                val_loss_epoch += loss.item()
                val_correct += (logits.argmax(1) == y).sum().item()
                val_total += x_tab.size(0)

        val_acc = val_correct / val_total
        avg_val_loss = val_loss_epoch / len(val_loader)

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        trial.report(val_acc, epoch)
        if trial.should_prune():
            print(f"TRIAL {trial.number} PRUNED at epoch {epoch}")
            raise optuna.exceptions.TrialPruned()

        es(avg_val_loss, model, "temp_best_model.pth") # This saves a temporary model, which is fine
        if es.early_stop:
            print(f"TRIAL {trial.number} Early Stopped at epoch {epoch}")
            break

    print(f"TRIAL {trial.number} finished. Best Val Acc: {best_val_acc:.4f}")
    return best_val_acc

if __name__ == "__main__":
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner()
    )
    study.optimize(objective, n_trials=N_TRIALS)

    print("\n\n--- TUNING COMPLETE ---")
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value (Best Val Acc): {trial.value:.4f}")
    print("  Best Hyperparameters:")

    final_params = {}
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
        final_params[key] = value

    # Re-create the tabular_hidden list from the best trial's parameters
    final_params['tabular_hidden'] = [trial.params[f'n_units_l{i}'] for i in range(trial.params['n_layers'])]

    PARAM_FILE = "best_params.json"
    with open(PARAM_FILE, 'w') as f:
        json.dump(final_params, f, indent=4)

    print(f"\nBest parameters saved to {PARAM_FILE}")

print("Overwrote tune_hyperparameters.py with arg fix and num_workers=16")