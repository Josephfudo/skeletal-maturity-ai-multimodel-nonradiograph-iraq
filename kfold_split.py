import pandas as pd
from sklearn.model_selection import StratifiedKFold
import os

# --- Configuration ---
N_SPLITS = 5
METADATA_FILE = 'final_metadata.csv'

# --- These must match the exact column names in final_metadata.csv ---
STRAT_KEY_1 = 'Gender_eng'
STRAT_KEY_2 = 'Growth_Phase_eng'

# --- End Configuration ---

print(f"Loading master metadata from: {METADATA_FILE}")
if not os.path.exists(METADATA_FILE):
    print(f"Error: {METADATA_FILE} not found!")
    print("Please create this file first by running 'final_metadata_creator.py'.")
    exit()
    
metadata = pd.read_csv(METADATA_FILE)

# Check if stratification columns exist
if STRAT_KEY_1 not in metadata.columns or STRAT_KEY_2 not in metadata.columns:
    print(f"Error: Stratification columns '{STRAT_KEY_1}' or '{STRAT_KEY_2}' not found in metadata.")
    print(f"Available columns are: {metadata.columns.tolist()}")
    exit()

# Stratify based on both growth phase and gender
metadata['stratify_col'] = metadata[STRAT_KEY_2].astype(str) + "_" + metadata[STRAT_KEY_1].astype(str)

print(f"Creating {N_SPLITS} stratified folds...")
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
folds = []

for fold_idx, (train_index, val_index) in enumerate(skf.split(metadata, metadata['stratify_col'])):
    fold_num = fold_idx + 1
    print(f"\n--- Processing FOLD {fold_num}/{N_SPLITS} ---")
    
    train_subset = metadata.iloc[train_index].reset_index(drop=True)
    val_subset = metadata.iloc[val_index].reset_index(drop=True)
    
    train_file = f'train_fold{fold_num}.csv'
    val_file = f'val_fold{fold_num}.csv'
    
    # Save the splits to CSV
    train_subset.to_csv(train_file, index=False)
    val_subset.to_csv(val_file, index=False)
    
    print(f"  Saved: {train_file} (Size: {len(train_subset)})")
    print(f"  Saved: {val_file} (Size: {len(val_subset)})")
    
    # Optional: Print distribution for this fold
    print("  Validation set distribution:")
    print(val_subset['stratify_col'].value_counts(normalize=True).sort_index())

print("\nK-Fold split files created successfully.")