import os
import torch
from torch.utils.data import DataLoader
import pandas as pd
from collections import Counter

from sm_multimodal_dataset import SMMDataset
from sm_multimodal_model import SMMultiModalNet

def check_file_exists(path):
    if not os.path.exists(path):
        print(f"Missing file: {path}")
        return False
    print(f"Found: {path}")
    return True

def check_metadata(path, required_cols):
    df = pd.read_csv(path)
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"Metadata {path} missing columns: {missing}")
    else:
        print(f"Metadata {path}: All required columns present")
    print("First few rows:\n", df.head())
    return len(missing) == 0

def check_class_balance(path, label_col, key='Gender_eng'):
    df = pd.read_csv(path)
    print(f"Class balance in {path} (by {label_col} and {key}):")
    grouped = df.groupby([label_col, key]).size().reset_index(name='counts')
    print(grouped)

def check_dataloader(dataset):
    dl = DataLoader(dataset, batch_size=4)
    batch = next(iter(dl))
    print("DL batch shapes:")
    for k, v in batch.items():
        if hasattr(v, 'shape'):
            print(f"  {k}: {v.shape}")
        else:
            print(f"  {k}: {type(v)}")
    return True

def check_model_forward(model, batch):
    try:
        out = model(batch['images'], batch['tabular'])
        print("Forward pass successful, output shape:", out.shape)
        return True
    except Exception as e:
        print("Model forward failed:", str(e))
        return False

if __name__ == '__main__':
    # Settings - update as needed
    required_metadata = ['final_metadata.csv', 'train_metadata.csv', 'val_metadata.csv', 'test_metadata.csv']
    tabular_features = ['Age', 'Body_Weight_kg', 'Body_Height_cm', 'BMI',
                        'weight_height_ratio', 'log_BMI', 'sqrt_age',
                        'Age_pre', 'Body_Weight_kg_pre', 'BMI_pre']
    image_columns = ['u_photo_eng', 'l_photo_eng', 'hp_photo_eng', 'hd_photo_eng', 'hdf_photo_eng']
    label_column = 'Growth_Phase_enc_eng'
    tabular_dim = len(tabular_features)
    n_images = len(image_columns)
    n_classes = 3
    image_root = './'

    # 1. Check metadata files and essential columns
    for file in required_metadata:
        check_file_exists(file)
    check_metadata('final_metadata.csv', tabular_features + image_columns + [label_column])

    # 2. Print class balance in all splits
    for split in ['train_metadata.csv', 'val_metadata.csv', 'test_metadata.csv']:
        check_class_balance(split, label_col=label_column)

    # 3. DataLoader + sample batch check
    ds = SMMDataset('train_metadata.csv', image_root, tabular_features, image_columns, label_column)
    check_dataloader(ds)

    # 4. Model sanity forward
    model = SMMultiModalNet(
        image_backbone='resnet18',
        tabular_dim=tabular_dim,
        tabular_hidden=[32, 16],
        n_images=n_images,
        n_classes=n_classes
    )
    model.eval()
    dl = DataLoader(ds, batch_size=4)
    batch = next(iter(dl))
    check_model_forward(model, batch)

    print("Preflight checks completed.")
