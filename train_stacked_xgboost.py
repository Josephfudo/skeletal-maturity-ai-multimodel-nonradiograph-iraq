
import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm
import os
import pickle
import json

from sm_multimodal_dataset import SMMDataset
from sm_multimodal_model import ImageEncoder

PARAM_FILE = "best_params.json"
if os.path.exists(PARAM_FILE):
    print(f"Loading tuned parameters from {PARAM_FILE}...")
    with open(PARAM_FILE, 'r') as f:
        TUNED_PARAMS = json.load(f)
else:
    print("WARNING: 'best_params.json' not found. Using default parameters.")
    TUNED_PARAMS = {'lr': 0.02, 'weight_decay': 1e-5} 

# --- CONFIGURATION ---
BACKBONES = ["resnet18", "resnet50", "densenet121", "efficientnet_b0", "vgg16"]
BATCH_SIZE = 16
IMAGE_ROOT = "./" 
TABULAR_FEATURES = [
    'Age', 'Body_Weight_kg', 'Body_Height_cm', 'BMI',
    'weight_height_ratio', 'log_BMI', 'sqrt_age',
    'Age_pre', 'Body_Weight_kg_pre', 'BMI_pre'
]
IMAGE_COLUMNS = ['u_photo_eng', 'l_photo_eng', 'hp_photo_eng', 'hd_photo_eng', 'hdf_photo_eng']
LABEL_COLUMN = 'Growth_Phase_enc_eng'
N_CLASSES = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS_FIX = 16 

def extract_image_features(loader, model):
    model.eval()
    all_features = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting image features"):
            images = batch['images'].to(DEVICE)
            labels = batch['label']

            B, N, C, H, W = images.shape
            images = images.view(B * N, C, H, W)
            feats = model(images)
            feats = feats.view(B, N, -1).mean(dim=1)

            all_features.append(feats.cpu().numpy())
            all_labels.append(labels.numpy())

    return np.concatenate(all_features), np.concatenate(all_labels)

def create_feature_datasets(backbone_name):
    print(f"\n--- Creating Features for: {backbone_name} ---")

    simclr_weight_path = f"simclr_{backbone}_backbone.pth"
    use_simclr = os.path.exists(simclr_weight_path)

    image_model = ImageEncoder(backbone_name=backbone_name, pretrained=not use_simclr).to(DEVICE)

    if use_simclr:
        print(f"Loading SimCLR weights from {simclr_weight_path}...")
        state_dict = torch.load(simclr_weight_path, map_location=DEVICE)

        if 'vgg16' in backbone_name:
             model_dict = image_model.state_dict()
             pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
             model_dict.update(pretrained_dict)
             image_model.load_state_dict(model_dict)
        else:
             image_model.cnn.load_state_dict(state_dict, strict=False)
    else:
        print(f"SimCLR weights not found. Using default ImageNet pretrained=True.")

    feature_cache = f'feature_cache_stacked_{backbone_name}_simclr_{use_simclr}.pkl'

    if os.path.exists(feature_cache):
        print("Loading features from cache...")
        with open(feature_cache, 'rb') as f:
            return pickle.load(f)

    print("No cache found. Creating new feature sets...")
    train_ds = SMMDataset('train_metadata.csv', IMAGE_ROOT, TABULAR_FEATURES, IMAGE_COLUMNS, LABEL_COLUMN, train=False)
    val_ds = SMMDataset('val_metadata.csv', IMAGE_ROOT, TABULAR_FEATURES, IMAGE_COLUMNS, LABEL_COLUMN, train=False)
    test_ds = SMMDataset('test_metadata.csv', IMAGE_ROOT, TABULAR_FEATURES, IMAGE_COLUMNS, LABEL_COLUMN, train=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS_FIX, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS_FIX, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS_FIX, pin_memory=True)

    train_img_feats, y_train = extract_image_features(train_loader, image_model)
    val_img_feats, y_val = extract_image_features(val_loader, image_model)
    test_img_feats, y_test = extract_image_features(test_loader, image_model)

    train_tab_feats = pd.read_csv('train_metadata.csv')[TABULAR_FEATURES].values
    val_tab_feats = pd.read_csv('val_metadata.csv')[TABULAR_FEATURES].values
    test_tab_feats = pd.read_csv('test_metadata.csv')[TABULAR_FEATURES].values

    X_train = np.concatenate([train_img_feats, train_tab_feats], axis=1)
    X_val = np.concatenate([val_img_feats, val_tab_feats], axis=1)
    X_test = np.concatenate([test_img_feats, test_tab_feats], axis=1)

    data = (X_train, y_train, X_val, y_val, X_test, y_test)
    with open(feature_cache, 'wb') as f:
        pickle.dump(data, f)

    return data

if __name__ == "__main__":

    all_results = []

    xgb_lr = TUNED_PARAMS.get('lr', 0.02) 

    for backbone in BACKBONES:
        X_train, y_train, X_val, y_val, X_test, y_test = create_feature_datasets(backbone)

        print(f"\n--- Training XGBoost Classifier for {backbone} ---")

        # --- THIS IS THE FIX ---
        # Changed 'gpu_hist' to 'hist' and added 'device=cuda'
        xgb_model = xgb.XGBClassifier(
            objective='multi:softmax',
            num_class=N_CLASSES,
            n_estimators=1000,
            learning_rate=xgb_lr,
            max_depth=6,
            subsample=0.7,
            colsample_bytree=0.7,
            use_label_encoder=False,
            eval_metric='mlogloss',
            early_stopping_rounds=50,
            tree_method='hist', # Use 'hist'
            device='cuda'       # Explicitly tell it to use the GPU
        )
        # --- END OF FIX ---

        xgb_model.fit(
            X_train, 
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        print(f"\n--- FINAL XGBOOST RESULTS ({backbone}) ---")
        y_pred = xgb_model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=['Pre-peak', 'Peak', 'Post-peak'])

        print(f"Stacked (CNN-Features + XGBoost) Accuracy: {accuracy * 100:.2f}%")
        print("Classification Report:")
        print(report)
        all_results.append({'backbone': backbone, 'accuracy': accuracy})

    print("\n\n--- ALL STACKED MODELS COMPLETE ---")
    for res in all_results:
        print(f"Backbone: {res['backbone']}, Test Accuracy: {res['accuracy']*100:.2f}%")

print("Overwrote train_stacked_xgboost.py with XGBoost fix.")
