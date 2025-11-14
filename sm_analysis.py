import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

from sm_multimodal_dataset import SMMDataset
from sm_multimodal_model import SMMultiModalNet

# Backbone comparison setup
backbones = ['resnet18', 'resnet50', 'densenet121', 'efficientnet_b0', 'vgg16']
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
IMAGE_ROOT = '/path/to/images'

# DataLoader setup
test_dataset = SMMDataset('test_metadata.csv', IMAGE_ROOT, TABULAR_FEATURES, IMAGE_COLUMNS, LABEL_COLUMN)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)

results = []

for backbone in backbones:
    # Load model weights (assumes each model's checkpoint is saved as e.g. 'best_model_{backbone}.pt')
    model = SMMultiModalNet(
        image_backbone=backbone,
        tabular_dim=TABULAR_DIM,
        tabular_hidden=[32, 16],
        n_images=N_IMAGES,
        n_classes=N_CLASSES
    )
    model.load_state_dict(torch.load(f'best_model_{backbone}.pt'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    test_labels = []
    test_preds  = []
    with torch.no_grad():
        for batch in test_loader:
            images = batch['images'].to(device)
            tabular = batch['tabular'].to(device)
            labels = batch['label'].to(device)
            outputs = model(images, tabular)
            preds = outputs.argmax(1)
            test_labels.extend(labels.cpu().numpy())
            test_preds.extend(preds.cpu().numpy())

    acc = accuracy_score(test_labels, test_preds)
    precision = precision_score(test_labels, test_preds, average='weighted', zero_division=0)
    recall    = recall_score(test_labels, test_preds, average='weighted', zero_division=0)
    cm        = confusion_matrix(test_labels, test_preds)

    results.append({
        'backbone': backbone,
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'confusion_matrix': cm
    })

    # Plot confusion matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Pre','Peak','Post'], yticklabels=['Pre','Peak','Post'])
    plt.title(f'Confusion Matrix: {backbone}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{backbone}.png')
    plt.close()

# Print and save results summary
import pandas as pd
summary = pd.DataFrame([{
    'Backbone': r['backbone'],
    'Accuracy': r['accuracy'],
    'Precision': r['precision'],
    'Recall': r['recall']
} for r in results])
summary.to_csv('backbone_comparison_results.csv', index=False)
print(summary)
