import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import numpy as np

from sm_multimodal_dataset import SMMDataset
from sm_multimodal_model import SMMultiModalNet

# --- Settings ---
BATCH_SIZE = 16
NUM_WORKERS = 4
EPOCHS = 40
LEARNING_RATE = 1e-4
BACKBONE = 'resnet18'   # Any of: 'resnet18', 'resnet50', 'densenet121', 'efficientnet_b0', 'vgg16'
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

# --- Load datasets ---
train_dataset = SMMDataset('train_metadata.csv', IMAGE_ROOT, TABULAR_FEATURES, IMAGE_COLUMNS, LABEL_COLUMN)
val_dataset   = SMMDataset('val_metadata.csv', IMAGE_ROOT, TABULAR_FEATURES, IMAGE_COLUMNS, LABEL_COLUMN)
test_dataset  = SMMDataset('test_metadata.csv', IMAGE_ROOT, TABULAR_FEATURES, IMAGE_COLUMNS, LABEL_COLUMN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# --- Model, optimizer, loss setup ---
model = SMMultiModalNet(
    image_backbone=BACKBONE,
    tabular_dim=TABULAR_DIM,
    tabular_hidden=[32, 16],
    n_images=N_IMAGES,
    n_classes=N_CLASSES
)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

best_val_acc = 0.0
best_model_path = 'best_model.pt'

# --- Training loop ---
for epoch in range(EPOCHS):
    model.train()
    train_losses = []
    all_train_labels = []
    all_train_preds = []
    for batch in train_loader:
        images = batch['images'].to(device)
        tabular = batch['tabular'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        outputs = model(images, tabular)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())
        preds = outputs.argmax(1)
        all_train_labels.extend(labels.cpu().numpy())
        all_train_preds.extend(preds.cpu().numpy())
    
    train_acc = accuracy_score(all_train_labels, all_train_preds)
    
    # --- Validation ---
    model.eval()
    val_losses = []
    val_labels = []
    val_preds = []
    with torch.no_grad():
        for batch in val_loader:
            images = batch['images'].to(device)
            tabular = batch['tabular'].to(device)
            labels = batch['label'].to(device)
            outputs = model(images, tabular)
            loss = criterion(outputs, labels)
            val_losses.append(loss.item())
            preds = outputs.argmax(1)
            val_labels.extend(labels.cpu().numpy())
            val_preds.extend(preds.cpu().numpy())
    
    val_acc = accuracy_score(val_labels, val_preds)
    val_precision = precision_score(val_labels, val_preds, average='weighted', zero_division=0)
    val_recall    = recall_score(val_labels, val_preds, average='weighted', zero_division=0)
    
    # --- Save best model ---
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), best_model_path)
    
    print(f"Epoch {epoch+1}: "
          f"Train loss {np.mean(train_losses):.4f}, acc {train_acc:.4f} | "
          f"Val loss {np.mean(val_losses):.4f}, acc {val_acc:.4f}, "
          f"prec {val_precision:.4f}, recall {val_recall:.4f}")

# --- Load best model and evaluate on test set ---
model.load_state_dict(torch.load(best_model_path))
model.eval()
test_labels = []
test_preds = []
with torch.no_grad():
    for batch in test_loader:
        images = batch['images'].to(device)
        tabular = batch['tabular'].to(device)
        labels = batch['label'].to(device)
        outputs = model(images, tabular)
        preds = outputs.argmax(1)
        test_labels.extend(labels.cpu().numpy())
        test_preds.extend(preds.cpu().numpy())

test_acc = accuracy_score(test_labels, test_preds)
test_precision = precision_score(test_labels, test_preds, average='weighted', zero_division=0)
test_recall    = recall_score(test_labels, test_preds, average='weighted', zero_division=0)
test_cm        = confusion_matrix(test_labels, test_preds)
print("\n--- Test Results ---")
print(f"Accuracy  : {test_acc:.4f}")
print(f"Precision : {test_precision:.4f}")
print(f"Recall    : {test_recall:.4f}")
print(f"Confusion matrix:\n{test_cm}")
