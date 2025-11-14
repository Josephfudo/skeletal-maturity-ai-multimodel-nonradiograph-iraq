import torch
from dataset_loader import MultiModalDataset
from multimodal_model import MultiModalNet
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

# Load validation data
num_features = ['Age', 'Body_Weight_kg', 'Body_Height_cm', 'BMI', 'Gender_enc']
image_cols = ['u_photo', 'l_photo', 'hp_photo', 'hd_photo', 'hdf_photo']
label_col = 'Growth_Phase_enc'
val_ds = MultiModalDataset('val_fold1.csv', num_features, image_cols, label_col)
val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=0)

# Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiModalNet(tabular_dim=5, n_classes=3, cnn_name='resnet18', pretrained=False).to(device)
model.load_state_dict(torch.load("multimodal_fold1.pth"))
model.eval()

# Collect predictions and labels
all_preds, all_labels = [], []
with torch.no_grad():
    for x_tab, x_imgs, y in val_loader:
        x_tab, x_imgs = x_tab.to(device), x_imgs.to(device)
        logits = model(x_tab, x_imgs)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y.numpy())

# Compute metrics
acc = accuracy_score(all_labels, all_preds)
prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
report = classification_report(all_labels, all_preds, target_names=['peak', 'post', 'pre'])

print(f"Accuracy: {acc:.3f}\nWeighted Precision: {prec:.3f}\nWeighted Recall: {rec:.3f}\nWeighted F1: {f1:.3f}")
print("\nDetailed Classification Report:\n", report)
