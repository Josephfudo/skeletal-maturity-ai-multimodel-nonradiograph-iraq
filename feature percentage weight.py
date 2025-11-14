import torch
import numpy as np
import matplotlib.pyplot as plt
from dataset_loader import MultiModalDataset
from multimodal_model import MultiModalNet
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

# ---- Tabular importances -----
# Load model as before
model = MultiModalNet(tabular_dim=5, n_classes=3, cnn_name='resnet18', pretrained=False)
model.load_state_dict(torch.load("multimodal_fold1.pth"))
model.eval()

first_linear = model.tabular_mlp[0]
tab_weights = first_linear.weight.detach().cpu().numpy()
tab_importance_vec = np.mean(np.abs(tab_weights), axis=0)
tab_feature_names = ['Age','Body_Weight_kg','Body_Height_cm','BMI','Gender_enc']

# ---- Image importance by ablation -----
image_names = ['u_photo','l_photo','hp_photo','hd_photo','hdf_photo']

# Function to measure effect of ablating each image channel
def image_ablation_importance(model_path, loader, which_img, base_acc=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MultiModalNet(tabular_dim=5, n_classes=3, cnn_name='resnet18', pretrained=False).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for x_tab, x_imgs, y in loader:
            x_tab, x_imgs = x_tab.to(device), x_imgs.to(device)
            # Zero-out the ablated image slot ONLY for the channel being tested
            x_imgs[:,:, :, :, :] = x_imgs[:,:, :, :, :].clone()
            x_imgs[:, which_img] = 0.0
            logits = model(x_tab, x_imgs)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())
    ablated_acc = accuracy_score(all_labels, all_preds)
    if base_acc is not None:
        # Importance = drop in accuracy when ablated
        return base_acc - ablated_acc
    return ablated_acc

# Load val loader as in your pipeline
val_ds = MultiModalDataset('val_fold1.csv', tab_feature_names, image_names, 'Growth_Phase_enc')
val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=0)

# Get reference accuracy with all images
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for x_tab, x_imgs, y in val_loader:
        x_tab, x_imgs = x_tab.to(device), x_imgs.to(device)
        logits = model(x_tab, x_imgs)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y.numpy())
base_acc = accuracy_score(all_labels, all_preds)

# Calc image importance via ablation
img_drop_accs = []
for img_idx in range(5):
    drop = image_ablation_importance("multimodal_fold1.pth", val_loader, img_idx, base_acc)
    img_drop_accs.append(max(drop, 0))  # importance can't be negative

# ---- Normalize to 100% ----
all_importances = np.concatenate([tab_importance_vec, img_drop_accs])
all_importances = all_importances / all_importances.sum() * 100

feature_labels = tab_feature_names + image_names
plt.bar(feature_labels, all_importances)
plt.ylabel("Relative importance (%)")
plt.title("Feature Importance (Tabular + 5 Images)")
for i, val in enumerate(all_importances):
    plt.text(i, val + 0.5, f"{val:.1f}%", ha='center')
plt.tight_layout()
plt.show()
