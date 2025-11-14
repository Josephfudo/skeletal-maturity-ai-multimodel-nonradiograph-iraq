
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
import time

# --- Import your project files ---
from data_augmentation import get_train_transforms 
from sm_multimodal_model import ImageEncoder 

# --- CONFIGURATION ---
BACKBONES = ["resnet18", "resnet50", "densenet121", "efficientnet_b0", "vgg16"]
EPOCHS = 50 
BATCH_SIZE = 32
IMAGE_ROOT = "./" # Patched path
IMAGE_COLUMNS = ['u_photo_eng', 'l_photo_eng', 'hp_photo_eng', 'hd_photo_eng', 'hdf_photo_eng']
PROJECTION_DIM = 128
TEMPERATURE = 0.1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS_FIX = 16 # <<< --- THIS IS THE FIX ---

class SimCLRDataset(Dataset):
    def __init__(self, metadata_csv, image_root, image_cols, transform):
        self.meta = pd.read_csv(metadata_csv)
        self.image_root = image_root
        self.image_cols = image_cols
        self.transform = transform

        self.all_image_paths = []
        for col in self.image_cols:
            for path_fragment in self.meta[col].unique():
                full_path = os.path.join(self.image_root, str(path_fragment).replace("\\", os.sep).replace("/", os.sep))
                if os.path.exists(full_path):
                    self.all_image_paths.append(full_path)

        self.all_image_paths = list(set(self.all_image_paths))
        print(f"Found {len(self.all_image_paths)} unique images for pre-training.")

    def __len__(self):
        return len(self.all_image_paths)

    def __getitem__(self, idx):
        img_path = self.all_image_paths[idx]
        try:
            img = Image.open(img_path).convert('RGB')
            img_i = self.transform(img)
            img_j = self.transform(img)
            return img_i, img_j
        except Exception as e:
            return torch.zeros(3, 224, 224), torch.zeros(3, 224, 224)

class SimCLRModel(nn.Module):
    def __init__(self, backbone, feature_dim, projection_dim):
        super().__init__()
        self.backbone = backbone
        self.projector = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, projection_dim)
        )

    def forward(self, x):
        features = self.backbone(x)
        projections = self.projector(features)
        return F.normalize(projections, dim=1)

def nt_xent_loss(z_i, z_j, temperature):
    batch_size = z_i.shape[0]
    z = torch.cat([z_i, z_j], dim=0)
    sim_matrix = (z @ z.T) / temperature
    mask = ~torch.eye(batch_size * 2, device=DEVICE).bool()
    sim_matrix = sim_matrix[mask].view(batch_size * 2, -1)
    labels = torch.cat([
        torch.arange(batch_size) + batch_size - 1,
        torch.arange(batch_size)
    ]).to(DEVICE)
    loss = F.cross_entropy(sim_matrix, labels)
    return loss

def pretrain_backbone(backbone_name):
    print(f"\n--- Starting SimCLR Pre-training for {backbone_name} ---")
    start_time = time.time()

    contrastive_transform = get_train_transforms(image_size=(224, 224))
    train_ds = SimCLRDataset('final_metadata.csv', IMAGE_ROOT, IMAGE_COLUMNS, contrastive_transform)
    # --- THIS IS THE FIX ---
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS_FIX, pin_memory=True)

    encoder = ImageEncoder(backbone_name=backbone_name, pretrained=True)
    feature_dim = encoder.n_feats
    model = SimCLRModel(encoder, feature_dim, PROJECTION_DIM).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [{backbone_name}]", leave=False)
        for (images_i, images_j) in pbar:
            images_i = images_i.to(DEVICE)
            images_j = images_j.to(DEVICE)

            optimizer.zero_grad()
            z_i = model(images_i)
            z_j = model(images_j)
            loss = nt_xent_loss(z_i, z_j, TEMPERATURE)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({"Loss": loss.item()})

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch}/{EPOCHS} [{backbone_name}], Avg. Loss: {avg_loss:.4f}")

    output_path = f"simclr_{backbone_name}_backbone.pth"
    torch.save(model.backbone.state_dict(), output_path)

    end_time = time.time()
    print(f"--- Pre-training for {backbone_name} complete ---")
    print(f"Time taken: {(end_time - start_time) / 60:.2f} minutes")
    print(f"Backbone weights saved to: {output_path}\n")

if __name__ == "__main__":
    for backbone in BACKBONES:
        pretrain_backbone(backbone)
    print("All backbones have been pre-trained.")

print("Overwrote pretrain_simclr.py with num_workers=16")
