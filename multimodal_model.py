import torch
import torch.nn as nn
import torchvision.models as models

class MultiModalNet(nn.Module):
    def __init__(self, tabular_dim, n_classes=3, cnn_name='resnet18', pretrained=True):
        super().__init__()
        # --------- Image branch: Pretrained ResNet, output pooled vector per image
        cnn = getattr(models, cnn_name)(pretrained=pretrained)
        cnn.fc = nn.Identity()  # Remove last classification layer
        self.cnn = cnn
        self.img_feat_dim = 512  # For ResNet18; adjust for others

        # --------- Tabular branch: Simple MLP
        self.tabular_mlp = nn.Sequential(
            nn.Linear(tabular_dim, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, 32),
            nn.ReLU()
        )

        # --------- Fusion + Classification
        fusion_dim = self.img_feat_dim + 32
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes)
        )

    def forward(self, x_tab, x_imgs):
        # x_tab: [B, tabular_dim]
        # x_imgs: [B, 5, 3, 224, 224]
        B, S, C, H, W = x_imgs.shape
        x_imgs = x_imgs.view(B*S, C, H, W)
        feats = self.cnn(x_imgs)        # [B*S, img_feat_dim]
        feats = feats.view(B, S, -1)    # [B, 5, img_feat_dim]
        feats = feats.mean(dim=1)       # [B, img_feat_dim] (mean-pool across images)

        tab_feats = self.tabular_mlp(x_tab)  # [B, 32]
        fused = torch.cat([feats, tab_feats], dim=1)  # [B, fusion_dim]
        logits = self.classifier(fused)  # [B, n_classes]
        return logits

# Save as multimodal_model.py
