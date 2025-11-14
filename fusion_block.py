import torch
import torch.nn as nn

class SimpleAttentionFusion(nn.Module):
    """Fuses image features and tabular features using a simple gating attention mechanism."""
    def __init__(self, image_feat_dim, tabular_feat_dim, fusion_dim):
        super().__init__()
        # Attention (gating) for each image and one for tabular
        self.img_attn = nn.Linear(image_feat_dim, 1, bias=False)
        self.tab_attn = nn.Linear(tabular_feat_dim, 1, bias=False)
        self.fc = nn.Linear(image_feat_dim + tabular_feat_dim, fusion_dim)

    def forward(self, img_feats, tab_feats):
        # img_feats: (B, img_feat_dim)
        # tab_feats: (B, tab_feat_dim)
        # Attention scores
        img_gate = torch.sigmoid(self.img_attn(img_feats))  # (B, 1)
        tab_gate = torch.sigmoid(self.tab_attn(tab_feats))  # (B, 1)
        img_feats_weighted = img_feats * img_gate
        tab_feats_weighted = tab_feats * tab_gate
        fused = torch.cat([img_feats_weighted, tab_feats_weighted], dim=1)
        fusion_out = self.fc(fused)
        return fusion_out

# Example: Integration into your main model_zoo
class MultiModalAttnNet(nn.Module):
    def __init__(self, tabular_dim, n_classes=3, backbone='resnet18', pretrained=True, fusion_dim=128):
        super().__init__()
        from model_zoo import ImageEncoder  # Make sure model_zoo.py is in project and imported
        self.img_encoder = ImageEncoder(backbone=backbone, pretrained=pretrained)
        img_feat_dim = self.img_encoder.n_feats
        self.tabular_mlp = nn.Sequential(
            nn.Linear(tabular_dim, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, 32),
            nn.ReLU()
        )
        self.fusion = SimpleAttentionFusion(img_feat_dim, 32, fusion_dim)
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(fusion_dim, n_classes)
        )

    def forward(self, x_tab, x_imgs):
        B, S, C, H, W = x_imgs.shape
        x_imgs = x_imgs.view(B*S, C, H, W)
        feats = self.img_encoder(x_imgs)   # (B*S, img_feat_dim)
        feats = feats.view(B, S, -1).mean(dim=1)  # (B, img_feat_dim)
        tab_feats = self.tabular_mlp(x_tab)        # (B, 32)
        fusion_out = self.fusion(feats, tab_feats)
        logits = self.classifier(fusion_out)
        return logits

if __name__ == "__main__":
    # Basic test setup
    net = MultiModalAttnNet(tabular_dim=5, n_classes=3, backbone='resnet18')
    batch_img = torch.randn(4, 5, 3, 224, 224)
    batch_tab = torch.randn(4, 5)
    out = net(batch_tab, batch_img)
    print("Output shape:", out.shape)
