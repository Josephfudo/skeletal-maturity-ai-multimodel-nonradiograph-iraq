import torch
import torch.nn as nn
import torchvision.models as models

class ImageEncoder(nn.Module):
    def __init__(self, backbone='resnet18', pretrained=True, output_dim=None):
        super().__init__()
        if backbone == 'resnet18':
            cnn = models.resnet18(pretrained=pretrained)
            n_feats = cnn.fc.in_features
            cnn.fc = nn.Identity()
        elif backbone == 'resnet50':
            cnn = models.resnet50(pretrained=pretrained)
            n_feats = cnn.fc.in_features
            cnn.fc = nn.Identity()
        elif backbone == 'efficientnet_b0':
            cnn = models.efficientnet_b0(pretrained=pretrained)
            n_feats = cnn.classifier[1].in_features
            cnn.classifier = nn.Identity()
        elif backbone == 'vgg16':
            cnn = models.vgg16(pretrained=pretrained)
            n_feats = cnn.classifier[-1].in_features
            cnn.classifier = nn.Identity()
        elif backbone == 'densenet121':
            cnn = models.densenet121(pretrained=pretrained)
            n_feats = cnn.classifier.in_features
            cnn.classifier = nn.Identity()
        else:
            raise ValueError('Unsupported backbone')
        self.cnn = cnn
        self.n_feats = n_feats if output_dim is None else output_dim

    def forward(self, x):
        return self.cnn(x)  # [batch, feat_dim]

class MultiModalNet(nn.Module):
    def __init__(self, tabular_dim, n_classes=3, backbone='resnet18', pretrained=True):
        super().__init__()
        self.img_encoder = ImageEncoder(backbone=backbone, pretrained=pretrained)
        img_feat_dim = self.img_encoder.n_feats
        self.tabular_mlp = nn.Sequential(
            nn.Linear(tabular_dim, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, 32),
            nn.ReLU()
        )
        fusion_dim = img_feat_dim + 32
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes)
        )

    def forward(self, x_tab, x_imgs):
        B, S, C, H, W = x_imgs.shape
        x_imgs = x_imgs.view(B*S, C, H, W)
        feats = self.img_encoder(x_imgs)
        feats = feats.view(B, S, -1)
        feats = feats.mean(dim=1)
        tab_feats = self.tabular_mlp(x_tab)
        fused = torch.cat([feats, tab_feats], dim=1)
        logits = self.classifier(fused)
        return logits

if __name__ == "__main__":
    # Simple test: swap backbones
    model_r18 = MultiModalNet(tabular_dim=5, n_classes=3, backbone='resnet18')
    model_r50 = MultiModalNet(tabular_dim=5, n_classes=3, backbone='resnet50')
    model_vgg = MultiModalNet(tabular_dim=5, n_classes=3, backbone='vgg16')
    print(model_r18)
    print(model_r50)
    print(model_vgg)
