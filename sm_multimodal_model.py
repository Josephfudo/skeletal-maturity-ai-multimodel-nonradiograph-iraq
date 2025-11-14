import torch
import torch.nn as nn
import torchvision.models as models
import os
import re # Import re for string matching

class ImageEncoder(nn.Module):
    """
    Encoder for image backbones.
    Takes a backbone name and returns the model and its output feature dimension.
    """
    def __init__(self, backbone_name='resnet18', pretrained=True):
        super().__init__()
        self.backbone_name = backbone_name
        
        if backbone_name == 'resnet18':
            model = models.resnet18(pretrained=pretrained)
            self.n_feats = model.fc.in_features
            model.fc = nn.Identity()
        elif backbone_name == 'resnet50':
            model = models.resnet50(pretrained=pretrained)
            self.n_feats = model.fc.in_features
            model.fc = nn.Identity()
        elif backbone_name == 'densenet121':
            model = models.densenet121(pretrained=pretrained)
            self.n_feats = model.classifier.in_features
            model.classifier = nn.Identity()
        elif backbone_name == 'efficientnet_b0':
            model = models.efficientnet_b0(pretrained=pretrained)
            self.n_feats = model.classifier[1].in_features
            model.classifier = nn.Identity()
        elif backbone_name == 'vgg16':
            model = models.vgg16(pretrained=pretrained)
            self.n_feats = 4096 # VGG's feature layer
            # VGG has a different structure, we use the features part
            model = model.features
            # We need an AdaptiveAvgPool to flatten the output to a fixed size
            self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
            self.vgg_classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(),
            )
            # Override the model's main component
            self.cnn = model
            return # Skip the final self.cnn assign
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
            
        self.cnn = model

    def forward(self, x):
        if self.backbone_name == 'vgg16':
            x = self.cnn(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.vgg_classifier(x)
            return x
        
        return self.cnn(x)

class SMMultiModalNet(nn.Module):
    """
    The main multimodal network, updated to accept tuned parameters.
    """
    def __init__(
        self,
        tabular_dim,
        n_classes=3,
        backbone='resnet18',
        pretrained=True,
        # --- Tunable Hyperparameters ---
        tabular_hidden=[32, 16], # Default if not tuning
        dropout_rate=0.3         # Default if not tuning
    ):
        super().__init__()
        
        # 1. Image Encoder
        self.img_encoder = ImageEncoder(backbone_name=backbone, pretrained=pretrained)
        img_feat_dim = self.img_encoder.n_feats

        # 2. Tabular MLP (now dynamically built)
        tab_layers = []
        in_dim = tabular_dim
        for h in tabular_hidden:
            tab_layers.append(nn.Linear(in_dim, h))
            tab_layers.append(nn.ReLU())
            tab_layers.append(nn.BatchNorm1d(h))
            tab_layers.append(nn.Dropout(dropout_rate)) # Use tuned dropout
            in_dim = h
        self.tabular_net = nn.Sequential(*tab_layers)
        tab_out_dim = tabular_hidden[-1] if tabular_hidden else tabular_dim

        # 3. Fusion and Classifier Head
        fusion_dim = img_feat_dim + tab_out_dim
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout_rate), # Use tuned dropout
            nn.Linear(128, n_classes)
        )

    def forward(self, images, tabular):
        # images: [B, N, C, H, W], tabular: [B, tabular_dim]
        B, N, C, H, W = images.shape
        images = images.view(B * N, C, H, W)
        img_feats = self.img_encoder(images)                 # [B*N, cnn_out]
        
        # Mean pool features from the 5 images
        img_feats = img_feats.view(B, N, -1).mean(dim=1)     # [B, cnn_out]
        
        tab_feats = self.tabular_net(tabular)                # [B, tab_out_dim]
        
        fusion_in = torch.cat([img_feats, tab_feats], dim=1) # [B, cnn_out + tab_out_dim]
        out = self.classifier(fusion_in)                     # [B, n_classes]
        return out

    def load_backbone_weights(self, weight_path):
        """
        Special loader for SimCLR weights.
        It only loads the 'backbone' part of the model.
        """
        if not os.path.exists(weight_path):
            print(f"Warning: Self-supervised weights not found at {weight_path}. Using ImageNet weights.")
            return

        print(f"Loading custom backbone weights from: {weight_path}")
        state_dict = torch.load(weight_path, map_location=torch.device('cpu'))
        
        # We need to match the keys from self.img_encoder.cnn
        # SimCLR saves it as `model.backbone.state_dict()`
        # Your ImageEncoder saves it as `self.cnn`
        
        # Create a new state_dict for the `cnn` module
        cnn_state_dict = {}
        for k, v in state_dict.items():
            # This logic depends on how `pretrain_simclr.py` saved the model.
            # Assuming it saved `model.backbone.state_dict()`, and `backbone` is an `ImageEncoder`
            # The keys might be `cnn.conv1.weight` etc.
            
            # If SimCLR saved the state_dict of the ImageEncoder:
            if k.startswith('cnn.'):
                 # Strip the 'cnn.' prefix
                new_key = k[len('cnn.'):]
                cnn_state_dict[new_key] = v
            # If SimCLR saved the state_dict of the ResNet/etc. directly:
            else:
                cnn_state_dict[k] = v

        # Filter out mismatched keys (like the VGG classifier part)
        model_keys = self.img_encoder.cnn.state_dict().keys()
        filtered_state_dict = {
            k: v for k, v in cnn_state_dict.items() if k in model_keys and \
            v.shape == self.img_encoder.cnn.state_dict()[k].shape
        }
        
        missing, unexpected = self.img_encoder.cnn.load_state_dict(filtered_state_dict, strict=False)
        print(f"Backbone weights loaded. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")

# --- Example usage ---
if __name__ == '__main__':
    print("--- Testing Model Initialization ---")
    
    # Test ResNet18
    model_r18 = SMMultiModalNet(
        backbone='resnet18',
        tabular_dim=10,
        tabular_hidden=[64, 32], # Tuned params
        dropout_rate=0.4         # Tuned param
    )
    
    # Test VGG16
    model_vgg = SMMultiModalNet(backbone='vgg16', tabular_dim=10)
    
    dummy_imgs = torch.randn(2, 5, 3, 224, 224)
    dummy_tab = torch.randn(2, 10)
    
    print("ResNet18 output shape:", model_r18(dummy_imgs, dummy_tab).shape)
    print("VGG16 output shape:", model_vgg(dummy_imgs, dummy_tab).shape)
    
    print("\n--- Testing SimCLR Weight Loading ---")
    # Create a dummy weight file
    DUMMY_WEIGHT_FILE = "simclr_resnet18_backbone.pth"
    torch.save(model_r18.img_encoder.state_dict(), DUMMY_WEIGHT_FILE)
    
    # Create a new model and load the weights
    new_model = SMMultiModalNet(backbone='resnet18', tabular_dim=10)
    new_model.load_backbone_weights(DUMMY_WEIGHT_FILE)
    os.remove(DUMMY_WEIGHT_FILE) # Clean up