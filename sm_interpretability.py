import torch
from torch.utils.data import DataLoader
import shap
from captum.attr import IntegratedGradients
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np

from sm_multimodal_dataset import SMMDataset
from sm_multimodal_model import SMMultiModalNet

TABULAR_FEATURES = [
    'Age', 'BodyWeightkg', 'BodyHeightcm', 'BMI',
    'weightheightratio', 'logBMI', 'sqrtage',
    'Age_pre', 'BodyWeightkg_pre', 'BMI_pre'
]
IMAGE_COLUMNS = ['uphoto_eng', 'lphoto_eng', 'hpphoto_eng', 'hdphoto_eng', 'hdfphoto_eng']
LABEL_COLUMN = 'GrowthPhaseenc_eng'
TABULAR_DIM = len(TABULAR_FEATURES)
N_IMAGES = len(IMAGE_COLUMNS)
N_CLASSES = 3

IMAGE_ROOT = '/path/to/images'
BACKBONE = 'resnet18'

test_dataset = SMMDataset('test_metadata.csv', IMAGE_ROOT, TABULAR_FEATURES, IMAGE_COLUMNS, LABEL_COLUMN)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=2)

model = SMMultiModalNet(
    image_backbone=BACKBONE,
    tabular_dim=TABULAR_DIM,
    tabular_hidden=[32, 16],
    n_images=N_IMAGES,
    n_classes=N_CLASSES
)
model.load_state_dict(torch.load('best_model_resnet18.pt'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.eval()

# SHAP feature importance for tabular
batch = next(iter(test_loader))
images = batch['images'].to(device)
tabular = batch['tabular'].to(device)
labels = batch['label'].to(device)

explainer = shap.DeepExplainer(model, [images, tabular])
shap_values = explainer.shap_values([images, tabular])

mean_tabular_shap = np.abs(shap_values[1]).mean(axis=0)
print("Tabular Feature Importances:")
for i, feat in enumerate(TABULAR_FEATURES):
    print(f"{feat}: {mean_tabular_shap[i]:.4f}")

# --- Grad-CAM for image interpretability ---
from torchvision.models import resnet18
from captum.attr import LayerGradCam, visualize_image_attr

def get_image_gradcam(model, images, tabular):
    gradcam = LayerGradCam(model, model.image_backbone.layer4[1].conv2)
    img_grads = []
    for i in range(images.shape[0]):
        attr = gradcam.attribute((images[i:i+1], tabular[i:i+1]), target=labels[i].item())
        img_grads.append(attr.squeeze().cpu().detach().numpy())
    return img_grads

cam_batch = get_image_gradcam(model, images, tabular)
for i, cam in enumerate(cam_batch):
    plt.imshow(cam, cmap='plasma')
    plt.title(f'GradCAM for sample {i}')
    plt.savefig(f'gradcam_sample_{i}.png')
    plt.close()

print("SHAP feature analysis and Grad-CAM completed for sample batch.")
