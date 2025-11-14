import torch
import pandas as pd
import numpy as np
import os
import random
import json
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from captum.attr import LayerGradCam

# --- Project imports ---
from sm_multimodal_model import SMMultiModalNet
from data_augmentation import get_val_test_transforms

# --- CONFIGURATION ---
PARAM_FILE = "best_params.json"
METADATA_FILE = 'final_metadata.csv'
MODEL_PATH = 'best_model_diff_lr_fold1_resnet18.pth'
MODEL_NAME = 'resnet18'
IMAGE_BASE_DIR = './'
IMAGE_COLUMNS = ['u_photo_eng', 'l_photo_eng', 'hp_photo_eng', 'hd_photo_eng', 'hdf_photo_eng']
TABULAR_COLS = [
    'Age', 'Body_Weight_kg', 'Body_Height_cm', 'BMI',
    'weight_height_ratio', 'log_BMI', 'sqrt_age',
    'Age_pre', 'Body_Weight_kg_pre', 'BMI_pre'
]
TABULAR_DIM = len(TABULAR_COLS)
N_CLASSES = 3
OUTPUT_GIRLS_DIR = 'girls_roi'
OUTPUT_BOYS_DIR = 'boys_roi'
NUM_CASES_PER_STAGE = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GENDER_COLUMN = 'Gender_enc_eng'
GIRLS_VALUE = 1
BOYS_VALUE = 0

def set_inplace_false(model):
    for module in model.modules():
        if hasattr(module, 'inplace'):
            module.inplace = False

def load_model_and_data():
    df = pd.read_csv(METADATA_FILE)
    with open(PARAM_FILE, 'r') as f:
        TUNED_PARAMS = json.load(f)
    tabular_hidden = TUNED_PARAMS.pop('tabular_hidden')
    model = SMMultiModalNet(
        tabular_dim=TABULAR_DIM,
        n_classes=N_CLASSES,
        backbone=MODEL_NAME,
        pretrained=False,
        tabular_hidden=tabular_hidden,
        dropout_rate=TUNED_PARAMS['dropout_rate']
    ).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    set_inplace_false(model)
    model.eval()
    target_layer = model.img_encoder.cnn.layer4
    val_transform = get_val_test_transforms((224, 224))  # Use for model input, not for visualization
    return model, df, val_transform, target_layer

def process_inputs_stacked(case_row, transform, image_cols_list, tabular_cols_list, base_dir):
    pil_images = []
    image_tensors = []
    orig_sizes = []
    for img_col in image_cols_list:
        path_fragment = str(case_row[img_col]).replace("\\", os.sep).replace("/", os.sep)
        img_path = os.path.join(base_dir, path_fragment)
        if not os.path.exists(img_path):
            img_pil = Image.new('RGB', (224, 224), (0, 0, 0))
        else:
            try:
                img_pil = Image.open(img_path).convert('RGB')
            except Exception:
                img_pil = Image.new('RGB', (224, 224), (0, 0, 0))
        pil_images.append(img_pil)
        orig_sizes.append(img_pil.size)
        image_tensors.append(transform(img_pil))
    # stack for shape [1, 5, 3, 224, 224] for five images per subject
    stacked_img_tensor = torch.stack(image_tensors).unsqueeze(0).to(DEVICE)
    tab_values = [case_row[col] for col in tabular_cols_list]
    tab_tensor = torch.tensor(tab_values, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    return pil_images, stacked_img_tensor, tab_tensor, orig_sizes

def save_gradcam_overlay(original_img, cam_map, filepath, alpha=0.4):
    orig_np = np.array(original_img)
    cam_min, cam_max = np.min(cam_map), np.max(cam_map)
    cam_map_norm = (cam_map - cam_min) / (cam_max - cam_min + 1e-8)
    cam_img = np.uint8(plt.cm.jet(cam_map_norm)[:, :, :3] * 255)
    cam_upsampled = np.array(Image.fromarray(cam_img).resize(original_img.size, Image.BILINEAR))
    blend = (alpha * cam_upsampled + (1 - alpha) * orig_np).astype(np.uint8)
    Image.fromarray(blend).save(filepath)

def generate_roi_gradcam(model, stacked_img_tensor, tab_tensor, target_class, target_layer):
    def model_forward(image_stack, tabular_data):
        return model(image_stack, tabular_data)
    lgc = LayerGradCam(model_forward, target_layer)
    attribution_map = lgc.attribute(
        inputs=stacked_img_tensor,
        additional_forward_args=(tab_tensor,),
        target=target_class
    )
    return attribution_map.squeeze(1).cpu().detach().numpy()

def main():
    model, df, val_transform, target_layer = load_model_and_data()
    class_map = {'pre': 0, 'peak': 1, 'post': 2}
    os.makedirs(OUTPUT_GIRLS_DIR, exist_ok=True)
    os.makedirs(OUTPUT_BOYS_DIR, exist_ok=True)
    for gender_val, gender_name, output_dir in [
        (GIRLS_VALUE, 'girls', OUTPUT_GIRLS_DIR),
        (BOYS_VALUE, 'boys', OUTPUT_BOYS_DIR)
    ]:
        for stage in class_map.keys():
            df_subset = df[
                (df[GENDER_COLUMN] == gender_val) &
                (df['Growth_Phase_eng'] == stage)
            ]
            num_to_sample = min(len(df_subset), NUM_CASES_PER_STAGE)
            if num_to_sample == 0:
                continue
            random_cases = df_subset.sample(num_to_sample)
            for _, case_row in random_cases.iterrows():
                case_id = case_row['Case_ID']
                target_class_idx = class_map[stage]
                pil_images, stacked_img_tensor, tab_tensor, orig_sizes = process_inputs_stacked(
                    case_row=case_row,
                    transform=val_transform,
                    image_cols_list=IMAGE_COLUMNS,
                    tabular_cols_list=TABULAR_COLS,
                    base_dir=IMAGE_BASE_DIR
                )
                attr_gradcam_all = generate_roi_gradcam(
                    model, stacked_img_tensor, tab_tensor, target_class_idx, target_layer
                )
                for i, img_type in enumerate(IMAGE_COLUMNS):
                    pil_img = pil_images[i]
                    cam_map = attr_gradcam_all[i]
                    save_path_gradcam = os.path.join(
                        output_dir, f"{case_id}_{img_type}_roi_gradcam.png"
                    )
                    save_gradcam_overlay(pil_img, cam_map, save_path_gradcam)
    print("Grad-CAM overlays generated for all subjects.")

if __name__ == "__main__":
    main()
