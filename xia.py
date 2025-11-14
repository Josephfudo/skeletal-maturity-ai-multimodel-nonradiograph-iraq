import torch
import pandas as pd
import numpy as np
import os
import random
from PIL import Image
from torchvision import transforms
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import torch.nn as nn
import cv2

# ---- CONFIGURATION ----
METADATA_FILE = "final_metadata.csv"
IMAGE_BASE_DIR = "."  # base folder for images
IMAGE_COLUMNS = ['u_photo_eng', 'l_photo_eng', 'hp_photo_eng', 'hd_photo_eng', 'hdf_photo_eng']
GENDER_COLUMN = 'Gender_enc_eng'
GIRLS_VALUE = 1
BOYS_VALUE = 0
GROWTH_PHASE_COLUMN = 'Growth_Phase_eng'
GROWTH_PHASES = ['pre', 'peak', 'post']
NUM_CASES_PER_STAGE = 2
OUTPUT_FOLDER = "Eigen-CAM photos_v3"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(None)  # Use system time for a different random sample

# ---- Function to find last Conv2d or C3 layer ----
def find_last_conv_or_c3(module):
    last = None
    for m in module.children():
        if isinstance(m, (nn.Conv2d,)) or m.__class__.__name__.startswith("C3"):
            last = m
        elif isinstance(m, (nn.Sequential, nn.Module)):
            t = find_last_conv_or_c3(m)
            if t: last = t
    return last

# ---- Load YOLOv5 model and CAM layer ----
yolo = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
yolo = yolo.to(device).eval()
target_layer = find_last_conv_or_c3(yolo.model.model)
if target_layer is None:
    raise RuntimeError("No Conv2d/C3 layer found for CAM!")

# ---- Load and re-sample metadata for new cases ----
df = pd.read_csv(METADATA_FILE)

# Optionally, add a list of old_case_ids to exclude, e.g.:
# old_case_ids = [52, 82, ...]  # list of previous Case_ID int values
# df = df[~df['Case_ID'].isin(old_case_ids)]

cases = []
used_ids = set()
for gender_val, gender_str in [(BOYS_VALUE, "boy"), (GIRLS_VALUE, "girl")]:
    for phase in GROWTH_PHASES:
        subset = df[(df[GENDER_COLUMN] == gender_val) & (df[GROWTH_PHASE_COLUMN] == phase)]
        subset = subset[~subset['Case_ID'].isin(used_ids)]  # filter out previously used IDs, if any
        selected = subset.sample(min(NUM_CASES_PER_STAGE, len(subset)), random_state=None)  # sys RNG
        for _, row in selected.iterrows():
            cases.append({'row': row, 'gender': gender_str, 'phase': phase})
            used_ids.add(row['Case_ID'])  # Avoid repeats

# ---- Run EigenCAM pipeline ----
for idx, case in enumerate(cases):
    row = case['row']
    case_id = row['Case_ID']
    gender = case['gender']
    phase = case['phase']
    print(f"\nProcessing {gender}, {phase}, {case_id}...")

    for img_col in IMAGE_COLUMNS:
        path_fragment = str(row[img_col]).replace("\\", os.sep).replace("/", os.sep)
        img_path = os.path.join(IMAGE_BASE_DIR, path_fragment)
        if not os.path.exists(img_path):
            print("  Image not found:", img_path)
            continue

        img = Image.open(img_path).convert('RGB').resize((640, 640))
        img_np = np.array(img).astype(np.float32) / 255.0
        input_tensor = transforms.ToTensor()(img_np).unsqueeze(0).to(device)

        cam = EigenCAM(yolo, [target_layer])
        with torch.no_grad():
            grayscale_cam = cam(input_tensor)[0]

        overlay = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
        out_name = f"{case_id}_{gender}_{phase}_{img_col}_eigencam.jpg"
        out_path = os.path.join(OUTPUT_FOLDER, out_name)
        cv2.imwrite(out_path, overlay)
        print("  Saved:", out_path)
