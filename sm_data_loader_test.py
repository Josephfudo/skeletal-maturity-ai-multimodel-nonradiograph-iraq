import pandas as pd
from torch.utils.data import DataLoader
from sm_multimodal_dataset import SMMDataset  # assumes your Dataset class is in 'sm_multimodal_dataset.py'

# Define tabular and image columns for input (customize as needed)
tabular_features = [
    'Age', 'Body_Weight_kg', 'Body_Height_cm', 'BMI',
    'weight_height_ratio', 'log_BMI', 'sqrt_age',
    'Age_pre', 'Body_Weight_kg_pre', 'BMI_pre'
]
image_columns = ['u_photo_eng', 'l_photo_eng', 'hp_photo_eng', 'hd_photo_eng', 'hdf_photo_eng']

# Create Datasets for each split
train_dataset = SMMDataset(
    metadata_csv='train_metadata.csv',
    image_root='/path/to/images',
    tabular_features=tabular_features,
    image_columns=image_columns,
    label_column='Growth_Phase_enc_eng',
    transform=None # apply torchvision transforms here if needed
)
val_dataset = SMMDataset(
    metadata_csv='val_metadata.csv',
    image_root='/path/to/images',
    tabular_features=tabular_features,
    image_columns=image_columns,
    label_column='Growth_Phase_enc_eng',
    transform=None
)
test_dataset = SMMDataset(
    metadata_csv='test_metadata.csv',
    image_root='/path/to/images',
    tabular_features=tabular_features,
    image_columns=image_columns,
    label_column='Growth_Phase_enc_eng',
    transform=None
)

# Build DataLoaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)

# Batch validation example
for batch in train_loader:
    print("Tabular batch shape:", batch['tabular'].shape)  # [batch, num_features]
    print("Images batch shape:", batch['images'].shape)     # [batch, num_images, C, H, W]
    print("Labels batch shape:", batch['label'].shape)      # [batch]
    break  # Just one batch for inspection

print("Train/Val/Test DataLoader setup complete!")
