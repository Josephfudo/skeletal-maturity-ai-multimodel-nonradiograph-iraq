import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import numpy as np
from data_augmentation import get_train_transforms, get_val_test_transforms

class MultiModalDataset(Dataset):
    def __init__(self, csv_file, num_features, image_cols, label_col, image_size=(224,224), train=True):
        self.meta = pd.read_csv(csv_file)
        self.num_features = num_features
        self.image_cols = image_cols
        self.label_col = label_col
        self.image_size = image_size
        if train:
            self.transform = get_train_transforms(image_size)
        else:
            self.transform = get_val_test_transforms(image_size)

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        tabular = row[self.num_features].astype(np.float32).values
        images = []
        for col in self.image_cols:
            img_path = row[col]
            try:
                img = Image.open(img_path).convert('RGB')
            except Exception:
                img = Image.new('RGB', self.image_size, (0,0,0))
            images.append(self.transform(img))
        images = torch.stack(images)
        label = int(row[self.label_col])
        return torch.tensor(tabular), images, label

if __name__ == "__main__":
    # Use all engineered features for tabular
    num_features = [
        'Age', 'Body_Weight_kg', 'Body_Height_cm', 'BMI',
        'weight_height_ratio', 'log_BMI', 'sqrt_age'
    ]
    image_cols = ['u_photo', 'l_photo', 'hp_photo', 'hd_photo', 'hdf_photo']
    label_col = 'Growth_Phase_enc'
    batch_size = 8

    train_ds = MultiModalDataset('train_fold1.csv', num_features, image_cols, label_col, train=True)
    val_ds   = MultiModalDataset('val_fold1.csv', num_features, image_cols, label_col, train=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    tab_batch, img_batch, label_batch = next(iter(train_loader))
    print('Tabular:', tab_batch.shape)
    print('Images:', img_batch.shape)
    print('Labels:', label_batch.shape)
