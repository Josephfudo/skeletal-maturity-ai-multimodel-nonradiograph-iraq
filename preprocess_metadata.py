import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load metadata
meta_path = 'metadata.csv'
metadata = pd.read_csv(meta_path)

# List of numerical and categorical columns for modeling
num_cols = ['Age', 'Body_Weight_kg', 'Body_Height_cm', 'BMI']
cat_cols = ['Gender']
label_col = 'Growth_Phase'  # Target

# Normalize numerical columns
scaler = StandardScaler()
metadata[num_cols] = scaler.fit_transform(metadata[num_cols])

# Encode categorical features
metadata['Gender_enc'] = LabelEncoder().fit_transform(metadata['Gender'])

# Encode target labels as integers for the model
metadata['Growth_Phase_enc'] = LabelEncoder().fit_transform(metadata[label_col])

# Preview - you will use these features for modeling:
print(metadata.head()[num_cols + ['Gender_enc', 'Growth_Phase', 'Growth_Phase_enc']])

# Save processed metadata for downstream use
metadata.to_csv('metadata_processed.csv', index=False)
