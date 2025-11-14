import pandas as pd

# Load the metadata
metadata = pd.read_csv('metadata.csv')

# Calculate BMI
def compute_bmi(row):
    weight_kg = row['Body_Weight_kg']
    height_m = row['Body_Height_cm'] / 100   # convert cm to m
    return round(weight_kg / (height_m ** 2), 2)

metadata['BMI'] = metadata.apply(compute_bmi, axis=1)

# Insert 'BMI' column after 'Body_Height_cm'
cols = list(metadata.columns)
idx = cols.index('Body_Height_cm') + 1
cols_new = cols[:idx] + ['BMI'] + cols[idx:-1]
metadata = metadata[cols_new]

# Save the updated metadata
metadata.to_csv('metadata_bmi.csv', index=False)
