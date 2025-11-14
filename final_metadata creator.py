import pandas as pd

engineered = pd.read_csv('metadata_bmi_engineered.csv')
preprocessed = pd.read_csv('metadata_bmi_preprocessed.csv')
raw = pd.read_csv('metadata_bmi.csv')  # contains the actual raw values

final_metadata = engineered.merge(preprocessed, on='Case_ID', how='outer', suffixes=('_eng', '_pre'))
final_metadata = final_metadata.merge(raw[['Case_ID', 'Age', 'Body_Weight_kg', 'Body_Height_cm', 'BMI']], on='Case_ID', how='left')

final_metadata.to_csv('final_metadata.csv', index=False)
