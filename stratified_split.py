import pandas as pd
from sklearn.model_selection import train_test_split

# Load the master metadata
df = pd.read_csv('final_metadata.csv')

# Make a stratification key (combining gender and growth phase)
df['stratify_key'] = df['Gender_eng'].astype(str) + '_' + df['Growth_Phase_eng'].astype(str)
# If your class/label encodings differ, replace 'Gendereng' / 'GrowthPhaseeng' accordingly!

# 1. Split off 20% for the test set stratified on the combined key
trainval_df, test_df = train_test_split(
    df, 
    test_size=0.20, 
    stratify=df['stratify_key'], 
    random_state=42
)

# 2. (Optional) Further split trainval into train/validation
train_df, val_df = train_test_split(
    trainval_df,
    test_size=0.2,  # 20% of trainval as validation (~16% of total)
    stratify=trainval_df['stratify_key'],
    random_state=42
)

# 3. Save splits
train_df.to_csv('train_metadata.csv', index=False)
val_df.to_csv('val_metadata.csv', index=False)
test_df.to_csv('test_metadata.csv', index=False)

# 4. Display class balance for check
print("Train class counts:\n", train_df['stratify_key'].value_counts())
print("Val class counts:\n", val_df['stratify_key'].value_counts())
print("Test class counts:\n", test_df['stratify_key'].value_counts())
