import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

def add_engineered_features(df):
    # Example new features:
    df['weight_height_ratio'] = df['Body_Weight_kg'] / (df['Body_Height_cm'] + 1)  # Add 1 to avoid division by zero
    df['log_BMI'] = np.log(df['BMI'] + 1e-5)
    df['sqrt_age'] = np.sqrt(df['Age'])
    # You can add more domain-specific features here
    return df

def make_tabular_pipeline(meta_path, output_path, numeric_cols, cat_cols, label_col, scaler_type='standard'):
    df = pd.read_csv(meta_path)
    df = add_engineered_features(df)

    # Combine with pre-existing features for model use
    all_numeric_cols = numeric_cols + ['weight_height_ratio', 'log_BMI', 'sqrt_age']
    
    if scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'minmax':
        scaler = MinMaxScaler()
    elif scaler_type == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError("Unknown scaler_type")
    
    df[all_numeric_cols] = scaler.fit_transform(df[all_numeric_cols])

    # Encode categoricals
    for col in cat_cols:
        if df[col].dtype == object:
            df[col + '_enc'] = pd.Categorical(df[col]).codes
        else:
            df[col + '_enc'] = df[col]

    # Encode label as integer if it's not already
    if df[label_col].dtype == object:
        df[label_col + '_enc'] = pd.Categorical(df[label_col]).codes
    else:
        df[label_col + '_enc'] = df[label_col]

    # Save new processed metadata
    df.to_csv(output_path, index=False)
    print(f"Saved engineered tabular data to {output_path}")
    print("Sample features:")
    print(df[all_numeric_cols + [col+'_enc' for col in cat_cols]].head())
    return all_numeric_cols, [col+'_enc' for col in cat_cols], label_col+'_enc'

if __name__ == "__main__":
    numeric_cols = ['Age', 'Body_Weight_kg', 'Body_Height_cm', 'BMI']
    cat_cols = ['Gender']
    label_col = 'Growth_Phase'
    meta_path = 'metadata_bmi.csv'
    output_path = 'metadata_bmi_engineered.csv'
    all_numeric_cols, encoded_cat_cols, encoded_label_col = make_tabular_pipeline(
        meta_path, output_path, numeric_cols, cat_cols, label_col, scaler_type='standard'
    )
