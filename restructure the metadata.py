import pandas as pd
from datetime import datetime

# Load the existing metadata
metadata = pd.read_csv('complete metadata.csv')

# Columns to remove
cols_to_drop = ['Group', 'Class', 'Subject_Number']
metadata = metadata.drop(columns=cols_to_drop)

# Calculate Age as of data collection date (2025-10-22)
collection_date = datetime.strptime('2025-10-22', '%Y-%m-%d')
def calculate_age(birth_str):
    birth_date = datetime.strptime(birth_str, '%Y-%m-%d')
    age = (collection_date - birth_date).days / 365.25
    return round(age, 2)

metadata['Age'] = metadata['Birthday'].apply(calculate_age)

# Move 'Age' column to immediately after 'Birthday'
cols = list(metadata.columns)
birthday_index = cols.index('Birthday')
new_order = cols[:birthday_index+1] + ['Age'] + cols[birthday_index+1:-1]
metadata = metadata[new_order]

# Save to CSV
metadata.to_csv('metadata.csv', index=False)
