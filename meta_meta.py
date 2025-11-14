import csv

# Define the desired column order
desired_order = [
    'Case_ID',
    'Gender',
    'Birthday',
    'Age',
    'Body_Weight_kg',
    'Body_Height_cm',
    'BMI',
    'radio_img',
    'Growth_Phase',
    'u_photo',
    'l_photo',
    'hp_photo',
    'hd_photo',
    'hdf_photo'
]

# Read metadata_raw.csv
input_file = "metadata_raw.csv"
rows = []

with open(input_file, 'r') as f:
    reader = csv.DictReader(f)
    current_columns = list(reader.fieldnames)
    
    # Read all rows
    for row in reader:
        rows.append(row)

# Create new column order: desired columns + any extra columns not in desired list
new_column_order = []
for col in desired_order:
    if col in current_columns:
        new_column_order.append(col)

# Add any remaining columns that weren't in the desired list
for col in current_columns:
    if col not in new_column_order:
        new_column_order.append(col)

# Write back with new column order
with open(input_file, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=new_column_order)
    writer.writeheader()
    writer.writerows(rows)

print(f"✅ Reordered columns in {input_file}")
print(f"\nNew column order:")
for i, col in enumerate(new_column_order, 1):
    print(f"  {i}. {col}")
