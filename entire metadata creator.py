import os
import json
import csv

base_dir = "Data"
csv_file = "complete metadata.csv"
header = ["Case_ID", "Gender", "Birthday", "Body_Weight_kg", "Body_Height_cm", "Growth_Phase",
          "Group", "Class", "Subject_Number", "u_photo", "l_photo", "hp_photo", "hd_photo", "hdf_photo"]

rows = []
for group in ['boys', 'girls']:
    for growth_class in ['pre-peak', 'peak', 'post-peak']:
        class_dir = os.path.join(base_dir, group, growth_class)
        if not os.path.exists(class_dir): continue
        for subject_folder in os.listdir(class_dir):
            subject_path = os.path.join(class_dir, subject_folder)
            info_json_path = os.path.join(subject_path, 'info.json')
            if os.path.isfile(info_json_path):
                try:
                    with open(info_json_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    # Find each photo's filename
                    def find_photo(prefix):
                        for ext in ['.jpg', '.png', '.jpeg']:
                            fname = os.path.join(subject_path, prefix + ext)
                            if os.path.isfile(fname): return fname
                        return ''
                    row = [
                        data.get('Case_ID', ''),
                        data.get('Gender', ''),
                        data.get('Birthday', ''),
                        data.get('Body_Weight_kg', ''),
                        data.get('Body_Height_cm', ''),
                        data.get('Growth_Phase', ''),
                        group,
                        growth_class,
                        subject_folder,
                        find_photo('u'),
                        find_photo('l'),
                        find_photo('hp'),
                        find_photo('hd'),
                        find_photo('hdf')
                    ]
                    rows.append(row)
                except Exception as e:
                    print(f"Error reading {info_json_path}: {e}")

output_dir = os.path.dirname(csv_file)
if output_dir and not os.path.exists(output_dir): os.makedirs(output_dir)
with open(csv_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(rows)
