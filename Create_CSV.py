import os
import csv

# Define input and output directories
image_dir = "Train_Images"

file_path = 'Labels.csv'

data = []

# Loop through each name folder inside Train_Images
for character in os.listdir(image_dir):
    folder_path = os.path.join(image_dir, character)

    # Process each image in the character name folder
    for img_name in os.listdir(folder_path):
        data.append([img_name, character])
        

with open(file_path, 'w', newline='') as file:
    writer = csv.writer(file)

    writer.writerow(['ID', 'Character'])

    writer.writerows(data)

print(f"CSV file '{file_path}' created successfully.")