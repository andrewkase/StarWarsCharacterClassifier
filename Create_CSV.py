import os
import csv

# Define input and output directories
image_dir = "Train_Images"

train_path = 'TrainingLabels.csv'
validation_path = 'ValidLabels.csv'

data = []
data2 = []

# Loop through each name folder inside Train_Images
for character in os.listdir(image_dir):
    folder_path = os.path.join(image_dir, character)

    i=0
    # Process each image in the character name folder
    for img_name in os.listdir(folder_path):
        base_name = os.path.splitext(img_name)[0] + ".jpg"  # Ensure .jpg extension

        if i < 6:
            data2.append([base_name, character])
        else:
            data.append([base_name, character])
        i+=1
        

with open(train_path, 'w', newline='') as file:
    writer = csv.writer(file)

    writer.writerow(['ID', 'Character'])

    writer.writerows(data)

with open(validation_path, 'w', newline='') as file:
    writer = csv.writer(file)

    writer.writerow(['ID', 'Character'])

    writer.writerows(data2)

print(f"CSV file '{train_path}' created successfully.")
print(f"CSV file '{validation_path}' created successfully.")
