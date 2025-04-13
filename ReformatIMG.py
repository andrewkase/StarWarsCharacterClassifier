'''This takes all of the images I have gathered from the path ML_Project -> Train_Images -> <Character_name>
and resizes all of the images to 256x256 for processing. They are then dumped into ML_Project/Train_Images_Resized'''

import os
from PIL import Image

# Define input and output directories
input_dir = "Train_Images"
output_dir = "Train_Images_Resized"
valid_path = "Validation_Images_Resized"

# Ensure the output directories exist
os.makedirs(output_dir, exist_ok=True)
os.makedirs(valid_path, exist_ok=True)

# Loop through each name folder inside Train_Images
for folder in os.listdir(input_dir):
    folder_path = os.path.join(input_dir, folder)

    i = 0
    # Process each image in the character name folder
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)

        try:
            with Image.open(img_path) as img:
                img = img.convert("RGB")  # Convert to RGB for JPEG format
                img_resized = img.resize((331, 331))  # Resize to 331x331
                
                # Create new filename with .jpg extension
                new_filename = os.path.splitext(img_name)[0] + ".jpg"
                
                # Determine output path
                if i < 6:
                    output_img_path = os.path.join(valid_path, new_filename)
                else:
                    output_img_path = os.path.join(output_dir, new_filename)

                img_resized.save(output_img_path, format="JPEG")  # Save as JPEG
                i += 1

        except Exception as e:
            print(f"Skipping {img_name}: {e}")

print("Image resizing complete. Check the resized directories.")
