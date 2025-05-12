# All imports 
import os
from PIL import Image
import matplotlib.pyplot as plt
import re
from collections import Counter
from sklearn.model_selection import train_test_split


# Define folder paths
bw_folder = "Black and White"
color_folder = "Colour (not validated)"


# Function to load images from a folder
def load_images_from_folder(folder):
    images = []
    if os.path.exists(folder):
        for filename in os.listdir(folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                img_path = os.path.join(folder, filename)
                img = Image.open(img_path)
                images.append((filename, img))
    else:
        print(f"Folder '{folder}' not found.")
    return images

# Load images
bw_images = load_images_from_folder(bw_folder)
color_images = load_images_from_folder(color_folder)




print(len(bw_images))
print(len(color_images))

# Function to extract label from filename
def extract_label(filename):
    match = re.search(r'(Pos|Neu|Neg)', filename, re.IGNORECASE)
    return match.group(0).capitalize() if match else "Unknown"

# Create list of tuples (filename, label)
image_labels = [(filename, extract_label(filename)) for filename, _ in color_images]

# Print the result
print(image_labels)


# Count occurrences of each class in the whole data 
label_counts = Counter(label for _, label in image_labels)

# Print the counts
print(label_counts) 


# Get image sizes and color type (mode) for all pictures 
image_details = [(filename, img.size, img.mode) for filename, img in color_images]

# Print the details
for filename, size, mode in image_details:
    print(f"{filename}: Size={size}, Mode={mode}")
