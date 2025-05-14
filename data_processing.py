
import re
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
import re
from collections import Counter
from datasets import load_dataset


train_colour_images = "train_colour"
dev_colour_images = "dev_colour"
test_colour_images = "test_colour"

# Load the data
train_colour = load_dataset(train_colour_images )
dev_colour = load_dataset(dev_colour_images)
test_colour = load_dataset(test_colour_images)


# Check if the have laoded 
print(train_colour)
print(dev_colour)
print(test_colour)




# Function to extract label from filename 
def extract_label(example):
    filename = getattr(example["image"], "filename", None)  
    if filename is None:
        example["label"] = "Unknown"
        return example

    match = re.search(r"(Pos|Neu|Neg)", filename, re.IGNORECASE)  
    
    if match:
        example["label"] = match.group(0).capitalize()  
    else:
        example["label"] = "Unknown"  
    return example

# Apply label extraction to train, dev, and test
train_colour = train_colour.map(extract_label)
dev_colour = dev_colour.map(extract_label)
test_colour = test_colour.map(extract_label)

print(train_colour["train"].features)  
print(train_colour["train"][0])  # Print the first sample
print(dev_colour["train"][0]) 
print(test_colour["train"][0]) 






# Count occurrences of each label in train set
label_counts_train = Counter(train_colour["train"]["label"])

# Count occurrences of each label in train set
label_counts_dev = Counter(dev_colour["train"]["label"])


# Count occurrences of each label in train set
label_counts_test = Counter(test_colour["train"]["label"])



# Print label distribution
print("Label Counts in Train Set:", label_counts_train)

# Print label distribution
print("Label Counts in Dev Set:", label_counts_dev)


# Print label distribution
print("Label Counts in Test Set:", label_counts_test)


##### AUTO IMAGE PROCESSOR ######
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from transformers import AutoImageProcessor

# Load the processor for the ViT model
image_processor = AutoImageProcessor.from_pretrained("motheecreator/vit-Facial-Expression-Recognition")


# Test the AutoImageProcessos for one image only 

##Doing it only for 1 image
image= train_colour["train"][0]['image']
label = train_colour["train"][0]["label"]  # Get the label
pixel_values = image_processor(image, return_tensors="pt")
print(pixel_values)
print("Processed Tensor Shape:", pixel_values["pixel_values"].shape)
print("Min Pixel Value After Processing:", pixel_values["pixel_values"].min().item())
print("Max Pixel Value After Processing:", pixel_values["pixel_values"].max().item())


# Do it for several images # Doing it for several images with normalization

# Initialize empty lists to store processed data (with normalization)
train_pixel_values_norm = []
dev_pixel_values_norm = []
test_pixel_values_norm = []

# Function to process dataset manually (WITH normalization)
def process_dataset_with_norm(dataset, dataset_name):
    pixel_values_list = []  

    for i in range(len(dataset["train"])):  
        image = dataset["train"][i]["image"]  
        pixel_values = image_processor(image, return_tensors="pt")["pixel_values"].squeeze(0)  
        pixel_values_list.append(pixel_values)  

        # Print details for debugging
        print(f"{dataset_name} - Image {i+1}/{len(dataset['train'])}:")
        print("Processed Tensor Shape:", pixel_values.shape)
        print("Min Pixel Value After Processing:", pixel_values.min().item())
        print("Max Pixel Value After Processing:", pixel_values.max().item())
        print("-" * 50)  

    return torch.stack(pixel_values_list)  

# Process each dataset (WITH normalization)
train_pixel_values_norm = process_dataset_with_norm(train_colour, "Train Dataset")
dev_pixel_values_norm = process_dataset_with_norm(dev_colour, "Dev Dataset")
test_pixel_values_norm = process_dataset_with_norm(test_colour, "Test Dataset")

# Print final dataset shapes
print("Final Train Dataset Shape (Normalized):", train_pixel_values_norm.shape)  # (num_images, 3, 224, 224)
print("Final Dev Dataset Shape (Normalized):", dev_pixel_values_norm.shape)  # (num_images, 3, 224, 224)
print("Final Test Dataset Shape (Normalized):", test_pixel_values_norm.shape)  # (num_images, 3, 224, 224)



## Do the same without Normalization 
# Raw pixel values
# Define a transformation that ensures raw pixel values [0-255]

transform_raw = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),  
    transforms.Lambda(lambda x: x * 255)  
])

# Function to process dataset manually (TRUE RAW, WITHOUT normalization)
def process_dataset_raw(dataset, dataset_name):
    pixel_values_list = []  # Store processed images

    for i in range(len(dataset["train"])):  
        image = dataset["train"][i]["image"]  
        pixel_values = transform_raw(image)  
        pixel_values_list.append(pixel_values)  

        # Print details for debugging
        print(f"{dataset_name} - Image {i+1}/{len(dataset['train'])}:")
        print("Processed Tensor Shape:", pixel_values.shape)
        print("Min Pixel Value After Processing:", pixel_values.min().item())  
        print("Max Pixel Value After Processing:", pixel_values.max().item())  
        print("-" * 50)  

    return torch.stack(pixel_values_list)  

# Process each dataset without normalization (RAW pixel values)
train_pixel_values_raw = process_dataset_raw(train_colour, "Train Dataset")
dev_pixel_values_raw = process_dataset_raw(dev_colour, "Dev Dataset")
test_pixel_values_raw = process_dataset_raw(test_colour, "Test Dataset")

# Print final dataset shapes
print("Final Train Dataset Shape (RAW Pixel Values):", train_pixel_values_raw.shape)  # (num_images, 3, 224, 224)
print("Final Dev Dataset Shape (RAW Pixel Values):", dev_pixel_values_raw.shape)  # (num_images, 3, 224, 224)
print("Final Test Dataset Shape (RAW Pixel Values):", test_pixel_values_raw.shape)  # (num_images, 3, 224, 224)



# Plot the first 1o images pixel values 
##Plot the first 10 images normalized pixel values
# Set up a figure with multiple subplots (2 rows, 5 columns)
fig, axes = plt.subplots(2, 5, figsize=(20, 8))  # 2 rows, 5 columns


for i in range(10):
    
    pixel_values_norm = train_pixel_values_norm[i]  

    # Convert to NumPy array & Flatten
    pixel_values_norm_np = pixel_values_norm.numpy().flatten()

    # Get row and column index for subplot
    row, col = i // 5, i % 5  # Arrange in a 2x5 grid

    # Plot histogram on corresponding subplot
    axes[row, col].hist(pixel_values_norm_np, bins=50, color="red", alpha=0.7)
    axes[row, col].set_xlabel("Pixel Intensity")
    axes[row, col].set_ylabel("Frequency")
    axes[row, col].set_title(f"Image {i+1}")


plt.tight_layout()
plt.show()



#convert both the pixel values and keep the laels 
import torch
from torchvision import transforms

# Define a transformation for normalization (assuming ViT normalization: [-1,1])
transform_norm = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  
])

# Create a label mapping (assign a unique number to each class)
label_mapping = {"Neu": 0, "Pos": 1, "Neg": 2}  

# Function to process dataset with normalization and numeric labels
def process_dataset_norm_with_labels(dataset, dataset_name):
    pixel_values_list = []  
    labels_list = []  

    for i in range(len(dataset["train"])):  
        image = dataset["train"][i]["image"]  
        label = dataset["train"][i]["label"]  

        # Convert label from string to number
        label_num = label_mapping[label]  # Map string label to numerical value

        pixel_values = transform_norm(image)  # Apply normalization
        pixel_values_list.append(pixel_values)  
        labels_list.append(label_num)  

        # Print details for debugging
        print(f"{dataset_name} - Image {i+1}/{len(dataset['train'])}: Label: {label} ({label_num})")
        print("Processed Tensor Shape:", pixel_values.shape)
        print("Min Pixel Value After Processing:", pixel_values.min().item())  
        print("Max Pixel Value After Processing:", pixel_values.max().item())  
        print("-" * 50)  

    return torch.stack(pixel_values_list), torch.tensor(labels_list, dtype=torch.long)  

# Process dataset while keeping labels (NORMALIZED)
train_pixel_values_norm, train_labels = process_dataset_norm_with_labels(train_colour, "Train Dataset")
dev_pixel_values_norm, dev_labels = process_dataset_norm_with_labels(dev_colour, "Dev Dataset")
test_pixel_values_norm, test_labels = process_dataset_norm_with_labels(test_colour, "Test Dataset")

# Print final dataset shapes
print("Final Train Dataset Shape (Normalized Pixel Values):", train_pixel_values_norm.shape)  # (num_images, 3, 224, 224)
print("Final Train Labels Shape:", train_labels.shape)  # (num_images,)


print(train_pixel_values_norm[0])
plt.hist(train_pixel_values_norm[0].numpy().flatten())
print(train_labels)


# Try the pytorch data loader 
## Create a pythorvh Dataset
import torch
from torch.utils.data import Dataset, DataLoader

class ImageDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images  # Normalized images
        self.labels = labels  # Class labels

    def __len__(self):
        return len(self.images)  

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]  


## Create dataset Objects
# Create dataset for training, validation, and testing
train_dataset = ImageDataset(train_pixel_values_norm, train_labels)
dev_dataset = ImageDataset(dev_pixel_values_norm, dev_labels)
test_dataset = ImageDataset(test_pixel_values_norm, test_labels)


# Create dataLoader
# Define batch size
batch_size = 32

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# Fetch a batch of images and labels
images, labels = next(iter(train_loader))

print("Image Batch Shape:", images.shape)  
print("Label Batch Shape:", labels.shape)  
print("Example Labels:", labels)  


__all__ = ["train_loader", "dev_loader", "test_loader"]
