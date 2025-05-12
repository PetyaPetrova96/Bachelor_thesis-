import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoImageProcessor, AutoModelForImageClassification

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load test dataset
test_colour = load_dataset("test_colour")  

# Load processor and model
image_processor = AutoImageProcessor.from_pretrained("motheecreator/vit-Facial-Expression-Recognition")

model = AutoModelForImageClassification.from_pretrained("motheecreator/vit-Facial-Expression-Recognition")
model.classifier = nn.Linear(model.config.hidden_size, 3)
model.config.num_labels = 3

# Load trained weights
model.load_state_dict(torch.load("trained_vit_model.pth", map_location=device))
model.to(device)
model.eval()

# SmoothGrad computation
def compute_smoothgrad(model, image_tensor, num_samples=50, noise_level=0.1):
    model.eval()
    total_gradients = None

    for _ in range(num_samples):
        noise = torch.randn_like(image_tensor) * noise_level
        noisy_image = (image_tensor + noise).detach().clone().requires_grad_(True)

        outputs = model(noisy_image)
        logits = outputs.logits
        target_class = logits.argmax(dim=1)

        model.zero_grad()
        logits[0, target_class].backward()

        gradients = noisy_image.grad.detach()
        if total_gradients is None:
            total_gradients = gradients
        else:
            total_gradients += gradients

    avg_gradients = total_gradients / num_samples
    avg_gradients = avg_gradients.abs().sum(dim=1).squeeze().cpu().numpy()
    return avg_gradients

# Visualization
def visualize_attention_single(images, attentions, scale=10):
    for i, (image, attention) in enumerate(zip(images, attentions)):
        img = np.array(image)
        h, w, _ = img.shape

        attn_resized = cv2.resize(attention, (w, h), interpolation=cv2.INTER_CUBIC)
        attn_resized = cv2.GaussianBlur(attn_resized, (5, 5), 0)
        p_low, p_high = np.percentile(attn_resized, (2, 98))
        attn_resized = np.clip((attn_resized - p_low) / (p_high - p_low), 0, 1)

        plt.figure(figsize=(scale, scale))
        plt.imshow(img)
        plt.imshow(attn_resized, cmap="jet", alpha=0.6)
        plt.title(f"Image {i+1}", fontsize=16)
        plt.axis("off")
        plt.tight_layout()
        plt.show()

# Main SmoothGrad pipeline
def process_images_smoothgrad(test_dataset, model, image_processor, device, num_images=25):
    images = []
    gradients = []

    for i in range(num_images):
        print(f"Processing Image {i+1}/{num_images}...")

        test_img = test_dataset["train"][i]["image"]
        images.append(test_img)

        image_tensor = image_processor(test_img, return_tensors="pt").pixel_values.to(device)
        image_tensor = image_tensor.detach().clone().requires_grad_(True)

        grad_map = compute_smoothgrad(model, image_tensor)
        gradients.append(grad_map)

    visualize_attention_single(images, gradients, scale=10)


process_images_smoothgrad(test_colour, model, image_processor, device, num_images=25)
