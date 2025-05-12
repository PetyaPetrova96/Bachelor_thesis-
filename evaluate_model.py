import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModelForImageClassification
from data_processing import dev_loader, test_loader

# Load label mapping
label_mapping = {0: "Neu", 1: "Pos", 2: "Neg"}

# Load model and processor
processor = AutoImageProcessor.from_pretrained("motheecreator/vit-Facial-Expression-Recognition")
model = AutoModelForImageClassification.from_pretrained(
    "motheecreator/vit-Facial-Expression-Recognition"
)

# Adjust classification head to match number of labels
num_classes = 3
model.classifier = nn.Linear(model.config.hidden_size, num_classes)
model.config.num_labels = num_classes

# Load saved weights
model.load_state_dict(torch.load("trained_vit_model.pth", map_location="cpu"))

# Move model to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Loss function
criterion = nn.CrossEntropyLoss()

# Evaluation function
def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs.logits, labels)

            total_loss += loss.item()
            predictions = torch.argmax(outputs.logits, dim=1)
            correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)

    avg_loss = total_loss / len(data_loader)
    accuracy = correct / total_samples * 100
    return avg_loss, accuracy

# Evaluate on dev set
dev_loss, dev_accuracy = evaluate(model, dev_loader, criterion, device)
print(f"[DEV] Loss: {dev_loss:.4f}, Accuracy: {dev_accuracy:.2f}%")

# Evaluate on test set
test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
print(f"[TEST] Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.2f}%")

# Show misclassified images
import matplotlib.pyplot as plt

def find_misclassified(model, dataloader, device, num_mistakes=4):
    model.eval()
    misclassified = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs.logits, dim=1)

            for img, pred, true in zip(images, preds, labels):
                if pred != true:
                    misclassified.append((img.cpu(), pred.item(), true.item()))
                if len(misclassified) >= num_mistakes:
                    return misclassified
    return misclassified

# Visualize
misclassified_samples = find_misclassified(model, test_loader, device, num_mistakes=4)

fig, axes = plt.subplots(1, len(misclassified_samples), figsize=(16, 4))
for i, (img, pred, true) in enumerate(misclassified_samples):
    img = img.permute(1, 2, 0).numpy()
    img = (img - img.min()) / (img.max() - img.min())
    axes[i].imshow(img)
    axes[i].set_title(f"Pred: {label_mapping[pred]}\nTrue: {label_mapping[true]}", fontsize=14)
    axes[i].axis("off")

plt.tight_layout()
plt.show()
