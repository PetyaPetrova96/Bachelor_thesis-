import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch

from data_processing import train_loader, dev_loader, test_loader

print("Train Loader Loaded:", len(train_loader.dataset))
print("Dev Loader Loaded:", len(dev_loader.dataset))
print("Test Loader Loaded:", len(test_loader.dataset))


# Load the processor (remains unchanged)
processor = AutoImageProcessor.from_pretrained("motheecreator/vit-Facial-Expression-Recognition")

# Load the pretrained model
model = AutoModelForImageClassification.from_pretrained(
    "motheecreator/vit-Facial-Expression-Recognition"
)


num_classes = 3  # My dataset has 3 classes: Neu, Pos, Neg
model.classifier = nn.Linear(model.config.hidden_size, num_classes)


model.config.num_labels = num_classes

# Move model to GPU if available
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model.to(device)

# Print updated model
print(model)



import torch
import torch.nn as nn
import torch.optim as optim


criterion = nn.CrossEntropyLoss()


optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)


### Training ###


# Training Process with Accuracy
from tqdm import tqdm  

# Training settings
num_epochs = 5  
train_loss_history = []
train_acc_history = []

# Loop over epochs
for epoch in range(num_epochs):
    model.train()  
    running_loss = 0.0
    correct = 0  
    total = 0  
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)

    for batch in progress_bar:
        images, labels = batch  
        images, labels = images.to(device), labels.to(device)  
        
        optimizer.zero_grad()  

        outputs = model(images)  # Forward pass
        loss = criterion(outputs.logits, labels)  # Compute loss

        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights

        running_loss += loss.item()

        # Compute Accuracy
        preds = torch.argmax(outputs.logits, dim=1)  # Get predicted labels
        correct += (preds == labels).sum().item()  # Count correct predictions
        total += labels.size(0)  # Total images in batch

        progress_bar.set_postfix(loss=loss.item())  # Update progress bar

    avg_loss = running_loss / len(train_loader)
    train_loss_history.append(avg_loss)

    # Calculate overall training accuracy
    train_accuracy = 100 * correct / total
    train_acc_history.append(train_accuracy)

    print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_loss:.4f} | Train Accuracy: {train_accuracy:.2f}%")

print("Training Complete!")

#### Save the model ####




# Define the filename to save the model
model_save_path = "trained_vit_model.pth"

# Save the model state dictionary
torch.save(model.state_dict(), model_save_path)

print(f"Model saved successfully to {model_save_path}")
