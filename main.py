import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
import os
import numpy as np
import wandb

import config

# Initialize Weights & Biases
wandb.init(project="plant_classification")


transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(), #ask about it
    transforms.Normalize(config.mean, config.std) #adjust these to the pretained values
])

labelled_dataset = torchvision.datasets.ImageFolder("data/labelled", transform=transforms)
train_data, val_data = train_test_split(labelled_dataset, test_size=config.val_size, random_state=42, shuffle=True)

train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=config.batch_size, shuffle=True)

model = models.resnet18(pretrained=config.pretrained)
final_layer_input_num = model.fc.in_features
model.fc = torch.nn.Linear(final_layer_input_num, config.num_classes)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
lr_scheduler = StepLR(optimizer, step_size=config.step_size, gamma=config.gamma)


for epoch in range(config.num_epochs):
    model.train()
    train_loss = 0
    
    for images, labels in train_loader:
        pred_labels = model(images)
        loss = loss_fn(pred_labels, labels)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Log training loss to W&B
        wandb.log({"Training Loss": loss.item()})

    print(f"Epoch {epoch+1}/{config.num_epochs} - Training Loss: {train_loss / len(train_loader):.4f}")

    #decay learning rate
    lr_scheduler.step()

    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Calculate validation accuracy
    val_accuracy = 100 * correct / total

    # Log validation loss and accuracy to W&B
    wandb.log({"Validation Loss": val_loss / len(val_loader),
              "Validation Accuracy": val_accuracy})
    print(f"Epoch {epoch+1}/{config.num_epochs} - Validation Loss: {val_loss / len(val_loader):.4f} - Validation Accuracy: {val_accuracy:.2f}%")

# Optional: Save trained model
wandb.save(config.save_model_path)

torch.save(model.state_dict(), config.save_model_path)
