from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from mnist_loader import load_data_wrapper
import os

class ClassificationNN(nn.Module):
    def __init__(self, input_value):
        super(ClassificationNN, self).__init__()
        self.fc1 = nn.Linear(input_value, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 128)
        self.fc5 = nn.Linear(128, 10)
        self.batchnorm = nn.BatchNorm1d(64)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        return x
    
    def save(self, filename):
        try:
            os.open(filename, "w")
        except():
            pass
# Load data
training_data, validation_data, test_data = load_data_wrapper()

# IMPORTANT: Convert validation_data from zip object to list
validation_data = list(validation_data)  # FIX: Convert to list
test_data = list(test_data)  # Also convert test_data if needed

# Extract training inputs
training_inputs = np.array([x.flatten() for x, _ in training_data])

# Convert training labels from one-hot to integers
training_labels_one_hot = np.array([y for _, y in training_data])
print(f"Original training_labels shape: {training_labels_one_hot.shape}")
training_labels = np.argmax(training_labels_one_hot.squeeze(), axis=1)
print(f"Converted training_labels shape: {training_labels.shape}")
print(f"Sample training labels: {training_labels[:10]}")

# Extract validation data (now a list)
validation_inputs = np.array([x.flatten() for x, _ in validation_data])
validation_labels = np.array([y for _, y in validation_data])  # These are integers already
print(f"Validation inputs shape: {validation_inputs.shape}")
print(f"Validation labels shape: {validation_labels.shape}")
print(f"Sample validation labels: {validation_labels[:10]}")

scaler = StandardScaler()
training_data_scaled = scaler.fit_transform(training_inputs)
validation_data_scaled = scaler.transform(validation_inputs)

trainingDataTensor = torch.tensor(training_data_scaled, dtype=torch.float32)
correspondingTrainingTensor = torch.tensor(training_labels, dtype=torch.long)

validationDataTensor = torch.tensor(validation_data_scaled, dtype=torch.float32)
validationLabelsTensor = torch.tensor(validation_labels, dtype=torch.long)

input_value = trainingDataTensor.shape[1]
model = ClassificationNN(input_value)

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

num_epochs = 50
batch_size = 64

# Create DataLoader
from torch.utils.data import DataLoader, TensorDataset

train_dataset = TensorDataset(trainingDataTensor, correspondingTrainingTensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

print("\nStarting training...")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    if (epoch + 1) % 5 == 0:
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch: {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}")

# Evaluate
model.eval()
with torch.no_grad():
    val_output = model(validationDataTensor)
    _, predicted = torch.max(val_output, 1)
    accuracy = (predicted == validationLabelsTensor).float().mean()
    print(f"\nValidation Accuracy: {accuracy.item()*100:.2f}%")
    
    # Show sample predictions
    print("\nSample predictions (first 10 validation images):")
    for i in range(10):
        print(f"Predicted: {predicted[i].item()}, Actual: {validationLabelsTensor[i].item()}")
        
