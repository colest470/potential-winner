from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from . import mnist_loader
from torch.utils.data import DataLoader, TensorDataset
import random

class ClassificationNN(nn.Module):
    def __init__(self, input_value):
        super(ClassificationNN, self).__init__()
        self.fc1 = nn.Linear(input_value, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 128)
        self.fc5 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        return x

    def save(self, filename="my_trained_network4.pth"):
        torch.save(self.state_dict(), filename)
        print(f"Model saved to {filename}")

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

validation_data = list(validation_data)
test_data = list(test_data)

training_inputs = np.array([x.flatten() for x, _ in training_data])
training_labels_one_hot = np.array([y for _, y in training_data])
print(f"Original training_labels shape: {training_labels_one_hot.shape}")
training_labels = np.argmax(training_labels_one_hot.squeeze(), axis=1)
print(f"Converted training_labels shape: {training_labels.shape}")
print(f"Sample training labels: {training_labels[:10]}")

validation_inputs = np.array([x.flatten() for x, _ in validation_data])
validation_labels = np.array([y for _, y in validation_data])
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

optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

num_epochs = 50
batch_size = 64

train_dataset = TensorDataset(trainingDataTensor, correspondingTrainingTensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

def test_network(model, test_data, n=5, scaler=None):
    """
    Tests the network on n random samples from test_data
    
    Args:
        model: The trained PyTorch model
        test_data: List of (image, label) tuples
        n: Number of random samples to test
        scaler: The fitted StandardScaler (if needed)
    """
    model.eval()
    samples = random.sample(test_data, min(n, len(test_data)))
    
    for i, (x, y) in enumerate(samples):
        actual_label = y if isinstance(y, (int, np.integer)) else np.argmax(y)
        
        x_flat = x.flatten().reshape(1, -1)
        if scaler:
            x_scaled = scaler.transform(x_flat)
            x_tensor = torch.tensor(x_scaled, dtype=torch.float32)
        else:
            x_tensor = torch.tensor(x_flat, dtype=torch.float32)
        
        with torch.no_grad():
            output = model(x_tensor)
            prediction = torch.argmax(output, dim=1).item()
            confidence = torch.softmax(output, dim=1)[0][prediction].item() * 100
        
        image_pixels = x.reshape(28, 28)
        ascii_img = ""
        for row in range(0, 28, 2):
            for col in range(0, 28, 2):
                char = "█" if image_pixels[row][col] > 0.5 else " "
                ascii_img += char
            ascii_img += "\n"
        
        print(f"\nTEST: {i+1}")
        print(ascii_img)
        print(f"Network Prediction: {prediction} ({confidence:.2f}% confidence)")
        print(f"Actual Label: {actual_label}")

def getShape(model):
    """Get the shape of linear layers in the model"""
    shape = []
    for layer in model.children():
        if isinstance(layer, nn.Linear):
            if not shape:
                shape.append(layer.in_features)
            shape.append(layer.out_features)
    return shape

# Load existing model if available
print("Checking architecture and saved models")
try:
    saved_state_dict = torch.load("my_trained_network4.pth", map_location=torch.device('cpu'))
    model.load_state_dict(saved_state_dict)
    print("✓ Model loaded successfully from .pth file!")
    print(f"  Model architecture: {getShape(model)}")
except FileNotFoundError:
    print("No pre-trained model found. Starting fresh training...")
except Exception as e:
    print(f"Error loading model: {e}")

# Training loop
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

# Validation
model.eval()
with torch.no_grad():
    val_output = model(validationDataTensor)
    _, predicted = torch.max(val_output, 1)
    accuracy = (predicted == validationLabelsTensor).float().mean()
    print(f"\n✓ Validation Accuracy: {accuracy.item()*100:.2f}%")
    
    # Save the trained model
    model.save("my_trained_network4.pth")
    print("✓ Model saved successfully!")

# Test on random samples from test_data
print("\n" + "="*50)
print("Testing on random samples:")
print("="*50)
test_network(model, test_data, n=10, scaler=scaler)

# Show sample predictions on validation set
print("\n" + "="*50)
print("Sample predictions (first 10 validation images):")
print("="*50)
for i in range(10):
    print(f"Predicted: {predicted[i].item()}, Actual: {validationLabelsTensor[i].item()}")