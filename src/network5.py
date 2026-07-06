from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import mnist_loader
from torch.utils.data import DataLoader, TensorDataset
import random
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

class ClassificationNN(nn.Module):
    def __init__(self, input_value):
        super(ClassificationNN, self).__init__()
        self.fc1 = nn.Linear(input_value, 128) # because of the vectors 784
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 16)
        self.fc5 = nn.Linear(16, 10)
        self.dropout = nn.Dropout(0.25)

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

training_data, validation_data, test_data = mnist_loader.load_data_wrapper() # [(28 x 28), 1]

validation_data = list(validation_data)
test_data = list(test_data)

training_inputs = np.array([x.flatten() for x, _ in training_data]) # 748 
training_labels_one_hot = np.array([y for _, y in training_data]) # [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
training_labels = np.argmax(training_labels_one_hot.squeeze(), axis=1)

validation_inputs = np.array([x.flatten() for x, _ in validation_data])
validation_labels = np.array([y for _, y in validation_data])
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
        
        image_pixels = x.reshape(28, 28)
        ascii_img = ""
        for row in range(0, 28, 2):
            for col in range(0, 28, 2):
                char = "█" if image_pixels[row][col] > 0.5 else " "
                ascii_img += char
            ascii_img += "\n"
        
        print(f"\nTEST: {i+1}")
        print(ascii_img)
        print(f"Network Prediction: {prediction}")
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
    print("Model loaded successfully from .pth file!")
    print(f"  Model architecture: {getShape(model)}")
except FileNotFoundError:
    print("No pre-trained model found. Starting fresh training...")
except Exception as e:
    print(f"Error loading model: {e}")

# Store training history for plotting
train_losses = []
val_accuracies = []

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
    
    avg_loss = running_loss / len(train_loader)
    train_losses.append(avg_loss)
    
    # Calculate validation accuracy for each epoch
    model.eval()
    with torch.no_grad():
        val_output = model(validationDataTensor)
        _, predicted = torch.max(val_output, 1)
        accuracy = (predicted == validationLabelsTensor).float().mean()
        val_accuracies.append(accuracy.item())
    
    if (epoch + 1) % 5 == 0:
        print(f"Epoch: {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f} - Val Acc: {accuracy.item()*100:.2f}%")

# Final validation
model.eval()
with torch.no_grad():
    val_output = model(validationDataTensor)
    _, predicted = torch.max(val_output, 1)
    final_accuracy = (predicted == validationLabelsTensor).float().mean()
    print(f"\n Final Validation Accuracy: {final_accuracy.item()*100:.2f}%")
    
    # Save the trained model
    model.save("my_trained_network5.pth")
    print("Model saved successfully!")

test_network(model, test_data, n=10, scaler=scaler)

# Show sample predictions on validation set
print("Sample predictions (first 10 validation images):")
for i in range(10):
    print(f"Predicted: {predicted[i].item()}, Actual: {validationLabelsTensor[i].item()}")

print("Generating result visualizations...")

# Create a figure with multiple subplots
fig = plt.figure(figsize=(15, 10))

# 1. Training Loss vs Epochs
plt.subplot(2, 2, 1)
plt.plot(range(1, num_epochs+1), train_losses, 'b-', linewidth=2)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Training Loss', fontsize=12)
plt.title('Training Loss vs Epochs', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.xticks(range(0, num_epochs+1, 5))

# 2. Validation Accuracy vs Epochs
plt.subplot(2, 2, 2)
plt.plot(range(1, num_epochs+1), np.array(val_accuracies)*100, 'g-', linewidth=2)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Validation Accuracy (%)', fontsize=12)
plt.title('Validation Accuracy vs Epochs', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.xticks(range(0, num_epochs+1, 5))
plt.ylim([0, 105])

# 3. Confusion Matrix
plt.subplot(2, 2, 3)
with torch.no_grad():
    all_predictions = []
    all_labels = []
    
    # Get predictions for validation set
    val_output = model(validationDataTensor)
    _, predicted_all = torch.max(val_output, 1)
    
    # Convert to numpy for sklearn
    predicted_np = predicted_all.cpu().numpy()
    actual_np = validationLabelsTensor.cpu().numpy()
    
    cm = confusion_matrix(actual_np, predicted_np)
    
    # Plot confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=range(10), yticklabels=range(10))
    plt.title('Confusion Matrix - Validation Set', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('Actual Label', fontsize=12)

# 4. Classification Report as Bar Chart
plt.subplot(2, 2, 4)
report = classification_report(actual_np, predicted_np, output_dict=True)
class_accuracies = [report[str(i)]['precision'] for i in range(10)]
bars = plt.bar(range(10), class_accuracies, color='orange', alpha=0.7)
plt.xlabel('Digit Class', fontsize=12)
plt.ylabel('Precision Score', fontsize=12)
plt.title('Precision per Class', fontsize=14, fontweight='bold')
plt.xticks(range(10))
plt.ylim([0, 1.05])
# Add value labels on bars
for bar, acc in zip(bars, class_accuracies):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{acc:.2f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('training_results.png', dpi=300, bbox_inches='tight')
plt.show()

# Print additional statistics
print("FINAL MODEL PERFORMANCE SUMMARY")
print(f"Best Validation Accuracy: {max(val_accuracies)*100:.2f}%")
print(f"Final Validation Accuracy: {val_accuracies[-1]*100:.2f}%")
print(f"Final Training Loss: {train_losses[-1]:.4f}")
print(f"Confusion Matrix saved in plot above")
print("\nClassification Report per class:")
for i in range(10):
    print(f"  Class {i}: Precision={report[str(i)]['precision']:.3f}, "
          f"Recall={report[str(i)]['recall']:.3f}, "
          f"F1={report[str(i)]['f1-score']:.3f}")

print(f"\nOverall Accuracy: {report['accuracy']*100:.2f}%")
print(f"Macro Avg F1: {report['macro avg']['f1-score']:.3f}")
print(f"Weighted Avg F1: {report['weighted avg']['f1-score']:.3f}")

# Save the plot
print(f"\n Plot saved as 'training_results.png'")