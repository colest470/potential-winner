import torch
import torch.nn as nn
import numpy as np
try:
    from . import mnist_loader
except ImportError:
    import mnist_loader
from PIL import Image
import os
from sklearn.preprocessing import StandardScaler


class ClassificationNN(nn.Module):
    def __init__(self, input_value=784):
        super(ClassificationNN, self).__init__()
        self.fc1 = nn.Linear(input_value, 128)
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

    def save(self, filename="my_trained_network5.pth"):
        torch.save(self.state_dict(), filename)
        print(f"Model saved to {filename}")


def build_model(input_value=784):
    return ClassificationNN(input_value)


def load_trained_model(model_path="my_trained_network5.pth", device='cpu'):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = build_model()
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def build_scaler():
    training_data, _, _ = mnist_loader.load_data_wrapper()
    training_inputs = np.array([x.flatten() for x, _ in training_data])
    mean = training_inputs.mean(axis=0)
    std = training_inputs.std(axis=0)
    std[std == 0] = 1.0
    return mean, std


def apply_scaler(arr, scaler):
    mean, std = scaler
    return (arr - mean) / std


def preprocess_image(pil_image, scaler=None):
    if pil_image.mode != 'L':
        pil_image = pil_image.convert('L')

    if hasattr(Image, 'Resampling'):
        pil_image = pil_image.resize((28, 28), Image.Resampling.LANCZOS)
    else:
        pil_image = pil_image.resize((28, 28), Image.ANTIALIAS)

    arr = np.array(pil_image).astype(np.float32)
    if arr.mean() > 127:
        arr = 255.0 - arr
    arr /= 255.0
    arr = arr.reshape(1, -1)

    if scaler is not None:
        arr = apply_scaler(arr, scaler)

    return torch.tensor(arr, dtype=torch.float32)


def predict_image(image, model, scaler=None):
    x = preprocess_image(image, scaler=scaler)
    with torch.no_grad():
        output = model(x)
        prediction = torch.argmax(output, dim=1).item()
    return prediction


if __name__ == '__main__':
    import sys

    print("Checking architecture and saved models")
    try:
        model = load_trained_model("my_trained_network5.pth")
        print("Model loaded successfully from saved file. Skipping training.")
        sys.exit(0)
    except FileNotFoundError:
        print("No pre-trained model found. Starting fresh training...")
    except Exception as e:
        print(f"Error loading model: {e}")

    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    import matplotlib.pyplot as plt
    import random
    from sklearn.metrics import confusion_matrix, classification_report
    import seaborn as sns

    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    validation_data = list(validation_data)
    test_data = list(test_data)

    training_inputs = np.array([x.flatten() for x, _ in training_data])
    training_labels_one_hot = np.array([y for _, y in training_data])
    training_labels = np.argmax(training_labels_one_hot.squeeze(), axis=1)

    validation_inputs = np.array([x.flatten() for x, _ in validation_data])
    validation_labels = np.array([y for _, y in validation_data])

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
            print(f"TEST: {i+1} | Prediction: {prediction} | Actual: {actual_label}")

    def getShape(model):
        shape = []
        for layer in model.children():
            if isinstance(layer, nn.Linear):
                if not shape:
                    shape.append(layer.in_features)
                shape.append(layer.out_features)
        return shape

    print("Checking architecture and saved models")
    try:
        model = load_trained_model("my_trained_network5.pth")
        print("Model loaded successfully from saved file. Skipping training.")
        sys.exit(0)
    except FileNotFoundError:
        print("No pre-trained model found. Starting fresh training...")
    except Exception as e:
        print(f"Error loading model: {e}")

    train_losses = []
    val_accuracies = []

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
        model.eval()
        with torch.no_grad():
            val_output = model(validationDataTensor)
            _, predicted = torch.max(val_output, 1)
            accuracy = (predicted == validationLabelsTensor).float().mean()
            val_accuracies.append(accuracy.item())
        if (epoch + 1) % 5 == 0:
            print(f"Epoch: {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f} - Val Acc: {accuracy.item()*100:.2f}%")

    model.eval()
    with torch.no_grad():
        val_output = model(validationDataTensor)
        _, predicted = torch.max(val_output, 1)
        final_accuracy = (predicted == validationLabelsTensor).float().mean()
        print(f"\nFinal Validation Accuracy: {final_accuracy.item()*100:.2f}%")
        model.save("my_trained_network5.pth")
        print("Model saved successfully!")

    test_network(model, test_data, n=10, scaler=scaler)

    print("Sample predictions (first 10 validation images):")
    for i in range(10):
        print(f"Predicted: {predicted[i].item()}, Actual: {validationLabelsTensor[i].item()}")

    print("Generating result visualizations...")
    fig = plt.figure(figsize=(15, 10))
    plt.subplot(2, 2, 1)
    plt.plot(range(1, num_epochs+1), train_losses, 'b-', linewidth=2)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Training Loss', fontsize=12)
    plt.title('Training Loss vs Epochs', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xticks(range(0, num_epochs+1, 5))

    plt.subplot(2, 2, 2)
    plt.plot(range(1, num_epochs+1), np.array(val_accuracies)*100, 'g-', linewidth=2)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Validation Accuracy (%)', fontsize=12)
    plt.title('Validation Accuracy vs Epochs', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xticks(range(0, num_epochs+1, 5))
    plt.ylim([0, 105])

    plt.subplot(2, 2, 3)
    with torch.no_grad():
        val_output = model(validationDataTensor)
        _, predicted_all = torch.max(val_output, 1)
        predicted_np = predicted_all.cpu().numpy()
        actual_np = validationLabelsTensor.cpu().numpy()
        cm = confusion_matrix(actual_np, predicted_np)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
        plt.title('Confusion Matrix - Validation Set', fontsize=14, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('Actual Label', fontsize=12)

    plt.subplot(2, 2, 4)
    report = classification_report(actual_np, predicted_np, output_dict=True)
    class_accuracies = [report[str(i)]['precision'] for i in range(10)]
    bars = plt.bar(range(10), class_accuracies, color='orange', alpha=0.7)
    plt.xlabel('Digit Class', fontsize=12)
    plt.ylabel('Precision Score', fontsize=12)
    plt.title('Precision per Class', fontsize=14, fontweight='bold')
    plt.xticks(range(10))
    plt.ylim([0, 1.05])
    for bar, acc in zip(bars, class_accuracies):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01, f'{acc:.2f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig('training_results.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("FINAL MODEL PERFORMANCE SUMMARY")
    print(f"Best Validation Accuracy: {max(val_accuracies)*100:.2f}%")
    print(f"Final Validation Accuracy: {val_accuracies[-1]*100:.2f}%")
    print(f"Final Training Loss: {train_losses[-1]:.4f}")
    print(f"Confusion Matrix saved in plot above")
    print("\nClassification Report per class:")
    for i in range(10):
        print(f"  Class {i}: Precision={report[str(i)]['precision']:.3f}, Recall={report[str(i)]['recall']:.3f}, F1={report[str(i)]['f1-score']:.3f}")
    print(f"\nOverall Accuracy: {report['accuracy']*100:.2f}%")
    print(f"Macro Avg F1: {report['macro avg']['f1-score']:.3f}")
    print(f"Weighted Avg F1: {report['weighted avg']['f1-score']:.3f}")
    print(f"\nPlot saved as 'training_results.png'")



# import Pkg; 
# Pkg.add("Images")
# Pkg.add("Colors")

# using Images
# using Colors

# function detectEdges(file_path)
#     image = load(file_path)
    
#     result = fill(RGB(0, 0, 0), size(image))

#     for y in axes(image, 1)
#         for x in axes(image, 2)
#             p = image[y, x]
#             # Simple thresholding logic
#             if red(p) > 0.4 && green(p) > 0.4 && blue(p) > 0.3
#                 result[y, x] = RGB(0, 0, 0)
#             else
#                 result[y, x] = RGB(1, 1, 1)
#             end
#         end
#     end

#     return channelview(result) 
end