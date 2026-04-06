from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
from mnist_loader import load_data_wrapper

class ClassificationNN(nn.Module):
    def __init__(self, input_value):
        super(ClassificationNN, self).__init__()
        self.fc1 = nn.Linear(input_value, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 128)
        self.fc5 = nn.Linear(128, 10)

    def foward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        x = self.fc5(x)

        return x
    
def returnActualResult(vector):
    if isinstance(vector, torch.Tensor):
            vector = vector.detach().numpy()

    return np.argmax(vector)

training_data, validation_data, _ = load_data_wrapper()

training_inputs = np.array([x.flatten() for x, _ in training_data])  # Shape: (50000, 784)
training_labels = np.array([y for _, y in training_data])  # Shape: (50000,) with integers

validation_inputs = np.array([x.flatten() for x, _ in validation_data])
validation_labels = np.array([y for _, y in validation_data])

scaler = StandardScaler()
training_data_scaled = scaler.fit_transform(training_inputs)
validation_data_scaled = scaler.transform(validation_inputs)

trainingDataTensor = torch.tensor(training_data_scaled, dtype=torch.float32)
correspondingTrainingTensor = torch.tensor(training_labels, dtype=torch.float32)

input_value = trainingDataTensor.shape[1]

model = ClassificationNN(input_value)

optimizer = optim.Adam(model.parameters(), lr=0.0001)
creterion = nn.CrossEntropyLoss()

num_epochs = 200
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    output = model.foward(trainingDataTensor)
    loss = creterion(output, correspondingTrainingTensor)
    loss.backward()
    optimizer.step()

    if((epoch + 1) % 10 == 0):
        print(f"Epoch:{epoch} / {num_epochs} loss: {loss.item():.4f}")

# gottenValues = 0
# for iteration in range(num_epochs):
#     output = model.foward(trainingDataTensor)
#     loss = creterion(returnActualResult(output), correspondingTrainingTensor)

#     if(loss):
#         pass
#     else:
#         gottenValues = gottenValues + 1
    
# print(f"Gotten values are {gottenValues} / {num_epochs}")
        
def sigmoid(z):
    return 1.0 /(1-np.exp(-z))