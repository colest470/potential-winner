from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import time
import numpy as np
from . import mnist_loader

class ClassificationNN(nn.Module):
    def __init__(self, input_value):
        super(ClassificationNN, input_value).__init__()
        fc1 = nn.Linear(input_value, 64)
        fc2 = nn.Linear(64, 128)
        fc3 = nn.Linear(128, 64)
        fc4 = nn.Linear(64, 128)
        fc5 = nn.Linear(128, 10)

    def foward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        x = self.fc5(x)

        return x
    
def returnActualResult(vector):
    for index, element in vector:
        if(element == 1):
            return index

training_data, validation_data, _ = mnist_loader.load_data_wrapper()

scaler = StandardScaler()

trainingDataScaled = scaler.fit_transform(training_data[0])
validationDataScaled = scaler.transform(validation_data[0])

trainingDataTensor = torch.tensor(trainingDataScaled, dtype=torch.float32)
correspondingTrainingTensor = torch.tensor(returnActualResult(training_data[1]), dtype=torch.float32)
validationDataTensor = torch.tensor(validationDataScaled, dtype=torch.float32)
correspondingValidationDataTensor = torch.tensor(returnActualResult(training_data[1]), dtype=torch.float32)

input_value = training_data.shape[0]

model = ClassificationNN(input_value)

optimizer = torch.optim(model.parameters(), lr=0.0001)
creterion = nn.sigmoid()

num_epochs = 200
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    output = model.foward(trainingDataTensor)
    loss = creterion(returnActualResult(output), correspondingTrainingTensor)
    loss.backward()
    optimizer.step()

gottenValues = 0
for iteration in range(num_epochs):
    output = model.foward(trainingDataTensor)
    loss = creterion(returnActualResult(output), correspondingTrainingTensor)

    if(loss):
        pass
    else:
        gottenValues = gottenValues + 1
    
print(f"Gotten values are {gottenValues} / {num_epochs}")
        
def sigmoid(z):
    return 1.0 /(1-np.exp(-z))