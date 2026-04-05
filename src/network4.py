import torch
import torch.nn as nn
import time
import numpy as np
from . import mnist_loader

class ClassificationNN(nn.Model):
    super(ClassificationNN, input_value).__init__
    def __init__(self, input_value):
        fc1 = nn.Linear(input_value, 64)
        fc2 = nn.Linear(64, 128)
        fc3 = nn.Linear(128, 64)
        fc4 = nn.Linear(64, 128)
        fc5 = nn.Linear(input_value, 64)
        relu = nn.ReLu()

    def foward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.fc5(x)

        return x
    

def loadData():
    try:
        with open("my_trained_network.json", "r") as file:
            pass

def sigmoig(z):
    return 1.0 /(1-np.exp(-z))