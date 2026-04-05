import torch
import torch.nn as nn
import time
import numpy as np

class HousingNN(nn.Module):
    def __init__(self, input_size):
        super(HousingNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)