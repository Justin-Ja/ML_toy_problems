import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Final
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from helper_funcs import createMoonInputImage

# Hyperparameters, adjust these to effect generated input and the effeciency of the model
N_SAMPLES: Final[int] = 1000
NOISE: Final[int] = 0.075
RAND_SEED: Final[int] = 42
DEFAULT_HIDDEN_UNITS: Final[int] = 5
LR: Final[int] = 0.1

# Creating and adjusting data into suitable training/testing tensors
points, groups = make_moons(N_SAMPLES, noise=NOISE, random_state=RAND_SEED)

points = torch.from_numpy(points).type(torch.float)
groups = torch.from_numpy(groups).type(torch.float)

x_train, x_test, y_train, y_test = train_test_split(points, groups, test_size=0.2)

#Device agnostic code
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

createMoonInputImage(points, groups)

#TODO: Consider possibly move this to its own file, and jsut have a new file for each class of model?
class MoonModel(nn.Module):
    def __init__(self, input_features, output_features, hidden_units=5):
        super().__init__()

        # Model has three hidden layers, with default of 5 neurons (hidden_units)
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(input_features, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, output_features),
        )

    def forward(self, x):
        return self.linear_layer_stack(x)
    
model_0 = MoonModel(input_features = 2, output_features = 1)

loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model_0.parameters(), LR) 

#TODO: Create a measure of accuracy