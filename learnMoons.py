import torch
from torch import nn
from typing import Final
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import helper_funcs as helper

# Hyperparameters, adjust these to effect the generated input and the effeciency of the model
N_SAMPLES: Final[int] = 1000
NOISE: Final[int] = 0.08
RAND_SEED: Final[int] = 42
DEFAULT_HIDDEN_UNITS: Final[int] = 5
LR: Final[float] = 0.1
EPOCHS: Final[int] = 500

# Creating and adjusting data into suitable training/testing tensors
points, groups = make_moons(N_SAMPLES, noise=NOISE, random_state=RAND_SEED)

points = torch.from_numpy(points).type(torch.float)
groups = torch.from_numpy(groups).type(torch.float)

points_train, points_test, groups_train, groups_test = train_test_split(points, groups, test_size=0.2)

# Allows for device agnostic code
device = helper.determineDevice()

helper.createInputImage(points, groups, name="moon_input.png")

# Consider possibly move this to its own file, and jsut have a new file for each class of model?
class MoonModel(nn.Module):
    def __init__(self, input_features, output_features, hidden_units=DEFAULT_HIDDEN_UNITS):
        super().__init__()

        # Model has three hidden layers, with default of 5 neurons (hidden_units)
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(input_features, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, output_features),
        )

    def forward(self, x):
        return self.linear_layer_stack(x)
    
model_0 = MoonModel(input_features = 2, output_features = 1, hidden_units=10)
#Binary groups so we can use BCE for loss calculation
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model_0.parameters(), LR) 


# Move data to same device before training
points_train, groups_train = points_train.to(device), groups_train.to(device)
points_test, groups_test = points_test.to(device), groups_test.to(device)
torch.manual_seed(RAND_SEED)


for epoch in range(EPOCHS + 1): #There's a plus one to get the final epoch stats printed out later
    ### TRAINING
    model_0.train()

    group_logits = model_0(points_train).squeeze()
    group_pred = torch.round(torch.sigmoid(group_logits)) # logits -> prediction probabilities -> prediction labels

    loss = loss_fn(group_logits, groups_train)
    accuracy = helper.accuracy_fn(groups_train, group_pred)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    ### TESTING
    model_0.eval()
    with torch.inference_mode():
        # Foward pass
        test_logits = model_0(points_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits)) # logits -> prediction probabilities -> prediction labels
        
        # Calcuate loss and accuracy
        test_loss = loss_fn(test_logits, groups_test)
        test_accuracy = helper.accuracy_fn(groups_test, test_pred)

    if epoch % 10 == 0:
        # Print out current status of the model
        print(f"Epoch: {epoch} | Accuracy: {accuracy: 0.2f}% | Loss: {test_loss: 0.5f} | - | Test Accuracy: {test_accuracy: 0.2f}% | Test Loss: {test_loss: 0.5f}")

helper.createOutputImage(model_0, points_train, groups_train, points_test, groups_test, name="moon_output.png")

